import asyncio
import logging
import time
import re
from collections import defaultdict
from contextlib import asynccontextmanager

from aiogram import Router
from aiogram.enums import ChatAction
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.ai.agents.chef import ChefAgent
from bot.ai.agents.expert import ExpertAgent, ExpertInput
from bot.ai.agents.manager import ManagerAgent
from bot.ai.schemas import ExpertDecision
from bot.fsm.session import (
    get_last_clarify_category_options,
    get_session_data,
    get_user_language,
    set_current_intent,
    set_last_clarify_category_options,
    set_last_recommendations,
    set_preferences,
    set_user_language,
    # new reco flow helpers
    reset_reco_flow,
    get_reco_seed_text,
    set_reco_seed_text,
    get_reco_round,
    set_reco_round,
    get_reco_candidates,
    set_reco_candidates,
    get_reco_filters,
    set_reco_filters,
    get_reco_last_clarify,
    set_reco_last_clarify,
    get_history_last_request,
    set_history_last_request,
    get_reco_shown_category_options,
    get_reco_shown_property_options,
    add_reco_shown_options,
)
from bot.fsm.states import FoodBotStates
from bot.menu.loader import load_menu
from bot.menu.render import render_dish_card
from bot.menu.search import find_dish_by_id, group_by_category, search_by_category
from bot.services.language import detect_language

logger = logging.getLogger(__name__)
router = Router()
MENU_CATALOG = load_menu()
MANAGER_AGENT = ManagerAgent()
EXPERT_AGENT = ExpertAgent(MENU_CATALOG)
CHEF_AGENT = ChefAgent(MENU_CATALOG)
_PROCESSING_LOCKS: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
_LAST_BUSY_NOTICE_AT: dict[int, float] = {}
_BUSY_NOTICE_COOLDOWN_SECONDS = 3.0


def _extract_category_options_from_clarify(text_or_question: str | None, allowed_options: list[str] | None = None) -> list[str] | str | None:
    """Parse category options from manager clarify text or map user answer to one allowed category.

    Usage:
      - _extract_category_options_from_clarify(question) -> list[str]
      - _extract_category_options_from_clarify(user_text, last_options) -> str | None
    """
    q = (text_or_question or "").lower()
    if not q:
        return [] if allowed_options is None else None

    mapping = [
        (["суп", "супи", "супы"], "soups"),
        (["локшин", "лапш", "рамен", "удон", "макарон"], "noodles"),
        (["рис", "боул", "боули", "боулы"], "rice"),
        (["рол", "суш"], "sushi_rolls"),
        (["закуск", "гедза", "спрінг", "спринг"], "snacks"),
        (["десерт", "моті", "моти"], "desserts"),
        (["напої", "напит", "чай", "лате"], "drinks"),
        (["вок"], "wok"),
    ]

    if allowed_options is None:
        result: list[str] = []
        for keys, cat in mapping:
            if any(k in q for k in keys) and cat not in result:
                result.append(cat)
        return result

    allowed = [str(x).strip().lower() for x in (allowed_options or []) if str(x).strip()]
    if not allowed:
        return None

    # direct canonical mention
    for cat in allowed:
        if cat in q:
            return cat

    # map keywords but restrict to options shown in the previous clarify question
    for keys, cat in mapping:
        if cat in allowed and any(k in q for k in keys):
            return cat

    if _text_means_other_option(q):
        # Explicit "other" means exclude listed options; caller handles exclusion path.
        return None

    return None

def _text_means_other_option(text: str) -> bool:
    t = (text or "").strip().lower().replace("’", "'").replace("`", "'")
    return t in {
        "щось інше", "щосьiнше", "інше", "iнше", "другое", "что-то другое", "что то другое", "другое что-то",
    }

@asynccontextmanager
async def _typing(message: Message):
    try:
        await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    except Exception:
        pass
    yield


def _build_menu_overview_text(language: str) -> str:
    grouped = group_by_category(MENU_CATALOG)
    if language == "ru":
        lines = ["📋 <b>TsuPosh — тестовое меню (демо)</b>", ""]
        category_labels = {"soups":"Супы","noodles":"Лапша","rice":"Рис / Боулы","wok":"Вок","sushi_rolls":"Роллы","snacks":"Закуски","desserts":"Десерты","drinks":"Напитки"}
    else:
        lines = ["📋 <b>TsuPosh — тестове меню (демо)</b>", ""]
        category_labels = {"soups":"Супи","noodles":"Локшина","rice":"Рис / Боули","wok":"Вок","sushi_rolls":"Роли","snacks":"Закуски","desserts":"Десерти","drinks":"Напої"}
    for category_key in ["soups","noodles","rice","wok","sushi_rolls","snacks","desserts","drinks"]:
        dishes = grouped.get(category_key, [])
        if not dishes:
            continue
        lines.append(f"<b>{category_labels.get(category_key, category_key)}</b>:")
        for dish in dishes[:3]:
            lines.append(f"• {dish.name.ru if language == 'ru' else dish.name.ua}")
        lines.append("")
    return "\n".join(lines)



# --- Clarification-limited recommendation flow helpers ---

_CATEGORY_LABELS = {
    "soups": {"ua": "Супи", "ru": "Супы"},
    "noodles": {"ua": "Локшина", "ru": "Лапша"},
    "rice": {"ua": "Рис / боули", "ru": "Рис / боулы"},
    "wok": {"ua": "Вок", "ru": "Вок"},
    "sushi_rolls": {"ua": "Роли", "ru": "Роллы"},
    "snacks": {"ua": "Закуски", "ru": "Закуски"},
    "desserts": {"ua": "Десерти", "ru": "Десерты"},
    "drinks": {"ua": "Напої", "ru": "Напитки"},
}

_PROPERTY_LABELS = {
    "spicy": {"ua": "гостре", "ru": "острое"},
    "not_spicy": {"ua": "не гостре", "ru": "не острое"},
    "chicken": {"ua": "з куркою", "ru": "с курицей"},
    "beef": {"ua": "з яловичиною", "ru": "с говядиной"},
    "seafood": {"ua": "з морепродуктами", "ru": "с морепродуктами"},
    "vegetarian": {"ua": "вегетаріанське", "ru": "вегетарианское"},
    "sweet": {"ua": "солодке", "ru": "сладкое"},
}

def _label_category(cat: str, lang: str) -> str:
    return _CATEGORY_LABELS.get(cat, {}).get(lang, cat)

def _label_property(prop: str, lang: str) -> str:
    return _PROPERTY_LABELS.get(prop, {}).get(lang, prop)

def _build_manager_clarify_question(
    lang: str,
    seed_text: str,
    category_options: list[str],
    property_options: list[str],
    current_filters: dict | None = None,
) -> tuple[str, str | None]:
    """Build ONE short, conversational clarification question with numbered options.

    Returns: (question_text, asked_type) where asked_type is 'category' or 'property'.
    """
    current_filters = dict(current_filters or {})
    wanted_cats = {str(x).lower() for x in (current_filters.get("wanted_categories") or [])}
    wanted_props = {str(x).lower() for x in (current_filters.get("wanted_properties") or [])}
    excluded_props = {str(x).lower() for x in (current_filters.get("not_properties") or [])}

    # Remove contradictory/duplicate props from suggestions
    def _is_contradictory(p: str) -> bool:
        p = str(p).lower().strip()
        if "spicy" in wanted_props and p == "not_spicy":
            return True
        if "not_spicy" in wanted_props and p == "spicy":
            return True
        if p in excluded_props:
            return True
        if p in wanted_props:
            return True
        return False

    safe_props = [p for p in (property_options or []) if not _is_contradictory(p)]

    # Decide what to ask now: category first (if not chosen yet), then properties.
    asked: str | None = None
    if category_options and not wanted_cats:
        asked = "category"
    elif safe_props:
        asked = "property"
    elif category_options:
        asked = "category"

    # A tiny reminder to keep it chatty (but no re-asking)
    preface: list[str] = []
    if "spicy" in wanted_props:
        preface.append("Ок, беру гостре 🔥" if lang == "ua" else "Ок, беру острое 🔥")
    if "not_spicy" in wanted_props:
        preface.append("Ок, беру не гостре 🙂" if lang == "ua" else "Ок, беру не острое 🙂")

    # Build one question with up to 3 options
    lines: list[str] = []
    if asked == "category":
        opts = list(category_options or [])[:3]
        if lang == "ua":
            lines.append("Щоб підібрати точніше, що тобі ближче?")
        else:
            lines.append("Чтобы подобрать точнее, что вам ближе?")
        for i, c in enumerate(opts, start=1):
            lines.append(f"{i}) {_label_category(c, lang)}")
        if lang == "ua":
            lines.append("Напиши номер або назву. Якщо хочеш інше — напиши «інше».")
        else:
            lines.append("Напишите номер или название. Если хотите другое — напишите «другое».")
    elif asked == "property":
        opts = list(safe_props or [])[:3]
        if lang == "ua":
            lines.append("Ок 🙂 А що з цього додати/уточнити?")
        else:
            lines.append("Ок 🙂 А что из этого добавить/уточнить?")
        for i, p in enumerate(opts, start=1):
            lines.append(f"{i}) {_label_property(p, lang)}")
        if lang == "ua":
            lines.append("Напиши номер(и) або слово. Якщо нічого — напиши «неважливо».")
        else:
            lines.append("Напишите номер(а) или слово. Если ничего — напишите «неважно».")
    else:
        # Fallback (should be rare)
        if lang == "ua":
            lines.append("Підкажи, будь ласка, трохи більше деталей 🙂")
        else:
            lines.append("Подскажите, пожалуйста, чуть больше деталей 🙂")

    question = "\n".join([*preface, *lines]).strip()
    return question, asked

def _parse_clarify_answer(text: str, lang: str, category_options: list[str], property_options: list[str], asked: str | None = None) -> dict:
    """Parse user's reply to a clarification question into filters delta."""
    q = (text or "").lower()
    delta = {
        "wanted_categories": [],
        "not_categories": [],
        "wanted_properties": [],
        "not_properties": [],
        "required_properties": [],
        "spice_max": None,
        "vegetarian": None,
    }

    
    # numeric answers support (e.g. "1" or "1 3")
    numbers = [int(n) for n in re.findall(r"\b\d+\b", q)]
# category selection
    if category_options:
        selected = _extract_category_options_from_clarify(text, category_options)
        if selected:
            delta["wanted_categories"] = [selected]
        elif _reply_is_other_choice(text, lang):
            delta["not_categories"] = list(category_options)

    
    # property selection by numbers (only if we asked properties)
    if property_options and numbers and asked == "property":
        for n in numbers:
            if 1 <= n <= len(property_options):
                p = property_options[n - 1]
                delta["wanted_properties"].append(p)
                delta["required_properties"].append(p)
# properties: simple keyword matching
    for prop in property_options or []:
        label = _label_property(prop, lang)
        if label and label in q:
            delta["wanted_properties"].append(prop)
            delta["required_properties"].append(prop)

    
    if any(x in q for x in ["неважливо", "не важно", "неважно", "байдуже", "все одно"]):
        # user doesn't want to add extra constraints
        delta["wanted_properties"] = []
        delta["required_properties"] = []
        delta["wanted_categories"] = delta.get("wanted_categories") or []
# extra synonyms
    if any(x in q for x in ["не гостре", "неостро", "не остро"]):
        delta["wanted_properties"].append("not_spicy")
        delta["required_properties"].append("not_spicy")
        delta["spice_max"] = 1
    if any(x in q for x in ["гостре", "остро"]):
        delta["wanted_properties"].append("spicy")
        delta["required_properties"].append("spicy")
    if any(x in q for x in ["вегет", "без мяса", "без м'яса", "без мʼяса"]):
        delta["wanted_properties"].append("vegetarian")
        delta["required_properties"].append("vegetarian")
        delta["vegetarian"] = True

    # de-duplicate
    delta["wanted_properties"] = list(dict.fromkeys(delta["wanted_properties"]))
    delta["required_properties"] = list(dict.fromkeys(delta["required_properties"]))
    return delta

def _merge_reco_filters(base: dict, delta: dict) -> dict:
    result = dict(base or {})
    for key in ["wanted_categories", "not_categories", "wanted_properties", "not_properties", "required_properties", "avoid_ingredients"]:
        merged = []
        for item in (result.get(key) or []) + (delta.get(key) or []):
            if item not in merged:
                merged.append(item)
        if merged:
            result[key] = merged
        else:
            result.pop(key, None)
    if delta.get("spice_max") is not None:
        result["spice_max"] = delta["spice_max"]
    if delta.get("vegetarian") is True:
        result["vegetarian"] = True
    
    if "with_rice" in delta and delta.get("with_rice") is not None:
        result["with_rice"] = bool(delta.get("with_rice"))
    return result

def _summarize_request_for_history(lang: str, filters: dict) -> str:
    cats = [_label_category(c, lang) for c in (filters.get("wanted_categories") or [])]
    props = [_label_property(p, lang) for p in (filters.get("wanted_properties") or [])]
    parts = []
    if props:
        parts.append(", ".join(props))
    if cats:
        parts.append(("з " if lang == "ua" else "из ") + ", ".join(cats))
    if not parts:
        return "щось смачне" if lang == "ua" else "что-то вкусное"
    return " ".join(parts)


def _merge_preferences(base: dict | None, extra: dict | None) -> dict:
    result = dict(base or {})
    for k, v in (extra or {}).items():
        if v is None:
            continue
        if isinstance(v, list):
            existing = result.get(k) or []
            merged = []
            for item in [*existing, *v]:
                if item not in merged:
                    merged.append(item)
            result[k] = merged
        else:
            result[k] = v
    return result




def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())





def _map_protein_to_property(value: str | None) -> str | None:
    """Map free-form UA/RU protein mentions to our canonical property tokens.
    Canonical tokens are the same ones used in menu tags and ExpertAgent filtering.
    """
    if not value:
        return None
    v = _norm(value)
    # chicken
    if any(x in v for x in ["курка", "курицу", "курица", "куряч", "chicken"]):
        return "chicken"
    # beef
    if any(x in v for x in ["ялович", "говядин", "beef"]):
        return "beef"
    # seafood / fish
    if any(x in v for x in ["риба", "рыба", "лосос", "сьомг", "семг", "тунец", "тунець", "кревет", "креветк", "shrimp", "fish", "seafood"]):
        return "seafood"
    # tofu
    if any(x in v for x in ["тофу", "tofu"]):
        return "tofu"
    return None


def _looks_like_greeting(text: str) -> bool:
    t = _norm(text)
    return bool(re.search(r"\b(привіт|добрий\s*(день|вечір|ранок)?|хай|hello|hi|hey|здрастуй|здравствуйте)\b", t))

def _looks_like_thanks_or_bye(text: str) -> bool:
    t = _norm(text)
    return bool(re.search(r"\b(дякую|спасиб(і|о)?|thx|thanks|thank\s+you|пока|пака|бувай|до\s+побачення|goodbye|bye)\b", t))

def _detect_category_in_text(text: str, lang: str) -> str | None:
    t = _norm(text)
    # try by localized labels first
    for key, labels in _CATEGORY_LABELS.items():
        lab = (labels.get(lang) or "").lower()
        if lab and lab in t:
            return key
    # fallback: common UA/RU keywords
    keywords = {
        "soups": ["суп", "супи", "супы"],
        "rice": ["рис", "боул", "боулы", "боулi", "боули"],
        "wok": ["вок"],
        "noodles": ["локшина", "лапша", "удон", "рамен"],
        "sushi_rolls": ["роли", "роллы", "суші", "суши"],
        "snacks": ["закуск", "стартер"],
        "desserts": ["десерт", "солодке", "сладкое"],
        "drinks": ["напої", "напитки", "пити", "пить", "чай", "кава", "кофе"],
    }
    for key, words in keywords.items():
        if any(w in t for w in words):
            return key
    return None

def _looks_like_show_category_request(text: str, lang: str) -> str | None:
    t = _norm(text)
    # "show all" / "list" patterns
    if not re.search(r"(покажи|покаж|виведи|дай|переліч|список|всі|усі|весь|все|хочу\s+подивитись|хочу\s+бачити|покажи\s+меню)", t):
        return None
    return _detect_category_in_text(t, lang)

def _normalize_dish_name_text(s: str) -> str:
    s = (s or "").lower().replace("’", "'")
    for ch in [",", ".", "!", "?", ":", ";", "(", ")"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())


def _find_available_dish_strict(catalog, query: str) -> object | None:
    q = _normalize_dish_name_text(query)
    if not q:
        return None
    for dish in catalog.dishes:
        if not getattr(dish, "available", True):
            continue
        ua = _normalize_dish_name_text(dish.name.ua)
        ru = _normalize_dish_name_text(dish.name.ru)
        # strict name equality in either language only (availability must not silently swap variants)
        if q == ua or q == ru:
            return dish
    return None

def _compose_manager_reply(reply_text: str | None, clarifying_question: str | None) -> str | None:
    """Return ONE final manager message.

    Rule for clarification flow: prefer a single clarifying_question and do not append
    reply_text, because LLM often duplicates the same meaning in both fields.
    """
    reply = (reply_text or "").strip()
    question = (clarifying_question or "").strip()

    if question:
        return question
    if reply:
        return reply
    return None


def _reply_is_other_choice(text: str, language: str) -> bool:
    n = _norm(text)
    if language == "ru":
        variants = ["другое", "что-то другое", "другое блюдо", "иное"]
    else:
        variants = ["інше", "щось інше", "друге", "інший варіант"]
    return any(v in n for v in variants)


def _expert_input_from_prefs(language: str, user_text: str, prefs: dict) -> ExpertInput:
    prefs = dict(prefs or {})
    excluded = prefs.pop("_exclude_categories", []) or []
    allowed = prefs.pop("_allowed_categories", None)
    return ExpertInput(
        language=language,
        user_text=user_text,
        preferences=prefs,
        allowed_categories=allowed,
        excluded_categories=excluded,
    )


async def _reply_manager_with_optional_clarification(message: Message, reply_text: str | None, clarifying_question: str | None) -> None:
    text = _compose_manager_reply(reply_text, clarifying_question)
    if text:
        await message.answer(text)

async def _send_expert_recommendations(message: Message, decision: ExpertDecision) -> None:
    await message.answer(decision.intro_text)
    if not decision.items:
        return
    reason_label = "Чому підійде" if decision.language == "ua" else "Почему подойдет"
    for item in decision.items:
        dish = find_dish_by_id(MENU_CATALOG, item.dish_id)
        if not dish:
            logger.warning("Expert returned unknown dish_id: %s", item.dish_id)
            continue
        card = render_dish_card(dish, decision.language)
        await message.answer(f"{card}\n<b>{reason_label}:</b> {item.reason}")



def _processing_lock_key(message: Message) -> int:
    return int(message.from_user.id) if message.from_user else int(message.chat.id)


@asynccontextmanager
async def _user_processing_guard(message: Message, language: str):
    key = _processing_lock_key(message)
    lock = _PROCESSING_LOCKS[key]
    if lock.locked():
        now = time.monotonic()
        last = _LAST_BUSY_NOTICE_AT.get(key, 0.0)
        if now - last >= _BUSY_NOTICE_COOLDOWN_SECONDS:
            _LAST_BUSY_NOTICE_AT[key] = now
            await message.answer(
                "⏳ Я ще обробляю попередній запит. Надішли наступне повідомлення за кілька секунд."
                if language != "ru"
                else "⏳ Я еще обрабатываю предыдущий запрос. Отправьте следующее сообщение через пару секунд."
            )
        yield False
        return

    await lock.acquire()
    try:
        yield True
    finally:
        if lock.locked():
            lock.release()


@router.message()
async def handle_text_message(message: Message, state: FSMContext) -> None:
    if not message.text:
        return
    text = message.text.strip()
    if not text:
        return

    detected_language = detect_language(text)
    await set_user_language(state, detected_language)
    current_language = await get_user_language(state)
    normalized = text.lower()

    if normalized in {"меню", "покажи меню"}:
        await message.answer(_build_menu_overview_text(current_language))
        return
    if normalized in {"тест суп", "покажи суп"}:
        soups = search_by_category(MENU_CATALOG, "soups")
        await message.answer(render_dish_card(soups[0], current_language) if soups else ("Супы не найдены." if current_language == "ru" else "Супів не знайдено."))
        return

    current_state = await state.get_state()

    if current_state == FoodBotStates.clarifying_recommendation.state:
        session_lang = await get_user_language(state)
        seed_text = await get_reco_seed_text(state)
        reco_round = await get_reco_round(state)
        reco_filters = await get_reco_filters(state)
        reco_candidates = await get_reco_candidates(state)
        last_clarify = await get_reco_last_clarify(state)

        # Stage: final step - restart or one relaxed retry (after 3 clarifications)
        if reco_filters.get("_final_stage") is True:
            t = _norm(text)

            if t in {"спочатку", "сначала", "заново", "з початку", "заново"}:
                await reset_reco_flow(state)
                await message.answer("Ок, починаємо з початку 🙂" if session_lang == "ua" else "Ок, начнем сначала 🙂")
                # Re-run as a new message
                await handle_text_message(message, state)
                return

            if t in {"ще раз", "еще раз", "ще", "еще", "спробувати ще", "попробовать еще", "м'якше", "мягче", "продовжити", "продолжить"}:
                # One more attempt, but less strict filtering
                reco_filters.pop("_final_stage", None)
                reco_filters["_relaxed"] = True
                await set_reco_filters(state, reco_filters)
            else:
                await message.answer(
                    ("Не знайшов точного варіанту після 3 уточнень. Напиши: «спочатку» (почати з нуля) або «ще раз» (спробувати менш жорстко)." if session_lang == "ua"
                     else "Не нашёл точный вариант после 3 уточнений. Напишите: «сначала» (начать заново) или «еще раз» (попробовать мягче).")
                )
                return

        # Parse user's clarification answer and update filters and update filters
        delta = _parse_clarify_answer(
            text,
            session_lang,
            category_options=list(last_clarify.get("category_options") or []),
            property_options=list(last_clarify.get("property_options") or []),
            asked=last_clarify.get("asked"),
        )
        reco_filters = _merge_reco_filters(reco_filters, delta)
        await set_reco_filters(state, reco_filters)

        async with _user_processing_guard(message, session_lang) as can_process:
            if not can_process:
                return
            async with _typing(message):
                expert_result = await EXPERT_AGENT.recommend_with_clarification(
                    language=session_lang,
                    seed_text=seed_text or text,
                    filters=reco_filters,
                    candidate_dish_ids=reco_candidates,
                    round_index=reco_round,
                )

        if expert_result.mode == "clarify":
            # Save expert candidate pool and new clarify options
            await set_reco_candidates(state, expert_result.candidate_dish_ids)
            clar = expert_result.clarification
            cat_opts = list(getattr(clar, "category_options", []) or [])
            prop_opts = list(getattr(clar, "property_options", []) or [])

            # Avoid repeating the same options across rounds
            shown_cats = set(await get_reco_shown_category_options(state))
            shown_props = set(await get_reco_shown_property_options(state))
            cat_opts = [c for c in cat_opts if c not in shown_cats]
            prop_opts = [p for p in prop_opts if p not in shown_props]
            await set_reco_last_clarify(state, cat_opts, prop_opts, asked)

            # Clarification limit reached (3 rounds). Offer restart or one relaxed retry.
            if reco_round >= 3:
                reco_filters["_final_stage"] = True
                await set_reco_filters(state, reco_filters)
                await message.answer(
                    ("Після 3 уточнень я все ще не бачу точного варіанту 😕\nНапиши: «спочатку» (почати з нуля) або «ще раз» (спробувати менш жорстко)." if session_lang == "ua"
                     else "После 3 уточнений я всё ещё не вижу точного варианта 😕\nНапишите: «сначала» (начать заново) или «еще раз» (попробовать мягче).")
                )
                return

            await set_reco_round(state, reco_round + 1)
            question, asked = _build_manager_clarify_question(session_lang, seed_text or "", cat_opts, prop_opts, reco_filters)
            await set_reco_last_clarify(state, cat_opts, prop_opts, asked)

            # Remember shown options so we don't repeat them next round
            if asked == "category":
                await add_reco_shown_options(state, categories=cat_opts[:3])
            elif asked == "property":
                await add_reco_shown_options(state, properties=prop_opts[:3])
            await _reply_manager_with_optional_clarification(message, expert_result.intro_text, question)
            return

        # Got final recommendations
        await state.set_state(FoodBotStates.showing_recommendations)
        await _send_expert_recommendations(message, expert_result)
        await set_last_recommendations(state, [x.dish_id for x in expert_result.items])

        # Move last request into simple history
        summary = _summarize_request_for_history(session_lang, reco_filters)
        await set_history_last_request(state, summary)

        # Reset recommendation flow state
        await reset_reco_flow(state)
        await set_last_clarify_category_options(state, [])
        return

    async with _user_processing_guard(message, current_language) as can_process:
        if not can_process:
            return
        async with _typing(message):
            decision = await MANAGER_AGENT.analyze_message(text)

    await set_user_language(state, decision.language)
    await set_current_intent(state, decision.intent)
    await set_preferences(state, decision.extracted_preferences.model_dump())

    logger.info("Manager decision | intent=%s lang=%s clarify=%s confidence=%.2f dish_query=%s prefs=%s", decision.intent, decision.language, decision.needs_clarification, decision.confidence, decision.dish_query, decision.extracted_preferences.model_dump())



    # If user explicitly wants to see the full menu of a category, show it directly (cards)
    cat_key = _looks_like_show_category_request(text, decision.language)
    if cat_key:
        async with _user_processing_guard(message, decision.language) as can_process:
            if not can_process:
                return
            async with _typing(message):
                expert_result = await EXPERT_AGENT.list_category(
                    language=decision.language,
                    category=cat_key,
                )
        await state.set_state(FoodBotStates.showing_recommendations)
        await _send_expert_recommendations(message, expert_result)
        await set_last_recommendations(state, [x.dish_id for x in expert_result.items])
        return



    if decision.intent == "recommendation_request":
        # Start / continue clarification-limited flow (expert decides when to narrow).
        existing_seed = await get_reco_seed_text(state)
        await set_reco_seed_text(state, existing_seed or text)
        seed_text = await get_reco_seed_text(state)
        await set_reco_round(state, 0)

        # If this is a new recommendation session, we may reference last request memory
        last_mem = await get_history_last_request(state)
        if last_mem and not existing_seed:
            if decision.language == "ru":
                decision.reply_text = f"{decision.reply_text}\n\nКстати, в прошлый раз вы искали: {last_mem}. Могу подобрать что-то похожее или новое 🙂"
            else:
                decision.reply_text = f"{decision.reply_text}\n\nДо речі, минулого разу ти шукав(ла): {last_mem}. Можу підібрати щось схоже або новеньке 🙂"

        # Initialize filters from manager extracted prefs (best-effort)
        filters: dict = {}

        # Quick keyword bootstrap (covers cases when manager prefs extraction misses it)
        seed_norm = _norm(seed_text or "")
        if "не гостр" in seed_norm or "неостр" in seed_norm or "не ост" in seed_norm:
            filters.setdefault("wanted_properties", [])
            if "not_spicy" not in filters["wanted_properties"]:
                filters["wanted_properties"].append("not_spicy")
            filters.setdefault("not_properties", [])
            if "spicy" not in filters["not_properties"]:
                filters["not_properties"].append("spicy")
        elif "гостр" in seed_norm or "остр" in seed_norm:
            filters.setdefault("wanted_properties", [])
            if "spicy" not in filters["wanted_properties"]:
                filters["wanted_properties"].append("spicy")
            filters.setdefault("not_properties", [])
            if "not_spicy" not in filters["not_properties"]:
                filters["not_properties"].append("not_spicy")

        
        # Carb / staple keyword bootstrap (example: rice). This helps when manager extraction misses it.
        if "рис" in seed_norm or "rice" in seed_norm:
            filters["with_rice"] = True
        if "без рис" in seed_norm or "безрис" in seed_norm:
            filters["with_rice"] = False
        prefs = decision.extracted_preferences.model_dump()
        if prefs.get("category"):
            filters["wanted_categories"] = [prefs["category"]]
        if prefs.get("vegetarian") is True:
            filters["vegetarian"] = True
            filters["wanted_properties"] = ["vegetarian"]
        if prefs.get("protein"):
            p = _map_protein_to_property(str(prefs["protein"]))
            if p:
                filters.setdefault("wanted_properties", [])
                if p not in filters["wanted_properties"]:
                    filters["wanted_properties"].append(p)
                filters.setdefault("required_properties", [])
                if p not in filters["required_properties"]:
                    filters["required_properties"].append(p)
        if prefs.get("spice_level"):
            val = str(prefs["spice_level"]).lower()
            if val in {"low", "mild"}:
                filters["spice_max"] = 1
                filters.setdefault("wanted_properties", [])
                if "not_spicy" not in filters["wanted_properties"]:
                    filters["wanted_properties"].append("not_spicy")
                filters.setdefault("required_properties", [])
                if "not_spicy" not in filters["required_properties"]:
                    filters["required_properties"].append("not_spicy")
            if val in {"high", "hot"}:
                filters.setdefault("wanted_properties", [])
                if "spicy" not in filters["wanted_properties"]:
                    filters["wanted_properties"].append("spicy")
                filters.setdefault("required_properties", [])
                if "spicy" not in filters["required_properties"]:
                    filters["required_properties"].append("spicy")
        if prefs.get("avoid_ingredients"):
            filters["avoid_ingredients"] = list(prefs.get("avoid_ingredients") or [])

        await set_reco_filters(state, filters)
        await set_reco_candidates(state, [])
        await set_reco_last_clarify(state, [], [])

        async with _user_processing_guard(message, decision.language) as can_process:
            if not can_process:
                return
            async with _typing(message):
                expert_result = await EXPERT_AGENT.recommend_with_clarification(
                    language=decision.language,
                    seed_text=seed_text or text,
                    filters=filters,
                    candidate_dish_ids=[],
                    round_index=0,
                )

        if expert_result.mode == "clarify":
            await state.set_state(FoodBotStates.clarifying_recommendation)
            await set_reco_candidates(state, expert_result.candidate_dish_ids)

            clar = expert_result.clarification
            cat_opts = list(getattr(clar, "category_options", []) or [])
            prop_opts = list(getattr(clar, "property_options", []) or [])

            # Avoid repeating the same options across rounds
            shown_cats = set(await get_reco_shown_category_options(state))
            shown_props = set(await get_reco_shown_property_options(state))
            cat_opts = [c for c in cat_opts if c not in shown_cats]
            prop_opts = [p for p in prop_opts if p not in shown_props]
            await set_reco_round(state, 1)

            question, asked = _build_manager_clarify_question(decision.language, seed_text or "", cat_opts, prop_opts, filters)
            await set_reco_last_clarify(state, cat_opts, prop_opts, asked)

            # Remember shown options so we don't repeat them next round
            if asked == "category":
                await add_reco_shown_options(state, categories=cat_opts[:3])
            elif asked == "property":
                await add_reco_shown_options(state, properties=prop_opts[:3])
            # Keep manager tone
            await _reply_manager_with_optional_clarification(message, decision.reply_text or expert_result.intro_text, question)
            return

        await state.set_state(FoodBotStates.showing_recommendations)
        await _send_expert_recommendations(message, expert_result)
        await set_last_recommendations(state, [x.dish_id for x in expert_result.items])

        summary = _summarize_request_for_history(decision.language, filters)
        await set_history_last_request(state, summary)
        await reset_reco_flow(state)
        await set_last_clarify_category_options(state, [])
        return

    if decision.intent == "menu_availability":
        await state.set_state(FoodBotStates.showing_dish_details)

        if decision.dish_query:
            dish = _find_available_dish_strict(MENU_CATALOG, decision.dish_query)

            if dish:
                if decision.language == "ru":
                    await message.answer("Да, в TsuPosh есть такая позиция ✅")
                else:
                    await message.answer("Так, у TsuPosh є така позиція ✅")

                await message.answer(render_dish_card(dish, decision.language))
                await set_last_recommendations(state, [dish.id])
                await set_last_clarify_category_options(state, [])
                return

            # Not found -> offer analogs
            if decision.language == "ru":
                await message.answer("Сейчас такой позиции в тестовом меню нет, но могу предложить похожие варианты 🙂")
            else:
                await message.answer("Зараз такої позиції в тестовому меню немає, але можу порадити схожі варіанти 🙂")

            async with _user_processing_guard(message, decision.language) as can_process:
                if not can_process:
                    return
                async with _typing(message):
                    expert_result = await EXPERT_AGENT.suggest_analogs_for_dish_query(
                    requested_query=decision.dish_query,
                    language=decision.language,
                )
            await _send_expert_recommendations(message, expert_result)
            await set_last_recommendations(state, [x.dish_id for x in expert_result.items])
            await set_last_clarify_category_options(state, [])
            return

        # Could not extract dish name
        if decision.language == "ru":
            await message.answer("Напишите, пожалуйста, название блюда, и я проверю наличие в меню.")
        else:
            await message.answer("Напиши, будь ласка, назву страви, і я перевірю наявність у меню.")
        return

    if decision.intent == "dish_question":
        await state.set_state(FoodBotStates.showing_dish_details)
        if decision.dish_query:
            async with _user_processing_guard(message, decision.language) as can_process:
                if not can_process:
                    return
                async with _typing(message):
                    chef_result = await CHEF_AGENT.describe_dish(dish_query=decision.dish_query, language=decision.language)
            if chef_result.found_in_menu and chef_result.details_text:
                await message.answer(chef_result.intro_text)
                await message.answer(chef_result.details_text)
                if chef_result.dish_id:
                    await set_last_recommendations(state, [chef_result.dish_id])
                    await set_last_clarify_category_options(state, [])
                return
            async with _user_processing_guard(message, decision.language) as can_process:
                if not can_process:
                    return
                async with _typing(message):
                    expert_result = await EXPERT_AGENT.suggest_analogs_for_dish_query(requested_query=decision.dish_query, language=decision.language)
            await _send_expert_recommendations(message, expert_result)
            await set_last_recommendations(state, [x.dish_id for x in expert_result.items])
            await set_last_clarify_category_options(state, [])
            return

        fallback_q = decision.clarifying_question or ("Напишите, пожалуйста, название блюда, и я расскажу о нём 👨‍🍳" if decision.language == "ru" else "Напиши, будь ласка, назву страви, і я розповім про неї 👨‍🍳")
        await _reply_manager_with_optional_clarification(message, decision.reply_text, fallback_q)
        return



    

    # Friendly memory: remind last request only when user greets (not on thanks/bye)
    if decision.intent == "smalltalk":
        last = await get_history_last_request(state)
        if last and _looks_like_greeting(text) and not _looks_like_thanks_or_bye(text):
            if decision.language == "ru":
                decision.reply_text = f"{decision.reply_text}\n\nКстати, в прошлый раз вы искали: {last}. Хотите что-то похожее или попробуем новое?"
            else:
                decision.reply_text = f"{decision.reply_text}\n\nДо речі, минулого разу ти шукав(ла): {last}. Хочеш щось схоже чи спробуємо новеньке?"
    await state.set_state(FoodBotStates.idle)
    await _reply_manager_with_optional_clarification(message, decision.reply_text, decision.clarifying_question if decision.needs_clarification else None)