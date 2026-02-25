import asyncio
import logging
import time
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
        session_data = await get_session_data(state)
        session_lang = session_data.get("language", current_language)
        existing_prefs = session_data.get("preferences", {}) or {}

        async with _user_processing_guard(message, session_lang) as can_process:
            if not can_process:
                return
            async with _typing(message):
                followup_decision = await MANAGER_AGENT.analyze_message(text)

        merged_prefs = _merge_preferences(existing_prefs, followup_decision.extracted_preferences.model_dump())

        # Handle explicit category selection from previous clarify question (including "other")
        last_cat_options = await get_last_clarify_category_options(state)
        selected_cat = _extract_category_options_from_clarify(text, last_cat_options)
        if selected_cat:
            merged_prefs["category"] = selected_cat
            merged_prefs.pop("_exclude_categories", None)
        elif last_cat_options and _reply_is_other_choice(text, session_lang):
            merged_prefs.pop("category", None)
            merged_prefs["_exclude_categories"] = list(last_cat_options)

        # Preserve language chosen for the dialog, unless user clearly switched and manager detected it.
        session_lang = followup_decision.language or session_lang
        await set_user_language(state, session_lang)
        await set_preferences(state, merged_prefs)

        if followup_decision.intent == "off_topic":
            await state.set_state(FoodBotStates.clarifying_recommendation)
            await message.answer(followup_decision.reply_text)
            return

        async with _user_processing_guard(message, session_lang) as can_process:
            if not can_process:
                return
            async with _typing(message):
                expert_result = await EXPERT_AGENT.recommend(_expert_input_from_prefs(session_lang, text, merged_prefs))

        if not expert_result.items:
            # If still nothing found, keep clarifying instead of ending up in a dead-end.
            await state.set_state(FoodBotStates.clarifying_recommendation)
            if followup_decision.needs_clarification or followup_decision.clarifying_question:
                await set_last_clarify_category_options(state, _extract_category_options_from_clarify(followup_decision.clarifying_question))
                await _reply_manager_with_optional_clarification(message, followup_decision.reply_text, followup_decision.clarifying_question)
            else:
                await message.answer(expert_result.intro_text)
            return

        await state.set_state(FoodBotStates.showing_recommendations)
        await _send_expert_recommendations(message, expert_result)
        await set_last_recommendations(state, [x.dish_id for x in expert_result.items])
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

    if decision.intent == "recommendation_request" and not decision.needs_clarification:
        await state.set_state(FoodBotStates.showing_recommendations)
        async with _user_processing_guard(message, decision.language) as can_process:
            if not can_process:
                return
            async with _typing(message):
                expert_result = await EXPERT_AGENT.recommend(_expert_input_from_prefs(decision.language, text, decision.extracted_preferences.model_dump()))
        await _send_expert_recommendations(message, expert_result)
        await set_last_recommendations(state, [x.dish_id for x in expert_result.items])
        await set_last_clarify_category_options(state, [])
        return

    if decision.intent == "recommendation_request" and decision.needs_clarification:
        await state.set_state(FoodBotStates.clarifying_recommendation)
        cat_options = _extract_category_options_from_clarify(decision.clarifying_question)
        await set_last_clarify_category_options(state, cat_options)
        await _reply_manager_with_optional_clarification(message, decision.reply_text, decision.clarifying_question)
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

    await state.set_state(FoodBotStates.idle)
    await _reply_manager_with_optional_clarification(message, decision.reply_text, decision.clarifying_question if decision.needs_clarification else None)
