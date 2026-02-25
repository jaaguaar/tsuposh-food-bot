from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

from bot.ai.client import AiClientError, AzureAiInferenceClient
from bot.ai.schemas import ExpertDecision, ExpertRecommendationItem
from bot.menu.loader import load_menu
from bot.menu.models import Dish, MenuCatalog
from bot.menu.search import find_dishes_by_name, recommend_simple, search_by_category

logger = logging.getLogger(__name__)


@dataclass
class ExpertInput:
    language: str = "ua"
    user_text: str = ""
    preferences: dict[str, Any] | None = None
    requested_dish_query: str | None = None
    allowed_categories: list[str] | None = None
    excluded_categories: list[str] | None = None


class ExpertAgent:
    def __init__(self, catalog: MenuCatalog | None = None, ai_client: AzureAiInferenceClient | None = None) -> None:
        self.catalog = catalog or load_menu()
        self.ai_client = ai_client or AzureAiInferenceClient()

    async def recommend(self, data: ExpertInput) -> ExpertDecision:
        lang = "ru" if data.language == "ru" else "ua"
        prefs = data.preferences or {}
        category = self._map_category(prefs.get("category"), lang) or self._infer_category_from_text(data.user_text)
        max_spice = self._map_spice_level(prefs.get("spice_level"))
        vegetarian_only = bool(prefs.get("vegetarian") is True)
        if self._text_means_no_meat(data.user_text):
            vegetarian_only = True
        include_tags = self._build_include_tags(data.user_text, prefs, lang)
        exclude_ingredients = self._normalize_string_list(prefs.get("avoid_ingredients"))

        dishes = recommend_simple(
            self.catalog,
            category=category,
            max_spice_level=max_spice,
            vegetarian_only=vegetarian_only,
            include_tags=include_tags,
            exclude_ingredients=exclude_ingredients,
            limit=8,
        )
        dishes = self._post_filter_required_constraints(dishes, prefs=prefs, user_text=data.user_text, vegetarian_only=vegetarian_only)

        if not dishes:
            dishes = recommend_simple(self.catalog, category=category, vegetarian_only=vegetarian_only, limit=8)
            dishes = self._post_filter_required_constraints(dishes, prefs=prefs, user_text=data.user_text, vegetarian_only=vegetarian_only)

        # Apply optional category constraints from clarification flow (FSM)
        allowed = set(data.allowed_categories or [])
        excluded = set(data.excluded_categories or [])
        if allowed or excluded:
            dishes = [
                d for d in dishes
                if (not allowed or d.category in allowed) and d.category not in excluded
            ]

        dishes = self._diversify(dishes, limit=3)

        if not dishes:
            return self._empty_recommendation(lang)

        items = [ExpertRecommendationItem(dish_id=d.id, reason=self._build_reason_for_recommendation(d, lang, category, include_tags, prefs, data.user_text)) for d in dishes]
        intro = self._build_intro(lang, len(items), mode="recommendation")

        ai_intro, ai_reasons = await self._refine_with_ai(lang=lang, user_text=data.user_text, prefs=prefs, dishes=dishes, mode="recommendation")
        if ai_intro:
            intro = ai_intro
        for i, reason in enumerate(ai_reasons[:len(items)]):
            if reason:
                items[i].reason = reason

        return ExpertDecision(language=lang, mode="recommendation", intro_text=intro, items=items, confidence=0.86 if ai_intro or ai_reasons else 0.75)

    async def suggest_analogs_for_dish_query(self, requested_query: str, language: str = "ua") -> ExpertDecision:
        lang = "ru" if language == "ru" else "ua"
        query = (requested_query or "").strip()
        if not query:
            return self._not_found(lang, requested_query)

        found = find_dishes_by_name(self.catalog, query, lang)
        if not found:
            found = find_dishes_by_name(self.catalog, query, "ua" if lang == "ru" else "ru")
        if found:
            items = [ExpertRecommendationItem(dish_id=d.id, reason=("Є в меню TsuPosh." if lang == "ua" else "Есть в меню TsuPosh.")) for d in found[:3]]
            return ExpertDecision(language=lang, mode="recommendation", intro_text=("Схоже, ця страва є в меню TsuPosh 👌" if lang == "ua" else "Похоже, это блюдо есть в меню TsuPosh 👌"), items=items, requested_dish_query=query, confidence=0.9)

        analogs = self._analogs_by_keywords(query, lang)
        if analogs:
            analogs = self._diversify(analogs, limit=3)
            items = [ExpertRecommendationItem(dish_id=d.id, reason=("Схожа страва за стилем або смаком." if lang == "ua" else "Похожее блюдо по стилю или вкусу.")) for d in analogs[:3]]
            intro = ("У тестовому меню TsuPosh такої страви не бачу, але ось схожі варіанти 🙂" if lang == "ua" else "В тестовом меню TsuPosh такого блюда не вижу, но вот похожие варианты 🙂")
            ai_intro, ai_reasons = await self._refine_with_ai(lang=lang, user_text=query, prefs={}, dishes=analogs[:3], mode="analogs")
            if ai_intro:
                intro = ai_intro
            for i, reason in enumerate(ai_reasons[:len(items)]):
                if reason:
                    items[i].reason = reason
            return ExpertDecision(language=lang, mode="analogs", intro_text=intro, items=items, requested_dish_query=query, confidence=0.72)

        return self._not_found(lang, query)

    async def _refine_with_ai(self, *, lang: str, user_text: str, prefs: dict[str, Any], dishes: list[Dish], mode: str) -> tuple[str | None, list[str]]:
        if not self.ai_client.enabled or not dishes:
            return None, []
        try:
            payload = {
                "brand": "TsuPosh",
                "language": lang,
                "mode": mode,
                "user_text": user_text,
                "preferences": prefs,
                "dishes": [
                    {
                        "id": d.id,
                        "name": d.name.ru if lang == "ru" else d.name.ua,
                        "category": d.category,
                        "description": d.description_short.ru if lang == "ru" else d.description_short.ua,
                        "spice_level": d.spice_level,
                        "is_vegetarian": d.is_vegetarian,
                        "ingredients": d.ingredients,
                        "tags": d.tags,
                    }
                    for d in dishes
                ],
            }
            system_prompt = (
                "Ти меню-експерт мережі азійського фастфуду TsuPosh. Поверни СТРОГО валідний JSON без markdown: "
                "{intro_text:string,reasons:[string,...]}. reasons має бути тієї ж довжини, що й dishes. "
                "Фрази короткі, конкретні, без вигадування відсутніх інгредієнтів."
            )
            data = await self.ai_client.acomplete_json(system_prompt=system_prompt, user_prompt=json.dumps(payload, ensure_ascii=False), max_tokens=700)
            intro_text = str(data.get("intro_text") or "").strip() or None
            reasons_raw = data.get("reasons")
            reasons = [str(x).strip() for x in reasons_raw] if isinstance(reasons_raw, list) else []
            return intro_text, reasons
        except (AiClientError, Exception) as ex:
            logger.warning("Expert AI phrasing fallback: %s", ex)
            return None, []

    def _post_filter_required_constraints(self, dishes: list[Dish], *, prefs: dict[str, Any], user_text: str, vegetarian_only: bool) -> list[Dish]:
        result = list(dishes)
        protein = (prefs.get("protein") or "").strip().lower()
        pnorm = protein.replace("’", "'").replace("`", "'")

        tag_map = {
            "курка": ["chicken"], "курицей": ["chicken"], "курица": ["chicken"],
            "яловичина": ["beef"], "говядина": ["beef"],
            "креветка": ["shrimp"], "креветки": ["shrimp"], "морепродукти": ["shrimp","seafood"], "морепродукты": ["shrimp","seafood"],
            "лосось": ["salmon"], "тунець": ["tuna"], "тунец": ["tuna"],
            "риба": ["fish","salmon","tuna"], "рыба": ["fish","salmon","tuna"],
        }
        if pnorm in {"м'ясо", "мясо", "з м'ясом", "с мясом"}:
            # Treat meat as poultry/beef; exclude fish/seafood-only dishes.
            result = [d for d in result if ({"chicken","beef"} & {t.lower() for t in d.tags})]
        elif pnorm in tag_map:
            wanted = set(tag_map[pnorm])
            result = [d for d in result if wanted & {t.lower() for t in d.tags}]

        if vegetarian_only:
            result = [d for d in result if d.is_vegetarian]

        # Enforce spice intent strictly (recommend_simple uses max<= and may be too broad for spicy requests)
        spice = str((prefs.get("spice_level") or "")).strip().lower()
        if spice:
            if any(x in spice for x in ["не гост", "неостр", "mild", "low"]):
                result = [d for d in result if d.spice_level == 0]
            elif any(x in spice for x in ["дуже гост", "очень остр", "максим", "пекуч"]):
                result = [d for d in result if d.spice_level >= 2]
            elif any(x in spice for x in ["гост", "остр", "spicy", "high"]):
                result = [d for d in result if d.spice_level >= 1]

        return result

    def _diversify(self, dishes: list[Dish], limit: int = 3) -> list[Dish]:
        selected: list[Dish] = []
        used_categories: set[str] = set()
        for d in dishes:
            if len(selected) >= limit:
                break
            if d.category not in used_categories:
                selected.append(d)
                used_categories.add(d.category)
        for d in dishes:
            if len(selected) >= limit:
                break
            if d.id not in {x.id for x in selected}:
                selected.append(d)
        return selected[:limit]

    def _build_intro(self, lang: str, count: int, mode: str) -> str:
        if lang == "ru":
            if count == 1:
                return "Вот что могу предложить в TsuPosh 🙂"
            return f"Подобрал {count} варианта(ов) в TsuPosh — посмотрите, что ближе по вкусу 🙂"
        if count == 1:
            return "Ось що можу запропонувати в TsuPosh 🙂"
        return f"Підібрав {count} варіант(и) у TsuPosh — подивись, що більше до смаку 🙂"

    def _text_means_no_meat(self, text: str) -> bool:
        t = (text or "").lower()
        return any(x in t for x in ["без м'яса", "без мяса", "без мʼяса", "не м’яс", "вегетар"])

    def _not_found(self, lang: str, query: str | None) -> ExpertDecision:
        text = ("Поки не знайшов точної відповідності в TsuPosh. Спробуй написати назву інакше або опиши, що хочеться 🙂" if lang == "ua" else "Пока не нашёл точного совпадения в TsuPosh. Попробуйте написать название иначе или описать, чего хочется 🙂")
        return ExpertDecision(language=lang, mode="not_found", intro_text=text, items=[], requested_dish_query=query, confidence=0.4)

    def _empty_recommendation(self, lang: str) -> ExpertDecision:
        text = ("Поки не зміг підібрати варіанти під цей запит у TsuPosh. Можемо уточнити категорію або інгредієнти 🙂" if lang == "ua" else "Пока не удалось подобрать варианты под этот запрос в TsuPosh. Можем уточнить категорию или ингредиенты 🙂")
        return ExpertDecision(language=lang, mode="not_found", intro_text=text, items=[], confidence=0.3)

    def _infer_category_from_text(self, text: str) -> str | None:
        t = (text or "").lower()
        # tolerate common typos in UA/RU
        if any(x in t for x in ["локш", "лапш", "рамен", "удон", "пад тай", "локг", "локщ"]):
            return "noodles"
        if "суп" in t or "том ям" in t or "місо" in t or "мисо" in t:
            return "soups"
        if "рис" in t or "боул" in t:
            return "rice"
        if "рол" in t or "суш" in t:
            return "sushi_rolls"
        if "закуск" in t or "гедза" in t or "спрінг" in t or "спринг" in t:
            return "snacks"
        if "десерт" in t or "моті" in t or "моти" in t:
            return "desserts"
        if "нап" in t or "чай" in t or "лате" in t:
            return "drinks"
        return None

    # --- existing helpers (kept/adapted) ---
    def _map_category(self, raw: str | None, lang: str) -> str | None:
        if not raw:
            return None
        value = raw.strip().lower()
        mapping = {
            "суп": "soups", "супи": "soups", "лапша": "noodles", "локшина": "noodles",
            "рис": "rice", "боул": "rice", "боули": "rice", "вок": "wok",
            "роли": "sushi_rolls", "роллы": "sushi_rolls", "закуска": "snacks", "закуски": "snacks",
            "десерт": "desserts", "десерти": "desserts", "десерты": "desserts",
            "напій": "drinks", "напої": "drinks", "напиток": "drinks", "напитки": "drinks",
            "soups": "soups", "noodles": "noodles", "rice": "rice", "wok": "wok", "sushi_rolls": "sushi_rolls", "snacks": "snacks", "desserts": "desserts", "drinks": "drinks",
        }
        return mapping.get(value)

    def _map_spice_level(self, raw: str | None) -> int | None:
        if not raw:
            return None
        v = raw.strip().lower()
        if v in {"не гостре", "неострое", "mild", "low", "none"}:
            return 0
        if v in {"середнє", "среднее", "medium"}:
            return 1
        if v in {"гостре", "острое", "high", "spicy"}:
            return 3
        if v.isdigit():
            return max(0, min(3, int(v)))
        return None

    def _build_include_tags(self, user_text: str, prefs: dict[str, Any], lang: str) -> list[str]:
        tags: set[str] = set()
        protein = (prefs.get("protein") or "").strip().lower()
        protein_map = {"курка": "chicken", "курицей": "chicken", "куркаю": "chicken", "курица": "chicken", "яловичина": "beef", "говядина": "beef", "креветка": "shrimp", "креветки": "shrimp", "лосось": "salmon", "тунець": "tuna", "тунец": "tuna"}
        if protein in protein_map:
            tags.add(protein_map[protein])
        text = (user_text or "").lower()
        keyword_map = {"гост": "spicy", "остр": "spicy", "легк": "light", "ситн": "hearty", "хруст": "crispy", "рол": "sushi", "суш": "sushi", "рамен": "japanese", "том ям": "thai", "пад тай": "thai", "вок": "wok"}
        for key, tag in keyword_map.items():
            if key in text:
                tags.add(tag)
        return list(tags)

    def _normalize_string_list(self, values: Any) -> list[str]:
        return [v.strip() for v in values if isinstance(v, str) and v.strip()] if isinstance(values, list) else []

    def _build_reason_for_recommendation(self, dish: Dish, lang: str, category: str | None, include_tags: list[str], prefs: dict[str, Any], user_text: str) -> str:
        reasons: list[str] = []
        tag_set = {t.lower() for t in dish.tags}
        if category and dish.category == category:
            reasons.append("підходить за категорією" if lang == "ua" else "подходит по категории")
        if (prefs.get("vegetarian") is True or self._text_means_no_meat(user_text)) and dish.is_vegetarian:
            reasons.append("без м’яса / вегетаріанський варіант" if lang == "ua" else "без мяса / вегетарианский вариант")
        for tag, ua_txt, ru_txt in [
            ("spicy", "має пікантний смак", "имеет пикантный вкус"),
            ("light", "легкий варіант", "лёгкий вариант"),
            ("hearty", "ситний варіант", "сытный вариант"),
            ("chicken", "з куркою", "с курицей"),
            ("beef", "з яловичиною", "с говядиной"),
            ("shrimp", "з креветкою", "с креветкой"),
        ]:
            if tag in include_tags and tag in tag_set:
                reasons.append(ua_txt if lang == "ua" else ru_txt)
        if not reasons:
            reasons.append("пасує до вашого запиту" if lang == "ua" else "подходит под ваш запрос")
        return ", ".join(reasons[:2]).capitalize() + "."

    def _analogs_by_keywords(self, query: str, lang: str) -> list[Dish]:
        q = query.lower()
        if "фо" in q or "pho" in q:
            return search_by_category(self.catalog, "soups")[:6]
        if "суш" in q or "рол" in q:
            return search_by_category(self.catalog, "sushi_rolls")[:6]
        if "рамен" in q:
            return [d for d in self.catalog.dishes if "ramen" in d.id][:6]
        if "лапш" in q or "локшин" in q or "noodle" in q:
            return (search_by_category(self.catalog, "noodles") + search_by_category(self.catalog, "wok"))[:6]
        if "суп" in q:
            return search_by_category(self.catalog, "soups")[:6]
        return recommend_simple(self.catalog, limit=6)
