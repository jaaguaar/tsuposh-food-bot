from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Optional

from bot.ai.client import AiClientError, AzureAiInferenceClient
from bot.ai.schemas import ExpertDecision, ExpertRecommendationItem, ExpertClarification
from bot.menu.loader import load_menu
from bot.menu.models import Dish, MenuCatalog
from bot.menu.search import find_dish_by_id, find_dishes_by_name, recommend_simple, search_by_category

def _diversify_list(dishes: list['Dish'], limit: int = 3) -> list['Dish']:
    """Pick up to `limit` dishes, preferring unique categories first."""
    selected: list['Dish'] = []
    used_categories: set[str] = set()
    for d in dishes:
        if len(selected) >= limit:
            break
        if getattr(d, 'category', None) not in used_categories:
            selected.append(d)
            used_categories.add(getattr(d, 'category', None))
    for d in dishes:
        if len(selected) >= limit:
            break
        if d.id not in {x.id for x in selected}:
            selected.append(d)
    return selected[:limit]


logger = logging.getLogger(__name__)


@dataclass
class ExpertInput:
    language: str = "ua"
    user_text: str = ""
    preferences:Optional[dict[str, Any]] = None
    requested_dish_query:Optional[str] = None
    allowed_categories:Optional[list[str]] = None
    excluded_categories:Optional[list[str]] = None


class ExpertAgent:
    def __init__(self, catalog:Optional[MenuCatalog] = None, ai_client:Optional[AzureAiInferenceClient] = None) -> None:
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

        dishes = getattr(self, '_diversify', _diversify_list)(dishes, limit=3)

        if not dishes:
            return self._empty_recommendation(lang)

        items = [ExpertRecommendationItem(dish_id=d.id, reason=getattr(self, '_build_reason_for_recommendation', _build_reason_fallback)(d, lang, category, include_tags, prefs, data.user_text)) for d in dishes]
        intro = self._build_intro(lang, len(items), mode="recommendation")

        ai_intro, ai_reasons = await self._refine_with_ai(lang=lang, user_text=data.user_text, prefs=prefs, dishes=dishes, mode="recommendation")
        if ai_intro:
            intro = ai_intro
        for i, reason in enumerate(ai_reasons[:len(items)]):
            if reason:
                items[i].reason = reason

        return ExpertDecision(language=lang, mode="recommendation", intro_text=intro, items=items, confidence=0.86 if ai_intro or ai_reasons else 0.75)

    async def recommend_with_clarification(
        self,
        *,
        language: str,
        seed_text: str,
        filters:Optional[dict[str, Any]],
        candidate_dish_ids:Optional[list[str]],
        round_index: int,
    ) -> ExpertDecision:
        """Recommendation flow that can return a clarification request (max 3 rounds controlled by manager).

        This method never asks the user directly. It either:
          - returns mode='recommendation' with up to 3 items, or
          - returns mode='clarify' with candidate_dish_ids + structured options.
        """
        lang = "ru" if language == "ru" else "ua"
        seed_text = (seed_text or "").strip()
        filters = dict(filters or {})
        round_index = int(round_index or 0)

        relaxed = bool(filters.get("_relaxed"))
        # Internal control keys should not affect filtering logic
        filters.pop("_relaxed", None)
        filters.pop("_final_stage", None)
        filters.pop("_restart_stage", None)

        base_dishes: list[Dish] = []
        if candidate_dish_ids:
            for did in candidate_dish_ids:
                d = find_dish_by_id(self.catalog, did)
                if d and d.available:
                    base_dishes.append(d)
        if not base_dishes:
            base_dishes = [d for d in self.catalog.dishes if d.available]

        # Apply filters
        filtered = self._apply_reco_filters(base_dishes, filters)

        # Decide whether to clarify
        if not filtered:
            # In relaxed mode we should stop asking questions and give "closest matches".
            if relaxed:
                minimal = {k: v for k, v in filters.items() if k in {"vegetarian", "avoid_ingredients"}}
                broadened = self._apply_reco_filters(base_dishes, minimal) or base_dishes
                candidates = self._rank_for_seed(broadened, seed_text, filters)[:12]
                picks = _diversify_list(candidates, limit=3)
                intro = ("Не знайшов точного збігу, але ось найближчі варіанти 🙂" if lang == "ua"
                         else "Не нашёл точного совпадения, но вот ближайшие варианты 🙂")
                items = []
                for d in picks:
                    reason = _build_reason_fallback(d, lang, None, [], filters, seed_text)
                    items.append(ExpertRecommendationItem(dish_id=d.id, title=d.name.ua if lang=="ua" else d.name.ru, reason=reason))
                intro2 = self._build_intro(lang, len(items), mode="recommendation")
                return ExpertDecision(
                    language=lang,
                    mode="recommendation",
                    intro_text=intro2 + "\n" + intro,
                    items=items,
                    clarification=None,
                    candidate_dish_ids=[d.id for d in candidates],
                    confidence=0.55,
                )

            # Build a broader candidate pool (keep hard constraints but loosen category/tag matching) (keep hard constraints but loosen category/tag matching)
            broadened = self._apply_reco_filters(base_dishes, {k: v for k, v in filters.items() if k in {"vegetarian", "avoid_ingredients", "spice_max"}})
            if not broadened:
                broadened = base_dishes

            candidates = self._rank_for_seed(broadened, seed_text, filters)[:12]
            candidates = self._limit_candidates_to_top_categories(candidates, max_categories=3)
            clarify = self._build_clarification_options(candidates)
            intro = "Щоб порадити найкраще, уточни, будь ласка, кілька деталей 🙂" if lang == "ua" else "Чтобы посоветовать лучше, уточните, пожалуйста, пару деталей 🙂"
            return ExpertDecision(
                language=lang,
                mode="clarify",
                intro_text=intro,
                items=[],
                clarification=clarify,
                candidate_dish_ids=[d.id for d in candidates],
                confidence=0.7,
            )

        # Too many results: ask to narrow (only while we still have rounds left)
        if (not relaxed) and len(filtered) > 5 and round_index < 3:
            candidates = self._rank_for_seed(filtered, seed_text, filters)[:12]
            candidates = self._limit_candidates_to_top_categories(candidates, max_categories=3)
            clarify = self._build_clarification_options(candidates)
            intro = "Є кілька хороших варіантів — давай трохи звузимо вибір 🙂" if lang == "ua" else "Есть несколько хороших вариантов — давайте чуть сузим выбор 🙂"
            return ExpertDecision(
                language=lang,
                mode="clarify",
                intro_text=intro,
                items=[],
                clarification=clarify,
                candidate_dish_ids=[d.id for d in candidates],
                confidence=0.75,
            )

        # Final recommendations
        ranked = self._rank_for_seed(filtered, seed_text, filters)
        final = getattr(self, '_diversify', _diversify_list)(ranked, limit=3)
        if not final:
            final = ranked[:3]

        items = [
            ExpertRecommendationItem(
                dish_id=d.id,
                reason=getattr(self, '_build_reason_for_recommendation', _build_reason_fallback)(d, lang, None, [], {}, seed_text),
            )
            for d in final
        ]
        intro = self._build_intro(lang, len(items), mode="recommendation")
        ai_intro, ai_reasons = await self._refine_with_ai(lang=lang, user_text=seed_text, prefs=filters, dishes=final, mode="recommendation")
        if ai_intro:
            intro = ai_intro
        for i, reason in enumerate(ai_reasons[:len(items)]):
            if reason:
                items[i].reason = reason

        return ExpertDecision(language=lang, mode="recommendation", intro_text=intro, items=items, confidence=0.86 if ai_intro or ai_reasons else 0.78)

    def _apply_reco_filters(self, dishes: list[Dish], filters: dict[str, Any]) -> list[Dish]:
        wanted_categories = set((filters.get("wanted_categories") or []))
        not_categories = set((filters.get("not_categories") or []))
        wanted_props = set((filters.get("wanted_properties") or []))
        required_props = set((filters.get("required_properties") or []))
        not_props = set((filters.get("not_properties") or []))
        spice_max = filters.get("spice_max")
        vegetarian_only = bool(filters.get("vegetarian") is True)
        avoid_ingredients = [str(x).strip().lower() for x in (filters.get("avoid_ingredients") or []) if str(x).strip()]
        with_rice = filters.get("with_rice")

        result: list[Dish] = []
        for d in dishes:
            if wanted_categories and d.category not in wanted_categories:
                continue
            if not_categories and d.category in not_categories:
                continue
            if vegetarian_only and not d.is_vegetarian:
                continue
            if spice_max is not None:
                try:
                    if d.spice_level > int(spice_max):
                        continue
                except Exception:
                    pass

            # properties -> tags/spice/veg
            dish_tags = {t.strip().lower() for t in (d.tags or [])}
            dish_props: set[str] = set()
            if d.is_vegetarian:
                dish_props.add("vegetarian")
            if d.spice_level >= 2 or "spicy" in dish_tags or "hot" in dish_tags:
                dish_props.add("spicy")
            if d.spice_level <= 1:
                dish_props.add("not_spicy")
            for p in ["chicken", "beef", "seafood", "shrimp", "fish", "tofu", "sweet"]:
                if p in dish_tags:
                    dish_props.add("seafood" if p in {"shrimp", "fish"} else p)
            if "dessert" in dish_tags:
                dish_props.add("sweet")

            if required_props and not required_props.issubset(dish_props):
                continue
            if wanted_props and not (dish_props & wanted_props):
                continue
            if not_props and (dish_props & not_props):
                continue

            
            # rice / carb constraint (best-effort heuristic)
            if with_rice is True:
                blob = " ".join((d.ingredients or [])).lower()
                try:
                    blob += " " + (d.description_short.ua or "").lower() + " " + (d.description_short.ru or "").lower()
                    blob += " " + (d.description_full.ua or "").lower() + " " + (d.description_full.ru or "").lower()
                except Exception:
                    pass
                if d.pair_with:
                    try:
                        blob += " " + (d.pair_with.ua or "").lower() + " " + (d.pair_with.ru or "").lower()
                    except Exception:
                        pass
                # category/tag based hints
                dish_tags_lower = {t.strip().lower() for t in (d.tags or [])}
                has_rice = (d.category == "rice") or ("rice" in dish_tags_lower) or ("рис" in blob) or ("рисов" in blob)
                if not has_rice:
                    continue
            elif with_rice is False:
                blob = " ".join((d.ingredients or [])).lower()
                if "рис" in blob or "рисов" in blob:
                    continue

            if avoid_ingredients:
                ing = " ".join((d.ingredients or [])).lower()
                if any(x in ing for x in avoid_ingredients):
                    continue

            result.append(d)

        return result

    def _rank_for_seed(self, dishes: list[Dish], seed_text: str, filters: dict[str, Any]) -> list[Dish]:
        tokens = [t for t in re.split(r"[^\wа-яА-ЯіїєґІЇЄҐ]+", (seed_text or "").lower()) if t]
        wanted_categories = set((filters.get("wanted_categories") or []))
        wanted_props = set((filters.get("wanted_properties") or []))

        def score(d: Dish) -> int:
            s = 0
            if d.category in wanted_categories:
                s += 6
            dish_tags = {t.lower() for t in (d.tags or [])}
            if "spicy" in wanted_props and (d.spice_level >= 2 or "spicy" in dish_tags or "hot" in dish_tags):
                s += 3
            if "not_spicy" in wanted_props and d.spice_level <= 1:
                s += 3
            if wanted_props:
                if ("chicken" in wanted_props and "chicken" in dish_tags):
                    s += 3
                if ("beef" in wanted_props and "beef" in dish_tags):
                    s += 3
                if ("seafood" in wanted_props and ({"seafood","shrimp","fish"} & dish_tags)):
                    s += 3
                if ("vegetarian" in wanted_props and d.is_vegetarian):
                    s += 3
                if ("sweet" in wanted_props and ("sweet" in dish_tags or "dessert" in dish_tags)):
                    s += 2

            hay = " ".join([
                d.name.ua, d.name.ru,
                d.description_short.ua, d.description_short.ru,
                " ".join(d.ingredients or []),
                " ".join(d.tags or []),
            ]).lower()
            for t in tokens:
                if len(t) <= 2:
                    continue
                if t in hay:
                    s += 1
            # slight preference for available + shorter list already ensured
            s += max(0, 3 - d.spice_level)  # mildly prefer less spicy by default
            return s

        return sorted(dishes, key=score, reverse=True)


    def _build_intro(self, lang: str, n_items: int, mode: str = "recommendation") -> str:
        # Short friendly intro for expert answers
        if lang == "ru":
            if mode == "clarify":
                return "Ок 🙂 Чтобы попасть точнее, уточню пару моментов."
            return f"Вот что могу посоветовать (вариантов: {n_items}) 🙂"
        else:
            if mode == "clarify":
                return "Ок 🙂 Щоб влучити точніше, уточню пару моментів."
            return f"Ось що можу порадити (варіантів: {n_items}) 🙂"

    def _limit_candidates_to_top_categories(self, candidates: list[Dish], max_categories: int = 3) -> list[Dish]:
        """Limit candidate pool to dishes from top-N categories to keep clarification short."""
        if not candidates:
            return []
        cats = [c.category for c in candidates if getattr(c, "category", None)]
        uniq_cats: list[str] = []
        for c in cats:
            if c not in uniq_cats:
                uniq_cats.append(c)
        if len(uniq_cats) <= max_categories:
            return candidates
        # Choose top categories by frequency
        freq: dict[str, int] = {}
        for c in cats:
            freq[c] = freq.get(c, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_categories]
        top_set = {c for c, _ in top}
        return [d for d in candidates if getattr(d, "category", None) in top_set]

    def _build_clarification_options(self, dishes: list[Dish]) -> ExpertClarification:
        """Build clarification options (categories + properties) from the current candidate pool."""
        # Category options: up to 6 unique, later UI will show up to 3 at once
        cats: list[str] = []
        for d in dishes:
            c = getattr(d, "category", None)
            if c and c not in cats:
                cats.append(c)
            if len(cats) >= 6:
                break

        # Property options are generic, based on tags/spice/veg signals
        props: list[str] = []
        # spicy vs not spicy suggestion
        spicy_cnt = sum(1 for d in dishes if getattr(d, "spice_level", 0) >= 3)
        mild_cnt = sum(1 for d in dishes if getattr(d, "spice_level", 0) <= 1)
        if spicy_cnt and "spicy" not in props:
            props.append("spicy")
        if mild_cnt and "not_spicy" not in props:
            props.append("not_spicy")

        # proteins / types (from tags where available)
        tag_set: set[str] = set()
        for d in dishes:
            for t in (getattr(d, "tags", None) or []):
                tag_set.add(str(t).lower())

        def _add(p: str) -> None:
            if p not in props:
                props.append(p)

        if "chicken" in tag_set:
            _add("chicken")
        if "beef" in tag_set:
            _add("beef")
        if {"fish","shrimp","seafood"} & tag_set:
            _add("seafood")
        if any(getattr(d, "is_vegetarian", False) for d in dishes):
            _add("vegetarian")
        if "sweet" in tag_set or "dessert" in tag_set:
            _add("sweet")

        return ExpertClarification(
            category_options=cats,
            property_options=props[:8],
        )

    def _post_filter_required_constraints(
        self,
        dishes: list[Dish],
        *,
        prefs: dict[str, object],
        user_text: str,
        vegetarian_only: bool,
    ) -> list[Dish]:
        """Hard constraints that must always apply (even in relaxed mode)."""
        if not dishes:
            return []
        res = list(dishes)

        # vegetarian constraint from prefs
        if vegetarian_only:
            res = [d for d in res if getattr(d, "is_vegetarian", False)]

        # avoid ingredients (string contains)
        avoid = prefs.get("avoid_ingredients") or []
        avoid = [str(x).lower().strip() for x in (avoid or []) if str(x).strip()]
        if avoid:
            out: list[Dish] = []
            for d in res:
                ing = " ".join([str(x).lower() for x in (getattr(d, "ingredients", None) or [])])
                if any(a and a in ing for a in avoid):
                    continue
                out.append(d)
            res = out

        return res

    def _limit_candidates_to_top_categories(self, candidates: list[Dish], max_categories: int = 3) -> list[Dish]:
        """If candidates span too many categories, keep only dishes from the top-N categories.

        We keep ranking order (candidates are already ranked), and choose categories by frequency
        within the candidate set. This prevents the manager from asking about a long list of categories
        and matches the UX requirement: show/keep only up to 3 categories in clarification.
        """
        if not candidates:
            return []

        cats = [c.category for c in candidates if c.category]
        uniq_cats: list[str] = []
        for c in cats:
            if c not in uniq_cats:
                uniq_cats.append(c)
        if len(uniq_cats) <= max_categories:
            return candidates

        # Frequency-based top categories
        freq: dict[str, int] = {}
        for c in cats:
            freq[c] = freq.get(c, 0) + 1
        top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:max_categories]
        allowed = {k for k, _ in top}

        return [d for d in candidates if d.category in allowed]

    def _build_clarification_options(self, dishes: list[Dish]) -> ExpertClarification:
        categories: list[str] = []
        for d in dishes:
            if d.category and d.category not in categories:
                categories.append(d.category)

        prop_set: set[str] = set()
        for d in dishes:
            tags = {t.lower() for t in (d.tags or [])}
            if d.is_vegetarian:
                prop_set.add("vegetarian")
            if d.spice_level >= 2 or "spicy" in tags or "hot" in tags:
                prop_set.add("spicy")
            if d.spice_level <= 1:
                prop_set.add("not_spicy")
            if "chicken" in tags:
                prop_set.add("chicken")
            if "beef" in tags:
                prop_set.add("beef")
            if {"seafood", "shrimp", "fish"} & tags:
                prop_set.add("seafood")
            if "sweet" in tags or "dessert" in tags:
                prop_set.add("sweet")

        # Keep it compact for the manager (best UX in chat)
        property_options = [
            p for p in ["spicy", "not_spicy", "chicken", "seafood", "beef", "vegetarian", "sweet"]
            if p in prop_set
        ][:6]

        return ExpertClarification(category_options=categories, property_options=property_options)

    async def list_category(self, *, language: str, category: str, limit: int = 12) -> ExpertDecision:
        """Return a list of dishes from a specific category as cards."""
        lang = "ru" if language == "ru" else "ua"
        category = (category or "").strip()
        dishes = [d for d in self.catalog.dishes if d.available and d.category == category]
        if not dishes:
            intro = "У цій категорії зараз нічого не бачу в меню 🙂" if lang == "ua" else "В этой категории сейчас ничего не вижу в меню 🙂"
            return ExpertDecision(language=lang, mode="recommendation", intro_text=intro, items=[], confidence=0.8)

        dishes_sorted = sorted(dishes, key=lambda d: (d.name_ua or d.name_ru or d.id))
        items = [
            ExpertRecommendationItem(
                dish_id=d.id,
                reason=("Ось що є в цій категорії." if lang == "ua" else "Вот что есть в этой категории."),
            )
            for d in dishes_sorted[: max(1, int(limit))]
        ]
        intro = "Ось що є в цій категорії 👇" if lang == "ua" else "Вот что есть в этой категории 👇"
        return ExpertDecision(language=lang, mode="recommendation", intro_text=intro, items=items, confidence=0.9)

    async def suggest_analogs_for_dish_query(self, requested_query: str, language: str = "ua") -> ExpertDecision:
        lang = "ru" if language == "ru" else "ua"
        query = (requested_query or "").strip()
        if not query:
            return self._not_found(lang, requested_query)

        found = find_dishes_by_name(self.catalog, query, lang)
        if not found:
            found = find_dishes_by_name(self.catalog, query, "ua" if lang == "ru" else "ru")
        if found:
            items = [
                ExpertRecommendationItem(
                    dish_id=d.id,
                    reason=("Є в меню TsuPosh." if lang == "ua" else "Есть в меню TsuPosh."),
                )
                for d in found[:3]
            ]
            return ExpertDecision(
                language=lang,
                mode="recommendation",
                intro_text=("Схоже, ця страва є в меню TsuPosh 👌" if lang == "ua" else "Похоже, это блюдо есть в меню TsuPosh 👌"),
                items=items,
                requested_dish_query=query,
                confidence=0.9,
            )

        analogs = self._analogs_by_keywords(query, lang)
        if analogs:
            analogs = getattr(self, '_diversify', _diversify_list)(analogs, limit=3)
            items = [
                ExpertRecommendationItem(
                    dish_id=d.id,
                    reason=("Схожа страва за стилем або смаком." if lang == "ua" else "Похожее блюдо по стилю или вкусу."),
                )
                for d in analogs[:3]
            ]
            intro = (
                "У тестовому меню TsuPosh такої страви не бачу, але ось схожі варіанти 🙂"
                if lang == "ua"
                else "В тестовом меню TsuPosh такого блюда не вижу, но вот похожие варианты 🙂"
            )
            ai_intro, ai_reasons = await self._refine_with_ai(lang=lang, user_text=query, prefs={}, dishes=analogs[:3], mode="analogs")
            if ai_intro:
                intro = ai_intro
            for i, reason in enumerate(ai_reasons[: len(items)]):
                if reason:
                    items[i].reason = reason
            return ExpertDecision(
                language=lang,
                mode="analogs",
                intro_text=intro,
                items=items,
                requested_dish_query=query,
                confidence=0.72,
            )

        return self._not_found(lang, query)

    async def _refine_with_ai(self, *, lang: str, user_text: str, prefs: dict[str, object], dishes: list[Dish], mode: str) -> tuple[str | None, list[str]]:
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
            data = await self.ai_client.acomplete_json(
                system_prompt=system_prompt,
                user_prompt=__import__('json').dumps(payload, ensure_ascii=False),
                max_tokens=700,
            )
            intro_text = str(data.get("intro_text") or "").strip() or None
            reasons_raw = data.get("reasons")
            reasons = [str(x).strip() for x in reasons_raw] if isinstance(reasons_raw, list) else []
            return intro_text, reasons
        except (AiClientError, Exception) as ex:
            logger.warning("Expert AI phrasing fallback: %s", ex)
            return None, []

    def _post_filter_required_constraints(self, dishes: list[Dish], *, prefs: dict[str, object], user_text: str, vegetarian_only: bool) -> list[Dish]:
        result = list(dishes)
        protein = str(prefs.get("protein") or "").strip().lower()
        pnorm = protein.replace("’", "'").replace("`", "'")

        tag_map = {
            "курка": ["chicken"], "курицей": ["chicken"], "курица": ["chicken"],
            "яловичина": ["beef"], "говядина": ["beef"],
            "креветка": ["shrimp"], "креветки": ["shrimp"], "морепродукти": ["shrimp", "seafood"], "морепродукты": ["shrimp", "seafood"],
            "лосось": ["salmon"], "тунець": ["tuna"], "тунец": ["tuna"],
            "риба": ["fish", "salmon", "tuna"], "рыба": ["fish", "salmon", "tuna"],
        }
        if pnorm in {"м'ясо", "мясо", "з м'ясом", "с мясом"}:
            result = [d for d in result if ({"chicken", "beef"} & {t.lower() for t in (d.tags or [])})]
        elif pnorm in tag_map:
            wanted = set(tag_map[pnorm])
            result = [d for d in result if wanted & {t.lower() for t in (d.tags or [])}]

        if vegetarian_only:
            result = [d for d in result if d.is_vegetarian]

        # Enforce spice intent strictly (recommend_simple uses max<= and may be too broad for spicy requests)
        if self._text_means_spicy(user_text) or str(prefs.get("spice_level") or "").strip().lower() in {"spicy", "hot", "гостре", "острое"}:
            result = [d for d in result if d.spice_level >= 2 or ("spicy" in {t.lower() for t in (d.tags or [])})]

        avoid_ingredients = self._normalize_string_list(prefs.get("avoid_ingredients"))
        if avoid_ingredients:
            lowered = [x.lower() for x in avoid_ingredients]
            def ok(d):
                hay = " ".join((d.ingredients or [])).lower()
                return not any(x and x in hay for x in lowered)
            result = [d for d in result if ok(d)]

        return result
def _build_reason_fallback(dish: 'Dish', lang: str, category:Optional[str], include_tags: list[str], prefs: dict, user_text: str) -> str:
    # Lightweight fallback if class method is missing for any reason.
    # Keep it short and conversational.
    parts: list[str] = []
    try:
        if category and getattr(dish, "category", None) == category:
            parts.append("по категорії" if lang == "ua" else "по категории")
        tags = {t.lower() for t in getattr(dish, "tags", [])}
        if "chicken" in tags:
            parts.append("з куркою" if lang == "ua" else "с курицей")
        if "beef" in tags:
            parts.append("з яловичиною" if lang == "ua" else "с говядиной")
        if "spicy" in tags:
            parts.append("пікантне 🔥" if lang == "ua" else "остренькое 🔥")
        if prefs.get("vegetarian") is True and getattr(dish, "is_vegetarian", False):
            parts.append("вегетаріанське" if lang == "ua" else "вегетарианское")
    except Exception:
        parts = []
    if not parts:
        return ("Добре пасує під запит." if lang == "ua" else "Хорошо подходит под запрос.")
    return (("Підійде: " if lang == "ua" else "Подойдёт: ") + ", ".join(parts[:2]) + ".")