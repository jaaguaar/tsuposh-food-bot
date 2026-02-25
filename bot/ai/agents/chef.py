from __future__ import annotations

import json
import logging

from bot.ai.client import AiClientError, AzureAiInferenceClient
from bot.ai.schemas import ChefDecision
from bot.menu.loader import load_menu
from bot.menu.models import Dish, MenuCatalog
from bot.menu.search import find_dish_by_id, find_dishes_by_name

logger = logging.getLogger(__name__)


class ChefAgent:
    def __init__(self, catalog: MenuCatalog | None = None, ai_client: AzureAiInferenceClient | None = None) -> None:
        self.catalog = catalog or load_menu()
        self.ai_client = ai_client or AzureAiInferenceClient()

    async def describe_dish(self, dish_query: str, language: str = "ua") -> ChefDecision:
        lang = "ru" if language == "ru" else "ua"
        query = (dish_query or "").strip()
        if not query:
            return self._not_found(lang, query)

        found = find_dishes_by_name(self.catalog, query, lang)
        if not found:
            other_lang = "ua" if lang == "ru" else "ru"
            found = find_dishes_by_name(self.catalog, query, other_lang)
        if not found:
            found = self._fallback_keyword_lookup(query)
        if not found:
            return self._not_found(lang, query)

        dish = found[0]
        intro = "З радістю від TsuPosh 👨‍🍳 Коротко про страву:" if lang == "ua" else "С удовольствием от TsuPosh 👨‍🍳 Коротко о блюде:"
        return ChefDecision(
            language=lang,
            found_in_menu=True,
            intro_text=intro,
            dish_id=dish.id,
            dish_query=query,
            details_text=await self._build_details_text(dish, lang),
            confidence=0.9 if self.ai_client.enabled else 0.85,
        )

    async def describe_dish_by_id(self, dish_id: str, language: str = "ua") -> ChefDecision:
        lang = "ru" if language == "ru" else "ua"
        dish = find_dish_by_id(self.catalog, dish_id)
        if not dish:
            return self._not_found(lang, dish_id)
        intro = "З радістю від TsuPosh 👨‍🍳 Коротко про страву:" if lang == "ua" else "С удовольствием от TsuPosh 👨‍🍳 Коротко о блюде:"
        return ChefDecision(
            language=lang,
            found_in_menu=True,
            intro_text=intro,
            dish_id=dish.id,
            dish_query=dish_id,
            details_text=await self._build_details_text(dish, lang),
            confidence=0.95 if self.ai_client.enabled else 0.9,
        )

    async def _build_details_text(self, dish: Dish, lang: str) -> str:
        ai_text = await self._build_details_with_ai(dish, lang)
        if ai_text:
            return ai_text
        return self._build_details_text_fallback(dish, lang)

    async def _build_details_with_ai(self, dish: Dish, lang: str) -> str | None:
        if not self.ai_client.enabled:
            return None
        try:
            payload = {
                "brand": "TsuPosh",
                "language": lang,
                "dish": {
                    "name": dish.name.ru if lang == "ru" else dish.name.ua,
                    "category": dish.category,
                    "description_short": dish.description_short.ru if lang == "ru" else dish.description_short.ua,
                    "description_full": dish.description_full.ru if lang == "ru" else dish.description_full.ua,
                    "ingredients": dish.ingredients,
                    "spice_level": dish.spice_level,
                    "is_vegetarian": dish.is_vegetarian,
                    "pair_with": (dish.pair_with.ru if lang == "ru" else dish.pair_with.ua) if dish.pair_with else None,
                    "tags": dish.tags,
                },
            }
            system_prompt = (
                "Ти шеф-консультант закладу TsuPosh. Поверни СТРОГО валідний JSON без markdown: "
                "{intro:string, cooking:string, taste:string, fun_fact:string}. "
                "Мова відповіді тільки ua або ru згідно з вхідними даними."
            )
            user_prompt = json.dumps(payload, ensure_ascii=False)
            data = await self.ai_client.acomplete_json(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=550)
            intro = str(data.get("intro") or "").strip()
            cooking = str(data.get("cooking") or "").strip()
            taste = str(data.get("taste") or "").strip()
            fun_fact = str(data.get("fun_fact") or "").strip()
            if not all([intro, cooking, taste, fun_fact]):
                return None
            return self._format_details_text(
                dish=dish,
                lang=lang,
                intro=intro,
                cooking=cooking,
                taste=taste,
                fun_fact=fun_fact,
            )
        except (AiClientError, Exception) as ex:
            logger.warning("Chef AI fallback: %s", ex)
            return None

    def _build_details_text_fallback(self, dish: Dish, lang: str) -> str:
        return self._format_details_text(
            dish=dish,
            lang=lang,
            intro=(dish.description_short.ru if lang == "ru" else dish.description_short.ua),
            cooking=self._cooking_style_ru(dish) if lang == "ru" else self._cooking_style_ua(dish),
            taste=self._taste_profile_ru(dish) if lang == "ru" else self._taste_profile_ua(dish),
            fun_fact=self._fun_fact_ru(dish) if lang == "ru" else self._fun_fact_ua(dish),
        )

    def _format_details_text(self, dish: Dish, lang: str, intro: str, cooking: str, taste: str, fun_fact: str) -> str:
        name = dish.name.ru if lang == "ru" else dish.name.ua
        pair_with = (dish.pair_with.ru if lang == "ru" else dish.pair_with.ua) if dish.pair_with else None
        if lang == "ru":
            lines = [
                f"🍽 <b>{name}</b>",
                intro,
                f"👨‍🍳 <b>Как готовится:</b> {cooking}",
                f"😋 <b>Вкус:</b> {taste}",
                f"📖 <b>Интересно:</b> {fun_fact}",
            ]
            if pair_with:
                lines.append(f"🥢 <b>Хорошо сочетается с:</b> {pair_with}")
        else:
            lines = [
                f"🍽 <b>{name}</b>",
                intro,
                f"👨‍🍳 <b>Як готується:</b> {cooking}",
                f"😋 <b>Смак:</b> {taste}",
                f"📖 <b>Цікаво:</b> {fun_fact}",
            ]
            if pair_with:
                lines.append(f"🥢 <b>Добре смакує з:</b> {pair_with}")
        return "\n".join(lines)

    def _not_found(self, lang: str, query: str | None) -> ChefDecision:
        text = "Поки не можу знайти цю страву в тестовому меню TsuPosh." if lang == "ua" else "Пока не могу найти это блюдо в тестовом меню TsuPosh."
        return ChefDecision(language=lang, found_in_menu=False, intro_text=text, dish_query=query, details_text=None, confidence=0.4)

    def _fallback_keyword_lookup(self, query: str) -> list[Dish]:
        q = query.lower()
        aliases = {
            "том ям": ["tom-yum"], "tom yum": ["tom-yum"], "місо": ["miso"], "мисо": ["miso"],
            "рамен": ["ramen"], "ramen": ["ramen"], "пад тай": ["pad-thai"], "pad thai": ["pad-thai"],
            "каліфорнія": ["california"], "калифорния": ["california"], "гедза": ["gyoza"], "гёдза": ["gyoza"],
            "моті": ["mochi"], "моти": ["mochi"], "матча": ["matcha"],
        }
        matched_ids: list[str] = []
        for key, id_parts in aliases.items():
            if key in q:
                for dish in self.catalog.dishes:
                    if any(part in dish.id for part in id_parts):
                        matched_ids.append(dish.id)
        result: list[Dish] = []
        seen: set[str] = set()
        for dish_id in matched_ids:
            if dish_id in seen:
                continue
            seen.add(dish_id)
            dish = find_dish_by_id(self.catalog, dish_id)
            if dish:
                result.append(dish)
        return result

    def _cooking_style_ua(self, dish: Dish) -> str:
        if dish.category in {"wok", "noodles", "rice"}:
            return "Швидко обсмажується на сильному вогні з соусом та основними інгредієнтами."
        if dish.category == "soups":
            return "Готується на ароматній основі зі спеціями та подається гарячою."
        if dish.category == "sushi_rolls":
            return "Формується з рису та начинки, загортається в норі та нарізається на шматочки."
        if dish.category == "snacks":
            return "Подається як закуска: обсмажується або готується до соковитої текстури."
        if dish.category == "desserts":
            return "Готується як легкий десерт із м’якою текстурою та солодким акцентом."
        if dish.category == "drinks":
            return "Готується шляхом змішування основи напою з додатковими інгредієнтами."
        return "Готується за фірмовим рецептом із добірних інгредієнтів."

    def _cooking_style_ru(self, dish: Dish) -> str:
        if dish.category in {"wok", "noodles", "rice"}:
            return "Быстро обжаривается на сильном огне с соусом и основными ингредиентами."
        if dish.category == "soups":
            return "Готовится на ароматной основе со специями и подается горячим."
        if dish.category == "sushi_rolls":
            return "Формируется из риса и начинки, заворачивается в нори и нарезается на кусочки."
        if dish.category == "snacks":
            return "Подается как закуска: обжаривается или готовится до сочной текстуры."
        if dish.category == "desserts":
            return "Готовится как легкий десерт с мягкой текстурой и сладким акцентом."
        if dish.category == "drinks":
            return "Готовится путем смешивания основы напитка с дополнительными ингредиентами."
        return "Готовится по фирменному рецепту из качественных ингредиентов."

    def _taste_profile_ua(self, dish: Dish) -> str:
        return ("помітно пікантний" if dish.spice_level >= 2 else "помірно пікантний" if dish.spice_level == 1 else "м’який за гостротою") + "."

    def _taste_profile_ru(self, dish: Dish) -> str:
        return ("заметно пикантный" if dish.spice_level >= 2 else "умеренно пикантный" if dish.spice_level == 1 else "мягкий по остроте") + "."

    def _fun_fact_ua(self, dish: Dish) -> str:
        if dish.category == "soups":
            return "Азійські супи цінують за баланс смаку: кисле, солоне, гостре та ароматне."
        if dish.category == "sushi_rolls":
            return "Роли часто обирають для знайомства з японською кухнею завдяки збалансованому смаку."
        return "У TsuPosh цю страву зручно обрати як швидкий, але яскравий за смаком варіант."

    def _fun_fact_ru(self, dish: Dish) -> str:
        if dish.category == "soups":
            return "Азиатские супы ценят за баланс вкуса: кислое, солёное, острое и ароматное."
        if dish.category == "sushi_rolls":
            return "Роллы часто выбирают для знакомства с японской кухней благодаря сбалансированному вкусу."
        return "В TsuPosh это удобный выбор, если хочется быстро и вкусно."
