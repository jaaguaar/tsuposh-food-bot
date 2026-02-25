from __future__ import annotations

import logging

from pydantic import ValidationError

from bot.ai.client import AiClientError, AzureAiInferenceClient
from bot.ai.prompts import MANAGER_SYSTEM_PROMPT
from bot.ai.schemas import ManagerDecision
from bot.services.language import detect_language

logger = logging.getLogger(__name__)


class ManagerAgent:
    def __init__(self, ai_client: AzureAiInferenceClient | None = None) -> None:
        self.ai_client = ai_client or AzureAiInferenceClient()

    async def analyze_message(self, user_text: str) -> ManagerDecision:
        """
        Returns structured manager decision.
        Falls back to a local heuristic response if AI is unavailable or fails.
        """
        if not user_text.strip():
            lang = "ua"
            return self._fallback_empty(lang)

        if not self.ai_client.enabled:
            return self._fallback(user_text, reason="ai_disabled")

        try:
            payload = await self.ai_client.acomplete_json(
                system_prompt=MANAGER_SYSTEM_PROMPT,
                user_prompt=user_text,
                temperature=0.15,
                max_tokens=450,
            )
            decision = ManagerDecision.model_validate(payload)
            return self._normalize_ai_decision(decision)

        except (AiClientError, ValidationError) as ex:
            logger.warning("ManagerAgent fallback triggered: %s", ex)
            return self._fallback(user_text, reason="ai_error")

    def _normalize_ai_decision(self, decision: ManagerDecision) -> ManagerDecision:
        """
        Small UX guardrails for AI output so Manager stays in its role.
        """
        # Ensure clarifying_question consistency
        if not decision.needs_clarification:
            decision.clarifying_question = None

        # Manager should not invent concrete menu items for recommendation flow.
        if decision.intent == "recommendation_request" and not decision.needs_clarification:
            if decision.language == "ru":
                decision.reply_text = "С удовольствием помогу подобрать что-то вкусное 🙂"
            else:
                decision.reply_text = "Із задоволенням допоможу підібрати щось смачне 🙂"

        # Keep off-topic redirect concise and on-topic
        if decision.intent == "off_topic":
            if decision.language == "ru":
                decision.reply_text = (
                    "Я помогаю с выбором блюд азиатской кухни 🍜 Напишите, что вам хочется — и я помогу подобрать вариант."
                )
            else:
                decision.reply_text = (
                    "Я допомагаю з вибором страв азіатської кухні 🍜 Напиши, що тобі хочеться — і я допоможу підібрати варіант."
                )
            decision.needs_clarification = False
            decision.clarifying_question = None

        return decision

    def _fallback_empty(self, lang: str) -> ManagerDecision:
        if lang == "ru":
            return ManagerDecision(
                language="ru",
                intent="clarify_needed",
                reply_text="Я помогу с выбором блюд азиатской кухни. Напишите, что вам хочется 🙂",
                needs_clarification=True,
                clarifying_question="Вам больше хочется суп, лапшу, рис, роллы или закуску?",
                confidence=0.3,
            )
        return ManagerDecision(
            language="ua",
            intent="clarify_needed",
            reply_text="Я допоможу з вибором страв азіатської кухні. Напишіть, що вам хочеться 🙂",
            needs_clarification=True,
            clarifying_question="Тобі більше хочеться суп, локшину, рис, роли чи закуску?",
            confidence=0.3,
        )

    def _detect_language_fallback(self, user_text: str) -> str:
        """
        Slightly smarter language detection for fallback mode (UA/RU),
        so we rely less on generic char heuristics.
        """
        text = (user_text or "").lower()

        ru_markers = [
            "привет",
            "здравств",
            "спасибо",
            "посовет",
            "что поесть",
            "что такое",
            "расскажи про",
            "хочу что-то",
            "не острое",
            "роллы",
            "лапша",
        ]
        ua_markers = [
            "привіт",
            "вітаю",
            "дякую",
            "порад",
            "що поїсти",
            "що таке",
            "розкажи про",
            "хочу щось",
            "не гостре",
            "роли",
            "локшина",
        ]

        ru_hits = sum(1 for marker in ru_markers if marker in text)
        ua_hits = sum(1 for marker in ua_markers if marker in text)

        if ru_hits > ua_hits:
            return "ru"
        if ua_hits > ru_hits:
            return "ua"

        return detect_language(user_text)

    def _extract_dish_query_from_text(self, user_text: str) -> str | None:
        """
        Extracts dish name from phrases like:
        - "Розкажи про том ям"
        - "Що таке пад тай?"
        - "Расскажи про рамен"
        - "Что такое мисо суп"
        """
        text = (user_text or "").strip()
        if not text:
            return None

        lowered = text.lower()

        prefixes = [
            "розкажи про ",
            "що таке ",
            "що за страва ",
            "расскажи про ",
            "что такое ",
            "что за блюдо ",
        ]

        for prefix in prefixes:
            if lowered.startswith(prefix):
                value = text[len(prefix):].strip(" ?!.,:;\"'«»()[]")
                return value or None

        return None

    def _fallback(self, user_text: str, reason: str) -> ManagerDecision:
        text = user_text.strip().lower()
        lang = self._detect_language_fallback(user_text)

        # Minimal local heuristic fallback (cheap + safe)
        if any(x in text for x in ["привіт", "вітаю", "дякую", "привет", "здравствуйте", "спасибо", "hello", "hi"]):
            if lang == "ru":
                return ManagerDecision(
                    language="ru",
                    intent="smalltalk",
                    reply_text="Привет! 👋 Я помогу подобрать блюда азиатской кухни. Напишите, что вам хочется.",
                    confidence=0.4,
                )
            return ManagerDecision(
                language="ua",
                intent="smalltalk",
                reply_text="Привіт! 👋 Я допоможу підібрати страви азіатської кухні. Напиши, що тобі хочеться.",
                confidence=0.4,
            )

        if any(
            x in text
            for x in [
                "порад",
                "посовет",
                "хочу щось",
                "хочу что-то",
                "що поїсти",
                "что поесть",
                "допоможи обрати",
                "помоги выбрать",
            ]
        ):
            if lang == "ru":
                return ManagerDecision(
                    language="ru",
                    intent="recommendation_request",
                    reply_text="С удовольствием помогу 🙂",
                    needs_clarification=True,
                    clarifying_question="Что вам больше хочется: суп, лапша, рис, роллы или закуска?",
                    confidence=0.4,
                )
            return ManagerDecision(
                language="ua",
                intent="recommendation_request",
                reply_text="Із задоволенням допоможу 🙂",
                needs_clarification=True,
                clarifying_question="Що тобі більше хочеться: суп, локшина, рис, роли чи закуска?",
                confidence=0.4,
            )

        if any(x in text for x in ["що таке", "розкажи про", "что такое", "расскажи про"]):
            dish_query = self._extract_dish_query_from_text(user_text)

            if lang == "ru":
                if dish_query:
                    return ManagerDecision(
                        language="ru",
                        intent="dish_question",
                        reply_text="Отличный вопрос 👨‍🍳 Сейчас расскажу.",
                        dish_query=dish_query,
                        needs_clarification=False,
                        confidence=0.5,
                    )
                return ManagerDecision(
                    language="ru",
                    intent="dish_question",
                    reply_text="Отличный вопрос 👨‍🍳",
                    needs_clarification=True,
                    clarifying_question="Напишите, пожалуйста, название блюда, и я расскажу о нём.",
                    confidence=0.4,
                )

            if dish_query:
                return ManagerDecision(
                    language="ua",
                    intent="dish_question",
                    reply_text="Чудове запитання 👨‍🍳 Зараз розповім.",
                    dish_query=dish_query,
                    needs_clarification=False,
                    confidence=0.5,
                )
            return ManagerDecision(
                language="ua",
                intent="dish_question",
                reply_text="Чудове запитання 👨‍🍳",
                needs_clarification=True,
                clarifying_question="Напиши, будь ласка, назву страви, і я розповім про неї.",
                confidence=0.4,
            )

        # Soft off-topic / generic fallback redirect to food topic
        if lang == "ru":
            return ManagerDecision(
                language="ru",
                intent="clarify_needed",
                reply_text="Я помогаю с выбором блюд азиатской кухни.",
                needs_clarification=True,
                clarifying_question="Напишите, что вам хочется: что-то острое, легкое, с курицей, роллы, суп и т.д.",
                confidence=0.35,
            )

        return ManagerDecision(
            language="ua",
            intent="clarify_needed",
            reply_text="Я допомагаю з вибором страв азіатської кухні.",
            needs_clarification=True,
            clarifying_question="Напиши, що тобі хочеться: щось гостре, легке, з куркою, роли, суп тощо.",
            confidence=0.35,
        )