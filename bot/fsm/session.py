from typing import Any

from aiogram.fsm.context import FSMContext

from bot.services.language import normalize_language


async def get_session_data(state: FSMContext) -> dict[str, Any]:
    data = await state.get_data()
    return data if isinstance(data, dict) else {}


async def get_user_language(state: FSMContext) -> str:
    data = await get_session_data(state)
    return normalize_language(data.get("language"))


async def set_user_language(state: FSMContext, language: str) -> None:
    await state.update_data(language=normalize_language(language))


async def update_conversation_summary(state: FSMContext, summary: str) -> None:
    await state.update_data(conversation_summary=summary)


async def set_current_intent(state: FSMContext, intent: str) -> None:
    await state.update_data(current_intent=intent)


async def set_preferences(state: FSMContext, preferences: dict[str, Any]) -> None:
    await state.update_data(preferences=preferences)


async def set_last_recommendations(state: FSMContext, dish_ids: list[str]) -> None:
    await state.update_data(last_recommendations=dish_ids)


async def set_last_selected_dish(state: FSMContext, dish_id: str) -> None:
    await state.update_data(last_selected_dish_id=dish_id)