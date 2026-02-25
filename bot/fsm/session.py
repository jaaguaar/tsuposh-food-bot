from __future__ import annotations

import time
from typing import Any

from aiogram.fsm.context import FSMContext

_SESSION_KEY = "session"


def _default_session() -> dict[str, Any]:
    return {
        "language": "ua",
        "current_intent": None,
        "preferences": {},
        "last_recommendations": [],
        "last_clarify_category_options": [],
        "last_clarify_category_options_set_at": 0,
    }


async def _ensure_session(state: FSMContext) -> dict[str, Any]:
    data = await state.get_data()
    session = data.get(_SESSION_KEY)
    if not isinstance(session, dict):
        session = _default_session()
        await state.update_data(**{_SESSION_KEY: session})
        return session

    # Backward-compatible keys
    changed = False
    for k, v in _default_session().items():
        if k not in session:
            session[k] = v
            changed = True
    if changed:
        await state.update_data(**{_SESSION_KEY: session})
    return session


async def get_session_data(state: FSMContext) -> dict[str, Any]:
    return await _ensure_session(state)


async def set_user_language(state: FSMContext, language: str) -> None:
    session = await _ensure_session(state)
    session["language"] = language
    await state.update_data(**{_SESSION_KEY: session})


async def get_user_language(state: FSMContext) -> str:
    session = await _ensure_session(state)
    return str(session.get("language") or "ua")


async def set_current_intent(state: FSMContext, intent: str | None) -> None:
    session = await _ensure_session(state)
    session["current_intent"] = intent
    await state.update_data(**{_SESSION_KEY: session})


async def set_preferences(state: FSMContext, preferences: dict[str, Any]) -> None:
    session = await _ensure_session(state)
    session["preferences"] = preferences or {}
    await state.update_data(**{_SESSION_KEY: session})


async def get_preferences(state: FSMContext) -> dict[str, Any]:
    session = await _ensure_session(state)
    return dict(session.get("preferences") or {})


async def set_last_recommendations(state: FSMContext, dish_ids: list[str]) -> None:
    session = await _ensure_session(state)
    session["last_recommendations"] = list(dish_ids or [])
    await state.update_data(**{_SESSION_KEY: session})


async def get_last_recommendations(state: FSMContext) -> list[str]:
    session = await _ensure_session(state)
    return list(session.get("last_recommendations") or [])


async def set_last_clarify_category_options(state: FSMContext, options: list[str]) -> None:
    session = await _ensure_session(state)
    session["last_clarify_category_options"] = list(options or [])
    session["last_clarify_category_options_set_at"] = int(time.time()) if options else 0
    await state.update_data(**{_SESSION_KEY: session})


async def get_last_clarify_category_options(state: FSMContext) -> list[str]:
    session = await _ensure_session(state)
    ttl_seconds = 2 * 60 * 60  # 2 hours
    options = list(session.get("last_clarify_category_options") or [])
    set_at = int(session.get("last_clarify_category_options_set_at") or 0)
    if options and set_at and (int(time.time()) - set_at) > ttl_seconds:
        session["last_clarify_category_options"] = []
        session["last_clarify_category_options_set_at"] = 0
        await state.update_data(**{_SESSION_KEY: session})
        return []
    return options
