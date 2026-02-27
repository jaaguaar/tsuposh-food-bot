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
        # New recommendation flow state
        "reco_seed_text": None,
        "reco_round": 0,
        "reco_candidate_dish_ids": [],
        "reco_filters": {},
        "reco_last_clarify": {"category_options": [], "property_options": []},
        "reco_shown_category_options": [],
        "reco_shown_property_options": [],
        # Simple per-user memory (last request summary)
        "history_last_request": None,
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


# --- New recommendation flow helpers (clarification-limited) ---

async def reset_reco_flow(state: FSMContext) -> None:
    session = await _ensure_session(state)
    session["reco_seed_text"] = None
    session["reco_round"] = 0
    session["reco_candidate_dish_ids"] = []
    session["reco_filters"] = {}
    session["reco_last_clarify"] = {"category_options": [], "property_options": []}
    session["reco_shown_category_options"] = []
    session["reco_shown_property_options"] = []
    await state.update_data(**{_SESSION_KEY: session})


async def get_reco_seed_text(state: FSMContext) -> str | None:
    session = await _ensure_session(state)
    value = session.get("reco_seed_text")
    return str(value) if value else None


async def set_reco_seed_text(state: FSMContext, seed_text: str | None) -> None:
    session = await _ensure_session(state)
    session["reco_seed_text"] = seed_text or None
    await state.update_data(**{_SESSION_KEY: session})


async def get_reco_round(state: FSMContext) -> int:
    session = await _ensure_session(state)
    try:
        return int(session.get("reco_round") or 0)
    except Exception:
        return 0


async def set_reco_round(state: FSMContext, value: int) -> None:
    session = await _ensure_session(state)
    session["reco_round"] = int(value)
    await state.update_data(**{_SESSION_KEY: session})


async def get_reco_candidates(state: FSMContext) -> list[str]:
    session = await _ensure_session(state)
    return list(session.get("reco_candidate_dish_ids") or [])


async def set_reco_candidates(state: FSMContext, dish_ids: list[str]) -> None:
    session = await _ensure_session(state)
    session["reco_candidate_dish_ids"] = list(dish_ids or [])
    await state.update_data(**{_SESSION_KEY: session})


async def get_reco_filters(state: FSMContext) -> dict[str, Any]:
    session = await _ensure_session(state)
    return dict(session.get("reco_filters") or {})


async def set_reco_filters(state: FSMContext, filters: dict[str, Any]) -> None:
    session = await _ensure_session(state)
    session["reco_filters"] = dict(filters or {})
    await state.update_data(**{_SESSION_KEY: session})


async def get_reco_last_clarify(state: FSMContext) -> dict[str, Any]:
    session = await _ensure_session(state)
    value = session.get("reco_last_clarify") or {}
    if not isinstance(value, dict):
        return {"category_options": [], "property_options": []}
    value.setdefault("category_options", [])
    value.setdefault("property_options", [])
    value.setdefault("asked", None)
    return value


async def set_reco_last_clarify(state: FSMContext, category_options: list[str], property_options: list[str], asked: str | None = None) -> None:
    session = await _ensure_session(state)
    session["reco_last_clarify"] = {
        "category_options": list(category_options or []),
        "property_options": list(property_options or []),
        "asked": asked,
    }
    await state.update_data(**{_SESSION_KEY: session})


async def get_history_last_request(state: FSMContext) -> str | None:
    session = await _ensure_session(state)
    value = session.get("history_last_request")
    return str(value) if value else None


async def set_history_last_request(state: FSMContext, value: str | None) -> None:
    session = await _ensure_session(state)
    session["history_last_request"] = value or None
    await state.update_data(**{_SESSION_KEY: session})

async def get_reco_shown_category_options(state: FSMContext) -> list[str]:
    session = await _ensure_session(state)
    return list(session.get("reco_shown_category_options") or [])

async def get_reco_shown_property_options(state: FSMContext) -> list[str]:
    session = await _ensure_session(state)
    return list(session.get("reco_shown_property_options") or [])

async def add_reco_shown_options(state: FSMContext, *, categories: list[str] | None = None, properties: list[str] | None = None) -> None:
    session = await _ensure_session(state)
    cats = list(session.get("reco_shown_category_options") or [])
    props = list(session.get("reco_shown_property_options") or [])
    for c in (categories or []):
        if c and c not in cats:
            cats.append(c)
    for p in (properties or []):
        if p and p not in props:
            props.append(p)
    session["reco_shown_category_options"] = cats
    session["reco_shown_property_options"] = props
    await state.update_data(**{_SESSION_KEY: session})
