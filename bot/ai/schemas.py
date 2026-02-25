from __future__ import annotations

from typing import Literal, Any

from pydantic import BaseModel, Field


Intent = Literal[
    "smalltalk",
    "recommendation_request",
    "dish_question",
    "menu_availability",
    "off_topic",
    "clarify_needed",
]


class ManagerExtractedPreferences(BaseModel):
    category: str | None = None
    protein: str | None = None
    spice_level: str | None = None  # low | medium | high (textual for now)
    vegetarian: bool | None = None
    avoid_ingredients: list[str] = Field(default_factory=list)


class ManagerDecision(BaseModel):
    language: Literal["ua", "ru"] = "ua"
    intent: Intent
    reply_text: str = Field(min_length=1)

    needs_clarification: bool = False
    clarifying_question: str | None = None

    dish_query: str | None = None
    extracted_preferences: ManagerExtractedPreferences = Field(
        default_factory=ManagerExtractedPreferences
    )

    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LlmJsonResult(BaseModel):
    raw_text: str
    parsed: dict[str, Any]

class ExpertRecommendationItem(BaseModel):
    dish_id: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=220)


class ExpertDecision(BaseModel):
    language: Literal["ua", "ru"] = "ua"
    mode: Literal["recommendation", "analogs", "not_found"] = "recommendation"

    intro_text: str = Field(min_length=1)
    items: list[ExpertRecommendationItem] = Field(default_factory=list)

    # If user asked about a specific dish
    requested_dish_query: str | None = None

    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

class ChefDecision(BaseModel):
    language: Literal["ua", "ru"] = "ua"
    found_in_menu: bool = False

    intro_text: str = Field(min_length=1)
    dish_id: str | None = None
    dish_query: str | None = None

    details_text: str | None = None
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)