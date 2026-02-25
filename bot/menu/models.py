from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


Category = Literal[
    "soups",
    "noodles",
    "rice",
    "wok",
    "sushi_rolls",
    "snacks",
    "desserts",
    "drinks",
]


class DishName(BaseModel):
    ua: str = Field(min_length=1)
    ru: str = Field(min_length=1)


class DishText(BaseModel):
    ua: str = Field(min_length=1)
    ru: str = Field(min_length=1)


class Dish(BaseModel):
    id: str = Field(min_length=1)
    category: Category

    name: DishName
    description_short: DishText
    description_full: DishText

    ingredients: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    spice_level: int = Field(ge=0, le=3, default=0)
    is_vegetarian: bool = False
    allergens: list[str] = Field(default_factory=list)
    pair_with: DishText | None = None

    available: bool = True

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        # Keep IDs simple and stable for internal references
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
        if any(ch not in allowed for ch in value.lower()):
            raise ValueError("Dish id may contain only a-z, 0-9, '-' and '_'")
        return value.lower()

    @field_validator("ingredients", "tags", "allergens")
    @classmethod
    def normalize_list_values(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for item in values:
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)
        return result


class MenuCatalog(BaseModel):
    version: str = "1"
    title: str = "TsuPosh Demo Menu"
    dishes: list[Dish]

    @field_validator("dishes")
    @classmethod
    def validate_unique_ids(cls, dishes: list[Dish]) -> list[Dish]:
        ids = [d.id for d in dishes]
        duplicates = sorted({x for x in ids if ids.count(x) > 1})
        if duplicates:
            raise ValueError(f"Duplicate dish ids found: {', '.join(duplicates)}")
        return dishes