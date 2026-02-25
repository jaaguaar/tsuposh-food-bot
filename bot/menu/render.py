from __future__ import annotations

from bot.menu.models import Dish

_CATEGORY_LABELS = {
    "ua": {"soups": "Суп", "noodles": "Локшина", "rice": "Рис", "wok": "Вок", "sushi_rolls": "Роли", "snacks": "Закуска", "desserts": "Десерт", "drinks": "Напій"},
    "ru": {"soups": "Суп", "noodles": "Лапша", "rice": "Рис", "wok": "Вок", "sushi_rolls": "Роллы", "snacks": "Закуска", "desserts": "Десерт", "drinks": "Напиток"},
}
_ALLERGEN_LABELS = {
    "ua": {"soy": "соя", "gluten": "глютен", "egg": "яйце", "fish": "риба", "sesame": "кунжут", "milk": "молоко", "shellfish": "морепродукти", "peanuts": "арахіс"},
    "ru": {"soy": "соя", "gluten": "глютен", "egg": "яйцо", "fish": "рыба", "sesame": "кунжут", "milk": "молоко", "shellfish": "морепродукты", "peanuts": "арахис"},
}

def _spice_badge(spice_level: int) -> str:
    return f"🌶️ {max(0, min(3, spice_level))}/3"

def _veg_badge(is_vegetarian: bool, lang: str) -> str:
    return (" • 🥬 Вегетаріанська" if lang == "ua" else " • 🥬 Вегетарианское") if is_vegetarian else ""

def _localize_allergens(allergens: list[str], lang: str) -> list[str]:
    mapping = _ALLERGEN_LABELS["ru" if lang == "ru" else "ua"]
    return [mapping.get(a.strip().lower(), a) for a in allergens]

def render_dish_card(dish: Dish, lang: str = "ua") -> str:
    lang = "ru" if lang == "ru" else "ua"
    name = dish.name.ru if lang == "ru" else dish.name.ua
    desc_short = dish.description_short.ru if lang == "ru" else dish.description_short.ua
    category = _CATEGORY_LABELS[lang].get(dish.category, dish.category)
    pair_with = dish.pair_with.ru if (lang == "ru" and dish.pair_with) else (dish.pair_with.ua if dish.pair_with else None)
    lines = [f"🍽 <b>{name}</b>", f"{category} • {_spice_badge(dish.spice_level)}{_veg_badge(dish.is_vegetarian, lang)}", desc_short]
    if pair_with:
        label = "Добре смакує з" if lang == "ua" else "Хорошо сочетается с"
        lines.append(f"🥢 <b>{label}:</b> {pair_with}")
    if dish.allergens:
        label = "Алергени" if lang == "ua" else "Аллергены"
        lines.append(f"⚠️ <b>{label}:</b> {', '.join(_localize_allergens(dish.allergens, lang))}")
    return "\n".join(lines)
