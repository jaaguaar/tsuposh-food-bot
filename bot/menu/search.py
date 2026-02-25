from __future__ import annotations

from collections import defaultdict

from bot.menu.models import Dish, MenuCatalog


def get_available_dishes(catalog: MenuCatalog) -> list[Dish]:
    return [d for d in catalog.dishes if d.available]


def find_dish_by_id(catalog: MenuCatalog, dish_id: str) -> Dish | None:
    normalized = dish_id.strip().lower()
    for dish in catalog.dishes:
        if dish.id == normalized:
            return dish
    return None


def find_dishes_by_name(catalog: MenuCatalog, query: str, language: str = "ua") -> list[Dish]:
    q = query.strip().lower()
    if not q:
        return []

    result: list[Dish] = []
    for dish in get_available_dishes(catalog):
        name = dish.name.ru if language == "ru" else dish.name.ua
        if q in name.lower():
            result.append(dish)
    return result


def search_by_category(catalog: MenuCatalog, category: str) -> list[Dish]:
    category = category.strip().lower()
    return [d for d in get_available_dishes(catalog) if d.category == category]


def search_by_tags(catalog: MenuCatalog, tags: list[str]) -> list[Dish]:
    wanted = {t.strip().lower() for t in tags if t.strip()}
    if not wanted:
        return []

    result: list[Dish] = []
    for dish in get_available_dishes(catalog):
        dish_tags = {t.lower() for t in dish.tags}
        if wanted.intersection(dish_tags):
            result.append(dish)
    return result


def recommend_simple(
    catalog: MenuCatalog,
    *,
    category: str | None = None,
    max_spice_level: int | None = None,
    vegetarian_only: bool = False,
    include_tags: list[str] | None = None,
    exclude_ingredients: list[str] | None = None,
    limit: int = 3,
) -> list[Dish]:
    candidates = get_available_dishes(catalog)

    if category:
        candidates = [d for d in candidates if d.category == category]

    if max_spice_level is not None:
        candidates = [d for d in candidates if d.spice_level <= max_spice_level]

    if vegetarian_only:
        candidates = [d for d in candidates if d.is_vegetarian]

    if include_tags:
        wanted_tags = {t.strip().lower() for t in include_tags if t.strip()}
        if wanted_tags:
            candidates = [
                d for d in candidates
                if wanted_tags.intersection({t.lower() for t in d.tags})
            ]

    if exclude_ingredients:
        excluded = {x.strip().lower() for x in exclude_ingredients if x.strip()}
        if excluded:
            candidates = [
                d for d in candidates
                if not excluded.intersection({i.lower() for i in d.ingredients})
            ]

    # Naive scoring to keep results varied and meaningful
    scored: list[tuple[int, Dish]] = []
    for dish in candidates:
        score = 0

        if category and dish.category == category:
            score += 3

        if include_tags:
            dish_tags = {t.lower() for t in dish.tags}
            score += len(dish_tags.intersection({t.lower() for t in include_tags}))

        if vegetarian_only and dish.is_vegetarian:
            score += 1

        # Slight preference to medium spice for "interesting" picks in demo
        if dish.spice_level == 1:
            score += 1

        scored.append((score, dish))

    scored.sort(key=lambda item: (-item[0], item[1].name.ua))
    return [dish for _, dish in scored[:limit]]


def group_by_category(catalog: MenuCatalog) -> dict[str, list[Dish]]:
    grouped: dict[str, list[Dish]] = defaultdict(list)
    for dish in get_available_dishes(catalog):
        grouped[dish.category].append(dish)
    return dict(grouped)