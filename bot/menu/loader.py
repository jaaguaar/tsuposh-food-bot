from __future__ import annotations

from pathlib import Path

import yaml

from bot.menu.models import MenuCatalog


DEFAULT_MENU_PATH = Path(__file__).with_name("menu.yaml")


def load_menu(menu_path: str | Path | None = None) -> MenuCatalog:
    path = Path(menu_path) if menu_path else DEFAULT_MENU_PATH

    if not path.exists():
        raise FileNotFoundError(f"Menu file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Menu file is empty: {path}")

    return MenuCatalog.model_validate(raw)