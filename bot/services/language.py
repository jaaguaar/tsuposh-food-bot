import re


_UA_CHARS_PATTERN = re.compile(r"[іїєґІЇЄҐ]")
_RU_CHARS_PATTERN = re.compile(r"[ёыэЁЫЭ]")


def detect_language(text: str) -> str:
    """
    Detects user language between Ukrainian (ua) and Russian (ru)
    using a simple character-based heuristic.

    Defaults to 'ua' if uncertain.
    """
    if not text:
        return "ua"

    if _UA_CHARS_PATTERN.search(text):
        return "ua"

    if _RU_CHARS_PATTERN.search(text):
        return "ru"

    # Fallback heuristic:
    # If no specific chars found, default to UA for MVP
    return "ua"


def normalize_language(value: str | None) -> str:
    if value in {"ua", "ru"}:
        return value
    return "ua"