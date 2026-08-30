"""Deterministic current-session shopping intent classification."""

from __future__ import annotations

import re

from .slot_extractor import SlotExtraction, extract_slots
from .state import SessionState
from .types import ShoppingIntent


NEGATED_BUYING_PATTERNS = (
    r"\bnot\s+(?:looking\s+to\s+buy|buying|ready\s+to\s+buy)\b",
    r"\b(?:do\s+not|don't)\s+(?:need|want)\b",
)
BUYING_PATTERNS = (
    r"\bi\s+need\b",
    r"\bi\s+want(?:\s+to\s+buy)?\b",
    r"\bi(?:'m|\s+am)\s+buying\b",
    r"\blooking\s+to\s+buy\b",
    r"\bneed\s+to\s+get\b",
    r"\bfind\s+me\b",
    r"\bhelp\s+me\s+find\b",
    r"\brecommend\s+me\b",
    r"\bwhich\s+.+?\s+should\s+i\s+get\b",
    r"\bwhat\s+should\s+i\s+buy\b",
)
BROWSING_PATTERNS = (
    r"\bjust\s+browsing\b",
    r"\bjust\s+looking\b",
    r"\blooking\s+around\b",
    r"\bnot\s+(?:looking\s+to\s+buy|buying|ready\s+to\s+buy)\b",
    r"\b(?:do\s+not|don't)\s+(?:need|want)\s+to\s+buy\b",
    r"\b(?:do\s+not|don't)\s+need\s+anything\s+specific\b",
    r"\bnot\s+sure\b",
    r"\bi(?:'m|\s+am)\s+exploring\b",
    r"\bi(?:'m|\s+am)\s+(?:interested\s+in|considering)\b",
    r"\bwhat\s+(?:kinds?|options|styles)\b",
    r"\btell\s+me\s+about\b",
    r"\bshow\s+me\s+some\s+ideas\b",
    r"\bany\s+ideas\b",
)


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether text matches at least one case-insensitive pattern."""

    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _without_negated_buying(text: str) -> str:
    """Remove negated buying spans before scoring positive buying evidence."""

    for pattern in NEGATED_BUYING_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def _historical_signal(state: SessionState) -> ShoppingIntent | None:
    """Return the most recent explicit signal from user history only."""

    for entry in reversed(state.message_history):
        if entry.get("role") != "user":
            continue
        content = entry.get("content", "")
        if _contains_any(content, BROWSING_PATTERNS):
            return "browsing"
        if _contains_any(_without_negated_buying(content), BUYING_PATTERNS):
            return "buying"
    return None


def detect_intent(
    user_message: str,
    state: SessionState,
    extraction: SlotExtraction | None = None,
) -> ShoppingIntent:
    """Classify current shopping intent without mutating session state.

    Explicit current-language signals dominate prior intent. With no decisive
    current signal, the previous intent persists. A first-turn ambiguity falls
    back to browsing, except for a highly structured active-search message.
    """

    text = user_message.strip()
    buying_score = 0
    browsing_score = 0

    if _contains_any(text, BROWSING_PATTERNS):
        browsing_score += 6
    if _contains_any(_without_negated_buying(text), BUYING_PATTERNS):
        buying_score += 6

    if state.intent == "buying":
        buying_score += 2
    elif state.intent == "browsing":
        browsing_score += 2
    else:
        history_signal = _historical_signal(state)
        if history_signal == "buying":
            buying_score += 1
        elif history_signal == "browsing":
            browsing_score += 1

    current_extraction = extraction if extraction is not None else extract_slots(user_message)
    populated_slots = {name for name, values in current_extraction.slots.items() if values}
    selection_anchors = {"category", "budget", "size"} & populated_slots
    if len(populated_slots) >= 4 and selection_anchors:
        buying_score += 4

    if buying_score > browsing_score:
        return "buying"
    if browsing_score > buying_score:
        return "browsing"
    return state.intent or _historical_signal(state) or "browsing"
