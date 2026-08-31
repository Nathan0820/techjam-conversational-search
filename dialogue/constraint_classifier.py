"""Classify accumulated shopping slots as hard constraints or preferences."""

from __future__ import annotations

import re

from .slot_extractor import SlotExtraction
from .state import SessionState
from .types import BudgetConstraint


HARD_LANGUAGE_RE = re.compile(
    r"\b(?:must|need|require(?:ment|d)?|cannot|can't|only|what matters is|"
    r"what i need is|under|below|less than|up to|maximum|max|between|from .+ to)\b",
    re.IGNORECASE,
)
SOFT_LANGUAGE_RE = re.compile(
    r"\b(?:prefer|preference|ideally|if possible|nice to have|around|about|"
    r"approximately|approx\.?|exploring|use your judgment)\b",
    re.IGNORECASE,
)


def _budget_is_firm(values: list[object]) -> bool:
    """Return whether any extracted budget represents a firm bound or range."""

    return any(
        isinstance(value, BudgetConstraint)
        and not value.approximate
        and (value.minimum is not None or value.maximum is not None)
        for value in values
    )


def classify_constraints(
    state: SessionState,
    user_message: str,
    extraction: SlotExtraction,
) -> None:
    """Assign strength labels to current-turn slots after accumulation/override.

    Explicit language controls all values revealed in the current message. When
    wording is neutral, category, size, and firm budget bounds are operational
    constraints; other attributes remain preferences. Existing labels on slots
    untouched by the current message are preserved.
    """

    current_slots = {
        name for name, values in extraction.slots.items()
        if values and state.slots.get(name)
    }
    if not current_slots:
        return

    explicitly_soft = bool(SOFT_LANGUAGE_RE.search(user_message))
    explicitly_hard = bool(HARD_LANGUAGE_RE.search(user_message)) and not explicitly_soft

    for name in current_slots:
        if explicitly_soft:
            strength = "soft"
        elif explicitly_hard or name in {"category", "size"}:
            strength = "hard"
        elif name == "budget" and _budget_is_firm(state.slots[name]):
            strength = "hard"
        else:
            strength = "soft"
        state.hard_constraints.discard(name)
        state.soft_preferences.discard(name)
        if strength == "hard":
            state.hard_constraints.add(name)
        else:
            state.soft_preferences.add(name)
