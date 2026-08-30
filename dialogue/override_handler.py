"""Resolve and apply explicit corrections to current shopping constraints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .slot_extractor import SlotExtraction, extract_slots
from .state import SUPPORTED_SLOTS, SessionState
from .types import SlotValue


REPLACEMENT_PATTERN = re.compile(
    r"\b(?:actually|instead|rather|change(?:d)?(?:\s+that)?\s+to|"
    r"make\s+it|switch\s+to|i\s+changed\s+my\s+mind)\b",
    re.IGNORECASE,
)
GENERAL_RETRACTION_PATTERN = re.compile(
    r"\b(?:ignore\s+my\s+earlier\s+preference|forget\s+that|scratch\s+that|"
    r"never\s+mind|i\s+changed\s+my\s+mind)\b",
    re.IGNORECASE,
)
SLOT_TERMS = {
    "category": ("category", "product type", "item type"),
    "material": ("material", "fabric"),
    "color": ("color", "colour"),
    "size": ("size", "sizing"),
    "style": ("style", "fit"),
    "brand": ("brand",),
    "budget": ("budget", "budget limit", "price limit"),
    "feature": ("feature",),
    "use_case": ("use case",),
}


@dataclass(frozen=True)
class OverrideResolution:
    """Proposed surgical corrections computed without mutating session state."""

    detected: bool = False
    clear_slots: tuple[str, ...] = ()
    remove_values: dict[str, tuple[SlotValue, ...]] = field(default_factory=dict)
    replacement_values: dict[str, tuple[SlotValue, ...]] = field(default_factory=dict)


def _clear_requested(message: str, terms: tuple[str, ...]) -> bool:
    """Return whether a message explicitly retracts a named slot constraint."""

    alternatives = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
    patterns = (
        rf"\b(?:do\s+not|don't)\s+care\s+about\s+(?:the\s+)?(?:{alternatives})\s+anymore\b",
        rf"\b(?:{alternatives})\s+(?:does\s+not|doesn't)\s+matter\s+anymore\b",
        rf"\b(?:forget|ignore|remove)\s+(?:the\s+)?(?:{alternatives})\b",
        rf"\bno\s+longer\s+(?:care\s+about\s+)?(?:the\s+)?(?:{alternatives})\b",
    )
    return any(re.search(pattern, message, re.IGNORECASE) for pattern in patterns)


def _value_rejected(message: str, value: str) -> bool:
    """Return whether an existing textual value is explicitly rejected."""

    escaped = re.escape(value)
    patterns = (
        rf"\bnot\s+{escaped}\b",
        rf"\bno\s+longer\s+{escaped}\b",
        rf"\banything\s+but\s+{escaped}\b",
        rf"\bwithout\s+{escaped}\b",
    )
    return any(re.search(pattern, message, re.IGNORECASE) for pattern in patterns)


def _last_user_extraction(state: SessionState) -> SlotExtraction | None:
    """Extract facts from the most recent user turn for explicit retraction."""

    for entry in reversed(state.message_history):
        if entry.get("role") == "user":
            return extract_slots(entry.get("content", ""))
    return None


def resolve_override(
    user_message: str,
    state: SessionState,
    extraction: SlotExtraction,
) -> OverrideResolution:
    """Compute explicit replacement, clearing, and targeted-removal actions."""

    clear_slots = tuple(
        slot_name
        for slot_name in SUPPORTED_SLOTS
        if state.slots[slot_name]
        and _clear_requested(user_message, SLOT_TERMS[slot_name])
    )

    remove_values: dict[str, tuple[SlotValue, ...]] = {}
    for slot_name in SUPPORTED_SLOTS:
        rejected = tuple(
            value
            for value in state.slots[slot_name]
            if isinstance(value, str) and _value_rejected(user_message, value)
        )
        if rejected:
            remove_values[slot_name] = rejected

    if GENERAL_RETRACTION_PATTERN.search(user_message):
        previous = _last_user_extraction(state)
        if previous is not None:
            for slot_name in SUPPORTED_SLOTS:
                if slot_name == "category":
                    continue
                active_previous = tuple(
                    value for value in previous.slots.get(slot_name, ())
                    if value in state.slots[slot_name]
                    and value not in extraction.slots.get(slot_name, ())
                )
                if active_previous:
                    remove_values[slot_name] = tuple(dict.fromkeys([
                        *remove_values.get(slot_name, ()),
                        *active_previous,
                    ]))

    replacement_values: dict[str, tuple[SlotValue, ...]] = {}
    if REPLACEMENT_PATTERN.search(user_message):
        for slot_name in SUPPORTED_SLOTS:
            if slot_name in clear_slots or not state.slots[slot_name]:
                continue
            rejected = remove_values.get(slot_name, ())
            candidates = tuple(
                value for value in extraction.slots.get(slot_name, ())
                if value not in rejected
            )
            if candidates and list(candidates) != state.slots[slot_name]:
                replacement_values[slot_name] = candidates

    detected = bool(clear_slots or remove_values or replacement_values)
    return OverrideResolution(
        detected=detected,
        clear_slots=clear_slots,
        remove_values=remove_values,
        replacement_values=replacement_values,
    )


def apply_override(state: SessionState, resolution: OverrideResolution) -> None:
    """Commit a previously resolved correction to operational slot state."""

    for slot_name in resolution.clear_slots:
        state.clear_constraint(slot_name)
    for slot_name, values in resolution.remove_values.items():
        if slot_name not in resolution.clear_slots:
            state.remove_slot_values(slot_name, values)
    for slot_name, values in resolution.replacement_values.items():
        if slot_name not in resolution.clear_slots:
            state.set_constraint(slot_name, values)
    if resolution.detected:
        state.override_detected = True
