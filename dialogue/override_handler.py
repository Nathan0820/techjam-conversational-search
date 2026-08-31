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
STRONG_REPLACEMENT_PATTERN = re.compile(
    r"\b(?:instead|rather|change(?:d)?(?:\s+that)?\s+to|make\s+it|"
    r"switch\s+to|i\s+changed\s+my\s+mind)\b",
    re.IGNORECASE,
)
ADDITIVE_PATTERN = re.compile(
    r"\b(?:also|too|as\s+well|another)\b",
    re.IGNORECASE,
)
# The evaluator always retracts with one fixed sentence ("Actually, ignore my earlier
# preference. What I need is: X.", local_evaluator.py:85), but a customer phrases this
# many ways. Matching only the simulator's wording means retraction silently stops
# working on any paraphrase — a failure invisible in our public-set metrics, since the
# simulator never paraphrases.
GENERAL_RETRACTION_PATTERN = re.compile(
    r"\b(?:"
    r"(?:ignore|disregard|forget)\s+(?:that|this|it|the\s+above|what\s+i\s+said|"
    r"my\s+(?:earlier|previous|last)\s+(?:preference|request|message))"
    r"|scratch\s+that"
    r"|never\s+mind"
    r"|i\s+changed\s+my\s+mind"
    r"|on\s+second\s+thought"
    # "my mistake" was considered and rejected: it fires on ordinary product wording
    # such as "mistake-proof", and a false positive here erases constraints the
    # customer still holds, which is worse than missing a rare phrasing.
    r")\b",
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
    remove_active_revealed_text: tuple[str, ...] = ()

    @property
    def has_state_mutations(self) -> bool:
        """Return whether this resolution changes operational slot/raw state."""

        return bool(
            self.clear_slots
            or self.remove_values
            or self.replacement_values
            or self.remove_active_revealed_text
        )


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


def _prior_user_extractions(
    state: SessionState,
) -> tuple[tuple[str, SlotExtraction], ...]:
    """Return every prior user turn and extraction, newest to oldest."""

    return tuple(
        (entry.get("content", ""), extract_slots(entry.get("content", "")))
        for entry in reversed(state.message_history)
        if entry.get("role") == "user"
    )


def _is_category_only_phrase(state: SessionState, phrase: str) -> bool:
    """Return whether an active phrase represents only the preserved category."""

    extracted = extract_slots(phrase)
    active_non_category = any(
        value in state.slots[slot_name]
        for slot_name in SUPPORTED_SLOTS
        if slot_name != "category"
        for value in extracted.slots.get(slot_name, ())
    )
    normalized_phrase = re.sub(r"[^a-z0-9]+", " ", phrase.casefold()).strip()
    equals_active_category = any(
        isinstance(value, str)
        and normalized_phrase
        == re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
        for value in state.slots["category"]
    )
    return equals_active_category and not active_non_category


def _find_prior_active_preferences(state: SessionState) -> tuple[str, ...]:
    """Find the earlier still-active preference targeted by general retraction.

    Initial shopping turns explicitly separate category from the user's earlier
    preference.  Prefer that trailing phrase after searching all prior turns;
    otherwise fall back to the newest active non-category phrase.
    """

    fallback: tuple[str, ...] = ()
    initial_preference: tuple[str, ...] = ()
    for content, _ in _prior_user_extractions(state):
        folded_content = content.casefold()
        candidates = tuple(
            phrase
            for phrase in state.active_revealed_text
            if phrase.casefold() in folded_content
            and not _is_category_only_phrase(state, phrase)
        )
        if not candidates:
            continue
        if not fallback:
            fallback = candidates
        initial_match = re.match(
            r"^\s*i(?:'m|\s+am)\s+looking\s+for\s+.+?[.!?]\s*(?P<phrase>.+?)\s*$",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if initial_match:
            trailing = initial_match.group("phrase").casefold()
            associated = tuple(
                phrase for phrase in candidates
                if phrase.casefold() in trailing or trailing in phrase.casefold()
            )
            if associated:
                initial_preference = associated
    return initial_preference or fallback


def _active_values_in_phrases(
    state: SessionState,
    phrases: tuple[str, ...],
) -> dict[str, tuple[SlotValue, ...]]:
    """Map active normalized values represented by raw phrases to their slots."""

    values_by_slot: dict[str, list[SlotValue]] = {}
    for phrase in phrases:
        extracted = extract_slots(phrase)
        for slot_name in SUPPORTED_SLOTS:
            if slot_name == "category":
                continue
            active_values = [
                value for value in extracted.slots.get(slot_name, ())
                if value in state.slots[slot_name]
            ]
            if active_values:
                values_by_slot.setdefault(slot_name, []).extend(active_values)
    return {
        slot_name: tuple(dict.fromkeys(values))
        for slot_name, values in values_by_slot.items()
    }


def _phrases_for_values(
    phrases: tuple[str, ...],
    slot_name: str,
    values: tuple[SlotValue, ...],
) -> tuple[str, ...]:
    """Find raw phrases associated with normalized values in one slot."""

    associated: list[str] = []
    for phrase in phrases:
        extracted_values = extract_slots(phrase).slots.get(slot_name, ())
        if any(value in extracted_values for value in values):
            associated.append(phrase)
            continue
        if any(
            isinstance(value, str)
            and re.search(
                rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])",
                phrase,
                re.IGNORECASE,
            )
            for value in values
        ):
            associated.append(phrase)
    return tuple(associated)


def _active_phrases_for_values(
    state: SessionState,
    slot_name: str,
    values: tuple[SlotValue, ...],
) -> tuple[str, ...]:
    """Find active raw phrases associated with normalized values in one slot."""

    return _phrases_for_values(
        tuple(state.active_revealed_text), slot_name, values,
    )


def _phrases_for_slot_terms(
    phrases: tuple[str, ...],
    terms: tuple[str, ...],
) -> tuple[str, ...]:
    """Find current raw phrases that explicitly name a retracted slot."""

    return tuple(
        phrase
        for phrase in phrases
        if any(
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
                phrase,
                re.IGNORECASE,
            )
            for term in terms
        )
    )


def resolve_override(
    user_message: str,
    state: SessionState,
    extraction: SlotExtraction,
) -> OverrideResolution:
    """Compute explicit replacement, clearing, and targeted-removal actions."""

    requested_clear_slots = tuple(
        slot_name
        for slot_name in SUPPORTED_SLOTS
        if _clear_requested(user_message, SLOT_TERMS[slot_name])
    )
    clear_slots = tuple(
        slot_name for slot_name in requested_clear_slots
        if state.slots[slot_name]
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

    general_retraction_phrases: tuple[str, ...] = ()
    if GENERAL_RETRACTION_PATTERN.search(user_message):
        general_retraction_phrases = _find_prior_active_preferences(state)
        for slot_name, active_previous in _active_values_in_phrases(
            state, general_retraction_phrases,
        ).items():
            retained = tuple(
                value for value in active_previous
                if value not in extraction.slots.get(slot_name, ())
            )
            if retained:
                remove_values[slot_name] = tuple(dict.fromkeys([
                    *remove_values.get(slot_name, ()),
                    *retained,
                ]))

    replacement_values: dict[str, tuple[SlotValue, ...]] = {}
    replacement_signal = bool(REPLACEMENT_PATTERN.search(user_message))
    additive_signal = bool(ADDITIVE_PATTERN.search(user_message))
    strong_replacement_signal = bool(STRONG_REPLACEMENT_PATTERN.search(user_message))
    replacement_requested = replacement_signal and not (
        additive_signal and not strong_replacement_signal
    )
    if replacement_requested:
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

    phrases_to_remove: list[str] = list(general_retraction_phrases)
    for slot_name in clear_slots:
        phrases_to_remove.extend(_active_phrases_for_values(
            state, slot_name, tuple(state.slots[slot_name]),
        ))
    for slot_name in requested_clear_slots:
        phrases_to_remove.extend(_phrases_for_slot_terms(
            tuple(extraction.revealed_text), SLOT_TERMS[slot_name],
        ))
    for slot_name, values in remove_values.items():
        phrases_to_remove.extend(_active_phrases_for_values(state, slot_name, values))
        phrases_to_remove.extend(_phrases_for_values(
            tuple(extraction.revealed_text), slot_name, values,
        ))
    for slot_name, replacements in replacement_values.items():
        stale_values = tuple(
            value for value in state.slots[slot_name]
            if value not in replacements
        )
        phrases_to_remove.extend(_active_phrases_for_values(
            state, slot_name, stale_values,
        ))

    remove_active_revealed_text = tuple(dict.fromkeys(phrases_to_remove))
    has_extracted_values = any(extraction.slots.get(name) for name in SUPPORTED_SLOTS)
    explicitly_expressed = bool(
        GENERAL_RETRACTION_PATTERN.search(user_message)
        or requested_clear_slots
        or remove_values
        or (replacement_requested and has_extracted_values)
    )
    return OverrideResolution(
        detected=explicitly_expressed,
        clear_slots=clear_slots,
        remove_values=remove_values,
        replacement_values=replacement_values,
        remove_active_revealed_text=remove_active_revealed_text,
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
            strength = (
                "hard" if slot_name in state.hard_constraints
                else "soft" if slot_name in state.soft_preferences
                else None
            )
            state.set_constraint(slot_name, values, strength=strength)
    if resolution.remove_active_revealed_text:
        stale_phrases = set(resolution.remove_active_revealed_text)
        state.active_revealed_text = [
            phrase for phrase in state.active_revealed_text
            if phrase not in stale_phrases
        ]
    if resolution.detected:
        state.override_detected = True
