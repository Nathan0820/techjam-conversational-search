"""Deterministic hard-versus-soft classification for current slot evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .slot_extractor import SlotExtraction, extract_slots
from .state import SUPPORTED_SLOTS, SessionState
from .types import BudgetConstraint, SlotValue


HARD_CUE_PATTERNS = (
    r"\bmust\b",
    r"\bneed(?:s|ed)?(?:\s+to)?\b",
    r"\b(?:have|has)\s+to\b",
    r"\brequir(?:e|es|ed|ement|ements)\b",
    r"\bessential\b",
    r"\bdefinitely\b",
    r"\bonly\b",
    r"\b(?:cannot|can't)\b",
    r"\bno\s+more\s+than\b",
    r"\bat\s+most\b",
    r"\b(?:under|below)\b",
    r"\b(?:maximum|max)\b",
)
SOFT_CUE_PATTERNS = (
    r"\bprefer(?:red|ably)?\b",
    r"\bwould\s+(?:prefer|like)\b",
    r"\bi(?:'d|\s+would)\s+(?:prefer|like)\b",
    r"\bideally\b",
    r"\bmaybe\b",
    r"\bif\s+possible\b",
    r"\bnice\s+to\s+have\b",
    r"\bi\s+like\b",
    r"\b(?:around|approximately|approx\.?|roughly)\b",
    r"\b(?:is|are|was|were)\s+not\s+(?:strictly\s+)?necessary\b",
    r"\b(?:isn't|aren't|wasn't|weren't)\s+(?:strictly\s+)?necessary\b",
    r"\bnot\s+(?:required|essential)\b",
    r"\b(?:does\s+not|doesn't)\s+have\s+to\b",
)
NEGATED_HARD_RE = re.compile(
    r"\b(?:do\s+not|don't|does\s+not|doesn't)\s+(?:really\s+)?"
    r"(?:need|have)\b|\bnot\s+(?:only|required|essential)\b",
    re.IGNORECASE,
)
CONTRAST_BOUNDARY_RE = re.compile(
    r"\s*(?:;|\bbut\b|\bhowever\b|\balthough\b|\bwhereas\b)\s*",
    re.IGNORECASE,
)
SLOT_REFERENCE_TERMS = {
    "category": ("category", "product type", "item type"),
    "material": ("material", "fabric"),
    "color": ("color", "colour"),
    "size": ("size", "sizing"),
    "style": ("style", "fit"),
    "brand": ("brand",),
    "budget": ("budget", "price", "price limit", "budget limit"),
    "feature": ("feature",),
    "use_case": ("use case",),
}


@dataclass(frozen=True)
class ConstraintClassification:
    """Strength assignments for slot names referenced by the current turn."""

    hard_slots: frozenset[str]
    soft_slots: frozenset[str]

    def __post_init__(self) -> None:
        """Reject overlapping or unsupported slot-name classifications."""

        overlap = self.hard_slots & self.soft_slots
        if overlap:
            raise ValueError(f"slots cannot be both hard and soft: {sorted(overlap)}")
        unsupported = (self.hard_slots | self.soft_slots) - set(SUPPORTED_SLOTS)
        if unsupported:
            raise ValueError(f"unsupported classified slot name(s): {sorted(unsupported)}")


def _matches(text: str, patterns: tuple[str, ...]) -> list[re.Match[str]]:
    """Return all deterministic cue matches ordered by message position."""

    matches = [
        match
        for pattern in patterns
        for match in re.finditer(pattern, text, re.IGNORECASE)
    ]
    return sorted(matches, key=lambda match: (match.start(), match.end()))


def _cue_matches(text: str) -> list[tuple[str, int, int]]:
    """Return hard and soft cues while ignoring explicitly negated hard cues."""

    hard_text = NEGATED_HARD_RE.sub(lambda match: " " * len(match.group(0)), text)
    cues = [
        *(('hard', match.start(), match.end()) for match in _matches(
            hard_text, HARD_CUE_PATTERNS,
        )),
        *(('soft', match.start(), match.end()) for match in _matches(
            text, SOFT_CUE_PATTERNS,
        )),
    ]
    return sorted(cues, key=lambda cue: (cue[1], cue[2]))


def _split_cued_commas(text: str) -> list[str]:
    """Split commas only when the following span introduces a new cue."""

    clauses: list[str] = []
    start = 0
    for match in re.finditer(r",", text):
        if not _cue_matches(text[match.end():]):
            continue
        clause = text[start:match.start()].strip()
        if clause:
            clauses.append(clause)
        start = match.end()
    final = text[start:].strip()
    if final:
        clauses.append(final)
    return clauses


def _clauses(user_message: str) -> list[str]:
    """Split a turn at contrast boundaries without breaking simple conjunctions."""

    clauses: list[str] = []
    for fragment in CONTRAST_BOUNDARY_RE.split(user_message):
        if fragment.strip():
            clauses.extend(_split_cued_commas(fragment))
    return clauses or [user_message]


def _value_terms(value: SlotValue) -> tuple[str, ...]:
    """Return textual forms useful for locating an extracted value in a clause."""

    if isinstance(value, BudgetConstraint):
        amounts = []
        for amount in (value.minimum, value.maximum):
            if amount is not None:
                amounts.append(str(amount).removesuffix(".0"))
        return tuple(dict.fromkeys(amounts))
    terms = [value]
    if value.casefold() == "gray":
        terms.append("grey")
    if value.casefold().endswith("s") and not value.casefold().endswith("ss"):
        terms.append(value[:-1])
    return tuple(terms)


def _slot_positions(clause: str, values: list[SlotValue]) -> list[int]:
    """Locate normalized slot values approximately in their source clause."""

    positions: list[int] = []
    for value in values:
        for term in _value_terms(value):
            match = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", clause, re.IGNORECASE)
            if match:
                positions.append((match.start() + match.end()) // 2)
                break
    return positions


def _referenced_existing_slots(text: str, state: SessionState) -> set[str]:
    """Find non-empty slots explicitly named without repeating their values."""

    referenced: set[str] = set()
    for slot_name, terms in SLOT_REFERENCE_TERMS.items():
        if not state.slots[slot_name]:
            continue
        if any(
            re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE)
            for term in terms
        ):
            referenced.add(slot_name)
    return referenced


def _nearest_cue_strength(
    position: int,
    cues: list[tuple[str, int, int]],
) -> str:
    """Choose the nearest cue, preferring a preceding cue on equal distance."""

    return min(
        cues,
        key=lambda cue: (
            min(abs(position - cue[1]), abs(position - cue[2])),
            0 if cue[2] <= position else 1,
            -cue[1],
        ),
    )[0]


def _explicit_clause_strengths(
    clause: str,
    clause_extraction: SlotExtraction,
    state: SessionState,
) -> dict[str, str]:
    """Classify extracted clause slots from local hard and soft cues."""

    cues = _cue_matches(clause)
    if not cues:
        return {}
    cue_strengths = {cue[0] for cue in cues}
    populated = {
        slot_name: values
        for slot_name, values in clause_extraction.slots.items()
        if values
    }
    for slot_name in _referenced_existing_slots(clause, state):
        populated.setdefault(slot_name, state.slots[slot_name])
    if len(cue_strengths) == 1:
        strength = next(iter(cue_strengths))
        return {slot_name: strength for slot_name in populated}

    strengths: dict[str, str] = {}
    for slot_name, values in populated.items():
        if slot_name == "budget" and any(cue[0] == "soft" for cue in cues):
            strengths[slot_name] = "soft"
            continue
        positions = _slot_positions(clause, values)
        if positions:
            strengths[slot_name] = _nearest_cue_strength(positions[-1], cues)
        else:
            strengths[slot_name] = cues[-1][0]
    return strengths


def _default_strength(
    slot_name: str,
    extraction: SlotExtraction,
    state: SessionState,
) -> str:
    """Apply multi-turn preservation and conservative neutral defaults."""

    if slot_name in state.hard_constraints:
        return "hard"
    if slot_name in state.soft_preferences:
        return "soft"
    if slot_name == "category":
        return "hard"
    if slot_name == "budget":
        budgets = extraction.slots[slot_name]
        if budgets and all(
            isinstance(value, BudgetConstraint) and value.approximate
            for value in budgets
        ):
            return "soft"
        return "hard"
    return "soft"


def classify_constraints(
    user_message: str,
    extraction: SlotExtraction,
    state: SessionState,
) -> ConstraintClassification:
    """Classify current extracted slot names without mutating session state."""

    current_slots = {
        slot_name for slot_name in SUPPORTED_SLOTS
        if extraction.slots.get(slot_name)
    }
    current_slots.update(_referenced_existing_slots(user_message, state))
    strengths: dict[str, str] = {}
    for clause in _clauses(user_message):
        clause_extraction = extract_slots(clause)
        for slot_name, strength in _explicit_clause_strengths(
            clause, clause_extraction, state,
        ).items():
            if slot_name in current_slots:
                strengths[slot_name] = strength

    for slot_name in current_slots:
        strengths.setdefault(slot_name, _default_strength(slot_name, extraction, state))

    return ConstraintClassification(
        hard_slots=frozenset(
            slot_name for slot_name, strength in strengths.items()
            if strength == "hard"
        ),
        soft_slots=frozenset(
            slot_name for slot_name, strength in strengths.items()
            if strength == "soft"
        ),
    )


def apply_constraint_classification(
    state: SessionState,
    classification: ConstraintClassification,
) -> None:
    """Commit current-turn slot strengths after the response succeeds."""

    for slot_name in classification.hard_slots:
        state.set_constraint(
            slot_name,
            state.slots[slot_name],
            strength="hard" if state.slots[slot_name] else None,
        )
    for slot_name in classification.soft_slots:
        state.set_constraint(
            slot_name,
            state.slots[slot_name],
            strength="soft" if state.slots[slot_name] else None,
        )
