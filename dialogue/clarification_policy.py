"""Deterministic, state-aware clarification decisions for shopping dialogue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .override_handler import OverrideResolution
from .slot_extractor import SlotExtraction
from .state import SUPPORTED_SLOTS, SessionState
from .types import ShoppingIntent, SlotValue


ALLOWED_ASK_ATTRIBUTES = frozenset((*SUPPORTED_SLOTS, "other"))
INFORMATION_WEIGHTS = {
    "material": 3,
    "size": 3,
    "budget": 3,
    "feature": 3,
    "use_case": 3,
    "color": 1,
    "style": 1,
    "brand": 1,
}
DEFAULT_PRIORITY = (
    "feature", "material", "other", "size", "use_case", "budget", "style", "color", "brand",
)
CLOTHING_PRIORITY = (
    "material", "feature", "other", "size", "style", "use_case", "color", "budget", "brand",
)
SHOES_PRIORITY = (
    "feature", "material", "other", "size", "use_case", "style", "color", "budget", "brand",
)
ACCESSORIES_PRIORITY = (
    "material", "feature", "other", "style", "color", "size", "budget", "brand", "use_case",
)
QUESTIONS = {
    "category": "What type of product are you looking for?",
    "material": "Do you have a preferred material?",
    "color": "Do you have a preferred color?",
    "size": "What size are you looking for?",
    "style": "Do you have a preferred style?",
    "brand": "Do you have a preferred brand?",
    "budget": "Do you have a budget range?",
    "feature": "Which product feature matters most to you?",
    "use_case": "What will you mainly use it for?",
    "other": "Is there another requirement that matters to you?",
}


@dataclass(frozen=True)
class ClarificationDecision:
    """A pure ask-or-act result for the current turn."""

    ask_attribute: str | None
    should_ask: bool

    def __post_init__(self) -> None:
        """Reject inconsistent decisions and attributes outside the API contract."""

        if self.ask_attribute is not None and self.ask_attribute not in ALLOWED_ASK_ATTRIBUTES:
            raise ValueError(f"unsupported ask attribute: {self.ask_attribute!r}")
        if self.should_ask != (self.ask_attribute is not None):
            raise ValueError("should_ask must match whether ask_attribute is set")


def evaluate_previous_ask_yield(
    state: SessionState,
    extraction: SlotExtraction,
) -> bool | None:
    """Return whether the immediately previous question yielded its attribute."""

    attribute = state.last_asked_attribute
    if attribute is None:
        return None
    if attribute == "other":
        return bool(
            extraction.revealed_text
            or any(extraction.slots.get(name) for name in SUPPORTED_SLOTS)
        )
    return bool(extraction.slots.get(attribute))


def _project_slots(
    state: SessionState,
    extraction: SlotExtraction,
    override_resolution: OverrideResolution | None,
) -> dict[str, list[SlotValue]]:
    """Project post-turn operational slots without changing session state."""

    projected = {
        slot_name: list(state.slots[slot_name])
        for slot_name in SUPPORTED_SLOTS
    }
    for slot_name in SUPPORTED_SLOTS:
        for value in extraction.slots.get(slot_name, ()):
            if value not in projected[slot_name]:
                projected[slot_name].append(value)

    if override_resolution is None:
        return projected
    for slot_name in override_resolution.clear_slots:
        projected[slot_name] = []
    for slot_name, values in override_resolution.remove_values.items():
        if slot_name not in override_resolution.clear_slots:
            projected[slot_name] = [
                value for value in projected[slot_name] if value not in values
            ]
    for slot_name, values in override_resolution.replacement_values.items():
        if slot_name not in override_resolution.clear_slots:
            projected[slot_name] = list(values)
    return projected


def _category_group(category_values: Sequence[SlotValue]) -> str:
    """Map granular catalog categories into one small priority group."""

    category_text = " ".join(
        value.casefold() for value in category_values if isinstance(value, str)
    )
    if any(term in category_text for term in (
        "shoe", "boot", "sneaker", "slipper", "sandal", "loafer", "mule",
        "clog", "footwear", "walking", "running",
    )):
        return "shoes"
    if any(term in category_text for term in (
        "accessor", "jewelry", "jewellery", "bracelet", "necklace", "earring",
        "ring", "watch", "belt", "wallet", "card case", "hat", "band",
    )):
        return "accessories"
    if any(term in category_text for term in (
        "shirt", "tee", "blouse", "tunic", "dress", "skirt", "pant", "jean",
        "legging", "short", "coat", "jacket", "sweater", "hoodie", "clothing",
        "camisole", "underwear", "sock",
    )):
        return "clothing"
    return "default"


def _priority_for(projected_slots: Mapping[str, Sequence[SlotValue]]) -> tuple[str, ...]:
    """Return a compact category-aware ordering of useful missing attributes."""

    group = _category_group(projected_slots["category"])
    if group == "clothing":
        return CLOTHING_PRIORITY
    if group == "shoes":
        # Size becomes decisive when the only known facts are weak selectors.
        weak_known = sum(bool(projected_slots[name]) for name in ("color", "brand", "style"))
        if weak_known >= 2 and not projected_slots["size"]:
            return ("size", *tuple(name for name in SHOES_PRIORITY if name != "size"))
        return SHOES_PRIORITY
    if group == "accessories":
        return ACCESSORIES_PRIORITY
    return DEFAULT_PRIORITY


def _has_enough_information(
    state: SessionState,
    projected_slots: Mapping[str, Sequence[SlotValue]],
    intent: ShoppingIntent,
    turn: int,
    previous_ask_yield: bool | None,
) -> bool:
    """Apply intent-aware stopping thresholds to current operational facts."""

    if not projected_slots["category"]:
        return False
    known = {
        name for name in INFORMATION_WEIGHTS if projected_slots[name]
    }
    if not known:
        return False

    score = sum(INFORMATION_WEIGHTS[name] for name in known)
    score += sum(
        1 for name in known
        if name in state.hard_constraints
    )
    # Exploratory users need less specificity, but one isolated attribute is
    # not a reliable stopping point in this catalog. Two strong facts (or an
    # equivalent mix) keep browsing concise while buying remains more exacting.
    #
    # Buying was 6 while intent detection was broken and classified 74 of 80 buying
    # sessions as browsing, so this branch almost never ran and 6 was never really
    # tested. With detection corrected (E8) it fires on all 80 and 6 asks too much:
    # 6 scores 0.7939 against 0.7975 at 5 and 0.7977 at 4. Five keeps intent
    # genuinely driving behaviour while costing nothing measurable.
    threshold = 4 if intent == "browsing" else 5
    if previous_ask_yield is False:
        threshold = max(4, threshold - 1)

    group = _category_group(projected_slots["category"])
    critical = {
        "shoes": {"size", "use_case", "feature", "material"},
        "clothing": {"size", "material", "style", "use_case", "feature"},
        "accessories": {"material", "feature", "style", "size"},
        "default": {"size", "material", "use_case", "feature", "budget"},
    }[group]
    if intent == "buying" and not (known & critical):
        return False
    if turn >= 6 and known:
        return True
    return score >= threshold


def decide_clarification(
    state: SessionState,
    extraction: SlotExtraction,
    turn: int,
    *,
    current_intent: ShoppingIntent | None = None,
    override_resolution: OverrideResolution | None = None,
    previous_ask_yield: bool | None = None,
) -> ClarificationDecision:
    """Choose one missing useful attribute, or act when current facts suffice."""

    projected = _project_slots(state, extraction, override_resolution)
    intent = current_intent or state.intent or "browsing"
    current_has_information = bool(
        extraction.revealed_text
        or any(extraction.slots.get(name) for name in SUPPORTED_SLOTS)
    )
    # A user who continues after a recommendation-only response is evidence that
    # the previous operational view was not sufficient. Resume with the next
    # untried attribute instead of returning the same results indefinitely.
    continuing_after_act = bool(
        state.turn > 0
        and state.last_asked_attribute is None
        and not current_has_information
    )
    if (
        not continuing_after_act
        and _has_enough_information(
            state, projected, intent, turn, previous_ask_yield,
        )
    ):
        return ClarificationDecision(None, False)

    if not projected["category"] and "category" not in state.asked_attributes:
        return ClarificationDecision("category", True)

    for attribute in _priority_for(projected):
        if (
            (attribute != "other" and projected[attribute])
            or attribute in state.asked_attributes
        ):
            continue
        return ClarificationDecision(attribute, True)
    return ClarificationDecision(None, False)


def select_response_ask_attribute(
    state: SessionState,
    decision: ClarificationDecision,
    *,
    previous_ask_yield: bool | None,
) -> str | None:
    """Expose other initially or after its immediately previous useful yield."""

    if decision.ask_attribute is not None:
        return decision.ask_attribute
    if "other" not in state.asked_attributes:
        return "other"
    if state.last_asked_attribute == "other" and previous_ask_yield is True:
        return "other"
    return None


def clarification_message(
    decision: ClarificationDecision,
    *,
    response_ask_attribute: str | None = None,
) -> str:
    """Build a response matching the policy or its API-facing fallback."""

    base = "Here are the closest matches I found."
    attribute = response_ask_attribute or decision.ask_attribute
    if attribute is None:
        return base
    return f"{base} {QUESTIONS[attribute]}"


def apply_clarification_decision(
    state: SessionState,
    decision: ClarificationDecision,
    previous_ask_yield: bool | None,
    *,
    response_ask_attribute: str | None = None,
) -> None:
    """Commit policy/output ask tracking only after response success."""

    attribute = response_ask_attribute or decision.ask_attribute
    state.last_ask_yielded = previous_ask_yield
    state.last_asked_attribute = attribute
    if attribute is not None:
        state.asked_attributes.add(attribute)
