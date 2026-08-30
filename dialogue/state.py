from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .types import SlotValue


SUPPORTED_SLOTS = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
)


def _empty_slots() -> dict[str, list[SlotValue]]:
    return {name: [] for name in SUPPORTED_SLOTS}


@dataclass
class SessionState:
    """Current agent-visible state for one shopping conversation.

    ``user_profile`` is historical preference context supplied at reset time,
    while ``message_history`` records the complete structured user and assistant
    conversation in chronological order.
    ``revealed_text`` preserves exact retrieval-useful wording identified later;
    its strings must not be normalized or rewritten in place. ``slots`` is the
    authoritative store for current constraint values. The hard and soft sets
    contain slot names only, classifying how values in ``slots`` should be used.

    The operational fields ``slots``, ``hard_constraints``, and
    ``soft_preferences`` must contain only the user's current valid preferences.
    Future override handling must erase or replace stale operational values;
    setting ``override_detected`` alone is not sufficient. Historical messages
    may remain in ``message_history`` for context and debugging.
    """

    session_id: str
    user_profile: dict = field(default_factory=dict)
    intent: str | None = None
    slots: dict[str, list[SlotValue]] = field(default_factory=_empty_slots)
    hard_constraints: set[str] = field(default_factory=set)
    soft_preferences: set[str] = field(default_factory=set)
    asked_attributes: set[str] = field(default_factory=set)
    last_ask_yielded: bool | None = None
    turn: int = 0
    override_detected: bool = False
    message_history: list[dict[str, str]] = field(default_factory=list)
    revealed_text: list[str] = field(default_factory=list)

    def set_constraint(
        self,
        slot_name: str,
        values: Iterable[SlotValue],
        strength: str | None = None,
    ) -> None:
        """Replace one slot's values and keep hard/soft labels consistent.

        Values live only in ``slots``. The hard and soft sets contain slot names
        solely as mutually exclusive classifications of non-empty slots.
        """

        self._validate_slot_name(slot_name)
        if strength not in {None, "hard", "soft"}:
            raise ValueError("strength must be 'hard', 'soft', or None")

        stored_values = list(values)
        self.slots[slot_name] = stored_values
        self.hard_constraints.discard(slot_name)
        self.soft_preferences.discard(slot_name)
        if not stored_values:
            return
        if strength == "hard":
            self.hard_constraints.add(slot_name)
        elif strength == "soft":
            self.soft_preferences.add(slot_name)

    def add_slot_values(self, slot_name: str, values: Iterable[SlotValue]) -> None:
        """Append new unclassified values while preserving order and labels."""

        self._validate_slot_name(slot_name)
        current_values = self.slots[slot_name]
        for value in values:
            if value not in current_values:
                current_values.append(value)

    def clear_constraint(self, slot_name: str) -> None:
        """Clear a slot's values and remove all strength classification."""

        self._validate_slot_name(slot_name)
        self.slots[slot_name] = []
        self.hard_constraints.discard(slot_name)
        self.soft_preferences.discard(slot_name)

    @staticmethod
    def _validate_slot_name(slot_name: str) -> None:
        if slot_name not in SUPPORTED_SLOTS:
            raise ValueError(f"unsupported slot name: {slot_name!r}")
