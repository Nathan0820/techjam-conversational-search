from __future__ import annotations

from dataclasses import dataclass, field


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


def _empty_slots() -> dict[str, list[str]]:
    return {name: [] for name in SUPPORTED_SLOTS}


@dataclass
class SessionState:
    """Current agent-visible state for one shopping conversation.

    ``user_profile`` is historical preference context supplied at reset time,
    while ``message_history`` records what happened in this conversation.
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
    slots: dict[str, list[str]] = field(default_factory=_empty_slots)
    hard_constraints: set[str] = field(default_factory=set)
    soft_preferences: set[str] = field(default_factory=set)
    asked_attributes: set[str] = field(default_factory=set)
    last_ask_yielded: bool | None = None
    turn: int = 0
    override_detected: bool = False
    message_history: list[str] = field(default_factory=list)
    revealed_text: list[str] = field(default_factory=list)
