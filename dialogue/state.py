from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    """Current agent-visible state for one shopping conversation.

    ``user_profile`` is historical preference context supplied at reset time,
    while ``message_history`` records what happened in this conversation.
    The slots and constraint mappings are the current operational truth that
    later retrieval and ranking logic may mutate when the user's intent changes.
    """

    session_id: str
    user_profile: dict = field(default_factory=dict)
    intent: str | None = None
    slots: dict = field(default_factory=dict)
    hard_constraints: dict = field(default_factory=dict)
    soft_preferences: dict = field(default_factory=dict)
    asked_attributes: set[str] = field(default_factory=set)
    turn: int = 0
    override_detected: bool = False
    message_history: list[str] = field(default_factory=list)
