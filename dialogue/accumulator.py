"""Fold one turn's extraction into the accumulated session state.

Deliberately narrow: it adds slot values and exact phrases and nothing else. Intent,
constraint strength, overrides and dialogue policy are each decided by their own
module, so that accumulating information cannot silently change a classification.
"""

from __future__ import annotations

from .slot_extractor import SlotExtraction
from .state import SUPPORTED_SLOTS, SessionState


def accumulate_information(state: SessionState, extraction: SlotExtraction) -> None:
    """Merge one temporary extraction into current operational shopping state.

    This function accumulates slot values and exact raw phrases only. It does
    not update history, intent, constraint strength, overrides, or dialogue
    policy state.
    """

    unsupported = set(extraction.slots) - set(SUPPORTED_SLOTS)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported extracted slot name(s): {names}")

    for slot_name in SUPPORTED_SLOTS:
        state.add_slot_values(slot_name, extraction.slots.get(slot_name, ()))

    for phrase in extraction.revealed_text:
        if phrase not in state.revealed_text:
            state.revealed_text.append(phrase)
        if phrase not in state.active_revealed_text:
            state.active_revealed_text.append(phrase)
