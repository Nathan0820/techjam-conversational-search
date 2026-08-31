"""Attribute a missed session to the stage that lost it.

A target can be missed two ways: retrieval never surfaced it, or retrieval surfaced it
and ranking pushed it out of the top 10. Those need different fixes and different
owners, so the distinction is recorded rather than inferred later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def ranking_failure_record(
    session_id: str,
    target_asin: str,
    retrieved_candidates: Sequence[Mapping[str, Any]],
    ranked_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a JSON-serializable record that assigns failures to the right stage."""

    initial_ids = [_asin(item) for item in retrieved_candidates]
    final_ids = [_asin(item.get("product", item)) for item in ranked_candidates]
    initial_rank = _rank(initial_ids, target_asin)
    final_rank = _rank(final_ids, target_asin)
    if initial_rank is None:
        reason = "target_not_retrieved"
    elif final_rank is None:
        reason = "target_dropped_by_reranker"
    elif final_rank > 1:
        reason = "target_ranked_below_competitor"
    else:
        reason = "none"
    competitor = final_ids[0] if final_ids and final_ids[0] != target_asin else None
    return {
        "session_id": session_id,
        "target_asin": target_asin,
        "target_retrieved": initial_rank is not None,
        "target_initial_rank": initial_rank,
        "target_final_rank": final_rank,
        "top_competing_product": competitor,
        "failure_reason": reason,
    }


def _asin(item: Mapping[str, Any]) -> str | None:
    """Read a product id under either key, or None when absent or blank."""

    value = item.get("parent_asin", item.get("asin"))
    return str(value) if value not in (None, "") else None


def _rank(values: Sequence[str | None], target: str) -> int | None:
    """Return the 1-based position of `target`, or None if it is not present."""

    try:
        return values.index(target) + 1
    except ValueError:
        return None
