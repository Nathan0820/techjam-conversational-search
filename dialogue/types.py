from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


Number: TypeAlias = int | float


@dataclass(frozen=True)
class BudgetConstraint:
    """A monetary bound extracted without performing currency conversion."""

    minimum: Number | None = None
    maximum: Number | None = None
    currency: str | None = None
    approximate: bool = False


SlotValue: TypeAlias = str | BudgetConstraint
