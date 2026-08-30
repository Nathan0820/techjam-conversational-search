from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias

from .state import SUPPORTED_SLOTS


Number: TypeAlias = int | float


@dataclass(frozen=True)
class BudgetConstraint:
    """A monetary bound extracted without performing currency conversion."""

    minimum: Number | None = None
    maximum: Number | None = None
    currency: str | None = None
    approximate: bool = False


SlotValue: TypeAlias = str | BudgetConstraint


def _empty_extracted_slots() -> dict[str, list[SlotValue]]:
    return {name: [] for name in SUPPORTED_SLOTS}


@dataclass
class SlotExtraction:
    """Temporary facts found in one message; it does not mutate session state."""

    slots: dict[str, list[SlotValue]] = field(default_factory=_empty_extracted_slots)
    revealed_text: list[str] = field(default_factory=list)


MATERIALS = (
    "stainless steel", "faux fur", "full grain leather", "polyester", "spandex",
    "leather", "cotton", "nylon", "wool", "silk", "rayon", "fabric", "alloy",
    "rubber", "textile", "denim", "canvas", "suede", "linen",
)
COLORS = (
    "rose gold", "navy blue", "light blue", "dark blue", "hot pink", "black",
    "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "beige", "navy", "gold", "silver", "teal",
    "maroon", "multicolor",
)
STYLE_PHRASES = (
    "slim fit", "relaxed fit", "regular fit", "loose fit", "crew neck", "v-neck",
    "long sleeve", "short sleeve", "sleeveless", "casual", "formal", "vintage",
    "classic", "modern", "minimalist", "oversized", "athletic fit",
)
FEATURE_PHRASES = (
    "water resistant", "machine washable", "machine wash", "drawstring closure",
    "buckle closure", "pull on closure", "rubber sole", "non-slip", "non slip",
    "waterproof", "breathable", "lightweight", "hypoallergenic", "insulated",
    "quick dry", "quick-dry", "wrinkle resistant", "stretchy", "adjustable",
    "moisture wicking", "uv protection", "fur lined",
)
USE_CASE_PHRASES = (
    "everyday wear", "casual wear", "cold weather", "snow", "winter", "hiking",
    "running", "walking", "basketball", "gym", "workout", "outdoor", "work",
    "travel", "wedding", "office", "school", "swimming", "cycling", "camping",
)
GENERIC_CATEGORIES = {
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry", "men",
    "women", "boys", "girls", "unisex", "accessories", "casual", "active",
}
GENERIC_BRANDS = {"unknown", "generic", "men", "women", "amazon"}
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'’+.-]*")


@lru_cache(maxsize=1)
def _catalog_vocabulary() -> tuple[dict[str, str], dict[str, str]]:
    """Load catalog brands/categories once, with a small-rule fallback."""

    brands: dict[str, str] = {}
    categories: dict[str, str] = {}
    catalog_path = Path(__file__).resolve().parents[1] / "data" / "catalog.jsonl"
    if not catalog_path.exists():
        return brands, categories
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            brand = str(product.get("store") or "").strip()
            if brand and brand.casefold() not in GENERIC_BRANDS:
                brands.setdefault(brand.casefold(), brand)
            for category in product.get("categories") or []:
                value = str(category).strip()
                if value and value.casefold() not in GENERIC_CATEGORIES:
                    categories.setdefault(value.casefold(), value)
                    if value.casefold().endswith("s") and not value.casefold().endswith("ss"):
                        categories.setdefault(value.casefold()[:-1], value)
    return brands, categories


def _ngrams(message: str, maximum: int = 5) -> list[tuple[str, str]]:
    tokens = list(TOKEN_RE.finditer(message))
    result: list[tuple[str, str]] = []
    for length in range(min(maximum, len(tokens)), 0, -1):
        for start in range(len(tokens) - length + 1):
            end = start + length - 1
            raw = message[tokens[start].start():tokens[end].end()]
            result.append((raw.casefold(), raw))
    return result


def _add_value(values: list[SlotValue], value: SlotValue) -> None:
    key = value.casefold() if isinstance(value, str) else value
    if all((item.casefold() if isinstance(item, str) else item) != key for item in values):
        values.append(value)


def _add_raw(result: SlotExtraction, raw: str) -> None:
    raw = raw.strip()
    if not raw:
        return
    folded = raw.casefold()
    if any(folded in existing.casefold() for existing in result.revealed_text):
        return
    result.revealed_text.append(raw)


def _extract_explicit_reveals(message: str, result: SlotExtraction) -> None:
    patterns = (
        r"(?i)key requirement is:\s*(.+?)(?=\s*$)",
        r"(?i)what matters is:\s*(.+?)(?=\s*$)",
        r"(?i)what i need is:\s*(.+?)(?=\s*$)",
    )
    for pattern in patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        for phrase in re.split(r"\s*;\s*", match.group(1)):
            _add_raw(result, phrase.rstrip(".?!"))


def _extract_terms(message: str, result: SlotExtraction, slot: str, terms: tuple[str, ...]) -> None:
    candidates: list[tuple[int, int, str, str]] = []
    for term in terms:
        for match in re.finditer(rf"(?i)(?<!\w){re.escape(term)}(?!\w)", message):
            candidates.append((match.start(), match.end(), term, match.group(0)))
    occupied: list[tuple[int, int]] = []
    for start, end, term, raw in sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        normalized = "gray" if slot == "color" and term == "grey" else term
        _add_value(result.slots[slot], normalized)
        _add_raw(result, raw)
        occupied.append((start, end))


def _number(value: str) -> Number:
    parsed = float(value.replace(",", ""))
    return int(parsed) if parsed.is_integer() else parsed


def _currency(raw: str) -> str | None:
    if re.search(r"(?i)\bSGD\b", raw):
        return "SGD"
    if re.search(r"(?i)\bUSD\b", raw):
        return "USD"
    return "$" if "$" in raw else None


def _extract_budget(message: str, result: SlotExtraction) -> None:
    amount = r"(?:SGD|USD|\$)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)"
    patterns = (
        (rf"(?i)\b(?:between)\s+{amount}\s+(?:and)\s+{amount}", "range"),
        (rf"(?i)\b(?:from)\s+{amount}\s+(?:to)\s+{amount}", "range"),
        (rf"(?i)\b(?:under|below|less than|up to|maximum|max)\s+{amount}", "max"),
        (rf"(?i)\b(?:over|above|more than|minimum|min)\s+{amount}", "min"),
        (rf"(?i)\b(?:around|about|approximately|approx\.?|budget around)\s+{amount}", "approx"),
    )
    for pattern, kind in patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        raw = match.group(0)
        if kind == "range":
            budget = BudgetConstraint(_number(match.group(1)), _number(match.group(2)), _currency(raw))
        elif kind == "max":
            budget = BudgetConstraint(maximum=_number(match.group(1)), currency=_currency(raw))
        elif kind == "min":
            budget = BudgetConstraint(minimum=_number(match.group(1)), currency=_currency(raw))
        else:
            value = _number(match.group(1))
            budget = BudgetConstraint(value, value, _currency(raw), approximate=True)
        _add_value(result.slots["budget"], budget)
        _add_raw(result, raw)
        return


def _extract_sizes(message: str, result: SlotExtraction) -> None:
    patterns = (
        r"(?i)\b(?:size|shoe size)\s*[:#-]?\s*(XXXS|XXS|XS|S|M|L|XL|XXL|XXXL|\d+(?:\.5)?)\b",
        r"(?<![A-Za-z0-9])((?:[2-6]X)|XXXL|XXL|XL|XS|XXS|XXXS)(?![A-Za-z0-9])",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, message):
            value = match.group(1).upper()
            _add_value(result.slots["size"], value)
            _add_raw(result, match.group(0))


def _extract_catalog_terms(message: str, result: SlotExtraction) -> None:
    brands, categories = _catalog_vocabulary()
    occupied: list[tuple[int, int]] = []
    for folded, raw in _ngrams(message):
        start = message.casefold().find(folded)
        span = (start, start + len(raw))
        if start < 0 or any(start < end and span[1] > begin for begin, end in occupied):
            continue
        looks_named = " " in raw or any(character.isupper() for character in raw)
        if folded in brands and looks_named:
            _add_value(result.slots["brand"], brands[folded])
            _add_raw(result, raw)
            occupied.append(span)
        if (
            folded in categories
            and folded not in GENERIC_CATEGORIES
            and not any(
                folded == str(value).casefold()
                for slot in ("material", "color", "style", "feature", "use_case")
                for value in result.slots[slot]
            )
        ):
            _add_value(result.slots["category"], categories[folded])
            _add_raw(result, raw)


def extract_slots(user_message: str) -> SlotExtraction:
    """Extract explicit shopping facts from one message without changing state."""

    result = SlotExtraction()
    if not user_message or not user_message.strip():
        return result
    _extract_explicit_reveals(user_message, result)
    _extract_budget(user_message, result)
    _extract_sizes(user_message, result)
    _extract_terms(user_message, result, "material", MATERIALS)
    _extract_terms(user_message, result, "color", COLORS)
    _extract_terms(user_message, result, "style", STYLE_PHRASES)
    _extract_terms(user_message, result, "feature", FEATURE_PHRASES)
    _extract_terms(user_message, result, "use_case", USE_CASE_PHRASES)
    _extract_catalog_terms(user_message, result)
    return result
