from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
FIELD_ALIASES = {
    "category": ("category", "categories", "target_category"),
    "brand": ("brand", "store", "manufacturer"),
    "color": ("color", "colour"),
    "size": ("size", "sizing", "width"),
    "material": ("material", "fabric"),
    "use_case": ("use_case", "use case", "occasion", "activity"),
}
PRICE_KEYS = ("max_price", "price_max", "budget", "price")


def state_value(state: object, key: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(key, default)
    return getattr(state, key, default)


def flatten(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            result.append(str(key))
            result.extend(flatten(item))
        return result
    if is_dataclass(value) and not isinstance(value, type):
        return flatten(asdict(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for item in value:
            result.extend(flatten(item))
        return result
    return [str(value)]


def normalize_text(value: object) -> str:
    return " ".join(TOKEN_RE.findall(" ".join(flatten(value)).lower()))


def product_text(product: Mapping[str, Any]) -> str:
    fields = ("title", "description", "features", "details", "categories", "category", "store", "brand")
    return normalize_text([product.get(field) for field in fields])


def _constraint_values(constraints: Mapping[str, Any], key: str) -> list[str]:
    for alias in FIELD_ALIASES.get(key, (key,)):
        if alias in constraints:
            return [value for value in flatten(constraints[alias]) if value]
    return []


def _phrase_match(values: list[str], text: str) -> float | None:
    if not values:
        return None
    text_tokens = set(text.split())
    matches = []
    for value in values:
        phrase = normalize_text(value)
        tokens = set(phrase.split())
        if not tokens:
            continue
        matches.append(1.0 if phrase in text else len(tokens & text_tokens) / len(tokens))
    return max(matches, default=0.0)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _preference_terms(value: object) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    negative: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            enabled = item is not False and item is not None
            terms = [str(key), *flatten(item)] if not isinstance(item, bool) else [str(key)]
            (positive if enabled else negative).extend(terms)
    else:
        positive.extend(flatten(value))
    return positive, negative


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = re.search(r"\d+(?:\.\d+)?", str(value or "").replace(",", ""))
        if not match:
            return None
        number = float(match.group())
    return number if math.isfinite(number) else None


def max_price(constraints: Mapping[str, Any]) -> float | None:
    for key in PRICE_KEYS:
        if key in constraints:
            value = constraints[key]
            values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
            maxima = []
            for item in values:
                if is_dataclass(item) and not isinstance(item, type):
                    item = asdict(item)
                if isinstance(item, Mapping):
                    maximum = _number(item.get("maximum"))
                else:
                    maximum = _number(item)
                if maximum is not None:
                    maxima.append(maximum)
            if maxima:
                return min(maxima)
    return None


def _constraint_views(state: object) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return active, hard, and soft values for both supported state schemas.

    Current ``SessionState`` keeps values in ``slots`` and strength labels in
    sets. Older callers and tests may still provide value mappings directly in
    ``hard_constraints`` and ``soft_preferences``.
    """

    slots = _mapping(state_value(state, "slots", {}))
    raw_hard = state_value(state, "hard_constraints", {})
    raw_soft = state_value(state, "soft_preferences", {})

    if isinstance(raw_hard, Mapping):
        hard = dict(raw_hard)
    else:
        hard = {name: slots.get(name, []) for name in raw_hard if name in slots}

    if isinstance(raw_soft, Mapping):
        soft = dict(raw_soft)
    else:
        soft = {name: slots.get(name, []) for name in raw_soft if name in slots}

    active = dict(slots)
    active.update(hard)
    # Extracted but not-yet-classified slots remain useful ranking evidence.
    for name, values in soft.items():
        active.setdefault(name, values)
    return active, hard, soft


@dataclass
class ProductFeatures:
    category_match: float | None = None
    brand_match: float | None = None
    color_match: float | None = None
    size_match: float | None = None
    material_match: float | None = None
    use_case_match: float | None = None
    price_match: float | None = None
    positive_preference_match: float | None = None
    negative_preference_match: float | None = None
    bm25_score: float | None = None
    dense_score: float | None = None
    rating: float | None = None
    review_count: float | None = None
    hard_violations: list[str] = field(default_factory=list)


def extract_features(product: Mapping[str, Any], state: object) -> ProductFeatures:
    active, hard, soft = _constraint_views(state)
    negative = state_value(state, "negative_preferences", state_value(state, "excluded_preferences", {}))
    positive_terms, soft_negative = _preference_terms(soft)
    explicit_negative, disabled_negative = _preference_terms(negative)
    negative_terms = [*soft_negative, *explicit_negative, *disabled_negative]
    text = product_text(product)

    result = ProductFeatures(
        category_match=_phrase_match(_constraint_values(active, "category"), normalize_text([product.get("category"), product.get("categories"), product.get("title")])),
        brand_match=_phrase_match(_constraint_values(active, "brand"), normalize_text([product.get("brand"), product.get("store"), product.get("details"), product.get("title")])),
        color_match=_phrase_match(_constraint_values(active, "color"), text),
        size_match=_phrase_match(_constraint_values(active, "size"), text),
        material_match=_phrase_match(_constraint_values(active, "material"), text),
        use_case_match=_phrase_match(_constraint_values(active, "use_case"), text),
        positive_preference_match=_phrase_match(positive_terms, text),
        negative_preference_match=_phrase_match(negative_terms, text),
        bm25_score=_number(product.get("bm25_score")),
        dense_score=_number(product.get("dense_score")),
        rating=_number(product.get("average_rating", product.get("rating"))),
        review_count=_number(product.get("rating_number", product.get("review_count"))),
    )

    # Budget values use the slot name in SessionState; legacy callers may use
    # max_price. All active budgets affect scoring, while only explicitly hard
    # budgets create a violation.
    limit = max_price(active)
    hard_limit = max_price(hard)
    price = _number(product.get("price"))
    if limit is not None and price is not None:
        result.price_match = 1.0 if price <= limit else 0.0

    hard_match_text = {
        "category": normalize_text([product.get("category"), product.get("categories"), product.get("title")]),
        "brand": normalize_text([product.get("brand"), product.get("store"), product.get("details"), product.get("title")]),
        "color": text,
        "size": text,
        "material": text,
        "use_case": text,
    }
    for name, candidate_text in hard_match_text.items():
        match = _phrase_match(_constraint_values(hard, name), candidate_text)
        if match == 0:
            result.hard_violations.append(name)
    if hard_limit is not None and price is not None and price > hard_limit:
        result.hard_violations.append("max_price")
    return result


def state_to_query(state: object) -> str:
    active = [
        state_value(state, "intent", ""),
        state_value(state, "slots", {}),
        state_value(state, "hard_constraints", {}),
        state_value(state, "soft_preferences", {}),
        state_value(state, "negative_preferences", {}),
        state_value(state, "query", ""),
    ]
    return " ".join(flatten(active)).strip()
