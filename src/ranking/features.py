"""Turn one product and the conversation state into comparable ranking signals.

Every signal is either a score in [0, 1] or None. None means "the customer never
mentioned this", which is different from zero, and the reranker skips it rather than
counting it against the product. That distinction matters because a customer who has
said nothing about colour should not penalise every product for failing to match a
colour they never asked for.

Products are read defensively. Catalog records vary in which fields they populate,
and callers may pass either the current SessionState or a plain mapping.
"""

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
PREPARED_TEXT_KEY = "_ranking_text"
PREPARED_CATEGORY_KEY = "_ranking_category_text"
PREPARED_BRAND_KEY = "_ranking_brand_text"


def state_value(state: object, key: str, default: Any = None) -> Any:
    """Read `key` from either a mapping or an object, so both state shapes work."""

    if isinstance(state, Mapping):
        return state.get(key, default)
    return getattr(state, key, default)


def flatten(value: object) -> list[str]:
    """Reduce a nested catalog or state value to a flat list of strings."""

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


def flatten_values(value: object) -> list[str]:
    """Flatten values while omitting mapping/dataclass schema field names."""

    if value is None:
        return []
    if is_dataclass(value) and not isinstance(value, type):
        return flatten_values(asdict(value))
    if isinstance(value, Mapping):
        return [term for item in value.values() for term in flatten_values(item)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [term for item in value for term in flatten_values(item)]
    return [str(value)]


def normalize_text(value: object) -> str:
    """Lowercase, strip punctuation, and join to a single space-separated string."""

    return " ".join(TOKEN_RE.findall(" ".join(flatten(value)).lower()))


def product_text(product: Mapping[str, Any]) -> str:
    """Return the product's searchable text, reusing a prepared value when present.

    prepare_product() caches this at catalog load time; recomputing it per candidate
    per turn would repeat the same normalisation thousands of times per session.
    """

    prepared = product.get(PREPARED_TEXT_KEY)
    if isinstance(prepared, str):
        return prepared
    fields = ("title", "description", "features", "details", "categories", "category", "store", "brand")
    return normalize_text([product.get(field) for field in fields])


def prepare_product(product: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a catalog product and cache immutable normalized ranking text."""

    prepared = dict(product)
    prepared[PREPARED_TEXT_KEY] = normalize_text([
        product.get(field)
        for field in ("title", "description", "features", "details", "categories", "category", "store", "brand")
    ])
    prepared[PREPARED_CATEGORY_KEY] = normalize_text([
        product.get("category"), product.get("categories"), product.get("title")
    ])
    prepared[PREPARED_BRAND_KEY] = normalize_text([
        product.get("brand"), product.get("store"), product.get("details"), product.get("title")
    ])
    return prepared


def _constraint_values(constraints: Mapping[str, Any], key: str) -> list[str]:
    """Look up a constraint by slot name, accepting the aliases in FIELD_ALIASES."""

    for alias in FIELD_ALIASES.get(key, (key,)):
        if alias in constraints:
            return [value for value in flatten(constraints[alias]) if value]
    return []


def _phrase_match(values: list[str], text: str) -> float | None:
    """Score how well `values` appear in `text`, in [0, 1], or None if none were given.

    A value whose words appear consecutively scores 1.0; otherwise it scores the
    fraction of its words present anywhere. Partial credit matters because a customer
    saying "high quality mesh" should still favour a mesh product that words it
    differently.
    """

    if not values:
        return None
    ordered_text_tokens = tuple(text.split())
    text_tokens = set(ordered_text_tokens)
    matches = []
    for value in values:
        phrase = normalize_text(value)
        ordered_phrase_tokens = tuple(phrase.split())
        if not ordered_phrase_tokens:
            continue
        width = len(ordered_phrase_tokens)
        exact = any(
            ordered_text_tokens[index:index + width] == ordered_phrase_tokens
            for index in range(len(ordered_text_tokens) - width + 1)
        )
        phrase_tokens = set(ordered_phrase_tokens)
        matches.append(
            1.0 if exact
            else len(phrase_tokens & text_tokens) / len(phrase_tokens)
        )
    return sum(matches) / len(matches) if matches else 0.0


def _mapping(value: object) -> Mapping[str, Any]:
    """Return `value` if it is a mapping, otherwise an empty one."""

    return value if isinstance(value, Mapping) else {}


def _preference_terms(value: object) -> tuple[list[str], list[str]]:
    """Split a preference structure into (wanted terms, unwanted terms)."""

    positive: list[str] = []
    negative: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            enabled = item is not False and item is not None
            # Legacy boolean maps encode the preference in the key. Current
            # SessionState maps slot names to values, where the schema key itself
            # (for example "brand") is not useful product evidence.
            terms = [str(key)] if isinstance(item, bool) else flatten_values(item)
            (positive if enabled else negative).extend(terms)
    else:
        positive.extend(flatten_values(value))
    return positive, negative


def _number(value: object) -> float | None:
    """Coerce a value to a float, digging a number out of text; None when there is none.

    Booleans are rejected explicitly, since Python would otherwise read True as 1.0.
    """

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


def price_bounds(
    constraints: Mapping[str, Any],
) -> tuple[float | None, float | None, bool] | None:
    """Return the strictest active minimum/maximum and approximation flag."""

    minima: list[float] = []
    maxima: list[float] = []
    approximate = False
    for key in PRICE_KEYS:
        if key in constraints:
            value = constraints[key]
            values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
            for item in values:
                if is_dataclass(item) and not isinstance(item, type):
                    item = asdict(item)
                if isinstance(item, Mapping):
                    minimum = _number(item.get("minimum"))
                    maximum = _number(item.get("maximum"))
                    approximate = approximate or bool(item.get("approximate"))
                else:
                    minimum = None
                    maximum = _number(item)
                if minimum is not None:
                    minima.append(minimum)
                if maximum is not None:
                    maxima.append(maximum)
    if not minima and not maxima:
        return None
    return (
        max(minima) if minima else None,
        min(maxima) if maxima else None,
        approximate,
    )


def max_price(constraints: Mapping[str, Any]) -> float | None:
    """Compatibility wrapper returning only the active upper price bound."""

    bounds = price_bounds(constraints)
    return bounds[1] if bounds is not None else None


def _price_match(
    bounds: tuple[float | None, float | None, bool] | None,
    price: float | None,
) -> float | None:
    """Score budget fit in [0, 1], or None when no budget was stated.

    An unknown price scores 0.25 rather than 0.0: most catalog records omit price, so
    treating a missing value as a failure would bury products for a fact we never had.
    Prices outside the stated bounds degrade with distance instead of dropping to zero,
    and an approximate budget scores by closeness to the stated figure.
    """

    if bounds is None:
        return None
    if price is None:
        # Unknown is weaker than known compliance but should not be excluded: most
        # catalog records do not expose price.
        return 0.25
    minimum, maximum, approximate = bounds
    if approximate and minimum is not None and maximum == minimum:
        scale = max(abs(minimum), 1.0)
        return max(0.0, 1.0 - abs(price - minimum) / scale)
    if minimum is not None and price < minimum:
        return max(0.0, price / max(minimum, 1.0))
    if maximum is not None and price > maximum:
        return max(0.0, 1.0 - (price - maximum) / max(maximum, 1.0))
    return 1.0


def _price_violates(
    bounds: tuple[float | None, float | None, bool] | None,
    price: float | None,
) -> str | None:
    """Name the budget bound this price breaks, or None.

    Only a known price against an exact budget counts as a violation; an approximate
    budget expresses a preference rather than a limit, so it never produces one.
    """

    if bounds is None or price is None:
        return None
    minimum, maximum, approximate = bounds
    if approximate:
        return None
    if minimum is not None and price < minimum:
        return "min_price"
    if maximum is not None and price > maximum:
        return "max_price"
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
    """Per-candidate ranking signals. None means the customer never mentioned it."""

    category_match: float | None = None
    brand_match: float | None = None
    color_match: float | None = None
    size_match: float | None = None
    material_match: float | None = None
    style_match: float | None = None
    feature_match: float | None = None
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
    """Compare one product against the conversation state.

    Fills every match signal the reranker consumes and records `hard_violations`,
    the names of requirements this product demonstrably breaks. Violations are
    reported rather than acted on here; the reranker decides what they cost.
    """

    active, hard, soft = _constraint_views(state)
    negative = state_value(state, "negative_preferences", state_value(state, "excluded_preferences", {}))
    positive_terms, soft_negative = _preference_terms(soft)
    explicit_negative, disabled_negative = _preference_terms(negative)
    negative_terms = [*soft_negative, *explicit_negative, *disabled_negative]
    text = product_text(product)
    category_text = str(product.get(PREPARED_CATEGORY_KEY) or normalize_text([
        product.get("category"), product.get("categories"), product.get("title")
    ]))
    brand_text = str(product.get(PREPARED_BRAND_KEY) or normalize_text([
        product.get("brand"), product.get("store"), product.get("details"), product.get("title")
    ]))

    result = ProductFeatures(
        category_match=_phrase_match(_constraint_values(active, "category"), category_text),
        brand_match=_phrase_match(_constraint_values(active, "brand"), brand_text),
        color_match=_phrase_match(_constraint_values(active, "color"), text),
        size_match=_phrase_match(_constraint_values(active, "size"), text),
        material_match=_phrase_match(_constraint_values(active, "material"), text),
        style_match=_phrase_match(_constraint_values(active, "style"), text),
        feature_match=_phrase_match(_constraint_values(active, "feature"), text),
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
    bounds = price_bounds(active)
    hard_bounds = price_bounds(hard)
    price = _number(product.get("price"))
    result.price_match = _price_match(bounds, price)

    hard_match_text = {
        "category": category_text,
        "brand": brand_text,
        "color": text,
        "size": text,
        "material": text,
        "style": text,
        "feature": text,
        "use_case": text,
    }
    for name, candidate_text in hard_match_text.items():
        match = _phrase_match(_constraint_values(hard, name), candidate_text)
        if match is not None and match < 1.0:
            result.hard_violations.append(name)
    price_violation = _price_violates(hard_bounds, price)
    if price_violation is not None:
        result.hard_violations.append(price_violation)
    return result


def state_to_query(state: object) -> str:
    """Serialize only active user evidence, without state-schema field names."""

    phrases = flatten_values(state_value(state, "active_revealed_text", []))
    slots = _mapping(state_value(state, "slots", {}))
    slot_values = [value for values in slots.values() for value in flatten_values(values)]
    hard = state_value(state, "hard_constraints", {})
    hard_values = flatten_values(hard) if isinstance(hard, Mapping) else []
    soft_values, _ = _preference_terms(state_value(state, "soft_preferences", {}))
    negatives = flatten_values(state_value(state, "negative_preferences", {}))
    profile = _mapping(state_value(state, "user_profile", {}))
    profile_tags = flatten_values(profile.get("preference_tags", []))
    query = state_value(state, "query", "")
    terms = [
        *phrases, *slot_values, *hard_values, *soft_values, *negatives,
        *profile_tags, *flatten_values(query),
    ]
    return " ".join(dict.fromkeys(term.strip() for term in terms if term.strip()))
