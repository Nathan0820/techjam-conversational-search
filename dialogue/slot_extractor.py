from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .state import SUPPORTED_SLOTS
from .types import BudgetConstraint, Number, SlotValue


def _empty_extracted_slots() -> dict[str, list[SlotValue]]:
    return {name: [] for name in SUPPORTED_SLOTS}


@dataclass
class SlotExtraction:
    """Temporary facts found in one message; it does not mutate session state."""

    slots: dict[str, list[SlotValue]] = field(default_factory=_empty_extracted_slots)
    revealed_text: list[str] = field(default_factory=list)
    retrieval_hints: list[str] = field(default_factory=list)


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
INITIAL_PREFERENCE_RE = re.compile(
    r"^\s*i(?:'m|\s+am)\s+looking\s+for\s+.+?[.!?]\s*(?P<phrase>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
EXPLICIT_REVEAL_RE = re.compile(
    r"\b(?:key\s+requirement\s+is|what\s+matters\s+is|what\s+i\s+need\s+is)\s*:",
    re.IGNORECASE,
)
OVERRIDE_SCAFFOLD_RE = re.compile(
    r"\b(?:actually|ignore\s+my\s+earlier\s+preference|forget\s+that|"
    r"scratch\s+that|never\s+mind|i\s+changed\s+my\s+mind)\b",
    re.IGNORECASE,
)
FILLER_RE = re.compile(
    r"^\s*(?:thanks?(?:\s+you)?|okay|ok|what\s+do\s+you\s+have|"
    r"show\s+me\s+(?:some\s+)?options?|anything\s+else)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
CONVERSATIONAL_FOLLOWUP_RE = re.compile(
    r"^\s*(?:thanks?\b|thank\s+you\b)|"
    r"\b(?:that\s+helped|good\s+advice|sounds\s+good)\b",
    re.IGNORECASE,
)
SHOPPING_TARGET_RE = re.compile(
    r"\b(?:looking\s+for|shopping\s+for|need|want|find|show\s+me|"
    r"prefer|interested\s+in|considering)\b",
    re.IGNORECASE,
)
COMPONENT_RELATION_RE = re.compile(
    r"\b(?:with|featuring|including|includes?|has|comes\s+with)\b",
    re.IGNORECASE,
)
PRODUCT_ATTRIBUTE_RE = re.compile(
    r"\b(?:band|bracelet|button|closure|collar|comfort|coverage|cuff|department|"
    r"fit|heel|length|lined|lining|midsole|neck|pocket|rise|size|sleeve|sole|"
    r"strap|torso|vamp|wash|waist|waterproof|width|zip|zipper)\b",
    re.IGNORECASE,
)
ATTRIBUTE_NAME_PATTERN = (
    r"(?:product\s+type|item\s+type|use\s+case|category|material|fabric|"
    r"color|colour|size|sizing|style|fit|brand|budget|price|feature|other)"
)
NO_PREFERENCE_CLAUSE_RE = re.compile(
    rf"\b(?:"
    rf"i\s+(?:(?:do\s+not|don['’]t)\s+have\s+"
    rf"(?:(?:an?|any)\s+)?(?:additional\s+)?preference|"
    rf"have\s+no\s+(?:additional\s+)?preference)|"
    rf"no\s+(?:additional\s+)?preference"
    rf")\s+(?:for|about|on)\s+(?:the\s+)?"
    rf"(?P<attribute>{ATTRIBUTE_NAME_PATTERN})"
    rf"(?:\s+anymore)?\b"
    rf"(?:\s*;\s*please\s+use\s+your\s+judg(?:e)?ment)?",
    re.IGNORECASE,
)
NON_EVIDENCE_SENTENCE_RE = re.compile(
    r"\b(?:those\s+options\s+(?:are\s+not|aren['’]t)\s+quite\s+right\s+yet|"
    r"ask\s+me\s+about\s+one\s+specific\s+attribute|"
    r"please\s+use\s+your\s+judg(?:e)?ment)\b",
    re.IGNORECASE,
)
USE_JUDGMENT_RE = re.compile(
    r"\bplease\s+use\s+your\s+judg(?:e)?ment\b",
    re.IGNORECASE,
)


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
            result.append((raw.rstrip(".,!?;:").casefold(), raw))
    return result


def _retrieval_evidence_text(message: str) -> tuple[str, list[str]]:
    """Remove dialogue clauses and return any one-turn attribute hints."""

    no_preference_matches = list(NO_PREFERENCE_CLAUSE_RE.finditer(message))
    cleaned = NO_PREFERENCE_CLAUSE_RE.sub(" ", message)
    cleaned = NON_EVIDENCE_SENTENCE_RE.sub(" ", cleaned)
    if cleaned == message:
        return message, []
    cleaned = cleaned.strip(" \t\r\n,;:.!?-")
    cleaned = re.sub(r"^(?:but|and)\b\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\s*(?:,|;)?\s*\b(?:but|and)\b\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" \t\r\n,;:.!?-")
    # A pure no-preference answer still identifies the attribute discussed. Keep
    # that word as a current-turn retrieval hint without treating it as a user
    # preference or persisting its surrounding dialogue boilerplate.
    hints = [] if cleaned or USE_JUDGMENT_RE.search(message) else [
        match.group("attribute")
        for match in no_preference_matches
    ]
    return cleaned, list(dict.fromkeys(hints))


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


def _has_known_product_term(message: str) -> bool:
    """Return whether text contains a deterministic shopping attribute term."""

    terms = (*MATERIALS, *COLORS, *STYLE_PHRASES, *FEATURE_PHRASES, *USE_CASE_PHRASES)
    return any(
        re.search(rf"(?i)(?<!\w){re.escape(term)}(?!\w)", message)
        for term in terms
    )


def _looks_like_product_description(phrase: str) -> bool:
    """Conservatively recognize a standalone catalog-like descriptive phrase."""

    words = TOKEN_RE.findall(phrase)
    if (
        not 2 <= len(words) <= 60
        or FILLER_RE.fullmatch(phrase)
        or CONVERSATIONAL_FOLLOWUP_RE.search(phrase)
    ):
        return False
    return bool(PRODUCT_ATTRIBUTE_RE.search(phrase) or _has_known_product_term(phrase))


def _extract_descriptive_reveals(message: str, result: SlotExtraction) -> None:
    """Preserve useful raw product wording without inventing normalized slots."""

    initial_match = INITIAL_PREFERENCE_RE.match(message)
    if initial_match:
        phrase = initial_match.group("phrase").strip()
        # An opening message can carry an explicit reveal after the category, e.g.
        # "I'm looking for Women Leggings. A key requirement is: polyester."
        # _extract_explicit_reveals already captures the value on its own, so adding
        # the framing sentence here would put its wording ("key", "requirement") into
        # the retrieval query. Those words are rare in the catalog and therefore score
        # highly under BM25, despite carrying no preference. This is the same guard
        # applied to non-opening messages below.
        if not EXPLICIT_REVEAL_RE.search(phrase) and not OVERRIDE_SCAFFOLD_RE.search(phrase):
            _add_raw(result, phrase)
        return
    if EXPLICIT_REVEAL_RE.search(message) or OVERRIDE_SCAFFOLD_RE.search(message):
        return
    phrase = message.strip().rstrip(".?!")
    if _looks_like_product_description(phrase):
        _add_raw(result, phrase)


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
        (
            rf"(?i)\b(?:under|below|less than|no more than|at most|up to|maximum|max)"
            rf"\s+{amount}",
            "max",
        ),
        (rf"(?i)\b(?:over|above|more than|minimum|min)\s+{amount}", "min"),
        (
            rf"(?i)\b(?:around|about|approximately|approx\.?|roughly|budget around)"
            rf"\s+{amount}",
            "approx",
        ),
    )
    for pattern, kind in patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        raw = match.group(0)
        if kind == "approx":
            context = message[
                max(0, match.start() - 24):min(len(message), match.end() + 24)
            ]
            if not re.search(
                r"(?:\$|\b(?:SGD|USD|dollars?|bucks?|budget|price|cost)\b)",
                context,
                re.IGNORECASE,
            ):
                continue
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


def _local_clause(message: str, start: int, end: int) -> str:
    """Return the sentence-like fragment containing one catalog match."""

    left_boundaries = [message.rfind(mark, 0, start) for mark in ".!?;"]
    left = max(left_boundaries) + 1
    right_candidates = [
        position
        for mark in ".!?;"
        if (position := message.find(mark, end)) >= 0
    ]
    right = min(right_candidates) if right_candidates else len(message)
    return message[left:right].strip()


def _brand_has_context(
    message: str,
    raw: str,
    start: int,
    end: int,
    categories: dict[str, str],
) -> bool:
    """Require a local shopping or explicit-brand context for catalog brands."""

    stripped = message.strip().rstrip(".,!?;:")
    if stripped.casefold() == raw.casefold():
        return True
    clause = _local_clause(message, start, end)
    escaped = re.escape(raw)
    if re.search(
        rf"(?i)(?<!\w)(?:brand|from|by|made\s+by)\s*:?[\s-]*{escaped}(?!\w)",
        clause,
    ):
        return True
    if re.search(
        rf"(?i)(?:\b(?:must|has\s+to|needs?\s+to)\s+be\s+{escaped}(?!\w)|"
        rf"(?<!\w){escaped}\s+(?:must\s+be\s+included|"
        rf"is\s+(?:(?:not|no\s+longer)\s+)?(?:required|essential|necessary)|"
        rf"isn't\s+(?:required|essential|necessary)))",
        clause,
    ):
        return True
    if SHOPPING_TARGET_RE.search(clause):
        return True
    return any(
        folded in categories and folded not in GENERIC_CATEGORIES
        for folded, _ in _ngrams(clause)
    )


def _category_has_context(message: str, start: int, end: int) -> bool:
    """Reject component nouns promoted from long descriptive feature text."""

    clause = _local_clause(message, start, end)
    if SHOPPING_TARGET_RE.search(clause):
        return True
    relative_start = clause.casefold().find(message[start:end].casefold())
    before_match = clause[:max(0, relative_start)]
    if COMPONENT_RELATION_RE.search(before_match):
        return False
    return not (
        len(TOKEN_RE.findall(clause)) > 8
        and PRODUCT_ATTRIBUTE_RE.search(clause)
    )


def _extract_catalog_terms(message: str, result: SlotExtraction) -> None:
    brands, categories = _catalog_vocabulary()
    occupied: list[tuple[int, int]] = []
    for folded, raw in _ngrams(message):
        start = message.casefold().find(folded)
        span = (start, start + len(raw))
        if start < 0 or any(start < end and span[1] > begin for begin, end in occupied):
            continue
        looks_named = " " in raw or any(character.isupper() for character in raw)
        if (
            folded in brands
            and looks_named
            and _brand_has_context(message, raw, start, span[1], categories)
        ):
            _add_value(result.slots["brand"], brands[folded])
            _add_raw(result, raw)
            occupied.append(span)
        if (
            folded in categories
            and folded not in GENERIC_CATEGORIES
            and not (
                result.slots["budget"]
                and re.search(
                    r"(?i)(?:\$|\b(?:sgd|usd|under|below|over|above|between|budget)\b|\d)",
                    raw,
                )
            )
            and _category_has_context(message, start, span[1])
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
    evidence_text, retrieval_hints = _retrieval_evidence_text(user_message)
    result.retrieval_hints.extend(retrieval_hints)
    if not evidence_text:
        return result
    _extract_explicit_reveals(evidence_text, result)
    _extract_descriptive_reveals(evidence_text, result)
    _extract_budget(evidence_text, result)
    _extract_sizes(evidence_text, result)
    _extract_terms(evidence_text, result, "material", MATERIALS)
    _extract_terms(evidence_text, result, "color", COLORS)
    _extract_terms(evidence_text, result, "style", STYLE_PHRASES)
    _extract_terms(evidence_text, result, "feature", FEATURE_PHRASES)
    _extract_terms(evidence_text, result, "use_case", USE_CASE_PHRASES)
    _extract_catalog_terms(evidence_text, result)
    return result
