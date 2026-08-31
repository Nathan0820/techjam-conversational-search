"""Order retrieved candidates against the current conversation state.

Retrieval decides which products compete; this decides the order they are shown in.
Each candidate is scored on how well it satisfies what the customer has said, with
the lexical retrieval score kept as the dominant term so a single noisy extraction
cannot outrank the evidence that found the product in the first place.

Products that violate a stated hard requirement are penalised rather than removed,
because extraction is imperfect and a wrongly-parsed constraint should cost a
product rank, not eliminate it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .features import ProductFeatures, extract_features, product_text, state_value, state_to_query


class CrossEncoder(Protocol):
    """Optional semantic scorer for (query, product text) pairs.

    Supplying one enables a second pass over the shortlist. The agent ships without
    one, so the feature score alone decides the order.
    """

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        """Return one score per pair, in the order the pairs were given."""


@dataclass(frozen=True)
class RankingWeights:
    """Relative influence of each ranking signal, and the cost of each violation."""

    # Retrieval remains the strongest signal. Structured state refines its
    # ordering without allowing a noisy extracted match to overwhelm BM25.
    constraint: float = 0.24
    semantic: float = 0.0
    keyword: float = 0.70
    category: float = 0.06
    quality: float = 0.001
    category_violation: float = 0.50
    price_violation: float = 0.50
    other_hard_violation: float = 0.30
    negative_preference: float = 0.50
    buying_penalty_multiplier: float = 1.10
    bm25_confidence_gap: float = 0.10
    feature_blend: float = 0.60
    cross_encoder_blend: float = 0.40


_PRICE_VIOLATIONS = {"min_price", "max_price"}
_CONFIRMED_VIOLATIONS = {"category", *_PRICE_VIOLATIONS}


def _minmax(values: list[float | None], *, log_scale: bool = False) -> list[float]:
    """Scale values into [0, 1] so signals on different scales can be summed.

    Missing and non-finite entries become 0.0, and a set of identical values becomes
    0.5 throughout rather than collapsing to zero. `log_scale` compresses long-tailed
    quantities such as review counts, where the difference between 10 and 100 reviews
    matters more than between 10,000 and 10,090.
    """

    present = [value for value in values if value is not None and math.isfinite(value)]
    if not present:
        return [0.0] * len(values)
    if log_scale:
        present = [math.log1p(max(0.0, value)) for value in present]
    low, high = min(present), max(present)
    result = []
    for value in values:
        if value is None or not math.isfinite(value):
            result.append(0.0)
            continue
        adjusted = math.log1p(max(0.0, value)) if log_scale else value
        result.append(0.5 if high == low else (adjusted - low) / (high - low))
    return result


def _mean_known(values: Sequence[float | None], default: float = 0.5) -> float:
    """Average the values that are known, ignoring absent ones.

    A constraint the customer never mentioned should neither help nor hurt a product,
    so it is skipped instead of counted as zero.
    """

    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else default


def _constraint_score(features: ProductFeatures) -> float:
    """Combine every per-attribute match into one satisfaction score in [0, 1]."""

    return _mean_known([
        features.brand_match,
        features.color_match,
        features.size_match,
        features.material_match,
        features.style_match,
        features.feature_match,
        features.use_case_match,
        features.price_match,
        features.positive_preference_match,
    ])


def _confident_bm25_leader(
    features: Sequence[ProductFeatures],
    minimum_relative_gap: float,
) -> int | None:
    """Return the clear BM25 leader, or None when retrieval is ambiguous."""

    present = [
        (index, item.bm25_score)
        for index, item in enumerate(features)
        if item.bm25_score is not None and math.isfinite(item.bm25_score)
    ]
    if len(present) < 2:
        return None
    present.sort(key=lambda pair: (-pair[1], pair[0]))
    leader_index, leader_score = present[0]
    runner_up_score = present[1][1]
    relative_gap = (leader_score - runner_up_score) / max(abs(leader_score), 1e-12)
    return leader_index if relative_gap >= minimum_relative_gap else None


class Reranker:
    """Scores and orders candidates for one turn.

    Holds no per-session state; everything it needs arrives with each call.
    """

    def __init__(
        self,
        weights: RankingWeights | None = None,
        cross_encoder: CrossEncoder | None = None,
        cross_encoder_candidates: int = 30,
    ) -> None:
        """Configure the signal weights and the optional semantic second pass.

        `cross_encoder_candidates` sets how deep the shortlist goes before that
        second pass, and has no effect without a cross-encoder.
        """

        self.weights = weights or RankingWeights()
        self.cross_encoder = cross_encoder
        self.cross_encoder_candidates = cross_encoder_candidates

    def rerank(self, cands: Sequence[Mapping[str, Any]], state: object, top_k: int = 10) -> list[dict[str, Any]]:
        """Return the best `top_k` candidates, best first.

        Each candidate is scored as a weighted blend of lexical retrieval, constraint
        satisfaction, semantic similarity, category match and review quality, minus
        penalties for violating anything the customer stated as a requirement. Buying
        sessions weight those penalties more heavily than browsing ones.

        A candidate that leads on retrieval score by a clear margin is held at the top
        unless it confirmedly violates a hard constraint, so an unambiguous lexical
        match is not displaced by a marginal feature difference.

        Returns dicts carrying the product, its final score and its extracted features;
        [] for an empty candidate list or a non-positive `top_k`.
        """

        if top_k <= 0 or not cands:
            return []
        features = [extract_features(product, state) for product in cands]
        bm25 = _minmax([item.bm25_score for item in features])
        dense = _minmax([item.dense_score for item in features])
        ratings = [min(1.0, max(0.0, (item.rating or 0.0) / 5.0)) for item in features]
        reviews = _minmax([item.review_count for item in features], log_scale=True)
        protected_index = _confident_bm25_leader(
            features,
            self.weights.bm25_confidence_gap,
        )
        scenario = str(
            state_value(state, "intent", None)
            or state_value(state, "scenario_type", "")
        ).lower()

        scored: list[dict[str, Any]] = []
        for index, (product, item) in enumerate(zip(cands, features)):
            category = item.category_match if item.category_match is not None else 0.5
            quality = 0.7 * ratings[index] + 0.3 * reviews[index]
            constraint = _constraint_score(item)
            base = (
                self.weights.constraint * constraint
                + self.weights.semantic * dense[index]
                + self.weights.keyword * bm25[index]
                + self.weights.category * category
                + self.weights.quality * quality
            )

            penalty = 0.0
            penalty += self.weights.price_violation if "max_price" in item.hard_violations else 0.0
            penalty += self.weights.price_violation if "min_price" in item.hard_violations else 0.0
            penalty += self.weights.category_violation if "category" in item.hard_violations else 0.0
            penalty += self.weights.negative_preference * (item.negative_preference_match or 0.0)
            for violation in item.hard_violations:
                if violation not in {"min_price", "max_price", "category"}:
                    penalty += self.weights.other_hard_violation
            if scenario == "buying":
                penalty *= self.weights.buying_penalty_multiplier

            feature_score = base - penalty
            scored.append({
                "product": product,
                "base_score": base,
                "feature_score": feature_score,
                "final_score": feature_score,
                "_features": item,
                "_penalty": penalty,
                "_bm25_protected": (
                    index == protected_index
                    and not (_CONFIRMED_VIOLATIONS & set(item.hard_violations))
                ),
                "_input_index": index,
            })

        scored.sort(key=lambda item: (
            -int(item["_bm25_protected"]),
            -item["feature_score"],
            item["_input_index"],
        ))
        shortlist_size = max(top_k, self.cross_encoder_candidates) if self.cross_encoder is not None else top_k
        shortlist = scored[:shortlist_size]
        if self.cross_encoder is not None and shortlist:
            query = state_to_query(state)
            raw_scores = list(self.cross_encoder.predict([(query, product_text(item["product"])) for item in shortlist]))
            if len(raw_scores) != len(shortlist):
                raise ValueError("cross-encoder returned a different number of scores than candidate pairs")
            normalized = _minmax([float(score) for score in raw_scores])
            feature_norm = _minmax([item["base_score"] for item in shortlist])
            for item, feature_value, cross_value in zip(shortlist, feature_norm, normalized):
                item["cross_encoder_score"] = cross_value
                item["final_score"] = (
                    self.weights.feature_blend * feature_value
                    + self.weights.cross_encoder_blend * cross_value
                    - item["_penalty"]
                )
            shortlist.sort(key=lambda item: (
                -int(item["_bm25_protected"]),
                -item["final_score"],
                item["_input_index"],
            ))

        for item in shortlist:
            item["features"] = asdict(item.pop("_features"))
            item.pop("base_score", None)
            item.pop("_input_index", None)
            item.pop("_penalty", None)
            item.pop("_bm25_protected", None)
        return shortlist[:top_k]


def rerank(cands: Sequence[Mapping[str, Any]], state: object, top_k: int = 10) -> list[dict[str, Any]]:
    """Rank with default weights and no cross-encoder, for callers holding no Reranker."""

    return Reranker().rerank(cands, state, top_k)


def to_evaluator_recommendations(scored: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce ranked results to the {parent_asin, score} shape the evaluator scores.

    Entries without a resolvable product id are dropped rather than emitted blank,
    since the evaluator discards ids outside the catalog anyway.
    """

    recommendations = []
    for item in scored:
        product = item.get("product", {})
        asin = product.get("parent_asin", product.get("asin")) if isinstance(product, Mapping) else None
        if asin:
            recommendations.append({"parent_asin": str(asin), "score": float(item["final_score"])})
    return recommendations
