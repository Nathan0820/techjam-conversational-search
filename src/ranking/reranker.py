from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .features import ProductFeatures, extract_features, product_text, state_value, state_to_query


class CrossEncoder(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


@dataclass(frozen=True)
class RankingWeights:
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
_PRICE_CONSTRAINT_NAMES = {"budget", "price", "max_price", "price_max", "min_price"}


def _minmax(values: list[float | None], *, log_scale: bool = False) -> list[float]:
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
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else default


def _constraint_score(features: ProductFeatures) -> float:
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


def _has_hard_budget(state: object) -> bool:
    raw = state_value(state, "hard_constraints", {})
    names = set(raw) if isinstance(raw, Mapping) else set(raw or ())
    return bool(names & _PRICE_CONSTRAINT_NAMES)


def _price_compliance_tier(features: ProductFeatures, hard_budget: bool) -> int:
    """Order confirmed budget matches, unknown prices, then violations."""

    if not hard_budget:
        return 0
    if _PRICE_VIOLATIONS & set(features.hard_violations):
        return 2
    return 0 if features.price_match == 1.0 else 1


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
    def __init__(
        self,
        weights: RankingWeights | None = None,
        cross_encoder: CrossEncoder | None = None,
        cross_encoder_candidates: int = 30,
    ) -> None:
        self.weights = weights or RankingWeights()
        self.cross_encoder = cross_encoder
        self.cross_encoder_candidates = cross_encoder_candidates

    def rerank(self, cands: Sequence[Mapping[str, Any]], state: object, top_k: int = 10) -> list[dict[str, Any]]:
        if top_k <= 0 or not cands:
            return []
        features = [extract_features(product, state) for product in cands]
        bm25 = _minmax([item.bm25_score for item in features])
        dense = _minmax([item.dense_score for item in features])
        ratings = [min(1.0, max(0.0, (item.rating or 0.0) / 5.0)) for item in features]
        reviews = _minmax([item.review_count for item in features], log_scale=True)
        hard_budget = _has_hard_budget(state)
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
            price_tier = _price_compliance_tier(item, hard_budget)
            scored.append({
                "product": product,
                "base_score": base,
                "feature_score": feature_score,
                "final_score": feature_score,
                "_features": item,
                "_penalty": penalty,
                "_price_tier": price_tier,
                "_bm25_protected": (
                    index == protected_index
                    and not (_CONFIRMED_VIOLATIONS & set(item.hard_violations))
                ),
                "_input_index": index,
            })

        scored.sort(key=lambda item: (
            item["_price_tier"],
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
                item["_price_tier"],
                -int(item["_bm25_protected"]),
                -item["final_score"],
                item["_input_index"],
            ))

        for item in shortlist:
            item["features"] = asdict(item.pop("_features"))
            item.pop("base_score", None)
            item.pop("_input_index", None)
            item.pop("_penalty", None)
            item.pop("_price_tier", None)
            item.pop("_bm25_protected", None)
        return shortlist[:top_k]


def rerank(cands: Sequence[Mapping[str, Any]], state: object, top_k: int = 10) -> list[dict[str, Any]]:
    return Reranker().rerank(cands, state, top_k)


def to_evaluator_recommendations(scored: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    recommendations = []
    for item in scored:
        product = item.get("product", {})
        asin = product.get("parent_asin", product.get("asin")) if isinstance(product, Mapping) else None
        if asin:
            recommendations.append({"parent_asin": str(asin), "score": float(item["final_score"])})
    return recommendations
