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
    feature_blend: float = 0.60
    cross_encoder_blend: float = 0.40


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
        features.use_case_match,
        features.price_match,
        features.positive_preference_match,
    ])


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
            penalty += 8.0 if "max_price" in item.hard_violations else 0.0
            penalty += 8.0 if "category" in item.hard_violations else 0.0
            penalty += 5.0 * (item.negative_preference_match or 0.0)
            for violation in item.hard_violations:
                if violation not in {"max_price", "category"}:
                    penalty += 4.0
            if scenario == "buying":
                penalty *= 1.15

            feature_score = base - penalty
            scored.append({
                "product": product,
                "feature_score": feature_score,
                "features": asdict(item),
                "final_score": feature_score,
                "_input_index": index,
            })

        scored.sort(key=lambda item: (-item["feature_score"], item["_input_index"]))
        shortlist = scored[: self.cross_encoder_candidates]
        if self.cross_encoder is not None and shortlist:
            query = state_to_query(state)
            raw_scores = list(self.cross_encoder.predict([(query, product_text(item["product"])) for item in shortlist]))
            if len(raw_scores) != len(shortlist):
                raise ValueError("cross-encoder returned a different number of scores than candidate pairs")
            normalized = _minmax([float(score) for score in raw_scores])
            feature_norm = _minmax([item["feature_score"] for item in shortlist])
            for item, feature_value, cross_value in zip(shortlist, feature_norm, normalized):
                item["cross_encoder_score"] = cross_value
                item["final_score"] = (
                    self.weights.feature_blend * feature_value
                    + self.weights.cross_encoder_blend * cross_value
                )
            shortlist.sort(key=lambda item: (-item["final_score"], item["_input_index"]))

        for item in shortlist:
            item.pop("_input_index", None)
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
