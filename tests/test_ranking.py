from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.ranking.features import extract_features, state_to_query
from src.ranking.diagnostics import ranking_failure_record
from src.ranking.reranker import Reranker, rerank, to_evaluator_recommendations


def product(asin: str, title: str, **fields: object) -> dict:
    return {
        "parent_asin": asin,
        "title": title,
        "categories": fields.pop("categories", []),
        "features": fields.pop("features", []),
        "description": fields.pop("description", []),
        "details": fields.pop("details", {}),
        "store": fields.pop("store", ""),
        "price": fields.pop("price", None),
        "average_rating": fields.pop("average_rating", None),
        "rating_number": fields.pop("rating_number", None),
        **fields,
    }


class FakeCrossEncoder:
    def predict(self, pairs):
        return [1.0 if "waterproof" in text else 0.0 for _, text in pairs]


class RankingTest(unittest.TestCase):
    def test_exact_brand_and_category_match(self) -> None:
        state = {"hard_constraints": {"brand": "Nike", "category": "boots"}}
        candidates = [
            product("wrong", "Adidas running shoe", store="Adidas", categories=["Shoes"]),
            product("target", "Nike hiking boots", store="Nike", categories=["Boots"]),
        ]
        self.assertEqual(rerank(candidates, state)[0]["product"]["parent_asin"], "target")

    def test_price_limit_enforcement(self) -> None:
        state = {"hard_constraints": {"category": "boots", "max_price": 150}}
        candidates = [product("over", "Premium boots", categories=["Boots"], price=151, dense_score=1), product("under", "Basic boots", categories=["Boots"], price=100)]
        ranked = rerank(candidates, state)
        self.assertEqual(ranked[0]["product"]["parent_asin"], "under")
        self.assertIn("max_price", ranked[1]["features"]["hard_violations"])

    def test_negative_preference_penalty(self) -> None:
        state = {"soft_preferences": {"comfortable": True}, "negative_preferences": ["leather"]}
        candidates = [product("bad", "Comfortable leather boot"), product("good", "Comfortable canvas boot")]
        self.assertEqual(rerank(candidates, state)[0]["product"]["parent_asin"], "good")

    def test_cross_encoder_semantic_similarity(self) -> None:
        candidates = [product("plain", "Hiking boot"), product("semantic", "Waterproof hiking boot")]
        ranked = Reranker(cross_encoder=FakeCrossEncoder()).rerank(candidates, {"soft_preferences": ["rain protection"]})
        self.assertEqual(ranked[0]["product"]["parent_asin"], "semantic")

    def test_overridden_preferences_do_not_leak(self) -> None:
        state = {"hard_constraints": {"category": "hiking boots"}, "soft_preferences": {}}
        candidates = [product("old", "Running shoes", categories=["Running Shoes"]), product("new", "Hiking boots", categories=["Hiking Boots"])]
        ranked = rerank(candidates, state)
        self.assertEqual(ranked[0]["product"]["parent_asin"], "new")
        self.assertNotIn("running", str(ranked[0]["features"]))

    def test_buying_penalizes_hard_violation(self) -> None:
        candidates = [product("wrong", "Red shoes", price=200, dense_score=1), product("right", "Red shoes", price=90)]
        state = {"scenario_type": "buying", "hard_constraints": {"max_price": 100}}
        self.assertEqual(rerank(candidates, state)[0]["product"]["parent_asin"], "right")

    def test_member_b_session_state_schema(self) -> None:
        state = {
            "intent": "buying",
            "slots": {"category": "boots", "color": "black", "brand": "Nike"},
            "hard_constraints": {"category": "boots", "budget": 150},
            "soft_preferences": {"waterproof": True},
            "asked_attributes": {"material", "color"},
            "turn": 3,
            "override_detected": False,
            "specificity": "high",
        }
        query = state_to_query(state)
        self.assertIn("boots", query)
        self.assertIn("waterproof", query)
        self.assertNotIn("asked_attributes", query)
        candidates = [product("over", "Black Nike boots", price=200), product("target", "Black Nike waterproof boots", price=120)]
        self.assertEqual(rerank(candidates, state)[0]["product"]["parent_asin"], "target")

    def test_missing_constraints(self) -> None:
        candidates = [product("low", "A", average_rating=2), product("high", "B", average_rating=5)]
        self.assertEqual(rerank(candidates, {})[0]["product"]["parent_asin"], "high")

    def test_missing_product_metadata(self) -> None:
        ranked = rerank([{"parent_asin": "A"}], {"hard_constraints": {"brand": "Nike"}})
        self.assertEqual(ranked[0]["product"]["parent_asin"], "A")

    def test_stable_deterministic_order_and_evaluator_shape(self) -> None:
        candidates = [product("A", "Same"), product("B", "Same")]
        first = rerank(candidates, {})
        second = rerank(candidates, {})
        self.assertEqual([x["product"]["parent_asin"] for x in first], ["A", "B"])
        self.assertEqual(first, second)
        self.assertEqual(set(to_evaluator_recommendations(first)[0]), {"parent_asin", "score"})

    def test_state_object_is_supported(self) -> None:
        @dataclass
        class State:
            hard_constraints: dict
            soft_preferences: dict

        features = extract_features(product("A", "Black Nike boot", store="Nike"), State({"brand": "Nike"}, {}))
        self.assertEqual(features.brand_match, 1.0)

    def test_ranking_failure_record(self) -> None:
        retrieved = [product("target", "Boot"), product("other", "Boot")]
        ranked = rerank(list(reversed(retrieved)), {})
        record = ranking_failure_record("session", "target", retrieved, ranked)
        self.assertTrue(record["target_retrieved"])
        self.assertEqual(record["target_initial_rank"], 1)
        self.assertEqual(record["target_final_rank"], 2)
        self.assertEqual(record["failure_reason"], "target_ranked_below_competitor")


if __name__ == "__main__":
    unittest.main()
