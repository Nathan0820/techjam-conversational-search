from __future__ import annotations

import unittest
from dataclasses import dataclass

from dialogue.accumulator import accumulate_information
from dialogue.slot_extractor import extract_slots
from dialogue.state import SessionState
from dialogue.types import BudgetConstraint
from src.ranking.features import extract_features, state_to_query
from src.ranking.diagnostics import ranking_failure_record
from src.ranking.reranker import RankingWeights, Reranker, rerank, to_evaluator_recommendations


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

    def test_confident_bm25_leader_is_protected_from_soft_evidence(self) -> None:
        """Keep a clearly separated retrieval winner ahead of soft evidence."""

        weights = RankingWeights(constraint=1.0, keyword=0.0, category=0.0, quality=0.0)
        state = {"soft_preferences": {"color": "black"}}
        candidates = [
            product("leader", "White shirt", bm25_score=10),
            product("soft-match", "Black shirt", bm25_score=5),
        ]

        ranked = Reranker(weights=weights).rerank(candidates, state)

        self.assertEqual(ranked[0]["product"]["parent_asin"], "leader")

    def test_confident_bm25_leader_cannot_override_budget_violation(self) -> None:
        """Never protect retrieval confidence against a numeric violation."""

        state = {"hard_constraints": {"max_price": 100}}
        candidates = [
            product(
                "leader",
                "Premium boots",
                price=150,
                bm25_score=10,
            ),
            product(
                "compliant",
                "Hiking boots",
                price=80,
                bm25_score=5,
            ),
        ]

        ranked = rerank(candidates, state)

        self.assertEqual(ranked[0]["product"]["parent_asin"], "compliant")

    def test_confident_bm25_leader_cannot_override_category_violation(self) -> None:
        """Treat an explicit category mismatch as confirmed evidence."""

        state = {"hard_constraints": {"category": "boots"}}
        candidates = [
            product(
                "leader",
                "Formal earrings",
                categories=["Earrings"],
                bm25_score=10,
            ),
            product(
                "compliant",
                "Hiking boots",
                categories=["Boots"],
                bm25_score=5,
            ),
        ]

        ranked = Reranker(
            weights=RankingWeights(category_violation=1.0),
        ).rerank(candidates, state)

        self.assertEqual(ranked[0]["product"]["parent_asin"], "compliant")

    def test_budget_compliance_tiers_precede_numeric_score(self) -> None:
        """Order known budget matches, unknown prices, then violations."""

        weights = RankingWeights(
            constraint=0.0,
            keyword=1.0,
            category=0.0,
            quality=0.0,
            price_violation=0.0,
        )
        state = {"hard_constraints": {"max_price": 100}}
        candidates = [
            product("violating", "Watch", price=150, bm25_score=10),
            product("unknown", "Watch", price=None, bm25_score=9),
            product("compliant", "Watch", price=80, bm25_score=1),
        ]

        ranked = Reranker(weights=weights).rerank(candidates, state, top_k=3)

        self.assertEqual(
            [item["product"]["parent_asin"] for item in ranked],
            ["compliant", "unknown", "violating"],
        )

    def test_close_bm25_scores_allow_feature_reranking(self) -> None:
        """Do not protect a retrieval leader when its score margin is small."""

        weights = RankingWeights(constraint=1.0, keyword=0.0, category=0.0, quality=0.0)
        state = {"soft_preferences": {"color": "black"}}
        candidates = [
            product("leader", "White shirt", bm25_score=10),
            product("soft-match", "Black shirt", bm25_score=9.5),
        ]

        ranked = Reranker(weights=weights).rerank(candidates, state)

        self.assertEqual(ranked[0]["product"]["parent_asin"], "soft-match")

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

    def test_accumulated_session_slots_affect_ranking_before_classification(self) -> None:
        state = SessionState(session_id="session")
        accumulate_information(state, extract_slots("black Nike boots"))
        candidates = [
            product("wrong", "Brown Adidas sandals", store="Adidas", categories=["Sandals"]),
            product("target", "Black Nike boots", store="Nike", categories=["Boots"]),
        ]

        ranked = rerank(candidates, state)

        self.assertEqual(ranked[0]["product"]["parent_asin"], "target")
        self.assertEqual(ranked[0]["features"]["brand_match"], 1.0)
        self.assertEqual(ranked[0]["features"]["color_match"], 1.0)
        self.assertEqual(ranked[1]["features"]["hard_violations"], [])

    def test_session_state_hard_constraints_use_values_from_slots(self) -> None:
        state = SessionState(session_id="session", intent="buying")
        state.set_constraint("category", ["boots"], strength="hard")
        state.set_constraint("budget", [
            BudgetConstraint(minimum=50, maximum=100, currency="SGD")
        ], strength="hard")
        candidates = [
            product("wrong-category", "Running shoes", categories=["Shoes"], price=80),
            product("over-budget", "Premium boots", categories=["Boots"], price=150),
            product("target", "Everyday boots", categories=["Boots"], price=90),
        ]

        ranked = rerank(candidates, state)

        self.assertEqual(ranked[0]["product"]["parent_asin"], "target")
        by_asin = {item["product"]["parent_asin"]: item for item in ranked}
        self.assertIn("category", by_asin["wrong-category"]["features"]["hard_violations"])
        self.assertIn("max_price", by_asin["over-budget"]["features"]["hard_violations"])

    def test_ranking_failure_record(self) -> None:
        retrieved = [product("target", "Boot"), product("other", "Boot")]
        ranked = rerank(list(reversed(retrieved)), {})
        record = ranking_failure_record("session", "target", retrieved, ranked)
        self.assertTrue(record["target_retrieved"])
        self.assertEqual(record["target_initial_rank"], 1)
        self.assertEqual(record["target_final_rank"], 2)
        self.assertEqual(record["failure_reason"], "target_ranked_below_competitor")

    def test_single_letter_size_requires_a_token_boundary(self) -> None:
        state = {"hard_constraints": {"size": "M"}}
        candidates = [
            product("false-positive", "Premium cotton shirt"),
            product("target", "Cotton shirt size M"),
        ]

        ranked = rerank(candidates, state)

        self.assertEqual(ranked[0]["product"]["parent_asin"], "target")
        self.assertIn("size", ranked[1]["features"]["hard_violations"])

    def test_partial_category_overlap_is_a_hard_violation(self) -> None:
        state = {"hard_constraints": {"category": "Women Leggings"}}
        candidates = [
            product("wrong", "Gold earrings", categories=["Women", "Earrings"]),
            product("target", "Training leggings", categories=["Women", "Leggings"]),
        ]

        ranked = rerank(candidates, state)

        self.assertEqual(ranked[0]["product"]["parent_asin"], "target")
        self.assertIn("category", ranked[1]["features"]["hard_violations"])

    def test_style_and_feature_slots_affect_ranking(self) -> None:
        state = {
            "slots": {"style": ["casual"], "feature": ["waterproof"]},
            "soft_preferences": {"style", "feature"},
        }
        candidates = [
            product("wrong", "Formal wool coat"),
            product("target", "Casual waterproof coat"),
        ]

        ranked = rerank(candidates, state)

        self.assertEqual(ranked[0]["product"]["parent_asin"], "target")
        self.assertEqual(ranked[0]["features"]["style_match"], 1.0)
        self.assertEqual(ranked[0]["features"]["feature_match"], 1.0)

    def test_minimum_price_and_unknown_price_are_distinguished(self) -> None:
        state = {
            "slots": {"budget": [BudgetConstraint(minimum=100)]},
            "hard_constraints": {"budget"},
        }
        candidates = [
            product("below", "Budget watch", price=50),
            product("unknown", "Unpriced watch", price=None),
            product("target", "Premium watch", price=120),
        ]

        ranked = rerank(candidates, state, top_k=3)
        by_asin = {item["product"]["parent_asin"]: item for item in ranked}

        self.assertEqual(ranked[0]["product"]["parent_asin"], "target")
        self.assertIn("min_price", by_asin["below"]["features"]["hard_violations"])
        self.assertEqual(by_asin["unknown"]["features"]["price_match"], 0.25)

    def test_semantic_blend_cannot_resurrect_hard_violation(self) -> None:
        class PreferWrong:
            def predict(self, pairs):
                return [1.0 if "earrings" in text else 0.0 for _, text in pairs]

        state = {"hard_constraints": {"category": "Leggings"}}
        candidates = [
            product("wrong", "Sparkly earrings", categories=["Earrings"]),
            product("target", "Training leggings", categories=["Leggings"]),
        ]

        ranked = Reranker(cross_encoder=PreferWrong()).rerank(candidates, state)

        self.assertEqual(ranked[0]["product"]["parent_asin"], "target")

    def test_semantic_query_omits_empty_slot_names(self) -> None:
        state = SessionState(session_id="session")
        state.add_slot_values("color", ["black"])

        query = state_to_query(state)

        self.assertIn("black", query)
        self.assertNotIn("material", query)
        self.assertNotIn("budget", query)

    def test_top_k_is_not_capped_without_cross_encoder(self) -> None:
        candidates = [product(str(index), f"Product {index}") for index in range(40)]

        self.assertEqual(len(rerank(candidates, {}, top_k=40)), 40)


if __name__ == "__main__":
    unittest.main()
