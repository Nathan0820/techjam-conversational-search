from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frontend.server import App, metrics_payload


class AlwaysHitAgent:
    def __init__(self) -> None:
        self.sessions = {"live": object()}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = object()

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        return {
            "message": "ok",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": "A"}],
        }


class FrontendEvaluationTest(unittest.TestCase):
    def test_metrics_payload_uses_evaluator_score(self) -> None:
        payload = metrics_payload({
            "sample_count": 2,
            "hit_rate_at_10": 0.5,
            "mrr": 0.25,
            "mttc": 6.5,
            "recommended_technical_score": 0.4,
        })

        self.assertTrue(payload["available"])
        self.assertEqual(payload["technical_score"], 0.4)

    def test_evaluation_publishes_results_and_preserves_live_sessions(self) -> None:
        app = App.__new__(App)
        app.agent = AlwaysHitAgent()
        app.catalog_ids = {"A"}
        app.categories_by_asin = {"A": ["Shoes"]}
        app.catalog_by_asin = {"A": {"parent_asin": "A"}}
        app.public_samples = [{
            "sample_id": "sample",
            "scenario_type": "buying",
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
            "intent_card": {"hard_constraints": [], "soft_preferences": []},
            "behavior": {},
        }]

        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "results.json"
            with patch("frontend.server.RESULTS_PATH", results_path):
                payload = app.evaluate_public_set()

            stored = json.loads(results_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["technical_score"], 1.0)
        self.assertEqual(stored["recommended_technical_score"], 1.0)
        self.assertEqual(set(app.agent.sessions), {"live"})


if __name__ == "__main__":
    unittest.main()
