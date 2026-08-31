from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from frontend.server import STATIC_DIR, App, metrics_payload


class AlwaysHitAgent:
    def __init__(self) -> None:
        self.sessions = {"live": self._state()}

    @staticmethod
    def _state() -> SimpleNamespace:
        return SimpleNamespace(
            intent="buying",
            turn=1,
            slots={},
            hard_constraints=set(),
            soft_preferences=set(),
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = self._state()

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
    def test_evaluation_runs_on_page_load_but_not_new_session(self) -> None:
        script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        button_handler = script.split(
            'newSessionButton.addEventListener("click"',
            maxsplit=1,
        )[1]
        button_handler, page_bootstrap = button_handler.split(
            'newSession({ reevaluate: true })',
            maxsplit=1,
        )

        self.assertIn('newSession({ reevaluate: false })', button_handler)
        self.assertNotIn('newSession({ reevaluate: true })', button_handler)
        self.assertIn('.catch(error =>', page_bootstrap)

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

    def test_chat_adds_catalog_highlights_to_recommendations(self) -> None:
        app = App.__new__(App)
        app.agent = AlwaysHitAgent()
        app.catalog_by_asin = {
            "A": {
                "parent_asin": "A",
                "title": "Trail jacket",
                "store": "Findly Test Store",
                "categories": ["Clothing", "Jackets"],
                "price": 79.0,
                "features": ["Waterproof", "Lightweight", "Packable"],
                "details": {"Department": "Unisex"},
            },
        }

        payload = app.chat({"session_id": "live", "message": "jacket", "turn": 1})
        recommendation = payload["recommendations"][0]

        self.assertEqual(recommendation["title"], "Trail jacket")
        self.assertEqual(
            recommendation["highlights"],
            ["Waterproof", "Lightweight", "Department: Unisex"],
        )


if __name__ == "__main__":
    unittest.main()
