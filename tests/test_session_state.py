from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from dialogue.state import SessionState
from starter.agent import Agent


def _catalog_file(directory: str) -> Path:
    path = Path(directory) / "catalog.jsonl"
    product = {
        "parent_asin": "A",
        "title": "Blue shirt",
        "categories": ["Clothing", "Shirts"],
        "features": ["cotton"],
        "details": {},
        "store": "Example",
        "description": [],
    }
    path.write_text(json.dumps(product) + "\n", encoding="utf-8")
    return path


class SessionStateTest(unittest.TestCase):
    def test_defaults_are_clean_and_contain_no_evaluator_only_fields(self) -> None:
        state = SessionState(session_id="one")

        self.assertIsNone(state.intent)
        self.assertEqual(state.user_profile, {})
        self.assertEqual(state.slots, {})
        self.assertEqual(state.hard_constraints, {})
        self.assertEqual(state.soft_preferences, {})
        self.assertEqual(state.asked_attributes, set())
        self.assertEqual(state.turn, 0)
        self.assertFalse(state.override_detected)
        self.assertEqual(state.message_history, [])

        field_names = {item.name for item in fields(SessionState)}
        evaluator_only = {
            "scenario_type", "difficulty_bucket", "category_bucket",
            "ground_truth", "intent_card", "behavior", "target_parent_asin",
            "parent_asin", "disclosed_attributes",
        }
        self.assertTrue(field_names.isdisjoint(evaluator_only))

    def test_mutable_fields_are_independent_between_states(self) -> None:
        first = SessionState(session_id="one")
        second = SessionState(session_id="two")

        first.slots["color"] = "blue"
        first.hard_constraints["material"] = "cotton"
        first.soft_preferences["style"] = "casual"
        first.asked_attributes.add("size")
        first.message_history.append("hello")

        self.assertEqual(second.slots, {})
        self.assertEqual(second.hard_constraints, {})
        self.assertEqual(second.soft_preferences, {})
        self.assertEqual(second.asked_attributes, set())
        self.assertEqual(second.message_history, [])


class AgentSessionLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.agent = Agent(_catalog_file(self.temporary_directory.name))

    def tearDown(self) -> None:
        self.agent.connection.close()
        self.temporary_directory.cleanup()

    def test_reset_creates_clean_state_and_safely_copies_profile(self) -> None:
        profile = {"summary": "Likes comfort", "preference_tags": ["comfort"]}
        self.agent.reset("session", profile)
        profile["summary"] = "changed"
        profile["preference_tags"].append("changed")

        state = self.agent.sessions["session"]
        self.assertEqual(state.session_id, "session")
        self.assertEqual(
            state.user_profile,
            {"summary": "Likes comfort", "preference_tags": ["comfort"]},
        )
        self.assertEqual(state.asked_attributes, set())
        self.assertEqual(state.message_history, [])

    def test_respond_appends_messages_in_order_and_updates_turn(self) -> None:
        self.agent.reset("session", {})

        first_response = self.agent.respond("session", "first message", 1, 10)
        second_response = self.agent.respond("session", "second message", 2, 10)

        state = self.agent.sessions["session"]
        self.assertEqual(state.message_history, ["first message", "second message"])
        self.assertEqual(state.turn, 2)
        self.assertIn("recommendations", first_response)
        self.assertIn("recommendations", second_response)

    def test_resetting_same_session_clears_previous_state(self) -> None:
        self.agent.reset("session", {"summary": "old"})
        old_state = self.agent.sessions["session"]
        old_state.slots["color"] = "blue"
        old_state.asked_attributes.add("size")
        self.agent.respond("session", "old message", 4, 10)

        self.agent.reset("session", {"summary": "new"})

        new_state = self.agent.sessions["session"]
        self.assertIsNot(new_state, old_state)
        self.assertEqual(new_state.user_profile, {"summary": "new"})
        self.assertEqual(new_state.slots, {})
        self.assertEqual(new_state.asked_attributes, set())
        self.assertEqual(new_state.message_history, [])
        self.assertEqual(new_state.turn, 0)

    def test_sessions_do_not_share_mutable_state(self) -> None:
        self.agent.reset("one", {})
        self.agent.reset("two", {})
        self.agent.sessions["one"].asked_attributes.add("color")

        self.assertEqual(self.agent.sessions["two"].asked_attributes, set())

    def test_respond_still_requires_reset(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            self.agent.respond("missing", "hello", 1, 10)


if __name__ == "__main__":
    unittest.main()
