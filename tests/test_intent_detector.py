"""Tests for deterministic intent detection and Agent integration."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dialogue.intent_detector import detect_intent
from dialogue.state import SessionState
from starter.agent import Agent


def _catalog_file(directory: str) -> Path:
    """Create the minimal catalog fixture used by Agent integration tests."""

    path = Path(directory) / "catalog.jsonl"
    product = {
        "parent_asin": "A",
        "title": "Black cotton running shoe",
        "categories": ["Clothing", "Shoes"],
        "features": ["waterproof"],
        "details": {},
        "store": "Example",
        "description": [],
    }
    path.write_text(json.dumps(product) + "\n", encoding="utf-8")
    return path


class IntentDetectorTest(unittest.TestCase):
    """Verify current-message scoring and conversation-context behavior."""

    def test_explicit_buying_signals(self) -> None:
        """Recognize direct purchase and recommendation requests as buying."""

        messages = (
            "I need a black cotton shirt under $50",
            "I'm looking to buy running shoes",
            "Help me find a backpack for university",
            "Which backpack should I get for university?",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(detect_intent(message, SessionState("session")), "buying")

    def test_explicit_browsing_signals(self) -> None:
        """Recognize direct exploration language as browsing."""

        messages = (
            "I'm just browsing",
            "What kinds of jackets are there?",
            "I'm interested in leather jackets",
            "Show me some ideas for summer outfits",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(detect_intent(message, SessionState("session")), "browsing")

    def test_explicit_want_to_browse_overrides_generic_want_signal(self) -> None:
        """Treat browse/look-around objects as browsing, even after buying."""

        for message in (
            "I want to browse",
            "I want to look around",
            "just want to browse",
            "just looking around",
        ):
            with self.subTest(message=message):
                state = SessionState("session", intent="buying")
                self.assertEqual(detect_intent(message, state), "browsing")

        self.assertEqual(
            detect_intent("I want black shoes", SessionState("session")),
            "buying",
        )

    def test_negated_buying_is_browsing(self) -> None:
        """Treat explicit rejection of purchase intent as browsing."""

        messages = (
            "I'm not looking to buy yet, just browsing",
            "I don't need anything specific yet",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(detect_intent(message, SessionState("session")), "browsing")

        prior_buying = SessionState("session", intent="buying")
        for message in ("I don't want to buy yet", "I do not need to buy yet"):
            with self.subTest(message=message, prior_intent="buying"):
                self.assertEqual(detect_intent(message, prior_buying), "browsing")

    def test_previous_buying_persists_through_short_refinement(self) -> None:
        """Preserve buying intent for an ambiguous constraint refinement."""

        state = SessionState("session", intent="buying")
        self.assertEqual(detect_intent("Preferably black", state), "buying")

    def test_previous_browsing_persists_through_exploratory_refinement(self) -> None:
        """Preserve browsing intent for an exploratory refinement."""

        state = SessionState("session", intent="browsing")
        self.assertEqual(detect_intent("Maybe leather or denim", state), "browsing")

    def test_current_buying_signal_overrides_previous_browsing(self) -> None:
        """Let explicit current buying evidence override browsing context."""

        state = SessionState("session", intent="browsing")
        self.assertEqual(detect_intent("Actually I need one under $80", state), "buying")

    def test_current_browsing_signal_overrides_previous_buying(self) -> None:
        """Let explicit current browsing evidence override buying context."""

        state = SessionState("session", intent="buying")
        self.assertEqual(
            detect_intent("I'm not buying yet, just looking around", state),
            "browsing",
        )

    def test_assistant_messages_are_not_user_intent_evidence(self) -> None:
        """Ignore assistant language when deriving user intent."""

        state = SessionState("session")
        state.message_history.append({
            "role": "assistant",
            "content": "I need you to buy this one.",
        })
        self.assertEqual(detect_intent("Maybe black", state), "browsing")

    def test_user_history_can_supply_context_when_intent_is_unset(self) -> None:
        """Use recent user history when stored intent has not been set."""

        state = SessionState("session")
        state.message_history.extend([
            {"role": "user", "content": "I need running shoes"},
            {"role": "assistant", "content": "Here are some choices."},
        ])
        self.assertEqual(detect_intent("Preferably black", state), "buying")

    def test_empty_or_ambiguous_first_message_defaults_to_browsing(self) -> None:
        """Use browsing as the conservative no-signal fallback."""

        for message in ("", "Maybe", "black cotton"):
            with self.subTest(message=message):
                self.assertEqual(detect_intent(message, SessionState("session")), "browsing")

    def test_strong_structured_search_can_imply_buying(self) -> None:
        """Infer buying from a sufficiently structured active search."""

        self.assertEqual(
            detect_intent(
                "black cotton shirt size M under $40 for work",
                SessionState("session"),
            ),
            "buying",
        )


class IntentAgentIntegrationTest(unittest.TestCase):
    """Verify intent lifecycle behavior inside the baseline Agent."""

    def setUp(self) -> None:
        """Create an Agent backed by a temporary one-product catalog."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.agent = Agent(_catalog_file(self.temporary_directory.name))

    def tearDown(self) -> None:
        """Close the Agent database and remove its temporary catalog."""

        self.agent.connection.close()
        self.temporary_directory.cleanup()

    def test_browsing_transitions_to_buying_after_successful_response(self) -> None:
        """Commit a browsing-to-buying transition after successful turns."""

        self.agent.reset("session", {})

        self.agent.respond("session", "What kinds of running shoes are good?", 1, 10)
        state = self.agent.sessions["session"]
        self.assertEqual(state.intent, "browsing")

        self.agent.respond("session", "Actually I need a black pair under $100.", 2, 10)

        self.assertEqual(state.intent, "buying")
        self.assertEqual(state.turn, 2)
        self.assertEqual(len(state.message_history), 4)
        self.assertIn("running", state.slots["use_case"])
        self.assertEqual(state.slots["color"], ["black"])
        self.assertEqual(len(state.slots["budget"]), 1)

    def test_failed_response_does_not_commit_new_intent(self) -> None:
        """Keep prior state unchanged when retrieval raises."""

        self.agent.reset("session", {})
        state = self.agent.sessions["session"]
        state.intent = "browsing"
        self.agent.connection.close()

        with self.assertRaises(sqlite3.ProgrammingError):
            self.agent.respond("session", "I need to buy black shoes now", 1, 10)

        self.assertEqual(state.intent, "browsing")
        self.assertEqual(state.turn, 0)
        self.assertEqual(state.message_history, [])
        self.assertTrue(all(not values for values in state.slots.values()))
        self.assertEqual(state.revealed_text, [])
        self.assertEqual(state.active_revealed_text, [])


if __name__ == "__main__":
    unittest.main()
