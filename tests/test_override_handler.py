"""Tests for explicit constraint correction and Agent integration."""

from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dialogue.accumulator import accumulate_information
from dialogue.override_handler import apply_override, resolve_override
from dialogue.slot_extractor import SlotExtraction, extract_slots
from dialogue.state import SessionState
from dialogue.types import BudgetConstraint
from starter.agent import Agent


def _apply_message(state: SessionState, message: str) -> None:
    """Extract, accumulate, and apply an override like a successful commit."""

    extraction = extract_slots(message)
    resolution = resolve_override(message, state, extraction)
    accumulate_information(state, extraction)
    apply_override(state, resolution)


def _record_prior_turn(state: SessionState, message: str) -> None:
    """Commit a prior successful turn needed for history-aware resolution."""

    accumulate_information(state, extract_slots(message))
    state.message_history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": "Here are the closest matches."},
    ])


def _catalog_file(directory: str) -> Path:
    """Create a minimal catalog used by Agent integration tests."""

    path = Path(directory) / "catalog.jsonl"
    product = {
        "parent_asin": "A",
        "title": "White waterproof cotton shirt",
        "categories": ["Clothing", "Shirts"],
        "features": ["waterproof"],
        "details": {},
        "store": "Example",
        "description": [],
    }
    path.write_text(json.dumps(product) + "\n", encoding="utf-8")
    return path


class OverrideHandlerTest(unittest.TestCase):
    """Verify isolated replacement, clearing, and removal semantics."""

    def test_color_replacement(self) -> None:
        """Replace an existing color after explicit correction language."""

        state = SessionState("session")
        state.add_slot_values("color", ["black"])
        _apply_message(state, "Actually white instead")
        self.assertEqual(state.slots["color"], ["white"])
        self.assertTrue(state.override_detected)

    def test_material_replacement(self) -> None:
        """Replace an existing material while preserving its normalized type."""

        state = SessionState("session")
        state.add_slot_values("material", ["cotton"])
        _apply_message(state, "Make it linen instead")
        self.assertEqual(state.slots["material"], ["linen"])

    def test_brand_replacement(self) -> None:
        """Replace an existing brand using extracted brand values."""

        state = SessionState("session")
        state.add_slot_values("brand", ["Nike"])
        extraction = SlotExtraction()
        extraction.slots["brand"] = ["Adidas"]
        message = "Actually Adidas instead"
        resolution = resolve_override(message, state, extraction)
        accumulate_information(state, extraction)
        apply_override(state, resolution)
        self.assertEqual(state.slots["brand"], ["Adidas"])

    def test_budget_replacement(self) -> None:
        """Replace a prior structured budget rather than accumulating both."""

        state = SessionState("session")
        state.add_slot_values("budget", [BudgetConstraint(maximum=100, currency="$")])
        _apply_message(state, "Actually up to $150")
        self.assertEqual(
            state.slots["budget"],
            [BudgetConstraint(maximum=150, currency="$")],
        )

    def test_clear_brand(self) -> None:
        """Clear a brand constraint when the user retracts it explicitly."""

        state = SessionState("session")
        state.add_slot_values("brand", ["Nike"])
        _apply_message(state, "I don't care about the brand anymore")
        self.assertEqual(state.slots["brand"], [])
        self.assertTrue(state.override_detected)

    def test_clear_color_and_strength_classification(self) -> None:
        """Clear both color values and any stale hard classification."""

        state = SessionState("session")
        state.set_constraint("color", ["black"], strength="hard")
        _apply_message(state, "Color doesn't matter anymore")
        self.assertEqual(state.slots["color"], [])
        self.assertNotIn("color", state.hard_constraints)
        self.assertNotIn("color", state.soft_preferences)

    def test_clear_budget_limit(self) -> None:
        """Recognize an explicit request to forget the budget limit."""

        state = SessionState("session")
        state.add_slot_values("budget", [BudgetConstraint(maximum=100)])
        _apply_message(state, "Forget the budget limit")
        self.assertEqual(state.slots["budget"], [])

    def test_replacement_preserves_existing_strength(self) -> None:
        """Keep a hard slot hard when only its value is replaced."""

        state = SessionState("session")
        state.set_constraint("color", ["black"], strength="hard")

        _apply_message(state, "Actually white instead")

        self.assertEqual(state.slots["color"], ["white"])
        self.assertIn("color", state.hard_constraints)
        self.assertNotIn("color", state.soft_preferences)

    def test_targeted_value_removal(self) -> None:
        """Remove only the rejected value when alternatives remain valid."""

        state = SessionState("session")
        state.add_slot_values("color", ["black", "blue"])
        state.revealed_text.extend(["black", "blue", "cotton"])
        state.active_revealed_text.extend(["black", "blue", "cotton"])
        extraction = SlotExtraction(revealed_text=["black"])
        extraction.slots["color"] = ["black"]
        resolution = resolve_override("Not black", state, extraction)
        accumulate_information(state, extraction)
        apply_override(state, resolution)
        self.assertEqual(state.slots["color"], ["blue"])
        self.assertEqual(state.revealed_text, ["black", "blue", "cotton"])
        self.assertEqual(state.active_revealed_text, ["blue", "cotton"])

    def test_general_retraction_removes_last_preference_across_slots(self) -> None:
        """Retract the last stated preference when a new slot is introduced."""

        state = SessionState("session")
        state.add_slot_values("category", ["Belts"])
        state.add_slot_values("feature", ["buckle closure"])
        state.revealed_text.extend(["Belts", "Buckle closure"])
        state.active_revealed_text.extend(["Belts", "Buckle closure"])
        state.message_history.extend([
            {"role": "user", "content": "I'm looking for Belts. Buckle closure"},
            {"role": "assistant", "content": "Here are the closest matches."},
        ])

        _apply_message(
            state,
            "Actually, ignore my earlier preference. What I need is: leather.",
        )

        self.assertEqual(state.slots["category"], ["Belts"])
        self.assertEqual(state.slots["feature"], [])
        self.assertEqual(state.slots["material"], ["leather"])
        self.assertEqual(
            state.revealed_text,
            ["Belts", "Buckle closure", "leather"],
        )
        self.assertEqual(state.active_revealed_text, ["Belts", "leather"])
        self.assertTrue(state.override_detected)

    def test_general_retraction_finds_turn_one_across_unrelated_turns(self) -> None:
        """Prune the initial active preference without pruning newer constraints."""

        for old_phrase in ("No Closure closure", "Hand Wash Only", "Pull On closure"):
            with self.subTest(old_phrase=old_phrase):
                state = SessionState("session")
                _record_prior_turn(state, f"I'm looking for Belts. {old_phrase}")
                _record_prior_turn(state, "For that, what matters is: black.")
                _record_prior_turn(state, "For that, what matters is: size M.")

                _apply_message(
                    state,
                    "Actually, ignore my earlier preference. What I need is: leather.",
                )

                self.assertIn(old_phrase, state.revealed_text)
                self.assertNotIn(old_phrase, state.active_revealed_text)
                self.assertIn("black", state.active_revealed_text)
                self.assertIn("size M", state.active_revealed_text)
                self.assertIn("Belts.", state.active_revealed_text)
                self.assertIn("Belts", state.slots["category"])
                self.assertTrue(state.override_detected)

    def test_general_retraction_removes_only_old_feature_value(self) -> None:
        """Keep unrelated active feature, material, color, and category state."""

        state = SessionState("session")
        _record_prior_turn(state, "I'm looking for Belts. Pull On closure")
        _record_prior_turn(state, "For that, what matters is: waterproof.")
        _record_prior_turn(state, "For that, what matters is: black.")

        _apply_message(
            state,
            "Actually, ignore my earlier preference. What I need is: leather.",
        )

        self.assertEqual(state.slots["feature"], ["waterproof"])
        self.assertEqual(state.slots["color"], ["black"])
        self.assertEqual(state.slots["material"], ["leather"])
        self.assertIn("Belts", state.slots["category"])
        self.assertIn("waterproof", state.active_revealed_text)
        self.assertNotIn("Pull On closure", state.active_revealed_text)

    def test_general_retraction_prunes_long_raw_phrase_without_slot_invention(self) -> None:
        """Deactivate a long raw preference while retaining historical wording."""

        old_phrase = (
            "Long torso camisole for extra coverage with spagetti adjustable "
            "strap for perfect fit"
        )
        state = SessionState("session")
        _record_prior_turn(state, f"I'm looking for Camisoles. {old_phrase}")
        _record_prior_turn(state, "For that, what matters is: black.")

        _apply_message(
            state,
            "Actually, ignore my earlier preference. What I need is: cotton.",
        )

        self.assertIn(old_phrase, state.revealed_text)
        self.assertNotIn(old_phrase, state.active_revealed_text)
        self.assertIn("black", state.active_revealed_text)

    def test_unrelated_constraints_are_preserved(self) -> None:
        """Keep category, material, and size during a color replacement."""

        state = SessionState("session")
        state.add_slot_values("category", ["Shirts"])
        state.add_slot_values("color", ["black"])
        state.add_slot_values("material", ["cotton"])
        state.add_slot_values("size", ["M"])
        _apply_message(state, "Actually white instead")
        self.assertEqual(state.slots["category"], ["Shirts"])
        self.assertEqual(state.slots["color"], ["white"])
        self.assertEqual(state.slots["material"], ["cotton"])
        self.assertEqual(state.slots["size"], ["M"])

    def test_additive_language_is_not_override(self) -> None:
        """Accumulate additive preferences without setting the event flag."""

        state = SessionState("session")
        state.add_slot_values("color", ["black"])
        _apply_message(state, "I also like white")
        self.assertEqual(state.slots["color"], ["black", "white"])
        self.assertFalse(state.override_detected)

    def test_weak_language_is_not_destructive(self) -> None:
        """Accumulate a tentative value without erasing previous values."""

        state = SessionState("session")
        state.add_slot_values("color", ["black"])
        _apply_message(state, "Maybe white")
        self.assertEqual(state.slots["color"], ["black", "white"])
        self.assertFalse(state.override_detected)

    def test_raw_text_and_history_remain_historical(self) -> None:
        """Preserve old raw wording and transcript during operational repair."""

        state = SessionState("session")
        state.add_slot_values("color", ["black"])
        state.revealed_text.append("black")
        state.active_revealed_text.append("black")
        state.message_history.append({"role": "user", "content": "black"})
        history_before = copy.deepcopy(state.message_history)
        _apply_message(state, "Actually white instead")
        self.assertEqual(state.revealed_text, ["black", "white"])
        self.assertEqual(state.active_revealed_text, ["white"])
        self.assertEqual(state.message_history, history_before)

    def test_explicit_override_without_stored_value_sets_flag(self) -> None:
        """Record explicit override intent even when no prior value can change."""

        state = SessionState("session")
        _apply_message(state, "Actually white instead")
        self.assertEqual(state.slots["color"], ["white"])
        self.assertTrue(state.override_detected)

    def test_evaluator_style_override_without_matching_value_sets_flag(self) -> None:
        """Detect evaluator-style correction language without a mutable match."""

        state = SessionState("session")
        _apply_message(
            state,
            "Actually, ignore my earlier preference. What I need is: leather.",
        )
        self.assertEqual(state.slots["material"], ["leather"])
        self.assertTrue(state.override_detected)

    def test_replacement_removes_only_stale_active_phrase(self) -> None:
        """Deactivate replaced color wording while preserving unrelated phrases."""

        state = SessionState("session")
        state.add_slot_values("color", ["black"])
        state.add_slot_values("material", ["cotton"])
        state.revealed_text.extend(["black", "cotton"])
        state.active_revealed_text.extend(["black", "cotton"])

        _apply_message(state, "Actually white instead")

        self.assertEqual(state.slots["color"], ["white"])
        self.assertEqual(state.revealed_text, ["black", "cotton", "white"])
        self.assertEqual(state.active_revealed_text, ["cotton", "white"])


class OverrideAgentIntegrationTest(unittest.TestCase):
    """Verify transactional override behavior in Agent.respond."""

    def setUp(self) -> None:
        """Create an Agent backed by a temporary catalog."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.agent = Agent(_catalog_file(self.temporary_directory.name))

    def tearDown(self) -> None:
        """Close the Agent database and delete the temporary catalog."""

        self.agent.connection.close()
        self.temporary_directory.cleanup()

    def test_multi_turn_replacement_then_normal_addition(self) -> None:
        """Replace color surgically and then accumulate a normal feature."""

        self.agent.reset("session", {})
        self.agent.respond("session", "I need a black cotton shirt size M", 1, 10)
        self.agent.respond("session", "Actually make it white instead", 2, 10)
        self.agent.respond("session", "And waterproof if possible", 3, 10)

        state = self.agent.sessions["session"]
        self.assertEqual(state.turn, 3)
        self.assertEqual(len(state.message_history), 6)
        self.assertIn("Shirts", state.slots["category"])
        self.assertEqual(state.slots["color"], ["white"])
        self.assertEqual(state.slots["material"], ["cotton"])
        self.assertEqual(state.slots["size"], ["M"])
        self.assertEqual(state.slots["feature"], ["waterproof"])
        self.assertTrue(state.override_detected)
        self.assertEqual(state.intent, "buying")

    def test_override_message_can_preserve_buying_intent(self) -> None:
        """Keep Step 5 buying classification while correcting a slot."""

        self.agent.reset("session", {})
        self.agent.respond("session", "I need a black shirt", 1, 10)
        self.agent.respond("session", "Actually I need a white one instead", 2, 10)
        state = self.agent.sessions["session"]
        self.assertEqual(state.intent, "buying")
        self.assertEqual(state.slots["color"], ["white"])

    def test_failed_response_does_not_commit_override(self) -> None:
        """Leave every state field unchanged when retrieval fails."""

        self.agent.reset("session", {})
        state = self.agent.sessions["session"]
        state.set_constraint("color", ["black"], strength="soft")
        state.revealed_text.extend(["black", "Hand Wash Only"])
        state.active_revealed_text.extend(["black", "Hand Wash Only"])
        state.message_history.extend([
            {"role": "user", "content": "I'm looking for Shirts. Hand Wash Only"},
            {"role": "assistant", "content": "Here are the closest matches."},
        ])
        state.intent = "buying"
        before = copy.deepcopy(state)
        self.agent.connection.close()

        with self.assertRaises(sqlite3.ProgrammingError):
            self.agent.respond(
                "session",
                "Actually, ignore my earlier preference. What I need is: leather.",
                1,
                10,
            )

        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
