"""Tests for deterministic hard/soft constraint classification."""

from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dialogue.accumulator import accumulate_information
from dialogue.constraint_classifier import (
    ConstraintClassification,
    apply_constraint_classification,
    classify_constraints,
)
from dialogue.override_handler import apply_override, resolve_override
from dialogue.slot_extractor import extract_slots
from dialogue.state import SessionState
from starter.agent import Agent


def _classify(
    message: str,
    state: SessionState | None = None,
) -> ConstraintClassification:
    """Extract and classify one message against optional prior state."""

    current_state = state or SessionState("session")
    return classify_constraints(message, extract_slots(message), current_state)


def _commit_turn(state: SessionState, message: str) -> None:
    """Apply extraction, override, and classification in Agent commit order."""

    extraction = extract_slots(message)
    resolution = resolve_override(message, state, extraction)
    classification = classify_constraints(message, extraction, state)
    accumulate_information(state, extraction)
    apply_override(state, resolution)
    apply_constraint_classification(state, classification)


def _catalog_file(directory: str) -> Path:
    """Create a minimal catalog for transaction-safety integration tests."""

    path = Path(directory) / "catalog.jsonl"
    product = {
        "parent_asin": "A",
        "title": "White cotton Nike shirt",
        "categories": ["Clothing", "Shirts"],
        "features": ["waterproof"],
        "details": {},
        "store": "Nike",
        "description": [],
    }
    path.write_text(json.dumps(product) + "\n", encoding="utf-8")
    return path


class ConstraintClassifierTest(unittest.TestCase):
    """Verify pure slot-level classification and multi-turn updates."""

    def test_need_is_hard(self) -> None:
        """Classify a needed material as hard."""

        result = _classify("I need cotton.")
        self.assertEqual(result.hard_slots, frozenset({"material"}))
        self.assertEqual(result.soft_slots, frozenset())

    def test_prefer_is_soft(self) -> None:
        """Classify a preferred material as soft."""

        result = _classify("I'd prefer cotton.")
        self.assertEqual(result.hard_slots, frozenset())
        self.assertEqual(result.soft_slots, frozenset({"material"}))

    def test_mixed_contrast_is_classified_per_clause(self) -> None:
        """Separate hard material from soft color across a contrast boundary."""

        result = _classify("I need cotton but I'd prefer black.")
        self.assertEqual(result.hard_slots, frozenset({"material"}))
        self.assertEqual(result.soft_slots, frozenset({"color"}))

    def test_and_propagates_hard_cue(self) -> None:
        """Apply one hard cue to both slots joined by and."""

        result = _classify("I need black and leather.")
        self.assertEqual(result.hard_slots, frozenset({"color", "material"}))
        self.assertEqual(result.soft_slots, frozenset())

    def test_nearest_cue_handles_mixed_strength_without_contrast_word(self) -> None:
        """Associate each slot with its nearby cue inside one conjunction."""

        result = _classify("I need cotton and would like black.")
        self.assertEqual(result.hard_slots, frozenset({"material"}))
        self.assertEqual(result.soft_slots, frozenset({"color"}))

    def test_cued_comma_splits_but_plain_comma_propagates(self) -> None:
        """Split a comma only when it introduces a new strength cue."""

        mixed = _classify("I need cotton, I'd prefer black.")
        propagated = _classify("I need black, leather.")
        self.assertEqual(mixed.hard_slots, frozenset({"material"}))
        self.assertEqual(mixed.soft_slots, frozenset({"color"}))
        self.assertEqual(propagated.hard_slots, frozenset({"color", "material"}))

    def test_neutral_color_is_soft(self) -> None:
        """Default an unqualified color mention to soft."""

        result = _classify("Black.")
        self.assertEqual(result.soft_slots, frozenset({"color"}))

    def test_must_is_hard(self) -> None:
        """Classify a mandatory color as hard."""

        result = _classify("It must be black.")
        self.assertEqual(result.hard_slots, frozenset({"color"}))

    def test_exact_budget_bound_is_hard(self) -> None:
        """Treat an unqualified numeric upper bound as hard."""

        for message in ("Under $100.", "At most $80.", "No more than $50."):
            with self.subTest(message=message):
                result = _classify(message)
                self.assertEqual(result.hard_slots, frozenset({"budget"}))

    def test_flexible_budget_language_is_soft(self) -> None:
        """Let explicit flexibility soften exact and approximate budgets."""

        for message in (
            "Ideally under $100.",
            "Around $100.",
            "Roughly $100.",
        ):
            with self.subTest(message=message):
                result = _classify(message)
                self.assertEqual(result.hard_slots, frozenset())
                self.assertEqual(result.soft_slots, frozenset({"budget"}))

    def test_category_defaults_to_hard(self) -> None:
        """Treat a confidently extracted shopping category as operationally hard."""

        result = _classify("I am looking for Shirts.")
        self.assertIn("category", result.hard_slots)

    def test_promotion_moves_slot_from_soft_to_hard(self) -> None:
        """Promote an existing material after explicit mandatory language."""

        state = SessionState("session")
        _commit_turn(state, "I'd prefer cotton.")
        _commit_turn(state, "It absolutely must be cotton.")
        self.assertEqual(state.hard_constraints, {"material"})
        self.assertEqual(state.soft_preferences, set())

    def test_demotion_moves_slot_from_hard_to_soft(self) -> None:
        """Demote an existing brand after explicit flexibility language."""

        state = SessionState("session")
        _commit_turn(state, "It must be Nike.")
        _commit_turn(state, "Nike isn't necessary, I just prefer it.")
        self.assertEqual(state.hard_constraints, set())
        self.assertEqual(state.soft_preferences, {"brand"})

    def test_slot_name_can_reclassify_without_repeating_value(self) -> None:
        """Use an explicit slot reference to promote or demote existing values."""

        state = SessionState("session")
        _commit_turn(state, "I'd prefer cotton but Nike must be included.")
        _commit_turn(state, "The material is essential.")
        self.assertIn("material", state.hard_constraints)
        self.assertIn("brand", state.hard_constraints)

        _commit_turn(state, "The brand isn't necessary.")
        self.assertIn("material", state.hard_constraints)
        self.assertIn("brand", state.soft_preferences)

    def test_neutral_replacement_preserves_strength(self) -> None:
        """Keep a hard color hard when Step 6 neutrally replaces its value."""

        state = SessionState("session")
        _commit_turn(state, "I need black.")
        _commit_turn(state, "Actually white instead.")
        self.assertEqual(state.slots["color"], ["white"])
        self.assertEqual(state.hard_constraints, {"color"})
        self.assertEqual(state.soft_preferences, set())

    def test_explicit_replacement_can_change_strength(self) -> None:
        """Let soft replacement wording demote a formerly hard color."""

        state = SessionState("session")
        _commit_turn(state, "I need black.")
        _commit_turn(state, "Actually I'd prefer white instead.")
        self.assertEqual(state.slots["color"], ["white"])
        self.assertEqual(state.hard_constraints, set())
        self.assertEqual(state.soft_preferences, {"color"})

    def test_clearing_removes_stale_strength(self) -> None:
        """Remove both labels when Step 6 clears the last brand value."""

        state = SessionState("session")
        _commit_turn(state, "It must be Nike.")
        _commit_turn(state, "I don't care about brand anymore.")
        self.assertEqual(state.slots["brand"], [])
        self.assertNotIn("brand", state.hard_constraints)
        self.assertNotIn("brand", state.soft_preferences)

    def test_unrelated_accumulated_strength_survives(self) -> None:
        """Leave earlier hard material untouched by a neutral color turn."""

        state = SessionState("session")
        _commit_turn(state, "I need cotton.")
        _commit_turn(state, "Black.")
        self.assertEqual(state.hard_constraints, {"material"})
        self.assertEqual(state.soft_preferences, {"color"})

    def test_strength_sets_remain_mutually_exclusive(self) -> None:
        """Maintain disjoint strength sets through repeated reclassification."""

        state = SessionState("session")
        for message in (
            "I'd prefer cotton.",
            "It must be cotton.",
            "Cotton isn't necessary, I'd prefer it.",
        ):
            _commit_turn(state, message)
            self.assertTrue(
                state.hard_constraints.isdisjoint(state.soft_preferences)
            )

    def test_classifier_is_pure_and_application_does_not_touch_raw_text(self) -> None:
        """Keep classification pure and raw-text fields outside Step 7 ownership."""

        state = SessionState("session")
        state.revealed_text.append("Black")
        state.active_revealed_text.append("Black")
        before = copy.deepcopy(state)
        extraction = extract_slots("It must be black.")
        classification = classify_constraints("It must be black.", extraction, state)
        self.assertEqual(state, before)

        state.add_slot_values("color", extraction.slots["color"])
        apply_constraint_classification(state, classification)
        self.assertEqual(state.revealed_text, ["Black"])
        self.assertEqual(state.active_revealed_text, ["Black"])


class ConstraintClassifierAgentIntegrationTest(unittest.TestCase):
    """Verify post-success classification in the current Agent lifecycle."""

    def setUp(self) -> None:
        """Create an Agent backed by a temporary catalog."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.agent = Agent(_catalog_file(self.temporary_directory.name))

    def tearDown(self) -> None:
        """Close the Agent database and remove its temporary catalog."""

        self.agent.connection.close()
        self.temporary_directory.cleanup()

    def test_agent_commits_mixed_strength_after_success(self) -> None:
        """Commit mixed classification through Agent.respond()."""

        self.agent.reset("session", {})
        self.agent.respond(
            "session", "I need cotton but I'd prefer black.", 1, 10,
        )
        state = self.agent.sessions["session"]
        self.assertEqual(state.hard_constraints, {"material"})
        self.assertEqual(state.soft_preferences, {"color"})

    def test_failed_respond_leaves_complete_state_unchanged(self) -> None:
        """Roll back strengths and every other state field when retrieval fails."""

        self.agent.reset("session", {})
        state = self.agent.sessions["session"]
        state.set_constraint("material", ["cotton"], strength="hard")
        state.set_constraint("color", ["black"], strength="soft")
        state.revealed_text.extend(["cotton", "black"])
        state.active_revealed_text.extend(["cotton", "black"])
        state.intent = "buying"
        state.override_detected = True
        before = copy.deepcopy(state)
        self.agent.connection.close()

        with self.assertRaises(sqlite3.ProgrammingError):
            self.agent.respond(
                "session", "Actually white must replace black.", 2, 10,
            )

        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
