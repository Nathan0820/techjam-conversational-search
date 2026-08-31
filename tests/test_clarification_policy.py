"""Focused tests for the Step 8 clarification policy and its integration."""

from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dialogue.clarification_policy import (
    ALLOWED_ASK_ATTRIBUTES,
    ClarificationDecision,
    apply_clarification_decision,
    decide_clarification,
    evaluate_previous_ask_yield,
    select_response_ask_attribute,
)
from dialogue.override_handler import OverrideResolution, resolve_override
from dialogue.slot_extractor import SlotExtraction, extract_slots
from dialogue.state import SUPPORTED_SLOTS, SessionState
from dialogue.types import ShoppingIntent, SlotValue
from starter.agent import Agent


def _extraction(**values: list[SlotValue]) -> SlotExtraction:
    """Build a compact extraction with values assigned by slot name."""

    extraction = SlotExtraction()
    for slot_name, slot_values in values.items():
        extraction.slots[slot_name] = list(slot_values)
    return extraction


def _state(
    intent: ShoppingIntent = "buying",
    **values: list[SlotValue],
) -> SessionState:
    """Build session state with selected operational slot values."""

    state = SessionState("session", intent=intent)
    for slot_name, slot_values in values.items():
        state.add_slot_values(slot_name, slot_values)
    return state


def _decide(
    state: SessionState,
    extraction: SlotExtraction | None = None,
    *,
    turn: int = 1,
    intent: ShoppingIntent | None = None,
    override: OverrideResolution | None = None,
    yielded: bool | None = None,
) -> ClarificationDecision:
    """Call the policy with concise defaults used throughout these tests."""

    return decide_clarification(
        state,
        extraction or SlotExtraction(),
        turn,
        current_intent=intent,
        override_resolution=override,
        previous_ask_yield=yielded,
    )


def _catalog_file(directory: str) -> Path:
    """Create a minimal searchable catalog for Agent lifecycle tests."""

    path = Path(directory) / "catalog.jsonl"
    products = (
        {
            "parent_asin": "A",
            "title": "Cotton shirt",
            "categories": ["Clothing", "Shirts"],
            "features": ["cotton", "black"],
            "details": {},
            "store": "Example",
            "description": [],
        },
        {
            "parent_asin": "B",
            "title": "Running shoes",
            "categories": ["Shoes", "Athletic Walking"],
            "features": ["rubber sole", "size 8"],
            "details": {},
            "store": "Example",
            "description": [],
        },
    )
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )
    return path


class ClarificationPolicyTest(unittest.TestCase):
    """Verify pure ask selection, stopping, projection, and yield rules."""

    def test_missing_category_is_requested_first(self) -> None:
        """Ask for the shopping target before secondary attributes."""

        decision = _decide(_state(material=["cotton"]))
        self.assertEqual(decision, ClarificationDecision("category", True))

    def test_weak_buying_state_asks_useful_missing_attribute(self) -> None:
        """Continue a buying dialogue that only identifies a category."""

        decision = _decide(_state(category=["Shirts"]))
        self.assertEqual(decision.ask_attribute, "material")

    def test_enough_buying_constraints_act_without_question(self) -> None:
        """Stop after two strongly informative buying constraints are known."""

        state = _state(category=["Shirts"], material=["cotton"], feature=["waterproof"])
        self.assertIsNone(_decide(state).ask_attribute)

    def test_act_decision_uses_one_api_fallback_without_changing_policy(self) -> None:
        """Expose other while preserving the internal should-ask false result."""

        state = _state(
            category=["Shirts"], material=["cotton"], feature=["waterproof"],
        )
        decision = _decide(state)

        self.assertEqual(decision, ClarificationDecision(None, False))
        self.assertEqual(select_response_ask_attribute(
            state, decision, previous_ask_yield=None,
        ), "other")
        self.assertFalse(decision.should_ask)
        self.assertEqual(state.asked_attributes, set())

    def test_other_fallback_repeats_only_after_its_own_useful_yield(self) -> None:
        """Repeat a useful immediately previous broad ask, but stop after failure."""

        state = _state(
            category=["Shirts"], material=["cotton"], feature=["waterproof"],
        )
        state.asked_attributes.add("other")
        state.last_asked_attribute = "other"
        state.last_ask_yielded = True
        decision = _decide(state)

        self.assertEqual(decision, ClarificationDecision(None, False))
        self.assertEqual(select_response_ask_attribute(
            state, decision, previous_ask_yield=True,
        ), "other")
        state.last_ask_yielded = False
        self.assertIsNone(select_response_ask_attribute(
            state, decision, previous_ask_yield=False,
        ))

    def test_old_other_yield_does_not_override_a_different_previous_ask(self) -> None:
        """Do not reuse broad-ask history when another attribute was asked last."""

        state = _state(
            category=["Shirts"], material=["cotton"], feature=["waterproof"],
        )
        state.asked_attributes.update({"other", "material"})
        state.last_asked_attribute = "material"
        state.last_ask_yielded = True
        decision = _decide(state)

        self.assertEqual(decision, ClarificationDecision(None, False))
        self.assertIsNone(select_response_ask_attribute(
            state, decision, previous_ask_yield=True,
        ))

    def test_targeted_policy_question_is_unchanged_by_response_adapter(self) -> None:
        """Keep useful material and feature selections ahead of the fallback."""

        state = _state(category=["Shirts"])
        material = _decide(state)
        self.assertEqual(material, ClarificationDecision("material", True))
        self.assertEqual(select_response_ask_attribute(
            state, material, previous_ask_yield=None,
        ), "material")

        state.asked_attributes.add("material")
        feature = _decide(state)
        self.assertEqual(feature, ClarificationDecision("feature", True))
        self.assertEqual(select_response_ask_attribute(
            state, feature, previous_ask_yield=False,
        ), "feature")

    def test_browsing_stops_earlier_than_buying(self) -> None:
        """Use a lower evidence threshold for exploratory users."""

        # Evidence score 4 (material 3 + color 1), which clears the browsing threshold
        # of 4 but not the buying threshold of 5. The previous fixture also included
        # style, scoring 5, which no longer distinguishes the two now that buying is 5
        # rather than 6 (see E8 in decisions.md).
        state = _state(category=["Shirts"], material=["cotton"], color=["black"])
        self.assertIsNone(_decide(state, intent="browsing").ask_attribute)
        self.assertIsNotNone(_decide(state, intent="buying").ask_attribute)

    def test_known_material_is_not_requested(self) -> None:
        """Exclude attributes already present in projected operational state."""

        state = _state(category=["Shirts"], material=["cotton"])
        decision = _decide(state, intent="buying")
        self.assertNotEqual(decision.ask_attribute, "material")

    def test_already_asked_material_is_not_repeated(self) -> None:
        """Treat asked_attributes as historical repetition protection."""

        state = _state(category=["Shirts"])
        state.asked_attributes.add("material")
        self.assertNotEqual(_decide(state).ask_attribute, "material")

    def test_all_useful_attributes_asked_means_act(self) -> None:
        """Return no question when every remaining candidate was attempted."""

        state = _state(category=["Shirts"])
        state.asked_attributes.update(ALLOWED_ASK_ATTRIBUTES)
        self.assertEqual(_decide(state), ClarificationDecision(None, False))

    def test_return_after_act_resumes_with_untried_attribute(self) -> None:
        """Ask again when recommendations were followed by no new user facts."""

        state = _state("browsing", category=["Shirts"], material=["cotton"])
        state.turn = 1
        state.last_asked_attribute = None
        self.assertEqual(_decide(state, turn=2).ask_attribute, "feature")

    def test_shoes_with_color_and_brand_prioritize_size(self) -> None:
        """Prefer shoe size when only weak selectors are currently known."""

        state = _state(category=["Shoes"], color=["black"], brand=["Nike"])
        self.assertEqual(_decide(state).ask_attribute, "size")

    def test_shoes_with_size_and_use_case_have_enough_information(self) -> None:
        """Act when two critical shoe selectors already provide specificity."""

        state = _state(category=["Shoes"], size=["8"], use_case=["running"])
        self.assertIsNone(_decide(state).ask_attribute)

    def test_clothing_weak_state_requests_material(self) -> None:
        """Use clothing-specific priority for an otherwise empty target."""

        state = _state(category=["Tees & Blouses Tunics"])
        self.assertEqual(_decide(state).ask_attribute, "material")

    def test_previous_material_ask_yields_material(self) -> None:
        """Mark the previous question useful when its requested slot appears."""

        state = _state()
        state.last_asked_attribute = "material"
        self.assertTrue(evaluate_previous_ask_yield(
            state, _extraction(material=["cotton"]),
        ))

    def test_previous_material_ask_does_not_yield_color(self) -> None:
        """Do not credit an unrelated extracted attribute to the prior ask."""

        state = _state()
        state.last_asked_attribute = "material"
        self.assertFalse(evaluate_previous_ask_yield(
            state, _extraction(color=["black"]),
        ))

    def test_no_previous_question_has_unknown_yield(self) -> None:
        """Keep yield unknown when the immediately previous response did not ask."""

        self.assertIsNone(evaluate_previous_ask_yield(_state(), SlotExtraction()))

    def test_low_yield_question_is_not_immediately_repeated(self) -> None:
        """Move to another missing attribute after an unanswered material ask."""

        state = _state(category=["Shirts"])
        state.asked_attributes.add("material")
        state.last_asked_attribute = "material"
        decision = _decide(state, yielded=False)
        self.assertNotEqual(decision.ask_attribute, "material")

    def test_current_extraction_counts_before_commit(self) -> None:
        """Stop using facts extracted on this turn even before accumulation."""

        state = _state(category=["Shoes"])
        extraction = _extraction(size=["8"], use_case=["running"])
        self.assertIsNone(_decide(state, extraction).ask_attribute)

    def test_override_clear_is_projected_as_missing(self) -> None:
        """Ignore a stale material value when the current resolution clears it."""

        state = _state(category=["Shirts"], material=["cotton"])
        override = OverrideResolution(detected=True, clear_slots=("material",))
        self.assertEqual(_decide(state, override=override).ask_attribute, "material")

    def test_override_replacement_counts_as_current_information(self) -> None:
        """Count the replacement value while excluding its superseded value."""

        state = _state("browsing", category=["Shirts"], material=["cotton"])
        extraction = _extraction(material=["linen"])
        override = OverrideResolution(
            detected=True,
            replacement_values={"material": ("linen",)},
        )
        decision = _decide(state, extraction, intent="browsing", override=override)
        self.assertNotEqual(decision.ask_attribute, "material")

    def test_real_override_resolution_is_used_without_reparsing(self) -> None:
        """Use Step 6 output to treat an explicit clear as operationally empty."""

        state = _state(category=["Shirts"], material=["cotton"])
        message = "I don't care about material anymore"
        extraction = extract_slots(message)
        override = resolve_override(message, state, extraction)
        self.assertEqual(_decide(state, extraction, override=override).ask_attribute, "material")

    def test_hard_soft_sets_only_strengthen_known_information(self) -> None:
        """Allow hard specificity without treating stale labels as slot values."""

        state = _state(
            category=["Shirts"], material=["cotton"], color=["black"], style=["casual"],
        )
        state.hard_constraints.update({"material", "color", "brand"})
        state.soft_preferences.add("style")
        decision = _decide(state)
        self.assertIsNone(decision.ask_attribute)

    def test_decision_function_does_not_mutate_state(self) -> None:
        """Keep projection and selection pure until an explicit apply step."""

        state = _state(category=["Shirts"], material=["cotton"])
        before = copy.deepcopy(state)
        _decide(state, _extraction(color=["black"]))
        self.assertEqual(state, before)

    def test_apply_commits_only_dialogue_policy_fields(self) -> None:
        """Record a successful question and the previous question's yield."""

        state = _state(category=["Shirts"])
        apply_clarification_decision(
            state, ClarificationDecision("material", True), False,
        )
        self.assertEqual(state.asked_attributes, {"material"})
        self.assertEqual(state.last_asked_attribute, "material")
        self.assertFalse(state.last_ask_yielded)

    def test_every_decision_uses_allowed_api_value(self) -> None:
        """Keep policy outputs within the response schema enumeration."""

        samples = (
            _state(),
            _state(category=["Shoes"]),
            _state(category=["Shirts"], material=["cotton"]),
            _state(category=["Belts"], feature=["buckle closure"]),
        )
        for state in samples:
            with self.subTest(slots=state.slots):
                attribute = _decide(state).ask_attribute
                self.assertTrue(attribute is None or attribute in ALLOWED_ASK_ATTRIBUTES)

    def test_every_response_adapter_value_is_allowed_or_null(self) -> None:
        """Keep fallback output within the public API enumeration."""

        states = (
            _state(),
            _state(category=["Shirts"], material=["cotton"], feature=["waterproof"]),
        )
        for state in states:
            with self.subTest(slots=state.slots):
                attribute = select_response_ask_attribute(
                    state, _decide(state), previous_ask_yield=None,
                )
                self.assertTrue(attribute is None or attribute in ALLOWED_ASK_ATTRIBUTES)

    def test_invalid_or_inconsistent_decision_is_rejected(self) -> None:
        """Enforce response-schema and should_ask consistency at construction."""

        with self.assertRaisesRegex(ValueError, "unsupported"):
            ClarificationDecision("invalid", True)
        with self.assertRaisesRegex(ValueError, "should_ask"):
            ClarificationDecision(None, True)


class ClarificationAgentIntegrationTest(unittest.TestCase):
    """Verify successful commits and complete rollback through Agent.respond."""

    def setUp(self) -> None:
        """Create a fresh indexed agent for each lifecycle test."""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.agent = Agent(_catalog_file(self.temporary_directory.name))
        self.agent.reset("session", {})

    def tearDown(self) -> None:
        """Close the index and remove the temporary catalog."""

        try:
            self.agent.connection.close()
        finally:
            self.temporary_directory.cleanup()

    def test_successful_questions_and_yield_are_committed(self) -> None:
        """Track a material question followed by a matching customer answer."""

        first = self.agent.respond("session", "I want Shirts", 1, 10)
        self.assertEqual(first["ask_attribute"], "material")
        self.assertIn("material", first["message"].casefold())

        second = self.agent.respond("session", "Cotton.", 2, 10)
        state = self.agent.sessions["session"]
        self.assertTrue(state.last_ask_yielded)
        self.assertEqual(state.last_asked_attribute, second["ask_attribute"])
        self.assertIn("material", state.asked_attributes)

    def test_useful_other_repeats_then_stops_after_an_empty_reply(self) -> None:
        """Commit a useful repeat and expose null after that repeat yields nothing."""

        first = self.agent.respond(
            "session", "I need waterproof cotton Shirts.", 1, 10,
        )
        state = self.agent.sessions["session"]
        self.assertEqual(first["ask_attribute"], "other")
        self.assertIn("another requirement", first["message"].casefold())
        self.assertEqual(state.asked_attributes, {"other"})
        self.assertEqual(state.last_asked_attribute, "other")

        second = self.agent.respond("session", "Silk lining.", 2, 10)
        self.assertEqual(second["ask_attribute"], "other")
        self.assertTrue(state.last_ask_yielded)
        self.assertEqual(state.last_asked_attribute, "other")
        self.assertEqual(state.asked_attributes, {"other"})

        third = self.agent.respond("session", "Nothing else.", 3, 10)
        self.assertIsNone(third["ask_attribute"])
        self.assertFalse(state.last_ask_yielded)
        self.assertEqual(state.asked_attributes, {"other"})

    def test_failed_retrieval_leaves_all_state_unchanged(self) -> None:
        """Roll back ask history, yield tracking, and every existing state field."""

        state = self.agent.sessions["session"]
        state.set_constraint("category", ["Shirts"], "hard")
        state.set_constraint("material", ["cotton"], "hard")
        state.set_constraint("color", ["black"], "soft")
        state.set_constraint("feature", ["waterproof"], "hard")
        state.asked_attributes.update({"material", "other"})
        state.last_asked_attribute = "other"
        state.last_ask_yielded = False
        state.intent = "buying"
        state.override_detected = True
        state.turn = 1
        state.message_history = [
            {"role": "user", "content": "I need waterproof cotton Shirts in black."},
            {"role": "assistant", "content": "Here are some options."},
        ]
        state.revealed_text = ["waterproof", "cotton", "Shirts", "black"]
        state.active_revealed_text = ["waterproof", "cotton", "Shirts", "black"]
        fallback_message = "Actually white isn't required."
        fallback_extraction = extract_slots(fallback_message)
        previous_ask_yield = evaluate_previous_ask_yield(
            state, fallback_extraction,
        )
        self.assertTrue(previous_ask_yield)
        decision = _decide(state, fallback_extraction, turn=2)
        self.assertEqual(decision, ClarificationDecision(None, False))
        self.assertEqual(select_response_ask_attribute(
            state,
            decision,
            previous_ask_yield=previous_ask_yield,
        ), "other")
        before = copy.deepcopy(state)
        self.agent.connection.close()

        with self.assertRaises(sqlite3.ProgrammingError):
            self.agent.respond("session", fallback_message, 2, 10)

        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
