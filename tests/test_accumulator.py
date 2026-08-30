from __future__ import annotations

import copy
import unittest

from dialogue.accumulator import accumulate_information
from dialogue.slot_extractor import BudgetConstraint, SlotExtraction, extract_slots
from dialogue.state import SessionState


class InformationAccumulatorTest(unittest.TestCase):
    def test_accumulates_one_extraction_into_empty_state(self) -> None:
        state = SessionState(session_id="session")
        extraction = extract_slots("A black cotton shirt under SGD 100")

        accumulate_information(state, extraction)

        self.assertEqual(state.slots["color"], ["black"])
        self.assertEqual(state.slots["material"], ["cotton"])
        self.assertIn("Shirts", state.slots["category"])
        self.assertEqual(
            state.slots["budget"],
            [BudgetConstraint(maximum=100, currency="SGD")],
        )

    def test_realistic_multi_turn_accumulation_preserves_order(self) -> None:
        state = SessionState(session_id="session")
        turn_one = SlotExtraction()
        turn_one.slots["category"] = ["Boots"]
        turn_one.slots["color"] = ["black"]
        turn_two = SlotExtraction()
        turn_two.slots["material"] = ["leather"]
        turn_two.slots["feature"] = ["waterproof"]
        turn_three = SlotExtraction()
        turn_three.slots["feature"] = ["breathable", "waterproof"]

        accumulate_information(state, turn_one)
        accumulate_information(state, turn_two)
        accumulate_information(state, turn_three)

        self.assertEqual(state.slots["color"], ["black"])
        self.assertIn("Boots", state.slots["category"])
        self.assertEqual(state.slots["material"], ["leather"])
        self.assertEqual(state.slots["feature"], ["waterproof", "breathable"])

    def test_preserves_existing_and_multiple_values_without_duplicates(self) -> None:
        state = SessionState(session_id="session")
        state.add_slot_values("color", ["black"])
        extraction = SlotExtraction()
        extraction.slots["color"] = ["black", "navy blue", "black"]
        extraction.slots["material"] = ["cotton", "polyester"]

        accumulate_information(state, extraction)

        self.assertEqual(state.slots["color"], ["black", "navy blue"])
        self.assertEqual(state.slots["material"], ["cotton", "polyester"])
        self.assertEqual(state.slots["size"], [])

    def test_accumulates_structured_budgets_without_stringifying(self) -> None:
        state = SessionState(session_id="session")
        first = BudgetConstraint(maximum=100, currency="SGD")
        second = BudgetConstraint(minimum=50, maximum=90, currency="SGD")

        extraction = SlotExtraction()
        extraction.slots["budget"] = [first]
        accumulate_information(state, extraction)
        extraction.slots["budget"] = [first, second]
        accumulate_information(state, extraction)

        self.assertEqual(state.slots["budget"], [first, second])
        self.assertIsInstance(state.slots["budget"][0], BudgetConstraint)

    def test_revealed_text_preserves_exact_first_seen_phrases(self) -> None:
        state = SessionState(session_id="session")
        first = SlotExtraction(revealed_text=["100% Leather", "Rubber sole"])
        second = SlotExtraction(revealed_text=["100% Leather", "leather"])

        accumulate_information(state, first)
        accumulate_information(state, second)

        self.assertEqual(
            state.revealed_text,
            ["100% Leather", "Rubber sole", "leather"],
        )

    def test_empty_extraction_changes_nothing(self) -> None:
        state = SessionState(session_id="session")
        state.add_slot_values("color", ["black"])
        state.revealed_text.append("Black")
        before = copy.deepcopy(state)

        accumulate_information(state, SlotExtraction())

        self.assertEqual(state, before)

    def test_rejects_unsupported_slot_before_mutating_state(self) -> None:
        state = SessionState(session_id="session")
        extraction = SlotExtraction(revealed_text=["raw"])
        extraction.slots["color"] = ["black"]
        extraction.slots["unsupported"] = ["value"]

        with self.assertRaisesRegex(ValueError, "unsupported extracted slot"):
            accumulate_information(state, extraction)

        self.assertEqual(state.slots["color"], [])
        self.assertEqual(state.revealed_text, [])

    def test_does_not_modify_other_dialogue_state(self) -> None:
        state = SessionState(session_id="session", intent="browsing")
        state.set_constraint("brand", ["Columbia"], strength="soft")
        state.set_constraint("material", ["cotton"], strength="hard")
        state.override_detected = True
        state.asked_attributes.add("color")
        state.last_ask_yielded = False
        state.message_history.append({"role": "user", "content": "earlier"})
        history_before = copy.deepcopy(state.message_history)

        extraction = SlotExtraction()
        extraction.slots["brand"] = ["Nike"]
        accumulate_information(state, extraction)

        self.assertEqual(state.intent, "browsing")
        self.assertEqual(state.slots["brand"], ["Columbia", "Nike"])
        self.assertEqual(state.hard_constraints, {"material"})
        self.assertEqual(state.soft_preferences, {"brand"})
        self.assertTrue(state.override_detected)
        self.assertEqual(state.asked_attributes, {"color"})
        self.assertFalse(state.last_ask_yielded)
        self.assertEqual(state.message_history, history_before)


if __name__ == "__main__":
    unittest.main()
