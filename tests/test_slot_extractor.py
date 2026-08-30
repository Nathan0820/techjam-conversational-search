from __future__ import annotations

import copy
import unittest

from dialogue.slot_extractor import BudgetConstraint, extract_slots
from dialogue.state import SUPPORTED_SLOTS, SessionState


class SlotExtractorTest(unittest.TestCase):
    def test_empty_and_irrelevant_messages_return_empty_result(self) -> None:
        for message in ("", "   ", "Thanks, I will think about it."):
            result = extract_slots(message)
            self.assertEqual(set(result.slots), set(SUPPORTED_SLOTS))
            self.assertTrue(all(not values for values in result.slots.values()))
            self.assertEqual(result.revealed_text, [])

    def test_extracts_multiple_slot_types_and_multiple_values(self) -> None:
        result = extract_slots(
            "I want a black or navy blue cotton and polyester shirt in size XL for running."
        )

        self.assertEqual(result.slots["color"], ["black", "navy blue"])
        self.assertEqual(result.slots["material"], ["cotton", "polyester"])
        self.assertEqual(result.slots["size"], ["XL"])
        self.assertIn("Shirts", result.slots["category"])
        self.assertEqual(result.slots["use_case"], ["running"])

    def test_normalizes_structured_values_but_preserves_raw_casing(self) -> None:
        result = extract_slots("A key requirement is: GREY Wool.")

        self.assertEqual(result.slots["color"], ["gray"])
        self.assertEqual(result.slots["material"], ["wool"])
        self.assertEqual(result.revealed_text, ["GREY Wool"])

    def test_detects_catalog_brand_and_category(self) -> None:
        result = extract_slots("I need a Columbia T-Shirts option.")

        self.assertIn("Columbia", result.slots["brand"])
        self.assertIn("T-Shirts", result.slots["category"])

    def test_extracts_sizes_with_punctuation_and_deduplicates(self) -> None:
        result = extract_slots("Size: xl, or XL; shoe size 8.5!")

        self.assertEqual(result.slots["size"], ["XL", "8.5"])

    def test_extracts_features_styles_and_use_cases(self) -> None:
        result = extract_slots(
            "For winter hiking, I want a casual, waterproof, breathable boot "
            "with drawstring closure and rubber sole."
        )

        self.assertEqual(result.slots["style"], ["casual"])
        self.assertEqual(
            result.slots["feature"],
            ["waterproof", "breathable", "drawstring closure", "rubber sole"],
        )
        self.assertEqual(result.slots["use_case"], ["winter", "hiking"])
        self.assertIn("Boots", result.slots["category"])

    def test_budget_upper_bound(self) -> None:
        result = extract_slots("Keep it under SGD 100.")
        self.assertEqual(
            result.slots["budget"],
            [BudgetConstraint(maximum=100, currency="SGD")],
        )

    def test_budget_lower_bound(self) -> None:
        result = extract_slots("I am looking above $80")
        self.assertEqual(
            result.slots["budget"],
            [BudgetConstraint(minimum=80, currency="$")],
        )

    def test_budget_range(self) -> None:
        result = extract_slots("My budget is between USD 80 and USD 120.")
        self.assertEqual(
            result.slots["budget"],
            [BudgetConstraint(minimum=80, maximum=120, currency="USD")],
        )

    def test_approximate_budget(self) -> None:
        result = extract_slots("Budget around $49.50")
        self.assertEqual(
            result.slots["budget"],
            [BudgetConstraint(49.5, 49.5, "$", approximate=True)],
        )

    def test_evaluator_style_reveals_preserve_exact_phrases(self) -> None:
        result = extract_slots(
            "For that, what matters is: Warm And Comfortable Women’s Winter Boots; "
            "Fully Lined With Soft Faux Fur To Keep The Feet Warm All Day Long."
        )

        self.assertEqual(result.slots["material"], ["faux fur"])
        self.assertEqual(
            result.revealed_text,
            [
                "Warm And Comfortable Women’s Winter Boots",
                "Fully Lined With Soft Faux Fur To Keep The Feet Warm All Day Long",
            ],
        )

    def test_duplicate_mentions_do_not_duplicate_values(self) -> None:
        result = extract_slots("Black, black, and BLACK cotton cotton.")
        self.assertEqual(result.slots["color"], ["black"])
        self.assertEqual(result.slots["material"], ["cotton"])

    def test_extraction_does_not_mutate_session_state(self) -> None:
        state = SessionState(session_id="session")
        state.set_constraint("color", ["red"])
        state.message_history.append({"role": "user", "content": "older message"})
        before = copy.deepcopy(state)

        extract_slots("I want a blue cotton shirt under $30")

        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
