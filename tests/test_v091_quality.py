from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from scripts import benchmark_engine


ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_PATH = ROOT / "benchmarks" / "prompts_v091_holdout.json"
REGRESSION_PATH = ROOT / "benchmarks" / "prompts_v02b_qa.json"
CATEGORIES = {
    "leviathan_concepts",
    "paraphrase_robustness",
    "negation_and_limitation_awareness",
    "package_runtime_usage",
    "sparse_mode_caveats",
    "simple_tinystories_comprehension",
    "malformed_or_ambiguous_requests",
}


class V091QualitySetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))

    def test_holdout_has_at_least_one_hundred_prompts(self) -> None:
        exact = self.data["exact_match_regression"]
        qualitative = self.data["qualitative_review"]
        self.assertGreaterEqual(len(exact) + len(qualitative), 100)
        self.assertEqual(len(exact), 70)
        self.assertEqual(len(qualitative), 35)

    def test_both_sections_cover_every_category(self) -> None:
        exact_counts = Counter(item["category"] for item in self.data["exact_match_regression"])
        qualitative_counts = Counter(item["category"] for item in self.data["qualitative_review"])
        self.assertEqual(set(exact_counts), CATEGORIES)
        self.assertEqual(set(qualitative_counts), CATEGORIES)
        self.assertTrue(all(count >= 10 for count in exact_counts.values()))
        self.assertTrue(all(count >= 5 for count in qualitative_counts.values()))

    def test_prompts_and_ids_are_unique(self) -> None:
        items = self.data["exact_match_regression"] + self.data["qualitative_review"]
        ids = [item["id"] for item in items]
        prompts = [item["prompt"].strip().casefold() for item in items]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(prompts), len(set(prompts)))

    def test_exact_prompts_do_not_copy_project_regression_prompts(self) -> None:
        existing = json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))
        existing_prompts = {item["prompt"].strip().casefold() for item in existing}
        holdout_prompts = {
            item["prompt"].strip().casefold() for item in self.data["exact_match_regression"]
        }
        self.assertFalse(existing_prompts & holdout_prompts)

    def test_exact_and_qualitative_contracts_are_separate(self) -> None:
        for item in self.data["exact_match_regression"]:
            rules = item["rules"]
            self.assertTrue(rules["required_terms"])
            self.assertIsInstance(rules["forbidden_terms"], list)
            self.assertGreater(rules["max_words"], 0)
            self.assertNotIn("review_criteria", item)
        for item in self.data["qualitative_review"]:
            self.assertGreaterEqual(len(item["review_criteria"]), 2)
            self.assertNotIn("rules", item)

    def test_runner_loads_only_exact_regression_items(self) -> None:
        prompts = benchmark_engine.load_prompts(HOLDOUT_PATH)
        self.assertEqual(len(prompts), 70)
        self.assertIn("Leviathan", prompts[0]["expected"])
        self.assertIn("general assistant", prompts[0]["forbidden"])
        self.assertEqual(prompts[0]["max_words"], 40)

    def test_disclaimer_rejects_comprehensive_evaluation_claim(self) -> None:
        self.assertIn("not a comprehensive", self.data["disclaimer"].casefold())


if __name__ == "__main__":
    unittest.main()
