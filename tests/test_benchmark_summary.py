from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.validate_benchmark_summary import percentage_delta, validate_summary


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "benchmarks" / "results" / "scaling_summary.json"


class BenchmarkSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    def test_percentage_delta(self) -> None:
        self.assertAlmostEqual(percentage_delta(110.0, 100.0), 10.0)
        self.assertAlmostEqual(percentage_delta(90.0, 100.0), -10.0)

    def test_canonical_summary_validates(self) -> None:
        self.assertEqual(validate_summary(self.summary), [])

    def test_mismatched_delta_fails(self) -> None:
        changed = copy.deepcopy(self.summary)
        changed["records"][0]["tokens_per_sec_delta_percent"] = 99.0
        errors = validate_summary(changed)
        self.assertTrue(any("tokens/sec delta mismatch" in error for error in errors))

    def test_confirmed_mode_cannot_reduce_qa(self) -> None:
        changed = copy.deepcopy(self.summary)
        changed["records"][0]["sparse_qa_pass"] = 900
        errors = validate_summary(changed)
        self.assertTrue(any("lower QA rate" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
