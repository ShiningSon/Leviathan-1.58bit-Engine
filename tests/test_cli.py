from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import benchmark_engine


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_benchmark_sparse_flags_parse(self) -> None:
        argv = [
            "benchmark_engine.py",
            "--model-dir",
            "model",
            "--prompts",
            "prompts.json",
            "--modes",
            "0",
            "0.10",
            "--mode-schedule",
            "interleave",
            "--sparse-min-density",
            "0.6",
            "--no-top-k-sort",
            "--sparse-scope",
            "down",
            "--top-k-select",
            "histogram",
        ]
        with patch.object(sys, "argv", argv):
            args = benchmark_engine.parse_args()
        self.assertEqual(args.modes, ["0", "0.10"])
        self.assertEqual(args.mode_schedule, "interleave")
        self.assertEqual(args.sparse_min_density, 0.6)
        self.assertTrue(args.no_top_k_sort)
        self.assertEqual(args.sparse_scope, "down")
        self.assertEqual(args.top_k_select, "histogram")

    def test_benchmark_defaults_to_dense_only(self) -> None:
        argv = [
            "benchmark_engine.py",
            "--model-dir",
            "model",
            "--prompts",
            "prompts.json",
        ]
        with patch.object(sys, "argv", argv):
            args = benchmark_engine.parse_args()
        self.assertEqual(args.modes, ["0"])
        self.assertEqual(args.out_dir, "benchmark_runs/latest")

    def test_engine_exposes_sparse_flags(self) -> None:
        source = (ROOT / "engine.py").read_text(encoding="utf-8")
        for flag in (
            "--sparse-min-density",
            "--no-top-k-sort",
            "--top-k-select",
            "--sparse-scope",
        ):
            self.assertIn(flag, source)
        self.assertIn('choices=["nth", "histogram"]', source)
        self.assertIn('choices=["all", "ffn", "down", "none"]', source)


if __name__ == "__main__":
    unittest.main()
