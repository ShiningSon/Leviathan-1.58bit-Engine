from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
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
            "--model-repo",
            "owner/model",
            "--model-revision",
            "a" * 40,
        ]
        with patch.object(sys, "argv", argv):
            args = benchmark_engine.parse_args()
        self.assertEqual(args.modes, ["0", "0.10"])
        self.assertEqual(args.mode_schedule, "interleave")
        self.assertEqual(args.sparse_min_density, 0.6)
        self.assertTrue(args.no_top_k_sort)
        self.assertEqual(args.sparse_scope, "down")
        self.assertEqual(args.top_k_select, "histogram")
        self.assertEqual(args.model_repo, "owner/model")
        self.assertEqual(args.model_revision, "a" * 40)

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
        self.assertIsNone(args.model_repo)
        self.assertIsNone(args.model_revision)

    def test_summary_keeps_individual_timing_samples(self) -> None:
        runs = [
            {
                "warmup": False,
                "process": {
                    "returncode": 0,
                    "parse_ok": True,
                    "prompt_results": [
                        {
                            "latency_ms": 12.5,
                            "tokens_per_sec": 80.0,
                            "qa_pass": True,
                        }
                    ],
                },
            }
        ]
        summary = benchmark_engine.summarize_mode("0", runs)
        self.assertEqual(summary["latency_samples_ms"], [12.5])
        self.assertEqual(summary["tokens_per_sec_samples"], [80.0])

    def test_benchmark_main_records_reproducibility_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            model_dir = temp / "fixture_model"
            model_dir.mkdir()
            (model_dir / "fixture.bin").write_bytes(b"fixture")
            (model_dir / "fixture_meta.json").write_text(
                json.dumps({"model_name": "fixture_model"}), encoding="utf-8"
            )
            prompts_path = temp / "prompts.json"
            prompts_path.write_text(
                json.dumps([{"prompt": "Prompt?", "expected": ["answer"]}]),
                encoding="utf-8",
            )
            out_dir = temp / "output"
            argv = [
                "benchmark_engine.py",
                "--model-dir",
                str(model_dir),
                "--engine",
                str(ROOT / "engine.py"),
                "--prompts",
                str(prompts_path),
                "--repeat",
                "1",
                "--warmup",
                "0",
                "--model-repo",
                "owner/model",
                "--model-revision",
                "b" * 40,
                "--out-dir",
                str(out_dir),
            ]
            process = {
                "command": ["python", "engine.py"],
                "cwd": str(model_dir),
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "prompt_results": [
                    {
                        "prompt": "Prompt?",
                        "output": "answer",
                        "latency_ms": 10.0,
                        "tokens_per_sec": 100.0,
                        "qa_pass": True,
                        "matched": ["answer"],
                        "missing": [],
                        "forbidden_hit": [],
                        "word_count": 1,
                        "max_words": None,
                    }
                ],
                "parse_ok": True,
            }
            with (
                patch.object(sys, "argv", argv),
                patch.object(benchmark_engine, "run_engine", return_value=process),
                patch.object(benchmark_engine, "git_commit", return_value="a" * 40),
                patch.object(
                    benchmark_engine,
                    "thread_environment",
                    return_value={"OMP_NUM_THREADS": "4"},
                ),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(benchmark_engine.main(), 0)

            result = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(result["metadata"]["engine_commit"], "a" * 40)
            self.assertEqual(result["metadata"]["model_repo"], "owner/model")
            self.assertEqual(result["metadata"]["model_revision"], "b" * 40)
            self.assertEqual(result["metadata"]["model_metadata_identifier"], "fixture_model")
            self.assertEqual(
                result["metadata"]["prompt_file_sha256"],
                hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(result["metadata"]["thread_environment"], {"OMP_NUM_THREADS": "4"})
            self.assertIsNotNone(result["started_at_utc"])
            self.assertIsNotNone(result["completed_at_utc"])
            self.assertEqual(result["modes"][0]["summary"]["latency_samples_ms"], [10.0])
            self.assertEqual(result["modes"][0]["summary"]["tokens_per_sec_samples"], [100.0])

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
