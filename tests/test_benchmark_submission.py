from __future__ import annotations

import copy
import hashlib
import json
import statistics
import tempfile
import unittest
from pathlib import Path

from scripts.export_benchmark_submission import export_submission
from scripts.validate_benchmark_submission import validate_submission_data


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "benchmark_submission_valid.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class BenchmarkSubmissionValidationTests(unittest.TestCase):
    def test_committed_fixture_is_valid(self) -> None:
        self.assertEqual(validate_submission_data(load_fixture()), [])

    def test_rejects_inconsistent_aggregate(self) -> None:
        submission = load_fixture()
        submission["modes"][0]["latency_ms"]["mean"] = 123.0
        errors = validate_submission_data(submission)
        self.assertTrue(any("latency_ms.mean is inconsistent" in error for error in errors))

    def test_rejects_private_machine_field(self) -> None:
        submission = load_fixture()
        submission["hardware"]["hostname"] = "private-machine"
        errors = validate_submission_data(submission)
        self.assertTrue(any("private machine-identifying field" in error for error in errors))

    def test_rejects_absolute_filesystem_path(self) -> None:
        submission = load_fixture()
        submission["hardware"]["benchmark_command"] = r"python C:\work\engine.py"
        errors = validate_submission_data(submission)
        self.assertTrue(any("private absolute path" in error for error in errors))

    def test_recommended_mode_must_preserve_dense_qa(self) -> None:
        submission = load_fixture()
        candidate = submission["modes"][1]
        candidate["qa"] = {"numerator": 3, "denominator": 4, "rate": 0.75}
        errors = validate_submission_data(submission)
        self.assertIn("recommended mode must preserve or improve dense QA rate", errors)


class BenchmarkSubmissionExportTests(unittest.TestCase):
    def test_export_recalculates_metrics_and_validates(self) -> None:
        prompts = [
            {"prompt": "Prompt A", "expected": ["A"]},
            {"prompt": "Prompt B", "expected": ["B"]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            prompts_path = Path(directory) / "prompts.json"
            prompts_path.write_text(json.dumps(prompts), encoding="utf-8")
            prompt_hash = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
            results = self._results(prompts, prompt_hash)
            hardware = copy.deepcopy(load_fixture()["hardware"])
            submission = export_submission(
                results,
                hardware,
                prompts_path,
                recommended_mode="0.10",
                caveats=["Test fixture only."],
            )
        self.assertEqual(validate_submission_data(submission), [])
        self.assertEqual(submission["modes"][0]["latency_ms"]["samples"], [100.0, 102.0, 98.0, 100.0])
        self.assertTrue(submission["modes"][1]["recommended"])

    def test_export_rejects_incomplete_mode(self) -> None:
        prompts = [{"prompt": "Prompt A", "expected": ["A"]}]
        with tempfile.TemporaryDirectory() as directory:
            prompts_path = Path(directory) / "prompts.json"
            prompts_path.write_text(json.dumps(prompts), encoding="utf-8")
            prompt_hash = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
            results = self._results(prompts, prompt_hash)
            results["modes"][0]["runs"][0]["process"]["parse_ok"] = False
            hardware = copy.deepcopy(load_fixture()["hardware"])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                export_submission(
                    results,
                    hardware,
                    prompts_path,
                    recommended_mode=None,
                    caveats=[],
                )

    @staticmethod
    def _results(prompts: list[dict], prompt_hash: str) -> dict:
        sample_sets = {
            "0": ([100.0, 102.0, 98.0, 100.0], [10.0, 11.0, 9.0, 10.0]),
            "0.10": ([90.0, 92.0, 88.0, 90.0], [11.0, 12.0, 10.0, 11.0]),
        }
        mode_results = []
        for mode, (latencies, speeds) in sample_sets.items():
            runs = []
            for repeat_index in range(2):
                prompt_results = []
                for prompt_index, prompt in enumerate(prompts):
                    sample_index = repeat_index * len(prompts) + prompt_index
                    prompt_results.append(
                        {
                            "prompt": prompt["prompt"],
                            "output": prompt["expected"][0],
                            "latency_ms": latencies[sample_index],
                            "tokens_per_sec": speeds[sample_index],
                            "qa_pass": True,
                        }
                    )
                runs.append(
                    {
                        "warmup": False,
                        "repeat_index": repeat_index + 1,
                        "process": {
                            "returncode": 0,
                            "parse_ok": True,
                            "prompt_results": prompt_results,
                        },
                    }
                )
            mode_results.append(
                {
                    "mode": mode,
                    "runs": runs,
                    "summary": {
                        "mode": mode,
                        "measured_prompt_count": 4,
                        "qa_pass_count": 4,
                        "qa_pass_rate": 1.0,
                        "avg_latency_ms": statistics.fmean(latencies),
                        "avg_tokens_per_sec": statistics.fmean(speeds),
                        "process_failures": 0,
                        "parse_failures": 0,
                    },
                }
            )
        return {
            "model": "leviathan_mlgru_200m_instruct_v09a",
            "metadata": {
                "engine_commit": "2eb51fa7feb1f9502690fc33fcb9d89474a74644",
                "model_repo": "ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a",
                "model_revision": "116a857bdaf2a1118d479d52aedba7e65cbff960",
                "model_metadata_identifier": "leviathan_mlgru_200m_instruct_v09a",
                "prompt_file_sha256": prompt_hash,
            },
            "settings": {
                "architecture": "mlgru",
                "prompt_template": "qa",
                "max_new": 80,
                "repeat": 2,
                "warmup": 1,
                "mode_schedule": "interleave",
                "sparse_min_density": 0.6,
                "no_top_k_sort": True,
                "top_k_select": "histogram",
                "sparse_scope": "down",
            },
            "prompts": prompts,
            "modes": mode_results,
        }


if __name__ == "__main__":
    unittest.main()
