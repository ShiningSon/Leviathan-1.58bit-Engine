from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import benchmark_engine, run_qualitative_holdout, validate_qualitative_review


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "benchmarks" / "prompts_v091_holdout.json"


def fixture_data() -> dict[str, object]:
    prompts = run_qualitative_holdout.load_qualitative_prompts(PROMPTS_PATH)
    records = []
    for item in prompts:
        dense_output = f"Dense response for {item['id']}."
        sparse_output = f"Sparse response for {item['id']}."
        records.append(
            {
                **item,
                "dense": {
                    "mode": "0",
                    "output": dense_output,
                    "output_sha256": run_qualitative_holdout.sha256_text(dense_output),
                    "process_returncode": 0,
                    "parse_ok": True,
                },
                "sparse": {
                    "mode": "0.10",
                    "output": sparse_output,
                    "output_sha256": run_qualitative_holdout.sha256_text(sparse_output),
                    "process_returncode": 0,
                    "parse_ok": True,
                },
            }
        )
    return {
        "schema_version": 1,
        "review_status": "unreviewed",
        "generated_at_utc": "2026-07-21T00:00:00+00:00",
        "engine_commit": "a" * 40,
        "model": {
            "repository": "owner/model",
            "revision": "b" * 40,
            "metadata_identifier": "fixture_model",
        },
        "prompt_set": {
            "file_name": PROMPTS_PATH.name,
            "sha256": hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest(),
            "qualitative_count": 35,
        },
        "settings": {
            "architecture": "mlgru",
            "prompt_template": "qa",
            "max_new": 80,
            "modes": ["0", "0.10"],
            "sparse_min_density": 0.6,
            "no_top_k_sort": True,
            "sparse_scope": "down",
            "top_k_select": "histogram",
            "engine_file": "engine.py",
            "model_package": "fixture_model",
            "binary_file": "fixture.bin",
            "metadata_file": "fixture_meta.json",
        },
        "prompts": records,
    }


def worksheet(data: dict[str, object]) -> tuple[list[dict[str, str]], list[str]]:
    reader = csv.DictReader(io.StringIO(run_qualitative_holdout.render_csv(data)))
    return list(reader), reader.fieldnames or []


class QualitativeHoldoutTests(unittest.TestCase):
    def test_prompt_loader_and_renderers_keep_exactly_35_paired_rows(self) -> None:
        prompts = run_qualitative_holdout.load_qualitative_prompts(PROMPTS_PATH)
        data = fixture_data()
        rows, fields = worksheet(data)
        markdown = run_qualitative_holdout.render_markdown(data)

        self.assertEqual(len(prompts), 35)
        self.assertEqual({item["category"] for item in prompts}, run_qualitative_holdout.EXPECTED_CATEGORIES)
        self.assertEqual(len(rows), 35)
        self.assertEqual(fields, list(run_qualitative_holdout.CSV_FIELDS))
        self.assertEqual(markdown.count("\n## "), 35)
        self.assertIn("Status: **unreviewed**", markdown)

    def test_loader_rejects_duplicate_prompt_ids(self) -> None:
        source = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
        source["qualitative_review"][1]["id"] = source["qualitative_review"][0]["id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate qualitative prompt id"):
                run_qualitative_holdout.load_qualitative_prompts(path)

    def test_generate_outputs_pairs_modes_without_automatic_verdicts(self) -> None:
        prompts = run_qualitative_holdout.load_qualitative_prompts(PROMPTS_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "fixture_model"
            model_dir.mkdir()
            (model_dir / "fixture.bin").write_bytes(b"fixture")
            (model_dir / "fixture_meta.json").write_text(
                json.dumps({"model_name": "fixture_model"}), encoding="utf-8"
            )
            (root / "engine.py").write_text("# fixture\n", encoding="utf-8")
            args = SimpleNamespace(
                model_dir="fixture_model",
                engine="engine.py",
                prompts=str(PROMPTS_PATH),
                architecture="mlgru",
                prompt_template="qa",
                max_new=80,
                modes=["0", "0.10"],
                sparse_min_density=0.6,
                no_top_k_sort=True,
                sparse_scope="down",
                top_k_select="histogram",
                model_repo="owner/model",
                model_revision="b" * 40,
                bin_name=None,
                meta_name=None,
                python="python",
                timeout=60,
            )

            def process_for(mode: str) -> dict[str, object]:
                return {
                    "returncode": 0,
                    "parse_ok": True,
                    "prompt_results": [
                        {
                            "prompt": item["prompt"],
                            "output": f"Output {mode} for {item['id']}",
                            "parse_error": None,
                        }
                        for item in prompts
                    ],
                }

            with (
                patch.object(
                    benchmark_engine,
                    "run_engine",
                    side_effect=[process_for("0"), process_for("0.10")],
                ),
                patch.object(benchmark_engine, "git_commit", return_value="a" * 40),
            ):
                data = run_qualitative_holdout.generate_outputs(args, root=root)

        self.assertEqual(len(data["prompts"]), 35)
        self.assertEqual(data["review_status"], "unreviewed")
        self.assertNotIn(str(root), json.dumps(data))
        for item in data["prompts"]:
            self.assertNotIn("qa_pass", item["dense"])
            self.assertNotIn("qa_pass", item["sparse"])

    def test_validator_accepts_consistent_unreviewed_outputs(self) -> None:
        data = fixture_data()
        rows, fields = worksheet(data)
        errors = validate_qualitative_review.validate_outputs(data, run_qualitative_holdout.load_qualitative_prompts(PROMPTS_PATH), PROMPTS_PATH)
        errors.extend(
            validate_qualitative_review.validate_worksheet(
                rows, fields, data, expect_unreviewed=True
            )
        )
        self.assertEqual(errors, [])

    def test_validator_rejects_tampered_output_hash(self) -> None:
        data = fixture_data()
        data["prompts"][0]["dense"]["output"] = "Tampered"
        errors = validate_qualitative_review.validate_outputs(
            data, run_qualitative_holdout.load_qualitative_prompts(PROMPTS_PATH), PROMPTS_PATH
        )
        self.assertTrue(any("does not match output" in error for error in errors))

    def test_validator_rejects_nondeterministic_prompt_order(self) -> None:
        data = fixture_data()
        data["prompts"][0], data["prompts"][1] = data["prompts"][1], data["prompts"][0]
        errors = validate_qualitative_review.validate_outputs(
            data, run_qualitative_holdout.load_qualitative_prompts(PROMPTS_PATH), PROMPTS_PATH
        )
        self.assertTrue(any("deterministic source order" in error for error in errors))

    def test_unreviewed_mode_rejects_filled_human_fields(self) -> None:
        data = fixture_data()
        rows, fields = worksheet(data)
        rows[0]["dense_verdict"] = "pass"
        errors = validate_qualitative_review.validate_worksheet(
            rows, fields, data, expect_unreviewed=True
        )
        self.assertTrue(any("must be blank" in error for error in errors))

    def test_completed_mode_requires_and_accepts_human_fields(self) -> None:
        data = fixture_data()
        rows, fields = worksheet(data)
        missing = validate_qualitative_review.validate_worksheet(
            copy.deepcopy(rows), fields, data, expect_unreviewed=False
        )
        self.assertTrue(any("reviewer is required" in error for error in missing))

        for row in rows:
            row["dense_verdict"] = "pass"
            row["sparse_verdict"] = "concern"
            row["semantic_equivalence"] = "partial"
            row["reviewer"] = "reviewer-1"
        self.assertEqual(
            validate_qualitative_review.validate_worksheet(
                rows, fields, data, expect_unreviewed=False
            ),
            [],
        )

    def test_validator_rejects_private_paths_in_shareable_data(self) -> None:
        data = fixture_data()
        private_path = "C:" + "\\Users\\Reviewer\\fixture.txt"
        data["prompts"][0]["dense"]["output"] = private_path
        data["prompts"][0]["dense"]["output_sha256"] = run_qualitative_holdout.sha256_text(
            data["prompts"][0]["dense"]["output"]
        )
        errors = validate_qualitative_review.validate_outputs(
            data, run_qualitative_holdout.load_qualitative_prompts(PROMPTS_PATH), PROMPTS_PATH
        )
        self.assertTrue(any("private absolute path" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
