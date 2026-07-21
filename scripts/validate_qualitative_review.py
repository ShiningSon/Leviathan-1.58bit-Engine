#!/usr/bin/env python3
"""Validate paired v0.9.1 qualitative outputs and their human-review worksheet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .run_qualitative_holdout import (
        CSV_FIELDS,
        EXPECTED_CATEGORIES,
        EXPECTED_QUALITATIVE_COUNT,
        REVIEW_FIELDS,
        load_qualitative_prompts,
        sha256_text,
    )
    from .validate_benchmark_submission import find_private_fields
except ImportError:
    from run_qualitative_holdout import (
        CSV_FIELDS,
        EXPECTED_CATEGORIES,
        EXPECTED_QUALITATIVE_COUNT,
        REVIEW_FIELDS,
        load_qualitative_prompts,
        sha256_text,
    )
    from validate_benchmark_submission import find_private_fields


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SECRET_RE = re.compile(
    r"(?:\bhf_[A-Za-z0-9]{20,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|"
    r"\b(?:HF_TOKEN|HUGGINGFACE_TOKEN|API_KEY|PASSWORD|SECRET)\s*=\s*\S+)",
    re.IGNORECASE,
)
VERDICTS = {"pass", "concern", "fail"}
EQUIVALENCE = {"yes", "partial", "no"}


def is_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def find_secret_strings(value: object, prefix: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            errors.extend(find_secret_strings(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(find_secret_strings(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and SECRET_RE.search(value):
        errors.append(f"possible credential in {prefix}")
    return errors


def expected_by_id(prompts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in prompts}


def validate_output_record(record: object, label: str) -> list[str]:
    if not isinstance(record, dict):
        return [f"{label} must be an object"]
    errors: list[str] = []
    expected_fields = {"mode", "output", "output_sha256", "process_returncode", "parse_ok"}
    if set(record) != expected_fields:
        errors.append(f"{label} fields do not match the expected contract")
    mode = record.get("mode")
    try:
        float(mode)
    except (TypeError, ValueError):
        errors.append(f"{label}.mode must be numeric text")
    output = record.get("output")
    if not isinstance(output, str) or not output.strip():
        errors.append(f"{label}.output must be nonempty")
    digest = record.get("output_sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        errors.append(f"{label}.output_sha256 is invalid")
    elif isinstance(output, str) and digest != sha256_text(output):
        errors.append(f"{label}.output_sha256 does not match output")
    if record.get("process_returncode") != 0:
        errors.append(f"{label}.process_returncode must be 0")
    if record.get("parse_ok") is not True:
        errors.append(f"{label}.parse_ok must be true")
    return errors


def validate_outputs(data: dict[str, Any], prompts: list[dict[str, Any]], prompt_path: Path) -> list[str]:
    errors: list[str] = []
    expected_top = {
        "schema_version",
        "review_status",
        "generated_at_utc",
        "engine_commit",
        "model",
        "prompt_set",
        "settings",
        "prompts",
    }
    if set(data) != expected_top:
        errors.append("output JSON top-level fields do not match the expected contract")
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if data.get("review_status") != "unreviewed":
        errors.append("generated output JSON must remain marked unreviewed")
    if not is_utc_timestamp(data.get("generated_at_utc")):
        errors.append("generated_at_utc must be a timezone-aware timestamp")
    if not isinstance(data.get("engine_commit"), str) or not COMMIT_RE.fullmatch(data["engine_commit"]):
        errors.append("engine_commit must be a full lowercase Git SHA")

    model = data.get("model")
    if not isinstance(model, dict):
        errors.append("model must be an object")
    else:
        if set(model) != {"repository", "revision", "metadata_identifier"}:
            errors.append("model fields do not match the expected contract")
        if not isinstance(model.get("repository"), str) or not REPO_RE.fullmatch(model["repository"]):
            errors.append("model.repository must use owner/repository format")
        if not isinstance(model.get("revision"), str) or not REVISION_RE.fullmatch(model["revision"]):
            errors.append("model.revision must be a pinned lowercase hex revision")
        if not isinstance(model.get("metadata_identifier"), str) or not model["metadata_identifier"].strip():
            errors.append("model.metadata_identifier must be nonempty")

    prompt_set = data.get("prompt_set")
    if not isinstance(prompt_set, dict):
        errors.append("prompt_set must be an object")
    else:
        if set(prompt_set) != {"file_name", "sha256", "qualitative_count"}:
            errors.append("prompt_set fields do not match the expected contract")
        if prompt_set.get("file_name") != prompt_path.name:
            errors.append("prompt_set.file_name does not match the source prompt file")
        source_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        if prompt_set.get("sha256") != source_hash:
            errors.append("prompt_set.sha256 does not match the source prompt file")
        if prompt_set.get("qualitative_count") != EXPECTED_QUALITATIVE_COUNT:
            errors.append(f"prompt_set.qualitative_count must be {EXPECTED_QUALITATIVE_COUNT}")

    settings = data.get("settings")
    if not isinstance(settings, dict):
        errors.append("settings must be an object")
    else:
        expected_settings = {
            "architecture",
            "prompt_template",
            "max_new",
            "modes",
            "sparse_min_density",
            "no_top_k_sort",
            "sparse_scope",
            "top_k_select",
            "engine_file",
            "model_package",
            "binary_file",
            "metadata_file",
        }
        if set(settings) != expected_settings:
            errors.append("settings fields do not match the expected contract")
        modes = settings.get("modes")
        if not isinstance(modes, list) or len(modes) != 2:
            errors.append("settings.modes must contain dense and one Top-K mode")
        else:
            try:
                values = [float(mode) for mode in modes]
            except (TypeError, ValueError):
                errors.append("settings.modes must be numeric text")
            else:
                if sum(value == 0 for value in values) != 1 or sum(value > 0 for value in values) != 1:
                    errors.append("settings.modes must contain one dense and one positive Top-K mode")

    records = data.get("prompts")
    if not isinstance(records, list) or len(records) != EXPECTED_QUALITATIVE_COUNT:
        errors.append(f"prompts must contain exactly {EXPECTED_QUALITATIVE_COUNT} records")
        records = records if isinstance(records, list) else []
    expected = expected_by_id(prompts)
    expected_order = [item["id"] for item in prompts]
    seen: set[str] = set()
    categories: set[str] = set()
    for index, record in enumerate(records):
        label = f"prompts[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{label} must be an object")
            continue
        if set(record) != {"id", "category", "prompt", "review_criteria", "dense", "sparse"}:
            errors.append(f"{label} fields do not match the expected contract")
        prompt_id = record.get("id")
        if not isinstance(prompt_id, str) or prompt_id not in expected:
            errors.append(f"{label}.id is unexpected")
            continue
        if prompt_id in seen:
            errors.append(f"duplicate prompt id: {prompt_id}")
        seen.add(prompt_id)
        source = expected[prompt_id]
        category = record.get("category")
        if isinstance(category, str):
            categories.add(category)
        if category != source["category"]:
            errors.append(f"{label}.category does not match source")
        if record.get("prompt") != source["prompt"]:
            errors.append(f"{label}.prompt does not match source")
        if record.get("review_criteria") != source["review_criteria"]:
            errors.append(f"{label}.review_criteria does not match source")
        errors.extend(validate_output_record(record.get("dense"), f"{label}.dense"))
        errors.extend(validate_output_record(record.get("sparse"), f"{label}.sparse"))
        if isinstance(settings, dict) and isinstance(settings.get("modes"), list):
            modes = settings["modes"]
            if len(modes) == 2:
                if isinstance(record.get("dense"), dict) and record["dense"].get("mode") != modes[0]:
                    errors.append(f"{label}.dense.mode does not match settings.modes")
                if isinstance(record.get("sparse"), dict) and record["sparse"].get("mode") != modes[1]:
                    errors.append(f"{label}.sparse.mode does not match settings.modes")
    if seen != set(expected):
        errors.append("prompt IDs do not exactly match the expected 35 IDs")
    record_order = [record.get("id") for record in records if isinstance(record, dict)]
    if record_order != expected_order:
        errors.append("prompt records do not follow deterministic source order")
    if categories != EXPECTED_CATEGORIES:
        errors.append("prompt records do not cover all seven categories")
    errors.extend(find_private_fields(data))
    errors.extend(find_secret_strings(data))
    return errors


def load_worksheet(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = reader.fieldnames or []
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"cannot load worksheet {path}: {exc}") from exc
    return rows, fields


def validate_worksheet(
    rows: list[dict[str, str]],
    fields: list[str],
    outputs: dict[str, Any],
    *,
    expect_unreviewed: bool,
) -> list[str]:
    errors: list[str] = []
    if fields != list(CSV_FIELDS):
        errors.append("worksheet columns do not match the expected deterministic order")
    if len(rows) != EXPECTED_QUALITATIVE_COUNT:
        errors.append(f"worksheet must contain exactly {EXPECTED_QUALITATIVE_COUNT} rows")
    output_by_id = {item["id"]: item for item in outputs.get("prompts", []) if isinstance(item, dict)}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        label = f"worksheet row {index + 2}"
        prompt_id = row.get("prompt_id", "")
        if prompt_id in seen:
            errors.append(f"duplicate worksheet prompt id: {prompt_id}")
        seen.add(prompt_id)
        source = output_by_id.get(prompt_id)
        if source is None:
            errors.append(f"{label} has an unexpected prompt id")
            continue
        expected_values = {
            "category": source["category"],
            "prompt": source["prompt"],
            "review_criteria_json": json.dumps(
                source["review_criteria"], ensure_ascii=False, separators=(",", ":")
            ),
            "dense_mode": source["dense"]["mode"],
            "dense_output": source["dense"]["output"],
            "dense_sha256": source["dense"]["output_sha256"],
            "sparse_mode": source["sparse"]["mode"],
            "sparse_output": source["sparse"]["output"],
            "sparse_sha256": source["sparse"]["output_sha256"],
        }
        for field, expected in expected_values.items():
            if row.get(field) != expected:
                errors.append(f"{label}.{field} does not match qualitative_outputs.json")
        if expect_unreviewed:
            for field in REVIEW_FIELDS:
                if row.get(field, "").strip():
                    errors.append(f"{label}.{field} must be blank in unreviewed mode")
        else:
            if row.get("dense_verdict", "").strip().lower() not in VERDICTS:
                errors.append(f"{label}.dense_verdict must be pass, concern, or fail")
            if row.get("sparse_verdict", "").strip().lower() not in VERDICTS:
                errors.append(f"{label}.sparse_verdict must be pass, concern, or fail")
            if row.get("semantic_equivalence", "").strip().lower() not in EQUIVALENCE:
                errors.append(f"{label}.semantic_equivalence must be yes, partial, or no")
            if not row.get("reviewer", "").strip():
                errors.append(f"{label}.reviewer is required in completed mode")
    if seen != set(output_by_id):
        errors.append("worksheet IDs do not exactly match qualitative output IDs")
    output_order = [item.get("id") for item in outputs.get("prompts", []) if isinstance(item, dict)]
    if [row.get("prompt_id") for row in rows] != output_order:
        errors.append("worksheet rows do not follow qualitative output order")
    errors.extend(find_private_fields(rows))
    errors.extend(find_secret_strings(rows))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("benchmarks/prompts_v091_holdout.json"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--expect-unreviewed", action="store_true")
    mode.add_argument("--expect-completed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        prompts = load_qualitative_prompts(args.prompts)
        outputs = load_json(args.outputs)
        rows, fields = load_worksheet(args.worksheet)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1
    errors = validate_outputs(outputs, prompts, args.prompts)
    errors.extend(
        validate_worksheet(
            rows,
            fields,
            outputs,
            expect_unreviewed=args.expect_unreviewed,
        )
    )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        print(f"Qualitative review validation failed with {len(errors)} issue(s).")
        return 1
    mode = "unreviewed" if args.expect_unreviewed else "completed human review"
    print(f"[OK] {len(rows)} paired qualitative rows; mode: {mode}")
    print("Qualitative review validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
