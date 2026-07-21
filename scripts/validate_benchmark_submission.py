#!/usr/bin/env python3
"""Validate a portable Leviathan benchmark submission without model weights."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
HF_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
PRIVATE_KEY_NAMES = {
    "credential",
    "credentials",
    "cwd",
    "filesystem_path",
    "home",
    "home_directory",
    "host",
    "host_name",
    "hostname",
    "ip",
    "ip_address",
    "mac",
    "mac_address",
    "secret",
    "token",
    "user",
    "user_name",
    "username",
    "working_directory",
}
ABSOLUTE_PRIVATE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"']+|(?<![:/.\w])/(?!/)[^\s\"']+)",
    re.IGNORECASE,
)
HARDWARE_FIELDS = {
    "schema_version",
    "timestamp_utc",
    "operating_system",
    "operating_system_version",
    "cpu_model",
    "cpu_vendor",
    "physical_cores",
    "logical_cores",
    "configured_threads",
    "ram_bytes",
    "compiler",
    "compiler_version",
    "python_version",
    "pytorch_version",
    "architecture",
    "model_repo",
    "model_revision",
    "engine_commit",
    "benchmark_command",
}
OPTIONAL_HARDWARE_STRINGS = {
    "operating_system_version",
    "cpu_model",
    "cpu_vendor",
    "compiler",
    "compiler_version",
    "pytorch_version",
}
OPTIONAL_HARDWARE_INTEGERS = {
    "physical_cores",
    "logical_cores",
    "configured_threads",
    "ram_bytes",
}


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def find_private_fields(value: object, prefix: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in PRIVATE_KEY_NAMES:
                errors.append(f"private machine-identifying field is not allowed: {path}")
            errors.extend(find_private_fields(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(find_private_fields(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and ABSOLUTE_PRIVATE_PATH_RE.search(value):
        errors.append(f"private absolute path is not allowed in {prefix}")
    return errors


def check_exact_fields(data: dict[str, Any], expected: set[str], label: str) -> list[str]:
    errors: list[str] = []
    missing = sorted(expected - set(data))
    extra = sorted(set(data) - expected)
    if missing:
        errors.append(f"{label} missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"{label} has unexpected fields: {', '.join(extra)}")
    return errors


def validate_hardware_manifest(data: object) -> list[str]:
    if not isinstance(data, dict):
        return ["hardware manifest must be an object"]

    errors = check_exact_fields(data, HARDWARE_FIELDS, "hardware manifest")
    if data.get("schema_version") != 1:
        errors.append("hardware.schema_version must be 1")
    if not is_utc_timestamp(data.get("timestamp_utc")):
        errors.append("hardware.timestamp_utc must be a timezone-aware ISO-8601 timestamp")

    for field in ("operating_system", "python_version", "architecture", "benchmark_command"):
        if not is_nonempty_string(data.get(field)):
            errors.append(f"hardware.{field} must be a nonempty string")
    for field in OPTIONAL_HARDWARE_STRINGS:
        value = data.get(field)
        if value is not None and not is_nonempty_string(value):
            errors.append(f"hardware.{field} must be null or a nonempty string")
    for field in OPTIONAL_HARDWARE_INTEGERS:
        value = data.get(field)
        if value is not None and not is_positive_integer(value):
            errors.append(f"hardware.{field} must be null or a positive integer")

    model_repo = data.get("model_repo")
    if not isinstance(model_repo, str) or not HF_REPO_RE.fullmatch(model_repo):
        errors.append("hardware.model_repo must be an owner/repository identifier")
    revision = data.get("model_revision")
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        errors.append("hardware.model_revision must be a pinned 40-64 character lowercase hex revision")
    commit = data.get("engine_commit")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        errors.append("hardware.engine_commit must be a full 40-character lowercase Git SHA")

    errors.extend(find_private_fields(data, "hardware"))
    return errors


def validate_metric(metric: object, label: str, expected_count: int) -> list[str]:
    if not isinstance(metric, dict):
        return [f"{label} must be an object"]
    errors = check_exact_fields(
        metric,
        {"samples", "mean", "median", "standard_deviation"},
        label,
    )
    samples = metric.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append(f"{label}.samples must be a nonempty array")
        return errors
    if len(samples) != expected_count:
        errors.append(f"{label}.samples must contain {expected_count} values, found {len(samples)}")
    if any(not is_finite_number(value) or float(value) <= 0 for value in samples):
        errors.append(f"{label}.samples must contain only positive finite numbers")
        return errors

    numeric = [float(value) for value in samples]
    expected = {
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "standard_deviation": statistics.pstdev(numeric),
    }
    for field, recalculated in expected.items():
        value = metric.get(field)
        if not is_finite_number(value):
            errors.append(f"{label}.{field} must be a finite number")
        elif not math.isclose(float(value), recalculated, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(
                f"{label}.{field} is inconsistent: recorded={value}, recalculated={recalculated}"
            )
    if is_finite_number(metric.get("standard_deviation")) and float(metric["standard_deviation"]) < 0:
        errors.append(f"{label}.standard_deviation must be nonnegative")
    return errors


def validate_submission_data(data: object) -> list[str]:
    if not isinstance(data, dict):
        return ["benchmark submission must be an object"]

    errors = check_exact_fields(
        data,
        {
            "schema_version",
            "created_at_utc",
            "hardware",
            "model",
            "engine_commit",
            "prompt_set",
            "benchmark",
            "modes",
            "caveats",
        },
        "submission",
    )
    if data.get("schema_version") != 1:
        errors.append("submission.schema_version must be 1")
    if not is_utc_timestamp(data.get("created_at_utc")):
        errors.append("submission.created_at_utc must be a timezone-aware ISO-8601 timestamp")

    hardware = data.get("hardware")
    errors.extend(validate_hardware_manifest(hardware))

    model = data.get("model")
    if not isinstance(model, dict):
        errors.append("model must be an object")
    else:
        errors.extend(
            check_exact_fields(
                model,
                {"identifier", "hf_repo", "hf_revision", "metadata_identifier"},
                "model",
            )
        )
        if not is_nonempty_string(model.get("identifier")):
            errors.append("model.identifier must be a nonempty string")
        if not isinstance(model.get("hf_repo"), str) or not HF_REPO_RE.fullmatch(model["hf_repo"]):
            errors.append("model.hf_repo must be an owner/repository identifier")
        if not isinstance(model.get("hf_revision"), str) or not REVISION_RE.fullmatch(model["hf_revision"]):
            errors.append("model.hf_revision must be a pinned lowercase hex revision")
        metadata_identifier = model.get("metadata_identifier")
        if metadata_identifier is not None and not is_nonempty_string(metadata_identifier):
            errors.append("model.metadata_identifier must be null or a nonempty string")

    engine_commit = data.get("engine_commit")
    if not isinstance(engine_commit, str) or not COMMIT_RE.fullmatch(engine_commit):
        errors.append("engine_commit must be a full lowercase Git SHA")
    if isinstance(hardware, dict) and hardware.get("engine_commit") != engine_commit:
        errors.append("engine_commit must match hardware.engine_commit")
    if isinstance(hardware, dict) and isinstance(model, dict):
        if hardware.get("model_repo") != model.get("hf_repo"):
            errors.append("model.hf_repo must match hardware.model_repo")
        if hardware.get("model_revision") != model.get("hf_revision"):
            errors.append("model.hf_revision must match hardware.model_revision")

    prompt_set = data.get("prompt_set")
    prompt_count = 0
    if not isinstance(prompt_set, dict):
        errors.append("prompt_set must be an object")
    else:
        errors.extend(
            check_exact_fields(prompt_set, {"file_name", "sha256", "prompt_count"}, "prompt_set")
        )
        if not is_nonempty_string(prompt_set.get("file_name")):
            errors.append("prompt_set.file_name must be a nonempty string")
        if not isinstance(prompt_set.get("sha256"), str) or not SHA256_RE.fullmatch(prompt_set["sha256"]):
            errors.append("prompt_set.sha256 must be a lowercase SHA-256 digest")
        if not is_positive_integer(prompt_set.get("prompt_count")):
            errors.append("prompt_set.prompt_count must be a positive integer")
        else:
            prompt_count = int(prompt_set["prompt_count"])

    benchmark = data.get("benchmark")
    repeats = 0
    if not isinstance(benchmark, dict):
        errors.append("benchmark must be an object")
    else:
        errors.extend(
            check_exact_fields(
                benchmark,
                {"architecture", "prompt_template", "max_new", "warmups", "repeats", "mode_schedule"},
                "benchmark",
            )
        )
        for field in ("architecture", "prompt_template"):
            if not is_nonempty_string(benchmark.get(field)):
                errors.append(f"benchmark.{field} must be a nonempty string")
        if not is_positive_integer(benchmark.get("max_new")):
            errors.append("benchmark.max_new must be a positive integer")
        if not is_nonnegative_integer(benchmark.get("warmups")):
            errors.append("benchmark.warmups must be a nonnegative integer")
        if not is_positive_integer(benchmark.get("repeats")):
            errors.append("benchmark.repeats must be a positive integer")
        else:
            repeats = int(benchmark["repeats"])
        if benchmark.get("mode_schedule") not in {"block", "interleave"}:
            errors.append("benchmark.mode_schedule must be block or interleave")
        if isinstance(hardware, dict) and benchmark.get("architecture") != hardware.get("architecture"):
            errors.append("benchmark.architecture must match hardware.architecture")

    expected_samples = prompt_count * repeats
    modes = data.get("modes")
    dense_modes: list[dict[str, Any]] = []
    recommended_modes: list[dict[str, Any]] = []
    mode_ids: set[str] = set()
    if not isinstance(modes, list) or not modes:
        errors.append("modes must be a nonempty array")
    else:
        for index, mode in enumerate(modes):
            label = f"modes[{index}]"
            if not isinstance(mode, dict):
                errors.append(f"{label} must be an object")
                continue
            errors.extend(
                check_exact_fields(
                    mode,
                    {
                        "mode_id",
                        "recommended",
                        "settings",
                        "latency_ms",
                        "tokens_per_sec",
                        "qa",
                        "output_checksum",
                        "process_failures",
                        "parse_failures",
                    },
                    label,
                )
            )
            mode_id = mode.get("mode_id")
            if not is_nonempty_string(mode_id):
                errors.append(f"{label}.mode_id must be a nonempty string")
            elif mode_id in mode_ids:
                errors.append(f"duplicate mode_id: {mode_id}")
            else:
                mode_ids.add(mode_id)
            if not isinstance(mode.get("recommended"), bool):
                errors.append(f"{label}.recommended must be boolean")
            elif mode["recommended"]:
                recommended_modes.append(mode)

            settings = mode.get("settings")
            if not isinstance(settings, dict):
                errors.append(f"{label}.settings must be an object")
            else:
                errors.extend(
                    check_exact_fields(
                        settings,
                        {"top_k", "sparse_min_density", "no_top_k_sort", "top_k_select", "sparse_scope"},
                        f"{label}.settings",
                    )
                )
                top_k = settings.get("top_k")
                if not is_finite_number(top_k) or not 0 <= float(top_k) <= 1:
                    errors.append(f"{label}.settings.top_k must be between 0 and 1")
                elif float(top_k) == 0:
                    dense_modes.append(mode)
                sparse_min_density = settings.get("sparse_min_density")
                if sparse_min_density is not None and (
                    not is_finite_number(sparse_min_density)
                    or not 0 <= float(sparse_min_density) <= 1
                ):
                    errors.append(f"{label}.settings.sparse_min_density must be null or between 0 and 1")
                if not isinstance(settings.get("no_top_k_sort"), bool):
                    errors.append(f"{label}.settings.no_top_k_sort must be boolean")
                if settings.get("top_k_select") not in {"nth", "histogram"}:
                    errors.append(f"{label}.settings.top_k_select is invalid")
                if settings.get("sparse_scope") not in {"all", "ffn", "down", "none"}:
                    errors.append(f"{label}.settings.sparse_scope is invalid")

            errors.extend(validate_metric(mode.get("latency_ms"), f"{label}.latency_ms", expected_samples))
            errors.extend(
                validate_metric(mode.get("tokens_per_sec"), f"{label}.tokens_per_sec", expected_samples)
            )

            qa = mode.get("qa")
            if not isinstance(qa, dict):
                errors.append(f"{label}.qa must be an object")
            else:
                errors.extend(check_exact_fields(qa, {"numerator", "denominator", "rate"}, f"{label}.qa"))
                numerator = qa.get("numerator")
                denominator = qa.get("denominator")
                rate = qa.get("rate")
                if not is_nonnegative_integer(numerator):
                    errors.append(f"{label}.qa.numerator must be a nonnegative integer")
                if not is_positive_integer(denominator):
                    errors.append(f"{label}.qa.denominator must be a positive integer")
                elif denominator != expected_samples:
                    errors.append(f"{label}.qa.denominator must equal {expected_samples}")
                if is_nonnegative_integer(numerator) and is_positive_integer(denominator):
                    if numerator > denominator:
                        errors.append(f"{label}.qa.numerator cannot exceed denominator")
                    recalculated_rate = numerator / denominator
                    if not is_finite_number(rate) or not math.isclose(
                        float(rate), recalculated_rate, rel_tol=1e-12, abs_tol=1e-12
                    ):
                        errors.append(f"{label}.qa.rate is inconsistent with numerator/denominator")

            checksum = mode.get("output_checksum")
            if not isinstance(checksum, str) or not SHA256_RE.fullmatch(checksum):
                errors.append(f"{label}.output_checksum must be a lowercase SHA-256 digest")
            for field in ("process_failures", "parse_failures"):
                if mode.get(field) != 0:
                    errors.append(f"{label}.{field} must be 0 for a portable submission")

    if len(dense_modes) > 1:
        errors.append("submission may contain only one dense mode")
    if len(recommended_modes) > 1:
        errors.append("submission may mark only one mode as recommended")
    if recommended_modes:
        if not dense_modes:
            errors.append("a recommended mode requires a dense baseline")
        else:
            dense_qa = dense_modes[0].get("qa", {})
            candidate_qa = recommended_modes[0].get("qa", {})
            dense_rate = dense_qa.get("rate") if isinstance(dense_qa, dict) else None
            candidate_rate = candidate_qa.get("rate") if isinstance(candidate_qa, dict) else None
            if is_finite_number(dense_rate) and is_finite_number(candidate_rate):
                if float(candidate_rate) < float(dense_rate):
                    errors.append("recommended mode must preserve or improve dense QA rate")

    caveats = data.get("caveats")
    if not isinstance(caveats, list) or not caveats or any(not is_nonempty_string(item) for item in caveats):
        errors.append("caveats must be a nonempty array of nonempty strings")

    errors.extend(find_private_fields(data))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Benchmark submission JSON file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = load_json_object(args.submission)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1
    errors = validate_submission_data(data)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        print(f"Submission validation failed with {len(errors)} issue(s).")
        return 1
    print(f"[OK] benchmark submission: {args.submission}")
    print(f"[OK] modes: {len(data['modes'])}; prompt count: {data['prompt_set']['prompt_count']}")
    print("Benchmark submission validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
