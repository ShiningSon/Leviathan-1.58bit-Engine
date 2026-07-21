#!/usr/bin/env python3
"""Export benchmark_engine.py results as a portable benchmark submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .validate_benchmark_submission import load_json_object, validate_submission_data
except ImportError:
    from validate_benchmark_submission import load_json_object, validate_submission_data


DEFAULT_CAVEAT = (
    "This is a hardware-specific measurement; it does not establish a general sparse speedup."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_checksum(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def metric(samples: list[float]) -> dict[str, Any]:
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("metric samples must be positive finite numbers")
    return {
        "samples": samples,
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "standard_deviation": statistics.pstdev(samples),
    }


def close_enough(left: object, right: float) -> bool:
    return isinstance(left, (int, float)) and not isinstance(left, bool) and math.isclose(
        float(left), right, rel_tol=1e-7, abs_tol=1e-7
    )


def model_metadata(results: dict[str, Any]) -> dict[str, Any]:
    metadata = results.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("results.json does not contain reproducibility metadata")
    required = ("engine_commit", "model_repo", "model_revision", "prompt_file_sha256")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError(f"results.json metadata is missing: {', '.join(missing)}")
    return metadata


def measured_prompt_results(mode_result: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    runs = mode_result.get("runs")
    if not isinstance(runs, list):
        raise ValueError("mode runs must be an array")
    measured = [run for run in runs if isinstance(run, dict) and not run.get("warmup")]
    process_failures = 0
    parse_failures = 0
    prompts: list[dict[str, Any]] = []
    for run in measured:
        process = run.get("process")
        if not isinstance(process, dict):
            raise ValueError("measured run is missing process data")
        if process.get("returncode") != 0:
            process_failures += 1
        if not process.get("parse_ok"):
            parse_failures += 1
        prompt_results = process.get("prompt_results")
        if not isinstance(prompt_results, list):
            raise ValueError("measured run is missing prompt_results")
        prompts.extend(item for item in prompt_results if isinstance(item, dict))
    return prompts, process_failures, parse_failures


def mode_settings(mode: str, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "top_k": float(mode),
        "sparse_min_density": settings.get("sparse_min_density"),
        "no_top_k_sort": bool(settings.get("no_top_k_sort", False)),
        "top_k_select": settings.get("top_k_select", "nth"),
        "sparse_scope": settings.get("sparse_scope", "all"),
    }


def export_submission(
    results: dict[str, Any],
    hardware: dict[str, Any],
    prompts_path: Path,
    *,
    recommended_mode: str | None,
    caveats: list[str],
) -> dict[str, Any]:
    metadata = model_metadata(results)
    settings = results.get("settings")
    prompts = results.get("prompts")
    modes = results.get("modes")
    if not isinstance(settings, dict) or not isinstance(prompts, list) or not isinstance(modes, list):
        raise ValueError("results.json has an unsupported structure")
    prompt_hash = sha256_file(prompts_path)
    if metadata["prompt_file_sha256"] != prompt_hash:
        raise ValueError("prompt file SHA-256 does not match results.json")
    repeats = settings.get("repeat")
    warmups = settings.get("warmup")
    if not isinstance(repeats, int) or repeats < 1 or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("results.json repeat/warmup settings are invalid")
    expected_count = repeats * len(prompts)

    exported_modes: list[dict[str, Any]] = []
    seen_recommended = False
    for mode_result in modes:
        if not isinstance(mode_result, dict) or not isinstance(mode_result.get("mode"), str):
            raise ValueError("results.json contains an invalid mode entry")
        mode = mode_result["mode"]
        prompt_results, process_failures, parse_failures = measured_prompt_results(mode_result)
        if process_failures or parse_failures:
            raise ValueError(
                f"mode {mode} is incomplete: process_failures={process_failures}, parse_failures={parse_failures}"
            )
        if len(prompt_results) != expected_count:
            raise ValueError(
                f"mode {mode} has {len(prompt_results)} measured prompts; expected {expected_count}"
            )
        latency = [float(item["latency_ms"]) for item in prompt_results if item.get("latency_ms") is not None]
        throughput = [
            float(item["tokens_per_sec"])
            for item in prompt_results
            if item.get("tokens_per_sec") is not None
        ]
        if len(latency) != expected_count or len(throughput) != expected_count:
            raise ValueError(f"mode {mode} has missing latency or throughput samples")
        qa_numerator = sum(item.get("qa_pass") is True for item in prompt_results)
        summary = mode_result.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(f"mode {mode} is missing summary data")
        expected_summary = {
            "measured_prompt_count": expected_count,
            "qa_pass_count": qa_numerator,
            "qa_pass_rate": qa_numerator / expected_count,
            "avg_latency_ms": statistics.fmean(latency),
            "avg_tokens_per_sec": statistics.fmean(throughput),
            "process_failures": 0,
            "parse_failures": 0,
        }
        for key, expected in expected_summary.items():
            actual = summary.get(key)
            if isinstance(expected, float):
                if not close_enough(actual, expected):
                    raise ValueError(f"mode {mode} summary.{key} is inconsistent")
            elif actual != expected:
                raise ValueError(f"mode {mode} summary.{key} is inconsistent")
        is_recommended = recommended_mode is not None and float(mode) == float(recommended_mode)
        seen_recommended = seen_recommended or is_recommended
        outputs = [
            {"prompt": item.get("prompt"), "output": item.get("output"), "qa_pass": item.get("qa_pass")}
            for item in prompt_results
        ]
        exported_modes.append(
            {
                "mode_id": mode,
                "recommended": is_recommended,
                "settings": mode_settings(mode, settings),
                "latency_ms": metric(latency),
                "tokens_per_sec": metric(throughput),
                "qa": {
                    "numerator": qa_numerator,
                    "denominator": expected_count,
                    "rate": qa_numerator / expected_count,
                },
                "output_checksum": canonical_checksum(outputs),
                "process_failures": 0,
                "parse_failures": 0,
            }
        )
    if recommended_mode is not None and not seen_recommended:
        raise ValueError(f"recommended mode {recommended_mode} was not measured")

    if hardware.get("engine_commit") != metadata["engine_commit"]:
        raise ValueError("hardware engine_commit does not match results.json")
    if hardware.get("model_repo") != metadata["model_repo"]:
        raise ValueError("hardware model_repo does not match results.json")
    if hardware.get("model_revision") != metadata["model_revision"]:
        raise ValueError("hardware model_revision does not match results.json")

    submission = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "model": {
            "identifier": str(results.get("model") or "unknown"),
            "hf_repo": metadata["model_repo"],
            "hf_revision": metadata["model_revision"],
            "metadata_identifier": metadata.get("model_metadata_identifier"),
        },
        "engine_commit": metadata["engine_commit"],
        "prompt_set": {
            "file_name": prompts_path.name,
            "sha256": prompt_hash,
            "prompt_count": len(prompts),
        },
        "benchmark": {
            "architecture": settings.get("architecture"),
            "prompt_template": settings.get("prompt_template"),
            "max_new": settings.get("max_new"),
            "warmups": warmups,
            "repeats": repeats,
            "mode_schedule": settings.get("mode_schedule", "block"),
        },
        "modes": exported_modes,
        "caveats": caveats or [DEFAULT_CAVEAT],
    }
    errors = validate_submission_data(submission)
    if errors:
        raise ValueError("submission validation failed:\n- " + "\n- ".join(errors))
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--hardware-manifest", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recommended-mode")
    parser.add_argument("--caveat", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        submission = export_submission(
            load_json_object(args.results),
            load_json_object(args.hardware_manifest),
            args.prompts,
            recommended_mode=args.recommended_mode,
            caveats=args.caveat,
        )
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(submission, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] wrote benchmark submission: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
