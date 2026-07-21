#!/usr/bin/env python3
"""Validate the canonical Leviathan scaling benchmark summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "model",
    "parameter_class",
    "architecture",
    "benchmark_repeats",
    "dense_latency_ms",
    "dense_tokens_per_sec",
    "dense_qa_pass",
    "dense_qa_total",
    "recommended_top_k",
    "selector",
    "sparse_scope",
    "sparse_latency_ms",
    "sparse_tokens_per_sec",
    "sparse_qa_pass",
    "sparse_qa_total",
    "latency_delta_percent",
    "tokens_per_sec_delta_percent",
    "status",
    "caveat",
}
EXPECTED_PARAMETER_CLASSES = {"30M", "70M", "100M", "200M"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def percentage_delta(value: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("baseline must be nonzero")
    return ((value - baseline) / baseline) * 100.0


def qa_rate(passed: int, total: int) -> float:
    if total <= 0:
        raise ValueError("QA total must be positive")
    if passed < 0 or passed > total:
        raise ValueError("QA pass count must be between zero and total")
    return passed / total


def validate_summary(data: dict[str, Any], tolerance: float = 0.02) -> list[str]:
    errors: list[str] = []
    records = data.get("records")
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(records, list) or not records:
        return errors + ["records must be a non-empty list"]

    seen_models: set[str] = set()
    seen_parameter_classes: set[str] = set()
    for index, record in enumerate(records):
        prefix = f"record {index}"
        if not isinstance(record, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        missing = sorted(REQUIRED_FIELDS - record.keys())
        if missing:
            errors.append(f"{prefix}: missing fields: {', '.join(missing)}")
            continue

        model = str(record["model"])
        prefix = model
        if model in seen_models:
            errors.append(f"{prefix}: duplicate model")
        seen_models.add(model)
        seen_parameter_classes.add(str(record["parameter_class"]))

        try:
            dense_latency = float(record["dense_latency_ms"])
            dense_speed = float(record["dense_tokens_per_sec"])
            sparse_latency = float(record["sparse_latency_ms"])
            sparse_speed = float(record["sparse_tokens_per_sec"])
            if min(dense_latency, dense_speed, sparse_latency, sparse_speed) <= 0:
                errors.append(f"{prefix}: latency and throughput values must be positive")

            expected_latency_delta = percentage_delta(sparse_latency, dense_latency)
            expected_speed_delta = percentage_delta(sparse_speed, dense_speed)
            if abs(float(record["latency_delta_percent"]) - expected_latency_delta) > tolerance:
                errors.append(
                    f"{prefix}: latency delta mismatch; stored={record['latency_delta_percent']} "
                    f"calculated={expected_latency_delta:.4f}"
                )
            if abs(float(record["tokens_per_sec_delta_percent"]) - expected_speed_delta) > tolerance:
                errors.append(
                    f"{prefix}: tokens/sec delta mismatch; stored={record['tokens_per_sec_delta_percent']} "
                    f"calculated={expected_speed_delta:.4f}"
                )

            dense_rate = qa_rate(int(record["dense_qa_pass"]), int(record["dense_qa_total"]))
            sparse_rate = qa_rate(int(record["sparse_qa_pass"]), int(record["sparse_qa_total"]))
            if "confirmed" in str(record["status"]).lower() and sparse_rate < dense_rate:
                errors.append(
                    f"{prefix}: confirmed recommended mode has lower QA rate than dense"
                )
            if float(record["recommended_top_k"]) <= 0:
                errors.append(f"{prefix}: recommended_top_k must be positive")
            if int(record["benchmark_repeats"]) <= 0:
                errors.append(f"{prefix}: benchmark_repeats must be positive")
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix}: invalid numeric value: {exc}")

    missing_classes = sorted(EXPECTED_PARAMETER_CLASSES - seen_parameter_classes)
    if missing_classes:
        errors.append(f"missing parameter classes: {', '.join(missing_classes)}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary",
        nargs="?",
        type=Path,
        default=repo_root() / "benchmarks" / "results" / "scaling_summary.json",
        help="Path to scaling_summary.json.",
    )
    parser.add_argument("--tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    errors = validate_summary(data, tolerance=args.tolerance)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    print(f"Validated {len(data['records'])} scaling records:")
    for record in data["records"]:
        print(
            "[OK] {parameter_class} {model}: Top-K {top_k:.2f}, "
            "{speed_delta:+.2f}% tok/s, {latency_delta:+.2f}% latency, "
            "QA {qa_pass}/{qa_total}".format(
                parameter_class=record["parameter_class"],
                model=record["model"],
                top_k=float(record["recommended_top_k"]),
                speed_delta=float(record["tokens_per_sec_delta_percent"]),
                latency_delta=float(record["latency_delta_percent"]),
                qa_pass=record["sparse_qa_pass"],
                qa_total=record["sparse_qa_total"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
