#!/usr/bin/env python3
"""Run repository-only checks for the Leviathan v0.9 research release."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

try:
    from .release_utils import (
        find_developer_absolute_paths,
        is_forbidden_tracked_path,
        markdown_internal_targets,
    )
    from .validate_benchmark_summary import validate_summary
except ImportError:
    from release_utils import (
        find_developer_absolute_paths,
        is_forbidden_tracked_path,
        markdown_internal_targets,
    )
    from validate_benchmark_summary import validate_summary


REQUIRED_DOCUMENTS = (
    "README.md",
    "BENCHMARK.md",
    "MODEL_ZOO.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "docs/BENCHMARK_AUTOMATION.md",
    "docs/HF_PUBLICATION_CHECKLIST.md",
    "docs/REPRODUCIBILITY.md",
    "docs/V09A_200M_SCALING_PROBE.md",
    "docs/V09_RELEASE_CHECKLIST.md",
    "docs/V09_RELEASE_NOTES.md",
)
REQUIRED_RELEASE_FILES = (
    "releases/v0.9.0_hf_manifest.json",
    "scripts/verify_hf_release.py",
)
REQUIRED_HF_CARDS = (
    "hf_cards/Leviathan-MLGRU-70M-TinyStories-Instruct-v07b_README.md",
    "hf_cards/Leviathan-MLGRU-100M-TinyStories-Instruct-v08a_README.md",
    "hf_cards/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a_README.md",
)
REQUIRED_CONFIGS = (
    "training/configs/v07b_70m_qa_repair.json",
    "training/configs/v08a_100m_mlgru.json",
    "training/configs/v09a_200m_mlgru.json",
    "training/configs/v09a_200m_mlgru_h100_fast.json",
    "training/configs/v09a_200m_mlgru_h100_safe.json",
    "training/configs/v09a_200m_mlgru_h200_fast.json",
)
SUMMARY_PATH = "benchmarks/results/scaling_summary.json"
RELEASE_MANIFEST_PATH = "releases/v0.9.0_hf_manifest.json"
V09_CARD_PATH = "hf_cards/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a_README.md"
V09_HF_REPO_ID = "ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a"
V09_HF_REVISION = "116a857bdaf2a1118d479d52aedba7e65cbff960"
V09_HF_FILES = {
    ".gitattributes",
    "README.md",
    "leviathan_mlgru_200m_instruct_v09a.bin",
    "leviathan_mlgru_200m_instruct_v09a_meta.json",
    "leviathan_mlgru_tokenizer/tokenizer.json",
    "leviathan_mlgru_tokenizer/tokenizer_config.json",
    "report.json",
    "sample_outputs.txt",
}
V09_CONSISTENCY_FILES = (
    "README.md",
    "BENCHMARK.md",
    "MODEL_ZOO.md",
    "docs/V09A_200M_SCALING_PROBE.md",
    "docs/V09_RELEASE_NOTES.md",
    V09_CARD_PATH,
)
SECRET_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(
        r"\b(?:HF_TOKEN|HUGGINGFACE_TOKEN|MODAL_TOKEN_ID|MODAL_TOKEN_SECRET)\s*=\s*['\"][^'\"]+['\"]",
        re.IGNORECASE,
    ),
)
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".gitignore",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_files(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [item for item in completed.stdout.decode("utf-8").split("\0") if item]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_required_files(root: Path, paths: tuple[str, ...], label: str) -> list[str]:
    return [f"{label} missing: {path}" for path in paths if not (root / path).is_file()]


def check_summary(root: Path) -> list[str]:
    path = root / SUMMARY_PATH
    if not path.is_file():
        return [f"benchmark summary missing: {SUMMARY_PATH}"]
    try:
        data = json.loads(read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"benchmark summary cannot be loaded: {exc}"]
    return [f"benchmark summary: {error}" for error in validate_summary(data)]


def validate_release_manifest(data: object) -> list[str]:
    if not isinstance(data, dict):
        return ["release manifest must contain a JSON object"]

    errors: list[str] = []
    expected_values = {
        "schema_version": 1,
        "repo_id": V09_HF_REPO_ID,
        "resolved_hf_revision": V09_HF_REVISION,
        "package_name": "leviathan_mlgru_200m_instruct_v09a",
        "architecture": "mlgru",
        "parameter_class": "200M-class",
        "estimated_trainable_parameters": 233388800,
        "recommended_top_k": 0.10,
        "selector": "histogram",
        "sparse_scope": "down",
        "sparse_min_density": 0.6,
    }
    for key, expected in expected_values.items():
        if data.get(key) != expected:
            errors.append(f"release manifest {key} must be {expected!r}")

    for key in ("verification_date_utc", "format", "expected_runtime"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            errors.append(f"release manifest {key} must be a nonempty string")

    benchmark = data.get("confirmed_benchmark")
    if not isinstance(benchmark, dict):
        errors.append("release manifest confirmed_benchmark must be an object")
    else:
        benchmark_values = {
            "reference": SUMMARY_PATH,
            "dense_latency_ms": 193.92,
            "dense_tokens_per_sec": 88.30,
            "sparse_latency_ms": 175.33,
            "sparse_tokens_per_sec": 97.54,
            "strict_qa": "600/600",
        }
        for key, expected in benchmark_values.items():
            if benchmark.get(key) != expected:
                errors.append(f"release manifest confirmed_benchmark.{key} must be {expected!r}")

    files = data.get("files")
    if not isinstance(files, list):
        errors.append("release manifest files must be an array")
        return errors

    paths: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            errors.append(f"release manifest files[{index}] must be an object")
            continue
        path = record.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"release manifest files[{index}].path must be nonempty")
            continue
        if path in paths:
            errors.append(f"release manifest contains duplicate file: {path}")
        paths.add(path)
        if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0:
            errors.append(f"release manifest file must have positive bytes: {path}")
        if not isinstance(record.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", record["sha256"]
        ):
            errors.append(f"release manifest file has invalid SHA-256: {path}")

    if paths != V09_HF_FILES:
        missing = sorted(V09_HF_FILES - paths)
        extra = sorted(paths - V09_HF_FILES)
        if missing:
            errors.append(f"release manifest is missing public files: {', '.join(missing)}")
        if extra:
            errors.append(f"release manifest has unexpected public files: {', '.join(extra)}")
    return errors


def check_release_manifest(root: Path) -> list[str]:
    path = root / RELEASE_MANIFEST_PATH
    if not path.is_file():
        return [f"release manifest missing: {RELEASE_MANIFEST_PATH}"]
    try:
        data = json.loads(read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"release manifest cannot be loaded: {exc}"]
    return [f"release manifest: {error}" for error in validate_release_manifest(data)]


def check_tracked_artifacts(paths: list[str]) -> list[str]:
    return [
        f"forbidden generated artifact is tracked: {path}"
        for path in paths
        if is_forbidden_tracked_path(path)
    ]


def check_developer_paths(root: Path, paths: list[str]) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        path = root / relative
        if path.suffix.lower() not in {".md", ".cff"}:
            continue
        for match in find_developer_absolute_paths(read_text(path)):
            errors.append(f"developer-specific absolute path in {relative}: {match}")
    return errors


def check_secrets(root: Path, paths: list[str]) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        path = root / relative
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        if path.stat().st_size > 1_000_000:
            continue
        text = read_text(path)
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible committed secret in {relative}")
                break
    return errors


def check_markdown_links(root: Path, paths: list[str]) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        path = root / relative
        if path.suffix.lower() != ".md":
            continue
        for target in markdown_internal_targets(path, read_text(path)):
            try:
                target.relative_to(root)
            except ValueError:
                errors.append(f"internal link escapes repository in {relative}: {target}")
                continue
            if not target.exists():
                errors.append(f"broken internal link in {relative}: {target.relative_to(root)}")
    return errors


def check_v09_consistency(root: Path) -> list[str]:
    errors: list[str] = []
    summary = json.loads(read_text(root / SUMMARY_PATH))
    record = next(
        (item for item in summary["records"] if item["model"].endswith("v09a")),
        None,
    )
    if record is None:
        return ["v09a record missing from benchmark summary"]
    if float(record["recommended_top_k"]) != 0.10:
        errors.append("v09a recommended_top_k must be 0.10")
    if (record["sparse_qa_pass"], record["sparse_qa_total"]) != (600, 600):
        errors.append("v09a sparse QA must be recorded as 600/600")
    if float(record["sparse_tokens_per_sec"]) != 97.54:
        errors.append("v09a sparse throughput must be recorded as 97.54 tok/s")
    if float(record["sparse_latency_ms"]) != 175.33:
        errors.append("v09a sparse latency must be recorded as 175.33 ms")

    for relative in V09_CONSISTENCY_FILES:
        path = root / relative
        if not path.is_file():
            continue
        text = read_text(path)
        for required in ("175.33", "97.54", "600/600"):
            if required not in text:
                errors.append(f"{relative} does not contain required v09a value {required}")

    card = read_text(root / V09_CARD_PATH).lower()
    card_requirements = (
        "not a general assistant",
        "not a general sparse speedup claim",
        "top-k is not always faster",
        "not automatically proven",
        "0.08",
    )
    for phrase in card_requirements:
        if phrase not in card:
            errors.append(f"v09a card missing required caveat: {phrase}")
    if "not recommended" not in card and "not the recommended" not in card:
        errors.append("v09a card does not reject Top-K 0.08 as the recommendation")
    return errors


def main() -> int:
    root = repo_root()
    paths = tracked_files(root)
    errors: list[str] = []
    errors.extend(check_required_files(root, REQUIRED_DOCUMENTS, "documentation"))
    errors.extend(check_required_files(root, REQUIRED_RELEASE_FILES, "release file"))
    errors.extend(check_required_files(root, REQUIRED_HF_CARDS, "HF card"))
    errors.extend(check_required_files(root, REQUIRED_CONFIGS, "training config"))
    errors.extend(check_summary(root))
    errors.extend(check_release_manifest(root))
    errors.extend(check_tracked_artifacts(paths))
    errors.extend(check_developer_paths(root, paths))
    errors.extend(check_secrets(root, paths))
    errors.extend(check_markdown_links(root, paths))
    if not errors:
        errors.extend(check_v09_consistency(root))

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        print(f"Release readiness failed with {len(errors)} issue(s).")
        return 1

    print(f"[OK] required documentation: {len(REQUIRED_DOCUMENTS)} files")
    print(f"[OK] release files: {len(REQUIRED_RELEASE_FILES)} files")
    print(f"[OK] Hugging Face cards: {len(REQUIRED_HF_CARDS)} files")
    print(f"[OK] training configs: {len(REQUIRED_CONFIGS)} files")
    print("[OK] canonical benchmark summary and v09a values")
    print("[OK] v0.9.0 public Hugging Face manifest")
    print(f"[OK] tracked-file hygiene and secret scan: {len(paths)} files")
    print("[OK] internal Markdown links and portable documentation paths")
    print("Release readiness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
