#!/usr/bin/env python3
"""Generate paired qualitative holdout outputs without judging their quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import benchmark_engine
except ImportError:
    import benchmark_engine


EXPECTED_QUALITATIVE_COUNT = 35
EXPECTED_CATEGORIES = {
    "leviathan_concepts",
    "paraphrase_robustness",
    "negation_and_limitation_awareness",
    "package_runtime_usage",
    "sparse_mode_caveats",
    "simple_tinystories_comprehension",
    "malformed_or_ambiguous_requests",
}
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
REVIEW_FIELDS = (
    "dense_verdict",
    "sparse_verdict",
    "semantic_equivalence",
    "reviewer",
    "notes",
    "disagreement_resolution",
)
CSV_FIELDS = (
    "prompt_id",
    "category",
    "prompt",
    "review_criteria_json",
    "dense_mode",
    "dense_output",
    "dense_sha256",
    "sparse_mode",
    "sparse_output",
    "sparse_sha256",
    *REVIEW_FIELDS,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_qualitative_prompts(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load qualitative prompt file {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("qualitative_review"), list):
        raise ValueError("prompt file must contain a qualitative_review list")

    prompts: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(data["qualitative_review"]):
        if not isinstance(item, dict):
            raise ValueError(f"qualitative_review[{index}] must be an object")
        prompt_id = item.get("id")
        category = item.get("category")
        prompt = item.get("prompt")
        criteria = item.get("review_criteria")
        if not isinstance(prompt_id, str) or not re.fullmatch(r"[a-z]{2}-q[0-9]{2}", prompt_id):
            raise ValueError(f"qualitative_review[{index}].id is invalid")
        if prompt_id in ids:
            raise ValueError(f"duplicate qualitative prompt id: {prompt_id}")
        if category not in EXPECTED_CATEGORIES:
            raise ValueError(f"qualitative_review[{index}].category is invalid")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"qualitative_review[{index}].prompt is invalid")
        if (
            not isinstance(criteria, list)
            or len(criteria) < 2
            or any(not isinstance(value, str) or not value.strip() for value in criteria)
        ):
            raise ValueError(f"qualitative_review[{index}].review_criteria is invalid")
        ids.add(prompt_id)
        prompts.append(
            {
                "id": prompt_id,
                "category": category,
                "prompt": prompt.strip(),
                "review_criteria": [value.strip() for value in criteria],
            }
        )

    if len(prompts) != EXPECTED_QUALITATIVE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_QUALITATIVE_COUNT} qualitative prompts, found {len(prompts)}"
        )
    if {item["category"] for item in prompts} != EXPECTED_CATEGORIES:
        raise ValueError("qualitative prompt set must cover all seven categories")
    return prompts


def resolve_modes(modes: list[str]) -> tuple[str, str]:
    if len(modes) != 2:
        raise ValueError("--modes must contain exactly one dense and one Top-K mode")
    try:
        values = [(mode, float(mode)) for mode in modes]
    except ValueError as exc:
        raise ValueError("--modes values must be numeric") from exc
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for _, value in values):
        raise ValueError("--modes values must be finite ratios from 0 through 1")
    dense = [mode for mode, value in values if value == 0.0]
    sparse = [mode for mode, value in values if value > 0.0]
    if len(dense) != 1 or len(sparse) != 1:
        raise ValueError("--modes must contain exactly one dense mode and one positive Top-K mode")
    return dense[0], sparse[0]


def qualitative_engine_prompts(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": item["prompt"],
            "expected": [],
            "forbidden": [],
            "max_words": None,
        }
        for item in prompts
    ]


def checked_process(process: dict[str, Any], mode: str, expected_count: int) -> list[dict[str, Any]]:
    if process.get("returncode") != 0:
        raise ValueError(f"engine process failed for mode {mode}: return code {process.get('returncode')}")
    if process.get("parse_ok") is not True:
        raise ValueError(f"engine output parsing failed for mode {mode}")
    results = process.get("prompt_results")
    if not isinstance(results, list) or len(results) != expected_count:
        raise ValueError(f"mode {mode} returned an unexpected number of prompt results")
    for index, result in enumerate(results):
        if not isinstance(result, dict) or result.get("parse_error") is not None:
            raise ValueError(f"mode {mode} prompt {index} has a parse error")
        if not isinstance(result.get("output"), str) or not result["output"].strip():
            raise ValueError(f"mode {mode} prompt {index} has an empty output")
    return results


def output_record(mode: str, result: dict[str, Any], returncode: int) -> dict[str, Any]:
    output = result["output"].strip()
    return {
        "mode": mode,
        "output": output,
        "output_sha256": sha256_text(output),
        "process_returncode": returncode,
        "parse_ok": result.get("parse_error") is None,
    }


def generate_outputs(args: argparse.Namespace, *, root: Path | None = None) -> dict[str, Any]:
    root = root or benchmark_engine.repo_root()
    model_dir = benchmark_engine.resolve_path(args.model_dir, base=root)
    engine_path = benchmark_engine.resolve_path(args.engine, base=root)
    prompts_path = benchmark_engine.resolve_path(args.prompts, base=root)
    if not model_dir.is_dir():
        raise ValueError(f"model directory not found: {model_dir}")
    if not engine_path.is_file():
        raise ValueError(f"engine file not found: {engine_path}")
    if not prompts_path.is_file():
        raise ValueError(f"prompt file not found: {prompts_path}")
    if args.max_new < 1:
        raise ValueError("--max-new must be at least 1")
    if args.prompt_template != "qa":
        raise ValueError("qualitative instruct-model runs must use --prompt-template qa")
    if not REPO_RE.fullmatch(args.model_repo):
        raise ValueError("--model-repo must use owner/repository format")
    if not REVISION_RE.fullmatch(args.model_revision):
        raise ValueError("--model-revision must be a pinned lowercase hexadecimal revision")
    if args.sparse_min_density is not None and (
        not math.isfinite(args.sparse_min_density)
        or args.sparse_min_density < 0.0
        or args.sparse_min_density > 1.0
    ):
        raise ValueError("--sparse-min-density must be a finite ratio from 0 through 1")

    dense_mode, sparse_mode = resolve_modes(args.modes)
    prompts = load_qualitative_prompts(prompts_path)
    engine_prompts = qualitative_engine_prompts(prompts)
    bin_path = model_dir / args.bin_name if args.bin_name else benchmark_engine.find_one(
        model_dir, "*.bin", "model binary"
    )
    meta_path = model_dir / args.meta_name if args.meta_name else benchmark_engine.find_one(
        model_dir, "*_meta.json", "metadata JSON"
    )
    if not bin_path.is_file() or not meta_path.is_file():
        raise ValueError("model binary or metadata file is missing")

    processes: dict[str, dict[str, Any]] = {}
    parsed: dict[str, list[dict[str, Any]]] = {}
    for mode in (dense_mode, sparse_mode):
        print(f"[qualitative] mode={mode}; prompts={len(prompts)}", flush=True)
        process = benchmark_engine.run_engine(
            python_exe=args.python,
            engine_path=engine_path,
            model_dir=model_dir,
            bin_name=bin_path.name,
            meta_name=meta_path.name,
            architecture=args.architecture,
            prompt_template=args.prompt_template,
            max_new=args.max_new,
            mode=mode,
            prompts=engine_prompts,
            timeout=args.timeout,
            sparse_min_density=args.sparse_min_density,
            no_top_k_sort=args.no_top_k_sort,
            top_k_select=args.top_k_select,
            sparse_scope=args.sparse_scope,
        )
        processes[mode] = process
        parsed[mode] = checked_process(process, mode, len(prompts))

    rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        dense_result = parsed[dense_mode][index]
        sparse_result = parsed[sparse_mode][index]
        if dense_result.get("prompt") != prompt["prompt"] or sparse_result.get("prompt") != prompt["prompt"]:
            raise ValueError(f"prompt ordering mismatch at {prompt['id']}")
        rows.append(
            {
                **prompt,
                "dense": output_record(
                    dense_mode, dense_result, int(processes[dense_mode]["returncode"])
                ),
                "sparse": output_record(
                    sparse_mode, sparse_result, int(processes[sparse_mode]["returncode"])
                ),
            }
        )

    commit = benchmark_engine.git_commit(root)
    if commit is None or not COMMIT_RE.fullmatch(commit):
        raise ValueError("could not resolve the engine Git commit")
    return {
        "schema_version": 1,
        "review_status": "unreviewed",
        "generated_at_utc": utc_now(),
        "engine_commit": commit,
        "model": {
            "repository": args.model_repo,
            "revision": args.model_revision,
            "metadata_identifier": benchmark_engine.model_metadata_identifier(meta_path),
        },
        "prompt_set": {
            "file_name": prompts_path.name,
            "sha256": benchmark_engine.sha256_file(prompts_path),
            "qualitative_count": len(prompts),
        },
        "settings": {
            "architecture": args.architecture,
            "prompt_template": args.prompt_template,
            "max_new": args.max_new,
            "modes": [dense_mode, sparse_mode],
            "sparse_min_density": args.sparse_min_density,
            "no_top_k_sort": args.no_top_k_sort,
            "sparse_scope": args.sparse_scope,
            "top_k_select": args.top_k_select,
            "engine_file": engine_path.name,
            "model_package": model_dir.name,
            "binary_file": bin_path.name,
            "metadata_file": meta_path.name,
        },
        "prompts": rows,
    }


def markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# v0.9.1 Qualitative Holdout Review Worksheet",
        "",
        "Status: **unreviewed**. Generated outputs and blank fields are not human judgments.",
        "AI-generated preliminary comments do not count as the required human review.",
        "",
        f"- Engine commit: `{data['engine_commit']}`",
        f"- Model: `{data['model']['repository']}`",
        f"- Model revision: `{data['model']['revision']}`",
        f"- Prompt SHA-256: `{data['prompt_set']['sha256']}`",
        f"- Paired review rows: `{len(data['prompts'])}`",
        "",
    ]
    for item in data["prompts"]:
        lines.extend(
            [
                f"## {item['id']} - {item['category']}",
                "",
                f"**Prompt:** {item['prompt']}",
                "",
                "**Review criteria:**",
                "",
            ]
        )
        lines.extend(f"- {criterion}" for criterion in item["review_criteria"])
        lines.extend(
            [
                "",
                f"| Dense `--top-k {item['dense']['mode']}` | Top-K `--top-k {item['sparse']['mode']}` |",
                "|---|---|",
                f"| {markdown_cell(item['dense']['output'])} | {markdown_cell(item['sparse']['output'])} |",
                "",
                "| Human review field | Value |",
                "|---|---|",
                "| Dense verdict (`pass` / `concern` / `fail`) |  |",
                "| Sparse verdict (`pass` / `concern` / `fail`) |  |",
                "| Semantic equivalence (`yes` / `partial` / `no`) |  |",
                "| Reviewer |  |",
                "| Notes |  |",
                "| Disagreement resolution |  |",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def csv_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in data["prompts"]:
        rows.append(
            {
                "prompt_id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "review_criteria_json": json.dumps(
                    item["review_criteria"], ensure_ascii=False, separators=(",", ":")
                ),
                "dense_mode": item["dense"]["mode"],
                "dense_output": item["dense"]["output"],
                "dense_sha256": item["dense"]["output_sha256"],
                "sparse_mode": item["sparse"]["mode"],
                "sparse_output": item["sparse"]["output"],
                "sparse_sha256": item["sparse"]["output_sha256"],
                **{field: "" for field in REVIEW_FIELDS},
            }
        )
    return rows


def render_csv(data: dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows(data))
    return output.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--engine", default="engine.py")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--architecture", default="mlgru")
    parser.add_argument("--prompt-template", default="qa")
    parser.add_argument("--max-new", type=int, default=80)
    parser.add_argument("--modes", nargs="+", default=["0", "0.10"])
    parser.add_argument("--sparse-min-density", type=float, default=None)
    parser.add_argument("--no-top-k-sort", action="store_true")
    parser.add_argument("--sparse-scope", choices=["all", "ffn", "down", "none"], default="all")
    parser.add_argument("--top-k-select", choices=["nth", "histogram"], default="nth")
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bin", dest="bin_name")
    parser.add_argument("--meta", dest="meta_name")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = generate_outputs(args)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    out_dir = benchmark_engine.resolve_path(args.out_dir, base=benchmark_engine.repo_root())
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = out_dir / "qualitative_outputs.json"
    markdown_path = out_dir / "qualitative_review.md"
    csv_path = out_dir / "qualitative_review.csv"
    outputs_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(data), encoding="utf-8")
    csv_path.write_text(render_csv(data), encoding="utf-8", newline="")
    print(f"Wrote {outputs_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {csv_path}")
    print("No qualitative verdicts were assigned. Human review remains required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
