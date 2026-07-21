#!/usr/bin/env python3
"""Run repeatable Leviathan engine benchmarks and export JSON/Markdown results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATS_RE = re.compile(
    r"ENGINE>\s*(?P<text>.*?)[\r\n]+\[Stats:\s*"
    r"(?P<latency>[0-9]+(?:\.[0-9]+)?)\s*ms\s*\|\s*"
    r"(?P<tps>[0-9]+(?:\.[0-9]+)?)\s*tokens/sec\]",
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Leviathan engine.py across dense and Top-K modes."
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing the exported model package.")
    parser.add_argument("--engine", default="engine.py", help="Path to engine.py, relative to repo root or cwd.")
    parser.add_argument("--prompts", required=True, help="JSON prompt file.")
    parser.add_argument("--architecture", default="mlgru", help="engine.py --architecture value.")
    parser.add_argument("--prompt-template", default="qa", help="engine.py --prompt-template value.")
    parser.add_argument("--max-new", type=int, default=80, help="engine.py --max-new value.")
    parser.add_argument("--modes", nargs="+", default=["0"], help="Top-K values to benchmark. Defaults to dense mode only; pass candidate densities explicitly.")
    parser.add_argument("--repeat", type=int, default=3, help="Measured repeats per mode.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per mode, excluded from averages.")
    parser.add_argument("--mode-schedule", choices=["block", "interleave"], default="block", help="Run modes in contiguous blocks or round-robin order per repeat.")
    parser.add_argument("--out-dir", default="benchmark_runs/latest", help="Output directory.")
    parser.add_argument("--bin", dest="bin_name", help="Model .bin filename inside --model-dir.")
    parser.add_argument("--meta", dest="meta_name", help="Model metadata JSON filename inside --model-dir.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run engine.py.")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds before each engine run times out.")
    parser.add_argument("--sparse-min-density", type=float, default=None, help="Forward engine.py --sparse-min-density. When set, high-density Top-K can auto-fallback to dense.")
    parser.add_argument("--no-top-k-sort", action="store_true", help="Forward engine.py --no-top-k-sort for the experimental Top-K no-sort selection path.")
    parser.add_argument("--top-k-select", choices=["nth", "histogram"], default="nth", help="Forward engine.py --top-k-select for selector experiments.")
    parser.add_argument("--sparse-scope", choices=["all", "ffn", "down", "none"], default="all", help="Forward engine.py --sparse-scope for projection-scoped sparse experiments.")
    parser.add_argument("--model-repo", help="Model repository identifier recorded in results.json.")
    parser.add_argument("--model-revision", help="Pinned model or Hugging Face revision recorded in results.json.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def thread_environment() -> dict[str, str | None]:
    names = (
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    return {name: os.environ.get(name) for name in names}


def resolve_path(value: str, *, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()

    candidates = []
    if base is not None:
        candidates.append(base / path)
    candidates.append(Path.cwd() / path)
    candidates.append(repo_root() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def find_one(directory: Path, pattern: str, description: str) -> Path:
    matches = sorted(p for p in directory.glob(pattern) if p.is_file())
    if not matches:
        raise FileNotFoundError(f"No {description} found in {directory} with pattern {pattern}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise ValueError(f"Multiple {description} files found in {directory}: {names}. Pass an explicit CLI value.")
    return matches[0]


def model_display_name(model_dir: Path, meta_path: Path) -> str:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return model_dir.name

    for key in ("display_name", "model_id", "release_name", "package_name"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip() and not value.strip().startswith("."):
            return value.strip()

    model_name = meta.get("model_name")
    if isinstance(model_name, str) and model_name.strip() and not model_name.strip().startswith("."):
        return model_name.strip()
    return model_dir.name


def model_metadata_identifier(meta_path: Path) -> str:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return meta_path.stem
    for key in ("model_id", "package_name", "model_name", "display_name", "release_name"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return meta_path.stem


def load_prompts(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        exact_items = data.get("exact_match_regression")
        if not isinstance(exact_items, list):
            raise ValueError("Structured prompt files must contain an exact_match_regression list.")
        structured_prompts: list[dict[str, Any]] = []
        for index, item in enumerate(exact_items):
            if not isinstance(item, dict):
                raise ValueError(f"Exact-match prompt item {index} must be an object.")
            rules = item.get("rules")
            if not isinstance(rules, dict):
                raise ValueError(f"Exact-match prompt item {index} must contain a rules object.")
            structured_prompts.append(
                {
                    "prompt": item.get("prompt"),
                    "expected": rules.get("required_terms", []),
                    "forbidden": rules.get("forbidden_terms", []),
                    "max_words": rules.get("max_words"),
                }
            )
        data = structured_prompts
    if not isinstance(data, list):
        raise ValueError("Prompt file must contain a JSON list or a structured holdout object.")

    prompts: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt item {index} must be an object.")
        prompt = item.get("prompt")
        expected = item.get("expected", [])
        forbidden = item.get("forbidden", [])
        max_words = item.get("max_words")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt item {index} has an invalid prompt.")
        if not isinstance(expected, list) or not all(isinstance(x, str) for x in expected):
            raise ValueError(f"Prompt item {index} has an invalid expected list.")
        if not isinstance(forbidden, list) or not all(isinstance(x, str) for x in forbidden):
            raise ValueError(f"Prompt item {index} has an invalid forbidden list.")
        if max_words is not None and (not isinstance(max_words, int) or max_words < 1):
            raise ValueError(f"Prompt item {index} has an invalid max_words value.")
        prompts.append(
            {
                "prompt": prompt.strip(),
                "expected": expected,
                "forbidden": forbidden,
                "max_words": max_words,
            }
        )
    return prompts


def run_engine(
    *,
    python_exe: str,
    engine_path: Path,
    model_dir: Path,
    bin_name: str,
    meta_name: str,
    architecture: str,
    prompt_template: str,
    max_new: int,
    mode: str,
    prompts: list[dict[str, Any]],
    timeout: int,
    sparse_min_density: float | None,
    no_top_k_sort: bool,
    top_k_select: str,
    sparse_scope: str,
) -> dict[str, Any]:
    command = [
        python_exe,
        str(engine_path),
        "--bin",
        bin_name,
        "--meta",
        meta_name,
        "--architecture",
        architecture,
        "--top-k",
        mode,
        "--max-new",
        str(max_new),
        "--prompt-template",
        prompt_template,
    ]
    if sparse_min_density is not None:
        command.extend(["--sparse-min-density", str(sparse_min_density)])
    if no_top_k_sort:
        command.append("--no-top-k-sort")
    if top_k_select != "nth":
        command.extend(["--top-k-select", top_k_select])
    if sparse_scope != "all":
        command.extend(["--sparse-scope", sparse_scope])
    stdin_text = "\n".join(item["prompt"] for item in prompts) + "\nexit\n"

    completed = subprocess.run(
        command,
        cwd=str(model_dir),
        input=stdin_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )

    parsed = parse_engine_output(completed.stdout, prompts)
    return {
        "command": command,
        "cwd": str(model_dir),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "prompt_results": parsed,
        "parse_ok": len(parsed) == len(prompts),
    }


def parse_engine_output(stdout: str, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = list(STATS_RE.finditer(stdout))
    results: list[dict[str, Any]] = []
    for index, prompt_item in enumerate(prompts):
        if index >= len(matches):
            results.append(
                {
                    "prompt": prompt_item["prompt"],
                    "expected": prompt_item["expected"],
                    "forbidden": prompt_item.get("forbidden", []),
                    "max_words": prompt_item.get("max_words"),
                    "output": "",
                    "latency_ms": None,
                    "tokens_per_sec": None,
                    "matched": [],
                    "missing": prompt_item["expected"],
                    "forbidden_hit": [],
                    "word_count": 0,
                    "too_long": False,
                    "qa_pass": False,
                    "parse_error": "missing ENGINE/Stats block",
                }
            )
            continue

        match = matches[index]
        output = clean_output(match.group("text"))
        latency_ms = float(match.group("latency"))
        tokens_per_sec = float(match.group("tps"))
        matched, missing = keyword_match(output, prompt_item["expected"])
        forbidden_hit = forbidden_match(output, prompt_item.get("forbidden", []))
        word_count = output_word_count(output)
        max_words = prompt_item.get("max_words")
        too_long = max_words is not None and word_count > max_words
        results.append(
            {
                "prompt": prompt_item["prompt"],
                "expected": prompt_item["expected"],
                "forbidden": prompt_item.get("forbidden", []),
                "max_words": max_words,
                "output": output,
                "latency_ms": latency_ms,
                "tokens_per_sec": tokens_per_sec,
                "matched": matched,
                "missing": missing,
                "forbidden_hit": forbidden_hit,
                "word_count": word_count,
                "too_long": too_long,
                "qa_pass": not missing and not forbidden_hit and not too_long,
                "parse_error": None,
            }
        )
    return results


def clean_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\nUSER>\s*$", "", text).strip()
    return text


def keyword_match(output: str, expected: list[str]) -> tuple[list[str], list[str]]:
    lower = output.lower()
    matched = [keyword for keyword in expected if keyword.lower() in lower]
    missing = [keyword for keyword in expected if keyword.lower() not in lower]
    return matched, missing


def forbidden_match(output: str, forbidden: list[str]) -> list[str]:
    lower = output.lower()
    hits: list[str] = []
    for keyword in forbidden:
        key = keyword.strip()
        if not key:
            continue
        key_lower = key.lower()
        if re.fullmatch(r"[a-zA-Z]+", key):
            if re.search(rf"\b{re.escape(key_lower)}\b", lower):
                hits.append(keyword)
        elif key_lower in lower:
            hits.append(keyword)
    return hits


def output_word_count(output: str) -> int:
    return len(re.findall(r"\b[\w.-]+\b", output))


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def mode_label(mode: str) -> str:
    try:
        value = float(mode)
    except ValueError:
        value = None
    if value == 0.0:
        return f"Dense `--top-k {mode}`"
    return f"Top-K `--top-k {mode}`"


def is_dense_mode(mode: str) -> bool:
    try:
        return float(mode) == 0.0
    except ValueError:
        return False


def summarize_mode(mode: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [run for run in runs if not run["warmup"]]
    prompt_results = [
        result
        for run in measured
        for result in run["process"]["prompt_results"]
        if result.get("latency_ms") is not None and result.get("tokens_per_sec") is not None
    ]
    latencies = [float(result["latency_ms"]) for result in prompt_results]
    speeds = [float(result["tokens_per_sec"]) for result in prompt_results]
    total = len(prompt_results)
    passes = sum(1 for result in prompt_results if result.get("qa_pass"))
    return {
        "mode": mode,
        "label": mode_label(mode),
        "measured_prompt_count": total,
        "qa_pass_count": passes,
        "qa_pass_rate": (passes / total) if total else None,
        "avg_latency_ms": mean_or_none(latencies),
        "avg_tokens_per_sec": mean_or_none(speeds),
        "latency_samples_ms": latencies,
        "tokens_per_sec_samples": speeds,
        "process_failures": sum(1 for run in measured if run["process"]["returncode"] != 0),
        "parse_failures": sum(1 for run in measured if not run["process"]["parse_ok"]),
    }


def fmt_number(value: float | None, suffix: str) -> str:
    if value is None:
        return "not recorded"
    return f"{value:.2f} {suffix}"


def fmt_pass_rate(summary: dict[str, Any]) -> str:
    if summary["qa_pass_rate"] is None:
        return "not recorded"
    pct = summary["qa_pass_rate"] * 100.0
    return f"{summary['qa_pass_count']}/{summary['measured_prompt_count']} ({pct:.1f}%)"


def note_for_summary(summary: dict[str, Any]) -> str:
    notes: list[str] = []
    if summary["process_failures"]:
        notes.append(f"{summary['process_failures']} process failure(s)")
    if summary["parse_failures"]:
        notes.append(f"{summary['parse_failures']} parse failure(s)")
    if summary["qa_pass_rate"] == 1.0:
        notes.append("QA matching preserved in this run")
    elif summary["qa_pass_rate"] is not None:
        notes.append("review QA guard failures")
    if not notes:
        notes.append("review output")
    return "; ".join(notes)


def pct_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return ((value - baseline) / baseline) * 100.0


def speed_comparison(summary: dict[str, Any], dense_summary: dict[str, Any]) -> str:
    value = summary["avg_tokens_per_sec"]
    baseline = dense_summary["avg_tokens_per_sec"]
    delta = pct_delta(value, baseline)
    if value is None or baseline is None or delta is None:
        return "tokens/sec comparison not recorded"
    if abs(delta) < 0.5:
        return f"equal/no clear difference versus dense ({fmt_number(value, 'tok/s')} vs {fmt_number(baseline, 'tok/s')}, {delta:+.2f}%)"
    if delta > 0:
        return f"faster than dense ({fmt_number(value, 'tok/s')} vs {fmt_number(baseline, 'tok/s')}, {delta:+.2f}%)"
    return f"slower than dense ({fmt_number(value, 'tok/s')} vs {fmt_number(baseline, 'tok/s')}, {delta:+.2f}%)"


def latency_comparison(summary: dict[str, Any], dense_summary: dict[str, Any]) -> str:
    value = summary["avg_latency_ms"]
    baseline = dense_summary["avg_latency_ms"]
    delta = pct_delta(value, baseline)
    if value is None or baseline is None or delta is None:
        return "latency comparison not recorded"
    if abs(delta) < 0.5:
        return f"equal/no clear difference versus dense ({fmt_number(value, 'ms')} vs {fmt_number(baseline, 'ms')}, {delta:+.2f}%)"
    if delta < 0:
        return f"lower than dense ({fmt_number(value, 'ms')} vs {fmt_number(baseline, 'ms')}, {delta:+.2f}%)"
    return f"higher than dense ({fmt_number(value, 'ms')} vs {fmt_number(baseline, 'ms')}, {delta:+.2f}%)"


def qa_comparison(summary: dict[str, Any], dense_summary: dict[str, Any]) -> str:
    value = summary["qa_pass_rate"]
    baseline = dense_summary["qa_pass_rate"]
    if value is None or baseline is None:
        return "comparison not recorded"
    current = fmt_pass_rate(summary)
    dense = fmt_pass_rate(dense_summary)
    if value > baseline:
        return f"higher than dense ({current} vs {dense})"
    if value < baseline:
        return f"lower than dense ({current} vs {dense})"
    return f"preserved versus dense ({current})"


def render_interpretation(result: dict[str, Any]) -> list[str]:
    summaries = [mode_result["summary"] for mode_result in result["modes"]]
    dense_summary = next((summary for summary in summaries if is_dense_mode(summary["mode"])), None)
    topk_summaries = [summary for summary in summaries if not is_dense_mode(summary["mode"])]

    lines = ["Interpretation:", ""]
    if not topk_summaries:
        lines.append("Only dense mode was measured in this run.")
    elif dense_summary is None:
        measured = ", ".join(summary["label"] for summary in topk_summaries)
        lines.append(f"Measured Top-K modes: {measured}. No dense `--top-k 0` baseline was included, so this report does not compare Top-K speed against dense.")
    else:
        lines.append(
            "Dense baseline: {speed}; QA pass rate {pass_rate}.".format(
                speed=fmt_number(dense_summary["avg_tokens_per_sec"], "tok/s"),
                pass_rate=fmt_pass_rate(dense_summary),
            )
        )
        for summary in topk_summaries:
            lines.append(
                "- {label}: {speed}; latency {latency}; QA pass rate {qa}.".format(
                    label=summary["label"],
                    speed=speed_comparison(summary, dense_summary),
                    latency=latency_comparison(summary, dense_summary),
                    qa=qa_comparison(summary, dense_summary),
                )
            )

    lines.extend(
        [
            "",
            "Do not claim Top-K speedup unless measured average tokens/sec is actually higher than dense under the same settings.",
            "",
        ]
    )
    return lines


def render_markdown(result: dict[str, Any]) -> str:
    settings = result["settings"]
    lines = [
        f"## Automated benchmark: {result['model']}",
        "",
        "Command settings:",
        "",
        f"- Architecture: `{settings['architecture']}`",
        f"- Prompt template: `{settings['prompt_template']}`",
        f"- Max new tokens: `{settings['max_new']}`",
        f"- Repeats: `{settings['repeat']}`",
        f"- Warmup runs per mode: `{settings['warmup']}`",
        f"- Mode schedule: `{settings.get('mode_schedule', 'block')}`",
        f"- Sparse min density: `{settings.get('sparse_min_density') if settings.get('sparse_min_density') is not None else 'not set'}`",
        f"- No Top-K sort: `{settings.get('no_top_k_sort', False)}`",
        f"- Top-K selector: `{settings.get('top_k_select', 'nth')}`",
        f"- Sparse scope: `{settings.get('sparse_scope', 'all')}`",
        "",
        "| Mode | Avg latency | Avg tokens/sec | QA pass rate | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for mode_result in result["modes"]:
        summary = mode_result["summary"]
        lines.append(
            "| {label} | {latency} | {speed} | {pass_rate} | {notes} |".format(
                label=summary["label"],
                latency=fmt_number(summary["avg_latency_ms"], "ms"),
                speed=fmt_number(summary["avg_tokens_per_sec"], "tok/s"),
                pass_rate=fmt_pass_rate(summary),
                notes=note_for_summary(summary),
            )
        )
    lines.extend([""])
    lines.extend(render_interpretation(result))
    return "\n".join(lines)


def render_samples(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for mode_result in result["modes"]:
        lines.append(f"Mode: {mode_result['summary']['label']}")
        lines.append("=" * 80)
        for run in mode_result["runs"]:
            if run["warmup"]:
                continue
            lines.append(f"Repeat: {run['repeat_index']}")
            for item in run["process"]["prompt_results"]:
                lines.append("")
                lines.append(f"USER> {item['prompt']}")
                lines.append(f"ENGINE> {item['output']}")
                lines.append(
                    "QA match: {status}; matched={matched}; missing={missing}; forbidden={forbidden}; words={words}".format(
                        status="pass" if item["qa_pass"] else "fail",
                        matched=", ".join(item["matched"]) or "none",
                        missing=", ".join(item["missing"]) or "none",
                        forbidden=", ".join(item.get("forbidden_hit", [])) or "none",
                        words=(
                            f"{item.get('word_count', 0)}/{item.get('max_words')}"
                            if item.get("max_words") is not None
                            else str(item.get("word_count", 0))
                        ),
                    )
                )
            lines.append("")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    started_at_utc = utc_now()
    root = repo_root()
    model_dir = resolve_path(args.model_dir, base=root)
    engine_path = resolve_path(args.engine, base=root)
    prompts_path = resolve_path(args.prompts, base=root)
    out_dir = resolve_path(args.out_dir, base=root)

    if args.prompt_template != "qa":
        print(
            "WARNING: Leviathan instruct proof-model QA benchmarks should use --prompt-template qa.",
            file=sys.stderr,
        )

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not engine_path.is_file():
        raise FileNotFoundError(f"engine.py not found: {engine_path}")
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or greater")

    bin_path = model_dir / args.bin_name if args.bin_name else find_one(model_dir, "*.bin", "model binary")
    meta_path = model_dir / args.meta_name if args.meta_name else find_one(model_dir, "*_meta.json", "metadata JSON")
    if not bin_path.is_file():
        raise FileNotFoundError(f"Model binary not found: {bin_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")

    prompts = load_prompts(prompts_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "schema_version": 2,
        "model": model_display_name(model_dir, meta_path),
        "created_at": started_at_utc,
        "started_at_utc": started_at_utc,
        "completed_at_utc": None,
        "metadata": {
            "engine_commit": git_commit(root),
            "model_metadata_identifier": model_metadata_identifier(meta_path),
            "model_repo": args.model_repo,
            "model_revision": args.model_revision,
            "prompt_file_sha256": sha256_file(prompts_path),
            "thread_environment": thread_environment(),
        },
        "settings": {
            "architecture": args.architecture,
            "prompt_template": args.prompt_template,
            "max_new": args.max_new,
            "modes": args.modes,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "mode_schedule": args.mode_schedule,
            "sparse_min_density": args.sparse_min_density,
            "no_top_k_sort": args.no_top_k_sort,
            "top_k_select": args.top_k_select,
            "sparse_scope": args.sparse_scope,
        },
        "paths": {
            "repo_root": str(root),
            "engine": str(engine_path),
            "model_dir": str(model_dir),
            "bin": bin_path.name,
            "meta": meta_path.name,
            "prompts": str(prompts_path),
            "out_dir": str(out_dir),
        },
        "prompts": prompts,
        "modes": [],
    }

    mode_results = [{"mode": mode, "runs": []} for mode in args.modes]

    def run_one(mode: str, warmup: bool, repeat_index: int) -> dict[str, Any]:
        print(
            f"[benchmark] mode={mode} {'warmup' if warmup else 'repeat'}={repeat_index}",
            flush=True,
        )
        process = run_engine(
            python_exe=args.python,
            engine_path=engine_path,
            model_dir=model_dir,
            bin_name=bin_path.name,
            meta_name=meta_path.name,
            architecture=args.architecture,
            prompt_template=args.prompt_template,
            max_new=args.max_new,
            mode=mode,
            prompts=prompts,
            timeout=args.timeout,
            sparse_min_density=args.sparse_min_density,
            no_top_k_sort=args.no_top_k_sort,
            top_k_select=args.top_k_select,
            sparse_scope=args.sparse_scope,
        )
        return {
            "warmup": warmup,
            "repeat_index": repeat_index,
            "process": process,
        }

    if args.mode_schedule == "block":
        for mode_result in mode_results:
            mode = mode_result["mode"]
            runs = mode_result["runs"]
            total_runs = args.warmup + args.repeat
            for run_index in range(total_runs):
                warmup = run_index < args.warmup
                repeat_index = run_index - args.warmup + 1 if not warmup else run_index + 1
                runs.append(run_one(mode, warmup, repeat_index))
    else:
        for warmup_index in range(args.warmup):
            for mode_result in mode_results:
                mode_result["runs"].append(run_one(mode_result["mode"], True, warmup_index + 1))
        for repeat_index in range(1, args.repeat + 1):
            for mode_result in mode_results:
                mode_result["runs"].append(run_one(mode_result["mode"], False, repeat_index))

    for mode_result in mode_results:
        result["modes"].append(
            {
                "mode": mode_result["mode"],
                "settings": {
                    "top_k": float(mode_result["mode"]),
                    "sparse_min_density": args.sparse_min_density,
                    "no_top_k_sort": args.no_top_k_sort,
                    "top_k_select": args.top_k_select,
                    "sparse_scope": args.sparse_scope,
                },
                "runs": mode_result["runs"],
                "summary": summarize_mode(mode_result["mode"], mode_result["runs"]),
            }
        )

    results_json = out_dir / "results.json"
    results_md = out_dir / "results.md"
    samples_txt = out_dir / "sample_outputs.txt"

    result["completed_at_utc"] = utc_now()
    results_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    results_md.write_text(render_markdown(result), encoding="utf-8")
    samples_txt.write_text(render_samples(result), encoding="utf-8")

    print(f"Wrote {results_json}")
    print(f"Wrote {results_md}")
    print(f"Wrote {samples_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
