#!/usr/bin/env python3
"""Run repeatable Leviathan engine benchmarks and export JSON/Markdown results."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--modes", nargs="+", default=["0", "0.9", "0.8"], help="Top-K values to benchmark.")
    parser.add_argument("--repeat", type=int, default=3, help="Measured repeats per mode.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per mode, excluded from averages.")
    parser.add_argument("--out-dir", default="benchmark_runs/v02b", help="Output directory.")
    parser.add_argument("--bin", dest="bin_name", help="Model .bin filename inside --model-dir.")
    parser.add_argument("--meta", dest="meta_name", help="Model metadata JSON filename inside --model-dir.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run engine.py.")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds before each engine run times out.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def load_prompts(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Prompt file must contain a JSON list.")

    prompts: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt item {index} must be an object.")
        prompt = item.get("prompt")
        expected = item.get("expected", [])
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt item {index} has an invalid prompt.")
        if not isinstance(expected, list) or not all(isinstance(x, str) for x in expected):
            raise ValueError(f"Prompt item {index} has an invalid expected list.")
        prompts.append({"prompt": prompt.strip(), "expected": expected})
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
                    "output": "",
                    "latency_ms": None,
                    "tokens_per_sec": None,
                    "matched": [],
                    "missing": prompt_item["expected"],
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
        results.append(
            {
                "prompt": prompt_item["prompt"],
                "expected": prompt_item["expected"],
                "output": output,
                "latency_ms": latency_ms,
                "tokens_per_sec": tokens_per_sec,
                "matched": matched,
                "missing": missing,
                "qa_pass": not missing,
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
        notes.append("review missing keyword matches")
    if not notes:
        notes.append("review output")
    return "; ".join(notes)


def render_markdown(result: dict[str, Any]) -> str:
    settings = result["settings"]
    lines = [
        "## Automated benchmark: Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2b",
        "",
        "Command settings:",
        "",
        f"- Architecture: `{settings['architecture']}`",
        f"- Prompt template: `{settings['prompt_template']}`",
        f"- Max new tokens: `{settings['max_new']}`",
        f"- Repeats: `{settings['repeat']}`",
        f"- Warmup runs per mode: `{settings['warmup']}`",
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
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "Top-K 0.9 and 0.8 should only be described as preserving tested QA matching if the pass rate remains high. Do not claim Top-K speedup unless measured average tokens/sec is actually higher than dense under the same settings.",
            "",
        ]
    )
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
                    "QA match: {status}; matched={matched}; missing={missing}".format(
                        status="pass" if item["qa_pass"] else "fail",
                        matched=", ".join(item["matched"]) or "none",
                        missing=", ".join(item["missing"]) or "none",
                    )
                )
            lines.append("")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    root = repo_root()
    model_dir = resolve_path(args.model_dir, base=root)
    engine_path = resolve_path(args.engine, base=root)
    prompts_path = resolve_path(args.prompts, base=root)
    out_dir = resolve_path(args.out_dir, base=root)

    if args.prompt_template != "qa":
        print(
            "WARNING: v0.2b project-specific QA benchmarks should use --prompt-template qa.",
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
        "model": "Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2b",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "architecture": args.architecture,
            "prompt_template": args.prompt_template,
            "max_new": args.max_new,
            "modes": args.modes,
            "repeat": args.repeat,
            "warmup": args.warmup,
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

    for mode in args.modes:
        runs: list[dict[str, Any]] = []
        total_runs = args.warmup + args.repeat
        for run_index in range(total_runs):
            warmup = run_index < args.warmup
            repeat_index = run_index - args.warmup + 1 if not warmup else run_index + 1
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
            )
            runs.append(
                {
                    "warmup": warmup,
                    "repeat_index": repeat_index,
                    "process": process,
                }
            )

        result["modes"].append(
            {
                "mode": mode,
                "runs": runs,
                "summary": summarize_mode(mode, runs),
            }
        )

    results_json = out_dir / "results.json"
    results_md = out_dir / "results.md"
    samples_txt = out_dir / "sample_outputs.txt"

    results_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    results_md.write_text(render_markdown(result), encoding="utf-8")
    samples_txt.write_text(render_samples(result), encoding="utf-8")

    print(f"Wrote {results_json}")
    print(f"Wrote {results_md}")
    print(f"Wrote {samples_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
