# Benchmark automation

Leviathan benchmark results should be repeatable enough to compare dense mode with Top-K experiments under the same prompt set and runtime settings. The benchmark runner records the prompts, generated answers, latency, tokens/sec, and simple expected-keyword matching so new results can be reviewed before they are copied into `BENCHMARK.md`.

## v0.2b prompt template

`Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2b` must be run with `--prompt-template qa` for project-specific QA prompts.

Use `--prompt-template plain` only for raw continuation experiments. Plain mode can break the QA matching behavior that v0.2b was trained to demonstrate.

Dense mode is the default recommendation for the current 30M model. Top-K `0.9` and `0.8` preserved the tested QA mappings in local checks, but do not claim Top-K speedup unless the measured average tokens/sec is actually higher than dense under the same settings.

The v0.2b prompt set has been expanded from the initial 4 prompts to a small 20-prompt regression set. It covers core concepts, runtime modes, prompt-template usage, model packaging, and GitHub/Hugging Face distribution. It is still not a full model evaluation benchmark.

For v0.4 sparse fallback checks, pass `--sparse-min-density 0.6` through the benchmark runner. This lets high-density Top-K modes fall back to the dense kernel instead of forcing the sparse path. Treat this as a runtime guardrail; do not describe it as a Top-K speedup unless measured average tokens/sec is higher than dense under the same settings.

## Example

From the repository root:

```powershell
python scripts/benchmark_engine.py `
  --model-dir .\leviathan_mlgru_30m_instruct_v02b `
  --engine .\engine.py `
  --prompts .\benchmarks\prompts_v02b_qa.json `
  --architecture mlgru `
  --prompt-template qa `
  --max-new 80 `
  --modes 0 0.9 0.8 `
  --repeat 3 `
  --sparse-min-density 0.6 `
  --out-dir .\benchmark_runs\v02b
```

By default, the runner performs one warmup run per mode before the measured repeats. Change this with `--warmup 0` or another integer.

## Output files

The runner writes:

```text
benchmark_runs/v02b/results.json
benchmark_runs/v02b/results.md
benchmark_runs/v02b/sample_outputs.txt
```

`results.json` contains the full structured benchmark record, including command settings, per-run process results, per-prompt outputs, parsed latency, parsed tokens/sec, and keyword-match details.

`results.md` is a review-friendly Markdown summary that can be copied into `BENCHMARK.md` after checking the generated answers.

`sample_outputs.txt` contains the prompt and answer text grouped by mode and repeat.

## Updating BENCHMARK.md

Run the benchmark, open `benchmark_runs/v02b/results.md`, and review the table plus sample outputs. If the outputs are sane, copy the Markdown summary into `BENCHMARK.md` in a separate documentation commit.

Do not copy generated files from `benchmark_runs/` into Git history. The folder is ignored because benchmark output is machine-local and can grow over time.

## Top-K interpretation

Top-K `0.9` and `0.8` should only be described as preserving tested QA matching if the pass rate remains high on the measured prompt set. Do not claim Top-K speedup unless measured average tokens/sec is actually higher than dense under the same architecture, prompt template, max-new setting, prompt set, and repeat count.
