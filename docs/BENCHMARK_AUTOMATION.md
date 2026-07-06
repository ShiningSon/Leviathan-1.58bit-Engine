# Benchmark automation

Leviathan benchmark results should be repeatable enough to compare dense mode with Top-K experiments under the same prompt set and runtime settings. The benchmark runner records the prompts, generated answers, latency, tokens/sec, and simple expected-keyword matching so new results can be reviewed before they are copied into `BENCHMARK.md`.

## v0.2 instruct prompt template

`Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2b` and later v02 instruct proof-model candidates such as v02g must be run with `--prompt-template qa` for project-specific QA prompts.

Use `--prompt-template plain` only for raw continuation experiments. Plain mode can break the QA matching behavior that v0.2b was trained to demonstrate.

Dense mode remains the stable default for the published 30M Hugging Face proof package. The current active local speed candidate is v08a 100M with histogram down-only Top-K `0.06`, but do not claim Top-K speedup unless the measured average tokens/sec is actually higher than dense under the same settings.

The v0.2b prompt set has been expanded from the initial 4 prompts to a small 20-prompt regression set. It covers core concepts, runtime modes, prompt-template usage, model packaging, and GitHub/Hugging Face distribution. It is still not a full model evaluation benchmark.

For v0.4 sparse fallback checks, pass `--sparse-min-density 0.6` through the benchmark runner. This lets high-density Top-K modes fall back to the dense kernel instead of forcing the sparse path. Treat this as a runtime guardrail; do not describe it as a Top-K speedup unless measured average tokens/sec is higher than dense under the same settings.

For experimental no-sort Top-K checks, add `--no-top-k-sort`. This forwards the engine option that skips the final index sort after Top-K selection. It is disabled by default, and low-density settings should be treated as experimental unless QA matching remains stable.

The strict QA prompt schema supports `expected`, `forbidden`, and `max_words`. Required keywords must be present, forbidden keywords must be absent, and outputs must stay within the configured word budget. `sample_outputs.txt` records forbidden hits and word counts for review.

For sparse-scope checks, use `--sparse-scope down` to restrict Top-K sparsity to the down projection while keeping recurrent/state projections dense.

For v0.6 and later speed-candidate checks, use `--top-k-select histogram` with `--mode-schedule interleave` so dense and Top-K modes are alternated by repeat instead of measured in large mode blocks. Treat each result as experimental and local-CPU-specific until it is repeated on additional hardware and larger models.

## CMD examples

From the repository root:

Current v08a 100M candidate check:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_100m_instruct_v08a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.06 0.08 0.10 --repeat 30 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir .\benchmark_runs\v08a_100m_histogram_candidate_repeat30
```

Historical high-density fallback check:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_30m_instruct_v02b --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.9 0.8 0.5 --repeat 3 --sparse-min-density 0.6 --out-dir .\benchmark_runs\v04_sparse_fallback
```

No-sort experiment example:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_30m_instruct_v02b --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.5 0.3 --repeat 3 --sparse-min-density 0.6 --no-top-k-sort --out-dir .\benchmark_runs\v04_topk_nosort
```

Strict QA sparse-scope example:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_30m_instruct_v02b --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.18 0.2 0.25 --repeat 10 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --out-dir .\benchmark_runs\v05_sparse_scope_down
```

v02g dense strict QA example:

Published 30M v02g HF package: https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02g

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_30m_instruct_v02g --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 --repeat 10 --out-dir .\benchmark_runs\v02g_dense_strict_qa
```

v02e down-only sparse-scope comparison example:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_30m_instruct_v02e --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.18 0.2 0.25 --repeat 10 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --out-dir .\benchmark_runs\v02e_sparse_down_strict_qa
```

v0.6 histogram Top-K interleaved candidate example:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_30m_instruct_v02g --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.12 --repeat 50 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir .\benchmark_runs\v06_histogram_topk_candidate
```

v07b 70M candidate example:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_70m_instruct_v07b --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.08 --repeat 30 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir .\benchmark_runs\v07b_70m_histogram_candidate_repeat30
```

By default, the runner performs one warmup run per mode before the measured repeats. Change this with `--warmup 0` or another integer.

## Output files

The runner writes:

```text
benchmark_runs/v04_sparse_fallback/results.json
benchmark_runs/v04_sparse_fallback/results.md
benchmark_runs/v04_sparse_fallback/sample_outputs.txt
```

`results.json` contains the full structured benchmark record, including command settings, per-run process results, per-prompt outputs, parsed latency, parsed tokens/sec, and keyword-match details.

`results.md` is a review-friendly Markdown summary that can be copied into `BENCHMARK.md` after checking the generated answers.

`sample_outputs.txt` contains the prompt and answer text grouped by mode and repeat.

## Updating BENCHMARK.md

Run the benchmark, open `benchmark_runs/v04_sparse_fallback/results.md`, and review the table plus sample outputs. If the outputs are sane, copy the Markdown summary into `BENCHMARK.md` in a separate documentation commit.

Do not copy generated files from `benchmark_runs/` into Git history. The folder is ignored because benchmark output is machine-local and can grow over time.

## Top-K interpretation

`results.md` now generates its interpretation from the actual `--modes` used for the run. When dense `--top-k 0` is present, every nonzero Top-K mode is compared against dense for tokens/sec, latency, and QA pass rate. Do not claim Top-K speedup unless measured average tokens/sec is actually higher than dense under the same architecture, prompt template, max-new setting, prompt set, and repeat count.
