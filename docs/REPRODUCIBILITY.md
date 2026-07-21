# Reproducibility

This document describes how to reproduce the Leviathan v0.9 proof-model workflow and benchmark method. It does not promise identical timings on every CPU.

## Training environment

The versioned training paths run on Modal and declare their Python packages inside each script. The v09a 200M route includes H100 and H200 variants under `training/`, with exact values in `training/configs/`. The original L40S attempt ran out of memory; the accelerated routes preserve the MLGRU architecture and export contract while adjusting batch size and step count.

Record the selected config, GPU class, package versions, training logs, final checkpoint, and exported `report.json`. Training data is TinyStories plus the committed project-specific QA mixture. These models remain narrow proof models and are not general assistants.

## Export format

The training route exports a Leviathan v2 runtime package:

```text
<package>.bin
<package>_meta.json
leviathan_mlgru_tokenizer/
report.json
sample_outputs.txt
```

Linear weights are row-major packed ternary values. Embeddings, normalization weights, recurrent tensors, and other structural values are recorded in the package binary and metadata. This is not a standard Transformers checkpoint; use `engine.py --architecture mlgru`.

The v09a public package is pinned by revision, byte size, and SHA-256 in [`releases/v0.9.0_hf_manifest.json`](../releases/v0.9.0_hf_manifest.json). Verify a clean public download without authentication:

```cmd
python scripts\verify_hf_release.py --download-dir hf_verify_v09a
```

## Benchmark prompt set

The strict QA regression set is [`benchmarks/prompts_v02b_qa.json`](../benchmarks/prompts_v02b_qa.json). It contains 20 project prompts. Each item may define:

- `expected`: required case-insensitive terms;
- `forbidden`: terms that invalidate contaminated or contradictory output;
- `max_words`: an answer-length guard.

A prompt passes only when all expected terms are present, no forbidden term is present, and the output stays within `max_words`. This catches residual TinyStories-style continuation that a required-keyword-only score could miss. It is a regression set, not a broad language-model evaluation.

## Timing policy

Confirmed scaling records use:

- `--prompt-template qa`;
- `--max-new 80`;
- one untimed warmup per mode;
- repeated measurements over the same 20 prompts;
- `--mode-schedule interleave` to reduce mode-order bias;
- the same local engine, package, prompt set, and process settings for dense and sparse modes.

The 30M record uses 50 repeats. The 70M, 100M, and 200M records use 30 repeats. The benchmark runner parses the engine's `[Stats: ...]` lines and writes local JSON, Markdown, and sample-output artifacts under `benchmark_runs/`.

## Sparse candidate method

The current sparse path uses:

```text
--sparse-min-density 0.6
--no-top-k-sort
--sparse-scope down
--top-k-select histogram
```

The histogram selector operates on INT8 activation magnitude buckets. `--sparse-scope down` limits Top-K to the FFN down projection while keeping recurrent/state projections dense. Recommended density is model-specific: v02g `0.12`, v07b `0.08`, v08a `0.06`, and v09a `0.10`.

## v09a reproduction command

Run from the repository root after downloading the package:

```bash
python scripts/benchmark_engine.py --model-dir ./leviathan_mlgru_200m_instruct_v09a --engine ./engine.py --prompts ./benchmarks/prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.08 0.10 0.12 --repeat 30 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir ./benchmark_runs/v09a_200m_histogram_candidates_repeat30
```

Review `results.md` and `sample_outputs.txt`, then compare the measured summary with [`benchmarks/results/scaling_summary.json`](../benchmarks/results/scaling_summary.json). Do not commit the generated directory.

## Hardware-specific interpretation

The committed results were observed on the Windows CPU environment documented in [`BENCHMARK.md`](../BENCHMARK.md). Compiler version, OpenMP scheduling, CPU frequency, thermals, memory, and background load affect timing. A valid reproduction follows the same method and reports its environment; it need not reproduce the exact milliseconds.

Do not claim a general sparse speedup from these records. Top-K `0.10` is the current v09a local CPU candidate because it preserved 600/600 strict QA and improved measured tokens/sec in the same repeat-30 interleaved run. Top-K `0.08` measured faster but is not recommended because QA fell to 570/600.
