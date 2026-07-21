# v0.9a 200M MLGRU speed candidate

This is the completed 200M-class scaling probe for the Leviathan MLGRU path. It confirms that the histogram Top-K down-only sparse speed margin observed at 30M, 70M, and 100M continues in the v09a local CPU benchmark.

This is a local CPU experimental proof-model result, not a general sparse speedup claim. Do not claim Top-K is always faster, and do not claim other hardware or larger-model scaling is automatically proven. The reviewed v09a package is published at [ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a](https://huggingface.co/ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a).

## Purpose

The current confirmed scaling pattern is:

| Model | Dense | Best Top-K | Strict QA | Interpretation |
|---|---:|---:|---:|---|
| 30M v0.6 / v02g route | 50.74 ms / 350.97 tok/s | Top-K `0.12`: 50.25 ms / 354.17 tok/s | 950/1000 (95.0%) both modes | Small local CPU candidate |
| 70M v07b | 72.85 ms / 238.50 tok/s | Top-K `0.08`: 68.54 ms / 254.12 tok/s | 600/600 (100.0%) both modes | Stronger 70M local CPU candidate |
| 100M v08a | 101.15 ms / 176.73 tok/s | Top-K `0.06`: 90.39 ms / 191.14 tok/s | Dense 570/600 (95.0%); Top-K 600/600 (100.0%) | Confirmed 100M local CPU candidate |
| 200M v09a | 193.92 ms / 88.30 tok/s | Top-K `0.10`: 175.33 ms / 97.54 tok/s | 600/600 (100.0%) both modes | Current active 200M local CPU candidate |

v09a tested a 200M-class MLGRU model with the same strict QA benchmark discipline and histogram down-only sparse path. Top-K `0.10` is the confirmed v09a candidate density.

## Final Result

Repeat-30 settings:

```text
Architecture: mlgru
Prompt template: qa
Max new tokens: 80
Repeats: 30
Warmup runs per mode: 1
Mode schedule: interleave
Sparse min density: 0.6
No Top-K sort: True
Top-K selector: histogram
Sparse scope: down
```

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate | Notes |
|---|---:|---:|---:|---|
| Dense `--top-k 0` | 193.92 ms | 88.30 tok/s | 600/600 (100.0%) | Dense baseline |
| Top-K `--top-k 0.08` histogram down-only | 173.07 ms | 98.80 tok/s | 570/600 (95.0%) | Faster but not recommended because QA dropped |
| Top-K `--top-k 0.10` histogram down-only | 175.33 ms | 97.54 tok/s | 600/600 (100.0%) | Confirmed v09a 200M local CPU speed candidate |
| Top-K `--top-k 0.12` histogram down-only | 177.04 ms | 96.57 tok/s | 600/600 (100.0%) | Preserved QA, slightly slower than 0.10 |

Top-K `0.10` improves tokens/sec by +10.46% and reduces latency by -9.58% versus dense while preserving strict QA at 600/600. Top-K `0.08` is faster but not recommended because strict QA drops to 570/600. Top-K `0.12` is also valid but slower than `0.10`.

v09b QA repair is not required for the current 20-prompt strict QA benchmark set. The next work is broader validation, additional CPU hardware, and optional larger-model work only after budget allows.

## Target Package

```text
leviathan_mlgru_200m_instruct_v09a
```

Published Hugging Face repository: https://huggingface.co/ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a

## Architecture

```text
architecture: mlgru
vocab_size: 8192
hidden_size: 1280
n_layers: 10
intermediate_size: 4096
seq_len: 96
heads: 1
kv_heads: 1
activation: relu2
tokenizer: reuse the v08a tokenizer path when present; otherwise train the same local tokenizer style
```

Estimated trainable parameters:

```text
233,388,800
```

This is about 233.39M trainable parameters using the existing `estimate_mlgru_params` formula. The estimate assumes tied input/output embeddings, seven ternary linear projections per layer, four RMSNorm vectors per layer, and one final RMSNorm vector.

## Config

The v09a run is configured in:

```text
training\configs\v09a_200m_mlgru.json
```

The config sets `base_ckpt_path` to an empty string because the v08a hidden/intermediate shapes are incompatible with the 200M-class architecture. The run starts from scratch while reusing the v08a tokenizer path when available.

Key values:

```text
run_name: leviathan_mlgru_200m_instruct_v09a
dataset: tinystories
vocab_size: 8192
hidden_size: 1280
n_layers: 10
intermediate_size: 4096
seq_len: 96
batch_size: 4
steps: 2200
lr: 0.0003
max_train_tokens: 4000000
tokenizer_docs: 10000
qa_ratio: 0.85
qa_seed_path: /root/instruction_qa_supervised_v02f.jsonl
base_ckpt_path:
base_tokenizer_dir: /data/exports/leviathan_mlgru_100m_instruct_v08a/leviathan_mlgru_tokenizer
training_stage: v0.9a-200m-scaling-probe
model_display_name: Leviathan-MLGRU-200M-TinyStories-Instruct-v0.9a
```

Accelerated v09a configs:

```text
training\configs\v09a_200m_mlgru_h100_fast.json
training\configs\v09a_200m_mlgru_h200_fast.json
training\configs\v09a_200m_mlgru_h100_safe.json
```

## GPU Memory And Batch-Size Plan

The original L40S path failed with CUDA OOM at `batch_size: 4` for the 200M-class v09a configuration. The failed run reported about 44.39 GiB usable GPU memory, about 44.37 GiB already in use, and then failed when trying to allocate another 20 MiB.

H100 is the first accelerated retry target. The primary H100 config uses `batch_size: 6` instead of retrying `batch_size: 4`, because the larger GPU should be used for better throughput and lower wall-clock time while keeping the same proof-model training recipe.

H200 is optional and uses `batch_size: 10` first. If H100 fast is unstable or still memory-bound, the H100 safe fallback returns to `batch_size: 4`.

Do not use `H100:2` or `H200:2` with the current script. The training route is single-process and single-GPU; no DDP or FSDP path is implemented.

The larger batches are for throughput. The learning rate remains `0.0003` to avoid introducing unnecessary QA instability. If dense QA drops below 90%, run a v09b QA repair stage rather than over-interpreting sparse results.

H100 fast config:

```text
batch_size: 6
steps: 1800
lr: 0.0003
training_stage: v0.9a-200m-scaling-probe-h100-fast
```

H200 fast config:

```text
batch_size: 10
steps: 1400
lr: 0.0003
training_stage: v0.9a-200m-scaling-probe-h200-fast
```

H100 safe fallback config:

```text
batch_size: 4
steps: 2200
lr: 0.0003
training_stage: v0.9a-200m-scaling-probe-h100-safe
```

## Dry-Run Parameter Check

Run from the repository root:

```cmd
python -m modal run training\08_finetune_supervised_qa_mlgru_modal_H100_v09.py --config-json training\configs\v09a_200m_mlgru_h100_fast.json --dry-run
```

Expected output should include:

```text
[estimate] params=233,388,800
```

## Training And Export Command

H100 fast path:

```cmd
python -m modal run training\08_finetune_supervised_qa_mlgru_modal_H100_v09.py --config-json training\configs\v09a_200m_mlgru_h100_fast.json
```

H100 safe fallback:

```cmd
python -m modal run training\08_finetune_supervised_qa_mlgru_modal_H100_v09.py --config-json training\configs\v09a_200m_mlgru_h100_safe.json
```

H200 fast path:

```cmd
python -m modal run training\08_finetune_supervised_qa_mlgru_modal_H200_v09.py --config-json training\configs\v09a_200m_mlgru_h200_fast.json
```

## Download Exported Package

```cmd
python -m modal volume get leviathan-mlgru /exports/leviathan_mlgru_200m_instruct_v09a.zip leviathan_mlgru_200m_instruct_v09a.zip
```

Unpack locally into a package folder:

```cmd
rmdir /S /Q leviathan_mlgru_200m_instruct_v09a 2>nul
mkdir leviathan_mlgru_200m_instruct_v09a
tar -xf leviathan_mlgru_200m_instruct_v09a.zip -C leviathan_mlgru_200m_instruct_v09a
```

Expected local package contents:

```text
leviathan_mlgru_200m_instruct_v09a.bin
leviathan_mlgru_200m_instruct_v09a_meta.json
leviathan_mlgru_tokenizer\
report.json
sample_outputs.txt
```

## Smoke Load Command

```cmd
echo What is Leviathan? | python engine.py --bin .\leviathan_mlgru_200m_instruct_v09a\leviathan_mlgru_200m_instruct_v09a.bin --meta .\leviathan_mlgru_200m_instruct_v09a\leviathan_mlgru_200m_instruct_v09a_meta.json --architecture mlgru --prompt-template qa --max-new 80 --top-k 0 --profile
```

## Benchmark Commands

Dense probe:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_200m_instruct_v09a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 --repeat 10 --mode-schedule interleave --out-dir .\benchmark_runs\v09a_200m_dense_probe
```

Histogram down-only quality recovery sweep:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_200m_instruct_v09a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.08 0.10 0.12 0.15 0.20 --repeat 10 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir .\benchmark_runs\v09a_200m_quality_recovery_sweep_repeat10
```

Confirmed repeat-30 candidates:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_200m_instruct_v09a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.08 0.10 0.12 --repeat 30 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir .\benchmark_runs\v09a_200m_histogram_candidates_repeat30
```

## Interpretation Rules

- Dense and Top-K `0.10` both reached 600/600 strict QA in the confirmed repeat-30 run.
- v09b QA repair is not required for the current benchmark set.
- Top-K `0.10` is the representative v09a speed candidate.
- Top-K `0.08` should not be used as the recommended mode because QA drops to 570/600.
- Do not claim general sparse speedup.
- Do not claim that Top-K is always faster.
- Do not claim larger-model scaling until measured.
- Test additional CPU hardware before treating the result as robust.

## Caveats

- v09a is a proof-model speed candidate, not a general assistant.
- The TinyStories plus small project QA mixture may be too narrow at this scale.
- Review the export, smoke test, benchmark output, and model card before any future Hugging Face package revision.
- Do not commit generated `benchmark_runs/`, model folders, `.bin`, `.zip`, checkpoints, or cache files to GitHub.
