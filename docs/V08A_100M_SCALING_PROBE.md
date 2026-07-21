# v0.8a 100M MLGRU scaling probe

This is the first 100M-class scaling probe for the Leviathan MLGRU path. It tests whether the v07b 70M experimental local CPU speed candidate scales to a larger MLGRU proof model.

This is a scaling probe, not a final model and not a general assistant. v08a is published on Hugging Face as an experimental Leviathan runtime package, but do not claim general sparse speedup from v08a unless strict QA and measured throughput support it.

## Current baseline

The latest confirmed benchmark is v07b 70M:

```text
Model: leviathan_mlgru_70m_instruct_v07b
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

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate |
|---|---:|---:|---:|
| Dense `--top-k 0` | 72.85 ms | 238.50 tok/s | 600/600 (100.0%) |
| Top-K `--top-k 0.08` | 68.54 ms | 254.12 tok/s | 600/600 (100.0%) |
| Top-K `--top-k 0.10` | 68.84 ms | 252.97 tok/s | 600/600 (100.0%) |
| Top-K `--top-k 0.12` | 69.48 ms | 250.62 tok/s | 600/600 (100.0%) |

v07b is the 70M experimental local CPU speed candidate. Top-K `0.08` histogram down-only improved tokens/sec by about 6.55% versus dense while preserving strict QA in repeat-30 interleaved testing. This is not a general sparse speedup claim, not a claim that Top-K is always faster, and not proof that similar results automatically scale to larger models or other hardware.

## v08a confirmed repeat-30 result

v08a is the first 100M MLGRU proof model where histogram Top-K down-only sparse inference produced a repeat-30 local CPU speed candidate.

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

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate |
|---|---:|---:|---:|
| Dense `--top-k 0` | 101.15 ms | 176.73 tok/s | 570/600 (95.0%) |
| Top-K `--top-k 0.06` | 90.39 ms | 191.14 tok/s | 600/600 (100.0%) |
| Top-K `--top-k 0.08` | 91.13 ms | 189.82 tok/s | 600/600 (100.0%) |
| Top-K `--top-k 0.10` | 93.45 ms | 189.88 tok/s | 570/600 (95.0%) |

Top-K `0.06` histogram down-only is the confirmed v08a 100M experimental local CPU speed candidate. It improved latency by about 10.64% and tokens/sec by about 8.15% versus dense in this repeat-30 interleaved run. Strict QA improved from 570/600 to 600/600 in the measured run.

Top-K `0.08` also preserved 600/600 QA and improved tokens/sec by about 7.41%. This is a local CPU 100M proof-model result. It is not a general sparse speedup claim, not a claim that Top-K is always faster, and not evidence that the result automatically scales to larger models or other hardware.

Known dense limitation:

```text
Dense mode still shows the familiar package-description failure: "What files are inside a Leviathan MLGRU model package?" It mentions .bin and metadata JSON but misses tokenizer and falls into a TinyStories-style continuation.
```

## Target package

```text
leviathan_mlgru_100m_instruct_v08a
```

## Architecture

```text
architecture: mlgru
vocab_size: 8192
layers: 8
hidden_size: 896
intermediate_size: 3072
heads: 1
kv_heads: 1
activation: relu2
tokenizer: reuse the v07b tokenizer path when present; otherwise train the same local tokenizer style
```

Estimated trainable parameters:

```text
99,137,408
```

This is about 99.14M trainable parameters using the existing `estimate_mlgru_params` formula. The estimate assumes tied input/output embeddings, seven ternary linear projections per layer, four RMSNorm vectors per layer, and one final RMSNorm vector.

## Config

The v08a run is configured in:

```text
training\configs\v08a_100m_mlgru.json
```

The config sets `base_ckpt_path` to an empty string because the v07b 70M checkpoint shape is incompatible with hidden size 896 and intermediate size 3072. The run starts from scratch while reusing the v07b tokenizer path or tokenizer strategy when available.

## Dry-run parameter check

Run from the repository root:

```cmd
python -m modal run training\07_finetune_supervised_qa_mlgru_modal_L40S_v02f.py --config-json training\configs\v08a_100m_mlgru.json --dry-run
```

Expected output includes:

```text
[estimate] params=99,137,408
```

## Training and export command

```cmd
python -m modal run training\07_finetune_supervised_qa_mlgru_modal_L40S_v02f.py --config-json training\configs\v08a_100m_mlgru.json
```

## Download exported package

```cmd
python -m modal volume get leviathan-mlgru /exports/leviathan_mlgru_100m_instruct_v08a.zip leviathan_mlgru_100m_instruct_v08a.zip
```

Unpack locally into a package folder:

```cmd
rmdir /S /Q leviathan_mlgru_100m_instruct_v08a 2>nul
mkdir leviathan_mlgru_100m_instruct_v08a
tar -xf leviathan_mlgru_100m_instruct_v08a.zip -C leviathan_mlgru_100m_instruct_v08a
```

Expected local package contents:

```text
leviathan_mlgru_100m_instruct_v08a.bin
leviathan_mlgru_100m_instruct_v08a_meta.json
leviathan_mlgru_tokenizer\
report.json
sample_outputs.txt
```

## Smoke load command

```cmd
echo What is Leviathan? | python engine.py --bin .\leviathan_mlgru_100m_instruct_v08a\leviathan_mlgru_100m_instruct_v08a.bin --meta .\leviathan_mlgru_100m_instruct_v08a\leviathan_mlgru_100m_instruct_v08a_meta.json --architecture mlgru --prompt-template qa --max-new 80 --top-k 0 --profile
```

## Benchmark commands

Dense probe:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_100m_instruct_v08a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 --repeat 10 --mode-schedule interleave --out-dir .\benchmark_runs\v08a_100m_dense_probe
```

Histogram down-only sweep:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_100m_instruct_v08a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.06 0.08 0.10 0.12 --repeat 10 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir .\benchmark_runs\v08a_100m_histogram_sweep_repeat10
```

If repeat-10 is promising:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_100m_instruct_v08a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.06 0.08 0.10 0.12 --repeat 30 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir .\benchmark_runs\v08a_100m_histogram_sweep_repeat30
```

Confirmed repeat-30 candidate run:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_100m_instruct_v08a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.06 0.08 0.10 --repeat 30 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir .\benchmark_runs\v08a_100m_histogram_candidate_repeat30
```

## Interpretation rules

- If dense QA is below 90%, v08a needs v08b QA repair before sparse speed conclusions.
- If dense QA is 90-95%, sparse sweep results should be treated as a scaling signal only.
- If dense QA is 95-100% and Top-K preserves QA while beating dense tokens/sec, v08a becomes the 100M speed candidate.
- The best Top-K density may shift lower than `0.08` because `intermediate_size` is larger.
- Do not claim general sparse speedup.
- Do not claim that Top-K is always faster.
- Do not claim that this result automatically scales to larger models or other hardware.
- The subsequent v09a work completed the 200M-class MLGRU scaling probe.
- Remaining validation targets are additional CPU hardware and broader QA coverage.
- At 200M+ scale, QA/instruction data likely needs to expand beyond TinyStories plus the small project QA seed.
