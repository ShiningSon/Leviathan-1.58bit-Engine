---
license: mit
language:
  - en
pipeline_tag: text-generation
library_name: pytorch
datasets:
  - roneneldan/TinyStories
tags:
  - text-generation
  - recurrent
  - mlgru
  - ternary
  - bitnet
  - 1.58bit
  - cpu-inference
  - tiny-stories
  - proof-of-concept
  - research
inference: false
---

# Leviathan-MLGRU-70M-TinyStories-Instruct-v07b

This is an experimental local CPU proof model for the Leviathan MLGRU runtime.
It is not a general assistant.

This repository contains a Leviathan runtime package, not a standard Transformers checkpoint. Use the Leviathan `engine.py` runtime to run the model locally.

## Model Summary

- Model package: `leviathan_mlgru_70m_instruct_v07b`
- Status: experimental local CPU proof model
- Architecture: MLGRU
- Format: native ternary / 1.58-bit / 2-bit packed Leviathan format
- Runtime: Leviathan `engine.py`
- Prompt template: `qa`
- Dataset base: TinyStories plus supervised Leviathan project QA repair

This package is not Transformers-compatible as a general LLM checkpoint. Hugging Face hosted inference is disabled because the model requires the Leviathan runtime.

## Benchmark

Benchmark settings:

- Architecture: `mlgru`
- Prompt template: `qa`
- Max new tokens: `80`
- Repeats: `30`
- Mode schedule: `interleave`
- Sparse min density: `0.6`
- No Top-K sort: `True`
- Top-K selector: `histogram`
- Sparse scope: `down`

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate | Notes |
|---|---:|---:|---:|---|
| Dense `--top-k 0` | 72.85 ms | 238.50 tok/s | 600/600 (100.0%) | Dense baseline |
| Top-K `--top-k 0.08` histogram down-only | 68.54 ms | 254.12 tok/s | 600/600 (100.0%) | Current 70M experimental local CPU speed candidate |

Interpretation:

Top-K `0.08` histogram down-only is the v07b 70M experimental local CPU speed candidate. It improved latency by about 5.92% and tokens/sec by about 6.55% versus dense in this repeat-30 interleaved local CPU run while preserving the measured strict QA pass rate.

This is not a general sparse speedup claim. Top-K is not always faster. These results are local CPU-specific, and larger-model or other-hardware scaling is not automatically proven.

## Local Usage

Run from the Leviathan repository root after placing the model package folder beside `engine.py`.

Dense QA check:

```cmd
echo What is Leviathan? | python engine.py --bin .\leviathan_mlgru_70m_instruct_v07b\leviathan_mlgru_70m_instruct_v07b.bin --meta .\leviathan_mlgru_70m_instruct_v07b\leviathan_mlgru_70m_instruct_v07b_meta.json --architecture mlgru --prompt-template qa --max-new 80 --top-k 0 --profile
```

Sparse candidate check:

```cmd
echo What is Leviathan? | python engine.py --bin .\leviathan_mlgru_70m_instruct_v07b\leviathan_mlgru_70m_instruct_v07b.bin --meta .\leviathan_mlgru_70m_instruct_v07b\leviathan_mlgru_70m_instruct_v07b_meta.json --architecture mlgru --prompt-template qa --max-new 80 --top-k 0.08 --sparse-scope down --sparse-min-density 0.6 --no-top-k-sort --top-k-select histogram --profile
```

## Package Contents

Expected files:

```text
README.md
leviathan_mlgru_70m_instruct_v07b.bin
leviathan_mlgru_70m_instruct_v07b_meta.json
leviathan_mlgru_tokenizer/
report.json
sample_outputs.txt
```

## Limitations

- This is a proof package for validating Leviathan training, export, and CPU runtime behavior.
- It is not a general assistant.
- It is not a standard Transformers checkpoint.
- It requires the Leviathan runtime.
- The sparse result is an experimental local CPU result, not a general sparse speedup claim.
- Top-K is not always faster.
- Larger-model or other-hardware scaling is not automatically proven.
