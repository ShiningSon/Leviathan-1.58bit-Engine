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

# Leviathan-MLGRU-100M-TinyStories-Instruct-v08a

This is an experimental local CPU proof model for the Leviathan MLGRU runtime.
It is not a general assistant.

This repository contains a Leviathan runtime package, not a standard Transformers checkpoint. Use the Leviathan `engine.py` runtime to run the model locally.

## Model Summary

- Model package: `leviathan_mlgru_100m_instruct_v08a`
- Status: experimental local CPU proof model
- Architecture: MLGRU
- Format: native ternary / 1.58-bit / 2-bit packed Leviathan format
- Runtime: Leviathan `engine.py`
- Prompt template: `qa`
- Dataset base: TinyStories plus supervised Leviathan project QA

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
| Dense `--top-k 0` | 101.15 ms | 176.73 tok/s | 570/600 (95.0%) | Dense baseline |
| Top-K `--top-k 0.06` histogram down-only | 90.39 ms | 191.14 tok/s | 600/600 (100.0%) | Current 100M experimental local CPU speed candidate |
| Top-K `--top-k 0.08` histogram down-only | 91.13 ms | 189.82 tok/s | 600/600 (100.0%) | Preserved 600/600 strict QA in this run |

Interpretation:

Top-K `0.06` histogram down-only is the current v08a 100M experimental local CPU speed candidate. It improved latency by about 10.64% and tokens/sec by about 8.15% versus dense in this repeat-30 interleaved local CPU run. Strict QA improved from 570/600 to 600/600 in the measured run.

Top-K `0.08` also preserved 600/600 strict QA and improved tokens/sec by about 7.41% versus dense in the same run.

This is not a general sparse speedup claim. Top-K is not always faster. These results are local CPU-specific, and larger-model or other-hardware scaling is not automatically proven.

## Local Usage

Run from the Leviathan repository root after placing the model package folder beside `engine.py`.

Dense QA check:

```cmd
echo What is Leviathan? | python engine.py --bin .\leviathan_mlgru_100m_instruct_v08a\leviathan_mlgru_100m_instruct_v08a.bin --meta .\leviathan_mlgru_100m_instruct_v08a\leviathan_mlgru_100m_instruct_v08a_meta.json --architecture mlgru --prompt-template qa --max-new 80 --top-k 0 --profile
```

Sparse candidate check:

```cmd
echo What is Leviathan? | python engine.py --bin .\leviathan_mlgru_100m_instruct_v08a\leviathan_mlgru_100m_instruct_v08a.bin --meta .\leviathan_mlgru_100m_instruct_v08a\leviathan_mlgru_100m_instruct_v08a_meta.json --architecture mlgru --prompt-template qa --max-new 80 --top-k 0.06 --sparse-scope down --sparse-min-density 0.6 --no-top-k-sort --top-k-select histogram --profile
```

## Package Contents

Expected files:

```text
README.md
leviathan_mlgru_100m_instruct_v08a.bin
leviathan_mlgru_100m_instruct_v08a_meta.json
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
