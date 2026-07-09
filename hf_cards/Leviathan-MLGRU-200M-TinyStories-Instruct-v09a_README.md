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

# Leviathan-MLGRU-200M-TinyStories-Instruct-v09a

This is an experimental local CPU proof model for the Leviathan MLGRU runtime.
It is not a general assistant.

This repository contains a Leviathan runtime package, not a standard Transformers checkpoint. Use the Leviathan `engine.py` runtime to run the model locally.

## Model Summary

- Model package: `leviathan_mlgru_200m_instruct_v09a`
- Status: experimental local CPU proof model
- Architecture: Leviathan MLGRU
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
- Warmup runs per mode: `1`
- Mode schedule: `interleave`
- Sparse min density: `0.6`
- No Top-K sort: `True`
- Top-K selector: `histogram`
- Sparse scope: `down`

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate | Notes |
|---|---:|---:|---:|---|
| Dense `--top-k 0` | 193.92 ms | 88.30 tok/s | 600/600 (100.0%) | Dense baseline |
| Top-K `--top-k 0.10` histogram down-only | 175.33 ms | 97.54 tok/s | 600/600 (100.0%) | Confirmed v09a 200M experimental local CPU speed candidate |
| Top-K `--top-k 0.12` histogram down-only | 177.04 ms | 96.57 tok/s | 600/600 (100.0%) | Preserved strict QA, slightly slower than 0.10 |

Top-K `0.08` reached 173.07 ms / 98.80 tok/s, but strict QA dropped to 570/600 (95.0%), so it is not the recommended mode.

Interpretation:

Top-K `0.10` histogram down-only is the confirmed v09a 200M experimental local CPU speed candidate. It improved tokens/sec by +10.46% and reduced latency by -9.58% versus dense in this repeat-30 interleaved local CPU run while preserving strict QA at 600/600.

This is not a general sparse speedup claim. Top-K is not always faster. These results are local CPU-specific, and larger-model or other-hardware scaling is not automatically proven.

## Local Usage

Run from the Leviathan repository root after placing the model package folder beside `engine.py`.

Dense QA check:

```cmd
echo What is Leviathan? | python engine.py --bin .\leviathan_mlgru_200m_instruct_v09a\leviathan_mlgru_200m_instruct_v09a.bin --meta .\leviathan_mlgru_200m_instruct_v09a\leviathan_mlgru_200m_instruct_v09a_meta.json --architecture mlgru --prompt-template qa --max-new 80 --top-k 0 --profile
```

Recommended sparse candidate check:

```cmd
echo What is Leviathan? | python engine.py --bin .\leviathan_mlgru_200m_instruct_v09a\leviathan_mlgru_200m_instruct_v09a.bin --meta .\leviathan_mlgru_200m_instruct_v09a\leviathan_mlgru_200m_instruct_v09a_meta.json --architecture mlgru --prompt-template qa --max-new 80 --top-k 0.10 --sparse-scope down --sparse-min-density 0.6 --no-top-k-sort --top-k-select histogram --profile
```

## Package Contents

Expected files:

```text
README.md
leviathan_mlgru_200m_instruct_v09a.bin
leviathan_mlgru_200m_instruct_v09a_meta.json
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
