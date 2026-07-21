# Leviathan v0.9.0 Research Artifact Release Notes

Version: `0.9.0`

Release date: 2026-07-21

This is the first tagged Leviathan v0.9 research artifact. Canonical release: [Leviathan v0.9.0](https://github.com/ShiningSon/Leviathan-1.58bit-Engine/releases/tag/v0.9.0).

## Highlights

- Completed the v09a 200M-class MLGRU training and Leviathan v2 export route.
- Preserved dense strict QA at 600/600 in the confirmed repeat-30 local benchmark.
- Confirmed histogram Top-K `0.10` on the down projection as the current experimental 200M-class local CPU candidate.
- Added a canonical compact scaling summary, validation scripts, release-readiness checks, lightweight tests, and Windows/Linux CI.
- Published the reviewed v09a Leviathan runtime package on Hugging Face.

## Training route

The initial v09a L40S attempt failed with CUDA out-of-memory pressure. The repository therefore added single-GPU H100 and H200 training paths plus H100 fast/safe and H200 fast configs. This was a training-capacity change, not a change to the inference benchmark method.

The exported architecture uses an 8192-token vocabulary, hidden size 1280, 10 MLGRU layers, and intermediate size 4096. It has approximately 233.4M estimated trainable parameters (233,388,800 under the existing tied-embedding estimate).

## Quality and sparse sweep

The repeat-10 quality-recovery sweep showed that Top-K `0.08` had the highest measured throughput but reduced strict QA to 570/600. It was therefore rejected as the recommendation. Top-K `0.10` and `0.12` both restored 600/600 strict QA and advanced to the confirmed repeat-30 comparison.

Confirmed repeat-30 interleaved result:

| Mode | Avg latency | Avg tokens/sec | Strict QA |
|---|---:|---:|---:|
| Dense `--top-k 0` | 193.92 ms | 88.30 tok/s | 600/600 (100.0%) |
| Top-K `--top-k 0.10` histogram down-only | 175.33 ms | 97.54 tok/s | 600/600 (100.0%) |

Measured against dense, Top-K `0.10` recorded +10.46% tokens/sec and -9.58% latency while preserving strict QA. This is the recommended v09a experimental mode.

## Package

Published Hugging Face repository:

[ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a](https://huggingface.co/ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a)

Verified public revision: `116a857bdaf2a1118d479d52aedba7e65cbff960`.

The package contains a Leviathan runtime binary, metadata, tokenizer, report, sample outputs, and model card. It is not a standard Transformers checkpoint and hosted inference is disabled. The verified public revision, byte sizes, and SHA-256 hashes are frozen in [`releases/v0.9.0_hf_manifest.json`](../releases/v0.9.0_hf_manifest.json).

## Known limitations

- v09a is a TinyStories plus project-QA proof model, not a general assistant.
- The 20-prompt strict QA set is a project regression guard, not a comprehensive evaluation.
- The benchmark is a local CPU result on one documented system.
- This is not a general sparse speedup claim, Top-K is not always faster, and scaling to other hardware or larger models is not automatically proven.
- Threshold sparsity and naive block Top-K remain negative experiments and are not active runtime paths.

All pre-release gates passed and are recorded in [`docs/V09_RELEASE_CHECKLIST.md`](V09_RELEASE_CHECKLIST.md). The authorized `v0.9.0` publication is completed by the release task after CI passes on the exact release commit.
