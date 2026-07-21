# v0.9.1 Reproducibility Plan

v0.9.1 is a post-release validation cycle for the immutable v0.9.0 research artifact. It improves how independent measurements and broader quality checks are collected without changing the published v0.9.0 benchmark record.

## Scope

- Validate dense and recommended sparse modes on additional CPU hardware.
- Expand QA regression coverage beyond the 20-prompt project set.
- Make benchmark submissions reproducible, privacy-limited, and machine-checkable.
- Improve runtime and benchmark usability where reproducibility work identifies friction.

## Explicit exclusions

- No new large-model training is part of this cycle.
- No universal sparse-speedup claim will be made from model-specific or single-machine results.
- No v0.9.0 benchmark value, release tag, or published Hugging Face package will be changed.

## Workstreams

### Cross-hardware validation

External runs pin the engine commit, Hugging Face revision, prompt-set SHA-256, sparse settings, thread configuration, warmups, repeats, and mode schedule. A privacy-limited hardware manifest records CPU, OS, memory, compiler, Python, and PyTorch information without collecting usernames, hostnames, IP addresses, or private filesystem paths.

### Broader quality validation

[`benchmarks/prompts_v091_holdout.json`](../benchmarks/prompts_v091_holdout.json) separates automatically scored regression rules from prompts that require human review. The set has 105 prompts across seven categories. It remains a project-focused holdout, not a comprehensive language-model evaluation.

### Portable submissions

Submission tooling recalculates latency and throughput aggregates from individual samples, checks strict-QA totals, hashes outputs, and rejects incomplete runs. A recommended sparse mode must preserve or improve the dense QA rate within the submitted comparison.

## Acceptance criteria

- The schemas and committed fixture pass local and CI validation.
- The system collector runs on Windows and Linux without private machine identity fields.
- The existing benchmark CLI remains compatible and records complete reproducibility metadata.
- At least one independent CPU submission is reviewed before cross-hardware conclusions are added.
- Qualitative holdout responses are reviewed by a person before any quality conclusion is published.

## Current boundary

This branch prepares the validation protocol and tooling. It contains no new external hardware result, no new model weights, and no modification to the v0.9.0 release evidence.
