# Leviathan v0.9.0 Release Candidate Checklist

This checklist records the final release-gate review before creating the deferred `v0.9.0` tag and GitHub Release. Verification used public Hugging Face revision `116a857bdaf2a1118d479d52aedba7e65cbff960` on 2026-07-21 UTC.

- [x] Local Python compile passed for `engine.py`, `quantizer.py`, `scripts/`, `training/`, and `tests/`.
- [x] Lightweight unit tests passed.
- [x] Canonical benchmark summary validation passed.
- [x] Repository release-readiness validation passed.
- [x] Ubuntu CI passed on the starting release-documentation commit (`ec83a86`, Actions run `29815447615`).
- [x] Windows CI passed on the starting release-documentation commit (`ec83a86`, Actions run `29815447615`).
- [x] The public v09a Hugging Face repository is visible without authentication.
- [x] The HF dry run listed all eight public files (77,442,152 bytes total).
- [x] A clean public HF snapshot download completed.
- [x] Dense mode loaded the clean package, resolved its tokenizer, and produced the expected coherent Leviathan answer.
- [x] Top-K `0.10` histogram down-only mode loaded the clean package and produced the same coherent answer.
- [x] The package-only repeat-3 strict QA regression passed at 60/60 for dense and 60/60 for Top-K `0.10`.
- [x] The public model card matches the reviewed repository card after normalizing only line endings and trailing whitespace.
- [x] SHA-256 and byte size are recorded for every public package file in [`releases/v0.9.0_hf_manifest.json`](../releases/v0.9.0_hf_manifest.json).
- [x] The confirmed benchmark is explicitly limited to one local CPU environment.
- [x] Additional CPU validation remains deferred and is documented as an open roadmap item.
- [ ] Ready for the `v0.9.0` tag after the final `main` push passes both CI jobs.

The tag and GitHub Release are not created by this verification task.
