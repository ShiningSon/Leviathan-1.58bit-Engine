# v0.9.1 Quality Protocol

The v0.9.1 quality work broadens project validation while keeping the original v0.9.0 evidence intact. The new holdout is useful for regression discovery, but it is not a comprehensive language-model evaluation.

## Project QA and holdout QA

[`benchmarks/prompts_v02b_qa.json`](../benchmarks/prompts_v02b_qa.json) is the 20-prompt strict project regression set used by the published scaling records. It checks known Leviathan concepts and known failure patterns with required terms, forbidden terms, and answer-length limits.

[`benchmarks/prompts_v091_holdout.json`](../benchmarks/prompts_v091_holdout.json) contains 105 different prompts:

- 70 `exact_match_regression` prompts, ten in each category;
- 35 `qualitative_review` prompts, five in each category.

The categories cover Leviathan concepts, paraphrase robustness, negation and limitation awareness, package/runtime usage, sparse-mode caveats, simple TinyStories comprehension, and malformed or ambiguous requests.

The simple-comprehension items use newly written short story contexts. They are not copied TinyStories dataset records.

The benchmark runner accepts the structured holdout file and automatically runs only `exact_match_regression`. Qualitative prompts are intentionally excluded from automatic scoring.

## Contamination risk

The 20 project prompts are public and related supervised QA data is committed, so a high score on that set is a regression result rather than clean held-out evidence. The v0.9.1 prompts do not copy those project questions verbatim, but they still discuss a narrow public project and cannot eliminate conceptual contamination. Results must be labeled accordingly.

## Exact-match limits

Each exact item defines:

- `required_terms`: all terms must appear case-insensitively;
- `forbidden_terms`: none may appear;
- `max_words`: the response must not exceed the limit.

These rules catch omissions, contradictions, and some story-tail contamination. They can still reject a correct synonym or accept an answer that contains the right words with weak reasoning. Automatic pass rates must be accompanied by sampled-output review.

## Qualitative review

Two reviewers are preferred for a report intended for canonical documentation. Review each answer against its item-level `review_criteria`, record disagreements, and preserve the raw output checksum. Reviewers should check factuality, instruction fit, concision, unsupported claims, entity consistency, and whether malformed requests trigger clarification rather than invention.

Do not convert a qualitative judgment into an exact percentage without publishing the rubric, reviewer count, disagreement handling, model revision, engine commit, and prompt-set hash.

Generate one paired row per qualitative prompt with dense and Top-K `0.10` outputs:

```cmd
python scripts\run_qualitative_holdout.py --model-dir .\leviathan_mlgru_200m_instruct_v09a --engine .\engine.py --prompts .\benchmarks\prompts_v091_holdout.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.10 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --model-repo ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a --model-revision 116a857bdaf2a1118d479d52aedba7e65cbff960 --out-dir .\benchmark_runs\v091_qualitative
```

The generated `qualitative_review.md` and `qualitative_review.csv` contain exactly 35 paired review rows. For each row, a human reviewer records:

- a dense verdict: `pass`, `concern`, or `fail`;
- a sparse verdict: `pass`, `concern`, or `fail`;
- semantic equivalence: `yes`, `partial`, or `no`;
- concise notes tied to the item criteria;
- a reviewer identifier or pseudonym;
- disagreement resolution when more than one reviewer participated.

Two reviewers are preferred. If only one reviewer is available, state that explicitly, do not report inter-rater agreement, and do not describe the result as independently adjudicated. AI-generated preliminary comments may assist a reviewer but do not count as the required human review and must not populate the reviewer field.

Validate a newly generated blank worksheet before review:

```cmd
python scripts\validate_qualitative_review.py .\benchmark_runs\v091_qualitative\qualitative_outputs.json --worksheet .\benchmark_runs\v091_qualitative\qualitative_review.csv --expect-unreviewed
```

After a person fills every required verdict, equivalence, and reviewer field, validate with `--expect-completed`. Keep generated worksheets under ignored `benchmark_runs/` until the human review and disclosure are complete.

## Running the exact holdout

The following command runs the 70 automatically scored items. It does not run the 35 qualitative prompts:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_200m_instruct_v09a --engine .\engine.py --prompts .\benchmarks\prompts_v091_holdout.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.10 --repeat 10 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --model-repo ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a --model-revision 116a857bdaf2a1118d479d52aedba7e65cbff960 --out-dir .\benchmark_runs\v091_holdout
```

Treat this as new v0.9.1 evidence. Do not replace or retroactively reinterpret the v0.9.0 20-prompt benchmark rows.

The compact repeat-10 record is published in [`reports/v091_exact_holdout_summary.md`](../reports/v091_exact_holdout_summary.md), with a machine-readable companion JSON. Raw `benchmark_runs/` outputs remain local artifacts.

## Why v0.9.0 remains unchanged

The v0.9.0 tag, release notes, manifest, and benchmark summary identify the exact artifact and method published at that point. New prompts, hardware, or tooling create new evidence under v0.9.1; changing historical values would make the release harder to reproduce and audit.
