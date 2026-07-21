# Contributing

Leviathan is a research prototype. Contributions should preserve reproducibility, keep model artifacts out of Git history, and distinguish measured observations from broader claims.

## Development setup

Use Python 3.11 for the repository-only validation path:

```bash
python -m unittest discover -s tests -v
python scripts/validate_benchmark_summary.py
python scripts/check_release_readiness.py
```

Running `engine.py` additionally requires PyTorch, a compatible C++ compiler, and OpenMP support. Install runtime and training dependencies with `python -m pip install -r requirements.txt`.

## Change scope

- Keep runtime changes, benchmark records, and release documentation in separate logical commits when practical.
- Do not change generation, scoring, or selector behavior while claiming a documentation-only update.
- Preserve dense behavior and existing CLI compatibility unless the change explicitly documents a migration.
- Add focused standard-library tests for repository logic that does not need weights, CUDA, Modal, or network access.
- Treat threshold sparsity and naive block Top-K as recorded negative experiments unless new measured evidence justifies a separate proposal.

## Benchmarks

Use `scripts/benchmark_engine.py` with the committed prompt set. Record the model, prompt template, max-new value, warmup count, repeat count, schedule, selector, scope, and CPU environment.

Only add confirmed compact records to `benchmarks/results/scaling_summary.json`. Run its validator before committing. Generated `benchmark_runs/` files remain local review artifacts.

Do not describe a Top-K mode as a speed candidate unless it preserves or improves the dense strict-QA rate and improves measured tokens/sec under the same settings. Never generalize one CPU result to all hardware or model sizes.

## Repository hygiene

Do not commit model packages, `.bin`, `.zip`, checkpoints, training runs, environment files, local caches, or benchmark dumps. Hugging Face model repositories receive reviewed package files; GitHub receives source, configs, cards, benchmark definitions, and documentation.

Before committing, run:

```bash
git diff --check
git diff --cached --name-only
```

## Documentation

Use repository-relative links for project files. Commands should state the shell when they are platform-specific; otherwise prefer paths and one-line commands that work from the repository root on Windows and Linux.

The project is licensed under MIT. By contributing, you agree that your contribution may be distributed under the repository license.
