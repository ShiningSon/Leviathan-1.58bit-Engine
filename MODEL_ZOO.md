# Leviathan Model Zoo

Leviathan model repositories are runtime packages, not standard Transformers checkpoints. Run them with this repository's `engine.py` and `--architecture mlgru`. Instruct proof models use `--prompt-template qa` for the project-specific QA benchmark.

Recommended Top-K values below are model-specific experimental local CPU candidates. They are not universal defaults and do not establish a general sparse speedup.

## Published Hugging Face packages

| Repository | Package | Class | Architecture | Recommended mode | Strict QA | Runtime compatibility | Status |
|---|---|---:|---|---|---:|---|---|
| [30M v02g](https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02g) | `leviathan_mlgru_30m_instruct_v02g` | 30M | MLGRU | Dense default; Top-K `0.12` is experimental | 950/1000 (95.0%) in the v0.6 route | Leviathan v2 package, `engine.py --architecture mlgru` | Published proof model |
| [70M v07b](https://huggingface.co/ShiningSon/Leviathan-MLGRU-70M-TinyStories-Instruct-v07b) | `leviathan_mlgru_70m_instruct_v07b` | 70M | MLGRU | Histogram Top-K `0.08`, down-only | 600/600 (100.0%) | Leviathan v2 package, `engine.py --architecture mlgru` | Published experimental candidate |
| [100M v08a](https://huggingface.co/ShiningSon/Leviathan-MLGRU-100M-TinyStories-Instruct-v08a) | `leviathan_mlgru_100m_instruct_v08a` | 100M | MLGRU | Histogram Top-K `0.06`, down-only | Dense 570/600; sparse 600/600 | Leviathan v2 package, `engine.py --architecture mlgru` | Published experimental candidate |
| [200M v09a](https://huggingface.co/ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a) | `leviathan_mlgru_200m_instruct_v09a` | 200M | MLGRU | Histogram Top-K `0.10`, down-only: 175.33 ms / 97.54 tok/s | 600/600 (100.0%) both modes | Leviathan v2 package, `engine.py --architecture mlgru` | Published current experimental candidate |

The v09a sparse command also requires `--sparse-min-density 0.6 --no-top-k-sort --top-k-select histogram --sparse-scope down`. Top-K `0.08` is not recommended for v09a because its strict QA result dropped to 570/600 even though it measured faster.

## Publication-ready packages

No reviewed active package is waiting for initial publication. The versioned cards remain under `hf_cards/` so future package revisions can be reviewed before upload.

## Historical or superseded experiments

| Model | Status | Notes |
|---|---|---|
| [30M v0.1 TinyStories](https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories) | Published historical proof | Validated training, export, and plain continuation through the MLGRU runtime. |
| [30M v0.2b QA proof](https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02b) | Published historical QA proof | Established the supervised project-QA route; dense remains the recommended mode for this small package. |
| v02c / v02d | Superseded negative experiments | Did not become successful release candidates. |
| v02e / v02f | Internal repair steps | Led to the published v02g package; not active distribution targets. |
| v07a 70M | Superseded scaling probe | Loaded successfully and showed a sparse scaling signal, but strict QA was only 85.0%; v07b replaced it. |

See [`benchmarks/results/scaling_summary.json`](benchmarks/results/scaling_summary.json) for the canonical compact measurements and [`BENCHMARK.md`](BENCHMARK.md) for the full history.
