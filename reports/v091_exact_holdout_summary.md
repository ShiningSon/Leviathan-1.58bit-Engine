# v0.9.1 Exact Holdout Evidence

Status: automated exact-match regression evidence. This is not a comprehensive language-quality evaluation and does not modify the v0.9.0 benchmark record.

## Reproducibility metadata

- Model: `ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a`
- Model revision: `116a857bdaf2a1118d479d52aedba7e65cbff960`
- Engine commit: `0f31a7eb0dea630f106a1ec5f46f23c08c6682a1`
- Prompt file: `benchmarks/prompts_v091_holdout.json`
- Prompt SHA-256: `b2e7b77a3fd73c92ff057f48f07cc2fa8af6fb50488322ebad5f34fa016af3ea`
- Automatically scored prompts: 70
- Modes: dense `--top-k 0` and histogram down-only Top-K `--top-k 0.10`
- Schedule: interleaved, one warmup per mode

## Probe

The repeat-1 probe completed with no process or parse failures.

| Mode | Avg latency | Avg tokens/sec | Exact QA pass rate | Process failures | Parse failures |
|---|---:|---:|---:|---:|---:|
| Dense `--top-k 0` | 276.08 ms | 63.10 tok/s | 16/70 (22.9%) | 0 | 0 |
| Top-K `--top-k 0.10` | 251.06 ms | 70.07 tok/s | 17/70 (24.3%) | 0 | 0 |

## Repeat-10 result

The measured run covered 700 responses per mode and completed with no process or parse failures.

| Mode | Avg latency | Avg tokens/sec | Exact QA pass rate | Process failures | Parse failures |
|---|---:|---:|---:|---:|---:|
| Dense `--top-k 0` | 278.03 ms | 62.75 tok/s | 160/700 (22.9%) | 0 | 0 |
| Top-K `--top-k 0.10` | 248.70 ms | 70.57 tok/s | 170/700 (24.3%) | 0 | 0 |

Top-K measured higher throughput than dense under these exact local settings, but the automatically scored holdout pass rate was low for both modes. This result is not evidence of broad language-quality improvement or universal sparse acceleration.

## Category results

Each category contains ten prompts measured ten times per mode.

| Category | Dense | Top-K 0.10 |
|---|---:|---:|
| Leviathan concepts | 40/100 | 40/100 |
| Paraphrase robustness | 40/100 | 40/100 |
| Negation and limitation awareness | 20/100 | 20/100 |
| Package/runtime usage | 10/100 | 10/100 |
| Sparse-mode caveats | 0/100 | 10/100 |
| Simple TinyStories comprehension | 0/100 | 0/100 |
| Malformed or ambiguous requests | 50/100 | 50/100 |

## Failed prompt IDs

The same failures repeated in all ten measured runs.

| Category | Dense failed IDs | Top-K 0.10 failed IDs |
|---|---|---|
| Leviathan concepts | `lc-e04`, `lc-e05`, `lc-e06`, `lc-e07`, `lc-e08`, `lc-e10` | `lc-e04`, `lc-e05`, `lc-e06`, `lc-e07`, `lc-e08`, `lc-e10` |
| Paraphrase robustness | `pr-e01`, `pr-e02`, `pr-e06`, `pr-e07`, `pr-e09`, `pr-e10` | `pr-e01`, `pr-e02`, `pr-e06`, `pr-e07`, `pr-e09`, `pr-e10` |
| Negation and limitation awareness | `nl-e02`, `nl-e03`, `nl-e04`, `nl-e06`, `nl-e07`, `nl-e08`, `nl-e09`, `nl-e10` | `nl-e02`, `nl-e03`, `nl-e04`, `nl-e06`, `nl-e07`, `nl-e08`, `nl-e09`, `nl-e10` |
| Package/runtime usage | `pu-e01`, `pu-e02`, `pu-e03`, `pu-e04`, `pu-e05`, `pu-e07`, `pu-e08`, `pu-e09`, `pu-e10` | `pu-e01`, `pu-e02`, `pu-e03`, `pu-e04`, `pu-e05`, `pu-e07`, `pu-e08`, `pu-e09`, `pu-e10` |
| Sparse-mode caveats | `sc-e01`, `sc-e02`, `sc-e03`, `sc-e04`, `sc-e05`, `sc-e06`, `sc-e07`, `sc-e08`, `sc-e09`, `sc-e10` | `sc-e01`, `sc-e02`, `sc-e04`, `sc-e05`, `sc-e06`, `sc-e07`, `sc-e08`, `sc-e09`, `sc-e10` |
| Simple TinyStories comprehension | `ts-e01`, `ts-e02`, `ts-e03`, `ts-e04`, `ts-e05`, `ts-e06`, `ts-e07`, `ts-e08`, `ts-e09`, `ts-e10` | `ts-e01`, `ts-e02`, `ts-e03`, `ts-e04`, `ts-e05`, `ts-e06`, `ts-e07`, `ts-e08`, `ts-e09`, `ts-e10` |
| Malformed or ambiguous requests | `ma-e01`, `ma-e06`, `ma-e07`, `ma-e09`, `ma-e10` | `ma-e01`, `ma-e06`, `ma-e07`, `ma-e09`, `ma-e10` |

All recorded failures were missing-required-term failures. The run recorded zero forbidden-term hits and zero `max_words` violations. These guards do not detect every semantic error or story-style contamination pattern: sampled outputs showed unrelated memorized project answers on some story-comprehension and package prompts.

## Interpretation limits

Exact keyword rules can reject a correct synonym and can accept a weak answer that happens to contain required terms. The holdout is conceptually related to a public project, so it cannot eliminate contamination risk. Review sampled outputs and the separate 35-row qualitative worksheet before drawing any quality conclusion.
