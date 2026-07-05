# v0.7b 70M QA repair fine-tune

This is a 70M QA repair fine-tune path for `leviathan_mlgru_70m_instruct_v07b`. It starts from the v07a 70M checkpoint and increases supervised QA emphasis before another sparse validation pass.

This remains a Leviathan proof-model path, not a general assistant. Do not upload v07b to Hugging Face, do not modify the v07a package, and do not claim sparse speedup from v07a because its dense QA benchmark is only 85.0%.

## v07a summary

The v07a package successfully loads in `engine.py` as an MLGRU model:

```text
layers: 8
hidden_size: 768
intermediate_size: 2304
architecture: mlgru
```

Dense QA benchmark:

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate |
|---|---:|---:|---:|
| Dense `--top-k 0` | 82.56 ms | 224.38 tok/s | 170/200 (85.0%) |

Histogram down-only sweep, repeat 10:

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate |
|---|---:|---:|---:|
| Dense `--top-k 0` | 79.94 ms | 230.85 tok/s | 170/200 (85.0%) |
| Top-K `--top-k 0.08` | 78.96 ms | 245.23 tok/s | 160/200 (80.0%) |
| Top-K `--top-k 0.10` | 74.61 ms | 245.38 tok/s | 160/200 (80.0%) |
| Top-K `--top-k 0.12` | 79.07 ms | 242.49 tok/s | 170/200 (85.0%) |
| Top-K `--top-k 0.15` | 78.69 ms | 239.33 tok/s | 170/200 (85.0%) |
| Top-K `--top-k 0.18` | 78.38 ms | 235.47 tok/s | 170/200 (85.0%) |

The v07a sweep is a promising 70M sparse scaling signal because Top-K `0.12` preserved the v07a dense QA rate and measured higher tokens/sec in the same repeat-10 sweep. It is not a valid speedup claim model because the underlying QA pass rate is only 85.0%.

## Why v07b

The v07a model needs QA repair before another sparse validation pass. Observed failure patterns:

- `Define Leviathan briefly.` often answers with the Top-K activation sparsity answer.
- `What files are inside a Leviathan MLGRU model package?` partially answers `.bin` and metadata JSON but misses tokenizer and falls into TinyStories-style continuation.
- `What files do I need to run the model?` sometimes answers TinyStories instead of package files.

Success criteria:

- Dense QA should improve from 85.0% toward 95.0%.
- Sparse sweep results are only meaningful if dense QA is at least near 90%, ideally 95%.
- If Top-K `0.12` or `0.15` preserves dense QA and improves tokens/sec, v07b becomes the real 70M speed candidate.

## Config

The v07b run is configured in:

```text
training\configs\v07b_70m_qa_repair.json
```

Key values:

```text
run_name: leviathan_mlgru_70m_instruct_v07b
architecture: mlgru
layers: 8
hidden_size: 768
intermediate_size: 2304
seq_len: 96
qa_ratio: 0.95
steps: 800
lr: 0.000075
base_ckpt_path: /data/runs/leviathan_mlgru_70m_instruct_v07a/checkpoints/final.pt
base_tokenizer_dir: /data/exports/leviathan_mlgru_70m_instruct_v07a/leviathan_mlgru_tokenizer
training_stage: v0.7b-70m-qa-repair
```

## Dry-run parameter check

Run from the repository root:

```cmd
python -m modal run training\07_finetune_supervised_qa_mlgru_modal_L40S_v02f.py --config-json training\configs\v07b_70m_qa_repair.json --dry-run
```

## Training and export command

```cmd
python -m modal run training\07_finetune_supervised_qa_mlgru_modal_L40S_v02f.py --config-json training\configs\v07b_70m_qa_repair.json
```

## Download exported package

```cmd
python -m modal volume get leviathan-mlgru /exports/leviathan_mlgru_70m_instruct_v07b.zip leviathan_mlgru_70m_instruct_v07b.zip
```

Unpack locally into a package folder:

```cmd
rmdir /S /Q leviathan_mlgru_70m_instruct_v07b 2>nul
mkdir leviathan_mlgru_70m_instruct_v07b
tar -xf leviathan_mlgru_70m_instruct_v07b.zip -C leviathan_mlgru_70m_instruct_v07b
```

## Smoke load command

```cmd
echo What is Leviathan? | python engine.py --bin .\leviathan_mlgru_70m_instruct_v07b\leviathan_mlgru_70m_instruct_v07b.bin --meta .\leviathan_mlgru_70m_instruct_v07b\leviathan_mlgru_70m_instruct_v07b_meta.json --architecture mlgru --prompt-template qa --max-new 80 --top-k 0 --profile
```

## Benchmark commands

Dense QA probe:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_70m_instruct_v07b --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 --repeat 10 --mode-schedule interleave --out-dir .\benchmark_runs\v07b_70m_dense_probe
```

Histogram down-only sweep:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_70m_instruct_v07b --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.08 0.10 0.12 0.15 0.18 --repeat 10 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir .\benchmark_runs\v07b_70m_histogram_sweep_repeat10
```

## Interpretation rules

- Treat v07b as a QA repair proof model, not a general assistant.
- Use `--prompt-template qa` for project-specific QA checks.
- Do not claim sparse speedup unless Top-K preserves dense QA and measured tokens/sec exceeds dense under the same benchmark settings.
- Do not upload v07b to Hugging Face until the package and benchmark are reviewed.
