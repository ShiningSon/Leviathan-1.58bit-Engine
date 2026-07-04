# v0.7a 70M MLGRU scaling probe

This is the first 70M-class scaling probe for the Leviathan MLGRU path. It exists to test whether the v0.6 histogram Top-K 30M speed candidate scales beyond `leviathan_mlgru_30m_instruct_v02g`.

This is a proof-model path, not a general assistant. Do not upload this model to Hugging Face and do not claim sparse speedup until the benchmark results support it.

## Target package

```text
leviathan_mlgru_70m_instruct_v07a
```

## Architecture

```text
architecture: mlgru
layers: 8
hidden_size: 768
intermediate_size: 2304
heads: 1
kv_heads: 1
activation: relu2
tokenizer: reuse the v02g tokenizer path when present; otherwise train the same local tokenizer style
```

Estimated trainable parameters:

```text
67,670,784
```

The estimate assumes tied input/output embeddings, seven ternary linear projections per layer, four RMSNorm vectors per layer, and one final RMSNorm vector.

## Config

The v07a run is configured in:

```text
training\configs\v07a_70m_mlgru.json
```

The config sets `base_ckpt_path` to an empty string because the 30M checkpoint shape is incompatible with the 70M architecture. The run starts from scratch while reusing the same QA/instruct data path and tokenizer strategy.

## Dry-run parameter check

Run from the repository root:

```cmd
python -m modal run training\07_finetune_supervised_qa_mlgru_modal_L40S_v02f.py --config-json training\configs\v07a_70m_mlgru.json --dry-run
```

Expected output includes:

```text
[estimate] params=67,670,784
```

## Training and export command

This Modal command trains the model and exports the engine-compatible package into the `leviathan-mlgru` Modal volume.

```cmd
python -m modal run training\07_finetune_supervised_qa_mlgru_modal_L40S_v02f.py --config-json training\configs\v07a_70m_mlgru.json
```

## Download exported package

```cmd
python -m modal volume get leviathan-mlgru /exports/leviathan_mlgru_70m_instruct_v07a.zip leviathan_mlgru_70m_instruct_v07a.zip
```

Unpack locally into a package folder:

```cmd
rmdir /S /Q leviathan_mlgru_70m_instruct_v07a 2>nul
mkdir leviathan_mlgru_70m_instruct_v07a
tar -xf leviathan_mlgru_70m_instruct_v07a.zip -C leviathan_mlgru_70m_instruct_v07a
```

Expected local package contents:

```text
leviathan_mlgru_70m_instruct_v07a.bin
leviathan_mlgru_70m_instruct_v07a_meta.json
leviathan_mlgru_tokenizer\
report.json
sample_outputs.txt
```

## Smoke load command

After export and unpacking, confirm that `engine.py` can load the package:

```cmd
echo What is Leviathan? | python engine.py --bin .\leviathan_mlgru_70m_instruct_v07a\leviathan_mlgru_70m_instruct_v07a.bin --meta .\leviathan_mlgru_70m_instruct_v07a\leviathan_mlgru_70m_instruct_v07a_meta.json --architecture mlgru --prompt-template qa --max-new 80 --top-k 0 --profile
```

## Benchmark commands

Dense probe:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_70m_instruct_v07a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 --repeat 10 --mode-schedule interleave --out-dir .\benchmark_runs\v07a_70m_dense_probe
```

Histogram down-only sweep:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_70m_instruct_v07a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.08 0.10 0.12 0.15 0.18 --repeat 10 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir .\benchmark_runs\v07a_70m_histogram_sweep_repeat10
```

If repeat-10 is promising, repeat with 30 measured runs:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_70m_instruct_v07a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.08 0.10 0.12 0.15 0.18 --repeat 30 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --out-dir .\benchmark_runs\v07a_70m_histogram_sweep_repeat30
```

## Interpretation rules

- Treat v07a as a scaling probe, not a final model.
- Use `--prompt-template qa` for project-specific QA checks.
- Do not claim sparse speedup unless measured tokens/sec exceeds dense under the same benchmark settings.
- Do not claim that the v0.6 30M result automatically scales to 70M.
- Do not upload v07a to Hugging Face until the package and benchmark are reviewed.
