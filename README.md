# Leviathan-1.58bit-Engine

Leviathan is a local CPU inference engine for native 1.58-bit / ternary language models.
Its target is not generic post-training quantization of ordinary FP16 LLMs; the runtime is built around QAT/native BitNet-style or Leviathan-trained checkpoints whose linear weights can live as ternary values `{-1, 0, 1}`.

The current direction is:

1. keep model packages compact with row-major 2-bit ternary weights;
2. execute bitlinear layers through a C++/OpenMP CPU runtime instead of PyTorch kernels;
3. experiment with ratio-based Top-K activation sparsity;
4. support an experimental attention-free MLGRU runtime path for recurrent checkpoints trained for that path.

> Status: research prototype. The dense ternary CPU path and MLGRU export path are working. Ratio Top-K is implemented, with dense still the default recommendation for the 30M proof model. `leviathan_mlgru_100m_instruct_v08a` is the current 100M experimental local CPU speed candidate: histogram down-projection-only Top-K `0.06` reached 90.39 ms / 191.14 tok/s versus dense 101.15 ms / 176.73 tok/s in repeat-30 interleaved testing, with QA improving from 570/600 to 600/600 in the measured run. This is not a general sparse speedup claim, not a claim that Top-K is always faster, and not proof that the result automatically scales to larger models or other hardware.

Published 30M Hugging Face proof package: https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02g

`leviathan_mlgru_100m_instruct_v08a` is currently a local/exported proof package and should not be described as a published Hugging Face model yet.

---

## What is implemented

### 1. C++ CPU ternary bitlinear runtime

`engine.py` compiles a C++ runtime through `torch.utils.cpp_extension`.
The runtime consumes INT8 activations and row-packed ternary weights, then performs sign/add accumulation on CPU.

Implemented pieces:

- AVX2 x86/x64 path using `_mm256_sign_epi8`-style sign/add logic.
- Portable scalar fallback for non-AVX2 systems.
- Row-major 2-bit ternary packing, four ternary weights per byte.
- Independent row padding so odd hidden sizes do not corrupt row boundaries.
- A shared `bitlinear_dispatch` boundary for dense and sparse paths.

Prepared extension points:

- AVX-512 VNNI-specialized kernels.
- ARM NEON / SDOT kernels.
- More efficient sparse activation scheduling.

### 2. Ratio-based Top-K activation sparsity

Leviathan supports Top-K activation sparsity through `--top-k`.

Current sparse-scope examples:

```bash
--top-k 0.06 --sparse-scope down --top-k-select histogram --sparse-min-density 0.6 --no-top-k-sort
--top-k 0.08 --sparse-scope down --top-k-select histogram --sparse-min-density 0.6 --no-top-k-sort
--top-k 0.12 --sparse-scope down --top-k-select histogram --sparse-min-density 0.6 --no-top-k-sort
```

Density guide:

- `0.06` is the current v08a 100M candidate density.
- `0.08` is the v07b 70M candidate density.
- `0.12` is the v0.6 30M candidate density.

The important change is that fractional values are treated as **per-layer density ratios** rather than fixed absolute K values.
This avoids the failure mode where a fixed K is acceptable for narrow projections but destroys wide FFN projections.

Historical and advanced examples:

```bash
--top-k 0.9    # historical high-density QA/fallback check
--top-k 0.8    # historical high-density QA/fallback check
--top-k 2048   # legacy absolute-K mode
```

Current benchmark result:

- Ratio Top-K preserves readable output in the 30M MLGRU proof model.
- Dense mode remains the default recommendation at 30M scale.
- v0.6 histogram Top-K `0.12` with `--sparse-scope down` produced a small local CPU speed candidate in repeat-50 interleaved testing while preserving strict QA at 95.0%.
- v07b 70M histogram Top-K `0.08` with `--sparse-scope down` is the confirmed 70M experimental local CPU speed candidate: 68.54 ms / 254.12 tok/s versus dense 72.85 ms / 238.50 tok/s, with both modes at 600/600 strict QA in repeat-30 interleaved testing.
- v08a 100M histogram Top-K `0.06` with `--sparse-scope down` is the current experimental local CPU speed candidate: 90.39 ms / 191.14 tok/s versus dense 101.15 ms / 176.73 tok/s, with QA improving from 570/600 to 600/600 in repeat-30 interleaved testing.
- Top-K is currently a scaling and kernel-optimization path, not a guaranteed speedup for every model size.

v0.4 guardrail:

Use `--sparse-min-density 0.6` when you want high-density Top-K settings such as `0.9` or `0.8` to fall back to dense kernels instead of forcing the sparse path. This avoids known sparse overhead at 30M scale. It is a guardrail, not a Top-K speedup claim.

```bash
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json --top-k 0.9 --sparse-min-density 0.6
```

`--no-top-k-sort` is experimental. It skips the final index sort after Top-K selection. It can reduce selection overhead, but low-density settings may degrade output quality. It is disabled by default.

`--sparse-scope down` is experimental. It restricts Top-K sparsity to the down projection while keeping recurrent/state projections dense. With the histogram selector, v08a down-only Top-K `0.06` is the current 100M experimental local CPU speed candidate, but it is not a general sparse speedup claim and does not automatically scale to larger models or other hardware.

See [`BENCHMARK.md`](BENCHMARK.md) for the current numbers.

### 3. MatMul-Free MLGRU runtime mode

Passing `--architecture mlgru` switches the runtime into an attention-free recurrent path.
This path replaces transformer attention maps with a GRU-like recurrent state update while still using Leviathan's ternary projection kernels.

Important limitation:

- Existing transformer checkpoints are not expected to work correctly in MLGRU mode.
- MLGRU mode needs a checkpoint trained for the recurrent path.

Historical 30M proof route:

- `Leviathan-MLGRU-30M-TinyStories-v0.1`
- trained from scratch on Modal T4
- dataset: TinyStories
- steps: 2000
- exported to Leviathan v2 format
- runs locally with `engine.py --architecture mlgru`
- model package: [ShiningSon/Leviathan-MLGRU-30M-TinyStories](https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories)

Published 30M instruct proof package:

- `Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2g`
- local package folder: `leviathan_mlgru_30m_instruct_v02g/`
- uploaded Hugging Face package: [ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02g](https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02g)
- dense strict QA: 190/200 keyword-guarded passes, or 95.0%
- known limitation: one package-description prompt still shows residual TinyStories-style continuation

Current active local speed candidate:

- `leviathan_mlgru_100m_instruct_v08a`
- architecture: `mlgru`
- hidden size: 896
- layers: 8
- intermediate size: 3072
- estimated trainable parameters: about 99.14M
- status: local/exported 100M speed candidate, not Hugging Face published, not a general assistant
- current candidate setting: `--top-k 0.06 --sparse-scope down --top-k-select histogram --sparse-min-density 0.6 --no-top-k-sort`

The early proof models are intentionally small. They validate the full route:

```text
Modal training -> fake ternary/QAT-style model -> Leviathan v2 export -> local CPU runtime
```

---

## Repository layout

```text
.
|-- engine.py                         # C++/OpenMP CPU runtime with dense, Top-K, and MLGRU paths
|-- quantizer.py                      # Streams compatible safetensors checkpoints into Leviathan v2 packages
|-- training/
|   |-- 01_train_export_mlgru_modal_T4_clean.py
|   |-- 03_train_supervised_qa_mlgru_modal_T4.py
|   |-- 07_finetune_supervised_qa_mlgru_modal_L40S_v02f.py
|   |-- configs/
|   |   |-- v07a_70m_mlgru.json
|   |   |-- v07b_70m_qa_repair.json
|   |   `-- v08a_100m_mlgru.json
|   `-- instruction_qa_supervised_v02b.jsonl
|-- scripts/
|   `-- benchmark_engine.py
|-- benchmarks/
|   `-- prompts_v02b_qa.json
|-- docs/
|   |-- BENCHMARK_AUTOMATION.md
|   |-- V07A_70M_SCALING_PROBE.md
|   |-- V07B_70M_QA_REPAIR.md
|   |-- V08A_100M_SCALING_PROBE.md
|   `-- V02B_SUPERVISED_QA_GUIDE.md
|-- BENCHMARK.md                      # Current runtime benchmark notes
|-- LICENSE                           # MIT License
|-- requirements.txt                  # Python dependencies
|-- .gitignore                        # Excludes model binaries, caches, venvs, and build outputs
`-- README.md                         # Project overview
```

The training/export proof route lives under `training/`.

---

## Quick start

### Install dependencies

You need Python, PyTorch, Hugging Face tools, and a working C++ compiler.
On Windows, PyTorch C++ extensions expect the MSVC toolchain, usually from Visual Studio Build Tools or a Developer PowerShell.

```bash
pip install -r requirements.txt
```

---

## Running a native BitNet / ternary package

Create a Leviathan package from a native BitNet/QAT checkpoint:

```bash
python quantizer.py --model microsoft/bitnet-b1.58-2B-4T-bf16 --out leviathan_native
```

Run the dense transformer path:

```bash
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json
```

Run with the current sparse-scope Top-K style:

```bash
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json --top-k 0.06 --sparse-scope down --top-k-select histogram --sparse-min-density 0.6 --no-top-k-sort
```

Historical high-density and legacy absolute-K checks:

```bash
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json --top-k 0.9 --sparse-min-density 0.6
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json --top-k 0.8 --sparse-min-density 0.6
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json --top-k 2048
```

---

## Running the MLGRU proof model

The 30M proof model is not a general assistant model. It is a small recurrent language model trained on TinyStories to validate the MLGRU runtime/export path.

Download the model package from Hugging Face:

[ShiningSon/Leviathan-MLGRU-30M-TinyStories](https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories)

Expected local package files:

```text
leviathan_mlgru_30m_t4.bin
leviathan_mlgru_30m_t4_meta.json
leviathan_mlgru_tokenizer/
```

Run dense MLGRU mode:

```bash
python engine.py \
  --bin leviathan_mlgru_30m_t4.bin \
  --meta leviathan_mlgru_30m_t4_meta.json \
  --architecture mlgru \
  --top-k 0 \
  --max-new 120 \
  --prompt-template plain
```

Run ratio Top-K MLGRU mode:

```bash
python engine.py \
  --bin leviathan_mlgru_30m_t4.bin \
  --meta leviathan_mlgru_30m_t4_meta.json \
  --architecture mlgru \
  --top-k 0.9 \
  --max-new 120 \
  --prompt-template plain
```

Recommended first prompt:

```text
Once upon a time
```

TinyStories-trained proof models are expected to produce simple story-like completions, not instruction-following answers.

---

## Running the v0.2b supervised QA proof model

`Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2b` is a supervised QA proof model. It is not a general assistant.

This release validates that a tiny Leviathan-trained MLGRU model can answer project-specific prompts after supervised QA training. For the tested project QA prompts, it should be run with `--prompt-template qa`. Use `--prompt-template plain` only for raw continuation experiments, where QA matching can break.

For the current 30M model, dense mode is the default recommendation. Top-K `0.9` and `0.8` preserved QA matching in local tests, but no Top-K speedup claim is made.

Published Hugging Face repository:

[ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02b](https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02b)

Expected local package files:

```text
leviathan_mlgru_30m_instruct_v02b.bin
leviathan_mlgru_30m_instruct_v02b_meta.json
leviathan_mlgru_tokenizer/
```

Recommended dense QA run:

```bash
python engine.py \
  --bin leviathan_mlgru_30m_instruct_v02b.bin \
  --meta leviathan_mlgru_30m_instruct_v02b_meta.json \
  --architecture mlgru \
  --top-k 0 \
  --max-new 80 \
  --prompt-template qa
```

Top-K QA check:

```bash
python engine.py \
  --bin leviathan_mlgru_30m_instruct_v02b.bin \
  --meta leviathan_mlgru_30m_instruct_v02b_meta.json \
  --architecture mlgru \
  --top-k 0.9 \
  --max-new 80 \
  --prompt-template qa
```

For high-density Top-K QA checks on the 30M model, add `--sparse-min-density 0.6` if you want `0.9` or `0.8` to fall back to dense when sparse overhead dominates.

```bash
python engine.py --bin leviathan_mlgru_30m_instruct_v02b.bin --meta leviathan_mlgru_30m_instruct_v02b_meta.json --architecture mlgru --top-k 0.9 --max-new 80 --prompt-template qa --sparse-min-density 0.6
```

Confirmed project QA examples:

```text
USER> What is Leviathan?
ENGINE> Leviathan is an experimental CPU inference engine for ternary and recurrent language models.

USER> What is MLGRU?
ENGINE> MLGRU is a recurrent runtime path that updates state step by step instead of computing attention maps.

USER> What is Top-K activation sparsity?
ENGINE> Top-K activation sparsity keeps the largest activations and skips smaller activations to reduce computation.

USER> What dataset was this proof model trained on?
ENGINE> The first proof model was trained on TinyStories.
```

---

## Running the published v0.2g 30M proof package

`Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2g` is the current published 30M Hugging Face proof package. It is a small model for validating Leviathan training, export, and local CPU runtime behavior, not a general assistant, and it is no longer the top active local speed candidate.

Uploaded Hugging Face package:

[ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02g](https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02g)

Dense strict QA reached 190/200 passes, or 95.0%, in the local benchmark. Earlier v02e down-only Top-K modes preserved the same strict QA pass rate, but dense remained faster in that comparison. In v0.6, histogram down-only Top-K `0.12` became the current experimental 30M speed candidate in repeat-50 interleaved testing: 50.25 ms / 354.17 tok/s versus dense 50.74 ms / 350.97 tok/s, with strict QA unchanged at 950/1000 (95.0%). This is not a general sparse speedup claim.

For sparse-scope speed-candidate experiments, `--sparse-scope down --top-k 0.12 --sparse-min-density 0.6 --no-top-k-sort --top-k-select histogram` is the 30M experimental setting. The v07b 70M candidate uses histogram down-only Top-K `0.08`; the v08a 100M candidate uses histogram down-only Top-K `0.06`. Broader validation still needs additional hardware and larger-model checks.

Known limitation: the v02g TinyStories proof model retains one package-description failure due to residual story-style continuation.

Recommended dense QA run:

```bash
python engine.py \
  --bin leviathan_mlgru_30m_instruct_v02g.bin \
  --meta leviathan_mlgru_30m_instruct_v02g_meta.json \
  --architecture mlgru \
  --top-k 0 \
  --max-new 80 \
  --prompt-template qa
```

---

## Training the MLGRU proof model on Modal

The training/export route trains a small recurrent model, exports packed ternary linear weights plus FP16 structural tensors, and zips the files for local `engine.py` execution.

Recommended first 30M run:

```bash
modal run training/01_train_export_mlgru_modal_T4_clean.py \
  --run-name leviathan_mlgru_30m_t4 \
  --dataset tinystories \
  --steps 2000 \
  --hidden-size 512 \
  --n-layers 8 \
  --intermediate-size 1536 \
  --batch-size 16 \
  --seq-len 96 \
  --max-train-tokens 3000000 \
  --tokenizer-docs 10000 \
  --lr 3e-4
```

Download the exported model package:

```bash
modal volume get leviathan-mlgru /exports/leviathan_mlgru_30m_t4.zip leviathan_mlgru_30m_t4.zip
```

Then unzip and run the model locally with `engine.py`.

---

## Current benchmark summary

Current winner:

```text
v08a 100M Top-K 0.06 histogram down-only
```

| Model | Dense result | Best Top-K result | QA | Interpretation |
|---|---:|---:|---:|---|
| 30M v0.6 / v02g route | 50.74 ms / 350.97 tok/s | Top-K `0.12` histogram down-only: 50.25 ms / 354.17 tok/s | 950/1000 (95.0%) both modes | Small local CPU speed candidate; v02g remains the published 30M HF proof package |
| 70M v07b | 72.85 ms / 238.50 tok/s | Top-K `0.08` histogram down-only: 68.54 ms / 254.12 tok/s | 600/600 (100.0%) both modes | Stronger 70M local CPU candidate |
| 100M v08a | 101.15 ms / 176.73 tok/s | Top-K `0.06` histogram down-only: 90.39 ms / 191.14 tok/s | Dense 570/600 (95.0%); Top-K 600/600 (100.0%) | Current active local CPU speed candidate |

The scaling pattern is encouraging but still narrow: 30M produced a small candidate, 70M produced a stronger candidate, and 100M currently has the strongest repeat-30 candidate. These are local CPU proof-model results, not a general sparse speedup claim, not a claim that Top-K is always faster, and not evidence that the result automatically scales to larger models or other hardware.

See [`BENCHMARK.md`](BENCHMARK.md) for details.

---

## Model distribution

Do not commit large generated model packages directly into the normal Git history unless the package is very small and intentionally versioned.

Recommended distribution:

1. **GitHub repo**: source code, README, benchmark, training scripts.
2. **GitHub Releases**: small zipped proof packages and screenshots.
3. **Hugging Face model repo**: model package, tokenizer, metadata, model card, sample outputs.

Published Hugging Face packages:

- Published 30M v02g proof package: [ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02g](https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02g)
- Earlier v0.2b QA proof model: [ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02b](https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02b)
- Historical v0.1 TinyStories proof model: [ShiningSon/Leviathan-MLGRU-30M-TinyStories](https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories)

Current local/exported candidates:

- `leviathan_mlgru_70m_instruct_v07b`: local 70M proof package and confirmed local CPU speed candidate.
- `leviathan_mlgru_100m_instruct_v08a`: local 100M proof package and current active local CPU speed candidate.

Do not describe v07b or v08a as Hugging Face-published packages unless they are explicitly uploaded and reviewed.

Suggested files for the published v02g package:

```text
README.md                         # Hugging Face model card
leviathan_mlgru_30m_instruct_v02g.bin
leviathan_mlgru_30m_instruct_v02g_meta.json
leviathan_mlgru_tokenizer/
report.json
sample_outputs.txt
benchmark_v02g_dense_results.md
benchmark_v02e_sparse_down_results.md
```

The published v02g package contains the exported MLGRU binary, metadata, tokenizer, model card, benchmark report, and sample outputs.

---

## Quantizer format

`quantizer.py` writes `leviathan-v2` metadata.

- Linear projection weights are stored as `ternary_2bit_packed`.
- Embeddings, norms, heads, biases, and other structural tensors remain FP16.
- Packed linear tensors include `shape`, `packed_shape`, `gamma`, `offset`, and `size`.

By default, the quantizer should avoid pretending that ordinary FP16 checkpoints are native 1.58-bit models.
PTQ conversion can be useful for runtime stress tests, but coherent generation requires compatible QAT/native ternary weights or a model trained for Leviathan's recurrent runtime.

---

## Limitations

- This is a research prototype, not a production LLM runtime.
- The 30M MLGRU model is a proof model, not a general assistant.
- TinyStories-trained models produce story-like completions, not broad factual answers.
- Top-K speed remains experimental; v08a has a 100M local CPU histogram Top-K candidate, not a general sparse speedup.
- MLGRU mode requires compatible recurrent weights.
- Larger models, better datasets, and kernel-level sparse optimizations are needed before claiming major speedups.

---

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).

---

## Roadmap

### v0.1: MLGRU proof route

- [x] Train 30M MLGRU on Modal T4.
- [x] Export to Leviathan v2 format.
- [x] Run locally through `engine.py --architecture mlgru`.
- [x] Record dense vs ratio Top-K benchmark.

### v0.2b: supervised QA proof model

- [x] Add TinyStories + supervised project-specific QA mixture.
- [x] Improve short project-specific QA behavior with `--prompt-template qa`.
- [x] Publish sample outputs and model card for the v0.2b proof model.
- [ ] Expand QA coverage beyond the small project-specific seed set.

For sparse-scope experiments, the earlier `--sparse-scope down --top-k 0.2 --sparse-min-density 0.6 --no-top-k-sort` setting was the v02e QA-stability candidate. The current v0.6 30M speed candidate uses histogram Top-K at density `0.12`, remains experimental, and is not the default runtime recommendation.

### v0.2g: checkpoint fine-tune strict QA repair

- [x] Fine-tune from the v02b/v02e checkpoint route instead of from-scratch calibration.
- [x] Restore dense strict QA to 190/200 (95.0%).
- [x] Publish the v02g Hugging Face proof-model package.
- [x] Document the remaining package-description known limitation.
- [ ] Expand beyond the current 20-prompt project-specific strict QA set.

### v0.3: benchmark automation

- [x] Add repeatable benchmark script.
- [x] Test multiple prompts and max-new values for v0.2b QA benchmark.
- [x] Record CPU model and compiler details.
- [x] Expand benchmark prompt set.
- [ ] Test on additional CPUs.

### v0.4: sparse kernel work

- [x] Add Top-K profiling mode.
- [x] Add sparse-min-density fallback for high-density Top-K.
- [x] Add experimental no-sort Top-K path.
- [ ] Reduce Top-K selection overhead further.
- [ ] Add quality-stable threshold or block sparsity path.
- [x] Benchmark the 70M proof model.
- [x] Benchmark the 100M proof model.

### v0.5: sparse-scope strict QA

- [x] Add sparse projection scope controls.
- [x] Add down-proj-only sparse experiment.
- [x] Add strict QA guards with forbidden keywords and max_words.
- [ ] Improve sparse kernel throughput.
- [x] Validate on the 70M proof model.
- [x] Validate on the 100M proof model.

### v0.7a: 70M MLGRU scaling probe

- [x] Add config-driven 70M MLGRU run path.
- [x] Add 70M parameter estimate and benchmark commands.
- [x] Train and export `leviathan_mlgru_70m_instruct_v07a`.
- [x] Run dense and histogram Top-K sweep benchmarks.
- [x] Identify QA repair need before treating v07a as a speed candidate.

### v0.7b: 70M QA repair fine-tune

- [x] Add v07b QA repair config from the v07a checkpoint.
- [x] Document v07a failure patterns and v07b benchmark plan.
- [x] Train and export `leviathan_mlgru_70m_instruct_v07b`.
- [x] Re-run dense QA and histogram down-only sweep.
- [x] Document repeat-30 Top-K `0.08` local CPU speed candidate.
- [x] Probe 100M scaling.

### v0.8a: 100M MLGRU scaling probe

- [x] Add config-driven 100M MLGRU run path.
- [x] Add 100M parameter estimate and benchmark commands.
- [x] Train and export `leviathan_mlgru_100m_instruct_v08a`.
- [x] Run dense QA and histogram down-only sweep.
- [x] Document repeat-30 Top-K `0.06` local CPU speed candidate.
- [ ] Test on additional hardware.

### v0.9: 150M/200M MLGRU scaling probe

- [ ] Add config-driven 150M/200M MLGRU run path.
- [ ] Prefer a 200M-class MLGRU target unless dry-run, VRAM, or cost says otherwise.
- [ ] Expand QA/instruction data beyond TinyStories plus the small project QA seed.
- [ ] Run dense QA and histogram down-only sweep on the selected larger model.
- [ ] Test on additional CPU hardware.

### v1.0 target

- [x] Publish initial GitHub source + proof benchmarks.
- [x] Publish initial Hugging Face proof model packages.
- [x] Demonstrate a trained recurrent ternary model running locally on CPU.
- [ ] Demonstrate sparse activation speedups on a larger model or improved sparse kernel.
- [x] Add repeatable benchmark automation.

---

## Goal

Leviathan aims to become a minimal, hardware-facing inference stack for 1.58-bit and matmul-free language models:

```text
compact ternary packaging
CPU-resident inference
SIMD bitlinear execution
activation sparsity experiments
recurrent MLGRU-style runtime paths
```

The immediate goal is not to claim a finished general-purpose LLM. The immediate goal is to build a reproducible route from training to ternary CPU inference and then scale it responsibly.
