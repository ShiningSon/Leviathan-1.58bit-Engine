# Leviathan-1.58bit-Engine

Leviathan is a local CPU inference engine for native 1.58-bit / ternary language models.
Its target is not generic post-training quantization of ordinary FP16 LLMs; the runtime is built around QAT/native BitNet-style or Leviathan-trained checkpoints whose linear weights can live as ternary values `{-1, 0, 1}`.

The current direction is:

1. keep model packages compact with row-major 2-bit ternary weights;
2. execute bitlinear layers through a C++/OpenMP CPU runtime instead of PyTorch kernels;
3. experiment with ratio-based Top-K activation sparsity;
4. support an experimental attention-free MLGRU runtime path for recurrent checkpoints trained for that path.

> Status: research prototype. The dense ternary CPU path and MLGRU export path are working. Ratio Top-K is implemented and preserves generation quality in the 30M proof model, but it is not yet faster at this small scale because selection overhead dominates.

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

Examples:

```bash
--top-k 0.9    # keep 90% of each bitlinear input, per layer
--top-k 0.8    # keep 80% of each bitlinear input, per layer
--top-k 2048   # legacy absolute-K mode
```

The important change is that fractional values are treated as **per-layer density ratios** rather than fixed absolute K values.
This avoids the failure mode where a fixed K is acceptable for narrow projections but destroys wide FFN projections.

Current benchmark result:

- Ratio Top-K preserves readable output in the 30M MLGRU proof model.
- Dense mode is still faster at 30M scale.
- Top-K is currently a scaling and kernel-optimization path, not a guaranteed speedup for every model size.

See [`BENCHMARK.md`](BENCHMARK.md) for the current numbers.

### 3. MatMul-Free MLGRU runtime mode

Passing `--architecture mlgru` switches the runtime into an attention-free recurrent path.
This path replaces transformer attention maps with a GRU-like recurrent state update while still using Leviathan's ternary projection kernels.

Important limitation:

- Existing transformer checkpoints are not expected to work correctly in MLGRU mode.
- MLGRU mode needs a checkpoint trained for the recurrent path.

Current proof model:

- `Leviathan-MLGRU-30M-TinyStories-v0.1`
- trained from scratch on Modal T4
- dataset: TinyStories
- steps: 2000
- exported to Leviathan v2 format
- runs locally with `engine.py --architecture mlgru`
- model package: [ShiningS04/Leviathan-MLGRU-30M-TinyStories](https://huggingface.co/ShiningS04/Leviathan-MLGRU-30M-TinyStories)

The proof model is intentionally small. It is meant to validate the full route:

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
|   `-- 01_train_export_mlgru_modal_T4_clean.py
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

Run with ratio Top-K:

```bash
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json --top-k 0.9
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json --top-k 0.8
```

Run with legacy absolute-K mode:

```bash
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json --top-k 2048
```

---

## Running the MLGRU proof model

The 30M proof model is not a general assistant model. It is a small recurrent language model trained on TinyStories to validate the MLGRU runtime/export path.

Download the model package from Hugging Face:

[ShiningS04/Leviathan-MLGRU-30M-TinyStories](https://huggingface.co/ShiningS04/Leviathan-MLGRU-30M-TinyStories)

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

Model:

```text
Leviathan-MLGRU-30M-TinyStories-v0.1
```

Local CPU runtime, prompt `Once upon a time`, `max-new=120`:

| Mode | Runtime setting | Observed tokens/sec | Notes |
|---|---:|---:|---|
| Dense MLGRU | `--top-k 0` | 383.90 tok/s | Best speed and coherence at 30M scale |
| Ratio Top-K | `--top-k 0.9` | 210.44 tok/s | Coherent, slower than dense |
| Ratio Top-K | `--top-k 0.8` | 205.81 tok/s | Coherent but altered, slower than dense |
| Ratio Top-K | `--top-k 0.7` | 238.95 tok/s | Still readable, slower than dense |

Interpretation:

```text
At 30M scale, dense mode is faster because Top-K selection overhead dominates.
Ratio Top-K preserves readable generation, but it is not yet a speed win at this model size.
Future work will focus on larger 70M/100M models and a lower-overhead sparse kernel.
```

See [`BENCHMARK.md`](BENCHMARK.md) for details.

---

## Model distribution

Do not commit large generated model packages directly into the normal Git history unless the package is very small and intentionally versioned.

Recommended distribution:

1. **GitHub repo**: source code, README, benchmark, training scripts.
2. **GitHub Releases**: small zipped proof packages and screenshots.
3. **Hugging Face model repo**: model package, tokenizer, metadata, model card, sample outputs.

Published Hugging Face model repository:

[ShiningS04/Leviathan-MLGRU-30M-TinyStories](https://huggingface.co/ShiningS04/Leviathan-MLGRU-30M-TinyStories)

Suggested files:

```text
README.md                         # Hugging Face model card
leviathan_mlgru_30m_t4.bin
leviathan_mlgru_30m_t4_meta.json
leviathan_mlgru_tokenizer/
report.json
sample_outputs.txt
```

The model package contains the exported MLGRU binary, metadata, tokenizer, model card, benchmark report, and sample outputs.

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
- Top-K is not faster at 30M scale in the current implementation.
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

### v0.2: instruction-mix model

- [ ] Add TinyStories + small instruction/QA mixture.
- [ ] Improve short QA behavior.
- [ ] Add sample outputs and model card.

### v0.3: benchmark automation

- [ ] Add repeatable benchmark script.
- [ ] Test multiple prompts and max-new values.
- [ ] Record CPU model and compiler details.

### v0.4: sparse kernel work

- [ ] Reduce Top-K selection overhead.
- [ ] Add threshold sparsity path.
- [ ] Avoid full sort where possible.
- [ ] Benchmark 70M/100M models.

### v1.0 target

- [ ] Publish GitHub source + benchmarks.
- [ ] Publish Hugging Face model package.
- [ ] Demonstrate a trained recurrent ternary model running locally on CPU.
- [ ] Demonstrate sparse activation speedups on a larger model or improved sparse kernel.

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
