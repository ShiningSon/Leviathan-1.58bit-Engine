# Leviathan-1.58bit-Engine

Leviathan is a local CPU inference engine for native 1.58-bit ternary language
models. Its target is not generic post-training quantization of normal FP16
LLMs; the engine is built for QAT/native BitNet-style checkpoints whose linear
weights can live as ternary values `{-1, 0, 1}`.

The core direction is simple:

1. keep the model package small with row-major 2-bit ternary weights;
2. run bitlinear layers through CPU SIMD instead of PyTorch/CUDA kernels;
3. reduce per-token work with Top-K activation sparsity and delta hashing;
4. expose an experimental MatMul-Free MLGRU path that removes attention-map
   computation for compatible recurrent checkpoints.

## What Is Implemented

### 1. Litespark-style SIMD bitlinear

`engine.py` compiles a C++ runtime through `torch.utils.cpp_extension`. The
runtime consumes INT8 activations and row-packed ternary weights, then applies
hardware-friendly sign/add accumulation.

- AVX2 x86/x64 path: `_mm256_sign_epi8` based fused bitlinear loop.
- Portable scalar fallback: same row-packed format for non-AVX2 targets.
- Packing fix: each weight row is padded independently, so odd hidden sizes no
  longer corrupt row boundaries.
- BitNet structure fix: grouped-query attention is handled with separate
  `num_attention_heads` and `num_key_value_heads`, so K/V projections such as
  `[640, 2560]` are no longer read as `[2560, 2560]`.
- BitNet fidelity fix: `attn_sub_norm`, `ffn_sub_norm`, `relu2`, and the model's
  RoPE theta are applied from the checkpoint configuration.

The code is structured so AVX-512 VNNI or ARM NEON SDOT kernels can be added
behind the same `bitlinear_dispatch` boundary.

### 2. Top-K activation sparsity + delta hashing

Passing `--top-k K` activates sparse bitlinear inference. For each bitlinear
input vector, the runtime selects the `K` largest-magnitude INT8 activations and
only reads the matching ternary columns. A per-layer/projection delta hash tracks
whether the active index set changed from the previous step.

This is the first concrete implementation of the image requirement: reduce the
effective dense `out_dim * in_dim` work to `out_dim * K` for bitlinear layers.

### 3. MatMul-Free MLGRU runtime mode

Passing `--architecture mlgru` replaces the transformer attention map with a
GRU-like recurrent state update. The mode reuses the ternary projection kernel
and avoids the `Q @ K.T` attention score matrix.

This path is experimental. Existing transformer checkpoints are not guaranteed
to produce meaningful text in MLGRU mode. It is intended for MLGRU-compatible
QAT checkpoints or kernel benchmarking.

## Repository Layout

```text
.
├── quantizer.py   # Streams safetensors into Leviathan v2 binary packages
├── engine.py      # C++/OpenMP SIMD runtime with dense, Top-K, and MLGRU paths
└── README.md      # Architecture and usage notes
```

## Quick Start

Install the Python dependencies and make sure a C++ compiler with OpenMP support
is available. On Windows, PyTorch C++ extensions expect the MSVC toolchain
(`cl.exe`), usually from Visual Studio Build Tools or a Developer PowerShell.

```bash
pip install torch safetensors huggingface_hub transformers ninja
```

Create a Leviathan package from a native BitNet/QAT checkpoint:

```bash
python quantizer.py --model microsoft/bitnet-b1.58-2B-4T-bf16 --out leviathan_native
```

Run the transformer path:

```bash
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json
```

The default `--prompt-template auto` wraps question-shaped inputs as a completion:

```text
Question: What is the capital of France?
Answer:
```

Use `--prompt-template plain` if you want raw continuation behavior.

Run with Top-K sparse bitlinear:

```bash
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json --top-k 512
```

Run the experimental MLGRU path:

```bash
python engine.py --bin leviathan_native.bin --meta leviathan_native_meta.json --architecture mlgru --top-k 512
```

## Quantizer Format

`quantizer.py` writes a `leviathan-v2` metadata file.

- Linear projection weights are stored as `ternary_2bit_packed`.
- Embeddings, lm_head/output heads, norms, biases, and other structural tensors
  remain FP16 so the engine can load them without semantic corruption.
- Packed linear tensors include `shape`, `packed_shape`, `gamma`, `offset`, and
  `size`.

By default, the quantizer refuses non-BitNet model ids because PTQ conversion of
ordinary FP16 models into 1.58-bit ternary weights usually causes quantization
collapse. For systems-only experiments, pass `--allow-ptq`.

## Limitations

- Coherent generation requires native 1.58-bit/QAT weights. PTQ of ordinary
  FP16 models is useful for stress testing the runtime, not for quality.
- The MLGRU mode is a real runtime path, but it needs compatible trained
  weights to be a faithful language model.
- Top-K sparsity trades accuracy for throughput unless the model was trained or
  calibrated for activation sparsity.
- AVX-512 VNNI and ARM NEON SDOT are prepared as extension points, not yet
  fully specialized kernels.

## Garbled Output Checklist

If the engine prints fluent-looking speed stats but nonsense text, check these
first:

- K/V projection rows must match the model's grouped-query attention shape. For
  `microsoft/bitnet-b1.58-2B-4T-bf16`, `k_proj` and `v_proj` are 640-wide, not
  2560-wide.
- The runtime should report `heads=20, kv_heads=5, kv_dim=640, act=relu2,
  rope_theta=500000`.
- Run dense mode first. Add `--top-k` only after dense generation is coherent.
- Use completion-style prompts for base models. The default `auto` template does
  this for inputs ending in `?`; the checkpoint is not an instruction-tuned
  assistant.

## Goal

Leviathan aims to become a minimal, hardware-facing inference engine for
1.58-bit models: compact binary packaging, CPU-resident weights, SIMD bitlinear
execution, sparse activation scheduling, and an attention-free recurrent path
for next-generation matmul-free architectures.
