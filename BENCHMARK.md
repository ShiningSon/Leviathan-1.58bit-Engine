# Leviathan Benchmarks

This file records observed runtime results for Leviathan models.

The current benchmark is an early proof benchmark. It is not a broad hardware study yet.
It is intended to document what currently works, what does not yet work, and what needs to be optimized next.

---

## Environment

```text
OS: Microsoft Windows 11 Pro 64-bit, version 10.0.26200, build 26200
CPU: 13th Gen Intel(R) Core(TM) i5-13600KF, 14 cores / 20 logical processors
RAM: 31.7 GiB visible system memory
Compiler: MSVC 19.51.36246 for x64 via Visual Studio Community / PyTorch C++ extension
Python: 3.10.11
PyTorch: 2.12.1+cpu
CUDA: not used for local inference benchmark
Engine commit: 6a05e63ab636
v0.1 model package: https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories
v0.2b model package: https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02b
```

Runtime notes:

- The first run includes C++ extension compilation.
- Reported tokens/sec below should be interpreted as interactive local runtime observations.
- Repeatable benchmark automation is available through `scripts/benchmark_engine.py`.
- Generated `benchmark_runs/` outputs are local artifacts and should not be committed.

---

## Model: Leviathan-MLGRU-30M-TinyStories-v0.1

### Training

```text
Architecture: Leviathan MLGRU
Dataset: TinyStories
Training platform: Modal T4
Steps: 2000
Hidden size: 512
Layers: 8
Intermediate size: 1536
Sequence length: 96
Batch size: 16
Export format: Leviathan v2
Linear weights: ternary_2bit_packed
Structural tensors: FP16
Tokenizer: local BPE tokenizer exported with package
```

### Runtime command pattern

Dense MLGRU:

```bash
python engine.py \
  --bin leviathan_mlgru_30m_t4.bin \
  --meta leviathan_mlgru_30m_t4_meta.json \
  --architecture mlgru \
  --top-k 0 \
  --max-new 120 \
  --prompt-template plain
```

Ratio Top-K MLGRU:

```bash
python engine.py \
  --bin leviathan_mlgru_30m_t4.bin \
  --meta leviathan_mlgru_30m_t4_meta.json \
  --architecture mlgru \
  --top-k 0.9 \
  --max-new 120 \
  --prompt-template plain
```

Prompt:

```text
Once upon a time
```

---

## Results

| Mode | Runtime setting | Observed latency | Observed speed | Output quality note |
|---|---:|---:|---:|---|
| Dense MLGRU | `--top-k 0` | 218.81 ms | 383.90 tok/s | Best speed and best coherence in this small model |
| Ratio Top-K | `--top-k 0.9` | 370.66 ms | 210.44 tok/s | Coherent, but slower than dense |
| Ratio Top-K | `--top-k 0.8` | 320.68 ms | 205.81 tok/s | Coherent but altered, slower than dense |
| Ratio Top-K | `--top-k 0.7` | 351.53 ms | 238.95 tok/s | Still readable, slower than dense |

---

## Sample outputs

### Dense MLGRU: `--top-k 0`

```text
USER> Once upon a time
ENGINE> , there was a little girl named Lily. She loved to play with her toys and go outside every day. One sunny day, she went for the beach when she saw lots of fun in the park!

The moral of this story is that it's important to be careful with things that don't belong to you are not so bad after all by herself before they could do anything like our friends again soon.
```

### Ratio Top-K: `--top-k 0.9`

```text
USER> Once upon a time
ENGINE> , there was a little girl named Lily. She loved to play with her toys and go outside every day. One sunny day, she saw the truck coming towards them from the fence and said it was too big for him!

"Look at that balloon!" said Max barked happily as he ran away quickly as fast as long as he could hurt his owner's cat again soon!
```

### Ratio Top-K: `--top-k 0.8`

```text
USER> Once upon a time
ENGINE> , there was a little girl named Lily. She loved to play outside and explore the world around her. One day she saw that it was not so big as long as she could!

The moral of this story is that when you are different things like you can do anything if we want to be friends with us.
```

### Ratio Top-K: `--top-k 0.7`

```text
USER> Once upon a time
ENGINE> , there was a little girl named Lily. She loved to play with her toys and go outside every day. One sunny day, she went for the beach and saw lots of fun on the swings when she heard someone else!

"What are you doing here?" asked Lucy smiled back as he said yes!

The moral of this story is that it's important to be kind to others too slow.
```

---

## Interpretation

### What worked

- A 30M recurrent MLGRU model was trained from scratch.
- The model was exported into Leviathan v2 format.
- The local CPU runtime loaded and generated text from the package.
- Ratio Top-K down to 70-80% preserved readable text in this small proof model.

### What did not yet work

- Ratio Top-K did not beat dense mode at 30M scale.
- Dense MLGRU was the fastest current path in the observed run.
- The proof model is still TinyStories-like and not an instruction-following assistant.

### Why dense wins at 30M

The model is too small for the current Top-K implementation to pay off.
At hidden size 512 and intermediate size 1536, dense ternary matvec work is relatively cheap.
Top-K selection, index sorting, and sparse memory access introduce overhead that dominates the runtime.

Current conclusion:

```text
At 30M scale, dense mode is faster because Top-K selection overhead dominates.
Ratio Top-K preserves readable generation, but it is not yet a speed win at this model size.
Future work will focus on larger 70M/100M models and a lower-overhead sparse kernel.
```

---

## Next benchmark targets

### v0.2: Instruction-mix 30M

Goal:

```text
TinyStories + small QA/instruction mixture
```

Measure:

- Dense speed.
- Top-K 0.9 / 0.8 quality.
- Simple QA output quality.

### v0.3: Automated benchmark script

Goal:

```text
Run the same prompt set across dense, top-k 0.9, top-k 0.8, top-k 0.7.
Repeat each mode multiple times.
Export results to JSON and Markdown.
```

### v0.4: Larger model

Goal:

```text
70M to 100M MLGRU model
```

Measure whether Top-K begins to catch up or beat dense when dense matvec work becomes larger.

### v0.5: Sparse kernel optimization

Potential changes:

- Threshold sparsity mode.
- Avoid full sort after Top-K selection.
- Cache active sets more aggressively.
- Explore block-sparse activation scheduling.
- Add AVX-512 / NEON specialized sparse paths.

---

## Current recommendation

For the 30M proof model:

```text
Use dense MLGRU mode for speed.
Use ratio Top-K for sparsity-quality experiments.
Do not claim Top-K speedup yet at this scale.
```

For future scaling:

```text
Top-K should be retested on 70M/100M models and after sparse-kernel optimization.
```

---

## Model: Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2b

```text
Prompt template: qa
Purpose: project-specific QA matching
```

### Runtime command pattern

Dense MLGRU QA:

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

### Automated benchmark results

Command settings:

- Architecture: `mlgru`
- Prompt template: `qa`
- Max new tokens: `80`
- Repeats: `3`
- Warmup runs per mode: `1`
- Prompt set: expanded `benchmarks/prompts_v02b_qa.json`, 20 prompts

| Mode | Setting | Avg latency | Avg tokens/sec | QA pass rate | Notes |
|---|---:|---:|---:|---:|---|
| Dense | `--top-k 0` | 54.17 ms | 371.08 tok/s | 54/60 (90.0%) | Expanded prompt set; review missing keyword matches |
| Top-K 0.9 | `--top-k 0.9` | 129.07 ms | 161.52 tok/s | 54/60 (90.0%) | Same QA pass rate as dense; slower than dense |
| Top-K 0.8 | `--top-k 0.8` | 112.35 ms | 172.15 tok/s | 54/60 (90.0%) | Same QA pass rate as dense; slower than dense |

### Interpretation

```text
The expanded prompt set increased the v0.2b QA regression check from 4 prompts to 20 prompts. Dense, Top-K 0.9, and Top-K 0.8 all reached 90.0% keyword-match pass rate. Dense remained the fastest observed mode. Top-K preserved the tested QA matching rate but did not show a speedup in this benchmark.

The remaining failures were concentrated around short paraphrases such as "What does Top-K do?" and one model-package wording. This is treated as a QA coverage limitation, not a runtime failure.
```

---

## v0.4 sparse-min-density fallback benchmark

```text
Model: Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2b
Prompt template: qa
Max new tokens: 80
Repeats: 3
Warmup runs per mode: 1
Prompt set: expanded benchmarks/prompts_v02b_qa.json, 20 prompts
Sparse min density: 0.6
```

| Mode | Setting | Avg latency | Avg tokens/sec | QA pass rate | Notes |
|---|---:|---:|---:|---:|---|
| Dense | `--top-k 0` | 56.97 ms | 353.06 tok/s | 54/60 (90.0%) | Dense baseline |
| Top-K 0.9 | `--top-k 0.9 --sparse-min-density 0.6` | 56.65 ms | 354.60 tok/s | 54/60 (90.0%) | High-density Top-K falls back to dense |
| Top-K 0.8 | `--top-k 0.8 --sparse-min-density 0.6` | 57.63 ms | 348.18 tok/s | 54/60 (90.0%) | High-density Top-K falls back to dense |
| Top-K 0.5 | `--top-k 0.5 --sparse-min-density 0.6` | 85.15 ms | 216.90 tok/s | 54/60 (90.0%) | Sparse path remains slower at 30M |

### Interpretation

```text
The v0.4 fallback benchmark shows that high-density Top-K should not be forced through the sparse path at 30M scale. With --sparse-min-density 0.6, Top-K 0.9 and 0.8 fall back to the dense kernel and preserve the same 90.0% QA keyword-match pass rate as dense. Top-K 0.5 still uses the sparse path but remains slower than dense. This is a runtime guardrail, not a Top-K speedup claim.
```

---

## v0.4 experimental no-sort Top-K benchmark

```text
Model: Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2b
Prompt template: qa
Max new tokens: 80
Repeats: 3
Warmup runs per mode: 1
Prompt set: expanded benchmarks/prompts_v02b_qa.json, 20 prompts
Sparse min density: 0.6
No Top-K sort: True
```

| Mode | Setting | Avg latency | Avg tokens/sec | QA pass rate | Notes |
|---|---:|---:|---:|---:|---|
| Dense | `--top-k 0` | 51.78 ms | 388.47 tok/s | 54/60 (90.0%) | Dense baseline |
| Top-K 0.5 no-sort | `--top-k 0.5 --sparse-min-density 0.6 --no-top-k-sort` | 63.34 ms | 294.97 tok/s | 54/60 (90.0%) | Preserved tested QA pass rate but slower than dense |
| Top-K 0.3 no-sort | `--top-k 0.3 --sparse-min-density 0.6 --no-top-k-sort` | 54.45 ms | 321.18 tok/s | 12/60 (20.0%) | Low-density no-sort degraded QA behavior |

### Interpretation

```text
The no-sort Top-K experiment reduces Top-K selection overhead by skipping the final sort of active indices after nth_element. In the 30M v0.2b benchmark, Top-K 0.5 with --no-top-k-sort preserved the same 90.0% QA keyword-match pass rate as dense but remained slower. Top-K 0.3 with --no-top-k-sort degraded QA behavior to 20.0%, so low-density no-sort should not be treated as stable. This is an experimental kernel path, not a speedup claim.
```
