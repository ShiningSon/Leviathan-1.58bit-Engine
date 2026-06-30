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
- Repeatable benchmark automation is planned.

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
- Prompt set: `benchmarks/prompts_v02b_qa.json`

| Mode | Setting | Avg latency | Avg tokens/sec | QA pass rate | Notes |
|---|---:|---:|---:|---:|---|
| Dense | `--top-k 0` | 57.64 ms | 313.84 tok/s | 12/12 (100.0%) | QA matching preserved in this run |
| Top-K 0.9 | `--top-k 0.9` | 113.83 ms | 140.05 tok/s | 12/12 (100.0%) | QA matching preserved in this run |
| Top-K 0.8 | `--top-k 0.8` | 105.03 ms | 153.49 tok/s | 12/12 (100.0%) | QA matching preserved in this run |

### Interpretation

```text
v0.2b demonstrates that a 30M Leviathan-trained MLGRU proof model can learn stable project-specific QA mappings when run with --prompt-template qa.
Dense remained the fastest observed mode for the 30M v0.2b model. Top-K 0.9 and 0.8 preserved the tested QA matching but did not show a speedup in this benchmark.
```
