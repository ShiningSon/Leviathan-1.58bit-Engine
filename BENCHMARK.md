# Leviathan Benchmarks

This file records observed runtime results for Leviathan models.

The current benchmark is an early proof benchmark. It is not a broad hardware study yet.
It is intended to document what currently works, what does not yet work, and what needs to be optimized next.

The canonical compact 30M/70M/100M/200M scaling summary is [`benchmarks/results/scaling_summary.json`](benchmarks/results/scaling_summary.json). The detailed sections below preserve historical context, negative experiments, and model-specific measurements.

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
Initial v0.1 engine commit: 6a05e63ab636
v0.1 model package: https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories
v0.2b model package: https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02b
v0.2g model package: https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02g
```

Runtime notes:

- The first run includes C++ extension compilation.
- Reported tokens/sec below should be interpreted as interactive local runtime observations.
- Repeatable benchmark automation is available through `scripts/benchmark_engine.py`.
- Generated `benchmark_runs/` outputs are local artifacts and should not be committed.

---

## Canonical scaling summary

| Model | Dense | Recommended sparse mode | Sparse result | Strict QA | Tokens/sec delta | Latency delta | Status |
|---|---:|---|---:|---:|---:|---:|---|
| 30M v02g route | 50.74 ms / 350.97 tok/s | Top-K `0.12`, histogram, down-only | 50.25 ms / 354.17 tok/s | 950/1000 both | +0.91% | -0.97% | Experimental local CPU candidate; small margin |
| 70M v07b | 72.85 ms / 238.50 tok/s | Top-K `0.08`, histogram, down-only | 68.54 ms / 254.12 tok/s | 600/600 both | +6.55% | -5.92% | Experimental local CPU candidate |
| 100M v08a | 101.15 ms / 176.73 tok/s | Top-K `0.06`, histogram, down-only | 90.39 ms / 191.14 tok/s | 570/600 -> 600/600 | +8.15% | -10.64% | Experimental local CPU candidate |
| 200M v09a | 193.92 ms / 88.30 tok/s | Top-K `0.10`, histogram, down-only | 175.33 ms / 97.54 tok/s | 600/600 both | +10.46% | -9.58% | Current experimental local CPU candidate |

These are model-specific observations from one local CPU environment. They do not prove that Top-K is always faster, that one density is universal, or that the measured behavior automatically transfers to other hardware or larger models. The JSON summary is validated by `scripts/validate_benchmark_summary.py`.

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
Subsequent v09a work completed the 200M-class scaling probe. Remaining work focuses on additional CPU hardware, broader QA coverage, and lower-overhead sparse kernels.
```

---

## Completed benchmark milestones and next targets

### Completed: v0.2 instruction-mix 30M

Goal:

```text
TinyStories + small QA/instruction mixture
```

Outcome:

- v02b established the supervised QA proof route.
- v02c and v02d were not successful final candidates.
- v02e/v02f/v02g checkpoint fine-tuning restored strict QA to 95.0%.
- Current published 30M Hugging Face proof package: `Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2g`.

### Completed: v0.3 automated benchmark script

Goal:

```text
Run the same prompt set across dense and sparse modes.
Repeat each mode multiple times.
Export results to JSON and Markdown.
```

Outcome:

- `scripts/benchmark_engine.py` is the repeatable benchmark path.
- Strict QA guards include expected keywords, forbidden keywords, and max word limits.

### Completed/active: v0.4 and v0.5 sparse-kernel experiments

Current findings:

- Threshold sparsity and naive block Top-K were negative experiments and should not be included in the active runtime path.
- Down-proj-only sparse scope was the most stable sparse setting.
- Down-only Top-K preserved strict QA in tested v02e modes, but did not improve measured latency or token throughput.
- Dense remains the recommended speed path for the 30M proof model.

### Completed/active: v0.6 histogram selector and interleaved scheduling

Current findings:

- Projection-slot profiling showed the sparse down projection can be faster than the dense down projection, but Top-K selection overhead can erase the gain.
- The experimental histogram Top-K selector reduces selection overhead by using int8 activation magnitude buckets.
- Interleaved benchmark scheduling reduces mode-order bias when comparing dense and sparse modes.
- On the v02g 30M proof model, down-only Top-K `0.12` is the current experimental 30M speed candidate, not a general sparse speedup claim.

### Completed/active: v0.7b 70M QA repair and speed candidate

Current findings:

- v07a showed a 70M sparse scaling signal but only reached 170/200 strict QA, so it was not a valid speed candidate.
- v07b repaired strict QA to 600/600 in repeat-30 testing.
- On the v07b 70M proof model, histogram down-only Top-K `0.08` is the confirmed 70M experimental local CPU speed candidate.
- This is not a general sparse speedup claim, not a claim that Top-K is always faster, and not evidence that the result automatically scales to larger models or other hardware.

### Completed/active: v0.8a 100M scaling validation

Current findings:

- v08a is the first 100M MLGRU proof model where histogram Top-K down-only sparse inference produced a repeat-30 local CPU speed candidate.
- Top-K `0.06` reached 90.39 ms / 191.14 tok/s versus dense 101.15 ms / 176.73 tok/s, while QA improved from 570/600 to 600/600 in the measured run.
- This is not a general sparse speedup claim, not a claim that Top-K is always faster, and not evidence that the result automatically scales to larger models or other hardware.

### Completed/active: v0.9a 200M scaling validation

Current findings:

- v09a is the current active 200M MLGRU local CPU speed candidate.
- Top-K `0.10` histogram down-only reached 175.33 ms / 97.54 tok/s versus dense 193.92 ms / 88.30 tok/s in repeat-30 interleaved testing.
- Strict QA stayed at 600/600 for dense and Top-K `0.10`.
- Top-K `0.08` was faster but dropped strict QA to 570/600, so it is not the recommended v09a mode.
- This is not a general sparse speedup claim, not a claim that Top-K is always faster, and not evidence that the result automatically scales to larger models or other hardware.

### Next: broader validation

Measure:

- Dense baseline speed.
- Down-only Top-K quality preservation.
- Actual tokens/sec speedup only if measured tokens/sec exceeds dense under the same settings.
- Additional hardware behavior.
- Optional 300M/500M behavior only after new budget or credits.

---

## Current recommendation

- For the published 30M Hugging Face package, dense remains the stable default. Histogram down-only Top-K `0.12` is a small experimental local CPU candidate from the v0.6 benchmark.
- For the 70M local package, histogram down-only Top-K `0.08` is the confirmed v07b local CPU candidate.
- For the 100M local package, histogram down-only Top-K `0.06` is the confirmed v08a local CPU candidate.
- For the 200M local package, histogram down-only Top-K `0.10` is the current confirmed v09a local CPU candidate.

The v07b 70M, v08a 100M, and v09a 200M packages are published experimental Leviathan proof packages on Hugging Face. Future package revisions should use the reviewed cards under `hf_cards/` before upload.

```text
ShiningSon/Leviathan-MLGRU-70M-TinyStories-Instruct-v07b
ShiningSon/Leviathan-MLGRU-100M-TinyStories-Instruct-v08a
ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a
```

All four Top-K candidates are experimental and local-CPU-specific. Do not claim general sparse speedup, do not claim Top-K is always faster, and do not assume the results automatically scale to larger models or other hardware.

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

---

## v0.5 sparse-scope strict QA benchmark

Earlier all-projection sparsity was unstable for the 30M v0.2b proof model, and FFN-only Top-K `0.3` degraded QA behavior. Down-proj-only sparsity was the most stable sparse scope in this local run.

Strict QA scoring adds forbidden-keyword and `max_words` guards so TinyStories contamination or story-tail continuations are not counted as clean QA passes. Top-K `0.2` with down-proj-only sparsity is the current representative quality-stable sparse-scope candidate.

```text
Model: Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2b
Prompt template: qa
Max new tokens: 80
Repeats: 10
Warmup runs per mode: 1
Prompt set: expanded benchmarks/prompts_v02b_qa.json, 20 prompts
Sparse min density: 0.6
No Top-K sort: True
Sparse scope: down
```

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate | Notes |
|---|---:|---:|---:|---|
| Dense `--top-k 0` | 54.07 ms | 372.08 tok/s | 170/200 (85.0%) | Strict QA baseline |
| Top-K `--top-k 0.18` down-only | 55.25 ms | 360.72 tok/s | 170/200 (85.0%) | Matched dense strict QA, slower throughput |
| Top-K `--top-k 0.2` down-only | 52.68 ms | 360.36 tok/s | 170/200 (85.0%) | Matched dense strict QA, best latency in this run |
| Top-K `--top-k 0.25` down-only | 55.86 ms | 352.83 tok/s | 170/200 (85.0%) | Matched dense strict QA, slower throughput |

### Interpretation

```text
This historical v0.5 run showed a latency-positive signal for down-proj-only Top-K 0.2 while matching dense strict-QA pass rate. The later v02e sparse comparison below did not reproduce a latency or throughput improvement over dense, so the current conclusion remains QA preservation only, not sparse speedup. Strict QA guards caught TinyStories contamination using forbidden keywords and max_words.
```

---

## v0.2g final proof-model candidate strict QA benchmark

`leviathan_mlgru_30m_instruct_v02g` is the published 30M Hugging Face proof package for the Leviathan MLGRU path. It is still a small proof model for validating training, export, and local CPU runtime behavior; it is not a general assistant and is no longer the top active local speed candidate.

```text
Model: Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2g
HF package: https://huggingface.co/ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct-v02g
Prompt template: qa
Max new tokens: 80
Repeats: 10
Warmup runs per mode: 1
Prompt set: expanded benchmarks/prompts_v02b_qa.json, 20 prompts
Mode: dense only
```

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate | Notes |
|---|---:|---:|---:|---|
| Dense `--top-k 0` | 49.45 ms | 360.50 tok/s | 190/200 (95.0%) | Uploaded v02g candidate dense strict-QA result |

Known limitation:

```text
The v02g TinyStories proof model retains one package-description failure due to residual story-style continuation. The prompt "What files are inside a Leviathan MLGRU model package?" returns .bin and metadata JSON, but misses tokenizer and adds a story-style "happy" continuation. This affects 1 of 20 strict QA prompts across repeats.
```

### Interpretation

```text
v02g restores strict QA to 95.0% on the local Leviathan MLGRU QA benchmark. This is a proof-model result, not a 100% QA claim and not a general-assistant claim.
```

---

## v0.2e sparse down-only strict QA benchmark

This table records the sparse-scope comparison that preceded v02g. Down-projection-only Top-K preserved strict QA in the tested modes, but dense remained faster in measured latency and tokens/sec. No sparse speedup is claimed.

```text
Model: Leviathan-MLGRU-30M-TinyStories-Instruct-v0.2e
Prompt template: qa
Max new tokens: 80
Repeats: 10
Warmup runs per mode: 1
Prompt set: expanded benchmarks/prompts_v02b_qa.json, 20 prompts
Sparse min density: 0.6
No Top-K sort: True
Sparse scope: down
```

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate | Notes |
|---|---:|---:|---:|---|
| Dense `--top-k 0` | 49.23 ms | 359.66 tok/s | 190/200 (95.0%) | v02e dense baseline |
| Top-K `--top-k 0.18` down-only | 50.42 ms | 355.52 tok/s | 190/200 (95.0%) | Preserved strict QA; slower than dense |
| Top-K `--top-k 0.20` down-only | 51.23 ms | 349.83 tok/s | 190/200 (95.0%) | Preserved strict QA; slower than dense |
| Top-K `--top-k 0.25` down-only | 51.64 ms | 350.50 tok/s | 190/200 (95.0%) | Preserved strict QA; slower than dense |

### Interpretation

```text
Down-only Top-K preserved the tested strict-QA pass rate across 0.18, 0.20, and 0.25. Dense still had the best measured latency and token throughput, so this is a QA-preservation result, not a sparse speedup result.
```

---

## v0.6 histogram Top-K selector candidate

This benchmark records the current experimental 30M speed candidate after adding projection-slot profiling, the histogram Top-K selector, and interleaved benchmark scheduling. It is a local CPU result on the v02g 30M MLGRU proof model, not a general sparse speedup claim and not evidence that the same result automatically scales to larger models.

```text
Automated benchmark: leviathan_mlgru_30m_instruct_v02g
Architecture: mlgru
Prompt template: qa
Max new tokens: 80
Repeats: 50
Warmup runs per mode: 1
Mode schedule: interleave
Sparse min density: 0.6
No Top-K sort: True
Top-K selector: histogram
Sparse scope: down
Prompt set: expanded benchmarks/prompts_v02b_qa.json, 20 prompts
```

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate | Notes |
|---|---:|---:|---:|---|
| Dense `--top-k 0` | 50.74 ms | 350.97 tok/s | 950/1000 (95.0%) | Dense baseline in repeat-50 interleaved run |
| Top-K `--top-k 0.12` histogram down-only | 50.25 ms | 354.17 tok/s | 950/1000 (95.0%) | Experimental 30M speed candidate; same strict QA pass rate |

### Interpretation

```text
v0.6 adds an experimental histogram Top-K selector and interleaved benchmark scheduling. On the v02g 30M MLGRU proof model, down-projection-only Top-K at density 0.12 produced a small local CPU speed candidate in repeat-50 interleaved testing: 50.25 ms / 354.17 tok/s versus dense 50.74 ms / 350.97 tok/s, while preserving strict QA at 950/1000 (95.0%).

The latency improvement is about 0.97%, and the tokens/sec improvement is about 0.91%. This is a 30M experimental result, not a general sparse speedup claim. The later v07b, v08a, and v09a sections record the completed 70M, 100M, and 200M follow-ups under the same strict QA and interleaved scheduling discipline.
```

Known limitation:

```text
The same package-description prompt remains the known QA failure: "What files are inside a Leviathan MLGRU model package?" The model says .bin and metadata JSON but misses tokenizer and adds a TinyStories-style "happy" continuation.
```

---

## v0.7b 70M histogram Top-K speed candidate

This benchmark records the first 70M MLGRU proof model where histogram Top-K down-only sparse inference produced a clear local CPU speed candidate while preserving strict QA. It is a local CPU proof-model result, not a general sparse speedup claim, not a claim that Top-K is always faster, and not evidence that the result automatically scales to larger models or other hardware.

```text
Automated benchmark: leviathan_mlgru_70m_instruct_v07b
Architecture: mlgru
Prompt template: qa
Max new tokens: 80
Repeats: 30
Warmup runs per mode: 1
Mode schedule: interleave
Sparse min density: 0.6
No Top-K sort: True
Top-K selector: histogram
Sparse scope: down
Prompt set: expanded benchmarks/prompts_v02b_qa.json, 20 prompts
```

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate | Notes |
|---|---:|---:|---:|---|
| Dense `--top-k 0` | 72.85 ms | 238.50 tok/s | 600/600 (100.0%) | Dense baseline in repeat-30 interleaved run |
| Top-K `--top-k 0.08` histogram down-only | 68.54 ms | 254.12 tok/s | 600/600 (100.0%) | Confirmed v07b 70M experimental local CPU speed candidate |
| Top-K `--top-k 0.10` histogram down-only | 68.84 ms | 252.97 tok/s | 600/600 (100.0%) | Preserved strict QA; slower than Top-K 0.08 in this run |
| Top-K `--top-k 0.12` histogram down-only | 69.48 ms | 250.62 tok/s | 600/600 (100.0%) | Preserved strict QA; slower than Top-K 0.08 in this run |

### Interpretation

```text
v07b is the first 70M MLGRU proof model where histogram Top-K down-only sparse inference produced a clear local CPU speed candidate while preserving strict QA. In interleaved repeat-30 testing, Top-K 0.08 reached 68.54 ms / 254.12 tok/s versus dense 72.85 ms / 238.50 tok/s, with both modes at 600/600 strict QA.

The latency improvement is about 5.92%, and the tokens/sec improvement is about 6.55%. This is an experimental 70M local CPU result, not a general sparse speedup claim. The v08a benchmark below is the 100M follow-up, and v09a is the completed 200M follow-up.
```

---

## v0.8a 100M MLGRU scaling probe

v08a is the completed 100M scaling validation target after the v07b 70M result. The repeat-30 benchmark below records the confirmed 100M experimental local CPU speed candidate.

```text
Target model: leviathan_mlgru_100m_instruct_v08a
Architecture: mlgru
Vocab size: 8192
Hidden size: 896
Layers: 8
Intermediate size: 3072
Heads: 1
KV heads: 1
Activation: relu2
Estimated trainable parameters: 99,137,408, about 99.14M
Config: training/configs/v08a_100m_mlgru.json
Runbook: docs/V08A_100M_SCALING_PROBE.md
```

Original interpretation rules:

```text
If dense QA is below 90%, v08a needs v08b QA repair before sparse speed conclusions.
If dense QA is 90-95%, sparse sweep results should be treated as a scaling signal only.
If dense QA is 95-100% and Top-K preserves QA while beating dense tokens/sec, v08a becomes the 100M speed candidate.
Do not claim general sparse speedup, do not claim Top-K is always faster, and do not claim that one local CPU result automatically scales to larger models or other hardware.
```

---

## v0.8a 100M histogram Top-K speed candidate

This benchmark records the first 100M MLGRU proof model where histogram Top-K down-only sparse inference produced a repeat-30 local CPU speed candidate. It is a local CPU proof-model result, not a general sparse speedup claim, not a claim that Top-K is always faster, and not evidence that the result automatically scales to larger models or other hardware.

```text
Automated benchmark: leviathan_mlgru_100m_instruct_v08a
Architecture: mlgru
Prompt template: qa
Max new tokens: 80
Repeats: 30
Warmup runs per mode: 1
Mode schedule: interleave
Sparse min density: 0.6
No Top-K sort: True
Top-K selector: histogram
Sparse scope: down
Prompt set: expanded benchmarks/prompts_v02b_qa.json, 20 prompts
```

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate | Notes |
|---|---:|---:|---:|---|
| Dense `--top-k 0` | 101.15 ms | 176.73 tok/s | 570/600 (95.0%) | Dense baseline; known package-description failure remains |
| Top-K `--top-k 0.06` histogram down-only | 90.39 ms | 191.14 tok/s | 600/600 (100.0%) | Confirmed v08a 100M experimental local CPU speed candidate |
| Top-K `--top-k 0.08` histogram down-only | 91.13 ms | 189.82 tok/s | 600/600 (100.0%) | Preserved 600/600 QA; slower than Top-K 0.06 in this run |
| Top-K `--top-k 0.10` histogram down-only | 93.45 ms | 189.88 tok/s | 570/600 (95.0%) | Matched dense QA pass rate; slower than Top-K 0.06 in this run |

### Interpretation

```text
v08a is the first 100M MLGRU proof model where histogram Top-K down-only sparse inference produced a repeat-30 local CPU speed candidate. Top-K 0.06 reached 90.39 ms / 191.14 tok/s versus dense 101.15 ms / 176.73 tok/s, while QA improved from 570/600 to 600/600 in the measured run.

The latency improvement is about 10.64%, and the tokens/sec improvement is about 8.15%. Top-K 0.08 also preserved 600/600 QA and improved tokens/sec by about 7.41%. This is an experimental local CPU result, not a general sparse speedup claim.
```

Known dense limitation:

```text
Dense mode still shows the familiar package-description failure: "What files are inside a Leviathan MLGRU model package?" It mentions .bin and metadata JSON but misses tokenizer and falls into a TinyStories-style continuation.
```

---

## v0.9a 200M histogram Top-K speed candidate

This benchmark records the confirmed v09a 200M local CPU speed candidate. It is a Leviathan MLGRU proof-model result on one local CPU setup, not a general sparse speedup claim, not a claim that Top-K is always faster, and not evidence that other hardware or larger-model scaling is automatically proven.

```text
Model: Leviathan-MLGRU-200M-TinyStories-Instruct-v0.9a
Package folder: leviathan_mlgru_200m_instruct_v09a
Architecture: mlgru
Prompt template: qa
Max new tokens: 80
Warmup runs per mode: 1
Mode schedule: interleave
Sparse min density: 0.6
No Top-K sort: True
Top-K selector: histogram
Sparse scope: down
Prompt set: expanded benchmarks/prompts_v02b_qa.json, 20 prompts
```

### Repeat-10 quality recovery sweep

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate | Notes |
|---|---:|---:|---:|---|
| Dense `--top-k 0` | 191.27 ms | 89.34 tok/s | 200/200 (100.0%) | Dense baseline |
| Top-K `--top-k 0.08` histogram down-only | 169.35 ms | 100.90 tok/s | 190/200 (95.0%) | Fastest sweep mode, but QA dropped |
| Top-K `--top-k 0.10` histogram down-only | 173.50 ms | 98.59 tok/s | 200/200 (100.0%) | Quality-preserving candidate |
| Top-K `--top-k 0.12` histogram down-only | 175.52 ms | 97.43 tok/s | 200/200 (100.0%) | Preserved QA, slower than 0.10 |
| Top-K `--top-k 0.15` histogram down-only | 180.00 ms | 95.03 tok/s | 200/200 (100.0%) | Preserved QA, slower than 0.10 |
| Top-K `--top-k 0.20` histogram down-only | 182.77 ms | 93.41 tok/s | 200/200 (100.0%) | Preserved QA, slower than 0.10 |

### Confirmed repeat-30 result

| Mode | Avg latency | Avg tokens/sec | Strict QA pass rate | Notes |
|---|---:|---:|---:|---|
| Dense `--top-k 0` | 193.92 ms | 88.30 tok/s | 600/600 (100.0%) | Dense baseline |
| Top-K `--top-k 0.08` histogram down-only | 173.07 ms | 98.80 tok/s | 570/600 (95.0%) | Faster but not recommended because QA dropped |
| Top-K `--top-k 0.10` histogram down-only | 175.33 ms | 97.54 tok/s | 600/600 (100.0%) | Confirmed v09a 200M local CPU speed candidate |
| Top-K `--top-k 0.12` histogram down-only | 177.04 ms | 96.57 tok/s | 600/600 (100.0%) | Preserved QA, slightly slower than 0.10 |

### Interpretation

```text
Top-K 0.10 is the confirmed v09a 200M local CPU speed candidate. It improves tokens/sec by +10.46% and reduces latency by -9.58% versus dense in the repeat-30 interleaved run while preserving strict QA at 600/600.

Top-K 0.08 is faster but not recommended because strict QA drops to 570/600. Top-K 0.12 is also valid but slower than 0.10. This is a local CPU experimental proof-model result, not a general sparse speedup claim.
```
