# Leviathan MLGRU v0.2 Instruction-Mix Training Guide

This guide is for the next proof model after `Leviathan-MLGRU-30M-TinyStories-v0.1`.

## Goal

`v0.1` proved the full route:

```text
Modal T4 training -> fake ternary / fake int8 MLGRU -> Leviathan v2 export -> local CPU engine
```

`v0.2` keeps the same 30M architecture, but changes the data mixture:

```text
80% TinyStories continuation text
20% short Question/Answer instruction examples
```

The goal is not to create a general assistant. The goal is to check whether the proof model can start following simple QA formatting while preserving the Leviathan export/runtime path.

## Files to add to the GitHub repo

Copy these files into the repo:

```text
training/02_train_mixed_mlgru_modal_T4.py
training/instruction_qa_seed.jsonl
```

Optional docs:

```text
docs/V02_INSTRUCTION_MIX_GUIDE.md
```

## First smoke test

Run only 100 steps first. This checks that the Modal file, QA seed file, tokenizer, data cache, model, and export path work.

```bash
modal run training/02_train_mixed_mlgru_modal_T4.py \
  --run-name leviathan_mlgru_30m_instruct_v02_smoke \
  --dataset tinystories \
  --steps 100 \
  --hidden-size 512 \
  --n-layers 8 \
  --intermediate-size 1536 \
  --batch-size 16 \
  --seq-len 96 \
  --max-train-tokens 500000 \
  --tokenizer-docs 5000 \
  --qa-ratio 0.2 \
  --lr 3e-4
```

Download the smoke export:

```bash
modal volume get leviathan-mlgru /exports/leviathan_mlgru_30m_instruct_v02_smoke.zip leviathan_mlgru_30m_instruct_v02_smoke.zip
```

## First real v0.2 run

Start with 1000 steps. Do not jump directly to a long run until the smoke test works.

```bash
modal run training/02_train_mixed_mlgru_modal_T4.py \
  --run-name leviathan_mlgru_30m_instruct_v02 \
  --dataset tinystories \
  --steps 1000 \
  --hidden-size 512 \
  --n-layers 8 \
  --intermediate-size 1536 \
  --batch-size 16 \
  --seq-len 96 \
  --max-train-tokens 2000000 \
  --tokenizer-docs 10000 \
  --qa-ratio 0.2 \
  --lr 3e-4
```

Download the export:

```bash
modal volume get leviathan-mlgru /exports/leviathan_mlgru_30m_instruct_v02.zip leviathan_mlgru_30m_instruct_v02.zip
```

## Local engine test

Unzip the package, then run dense mode first:

```bash
python engine.py \
  --bin leviathan_mlgru_30m_instruct_v02.bin \
  --meta leviathan_mlgru_30m_instruct_v02_meta.json \
  --architecture mlgru \
  --top-k 0 \
  --max-new 120 \
  --prompt-template plain
```

Test prompts:

```text
Once upon a time
```

```text
Question: What is Leviathan?
Answer:
```

```text
Question: What is MLGRU?
Answer:
```

```text
Question: What is Top-K activation sparsity?
Answer:
```

## Success criteria

`v0.2` is successful if:

```text
1. The model still exports to Leviathan v2.
2. engine.py can run it with --architecture mlgru.
3. Dense mode works first.
4. The model starts following Question/Answer formatting at least partially.
5. TinyStories-style continuation is not completely destroyed.
```

Perfect factual answers are not required. This is still a 30M proof model.

## If QA is too weak

Try a higher QA ratio for a short run:

```bash
--qa-ratio 0.35
```

Do not go too high at first. If QA dominates the data, the model may forget story continuation and overfit the small seed file.

## If training is too slow

Use shorter runs:

```bash
--steps 500
--max-train-tokens 1000000
```

## If output becomes repetitive

That is expected. The current engine path is mostly greedy generation. Sampling options can be added later.

## What to publish after v0.2

If the result is better than v0.1, publish it as either:

```text
ShiningSon/Leviathan-MLGRU-30M-TinyStories-Instruct
```

or as a new revision/file set under the existing Hugging Face model repo.

Recommended v0.2 model package files:

```text
leviathan_mlgru_30m_instruct_v02.bin
leviathan_mlgru_30m_instruct_v02_meta.json
leviathan_mlgru_tokenizer/
report.json
sample_outputs.txt
README.md
```
