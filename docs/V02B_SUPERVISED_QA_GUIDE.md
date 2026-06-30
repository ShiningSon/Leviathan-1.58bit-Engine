# Leviathan MLGRU v0.2b: supervised QA fix

The v0.2-alpha model learned the `Question: ... Answer:` format, but it mixed up
concept-to-answer grounding. Dense mode showed the same issue, so this is a data/training
problem rather than a Top-K problem.

## Key change

`03_train_supervised_qa_mlgru_modal_T4.py` changes QA sampling:

- v0.2-alpha sampled random windows from one concatenated QA token buffer.
- v0.2b samples complete QA examples.
- Prompt tokens are masked out of the QA loss.
- The model is trained mainly to predict the answer for the selected question.

## Files

Put these in your GitHub repo:

```text
training/03_train_supervised_qa_mlgru_modal_T4.py
training/instruction_qa_supervised_v02b.jsonl
docs/V02B_SUPERVISED_QA_GUIDE.md
```

## Smoke test

Run from the repo root:

```powershell
python -m modal run training/03_train_supervised_qa_mlgru_modal_T4.py --run-name leviathan_mlgru_30m_instruct_v02b_smoke --dataset tinystories --steps 100 --hidden-size 512 --n-layers 8 --intermediate-size 1536 --batch-size 16 --seq-len 96 --max-train-tokens 500000 --tokenizer-docs 5000 --qa-ratio 0.5 --lr 3e-4
```

## Main v0.2b run

```powershell
python -m modal run training/03_train_supervised_qa_mlgru_modal_T4.py --run-name leviathan_mlgru_30m_instruct_v02b --dataset tinystories --steps 1500 --hidden-size 512 --n-layers 8 --intermediate-size 1536 --batch-size 16 --seq-len 96 --max-train-tokens 2500000 --tokenizer-docs 10000 --qa-ratio 0.5 --lr 3e-4
```

## Download

```powershell
python -m modal volume get leviathan-mlgru /exports/leviathan_mlgru_30m_instruct_v02b.zip leviathan_mlgru_30m_instruct_v02b.zip
```

## Run locally

```powershell
Expand-Archive .\leviathan_mlgru_30m_instruct_v02b.zip -DestinationPath .\leviathan_mlgru_30m_instruct_v02b -Force
cd .\leviathan_mlgru_30m_instruct_v02b
python ..\engine.py --bin .\leviathan_mlgru_30m_instruct_v02b.bin --meta .\leviathan_mlgru_30m_instruct_v02b_meta.json --architecture mlgru --top-k 0 --max-new 80 --prompt-template qa
```

Use `--prompt-template qa` for project-specific QA checks. `--prompt-template plain` is only for raw continuation experiments and may break QA matching.

For the 30M proof model, dense mode is the default recommendation. Top-K `0.9` and `0.8` preserved the tested QA mappings, but this guide does not claim Top-K speedup.

## Test prompts

```text
What is Leviathan?
```

```text
What is MLGRU?
```

```text
What is Top-K activation sparsity?
```

```text
What dataset was this proof model trained on?
```

With `--prompt-template qa`, `engine.py` wraps each user prompt into the `Question: ... Answer:` format expected by the supervised QA model.

## Success criterion

v0.2b succeeds if it maps the core project concepts to the correct short answers with `--prompt-template qa`.
It does not need to become a general assistant.
