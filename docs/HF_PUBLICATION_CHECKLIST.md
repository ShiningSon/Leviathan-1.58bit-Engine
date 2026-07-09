# Hugging Face Publication Checklist

This checklist prepares the v07b 70M, v08a 100M, and v09a 200M Leviathan MLGRU proof packages for Hugging Face publication.

Do not commit or upload unrelated local artifacts. GitHub keeps source, docs, benchmark definitions, and helper scripts. Hugging Face receives the model package files.

## Target Repositories

```text
ShiningSon/Leviathan-MLGRU-70M-TinyStories-Instruct-v07b
ShiningSon/Leviathan-MLGRU-100M-TinyStories-Instruct-v08a
ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a
```

## Required Files

v07b package:

```text
leviathan_mlgru_70m_instruct_v07b/
  README.md
  leviathan_mlgru_70m_instruct_v07b.bin
  leviathan_mlgru_70m_instruct_v07b_meta.json
  leviathan_mlgru_tokenizer/
  report.json
  sample_outputs.txt
```

v08a package:

```text
leviathan_mlgru_100m_instruct_v08a/
  README.md
  leviathan_mlgru_100m_instruct_v08a.bin
  leviathan_mlgru_100m_instruct_v08a_meta.json
  leviathan_mlgru_tokenizer/
  report.json
  sample_outputs.txt
```

v09a package:

```text
leviathan_mlgru_200m_instruct_v09a/
  README.md
  leviathan_mlgru_200m_instruct_v09a.bin
  leviathan_mlgru_200m_instruct_v09a_meta.json
  leviathan_mlgru_tokenizer/
  report.json
  sample_outputs.txt
```

## Do Not Upload

```text
v07a packages
failed intermediate checkpoints
training checkpoints
benchmark_runs/
exports/
checkpoints/
runs/
.venv/
venv/
.env
.env.*
*.pt
*.pth
*.ckpt
*.safetensors
*.zip
local patch or backup files
```

## Copy Model Cards

Run from the repository root:

```cmd
copy hf_cards\Leviathan-MLGRU-70M-TinyStories-Instruct-v07b_README.md leviathan_mlgru_70m_instruct_v07b\README.md
copy hf_cards\Leviathan-MLGRU-100M-TinyStories-Instruct-v08a_README.md leviathan_mlgru_100m_instruct_v08a\README.md
copy hf_cards\Leviathan-MLGRU-200M-TinyStories-Instruct-v09a_README.md leviathan_mlgru_200m_instruct_v09a\README.md
```

## Upload With Hugging Face CLI

Authenticate first with the normal Hugging Face CLI login flow.

```cmd
hf upload ShiningSon/Leviathan-MLGRU-70M-TinyStories-Instruct-v07b .\leviathan_mlgru_70m_instruct_v07b .
hf upload ShiningSon/Leviathan-MLGRU-100M-TinyStories-Instruct-v08a .\leviathan_mlgru_100m_instruct_v08a .
hf upload ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a .\leviathan_mlgru_200m_instruct_v09a .
```

## Optional Python Upload Helper

The helper script creates the target model repo if needed and uploads a folder. It does not hardcode a token and uses the normal Hugging Face login/cache.

```cmd
python scripts\publish_hf_package.py --repo-id ShiningSon/Leviathan-MLGRU-70M-TinyStories-Instruct-v07b --folder .\leviathan_mlgru_70m_instruct_v07b
python scripts\publish_hf_package.py --repo-id ShiningSon/Leviathan-MLGRU-100M-TinyStories-Instruct-v08a --folder .\leviathan_mlgru_100m_instruct_v08a
python scripts\publish_hf_package.py --repo-id ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a --folder .\leviathan_mlgru_200m_instruct_v09a
```

To create a private repo:

```cmd
python scripts\publish_hf_package.py --repo-id ShiningSon/Leviathan-MLGRU-70M-TinyStories-Instruct-v07b --folder .\leviathan_mlgru_70m_instruct_v07b --private true
```

## Post-Upload Verification

- Confirm each Hugging Face page opens.
- Confirm the model card starts with YAML metadata and has no YAML metadata warning.
- Confirm `README.md` appears at the repo root.
- Confirm `.bin`, metadata JSON, tokenizer folder, `report.json`, and `sample_outputs.txt` are present.
- Confirm v07b, v08a, and v09a pages say they are Leviathan runtime packages, not standard Transformers checkpoints.
- Confirm the cards say the models are not general assistants.
- Confirm the cards say this is not a general sparse speedup claim.
- Confirm Top-K is not described as always faster.
- Confirm larger-model and other-hardware scaling is not described as automatically proven.
- Confirm v07a, failed checkpoints, training checkpoints, and `benchmark_runs/` were not uploaded.
