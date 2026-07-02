"""
Train a Leviathan-compatible MLGRU v0.2c QA calibration proof model on Modal T4 using a
TinyStories + supervised instruction/QA mixture, then export it to the Leviathan v2
binary format that engine.py can run with --architecture mlgru.

This is the v0.2 route:
1) train the same 30M recurrent proof architecture with fake ternary weights and
   fake int8 activations,
2) mix TinyStories continuation text with supervised Question/Answer examples,
3) export packed ternary linear weights + FP16 structural tensors,
4) zip the exported files, report.json, tokenizer, and sample_outputs.txt.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "leviathan-mlgru-train-export-v02c-calibration"
VOLUME_NAME = "leviathan-mlgru"
VOL_PATH = "/data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "datasets",
        "transformers",
        "tokenizers",
        "tqdm",
    )
)

# Include the editable seed QA file in the Modal image when this script is run
# from the repository's training/ folder. If the file is missing, the script
# falls back to the embedded examples below.
LOCAL_QA_SEED = Path(__file__).with_name("instruction_qa_supervised_v02c.jsonl")
if LOCAL_QA_SEED.exists():
    try:
        image = image.add_local_file(str(LOCAL_QA_SEED), remote_path="/root/instruction_qa_supervised_v02c.jsonl")
    except AttributeError:
        # Older Modal clients may not expose add_local_file. The script still has
        # embedded QA examples below, so training can continue.
        pass


@app.function(
    image=image,
    gpu="T4",
    volumes={VOL_PATH: volume},
    timeout=6 * 60 * 60,
)
def train_and_export(
    run_name: str = "leviathan_mlgru_30m_instruct_v02c",
    dataset: str = "tinystories",
    vocab_size: int = 8192,
    hidden_size: int = 256,
    n_layers: int = 4,
    intermediate_size: int = 768,
    seq_len: int = 128,
    batch_size: int = 16,
    steps: int = 500,
    lr: float = 3e-4,
    max_train_tokens: int = 1_000_000,
    tokenizer_docs: int = 5000,
    qa_ratio: float = 0.7,
    qa_seed_path: str = "/root/instruction_qa_supervised_v02c.jsonl",
    seed: int = 1337,
):
    import json
    import math
    import os
    import random
    import shutil
    import time
    import zipfile
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    run_dir = Path(VOL_PATH) / "runs" / run_name
    export_dir = Path(VOL_PATH) / "exports" / run_name
    tokenizer_dir = export_dir / "leviathan_mlgru_tokenizer"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    def dataset_iter(name: str):
        if name == "tinystories":
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        elif name == "fineweb-edu":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        else:
            raise ValueError("dataset must be 'tinystories' or 'fineweb-edu'")
        for row in ds:
            text = row.get("text")
            if isinstance(text, str) and len(text.strip()) > 0:
                yield text.strip()

    EMBEDDED_QA_TEXTS = [
        "Question: What is Leviathan?\nAnswer: Leviathan is an experimental CPU inference engine for ternary and recurrent language models.",
        "USER: What is Leviathan?\nASSISTANT: Leviathan is an experimental CPU inference engine for ternary and recurrent language models.",
        "Question: What is Leviathan-1.58bit-Engine?\nAnswer: Leviathan-1.58bit-Engine is a research prototype for running native ternary model packages on a local CPU runtime.",
        "USER: What is Leviathan-1.58bit-Engine?\nASSISTANT: Leviathan-1.58bit-Engine is a research prototype for running native ternary model packages on a local CPU runtime.",
        "Question: What is MLGRU?\nAnswer: MLGRU is a recurrent model path that updates hidden state step by step instead of using attention maps.",
        "USER: What is MLGRU?\nASSISTANT: MLGRU is a recurrent model path that updates hidden state step by step instead of using attention maps.",
        "Question: What is Top-K activation sparsity?\nAnswer: Top-K activation sparsity keeps the largest activations and skips smaller activations to reduce computation.",
        "USER: What is Top-K activation sparsity?\nASSISTANT: Top-K activation sparsity keeps the largest activations and skips smaller activations to reduce computation.",
        "Question: What does dense mode mean?\nAnswer: Dense mode uses all quantized activations instead of selecting a sparse Top-K subset.",
        "USER: What does dense mode mean?\nASSISTANT: Dense mode uses all quantized activations instead of selecting a sparse Top-K subset.",
        "Question: What does --top-k 0 mean?\nAnswer: It means dense mode. Leviathan uses all activations instead of sparse Top-K activation selection.",
        "USER: What does --top-k 0 mean?\nASSISTANT: It means dense mode. Leviathan uses all activations instead of sparse Top-K activation selection.",
        "Question: What does --top-k 0.9 mean?\nAnswer: It keeps about 90 percent of each bitlinear input per layer as a ratio-based Top-K setting.",
        "USER: What does --top-k 0.9 mean?\nASSISTANT: It keeps about 90 percent of each bitlinear input per layer as a ratio-based Top-K setting.",
        "Question: What does --top-k 0.8 mean?\nAnswer: It keeps about 80 percent of each bitlinear input per layer and skips the smaller activations.",
        "USER: What does --top-k 0.8 mean?\nASSISTANT: It keeps about 80 percent of each bitlinear input per layer and skips the smaller activations.",
        "Question: Why is ratio Top-K used?\nAnswer: Ratio Top-K scales K with each layer width, so wide layers are not damaged by a fixed absolute K.",
        "USER: Why is ratio Top-K used?\nASSISTANT: Ratio Top-K scales K with each layer width, so wide layers are not damaged by a fixed absolute K.",
        "Question: Why was fixed K risky?\nAnswer: Fixed K can keep enough activations in narrow layers but too few activations in wide layers, causing generation to collapse.",
        "USER: Why was fixed K risky?\nASSISTANT: Fixed K can keep enough activations in narrow layers but too few activations in wide layers, causing generation to collapse.",
        "Question: Why is Top-K not faster at 30M scale?\nAnswer: At 30M scale, Top-K selection and sparse memory overhead can be larger than the dense ternary computation.",
        "USER: Why is Top-K not faster at 30M scale?\nASSISTANT: At 30M scale, Top-K selection and sparse memory overhead can be larger than the dense ternary computation.",
        "Question: Is this model a general assistant?\nAnswer: No. This is a small proof model for testing the Leviathan training, export, and CPU runtime path.",
        "USER: Is this model a general assistant?\nASSISTANT: No. This is a small proof model for testing the Leviathan training, export, and CPU runtime path.",
        "Question: What dataset was the first proof model trained on?\nAnswer: The first proof model was trained on TinyStories.",
        "USER: What dataset was the first proof model trained on?\nASSISTANT: The first proof model was trained on TinyStories.",
        "Question: What is the goal of the v0.2 model?\nAnswer: The goal is to mix TinyStories with short Question and Answer examples so the model can follow simple QA format.",
        "USER: What is the goal of the v0.2 model?\nASSISTANT: The goal is to mix TinyStories with short Question and Answer examples so the model can follow simple QA format.",
        "Question: What files are inside a Leviathan MLGRU model package?\nAnswer: The package contains a .bin file, a metadata JSON file, a tokenizer folder, a report, and sample outputs.",
        "USER: What files are inside a Leviathan MLGRU model package?\nASSISTANT: The package contains a .bin file, a metadata JSON file, a tokenizer folder, a report, and sample outputs.",
        "Question: What is stored in the .bin file?\nAnswer: The .bin file stores packed ternary linear weights and FP16 structural tensors for the Leviathan runtime.",
        "USER: What is stored in the .bin file?\nASSISTANT: The .bin file stores packed ternary linear weights and FP16 structural tensors for the Leviathan runtime.",
        "Question: What is stored in the meta JSON file?\nAnswer: The meta JSON file describes tensor names, shapes, offsets, packing format, model configuration, and tokenizer location.",
        "USER: What is stored in the meta JSON file?\nASSISTANT: The meta JSON file describes tensor names, shapes, offsets, packing format, model configuration, and tokenizer location.",
        "Question: What is ternary weight?\nAnswer: A ternary weight is a weight represented with three values: negative one, zero, or positive one.",
        "USER: What is ternary weight?\nASSISTANT: A ternary weight is a weight represented with three values: negative one, zero, or positive one.",
        "Question: What is 1.58-bit modeling?\nAnswer: It refers to ternary weight representation, where three states carry about log2 of three bits of information.",
        "USER: What is 1.58-bit modeling?\nASSISTANT: It refers to ternary weight representation, where three states carry about log2 of three bits of information.",
        "Question: What is fake ternary training?\nAnswer: Fake ternary training simulates ternary weights during training while still updating floating point parameters.",
        "USER: What is fake ternary training?\nASSISTANT: Fake ternary training simulates ternary weights during training while still updating floating point parameters.",
        "Question: What is fake int8 activation training?\nAnswer: Fake int8 activation training simulates the quantized activations that the runtime will use during inference.",
        "USER: What is fake int8 activation training?\nASSISTANT: Fake int8 activation training simulates the quantized activations that the runtime will use during inference.",
        "Question: Why train with fake quantization?\nAnswer: Fake quantization helps the model adapt to the low precision behavior that the engine will use after export.",
        "USER: Why train with fake quantization?\nASSISTANT: Fake quantization helps the model adapt to the low precision behavior that the engine will use after export.",
        "Question: What does Leviathan v2 export mean?\nAnswer: It means the model is exported into the Leviathan binary and metadata format that engine.py can load.",
        "USER: What does Leviathan v2 export mean?\nASSISTANT: It means the model is exported into the Leviathan binary and metadata format that engine.py can load.",
        "Question: How do I run the MLGRU proof model?\nAnswer: Run engine.py with the model .bin file, the meta JSON file, --architecture mlgru, and a prompt template.",
        "USER: How do I run the MLGRU proof model?\nASSISTANT: Run engine.py with the model .bin file, the meta JSON file, --architecture mlgru, and a prompt template.",
        "Question: What command runs dense MLGRU mode?\nAnswer: Use python engine.py with --architecture mlgru and --top-k 0.",
        "USER: What command runs dense MLGRU mode?\nASSISTANT: Use python engine.py with --architecture mlgru and --top-k 0.",
        "Question: What command runs sparse MLGRU mode?\nAnswer: Use python engine.py with --architecture mlgru and a ratio such as --top-k 0.9.",
        "USER: What command runs sparse MLGRU mode?\nASSISTANT: Use python engine.py with --architecture mlgru and a ratio such as --top-k 0.9.",
        "Question: What prompt should I test first?\nAnswer: A simple first prompt is Once upon a time, because the proof model was trained on TinyStories.",
        "USER: What prompt should I test first?\nASSISTANT: A simple first prompt is Once upon a time, because the proof model was trained on TinyStories.",
        "Question: Why does the model produce story-like text?\nAnswer: It produces story-like text because TinyStories is the main training dataset.",
        "USER: Why does the model produce story-like text?\nASSISTANT: It produces story-like text because TinyStories is the main training dataset.",
        "Question: Can this model answer broad factual questions?\nAnswer: Not reliably. It is a small proof model and not a broad knowledge assistant.",
        "USER: Can this model answer broad factual questions?\nASSISTANT: Not reliably. It is a small proof model and not a broad knowledge assistant.",
        "Question: What is the current best mode for the 30M model?\nAnswer: Dense MLGRU mode is currently the best mode for speed and coherence at 30M scale.",
        "USER: What is the current best mode for the 30M model?\nASSISTANT: Dense MLGRU mode is currently the best mode for speed and coherence at 30M scale.",
        "Question: What is the purpose of BENCHMARK.md?\nAnswer: BENCHMARK.md records measured speed, sample outputs, limitations, and the next optimization targets.",
        "USER: What is the purpose of BENCHMARK.md?\nASSISTANT: BENCHMARK.md records measured speed, sample outputs, limitations, and the next optimization targets.",
        "Question: What is the purpose of the training script?\nAnswer: The training script trains a small MLGRU model on Modal and exports a Leviathan-compatible package.",
        "USER: What is the purpose of the training script?\nASSISTANT: The training script trains a small MLGRU model on Modal and exports a Leviathan-compatible package.",
        "Question: What GPU is used for the v0.2 training script?\nAnswer: The v0.2 training script is designed to use a Modal T4 GPU.",
        "USER: What GPU is used for the v0.2 training script?\nASSISTANT: The v0.2 training script is designed to use a Modal T4 GPU.",
        "Question: Why use T4 instead of A100 for this proof model?\nAnswer: The recurrent training loop did not benefit enough from A100, so T4 was more cost efficient in testing.",
        "USER: Why use T4 instead of A100 for this proof model?\nASSISTANT: The recurrent training loop did not benefit enough from A100, so T4 was more cost efficient in testing.",
        "Question: What is the next scaling target?\nAnswer: The next scaling target is a larger 70M to 100M model after the 30M proof route is stable.",
        "USER: What is the next scaling target?\nASSISTANT: The next scaling target is a larger 70M to 100M model after the 30M proof route is stable.",
        "Question: What is the sparse kernel target?\nAnswer: The sparse kernel target is to reduce selection overhead and make sparse activation execution faster on larger models.",
        "USER: What is the sparse kernel target?\nASSISTANT: The sparse kernel target is to reduce selection overhead and make sparse activation execution faster on larger models.",
        "Question: What is threshold sparsity?\nAnswer: Threshold sparsity keeps activations above a threshold instead of performing an exact Top-K selection.",
        "USER: What is threshold sparsity?\nASSISTANT: Threshold sparsity keeps activations above a threshold instead of performing an exact Top-K selection.",
        "Question: Why might threshold sparsity be useful?\nAnswer: It may reduce sorting and selection overhead compared with exact Top-K.",
        "USER: Why might threshold sparsity be useful?\nASSISTANT: It may reduce sorting and selection overhead compared with exact Top-K.",
        "Question: What is the immediate project goal?\nAnswer: The immediate goal is a reproducible route from training to ternary CPU inference, then responsible scaling.",
        "USER: What is the immediate project goal?\nASSISTANT: The immediate goal is a reproducible route from training to ternary CPU inference, then responsible scaling.",
        "Question: What should the project not claim yet?\nAnswer: It should not claim major Top-K speedups at 30M scale or general assistant quality.",
        "USER: What should the project not claim yet?\nASSISTANT: It should not claim major Top-K speedups at 30M scale or general assistant quality.",
        "Question: What did v0.1 prove?\nAnswer: It proved that a 30M MLGRU model can be trained, exported to Leviathan v2, and run locally through engine.py.",
        "USER: What did v0.1 prove?\nASSISTANT: It proved that a 30M MLGRU model can be trained, exported to Leviathan v2, and run locally through engine.py.",
        "Question: What should v0.2 prove?\nAnswer: It should prove that the same route can learn simple QA formatting while preserving story continuation behavior.",
        "USER: What should v0.2 prove?\nASSISTANT: It should prove that the same route can learn simple QA formatting while preserving story continuation behavior.",
        "Question: What is a model card?\nAnswer: A model card is a README for a model repository that explains the model, files, usage, limitations, and benchmark results.",
        "USER: What is a model card?\nASSISTANT: A model card is a README for a model repository that explains the model, files, usage, limitations, and benchmark results.",
        "Question: Where should model files be uploaded?\nAnswer: Model files should be uploaded to Hugging Face or a release asset, not committed directly to the GitHub main branch.",
        "USER: Where should model files be uploaded?\nASSISTANT: Model files should be uploaded to Hugging Face or a release asset, not committed directly to the GitHub main branch.",
        "Question: Where should source code be uploaded?\nAnswer: Source code, README files, benchmark files, and training scripts belong in the GitHub repository.",
        "USER: Where should source code be uploaded?\nASSISTANT: Source code, README files, benchmark files, and training scripts belong in the GitHub repository.",
        "Question: Why keep model binaries out of GitHub history?\nAnswer: Large binary files make the repository harder to clone, review, and maintain.",
        "USER: Why keep model binaries out of GitHub history?\nASSISTANT: Large binary files make the repository harder to clone, review, and maintain.",
        "Question: What does CPU-resident inference mean?\nAnswer: It means the model weights are loaded and executed on the local CPU runtime instead of depending on a GPU.",
        "USER: What does CPU-resident inference mean?\nASSISTANT: It means the model weights are loaded and executed on the local CPU runtime instead of depending on a GPU.",
        "Question: What does SIMD bitlinear execution mean?\nAnswer: It means the runtime uses CPU vector instructions to accelerate low precision linear projection work.",
        "USER: What does SIMD bitlinear execution mean?\nASSISTANT: It means the runtime uses CPU vector instructions to accelerate low precision linear projection work.",
        "Question: What is a recurrent runtime path?\nAnswer: A recurrent runtime path updates state step by step and avoids computing a full attention score matrix.",
        "USER: What is a recurrent runtime path?\nASSISTANT: A recurrent runtime path updates state step by step and avoids computing a full attention score matrix.",
        "Question: Why avoid attention maps?\nAnswer: Avoiding attention maps can reduce memory and compute patterns for compatible recurrent models.",
        "USER: Why avoid attention maps?\nASSISTANT: Avoiding attention maps can reduce memory and compute patterns for compatible recurrent models.",
        "Question: Can transformer checkpoints run in MLGRU mode?\nAnswer: Existing transformer checkpoints are not expected to work correctly in MLGRU mode without compatible training.",
        "USER: Can transformer checkpoints run in MLGRU mode?\nASSISTANT: Existing transformer checkpoints are not expected to work correctly in MLGRU mode without compatible training.",
        "Question: Why does MLGRU need its own checkpoint?\nAnswer: MLGRU has a different recurrent computation path, so the weights need to be trained for that path.",
        "USER: Why does MLGRU need its own checkpoint?\nASSISTANT: MLGRU has a different recurrent computation path, so the weights need to be trained for that path.",
        "Question: What is the safest first test after training?\nAnswer: Run dense mode first with --top-k 0, then compare ratio Top-K settings later.",
        "USER: What is the safest first test after training?\nASSISTANT: Run dense mode first with --top-k 0, then compare ratio Top-K settings later.",
        "Question: What is the first sign that v0.2 worked?\nAnswer: The first sign is that the model follows Question and Answer formatting for simple project-related prompts.",
        "USER: What is the first sign that v0.2 worked?\nASSISTANT: The first sign is that the model follows Question and Answer formatting for simple project-related prompts.",
        "Question: What should I do if output is random?\nAnswer: Check dense mode first, verify the correct .bin and meta files, and confirm the tokenizer folder is present.",
        "USER: What should I do if output is random?\nASSISTANT: Check dense mode first, verify the correct .bin and meta files, and confirm the tokenizer folder is present.",
        "Question: What should I do if Top-K output collapses?\nAnswer: Use dense mode as the baseline and test less aggressive ratios such as 0.9 before trying lower ratios.",
        "USER: What should I do if Top-K output collapses?\nASSISTANT: Use dense mode as the baseline and test less aggressive ratios such as 0.9 before trying lower ratios.",
        "Question: What should I do if Modal training is too slow?\nAnswer: Run shorter 1000 step experiments and compare results before spending time on longer runs.",
        "USER: What should I do if Modal training is too slow?\nASSISTANT: Run shorter 1000 step experiments and compare results before spending time on longer runs.",
        "Question: What should I do before publishing a benchmark?\nAnswer: Record the CPU, OS, Python version, PyTorch version, compiler, engine commit, command, prompt, and sample output.",
        "USER: What should I do before publishing a benchmark?\nASSISTANT: Record the CPU, OS, Python version, PyTorch version, compiler, engine commit, command, prompt, and sample output.",
        "Question: What is the honest conclusion for 30M Top-K?\nAnswer: The honest conclusion is that ratio Top-K preserves readable text but is not yet a speed win at 30M scale.",
        "USER: What is the honest conclusion for 30M Top-K?\nASSISTANT: The honest conclusion is that ratio Top-K preserves readable text but is not yet a speed win at 30M scale.",
        "Question: What is the honest conclusion for 30M MLGRU?\nAnswer: The honest conclusion is that the MLGRU training, export, and CPU runtime path works as a proof model.",
        "USER: What is the honest conclusion for 30M MLGRU?\nASSISTANT: The honest conclusion is that the MLGRU training, export, and CPU runtime path works as a proof model.",
        "Question: What is the next GitHub milestone?\nAnswer: The next milestone is an instruction-mix v0.2 model and repeatable benchmark automation.",
        "USER: What is the next GitHub milestone?\nASSISTANT: The next milestone is an instruction-mix v0.2 model and repeatable benchmark automation.",
    ]

    def split_qa_text(text: str) -> tuple[str, str] | None:
        text = "\n".join(part.strip() for part in text.splitlines() if part.strip())
        if "\nAnswer:" in text:
            q, a = text.split("\nAnswer:", 1)
            return q.strip() + "\nAnswer:", a.strip()
        if "\nASSISTANT:" in text:
            q, a = text.split("\nASSISTANT:", 1)
            return q.strip() + "\nASSISTANT:", a.strip()
        return None

    def load_instruction_pairs(path: str) -> list[dict[str, str]]:
        pairs: list[dict[str, str]] = []
        p = Path(path) if path else None
        if p is not None and p.exists():
            print(f"[qa] loading supervised QA seed file: {p}")
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prompt = row.get("prompt")
                    answer = row.get("answer")
                    if isinstance(prompt, str) and isinstance(answer, str) and prompt.strip() and answer.strip():
                        pairs.append({"prompt": prompt.strip(), "answer": answer.strip()})
                        continue
                    text = row.get("text")
                    if isinstance(text, str):
                        split = split_qa_text(text)
                        if split is not None:
                            pairs.append({"prompt": split[0], "answer": split[1]})
        if not pairs:
            print("[qa] supervised seed file missing or empty; using embedded QA fallback")
            for text in EMBEDDED_QA_TEXTS:
                split = split_qa_text(text)
                if split is not None:
                    pairs.append({"prompt": split[0], "answer": split[1]})

        # Deduplicate while preserving order.
        seen: set[tuple[str, str]] = set()
        clean: list[dict[str, str]] = []
        for pair in pairs:
            prompt = "\n".join(part.strip() for part in pair["prompt"].splitlines() if part.strip())
            answer = " ".join(part.strip() for part in pair["answer"].splitlines() if part.strip())
            key = (prompt, answer)
            if prompt and answer and key not in seen:
                seen.add(key)
                clean.append({"prompt": prompt, "answer": answer})
        print(f"[qa] supervised examples: {len(clean)} qa_ratio={qa_ratio}")
        return clean

    qa_ratio = max(0.0, min(1.0, float(qa_ratio)))
    instruction_pairs = load_instruction_pairs(qa_seed_path)
    instruction_texts = [p["prompt"] + " " + p["answer"] for p in instruction_pairs]

    # ---------------------------------------------------------------------
    # 1) Train or load a small local tokenizer.
    # ---------------------------------------------------------------------
    if not (tokenizer_dir / "tokenizer.json").exists():
        print(f"[tokenizer] training BPE tokenizer vocab_size={vocab_size} from {dataset}...")
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        )
        texts = []
        for i, text in enumerate(dataset_iter(dataset)):
            texts.append(text)
            if i + 1 >= tokenizer_docs:
                break
        # Make sure the tokenizer sees the Question/Answer style before training.
        texts.extend(instruction_texts)
        tok.train_from_iterator(texts, trainer=trainer)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
        tokenizer.save_pretrained(str(tokenizer_dir))
    else:
        print("[tokenizer] using existing tokenizer in volume")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))

    vocab_size_actual = len(tokenizer)
    print("[tokenizer] vocab size:", vocab_size_actual)

    # ---------------------------------------------------------------------
    # 2) Build training data.
    #    Base TinyStories uses random token windows.
    #    QA examples use supervised prompt->answer sequences, with prompt tokens
    #    masked out of the loss. This is the key v0.2c calibration change: the model is
    #    trained to predict the answer for the selected question instead of
    #    learning from arbitrary snippets of a concatenated QA token stream.
    # ---------------------------------------------------------------------
    def encode_stream_to_cache(cache_path: Path, text_source, token_budget: int, label: str) -> np.ndarray:
        if cache_path.exists():
            print(f"[data] loading cached {label} token buffer:", cache_path)
            return np.load(cache_path).astype(np.int64)
        print(f"[data] streaming/tokenizing up to {token_budget:,} {label} tokens...")
        ids: list[int] = []
        eos = int(tokenizer.eos_token_id)
        for text in text_source:
            enc = tokenizer.encode(text, add_special_tokens=False)
            if not enc:
                continue
            ids.extend(enc)
            ids.append(eos)
            if len(ids) >= token_budget:
                break
        arr = np.array(ids[:token_budget], dtype=np.int64)
        np.save(cache_path, arr)
        return arr

    base_cache = run_dir / f"tokens_{dataset}_base_{vocab_size_actual}_{max_train_tokens}.npy"
    base_tokens = encode_stream_to_cache(base_cache, dataset_iter(dataset), max_train_tokens, dataset)
    base_token_tensor = torch.tensor(base_tokens, dtype=torch.long)

    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    eos_id = int(tokenizer.eos_token_id)

    def sample_from_base_buffer(buf: torch.Tensor, n: int):
        if n <= 0:
            return None, None
        if len(buf) <= seq_len + 2:
            raise ValueError(f"token buffer too short for seq_len={seq_len}: {len(buf)}")
        ix = torch.randint(0, len(buf) - seq_len - 1, (n,))
        x = torch.stack([buf[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([buf[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y

    def encode_supervised_pair(pair: dict[str, str]) -> tuple[list[int], list[int]]:
        prompt = pair["prompt"].strip()
        answer = pair["answer"].strip()
        # Keep a single space after Answer:/ASSISTANT: so generation starts naturally.
        full_text = prompt + " " + answer
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False) + [eos_id]
        if len(full_ids) < 2:
            full_ids = [eos_id, eos_id]

        x_ids = full_ids[:-1][:seq_len]
        y_ids = full_ids[1:][:seq_len]

        # Mask prompt-only targets. Position t predicts full_ids[t+1].
        # The first answer token is predicted at t = len(prompt_ids)-1, so keep
        # labels where target_index >= len(prompt_ids).
        labels: list[int] = []
        for pos, target_id in enumerate(y_ids):
            target_index = pos + 1
            labels.append(target_id if target_index >= len(prompt_ids) else -100)

        if len(x_ids) < seq_len:
            pad = seq_len - len(x_ids)
            x_ids.extend([pad_id] * pad)
            labels.extend([-100] * pad)
        return x_ids, labels

    supervised_cache: list[tuple[list[int], list[int]]] = [encode_supervised_pair(p) for p in instruction_pairs]
    supervised_cache = [(x, y) for x, y in supervised_cache if any(v != -100 for v in y)]
    if not supervised_cache:
        raise ValueError("No usable supervised QA examples were loaded")
    print("[data] base tokens:", len(base_tokens))
    print("[data] supervised QA sequences:", len(supervised_cache))

    def sample_supervised_qa(n: int):
        if n <= 0:
            return None, None
        xs, ys = [], []
        for _ in range(n):
            x_ids, y_ids = random.choice(supervised_cache)
            xs.append(torch.tensor(x_ids, dtype=torch.long))
            ys.append(torch.tensor(y_ids, dtype=torch.long))
        return torch.stack(xs).to(device), torch.stack(ys).to(device)

    def get_batch():
        n_qa = int(round(batch_size * qa_ratio))
        n_qa = max(0, min(batch_size, n_qa))
        n_base = batch_size - n_qa
        parts_x, parts_y = [], []
        xb, yb = sample_from_base_buffer(base_token_tensor, n_base)
        if xb is not None:
            parts_x.append(xb)
            parts_y.append(yb)
        xq, yq = sample_supervised_qa(n_qa)
        if xq is not None:
            parts_x.append(xq)
            parts_y.append(yq)
        x = torch.cat(parts_x, dim=0)
        y = torch.cat(parts_y, dim=0)
        perm = torch.randperm(x.size(0), device=device)
        return x[perm], y[perm]

    # ---------------------------------------------------------------------
    # 3) Define a Leviathan-compatible MLGRU model.
    #    Tensor names and shapes are exported to match engine.py's loader.
    # ---------------------------------------------------------------------
    def fake_int8_activation(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # Per-token absmax fake quantization, close to engine.py quantize_absmax.
        max_abs = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        scale = 127.0 / max_abs
        q = torch.round(x * scale).clamp(-128, 127)
        dq = q / scale
        return x + (dq - x).detach()

    class TernaryLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

        def effective_weight(self) -> torch.Tensor:
            # Global gamma matches Leviathan metadata: one gamma per tensor.
            gamma = self.weight.detach().abs().mean().clamp_min(1e-6)
            q = torch.round(self.weight / gamma).clamp(-1, 1)
            ternary = q * gamma
            return self.weight + (ternary - self.weight).detach()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Original proof route: train while simulating engine-side int8 activations
            # and ternary weights. This is slower than BF16-only training, but safer
            # for direct Leviathan export.
            x = fake_int8_activation(x)
            return F.linear(x, self.effective_weight())

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

    class MLGRUBlock(nn.Module):
        def __init__(self, hidden: int, inter: int):
            super().__init__()
            self.input_layernorm = RMSNorm(hidden)
            self.post_attention_layernorm = RMSNorm(hidden)
            self.attn_sub_norm = RMSNorm(hidden)
            self.ffn_sub_norm = RMSNorm(inter)
            self.q_proj = TernaryLinear(hidden, hidden)
            self.k_proj = TernaryLinear(hidden, hidden)
            self.v_proj = TernaryLinear(hidden, hidden)
            self.o_proj = TernaryLinear(hidden, hidden)
            self.gate_proj = TernaryLinear(hidden, inter)
            self.up_proj = TernaryLinear(hidden, inter)
            self.down_proj = TernaryLinear(inter, hidden)

        def forward_step(self, h: torch.Tensor, recurrent: torch.Tensor):
            # Mirrors engine.py MLGRU path: q/k/v -> GRU-like recurrent state -> o_proj -> FFN.
            x = self.input_layernorm(h)
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            reset = torch.sigmoid(q)
            update = torch.sigmoid(k)
            candidate = torch.tanh(v) * reset
            recurrent = update * recurrent + (1.0 - update) * candidate
            attn_out = recurrent

            attn_normed = self.attn_sub_norm(attn_out)
            h = h + self.o_proj(attn_normed)

            ffn_x = self.post_attention_layernorm(h)
            gate = self.gate_proj(ffn_x)
            up = self.up_proj(ffn_x)
            relu = torch.relu(gate)
            ffn_act = relu * relu * up
            ffn_normed = self.ffn_sub_norm(ffn_act)
            h = h + self.down_proj(ffn_normed)
            return h, recurrent

    class LeviathanMLGRULM(nn.Module):
        def __init__(self, vocab: int, hidden: int, layers: int, inter: int):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([MLGRUBlock(hidden, inter) for _ in range(layers)])
            self.norm = RMSNorm(hidden)

        def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
            B, T = input_ids.shape
            states = [torch.zeros(B, hidden_size, device=input_ids.device, dtype=torch.float32) for _ in self.layers]
            logits_out = []
            for t in range(T):
                h = self.embed_tokens(input_ids[:, t])
                for li, block in enumerate(self.layers):
                    h, states[li] = block.forward_step(h, states[li])
                h = self.norm(h)
                logits = F.linear(h, self.embed_tokens.weight)
                logits_out.append(logits)
            logits = torch.stack(logits_out, dim=1)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            return logits, loss

        @torch.no_grad()
        def generate(self, prompt: str, max_new: int = 120):
            self.eval()
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not ids:
                ids = [int(tokenizer.bos_token_id)]
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            states = [torch.zeros(1, hidden_size, device=device, dtype=torch.float32) for _ in self.layers]
            generated = ids[:]
            # Prime recurrent state on the prompt.
            for t in range(input_ids.shape[1]):
                h = self.embed_tokens(input_ids[:, t])
                for li, block in enumerate(self.layers):
                    h, states[li] = block.forward_step(h, states[li])
            last_id = torch.tensor([generated[-1]], dtype=torch.long, device=device)
            for _ in range(max_new):
                h = self.embed_tokens(last_id)
                for li, block in enumerate(self.layers):
                    h, states[li] = block.forward_step(h, states[li])
                h = self.norm(h)
                logits = F.linear(h, self.embed_tokens.weight)
                next_id = int(torch.argmax(logits, dim=-1).item())
                generated.append(next_id)
                last_id = torch.tensor([next_id], dtype=torch.long, device=device)
                if next_id == tokenizer.eos_token_id:
                    break
            return tokenizer.decode(generated, skip_special_tokens=True)

    model = LeviathanMLGRULM(vocab_size_actual, hidden_size, n_layers, intermediate_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[model] params={param_count:,} hidden={hidden_size} layers={n_layers} inter={intermediate_size}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    start = time.time()
    losses: list[float] = []
    model.train()
    for step in range(1, steps + 1):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses.append(float(loss.item()))

        if step == 1 or step % 25 == 0 or step == steps:
            elapsed = time.time() - start
            trained_tokens = step * batch_size * seq_len
            tok_per_sec = trained_tokens / max(elapsed, 1e-6)
            print(
                f"step {step:5d}/{steps} "
                f"loss={loss.item():.4f} "
                f"elapsed={elapsed/60:.1f}m "
                f"train_tok/s={tok_per_sec:.0f}"
            )

        if step % 500 == 0:
            ckpt_path = ckpt_dir / f"step_{step}.pt"
            torch.save({"model": model.state_dict(), "config": {
                "vocab_size": vocab_size_actual,
                "hidden_size": hidden_size,
                "n_layers": n_layers,
                "intermediate_size": intermediate_size,
                "seq_len": seq_len,
                "param_count": param_count,
            }}, ckpt_path)
            volume.commit()
            print("[checkpoint] saved", ckpt_path)

    model.eval()
    sample_prompts = [
        "Once upon a time",
        "Question: What does Top-K do?\nAnswer:",
        "Question: What files are inside a Leviathan MLGRU model package?\nAnswer:",
        "Question: What is Leviathan?\nAnswer:",
        "Question: What is MLGRU?\nAnswer:",
        "Question: Is this model a general assistant?\nAnswer:",
    ]
    sample_outputs: dict[str, str] = {}
    with torch.no_grad():
        for prompt in sample_prompts:
            sample_outputs[prompt] = model.generate(prompt, max_new=120)
    sample = sample_outputs["Once upon a time"]
    print("[samples]")
    for prompt, output in sample_outputs.items():
        print("--- PROMPT ---")
        print(prompt)
        print("--- OUTPUT ---")
        print(output)

    sample_outputs_path = export_dir / "sample_outputs.txt"
    with open(sample_outputs_path, "w", encoding="utf-8") as sf:
        for prompt, output in sample_outputs.items():
            sf.write("PROMPT:\n")
            sf.write(prompt + "\n\n")
            sf.write("OUTPUT:\n")
            sf.write(output + "\n")
            sf.write("\n" + "=" * 80 + "\n\n")

    final_ckpt = ckpt_dir / "final.pt"
    torch.save({"model": model.state_dict(), "losses": losses}, final_ckpt)

    # ---------------------------------------------------------------------
    # 4) Export to Leviathan v2 format.
    # ---------------------------------------------------------------------
    def map_ternary_to_uint8(w_ternary: torch.Tensor) -> torch.Tensor:
        mapped = torch.zeros_like(w_ternary, dtype=torch.uint8)
        mapped[w_ternary == 1] = 1
        mapped[w_ternary == -1] = 2
        return mapped

    def pack_ternary_to_uint8(w_ternary: torch.Tensor):
        if w_ternary.ndim != 2:
            raise ValueError("Only 2D linear weights can be packed")
        rows, cols = w_ternary.shape
        mapped = map_ternary_to_uint8(w_ternary.cpu().contiguous())
        packed_cols = (cols + 3) // 4
        pad_cols = packed_cols * 4 - cols
        if pad_cols:
            mapped = F.pad(mapped, (0, pad_cols))
        groups = mapped.view(rows, packed_cols, 4)
        packed = ((groups[:, :, 0] << 6) | (groups[:, :, 1] << 4) | (groups[:, :, 2] << 2) | groups[:, :, 3]).contiguous()
        return packed.numpy().tobytes(), [rows, packed_cols]

    def fp16_bytes(t: torch.Tensor) -> bytes:
        return t.detach().cpu().contiguous().to(torch.float16).numpy().tobytes()

    def append_fp16(name: str, t: torch.Tensor, f, meta: dict, offset: int) -> int:
        raw = fp16_bytes(t)
        meta["tensors"][name] = {
            "type": "float16",
            "role": "structural",
            "shape": list(t.shape),
            "offset": offset,
            "size": len(raw),
        }
        f.write(raw)
        return offset + len(raw)

    def append_ternary(name: str, layer: TernaryLinear, f, meta: dict, offset: int) -> int:
        w = layer.weight.detach().cpu().float().contiguous()
        gamma = max(float(w.abs().mean().item()), 1e-8)
        w_ternary = torch.clamp(torch.round(w / gamma), -1, 1).to(torch.int8)
        raw, packed_shape = pack_ternary_to_uint8(w_ternary)
        meta["tensors"][name] = {
            "type": "ternary_2bit_packed",
            "role": "linear_weight",
            "shape": list(w.shape),
            "packed_shape": packed_shape,
            "gamma": gamma,
            "offset": offset,
            "size": len(raw),
        }
        f.write(raw)
        return offset + len(raw)

    bin_name = f"{run_name}.bin"
    meta_name = f"{run_name}_meta.json"
    bin_path = export_dir / bin_name
    meta_path = export_dir / meta_name

    meta = {
        "format": "leviathan-v2",
        # The engine loads AutoTokenizer.from_pretrained(model_name).
        # After unzipping locally, run engine.py from the export folder so this relative path resolves.
        "model_name": "./leviathan_mlgru_tokenizer",
        "model_config": {
            "model_type": "leviathan_mlgru",
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": n_layers,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": hidden_size,
            "hidden_act": "relu2",
            "rope_theta": 10000.0,
            "vocab_size": vocab_size_actual,
            "tie_word_embeddings": True,
            "architecture": "mlgru",
            "trained_for_recurrent_runtime": True,
            "training_stage": "v0.2c-qa-calibration",
            "base_dataset": dataset,
            "qa_ratio": qa_ratio,
        },
        "packing": "row-major ternary 2-bit, four weights per byte",
        "tensors": {},
    }

    offset = 0
    with open(bin_path, "wb") as f:
        offset = append_fp16("model.embed_tokens.weight", model.embed_tokens.weight, f, meta, offset)
        for li, block in enumerate(model.layers):
            prefix = f"model.layers.{li}"
            offset = append_fp16(f"{prefix}.input_layernorm.weight", block.input_layernorm.weight, f, meta, offset)
            offset = append_fp16(f"{prefix}.post_attention_layernorm.weight", block.post_attention_layernorm.weight, f, meta, offset)
            offset = append_fp16(f"{prefix}.attn_sub_norm.weight", block.attn_sub_norm.weight, f, meta, offset)
            offset = append_fp16(f"{prefix}.ffn_sub_norm.weight", block.ffn_sub_norm.weight, f, meta, offset)
            offset = append_ternary(f"{prefix}.q_proj.weight", block.q_proj, f, meta, offset)
            offset = append_ternary(f"{prefix}.k_proj.weight", block.k_proj, f, meta, offset)
            offset = append_ternary(f"{prefix}.v_proj.weight", block.v_proj, f, meta, offset)
            offset = append_ternary(f"{prefix}.o_proj.weight", block.o_proj, f, meta, offset)
            offset = append_ternary(f"{prefix}.gate_proj.weight", block.gate_proj, f, meta, offset)
            offset = append_ternary(f"{prefix}.up_proj.weight", block.up_proj, f, meta, offset)
            offset = append_ternary(f"{prefix}.down_proj.weight", block.down_proj, f, meta, offset)
        offset = append_fp16("model.norm.weight", model.norm.weight, f, meta, offset)

    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    # Save a small report before zipping so the package is self-contained.
    zip_path = Path(VOL_PATH) / "exports" / f"{run_name}.zip"
    report = {
        "run_name": run_name,
        "dataset": dataset,
        "dataset_mix": {
            "base_dataset": dataset,
            "qa_ratio": qa_ratio,
            "instruction_examples": len(instruction_pairs),
            "supervised_answer_loss": True,
        },
        "params": param_count,
        "final_loss": losses[-1] if losses else None,
        "sample": sample,
        "samples": sample_outputs,
        "export_zip": str(zip_path),
        "local_engine_command_dense": f"python engine.py --bin {bin_name} --meta {meta_name} --architecture mlgru --top-k 0 --max-new 120 --prompt-template plain",
        "local_engine_command_topk_09": f"python engine.py --bin {bin_name} --meta {meta_name} --architecture mlgru --top-k 0.9 --max-new 120 --prompt-template plain",
    }
    report_path = export_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)

    # Create a zip with everything needed locally.
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(bin_path, arcname=bin_name)
        z.write(meta_path, arcname=meta_name)
        z.write(report_path, arcname="report.json")
        z.write(sample_outputs_path, arcname="sample_outputs.txt")
        for p in tokenizer_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(Path("leviathan_mlgru_tokenizer") / p.relative_to(tokenizer_dir)))

    volume.commit()
    print("[export] bin:", bin_path)
    print("[export] meta:", meta_path)
    print("[export] zip:", zip_path)
    print("[local command after unzip]")
    print(report["local_engine_command_dense"])
    return report


@app.local_entrypoint()
def main(
    run_name: str = "leviathan_mlgru_30m_instruct_v02c",
    dataset: str = "tinystories",
    vocab_size: int = 8192,
    hidden_size: int = 256,
    n_layers: int = 4,
    intermediate_size: int = 768,
    seq_len: int = 128,
    batch_size: int = 16,
    steps: int = 500,
    lr: float = 3e-4,
    max_train_tokens: int = 1_000_000,
    tokenizer_docs: int = 5000,
    qa_ratio: float = 0.7,
    qa_seed_path: str = "/root/instruction_qa_supervised_v02c.jsonl",
):
    result = train_and_export.remote(
        run_name=run_name,
        dataset=dataset,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        intermediate_size=intermediate_size,
        seq_len=seq_len,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        max_train_tokens=max_train_tokens,
        tokenizer_docs=tokenizer_docs,
        qa_ratio=qa_ratio,
        qa_seed_path=qa_seed_path,
    )
    print(result)
