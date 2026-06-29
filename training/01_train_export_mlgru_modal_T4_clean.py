"""
Train a tiny Leviathan-compatible MLGRU model on Modal, then export it to the
Leviathan v2 binary format that engine.py can run with --architecture mlgru.

This script is intentionally small and beginner-friendly. It is a proof route:
1) train a small recurrent model with fake ternary weights and fake int8 activations,
2) export packed ternary linear weights + FP16 structural tensors,
3) zip the exported files so you can download them and run engine.py locally.
"""

from __future__ import annotations

import modal

APP_NAME = "leviathan-mlgru-train-export"
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


@app.function(
    image=image,
    gpu="T4",
    volumes={VOL_PATH: volume},
    timeout=6 * 60 * 60,
)
def train_and_export(
    run_name: str = "leviathan_mlgru_tiny",
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
    # 2) Build a token buffer. For the first run we keep this intentionally
    #    small. Increase max_train_tokens after the pipeline is working.
    # ---------------------------------------------------------------------
    token_cache = run_dir / f"tokens_{dataset}_{vocab_size_actual}_{max_train_tokens}.npy"
    if token_cache.exists():
        print("[data] loading cached token buffer:", token_cache)
        tokens = np.load(token_cache).astype(np.int64)
    else:
        print(f"[data] streaming and tokenizing up to {max_train_tokens:,} tokens from {dataset}...")
        ids: list[int] = []
        eos = int(tokenizer.eos_token_id)
        for text in dataset_iter(dataset):
            enc = tokenizer.encode(text, add_special_tokens=False)
            ids.extend(enc)
            ids.append(eos)
            if len(ids) >= max_train_tokens:
                break
        tokens = np.array(ids[:max_train_tokens], dtype=np.int64)
        np.save(token_cache, tokens)
    print("[data] tokens:", len(tokens))
    token_tensor = torch.tensor(tokens, dtype=torch.long)

    def get_batch():
        ix = torch.randint(0, len(token_tensor) - seq_len - 1, (batch_size,))
        x = torch.stack([token_tensor[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([token_tensor[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y

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
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
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

    sample = model.generate("Once upon a time", max_new=120)
    print("[sample]")
    print(sample)

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

    # Copy tokenizer dir into export dir if it already isn't there.
    # It is already tokenizer_dir. Create a zip with everything needed locally.
    zip_path = Path(VOL_PATH) / "exports" / f"{run_name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(bin_path, arcname=bin_name)
        z.write(meta_path, arcname=meta_name)
        for p in tokenizer_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(Path("leviathan_mlgru_tokenizer") / p.relative_to(tokenizer_dir)))

    # Save a small report.
    report = {
        "run_name": run_name,
        "dataset": dataset,
        "params": param_count,
        "final_loss": losses[-1] if losses else None,
        "sample": sample,
        "export_zip": str(zip_path),
        "local_engine_command": f"python engine.py --bin {bin_name} --meta {meta_name} --architecture mlgru --top-k 0.9 --max-new 80",
    }
    with open(export_dir / "report.json", "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)

    volume.commit()
    print("[export] bin:", bin_path)
    print("[export] meta:", meta_path)
    print("[export] zip:", zip_path)
    print("[local command after unzip]")
    print(report["local_engine_command"])
    return report


@app.local_entrypoint()
def main(
    run_name: str = "leviathan_mlgru_tiny",
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
    )
    print(result)
