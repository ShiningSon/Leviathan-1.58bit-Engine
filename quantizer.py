"""Out-of-core 1.58-bit compressor for Leviathan.

The engine expects structural tensors such as embeddings, lm_head, norms, and
biases to remain FP16. Only dense projection weights are converted to ternary
{-1, 0, 1} and packed four values per byte.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

os.environ.setdefault("TORCH_USE_NINJA", "0")

DEFAULT_TARGET_MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"
DEFAULT_OUTPUT_STEM = "leviathan_native"

STRUCTURAL_NAME_PARTS = (
    "embed_tokens",
    "tok_embeddings",
    "word_embeddings",
    ".wte",
    "lm_head",
    "embed_out",
    "layernorm",
    "layer_norm",
    "input_norm",
    "post_attention_norm",
    "post_attention_layernorm",
    "norm",
    "bias",
    "rotary_emb",
    "inv_freq",
)


def is_structural_tensor(tensor_name: str, tensor: torch.Tensor) -> bool:
    """Return True for tensors that must keep their FP16 numerical values."""
    lower_name = tensor_name.lower()
    if tensor.ndim != 2:
        return True
    if lower_name.endswith("output.weight"):
        return True
    return any(part in lower_name for part in STRUCTURAL_NAME_PARTS)


def _map_ternary_to_uint8(w_ternary: torch.Tensor) -> torch.Tensor:
    mapped = torch.zeros_like(w_ternary, dtype=torch.uint8)
    mapped[w_ternary == 1] = 1
    mapped[w_ternary == -1] = 2
    return mapped


def pack_ternary_to_uint8(w_ternary: torch.Tensor) -> tuple[bytes, list[int]]:
    """Pack a 2D ternary tensor row-by-row into a compact uint8 stream."""
    if w_ternary.ndim != 2:
        raise ValueError("Leviathan packs only 2D linear weights")

    rows, cols = w_ternary.shape
    mapped = _map_ternary_to_uint8(w_ternary.contiguous())
    packed_cols = (cols + 3) // 4
    pad_cols = packed_cols * 4 - cols
    if pad_cols:
        mapped = torch.nn.functional.pad(mapped, (0, pad_cols))

    groups = mapped.view(rows, packed_cols, 4)
    packed = (
        (groups[:, :, 0] << 6)
        | (groups[:, :, 1] << 4)
        | (groups[:, :, 2] << 2)
        | groups[:, :, 3]
    ).contiguous()
    return packed.numpy().tobytes(), [rows, packed_cols]


def tensor_to_fp16_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().to(torch.float16).numpy().tobytes()


def tensor_to_ternary_bytes(tensor: torch.Tensor) -> tuple[bytes, float, list[int]]:
    tensor = tensor.detach().cpu().contiguous().to(torch.float32)
    gamma = max(float(tensor.abs().mean().item()), 1e-8)
    w_ternary = torch.clamp(torch.round(tensor / gamma), -1, 1).to(torch.int8)
    packed_bytes, packed_shape = pack_ternary_to_uint8(w_ternary)
    return packed_bytes, gamma, packed_shape


def process_and_append_tensor(
    tensor_name: str,
    tensor: torch.Tensor,
    bin_file_handle: Any,
    metadata: dict[str, Any],
    current_offset: int,
) -> int:
    """Compress or preserve one tensor and append it to the binary package."""
    shape = list(tensor.shape)
    if is_structural_tensor(tensor_name, tensor):
        raw_bytes = tensor_to_fp16_bytes(tensor)
        tensor_meta: dict[str, Any] = {
            "type": "float16",
            "role": "structural",
            "shape": shape,
            "offset": current_offset,
            "size": len(raw_bytes),
        }
    else:
        raw_bytes, gamma, packed_shape = tensor_to_ternary_bytes(tensor)
        tensor_meta = {
            "type": "ternary_2bit_packed",
            "role": "linear_weight",
            "shape": shape,
            "packed_shape": packed_shape,
            "gamma": gamma,
            "offset": current_offset,
            "size": len(raw_bytes),
        }

    bin_file_handle.write(raw_bytes)
    metadata["tensors"][tensor_name] = tensor_meta
    return current_offset + len(raw_bytes)


def iter_safetensor_files(repo_path: Path) -> list[Path]:
    return sorted(path for path in repo_path.rglob("*.safetensors") if path.is_file())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forge a Leviathan 1.58-bit binary package.")
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL, help="Hugging Face model id to convert")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_STEM, help="Output filename stem")
    parser.add_argument(
        "--allow-ptq",
        action="store_true",
        help="Allow post-training conversion of non-native BitNet models despite quality collapse risk",
    )
    return parser


def model_config_summary(model_id: str) -> dict[str, Any]:
    try:
        cfg = AutoConfig.from_pretrained(model_id)
    except Exception:
        return {}
    rope_parameters = getattr(cfg, "rope_parameters", None) or {}
    rope_theta = (
        rope_parameters.get("rope_theta", getattr(cfg, "rope_theta", None))
        if isinstance(rope_parameters, dict)
        else getattr(cfg, "rope_theta", None)
    )
    return {
        "model_type": getattr(cfg, "model_type", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "head_dim": getattr(cfg, "head_dim", None),
        "hidden_act": getattr(cfg, "hidden_act", None),
        "rope_theta": rope_theta,
        "vocab_size": getattr(cfg, "vocab_size", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
    }


def main() -> None:
    args = build_parser().parse_args()
    target_model = args.model
    lower_model = target_model.lower()
    if not args.allow_ptq and "bitnet" not in lower_model and "1bit" not in lower_model:
        raise SystemExit(
            "Refusing PTQ by default. Use a native BitNet/QAT checkpoint, or pass --allow-ptq "
            "if you only need a systems test package."
        )

    print(f"--- Launching Leviathan streaming quantizer for: {target_model} ---")
    repo_path = Path(snapshot_download(repo_id=target_model, allow_patterns=["*.safetensors"]))
    safetensor_files = iter_safetensor_files(repo_path)
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found for {target_model}")
    print(f"[INFO] Resolved {len(safetensor_files)} safetensors shard(s).")

    bin_file = f"{args.out}.bin"
    meta_file = f"{args.out}_meta.json"
    metadata: dict[str, Any] = {
        "format": "leviathan-v2",
        "model_name": target_model,
        "model_config": model_config_summary(target_model),
        "packing": "row-major ternary 2-bit, four weights per byte",
        "tensors": {},
    }

    current_offset = 0
    start_time = time.perf_counter()
    packed_count = 0
    fp16_count = 0

    print("[PROCESSING] Streaming tensors with bounded RAM usage...")
    with open(bin_file, "wb") as f_out:
        for st_file in safetensor_files:
            print(f"  -> Processing shard: {st_file.name}")
            with safe_open(st_file, framework="pt", device="cpu") as f_in:
                for tensor_name in f_in.keys():
                    tensor = f_in.get_tensor(tensor_name)
                    before = current_offset
                    current_offset = process_and_append_tensor(
                        tensor_name, tensor, f_out, metadata, current_offset
                    )
                    role = metadata["tensors"][tensor_name]["role"]
                    packed_count += int(role == "linear_weight")
                    fp16_count += int(role == "structural")
                    print(f"     {role:13s} {tensor_name} ({current_offset - before:,} bytes)")
                    del tensor
                    gc.collect()

    with open(meta_file, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    elapsed = time.perf_counter() - start_time
    final_size_gb = current_offset / (1024**3)
    print("\n" + "=" * 60)
    print("[SUCCESS] Leviathan binary package forged")
    print(f"> Packed linear tensors : {packed_count}")
    print(f"> FP16 structural tensors: {fp16_count}")
    print(f"> Processing time       : {elapsed:.2f} seconds")
    print(f"> Binary package size   : {final_size_gb:.2f} GB")
    print(f"> Outputs               : {bin_file}, {meta_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
