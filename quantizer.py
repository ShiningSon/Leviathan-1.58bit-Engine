# ==============================================================================
# The Leviathan Engine - Out-of-Core Streaming 1.58-bit Compressor
# Author: Architecture Engineering Core
# Description: Streams massive model tensors from disk/Hub sequentially,
#              compressing them to a dense 1.58-bit ternary configuration
#              with a maximum RAM consumption boundary of < 4GB.
# ==============================================================================

import os
import json
import gc
import time
import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download

os.environ.setdefault("TORCH_USE_NINJA", "0")

TARGET_MODEL = "EleutherAI/gpt-neox-20b" 

def pack_ternary_to_uint8(w_ternary: torch.Tensor) -> bytes:
    """Pack four individual ternary values {-1, 0, 1} tightly into a single uint8 byte."""
    w_mapped = torch.zeros_like(w_ternary, dtype=torch.uint8)
    w_mapped[w_ternary == 1] = 1
    w_mapped[w_ternary == -1] = 2 
    
    pad_len = (4 - (len(w_mapped) % 4)) % 4
    if pad_len > 0:
        w_mapped = torch.cat([w_mapped, torch.zeros(pad_len, dtype=torch.uint8)])
        
    w_reshaped = w_mapped.view(-1, 4)
    packed = (w_reshaped[:, 0] << 6) | (w_reshaped[:, 1] << 4) | \
             (w_reshaped[:, 2] << 2) | (w_reshaped[:, 3])
    return packed.numpy().tobytes()

def process_and_append_tensor(tensor_name, tensor, bin_file_handle, metadata, current_offset):
    """Squeezes a single tensor block down to 1.58-bit and streams it directly to storage."""
    if len(tensor.shape) == 2:
        # 2D Weight Matrix: Extract scaling matrix profiles and cast to ternary values
        gamma = tensor.abs().mean().item()
        w_scaled = tensor / (gamma + 1e-8)
        w_ternary = torch.clamp(torch.round(w_scaled), -1, 1).to(torch.int8)
        
        packed_bytes = pack_ternary_to_uint8(w_ternary)
        bin_file_handle.write(packed_bytes)
        byte_size = len(packed_bytes)
        
        metadata["tensors"][tensor_name] = {
            "type": "2bit_packed",
            "shape": list(tensor.shape),
            "gamma": gamma,
            "offset": current_offset,
            "size": byte_size
        }
    else:
        # 1D Vector Arrays (LayerNorms, Biases): Retain FP16 structural fidelity
        raw_bytes = tensor.to(torch.float16).numpy().tobytes()
        bin_file_handle.write(raw_bytes)
        byte_size = len(raw_bytes)
        
        metadata["tensors"][tensor_name] = {
            "type": "float16",
            "shape": list(tensor.shape),
            "offset": current_offset,
            "size": byte_size
        }
        
    return current_offset + byte_size

def main():
    print(f"--- Launching Out-of-Core Streaming Quantizer Pipeline for: {TARGET_MODEL} ---")
    
    # Secure storage file chunks without pulling them into runtime memory
    repo_path = snapshot_download(repo_id=TARGET_MODEL, allow_patterns=["*.safetensors"])
    safetensor_files = [os.path.join(repo_path, f) for f in os.listdir(repo_path) if f.endswith(".safetensors")]
    print(f"[INFO] Successfully resolved {len(safetensor_files)} distinct weight matrix fragments.\n")

    bin_file = "extreme_20b_weights.bin"
    meta_file = "extreme_20b_meta.json"
    
    metadata = {"model_name": TARGET_MODEL, "tensors": {}}
    current_offset = 0
    start_time = time.perf_counter()

    print("[PROCESSING] Memory optimization shield active. Compressing layer sequences...")
    with open(bin_file, "wb") as f_out:
        for st_file in safetensor_files:
            print(f"  -> Dismantling storage chunk: {os.path.basename(st_file)}")
            
            with safe_open(st_file, framework="pt", device="cpu") as f_in:
                for tensor_name in f_in.keys():
                    # Isolate a single layer to RAM bounds
                    tensor = f_in.get_tensor(tensor_name)
                    
                    # Convert and append to structural binary binary block
                    current_offset = process_and_append_tensor(
                        tensor_name, tensor, f_out, metadata, current_offset
                    )
                    
                    # Enforce immediate garbage collection release
                    del tensor
                    gc.collect() 

    # Serialize architectural metadata index map
    with open(meta_file, "w") as mf:
        json.dump(metadata, mf, indent=2)

    elapsed = (time.perf_counter() - start_time)
    final_size_gb = current_offset / (1024**3)
    
    print("\n" + "=" * 60)
    print("🔥 [SUCCESS] Extreme 1.58-bit Engine Ammo Forged! 🔥")
    print(f"> Processing Time     : {elapsed:.2f} seconds")
    print(f"> Binary Package Size : {final_size_gb:.2f} GB")
    print("=" * 60)
    print("[SYSTEM] Streaming build completed safely. Engine core is ready for allocation.")

if __name__ == "__main__":
    main()
