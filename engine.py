import os
import json
import time
import argparse
import platform
import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoTokenizer

os.environ.setdefault("TORCH_USE_NINJA", "0")

print("==================================================")
print("[THE LEVIATHAN] Phase 49: True Architecture (정규화 가중치 복원 및 메모리 최적화)")
print("==================================================\n")

# ============================================================
# [1] C++ 엔진룸: 병목 제거 및 LayerNorm 가중치 복원
# ============================================================
cpp_source = r"""
#include <torch/extension.h>
#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

static const float TERNARY_LUT_F[4] = {0.0f, 1.0f, -1.0f, 0.0f}; 

// [진짜 AVX2 SIMD 가속 커널] 
// _mm256 인트린직을 사용하여 한 번에 32개의 파라미터를 동시 연산합니다.
void avx2_bitlinear_intrinsic_fused(const int8_t* x_quant, const uint8_t* packed_w, int in_dim, int out_dim, int32_t* out_accum) {
    int in_packed_bytes = in_dim / 4;
    
    #pragma omp parallel for schedule(dynamic, 8)
    for (int o = 0; o < out_dim; ++o) {
        const uint8_t* w_row = packed_w + (o * in_packed_bytes);
        
        __m256i vec_sum = _mm256_setzero_si256(); // 256비트 누산기 초기화
        int32_t scalar_sum = 0;
        int i = 0;
        
        // AVX2 SIMD 파이프라인: 8바이트(32개의 1.58비트 가중치)씩 한 번에 처리
        for (; i <= in_packed_bytes - 8; i += 8) {
            // 1. 압축된 2비트 가중치 32개를 int8_t 배열로 L1 캐시에 언패킹
            int8_t unpacked_w[32];
            for (int j = 0; j < 8; ++j) {
                uint8_t p = w_row[i + j];
                unpacked_w[j*4 + 0] = (int8_t)TERNARY_LUT_F[(p >> 6) & 0x03];
                unpacked_w[j*4 + 1] = (int8_t)TERNARY_LUT_F[(p >> 4) & 0x03];
                unpacked_w[j*4 + 2] = (int8_t)TERNARY_LUT_F[(p >> 2) & 0x03];
                unpacked_w[j*4 + 3] = (int8_t)TERNARY_LUT_F[p & 0x03];
            }
            
            // 2. AVX2 메모리 로드 (32바이트)
            __m256i vx = _mm256_loadu_si256((const __m256i*)&x_quant[i * 4]);
            __m256i vw = _mm256_loadu_si256((const __m256i*)unpacked_w);
            
            // 3. 3항 곱셈 마법 (vw가 -1이면 vx부호 반전, 0이면 0, 1이면 vx 유지)
            __m256i v_prod = _mm256_sign_epi8(vx, vw);
            
            // 4. 오버플로우 방지를 위한 16비트 확장 및 수평 덧셈(Horizontal Add)
            __m128i v_prod_lo = _mm256_castsi256_si128(v_prod);
            __m128i v_prod_hi = _mm256_extracti128_si256(v_prod, 1);
            
            __m256i v_16_lo = _mm256_cvtepi8_epi16(v_prod_lo);
            __m256i v_16_hi = _mm256_cvtepi8_epi16(v_prod_hi);
            
            __m256i ones = _mm256_set1_epi16(1);
            __m256i sum32_lo = _mm256_madd_epi16(v_16_lo, ones);
            __m256i sum32_hi = _mm256_madd_epi16(v_16_hi, ones);
            
            // 5. 32비트 누산기에 최종 합산
            vec_sum = _mm256_add_epi32(vec_sum, sum32_lo);
            vec_sum = _mm256_add_epi32(vec_sum, sum32_hi);
        }
        
        // 256비트 벡터 레지스터에 쌓인 값을 스칼라로 추출
        int32_t sums[8];
        _mm256_storeu_si256((__m256i*)sums, vec_sum);
        for(int k = 0; k < 8; ++k) scalar_sum += sums[k];
        
        // SIMD로 처리하고 남은 자투리(Tail) 처리
        for (; i < in_packed_bytes; ++i) {
            uint8_t p = w_row[i];
            int base_x = i * 4;
            scalar_sum += x_quant[base_x + 0] * (int8_t)TERNARY_LUT_F[(p >> 6) & 0x03];
            scalar_sum += x_quant[base_x + 1] * (int8_t)TERNARY_LUT_F[(p >> 4) & 0x03];
            scalar_sum += x_quant[base_x + 2] * (int8_t)TERNARY_LUT_F[(p >> 2) & 0x03];
            scalar_sum += x_quant[base_x + 3] * (int8_t)TERNARY_LUT_F[p & 0x03];
        }
        out_accum[o] = scalar_sum;
    }
}

void quantize_absmax(int8_t* out_quant, float& out_scale, const float* in, int dim) {
    float max_abs = 1e-8f;
    for (int i = 0; i < dim; ++i) { float v = std::abs(in[i]); if (v > max_abs) max_abs = v; }
    out_scale = 127.0f / max_abs;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dim; ++i) {
        float q = std::round(in[i] * out_scale);
        out_quant[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, q)));
    }
}

void dequantize(float* out, const int32_t* in, float scale_x, float gamma, int dim) {
    float factor = gamma / scale_x;
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < dim; ++i) out[i] = in[i] * factor;
}

// [핵심 패치 1] 모델이 학습한 FP16 가중치(weight)를 반영하는 정규화 함수
void rms_norm_weighted(float* out, const float* in, const float* weight, int dim, float eps = 1e-5f) {
    float ss = 0.0f;
    for(int i = 0; i < dim; ++i) ss += in[i] * in[i];
    ss = 1.0f / std::sqrt((ss / dim) + eps);
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < dim; ++i) out[i] = in[i] * ss * weight[i];
}

void apply_rope(float* vec, int seq_pos, int dim, int num_heads, float rope_theta = 10000.0f) {
    int head_dim = dim / num_heads;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < num_heads; ++h) {
        float* head_vec = vec + (h * head_dim);
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / std::pow(rope_theta, static_cast<float>(i) / head_dim);
            float val = static_cast<float>(seq_pos) * freq;
            float cos_val = std::cos(val), sin_val = std::sin(val);
            float v0 = head_vec[i], v1 = head_vec[i + 1];
            head_vec[i]   = v0 * cos_val - v1 * sin_val;
            head_vec[i+1] = v0 * sin_val + v1 * cos_val;
        }
    }
}

void attention_forward_omp(float* attn_out, const float* q, float* k_cache, float* v_cache, int current_pos, int num_heads, int head_dim) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    #pragma omp parallel for schedule(dynamic, 4)
    for (int h = 0; h < num_heads; ++h) {
        const float* q_head = q + (h * head_dim);
        float* out_head = attn_out + (h * head_dim);
        std::vector<float> scores(current_pos + 1, 0.0f);
        float max_score = -1e9f;
        for (int t = 0; t <= current_pos; ++t) {
            const float* k_head = k_cache + (t * num_heads * head_dim) + (h * head_dim);
            float score = 0.0f;
            for (int i = 0; i < head_dim; ++i) score += q_head[i] * k_head[i];
            score *= scale;
            scores[t] = score;
            if (score > max_score) max_score = score;
        }
        float sum_exp = 0.0f;
        for (int t = 0; t <= current_pos; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        std::fill(out_head, out_head + head_dim, 0.0f);
        for (int t = 0; t <= current_pos; ++t) {
            float s = scores[t] / sum_exp;
            const float* v_head = v_cache + (t * num_heads * head_dim) + (h * head_dim);
            for (int i = 0; i < head_dim; ++i) out_head[i] += s * v_head[i];
        }
    }
}

void unpack_embedding_fp32(int token_id, const float* embed_w, float* out_hidden, int hidden_dim) {
    const float* row_ptr = embed_w + (token_id * hidden_dim);
    std::copy(row_ptr, row_ptr + hidden_dim, out_hidden);
}

int64_t lm_head_argmax_fp32(const float* hidden, const float* lm_head_w, int vocab_size, int hidden_dim, const std::vector<int64_t>& past_tokens) {
    std::vector<float> logits(vocab_size, 0.0f);
    #pragma omp parallel for schedule(dynamic, 1024)
    for(int v = 0; v < vocab_size; ++v) {
        const float* row_ptr = lm_head_w + (v * hidden_dim);
        float score = 0.0f;
        for(int i = 0; i < hidden_dim; ++i) {
            score += hidden[i] * row_ptr[i];
        }
        logits[v] = score;
    }
    float penalty = 1.2f;
    for(int64_t pt : past_tokens) {
        if(logits[pt] > 0) logits[pt] /= penalty;
        else logits[pt] *= penalty;
    }
    int64_t best_id = 0;
    float best_val = -1e9f;
    for(int v = 0; v < vocab_size; ++v) {
        if(logits[v] > best_val) { best_val = logits[v]; best_id = v; }
    }
    return best_id;
}

std::vector<int64_t> force_evolved_generate_cpp(
    std::vector<int64_t> prompt_tokens, int64_t max_new_tokens, int64_t hidden_dim,
    int64_t num_layers, int64_t num_heads, int64_t vocab_size, int64_t eos_id,
    std::vector<torch::Tensor> layer_weights, std::vector<float> layer_gammas,
    std::vector<torch::Tensor> layer_norms, torch::Tensor final_norm_tensor,
    torch::Tensor embed_tensor, torch::Tensor lm_head_tensor 
) {
    std::vector<int64_t> generated = prompt_tokens;
    
    const float* embed_ptr = embed_tensor.data_ptr<float>();
    const float* lm_head_ptr = lm_head_tensor.data_ptr<float>();
    const float* final_norm_w = final_norm_tensor.data_ptr<float>();
    
    int max_seq_len = prompt_tokens.size() + max_new_tokens;
    int head_dim = hidden_dim / num_heads;
    
    // [핵심 패치 2] 지연 시간(84초)을 박살내는 메모리 사전 할당 (Pre-allocation)
    // 루프 내부에서 발생하던 수만 번의 동적 할당을 외부로 완전히 빼냈습니다.
    int inter_dim = layer_weights[4].numel() * 4 / hidden_dim;
    if (inter_dim <= 0) inter_dim = hidden_dim * 4; 
    
    std::vector<float> k_cache(num_layers * max_seq_len * hidden_dim, 0.0f);
    std::vector<float> v_cache(num_layers * max_seq_len * hidden_dim, 0.0f);
    std::vector<float> hidden_state(hidden_dim, 0.0f), norm_state(hidden_dim, 0.0f);
    std::vector<int8_t> hidden_q(hidden_dim, 0);
    std::vector<int32_t> accum_buf(hidden_dim * 4, 0); 
    std::vector<float> q(hidden_dim, 0.0f), k(hidden_dim, 0.0f), v(hidden_dim, 0.0f), attn_out(hidden_dim, 0.0f);
    std::vector<float> ffn_norm_state(hidden_dim, 0.0f);
    std::vector<float> gate_out(inter_dim, 0.0f), up_out(inter_dim, 0.0f), ffn_act(inter_dim, 0.0f);
    std::vector<int8_t> ffn_act_q(inter_dim, 0);
    
    int total_steps = prompt_tokens.size() + max_new_tokens - 1;
    for (int step = 0; step < total_steps; ++step) {
        int current_pos = step;
        int current_token = (step < prompt_tokens.size()) ? prompt_tokens[step] : generated.back();
        
        unpack_embedding_fp32(current_token, embed_ptr, hidden_state.data(), hidden_dim);
        
        for (int l = 0; l < num_layers; ++l) {
            int w_idx = l * 7;
            if (layer_weights[w_idx+0].numel() == 0) continue; 

            // 레이어별 고유 정규화 가중치 매핑
            const float* attn_norm_w = layer_norms[l * 2].data_ptr<float>();
            const float* ffn_norm_w = layer_norms[l * 2 + 1].data_ptr<float>();

            // 가중치가 반영된 정확한 정규화 수행
            rms_norm_weighted(norm_state.data(), hidden_state.data(), attn_norm_w, hidden_dim);
            
            float scale_x;
            quantize_absmax(hidden_q.data(), scale_x, norm_state.data(), hidden_dim);
            
            avx2_bitlinear_intrinsic_fused(hidden_q.data(), layer_weights[w_idx+0].data_ptr<uint8_t>(), hidden_dim, hidden_dim, accum_buf.data());
            dequantize(q.data(), accum_buf.data(), scale_x, layer_gammas[w_idx+0], hidden_dim);
            
            avx2_bitlinear_intrinsic_fused(hidden_q.data(), layer_weights[w_idx+1].data_ptr<uint8_t>(), hidden_dim, hidden_dim, accum_buf.data());
            dequantize(k.data(), accum_buf.data(), scale_x, layer_gammas[w_idx+1], hidden_dim);
            
            avx2_bitlinear_intrinsic_fused(hidden_q.data(), layer_weights[w_idx+2].data_ptr<uint8_t>(), hidden_dim, hidden_dim, accum_buf.data());
            dequantize(v.data(), accum_buf.data(), scale_x, layer_gammas[w_idx+2], hidden_dim);
            
            apply_rope(q.data(), current_pos, hidden_dim, num_heads);
            apply_rope(k.data(), current_pos, hidden_dim, num_heads);
            
            int cache_offset = (l * max_seq_len * hidden_dim) + (current_pos * hidden_dim);
            std::copy(k.begin(), k.end(), k_cache.begin() + cache_offset);
            std::copy(v.begin(), v.end(), v_cache.begin() + cache_offset);
            
            attention_forward_omp(attn_out.data(), q.data(), k_cache.data() + (l * max_seq_len * hidden_dim), v_cache.data() + (l * max_seq_len * hidden_dim), current_pos, num_heads, head_dim);
            
            float scale_attn;
            quantize_absmax(hidden_q.data(), scale_attn, attn_out.data(), hidden_dim);
            avx2_bitlinear_intrinsic_fused(hidden_q.data(), layer_weights[w_idx+3].data_ptr<uint8_t>(), hidden_dim, hidden_dim, accum_buf.data());
            
            float factor = layer_gammas[w_idx+3] / scale_attn;
            #pragma omp parallel for schedule(static)
            for(int i=0; i<hidden_dim; ++i) hidden_state[i] += accum_buf[i] * factor;
            
            // FFN 정규화 가중치 적용
            rms_norm_weighted(ffn_norm_state.data(), hidden_state.data(), ffn_norm_w, hidden_dim);
            quantize_absmax(hidden_q.data(), scale_x, ffn_norm_state.data(), hidden_dim);
            
            avx2_bitlinear_intrinsic_fused(hidden_q.data(), layer_weights[w_idx+4].data_ptr<uint8_t>(), hidden_dim, inter_dim, accum_buf.data());
            dequantize(gate_out.data(), accum_buf.data(), scale_x, layer_gammas[w_idx+4], inter_dim);
            
            bool has_up_proj = (layer_weights[w_idx+5].numel() > 0);
            if (has_up_proj) {
                avx2_bitlinear_intrinsic_fused(hidden_q.data(), layer_weights[w_idx+5].data_ptr<uint8_t>(), hidden_dim, inter_dim, accum_buf.data());
                dequantize(up_out.data(), accum_buf.data(), scale_x, layer_gammas[w_idx+5], inter_dim);
            }
            
            #pragma omp parallel for schedule(static)
            for(int i=0; i<inter_dim; ++i) {
                float g = gate_out[i];
                if (has_up_proj) {
                    float silu = g / (1.0f + std::exp(-g));
                    ffn_act[i] = silu * up_out[i];
                } else {
                    float cdf = 0.5f * (1.0f + std::tanh(0.79788456f * (g + 0.044715f * g * g * g)));
                    ffn_act[i] = g * cdf;
                }
            }
            
            float scale_ffn;
            quantize_absmax(ffn_act_q.data(), scale_ffn, ffn_act.data(), inter_dim);
            
            avx2_bitlinear_intrinsic_fused(ffn_act_q.data(), layer_weights[w_idx+6].data_ptr<uint8_t>(), inter_dim, hidden_dim, accum_buf.data());
            
            float down_factor = layer_gammas[w_idx+6] / scale_ffn;
            #pragma omp parallel for schedule(static)
            for(int i=0; i<hidden_dim; ++i) hidden_state[i] += accum_buf[i] * down_factor;
        }

        // 최종 출력 전 정규화 가중치 적용
        rms_norm_weighted(norm_state.data(), hidden_state.data(), final_norm_w, hidden_dim);
        
        if (step >= prompt_tokens.size() - 1) {
            int64_t next_token = lm_head_argmax_fp32(norm_state.data(), lm_head_ptr, vocab_size, hidden_dim, generated);
            generated.push_back(next_token);
            if (next_token == eos_id || next_token == 0 || next_token == 2 || next_token == 128001) break;
        }
    }
    
    return generated;
}
"""

if platform.system().lower().startswith("win"):
    extra_cflags = ["/O2", "/arch:AVX2", "/fp:fast", "/openmp"]
else:
    extra_cflags = ["-O3", "-march=native", "-ffast-math", "-fopenmp"]

print("[BUILD] 구조적 결함 제거 및 런타임 최적화 커널 컴파일 중...")
evolved_engine = load_inline(
    name="phase49_master",
    cpp_sources=[cpp_source],
    functions=["force_evolved_generate_cpp"],
    extra_cflags=extra_cflags,
    verbose=False,
)
print("[BUILD] 컴파일 완료!\n")


# ============================================================
# [2] 파이썬 로더
# ============================================================
class RestoredBitNet:
    def __init__(self, bin_path="leviathan_native_10b.bin", meta_path="leviathan_native_10b_meta.json"):
        with open(meta_path, "r") as f: self.meta = json.load(f)
        target_model = self.meta.get("model_name", "1bitLLM/bitnet_b1_58-3B")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model)
        
        embed_keys = [k for k in self.meta["tensors"].keys() if "embed" in k or "wte" in k]
        embed_info = self.meta["tensors"][embed_keys[0]]
        self.vocab_size = embed_info["shape"][0]
        self.hidden_dim = embed_info["shape"][1]
        
        layer_nums = [int(k.split(".")[2]) for k in self.meta["tensors"].keys() if "layers" in k and k.split(".")[2].isdigit()]
        self.num_layers = max(layer_nums) + 1 if layer_nums else 24
        self.num_heads = self.hidden_dim // 128
        if self.num_heads == 0: self.num_heads = 32
            
        file_size = os.path.getsize(bin_path)
        print(f"[SYSTEM] 모델 캐싱 중... ({file_size / (1024**3):.2f}GB 물리 RAM 매핑)")
        self.mmap_tensor = torch.from_file(bin_path, shared=True, size=file_size, dtype=torch.uint8).clone()

        def get_slice_fp16(tensor_name):
            if tensor_name not in self.meta["tensors"]: return None
            info = self.meta["tensors"][tensor_name]
            return self.mmap_tensor[info["offset"] : info["offset"] + info["size"]].view(torch.float16).to(torch.float32)

        def get_slice_uint8(tensor_name):
            if tensor_name not in self.meta["tensors"]: return None
            info = self.meta["tensors"][tensor_name]
            return self.mmap_tensor[info["offset"] : info["offset"] + info["size"]]

        self.embed_tensor = get_slice_fp16(embed_keys[0])
        lm_head_keys = [k for k in self.meta["tensors"].keys() if "lm_head" in k or "output" in k]
        self.lm_head_tensor = get_slice_fp16(lm_head_keys[0]) if lm_head_keys else self.embed_tensor

        def find_norm_weight(l, patterns):
            for p in patterns:
                for k in self.meta["tensors"].keys():
                    if f"layers.{l}." in k and p in k and "weight" in k:
                        return get_slice_fp16(k)
            return torch.ones(self.hidden_dim, dtype=torch.float32)

        self.layer_norms = []
        for l in range(self.num_layers):
            attn_norm = find_norm_weight(l, ["input_layernorm", "ln_1"])
            ffn_norm = find_norm_weight(l, ["post_attention_layernorm", "ln_2"])
            self.layer_norms.extend([attn_norm, ffn_norm])
            
        final_norm_keys = [k for k in self.meta["tensors"].keys() if "model.norm.weight" in k or "ln_f" in k]
        self.final_norm = get_slice_fp16(final_norm_keys[0]) if final_norm_keys else torch.ones(self.hidden_dim, dtype=torch.float32)

        def get_target_tensor(patterns):
            for p in patterns:
                for k in self.meta["tensors"].keys():
                    if f"layers.{l}." in k and p in k and "weight" in k and "norm" not in k:
                        return get_slice_uint8(k), float(self.meta["tensors"][k].get("gamma", 1.0))
            return torch.zeros(0, dtype=torch.uint8), 1.0

        self.layer_tensors, self.layer_gammas = [], []
        for l in range(self.num_layers):
            q_t, q_g = get_target_tensor(["q_proj", "query_key_value"])
            k_t, k_g = get_target_tensor(["k_proj", "query_key_value"])
            v_t, v_g = get_target_tensor(["v_proj", "query_key_value"])
            o_t, o_g = get_target_tensor(["o_proj", "dense", "out"])
            gate_t, gate_g = get_target_tensor(["gate_proj", "dense_h_to_4h", "w1"])
            up_t, up_g = get_target_tensor(["up_proj", "w3"])
            down_t, down_g = get_target_tensor(["down_proj", "dense_4h_to_h", "w2"])
            for t, g in [(q_t, q_g), (k_t, k_g), (v_t, v_g), (o_t, o_g), (gate_t, gate_g), (up_t, up_g), (down_t, down_g)]:
                self.layer_tensors.append(t)
                self.layer_gammas.append(g)
        print(f"[SYSTEM] 아키텍처 및 정규화 가중치 로드 완료 (Layers: {self.num_layers}, Hidden: {self.hidden_dim})")

    def generate(self, prompt_text, max_new_tokens=150):
        formatted_prompt = prompt_text 
        prompt_tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=True)
        if not prompt_tokens: prompt_tokens = [0]
            
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None: eos_id = 2 
        if isinstance(eos_id, list): eos_id = eos_id[0]

        start_time = time.perf_counter()
        
        output_tokens = evolved_engine.force_evolved_generate_cpp(
            prompt_tokens, max_new_tokens, self.hidden_dim, self.num_layers, self.num_heads, self.vocab_size, eos_id,
            self.layer_tensors, self.layer_gammas, self.layer_norms, self.final_norm,
            self.embed_tensor, self.lm_head_tensor
        )
        elapsed = (time.perf_counter() - start_time) * 1000.0
        
        gen_tokens = output_tokens[len(prompt_tokens):]
        output_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        gen_len = max(len(gen_tokens), 1)
        return output_text, elapsed, gen_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", nargs="?", default="chat")
    parser.add_argument("--max-new", type=int, default=150)
    args = parser.parse_args()

    print("=" * 100)
    print("[PHASE 49] THE LEVIATHAN - True Architecture (Final Resolution)")
    print("=" * 100)
    
    engine = RestoredBitNet()
    while True:
        try: prompt = input("\nUSER> ").strip()
        except EOFError: break
        if not prompt or prompt.lower() in {"/exit", "exit", "quit"}: break

        text, elapsed, gen_len = engine.generate(prompt, args.max_new)
        print(f"\nENGINE> {text.strip()}")
        print(f"[Stats: {elapsed:.2f} ms | {gen_len / (max(elapsed / 1000.0, 1e-5)):.2f} tokens/sec]")

if __name__ == "__main__":
    main()
