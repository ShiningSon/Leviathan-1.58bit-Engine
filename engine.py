from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import Iterable

import torch
from torch.utils.cpp_extension import load_inline
from transformers import AutoConfig, AutoTokenizer

os.environ.setdefault("TORCH_USE_NINJA", "0")

try:
    import ninja

    ninja_bin_dir = getattr(ninja, "BIN_DIR", "")
    if ninja_bin_dir and ninja_bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ninja_bin_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

print("==================================================")
print("[THE LEVIATHAN] Phase 50: sparse SIMD + MLGRU runtime")
print("==================================================\n")


cpp_source = r"""
#include <torch/extension.h>

#if defined(LEVIATHAN_AVX2)
#include <immintrin.h>
#endif

#if defined(LEVIATHAN_NEON_DOT)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#include <chrono>
#include <pybind11/pybind11.h>

namespace py = pybind11;

static const int8_t TERNARY_LUT_I8[4] = {0, 1, -1, 0};

struct TopKDeltaState {
    std::vector<int> indices;
    uint64_t hash = 0;
    int delta_count = 0;
};

struct ProfileStats {
    int64_t dense_calls = 0;
    int64_t topk_calls = 0;
    int64_t topk_fallback_calls = 0;
    int64_t topk_select_us = 0;
    int64_t dense_kernel_us = 0;
    int64_t sparse_kernel_us = 0;
    int64_t active_k_sum = 0;
    int64_t input_dim_sum = 0;
};

static bool g_profile_enabled = false;
static ProfileStats g_profile;

static inline int64_t micros_since(std::chrono::high_resolution_clock::time_point start) {
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static inline int packed_cols_for_dim(int dim) {
    return (dim + 3) / 4;
}

static inline int8_t read_packed_weight(const uint8_t* row, int input_idx) {
    const uint8_t packed = row[input_idx >> 2];
    const int shift = 6 - ((input_idx & 3) * 2);
    return TERNARY_LUT_I8[(packed >> shift) & 0x03];
}

void bitlinear_dense_fused(
    const int8_t* x_quant,
    const uint8_t* packed_w,
    int in_dim,
    int out_dim,
    int32_t* out_accum
) {
    const int row_packed_bytes = packed_cols_for_dim(in_dim);

    #pragma omp parallel for schedule(dynamic, 8)
    for (int o = 0; o < out_dim; ++o) {
        const uint8_t* w_row = packed_w + (o * row_packed_bytes);
        int32_t scalar_sum = 0;
        int i = 0;

#if defined(LEVIATHAN_AVX2)
        __m256i vec_sum = _mm256_setzero_si256();
        for (; i <= in_dim - 32; i += 32) {
            int8_t unpacked_w[32];
            const int byte_base = i >> 2;
            for (int j = 0; j < 8; ++j) {
                const uint8_t p = w_row[byte_base + j];
                unpacked_w[j * 4 + 0] = TERNARY_LUT_I8[(p >> 6) & 0x03];
                unpacked_w[j * 4 + 1] = TERNARY_LUT_I8[(p >> 4) & 0x03];
                unpacked_w[j * 4 + 2] = TERNARY_LUT_I8[(p >> 2) & 0x03];
                unpacked_w[j * 4 + 3] = TERNARY_LUT_I8[p & 0x03];
            }

            const __m256i vx = _mm256_loadu_si256((const __m256i*)&x_quant[i]);
            const __m256i vw = _mm256_loadu_si256((const __m256i*)unpacked_w);
            const __m256i v_prod = _mm256_sign_epi8(vx, vw);

            const __m128i v_prod_lo = _mm256_castsi256_si128(v_prod);
            const __m128i v_prod_hi = _mm256_extracti128_si256(v_prod, 1);
            const __m256i v_16_lo = _mm256_cvtepi8_epi16(v_prod_lo);
            const __m256i v_16_hi = _mm256_cvtepi8_epi16(v_prod_hi);
            const __m256i ones = _mm256_set1_epi16(1);
            vec_sum = _mm256_add_epi32(vec_sum, _mm256_madd_epi16(v_16_lo, ones));
            vec_sum = _mm256_add_epi32(vec_sum, _mm256_madd_epi16(v_16_hi, ones));
        }

        alignas(32) int32_t sums[8];
        _mm256_store_si256((__m256i*)sums, vec_sum);
        for (int k = 0; k < 8; ++k) scalar_sum += sums[k];
#endif

        for (; i < in_dim; ++i) {
            scalar_sum += static_cast<int32_t>(x_quant[i]) * read_packed_weight(w_row, i);
        }
        out_accum[o] = scalar_sum;
    }
}

uint64_t hash_indices(const std::vector<int>& indices) {
    uint64_t h = 1469598103934665603ULL;
    for (int idx : indices) {
        h ^= static_cast<uint64_t>(idx + 0x9e3779b9);
        h *= 1099511628211ULL;
    }
    return h;
}

int symmetric_delta_count(const std::vector<int>& a, const std::vector<int>& b) {
    int i = 0;
    int j = 0;
    int delta = 0;
    while (i < static_cast<int>(a.size()) && j < static_cast<int>(b.size())) {
        if (a[i] == b[j]) {
            ++i;
            ++j;
        } else if (a[i] < b[j]) {
            ++delta;
            ++i;
        } else {
            ++delta;
            ++j;
        }
    }
    return delta + (static_cast<int>(a.size()) - i) + (static_cast<int>(b.size()) - j);
}

void select_topk_abs(const int8_t* x_quant, int dim, int top_k, std::vector<int>& out, bool sort_indices) {
    top_k = std::max(0, std::min(top_k, dim));
    out.resize(dim);
    std::iota(out.begin(), out.end(), 0);
    if (top_k < dim) {
        std::nth_element(
            out.begin(),
            out.begin() + top_k,
            out.end(),
            [x_quant](int lhs, int rhs) {
                return std::abs(static_cast<int>(x_quant[lhs])) > std::abs(static_cast<int>(x_quant[rhs]));
            }
        );
        out.resize(top_k);
    }
    if (sort_indices) {
        std::sort(out.begin(), out.end());
    }
}

const std::vector<int>& update_topk_delta_state(
    TopKDeltaState& state,
    const int8_t* x_quant,
    int dim,
    int top_k,
    bool sort_indices
) {
    std::vector<int> next_indices;
    select_topk_abs(x_quant, dim, top_k, next_indices, sort_indices);
    const uint64_t next_hash = hash_indices(next_indices);
    if (state.indices.empty() || next_hash != state.hash) {
        // symmetric_delta_count assumes sorted index vectors. When the experiment
        // disables sorting, delta_count is intentionally not meaningful.
        state.delta_count = sort_indices ? symmetric_delta_count(state.indices, next_indices) : -1;
        state.indices.swap(next_indices);
        state.hash = next_hash;
    } else {
        state.delta_count = 0;
    }
    return state.indices;
}

void bitlinear_topk_sparse(
    const int8_t* x_quant,
    const uint8_t* packed_w,
    int in_dim,
    int out_dim,
    const std::vector<int>& active_indices,
    int32_t* out_accum
) {
    const int row_packed_bytes = packed_cols_for_dim(in_dim);
    // Magnitude-based Top-K drops only the smallest activations, whose signed
    // contributions roughly cancel, so the partial sum already tracks the dense
    // sum. Do NOT rescale by in_dim/K -- that amplifies the sum and corrupts
    // downstream distributions (verified: produces gibberish output).
    #pragma omp parallel for schedule(dynamic, 8)
    for (int o = 0; o < out_dim; ++o) {
        const uint8_t* w_row = packed_w + (o * row_packed_bytes);
        int32_t sum = 0;
        for (int idx : active_indices) {
            sum += static_cast<int32_t>(x_quant[idx]) * read_packed_weight(w_row, idx);
        }
        out_accum[o] = sum;
    }
}


static inline bool projection_allows_sparse(int sparse_scope, int projection_slot) {
    // 0 = all projections
    // 1 = FFN projections only: gate/up/down
    // 2 = down projection only
    // 3 = none
    if (sparse_scope == 0) return true;
    if (sparse_scope == 1) return projection_slot >= 4 && projection_slot <= 6;
    if (sparse_scope == 2) return projection_slot == 6;
    return false;
}

void bitlinear_dispatch(
    const int8_t* x_quant,
    const uint8_t* packed_w,
    int in_dim,
    int out_dim,
    int32_t* out_accum,
    int top_k_param,
    bool top_k_is_ratio,
    float sparse_min_density,
    bool sort_topk,
    bool allow_sparse,
    TopKDeltaState* topk_state
) {
    // Resolve the effective K for THIS layer. When --top-k is given as a ratio
    // (0..100), K scales with in_dim so wide layers (e.g. down_proj in_dim=6912)
    // keep the same fraction of their activations as narrow layers. A flat
    // absolute K punished wide layers disproportionately: K=512 on in_dim=6912
    // kept only 7% of activations and collapsed generation.
    int effective_k = top_k_param;
    if (top_k_is_ratio && top_k_param > 0) {
        const float density = std::min(1.0f, std::max(0.0f, top_k_param / 100.0f));
        effective_k = static_cast<int>(std::ceil(density * static_cast<float>(in_dim)));
    }

    const float effective_density = in_dim > 0
        ? static_cast<float>(effective_k) / static_cast<float>(in_dim)
        : 1.0f;
    const bool requested_sparse = allow_sparse && effective_k > 0 && effective_k < in_dim && topk_state != nullptr;
    const bool use_sparse = requested_sparse && effective_density <= sparse_min_density;

    if (use_sparse) {
        std::chrono::high_resolution_clock::time_point t_select;
        if (g_profile_enabled) t_select = std::chrono::high_resolution_clock::now();

        const std::vector<int>& active_indices = update_topk_delta_state(*topk_state, x_quant, in_dim, effective_k, sort_topk);

        if (g_profile_enabled) {
            g_profile.topk_select_us += micros_since(t_select);
            g_profile.active_k_sum += static_cast<int64_t>(active_indices.size());
            g_profile.input_dim_sum += static_cast<int64_t>(in_dim);
        }

        std::chrono::high_resolution_clock::time_point t_kernel;
        if (g_profile_enabled) t_kernel = std::chrono::high_resolution_clock::now();

        bitlinear_topk_sparse(x_quant, packed_w, in_dim, out_dim, active_indices, out_accum);

        if (g_profile_enabled) {
            g_profile.sparse_kernel_us += micros_since(t_kernel);
            g_profile.topk_calls += 1;
        }
    } else {
        std::chrono::high_resolution_clock::time_point t_kernel;
        if (g_profile_enabled) t_kernel = std::chrono::high_resolution_clock::now();

        bitlinear_dense_fused(x_quant, packed_w, in_dim, out_dim, out_accum);

        if (g_profile_enabled) {
            g_profile.dense_kernel_us += micros_since(t_kernel);
            g_profile.dense_calls += 1;
            if (requested_sparse) g_profile.topk_fallback_calls += 1;
        }
    }
}

void reset_profile_cpp() {
    g_profile = ProfileStats();
    g_profile_enabled = true;
}

void set_profile_enabled_cpp(bool enabled) {
    g_profile_enabled = enabled;
}

py::dict get_profile_cpp() {
    py::dict d;
    d["dense_calls"] = g_profile.dense_calls;
    d["topk_calls"] = g_profile.topk_calls;
    d["topk_fallback_calls"] = g_profile.topk_fallback_calls;
    d["topk_select_ms"] = static_cast<double>(g_profile.topk_select_us) / 1000.0;
    d["dense_kernel_ms"] = static_cast<double>(g_profile.dense_kernel_us) / 1000.0;
    d["sparse_kernel_ms"] = static_cast<double>(g_profile.sparse_kernel_us) / 1000.0;
    d["avg_active_k"] = g_profile.topk_calls ? static_cast<double>(g_profile.active_k_sum) / static_cast<double>(g_profile.topk_calls) : 0.0;
    d["avg_input_dim"] = g_profile.topk_calls ? static_cast<double>(g_profile.input_dim_sum) / static_cast<double>(g_profile.topk_calls) : 0.0;
    d["avg_density"] = g_profile.input_dim_sum ? static_cast<double>(g_profile.active_k_sum) / static_cast<double>(g_profile.input_dim_sum) : 0.0;
    return d;
}

void quantize_absmax(int8_t* out_quant, float& out_scale, const float* in, int dim) {
    float max_abs = 1e-8f;
    for (int i = 0; i < dim; ++i) {
        const float v = std::abs(in[i]);
        if (v > max_abs) max_abs = v;
    }
    out_scale = 127.0f / max_abs;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dim; ++i) {
        const float q = std::round(in[i] * out_scale);
        out_quant[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, q)));
    }
}

void dequantize(float* out, const int32_t* in, float scale_x, float gamma, int dim) {
    const float factor = gamma / std::max(scale_x, 1e-8f);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dim; ++i) out[i] = in[i] * factor;
}

void rms_norm_weighted(float* out, const float* in, const float* weight, int dim, float eps = 1e-5f) {
    float ss = 0.0f;
    for (int i = 0; i < dim; ++i) ss += in[i] * in[i];
    const float inv = 1.0f / std::sqrt((ss / dim) + eps);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dim; ++i) out[i] = in[i] * inv * weight[i];
}

void apply_rope(float* vec, int seq_pos, int dim, int num_heads, float rope_theta = 10000.0f) {
    const int head_dim = dim / num_heads;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < num_heads; ++h) {
        float* head_vec = vec + (h * head_dim);
        for (int i = 0; i + 1 < head_dim; i += 2) {
            const float freq = 1.0f / std::pow(rope_theta, static_cast<float>(i) / head_dim);
            const float val = static_cast<float>(seq_pos) * freq;
            const float cos_val = std::cos(val);
            const float sin_val = std::sin(val);
            const float v0 = head_vec[i];
            const float v1 = head_vec[i + 1];
            head_vec[i] = v0 * cos_val - v1 * sin_val;
            head_vec[i + 1] = v0 * sin_val + v1 * cos_val;
        }
    }
}

void attention_forward_omp(
    float* attn_out,
    const float* q,
    float* k_cache,
    float* v_cache,
    int current_pos,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int kv_groups = std::max(1, num_heads / std::max(1, num_kv_heads));
    #pragma omp parallel for schedule(dynamic, 4)
    for (int h = 0; h < num_heads; ++h) {
        const int kv_head = std::min(num_kv_heads - 1, h / kv_groups);
        const float* q_head = q + (h * head_dim);
        float* out_head = attn_out + (h * head_dim);
        std::vector<float> scores(current_pos + 1, 0.0f);
        float max_score = -1e9f;
        for (int t = 0; t <= current_pos; ++t) {
            const float* k_head = k_cache + (t * num_kv_heads * head_dim) + (kv_head * head_dim);
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
            const float s = scores[t] / std::max(sum_exp, 1e-8f);
            const float* v_head = v_cache + (t * num_kv_heads * head_dim) + (kv_head * head_dim);
            for (int i = 0; i < head_dim; ++i) out_head[i] += s * v_head[i];
        }
    }
}

static inline float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

void unpack_embedding_fp32(int token_id, const float* embed_w, float* out_hidden, int hidden_dim) {
    const float* row_ptr = embed_w + (token_id * hidden_dim);
    std::copy(row_ptr, row_ptr + hidden_dim, out_hidden);
}

int64_t lm_head_argmax_fp32(
    const float* hidden,
    const float* lm_head_w,
    int vocab_size,
    int hidden_dim,
    const std::vector<int64_t>& past_tokens
) {
    std::vector<float> logits(vocab_size, 0.0f);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int v = 0; v < vocab_size; ++v) {
        const float* row_ptr = lm_head_w + (v * hidden_dim);
        float score = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) score += hidden[i] * row_ptr[i];
        logits[v] = score;
    }

    const float penalty = 1.2f;
    for (int64_t pt : past_tokens) {
        if (pt >= 0 && pt < vocab_size) {
            if (logits[pt] > 0) logits[pt] /= penalty;
            else logits[pt] *= penalty;
        }
    }

    int64_t best_id = 0;
    float best_val = -1e9f;
    for (int v = 0; v < vocab_size; ++v) {
        if (logits[v] > best_val) {
            best_val = logits[v];
            best_id = v;
        }
    }
    return best_id;
}

std::vector<int64_t> force_evolved_generate_cpp(
    std::vector<int64_t> prompt_tokens,
    int64_t max_new_tokens,
    int64_t hidden_dim_i64,
    int64_t inter_dim_i64,
    int64_t kv_dim_i64,
    int64_t num_layers_i64,
    int64_t num_heads_i64,
    int64_t num_kv_heads_i64,
    int64_t vocab_size_i64,
    int64_t eos_id,
    std::vector<int64_t> stop_token_ids,
    int64_t top_k_i64,
    bool top_k_is_ratio,
    double sparse_min_density,
    bool sort_topk,
    int64_t sparse_scope_i64,
    int64_t architecture_mode,
    int64_t activation_mode,
    double rope_theta,
    std::vector<torch::Tensor> layer_weights,
    std::vector<float> layer_gammas,
    std::vector<torch::Tensor> layer_norms,
    torch::Tensor final_norm_tensor,
    torch::Tensor embed_tensor,
    torch::Tensor lm_head_tensor
) {
    std::vector<int64_t> generated = prompt_tokens;

    const int hidden_dim = static_cast<int>(hidden_dim_i64);
    const int inter_dim = static_cast<int>(inter_dim_i64);
    const int kv_dim = static_cast<int>(kv_dim_i64);
    const int num_layers = static_cast<int>(num_layers_i64);
    const int num_heads = static_cast<int>(num_heads_i64);
    const int num_kv_heads = static_cast<int>(num_kv_heads_i64);
    const int vocab_size = static_cast<int>(vocab_size_i64);
    const int top_k = static_cast<int>(std::max<int64_t>(0, top_k_i64));
    const float sparse_min_density_f = static_cast<float>(std::min(1.0, std::max(0.0, sparse_min_density)));
    const int sparse_scope = static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(3, sparse_scope_i64)));
    const bool use_mlgru = architecture_mode == 1;
    const bool use_relu2 = activation_mode == 1;

    const float* embed_ptr = embed_tensor.data_ptr<float>();
    const float* lm_head_ptr = lm_head_tensor.data_ptr<float>();
    const float* final_norm_w = final_norm_tensor.data_ptr<float>();

    const int max_seq_len = static_cast<int>(prompt_tokens.size() + max_new_tokens);
    const int head_dim = std::max(1, hidden_dim / std::max(1, num_heads));
    const int accum_dim = std::max(std::max(hidden_dim, inter_dim), kv_dim);

    std::vector<float> k_cache(num_layers * max_seq_len * kv_dim, 0.0f);
    std::vector<float> v_cache(num_layers * max_seq_len * kv_dim, 0.0f);
    std::vector<float> mlgru_state(num_layers * hidden_dim, 0.0f);
    std::vector<TopKDeltaState> topk_states(num_layers * 7);

    std::vector<float> hidden_state(hidden_dim, 0.0f);
    std::vector<float> norm_state(hidden_dim, 0.0f);
    std::vector<int8_t> hidden_q(std::max(hidden_dim, inter_dim), 0);
    std::vector<int32_t> accum_buf(accum_dim, 0);
    std::vector<float> q(hidden_dim, 0.0f);
    std::vector<float> k(kv_dim, 0.0f);
    std::vector<float> v(kv_dim, 0.0f);
    std::vector<float> attn_out(hidden_dim, 0.0f);
    std::vector<float> attn_normed(hidden_dim, 0.0f);
    std::vector<float> ffn_norm_state(hidden_dim, 0.0f);
    std::vector<float> gate_out(inter_dim, 0.0f);
    std::vector<float> up_out(inter_dim, 0.0f);
    std::vector<float> ffn_act(inter_dim, 0.0f);
    std::vector<float> ffn_sub_normed(inter_dim, 0.0f);

    auto state_slot = [&](int layer, int projection) -> TopKDeltaState* {
        if (top_k <= 0) return nullptr;
        return &topk_states[layer * 7 + projection];
    };

    const int total_steps = static_cast<int>(prompt_tokens.size() + max_new_tokens - 1);
    for (int step = 0; step < total_steps; ++step) {
        const int current_pos = step;
        const int current_token = (step < static_cast<int>(prompt_tokens.size()))
            ? static_cast<int>(prompt_tokens[step])
            : static_cast<int>(generated.back());

        unpack_embedding_fp32(current_token, embed_ptr, hidden_state.data(), hidden_dim);

        for (int l = 0; l < num_layers; ++l) {
            const int w_idx = l * 7;
            if (w_idx + 6 >= static_cast<int>(layer_weights.size()) || layer_weights[w_idx + 0].numel() == 0) {
                continue;
            }

            const float* attn_norm_w = layer_norms[l * 4].data_ptr<float>();
            const float* ffn_norm_w = layer_norms[l * 4 + 1].data_ptr<float>();
            const float* attn_sub_norm_w = layer_norms[l * 4 + 2].data_ptr<float>();
            const float* ffn_sub_norm_w = layer_norms[l * 4 + 3].data_ptr<float>();

            rms_norm_weighted(norm_state.data(), hidden_state.data(), attn_norm_w, hidden_dim);

            float scale_x = 1.0f;
            quantize_absmax(hidden_q.data(), scale_x, norm_state.data(), hidden_dim);

            bitlinear_dispatch(hidden_q.data(), layer_weights[w_idx + 0].data_ptr<uint8_t>(), hidden_dim, hidden_dim, accum_buf.data(), top_k, top_k_is_ratio, sparse_min_density_f, sort_topk, projection_allows_sparse(sparse_scope, 0), state_slot(l, 0));
            dequantize(q.data(), accum_buf.data(), scale_x, layer_gammas[w_idx + 0], hidden_dim);

            bitlinear_dispatch(hidden_q.data(), layer_weights[w_idx + 1].data_ptr<uint8_t>(), hidden_dim, kv_dim, accum_buf.data(), top_k, top_k_is_ratio, sparse_min_density_f, sort_topk, projection_allows_sparse(sparse_scope, 1), state_slot(l, 1));
            dequantize(k.data(), accum_buf.data(), scale_x, layer_gammas[w_idx + 1], kv_dim);

            bitlinear_dispatch(hidden_q.data(), layer_weights[w_idx + 2].data_ptr<uint8_t>(), hidden_dim, kv_dim, accum_buf.data(), top_k, top_k_is_ratio, sparse_min_density_f, sort_topk, projection_allows_sparse(sparse_scope, 2), state_slot(l, 2));
            dequantize(v.data(), accum_buf.data(), scale_x, layer_gammas[w_idx + 2], kv_dim);

            if (use_mlgru) {
                float* recurrent = mlgru_state.data() + (l * hidden_dim);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < hidden_dim; ++i) {
                    const float reset = sigmoid_stable(q[i]);
                    const int kv_i = i % kv_dim;
                    const float update = sigmoid_stable(k[kv_i]);
                    const float candidate = std::tanh(v[kv_i]) * reset;
                    recurrent[i] = update * recurrent[i] + (1.0f - update) * candidate;
                    attn_out[i] = recurrent[i];
                }
            } else {
                apply_rope(q.data(), current_pos, hidden_dim, num_heads, static_cast<float>(rope_theta));
                apply_rope(k.data(), current_pos, kv_dim, num_kv_heads, static_cast<float>(rope_theta));

                const int cache_offset = (l * max_seq_len * kv_dim) + (current_pos * kv_dim);
                std::copy(k.begin(), k.end(), k_cache.begin() + cache_offset);
                std::copy(v.begin(), v.end(), v_cache.begin() + cache_offset);

                attention_forward_omp(
                    attn_out.data(),
                    q.data(),
                    k_cache.data() + (l * max_seq_len * kv_dim),
                    v_cache.data() + (l * max_seq_len * kv_dim),
                    current_pos,
                    num_heads,
                    num_kv_heads,
                    head_dim
                );
            }

            float scale_attn = 1.0f;
            rms_norm_weighted(attn_normed.data(), attn_out.data(), attn_sub_norm_w, hidden_dim);
            quantize_absmax(hidden_q.data(), scale_attn, attn_normed.data(), hidden_dim);
            bitlinear_dispatch(hidden_q.data(), layer_weights[w_idx + 3].data_ptr<uint8_t>(), hidden_dim, hidden_dim, accum_buf.data(), top_k, top_k_is_ratio, sparse_min_density_f, sort_topk, projection_allows_sparse(sparse_scope, 3), state_slot(l, 3));

            const float attn_factor = layer_gammas[w_idx + 3] / std::max(scale_attn, 1e-8f);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < hidden_dim; ++i) hidden_state[i] += accum_buf[i] * attn_factor;

            rms_norm_weighted(ffn_norm_state.data(), hidden_state.data(), ffn_norm_w, hidden_dim);
            quantize_absmax(hidden_q.data(), scale_x, ffn_norm_state.data(), hidden_dim);

            bitlinear_dispatch(hidden_q.data(), layer_weights[w_idx + 4].data_ptr<uint8_t>(), hidden_dim, inter_dim, accum_buf.data(), top_k, top_k_is_ratio, sparse_min_density_f, sort_topk, projection_allows_sparse(sparse_scope, 4), state_slot(l, 4));
            dequantize(gate_out.data(), accum_buf.data(), scale_x, layer_gammas[w_idx + 4], inter_dim);

            const bool has_up_proj = layer_weights[w_idx + 5].numel() > 0;
            if (has_up_proj) {
                bitlinear_dispatch(hidden_q.data(), layer_weights[w_idx + 5].data_ptr<uint8_t>(), hidden_dim, inter_dim, accum_buf.data(), top_k, top_k_is_ratio, sparse_min_density_f, sort_topk, projection_allows_sparse(sparse_scope, 5), state_slot(l, 5));
                dequantize(up_out.data(), accum_buf.data(), scale_x, layer_gammas[w_idx + 5], inter_dim);
            }

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < inter_dim; ++i) {
                const float g = gate_out[i];
                if (has_up_proj) {
                    if (use_relu2) {
                        const float relu = std::max(0.0f, g);
                        ffn_act[i] = relu * relu * up_out[i];
                    } else {
                        const float silu = g / (1.0f + std::exp(-g));
                        ffn_act[i] = silu * up_out[i];
                    }
                } else {
                    const float cdf = 0.5f * (1.0f + std::tanh(0.79788456f * (g + 0.044715f * g * g * g)));
                    ffn_act[i] = g * cdf;
                }
            }

            float scale_ffn = 1.0f;
            rms_norm_weighted(ffn_sub_normed.data(), ffn_act.data(), ffn_sub_norm_w, inter_dim);
            quantize_absmax(hidden_q.data(), scale_ffn, ffn_sub_normed.data(), inter_dim);
            bitlinear_dispatch(hidden_q.data(), layer_weights[w_idx + 6].data_ptr<uint8_t>(), inter_dim, hidden_dim, accum_buf.data(), top_k, top_k_is_ratio, sparse_min_density_f, sort_topk, projection_allows_sparse(sparse_scope, 6), state_slot(l, 6));

            const float down_factor = layer_gammas[w_idx + 6] / std::max(scale_ffn, 1e-8f);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < hidden_dim; ++i) hidden_state[i] += accum_buf[i] * down_factor;
        }

        rms_norm_weighted(norm_state.data(), hidden_state.data(), final_norm_w, hidden_dim);

        if (step >= static_cast<int>(prompt_tokens.size()) - 1) {
            const int64_t next_token = lm_head_argmax_fp32(norm_state.data(), lm_head_ptr, vocab_size, hidden_dim, generated);
            generated.push_back(next_token);
            if (next_token == eos_id || next_token == 0 || next_token == 2 || next_token == 128001) break;
            // Base completion models keep generating past the answer (e.g. extra
            // "Question:" lines). For Q&A prompts, the model emits a newline at
            // the end of its answer, so stopping there yields a clean reply.
            if (!stop_token_ids.empty()) {
                for (int64_t stop_id : stop_token_ids) {
                    if (next_token == stop_id) goto generation_done;
                }
            }
        }
    }

    generation_done:;

    return generated;
}
"""


def _extension_flags() -> list[str]:
    machine = platform.machine().lower()
    system = platform.system().lower()
    if machine in {"amd64", "x86_64", "x64"}:
        if system.startswith("win"):
            return ["/O2", "/arch:AVX2", "/fp:fast", "/openmp", "/DLEVIATHAN_AVX2"]
        return ["-O3", "-march=native", "-ffast-math", "-fopenmp", "-DLEVIATHAN_AVX2"]
    if "arm" in machine or "aarch64" in machine:
        if system.startswith("darwin"):
            return ["-O3", "-ffast-math"]
        return ["-O3", "-ffast-math", "-fopenmp", "-DLEVIATHAN_NEON_DOT"]
    return ["/O2", "/openmp"] if system.startswith("win") else ["-O3", "-fopenmp"]


_evolved_engine = None


def get_runtime():
    global _evolved_engine
    if _evolved_engine is not None:
        return _evolved_engine

    print("[BUILD] Compiling Leviathan sparse SIMD runtime...")
    try:
        _evolved_engine = load_inline(
            name="leviathan_phase58_sparse_scope",
            cpp_sources=[cpp_source],
            functions=["force_evolved_generate_cpp", "reset_profile_cpp", "set_profile_enabled_cpp", "get_profile_cpp"],
            extra_cflags=_extension_flags(),
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Leviathan needs Ninja plus a PyTorch-compatible C++ compiler. "
            "On Windows, install Visual Studio Build Tools with the MSVC C++ toolchain "
            "or run from a Developer PowerShell where cl.exe is available."
        ) from exc
    print("[BUILD] Compile complete.\n")
    return _evolved_engine


def _lower_keys(meta: dict) -> Iterable[tuple[str, str]]:
    for key in meta["tensors"].keys():
        yield key, key.lower()


def _layer_number_from_key(key: str) -> int | None:
    parts = key.split(".")
    for marker in ("layers", "h", "block", "blocks"):
        if marker in parts:
            idx = parts.index(marker) + 1
            if idx < len(parts) and parts[idx].isdigit():
                return int(parts[idx])
    return None


class RestoredBitNet:
    def __init__(
        self,
        bin_path: str = "leviathan_native.bin",
        meta_path: str = "leviathan_native_meta.json",
        architecture: str = "transformer",
        top_k: float = 0.0,
        prompt_template: str = "auto",
        profile: bool = False,
        sparse_min_density: float = 1.0,
        sort_topk: bool = True,
        sparse_scope: str = "all",
    ):
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.architecture = architecture
        self.architecture_mode = 1 if architecture == "mlgru" else 0

        # Interpret --top-k: a fraction 0<f<=1 is a per-layer density ratio
        # (recommended, scales K with in_dim); a value >1 is a flat absolute K
        # (legacy). The C++ runtime receives both and resolves the effective K
        # per layer in bitlinear_dispatch.
        top_k = float(top_k)
        if 0 < top_k <= 1.0:
            self.top_k = int(round(top_k * 100))   # density as 0..100
            self.top_k_is_ratio = True
        else:
            self.top_k = max(0, int(top_k))
            self.top_k_is_ratio = False
        self.prompt_template = prompt_template
        self.profile = bool(profile)
        self.sparse_min_density = max(0.0, min(1.0, float(sparse_min_density)))
        self.sort_topk = bool(sort_topk)
        sparse_scope_map = {"all": 0, "ffn": 1, "down": 2, "none": 3}
        if sparse_scope not in sparse_scope_map:
            raise ValueError(f"Unsupported sparse scope: {sparse_scope}")
        self.sparse_scope = sparse_scope_map[sparse_scope]

        target_model = self.meta.get("model_name", "microsoft/bitnet-b1.58-2B-4T-bf16")
        try:
            self.config = AutoConfig.from_pretrained(target_model)
        except Exception:
            self.config = None
        self.meta_config = self.meta.get("model_config", {})
        self.tokenizer = AutoTokenizer.from_pretrained(target_model)

        embed_keys = [
            key
            for key, lower in _lower_keys(self.meta)
            if "embed_tokens" in lower or "tok_embeddings" in lower or "word_embeddings" in lower or ".wte" in lower
        ]
        if not embed_keys:
            raise KeyError("No FP16 embedding tensor found in metadata.")

        embed_info = self.meta["tensors"][embed_keys[0]]
        if embed_info.get("type") != "float16":
            raise TypeError("Embedding tensor must be stored as float16. Re-run quantizer.py with the v2 format.")
        self.vocab_size = int(embed_info["shape"][0])
        self.hidden_dim = int(embed_info["shape"][1])

        layer_nums = [_layer_number_from_key(key) for key in self.meta["tensors"].keys()]
        layer_nums = [num for num in layer_nums if num is not None]
        self.num_layers = max(layer_nums) + 1 if layer_nums else 0
        if self.num_layers <= 0:
            raise KeyError("No transformer/MLGRU layer tensors found in metadata.")

        def cfg_value(name: str, default=None):
            value = getattr(self.config, name, None) if self.config is not None else None
            if value is None:
                value = self.meta_config.get(name, default)
            return default if value is None else value

        self.num_heads = int(cfg_value("num_attention_heads", max(1, self.hidden_dim // 128)))
        self.head_dim = int(cfg_value("head_dim", max(1, self.hidden_dim // self.num_heads)))
        self.num_kv_heads = int(cfg_value("num_key_value_heads", self.num_heads))
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.inter_dim = self.hidden_dim * 4
        self.hidden_act = str(cfg_value("hidden_act", "silu") or "silu").lower()
        self.activation_mode = 1 if self.hidden_act == "relu2" else 0
        rope_parameters = getattr(self.config, "rope_parameters", None) or {}
        self.rope_theta = float(
            rope_parameters.get("rope_theta", cfg_value("rope_theta", 10000.0))
            if isinstance(rope_parameters, dict)
            else cfg_value("rope_theta", 10000.0)
        )

        file_size = os.path.getsize(bin_path)
        print(f"[SYSTEM] Cloning {file_size / (1024**3):.2f}GB package into RAM.")
        self.mmap_tensor = torch.from_file(bin_path, shared=True, size=file_size, dtype=torch.uint8).clone()

        def ensure_float32(tensor: torch.Tensor, shape: list[int]) -> torch.Tensor:
            return tensor.view(torch.float16).to(torch.float32).reshape(shape).contiguous()

        def get_slice_fp16(tensor_name: str) -> torch.Tensor | None:
            info = self.meta["tensors"].get(tensor_name)
            if not info:
                return None
            if info.get("type") != "float16":
                raise TypeError(f"{tensor_name} is {info.get('type')}, expected float16.")
            raw = self.mmap_tensor[info["offset"] : info["offset"] + info["size"]]
            return ensure_float32(raw, info["shape"])

        def packed_row_bytes(info: dict) -> int:
            if "packed_shape" in info:
                return int(info["packed_shape"][1])
            return (int(info["shape"][1]) + 3) // 4

        def get_packed_rows(tensor_name: str, row_start: int = 0, row_count: int | None = None) -> torch.Tensor:
            info = self.meta["tensors"][tensor_name]
            if info.get("type") not in {"ternary_2bit_packed", "2bit_packed"}:
                raise TypeError(f"{tensor_name} is {info.get('type')}, expected packed ternary.")
            rows = int(info["shape"][0])
            row_count = rows - row_start if row_count is None else row_count
            row_bytes = packed_row_bytes(info)
            offset = int(info["offset"]) + (row_start * row_bytes)
            size = row_count * row_bytes
            return self.mmap_tensor[offset : offset + size].contiguous()

        self.embed_tensor = get_slice_fp16(embed_keys[0])
        lm_head_keys = [
            key
            for key, lower in _lower_keys(self.meta)
            if "lm_head" in lower or "embed_out" in lower or lower.endswith("output.weight")
        ]
        self.lm_head_tensor = get_slice_fp16(lm_head_keys[0]) if lm_head_keys else self.embed_tensor

        def find_norm_weight(layer: int, patterns: list[str], fallback_dim: int) -> torch.Tensor:
            for pattern in patterns:
                for key, lower in _lower_keys(self.meta):
                    if f".{layer}." in key and pattern in lower and "weight" in lower:
                        found = get_slice_fp16(key)
                        if found is not None:
                            return found.reshape(-1).contiguous()
            return torch.ones(fallback_dim, dtype=torch.float32)

        final_norm_keys = [
            key
            for key, lower in _lower_keys(self.meta)
            if lower == "model.norm.weight" or lower.endswith(".ln_f.weight") or lower.endswith(".final_layernorm.weight")
        ]
        self.final_norm = get_slice_fp16(final_norm_keys[0]) if final_norm_keys else torch.ones(self.hidden_dim, dtype=torch.float32)
        self.final_norm = self.final_norm.reshape(self.hidden_dim).contiguous()

        def layer_key_matches(key: str, lower: str, layer: int, pattern: str) -> bool:
            return f".{layer}." in key and pattern in lower and "weight" in lower and "norm" not in lower

        def get_linear_tensor(layer: int, patterns: list[str], qkv_part: int | None = None) -> tuple[torch.Tensor, float, int]:
            for pattern in patterns:
                for key, lower in _lower_keys(self.meta):
                    if not layer_key_matches(key, lower, layer, pattern):
                        continue
                    info = self.meta["tensors"][key]
                    gamma = float(info.get("gamma", 1.0))
                    rows = int(info["shape"][0])
                    if "query_key_value" in lower and qkv_part is not None and rows >= self.hidden_dim * 3:
                        return get_packed_rows(key, qkv_part * self.hidden_dim, self.hidden_dim), gamma, self.hidden_dim
                    if rows > self.hidden_dim and any(part in lower for part in ("gate", "up", "w1", "w3", "dense_h_to_4h")):
                        self.inter_dim = rows
                    return get_packed_rows(key), gamma, rows
            return torch.zeros(0, dtype=torch.uint8), 1.0, 0

        self.layer_tensors: list[torch.Tensor] = []
        self.layer_gammas: list[float] = []
        for layer in range(self.num_layers):
            q_t, q_g, _ = get_linear_tensor(layer, ["q_proj", "query_key_value"], qkv_part=0)
            k_t, k_g, k_rows = get_linear_tensor(layer, ["k_proj", "query_key_value"], qkv_part=1)
            v_t, v_g, v_rows = get_linear_tensor(layer, ["v_proj", "query_key_value"], qkv_part=2)
            o_t, o_g, _ = get_linear_tensor(layer, ["o_proj", "dense", "out_proj"])
            gate_t, gate_g, gate_rows = get_linear_tensor(layer, ["gate_proj", "dense_h_to_4h", "w1"])
            up_t, up_g, up_rows = get_linear_tensor(layer, ["up_proj", "w3"])
            down_t, down_g, _ = get_linear_tensor(layer, ["down_proj", "dense_4h_to_h", "w2"])
            if k_rows:
                self.kv_dim = k_rows
            if v_rows and v_rows != self.kv_dim:
                raise ValueError(f"K/V projection mismatch in layer {layer}: k={self.kv_dim}, v={v_rows}")
            if gate_rows:
                self.inter_dim = gate_rows
            elif up_rows:
                self.inter_dim = up_rows

            for tensor, gamma in (
                (q_t, q_g),
                (k_t, k_g),
                (v_t, v_g),
                (o_t, o_g),
                (gate_t, gate_g),
                (up_t, up_g),
                (down_t, down_g),
            ):
                self.layer_tensors.append(tensor)
                self.layer_gammas.append(gamma)

        if self.kv_dim <= 0:
            self.kv_dim = self.hidden_dim
        if self.kv_dim % self.head_dim == 0:
            self.num_kv_heads = max(1, self.kv_dim // self.head_dim)
        else:
            self.num_kv_heads = self.num_heads
            self.kv_dim = self.num_kv_heads * self.head_dim

        self.layer_norms: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            attn_norm = find_norm_weight(layer, ["input_layernorm", "ln_1", "attention_norm"], self.hidden_dim)
            ffn_norm = find_norm_weight(layer, ["post_attention_layernorm", "ln_2"], self.hidden_dim)
            attn_sub_norm = find_norm_weight(layer, ["attn_sub_norm"], self.hidden_dim)
            ffn_sub_norm = find_norm_weight(layer, ["ffn_sub_norm"], self.inter_dim)
            self.layer_norms.extend([attn_norm, ffn_norm, attn_sub_norm, ffn_sub_norm])

        top_k_desc = "dense"
        if self.top_k:
            top_k_desc = f"{self.top_k / 100.0:.0%} per layer" if self.top_k_is_ratio else str(self.top_k)
        print(
            "[SYSTEM] Runtime ready "
            f"(layers={self.num_layers}, hidden={self.hidden_dim}, inter={self.inter_dim}, "
            f"heads={self.num_heads}, kv_heads={self.num_kv_heads}, kv_dim={self.kv_dim}, "
            f"act={self.hidden_act}, rope_theta={self.rope_theta:g}, "
            f"architecture={self.architecture}, top_k={top_k_desc})."
        )

    def format_prompt(self, prompt_text: str) -> str:
        prompt_text = prompt_text.strip()
        if self.prompt_template == "qa" or (self.prompt_template == "auto" and prompt_text.endswith("?")):
            return f"Question: {prompt_text}\nAnswer:"
        return prompt_text

    def generate(self, prompt_text: str, max_new_tokens: int = 150):
        formatted_prompt = self.format_prompt(prompt_text)
        prompt_tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=True) or [0]
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = 2
        if isinstance(eos_id, list):
            eos_id = eos_id[0]

        # Base completion checkpoints (e.g. bitnet-b1.58-2B-4T) do not emit EOS at
        # the end of an answer; they continue with the next "Question:" line. When
        # a Q&A template is active, stop at the first newline so the reply stays
        # clean. A plain prompt keeps raw continuation behavior.
        stop_token_ids: list[int] = []
        is_qa_prompt = self.prompt_template == "qa" or (
            self.prompt_template == "auto" and prompt_text.strip().endswith("?")
        )
        if is_qa_prompt:
            for stop_text in ("\n", "\n\n"):
                enc = self.tokenizer.encode(stop_text, add_special_tokens=False)
                if enc and enc[0] not in stop_token_ids:
                    stop_token_ids.append(enc[0])

        start_time = time.perf_counter()
        runtime = get_runtime()
        if self.profile:
            runtime.reset_profile_cpp()
        else:
            runtime.set_profile_enabled_cpp(False)
        output_tokens = runtime.force_evolved_generate_cpp(
            prompt_tokens,
            max_new_tokens,
            self.hidden_dim,
            self.inter_dim,
            self.kv_dim,
            self.num_layers,
            self.num_heads,
            self.num_kv_heads,
            self.vocab_size,
            eos_id,
            stop_token_ids,
            self.top_k,
            self.top_k_is_ratio,
            self.sparse_min_density,
            self.sort_topk,
            self.sparse_scope,
            self.architecture_mode,
            self.activation_mode,
            self.rope_theta,
            self.layer_tensors,
            self.layer_gammas,
            self.layer_norms,
            self.final_norm,
            self.embed_tensor,
            self.lm_head_tensor,
        )
        elapsed = (time.perf_counter() - start_time) * 1000.0

        profile_data = runtime.get_profile_cpp() if self.profile else None
        if self.profile:
            runtime.set_profile_enabled_cpp(False)

        gen_tokens = output_tokens[len(prompt_tokens) :]
        output_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        gen_len = max(len(gen_tokens), 1)
        return output_text, elapsed, gen_len, profile_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", nargs="?", default="chat")
    parser.add_argument("--bin", default="leviathan_native.bin")
    parser.add_argument("--meta", default="leviathan_native_meta.json")
    parser.add_argument("--max-new", type=int, default=150)
    parser.add_argument("--top-k", type=float, default=0.0,
                        help="Activation sparsity. A fraction 0<f<=1 (e.g. 0.8 keeps the largest 80%% "
                             "per layer) is recommended: it scales K with each layer's width so wide "
                             "layers (down_proj in_dim=6912) don't collapse. An integer >1 (e.g. 512) "
                             "is treated as a flat absolute K (legacy, harsh on wide layers).")
    parser.add_argument("--architecture", choices=["transformer", "mlgru"], default="transformer")
    parser.add_argument("--prompt-template", choices=["auto", "plain", "qa"], default="auto")
    parser.add_argument("--profile", action="store_true", help="Print C++ bitlinear timing counters for each prompt.")
    parser.add_argument("--sparse-min-density", type=float, default=1.0, help="Use sparse Top-K only when effective density is <= this value. Default 1.0 preserves previous behavior; try 0.6 to auto-fallback 0.8/0.9 to dense.")
    parser.add_argument("--no-top-k-sort", action="store_true", help="Experimental: skip sorting active Top-K indices after nth_element. Preserves the selected set but may change sparse memory access locality.")
    parser.add_argument("--sparse-scope", choices=["all", "ffn", "down", "none"], default="all", help="Experimental sparse projection scope: all projections, FFN only, down projection only, or none.")
    args = parser.parse_args()

    print("=" * 100)
    print("[PHASE 50] THE LEVIATHAN - Sparse SIMD / MLGRU Runtime")
    print("=" * 100)

    engine = RestoredBitNet(
        bin_path=args.bin,
        meta_path=args.meta,
        architecture=args.architecture,
        top_k=args.top_k,
        prompt_template=args.prompt_template,
        profile=args.profile,
        sparse_min_density=args.sparse_min_density,
        sort_topk=not args.no_top_k_sort,
        sparse_scope=args.sparse_scope,
    )
    while True:
        try:
            prompt = input("\nUSER> ").strip()
        except EOFError:
            break
        if not prompt or prompt.lower() in {"/exit", "exit", "quit"}:
            break

        text, elapsed, gen_len, profile_data = engine.generate(prompt, args.max_new)
        print(f"\nENGINE> {text.strip()}")
        print(f"[Stats: {elapsed:.2f} ms | {gen_len / (max(elapsed / 1000.0, 1e-5)):.2f} tokens/sec]")
        if profile_data:
            print(
                "[Profile: "
                f"dense_calls={profile_data['dense_calls']} "
                f"dense_kernel_ms={profile_data['dense_kernel_ms']:.2f} "
                f"topk_calls={profile_data['topk_calls']} "
                f"topk_fallback_calls={profile_data['topk_fallback_calls']} "
                f"topk_select_ms={profile_data['topk_select_ms']:.2f} "
                f"sparse_kernel_ms={profile_data['sparse_kernel_ms']:.2f} "
                f"avg_active_k={profile_data['avg_active_k']:.1f} "
                f"avg_input_dim={profile_data['avg_input_dim']:.1f} "
                f"avg_density={profile_data['avg_density']:.3f}"
                "]"
            )


if __name__ == "__main__":
    main()
