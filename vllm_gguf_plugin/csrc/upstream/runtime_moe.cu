// SPDX-License-Identifier: Apache-2.0
#include "ggml_dypes.cuh"
#include "torch_context.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"
#include "mmvf.cuh"
#include "mmf.cuh"

#include <algorithm>
#include <limits>
#include <torch/csrc/stable/ops.h>

namespace {
// Shared with kernel_support.py; auto mode recognizes this exact marker.
constexpr const char* kMoeNotEligibleMarker = "VLLM_GGUF_MOE_NOT_ELIGIBLE";

void check_moe_inputs(const Tensor& X, const Tensor& W, const Tensor& topk_ids,
                      int64_t type, int64_t row, int64_t top_k,
                      int64_t tokens) {
  STD_TORCH_CHECK(X.is_cuda() && W.is_cuda() && topk_ids.is_cuda(),
                  kMoeNotEligibleMarker,
                  ": all "
                  "tensors must be CUDA tensors");
  STD_TORCH_CHECK(X.get_device_index() == W.get_device_index() &&
                      X.get_device_index() == topk_ids.get_device_index(),
                  kMoeNotEligibleMarker,
                  ": "
                  "tensors must be on the same CUDA device");
  STD_TORCH_CHECK(X.dim() == 2 && W.dim() == 3 && topk_ids.dim() == 2,
                  kMoeNotEligibleMarker,
                  ": "
                  "expected X[ tokens, K ], W[ experts, rows, packed ], and "
                  "topk_ids[ tokens, top_k ]");
  STD_TORCH_CHECK(
      X.is_contiguous() && W.is_contiguous() && topk_ids.is_contiguous(),
      kMoeNotEligibleMarker,
      ": "
      "tensors must be contiguous");
  STD_TORCH_CHECK(topk_ids.scalar_type() == ScalarType::Int,
                  kMoeNotEligibleMarker,
                  ": "
                  "topk_ids must be int32");
  STD_TORCH_CHECK(X.scalar_type() == ScalarType::Float ||
                      X.scalar_type() == ScalarType::Half ||
                      X.scalar_type() == ScalarType::BFloat16,
                  kMoeNotEligibleMarker,
                  ": X "
                  "must have dtype fp32, fp16, or bf16");
  STD_TORCH_CHECK(
      float_type_matches(W, type) || (is_upstream_weight_type(type) &&
                                      W.scalar_type() == ScalarType::Byte),
      kMoeNotEligibleMarker,
      ": unsupported weight type or dtype mismatch: ", type);
  STD_TORCH_CHECK(tokens > 0 && top_k > 0 && row > 0, kMoeNotEligibleMarker,
                  ": "
                  "tokens, top_k, and row must be positive");
  STD_TORCH_CHECK(X.size(0) == tokens && topk_ids.size(0) == tokens &&
                      topk_ids.size(1) == top_k,
                  kMoeNotEligibleMarker,
                  ": "
                  "shape arguments do not match X/topk_ids");
  STD_TORCH_CHECK(W.size(0) > 0 && W.size(1) == row && W.size(2) > 0,
                  kMoeNotEligibleMarker,
                  ": "
                  "shape arguments do not match W");
  STD_TORCH_CHECK(top_k <= W.size(0), kMoeNotEligibleMarker,
                  ": top_k exceeds the number of experts");
}

int64_t logical_k_from_moe_weight(const Tensor& W, int64_t type,
                                  const char* op_name) {
  return logical_k_from_packed_row_bytes(W.size(2), type, op_name,
                                         "expert row");
}

int64_t checked_moe_product(int64_t lhs, int64_t rhs, const char* name) {
  STD_TORCH_CHECK(
      lhs > 0 && rhs > 0 && lhs <= std::numeric_limits<int64_t>::max() / rhs,
      kMoeNotEligibleMarker, ": ", name, " overflows int64");
  return lhs * rhs;
}

enum class MoeKernel { kMmvq, kMmq, kIq1MChunks, kMmvf, kMmf };

MoeKernel select_moe_kernel(const Tensor& W, int64_t type, int64_t row,
                            int64_t tokens, int cc, int warp_size, size_t smpbo,
                            int* mmvq_max) {
  const auto ggml_type = static_cast<enum ggml_type>(type);
  if (is_upstream_float_type(type)) {
    const ggml_tensor weight = make_moe_float_weight_tensor(W, type);
    const bool aligned =
        reinterpret_cast<uintptr_t>(W.data_ptr()) % (2 * W.element_size()) == 0;
    STD_TORCH_CHECK(aligned, kMoeNotEligibleMarker,
                    ": floating weight pointer is not aligned");
    if (tokens <= MMVF_MAX_BATCH_SIZE &&
        ggml_cuda_should_use_mmvf(ggml_type, cc, weight.ne, weight.nb,
                                  tokens)) {
      return MoeKernel::kMmvf;
    }
    if (ggml_cuda_should_use_mmf(ggml_type, cc, warp_size, weight.ne, weight.nb,
                                 tokens, /*mul_mat_id=*/true)) {
      return MoeKernel::kMmf;
    }
    STD_TORCH_CHECK(false, kMoeNotEligibleMarker,
                    ": neither MMVF nor MMF supports this type/shape/device");
  }

  *mmvq_max = std::min<int>(MMVQ_MAX_BATCH_SIZE,
                            get_mmvq_mmid_max_batch(ggml_type, cc));
  if (*mmvq_max > 0 && tokens <= *mmvq_max) {
    return MoeKernel::kMmvq;
  }
  // Preserve the MoE MMQ policy: template availability, shared memory and
  // J tile size. The upstream top-level predicate is intentionally not used.
  const bool fallback = row % 128 != 0;
  if (upstream_mmq_type_supported(type) && smpbo >= 48 * 1024 &&
      ggml_cuda_mmq_get_J_max(ggml_type, fallback, cc, tokens) > 0) {
    return MoeKernel::kMmq;
  }
  if (type == GGML_TYPE_IQ1_M && *mmvq_max > 0) {
    return MoeKernel::kIq1MChunks;
  }
  STD_TORCH_CHECK(false, kMoeNotEligibleMarker,
                  ": neither MMVQ nor MMQ supports this type/shape/device");
}

Tensor run_upstream_moe_projection(const Tensor& W, const Tensor& X,
                                   const Tensor& topk_ids, int64_t type,
                                   int64_t row, int64_t top_k, int64_t tokens,
                                   int64_t k) {
  const int32_t device_index = X.get_device_index();
  const DeviceGuard device_guard(device_index);
  const auto& device = ggml_cuda_info().devices[device_index];
  int mmvq_max = 0;
  const MoeKernel kernel =
      select_moe_kernel(W, type, row, tokens, device.cc, device.warp_size,
                        device.smpbo, &mmvq_max);

  UpstreamCall call(X);
  const int64_t output_rows = tokens * top_k;
  const ProjectionBuffers buffers = projection_buffers(
      X, output_rows, row, 0, *call.scratch_pool, call.stream);

  ggml_tensor src0 = kernel == MoeKernel::kMmvf || kernel == MoeKernel::kMmf
                         ? make_moe_float_weight_tensor(W, type)
                         : make_moe_weight_tensor(W, type, k, row);
  ggml_tensor src1 = make_moe_input_tensor(buffers.input, k, tokens);
  ggml_tensor ids_tensor = make_moe_ids_tensor(
      static_cast<const int32_t*>(topk_ids.data_ptr()), top_k, tokens);
  ggml_tensor dst = make_moe_output_tensor(buffers.result, row, top_k, tokens);

  if (kernel == MoeKernel::kMmvq) {
    ggml_cuda_mul_mat_vec_q(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMVQ");
  } else if (kernel == MoeKernel::kMmq) {
    ggml_cuda_mul_mat_q(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMQ");
  } else if (kernel == MoeKernel::kMmvf) {
    ggml_cuda_mul_mat_vec_f(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMVF");
  } else if (kernel == MoeKernel::kMmf) {
    ggml_cuda_mul_mat_f(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMF");
  } else {
    // IQ1_M has no MMQ instance; keep MMVQ within its per-call token limit.
    for (int64_t start = 0; start < tokens; start += mmvq_max) {
      const int64_t count = std::min<int64_t>(mmvq_max, tokens - start);
      ggml_tensor chunk_input =
          make_moe_input_tensor(buffers.input + start * k, k, count);
      ggml_tensor chunk_ids = make_moe_ids_tensor(
          static_cast<const int32_t*>(topk_ids.data_ptr()) + start * top_k,
          top_k, count);
      ggml_tensor chunk_output = make_moe_output_tensor(
          buffers.result + start * top_k * row, row, top_k, count);
      ggml_cuda_mul_mat_vec_q(call.context, &src0, &chunk_input, &chunk_ids,
                              &chunk_output);
      check_launch("upstream MoE IQ1_M MMVQ");
    }
  }
  return finish_output(buffers, call.stream);
}

}  // namespace

Tensor ggml_moe_a8_upstream(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                            int64_t row, int64_t top_k, int64_t tokens) {
  check_moe_inputs(X, W, topk_ids, type, row, top_k, tokens);
  const bool float_weight = is_upstream_float_type(type);
  const int64_t k =
      float_weight ? W.size(2)
                   : logical_k_from_moe_weight(W, type, "ggml_moe_a8_upstream");
  STD_TORCH_CHECK(X.size(1) == k, kMoeNotEligibleMarker,
                  ": X K dimension does not match the expert row");
  if (!float_weight) {
    STD_TORCH_CHECK(has_weight_padding(W, k, type, "ggml_moe_a8_upstream"),
                    kMoeNotEligibleMarker,
                    ": W lacks required MATRIX_ROW_PADDING storage");
  }
  const int64_t routes = checked_moe_product(tokens, top_k, "route count");
  checked_moe_product(routes, k, "routed input element count");
  checked_moe_product(routes, row, "routed output element count");
  return run_upstream_moe_projection(W, X, topk_ids, type, row, top_k, tokens,
                                     k);
}

Tensor ggml_moe_upstream(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                         int64_t row, int64_t top_k, int64_t tokens) {
  return ggml_moe_a8_upstream(X, W, topk_ids, type, row, top_k, tokens);
}
