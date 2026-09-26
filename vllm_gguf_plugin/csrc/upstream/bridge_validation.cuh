// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "bridge_support.cuh"

namespace {
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
  STD_TORCH_CHECK(W.element_size() == 1, kMoeNotEligibleMarker,
                  ": W "
                  "must contain packed byte data");
  STD_TORCH_CHECK(is_upstream_type(type), kMoeNotEligibleMarker,
                  ": "
                  "unsupported quantization type ",
                  type);
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

}  // namespace
