// SPDX-License-Identifier: Apache-2.0
#include "bridge_common.cuh"

namespace {
bool is_legacy_mmq_type(int64_t type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
      return true;
    default:
      return false;
  }
}

}  // namespace

bool ggml_should_use_mmvq(int64_t type, int64_t cc, int64_t batch) {
  // No device lookup here: Python supplies the tensor device's capability,
  // and policy tests can exercise every architecture without that hardware.
  return is_upstream_type(type) && batch > 0 && batch <= kMmvqMaxBatchSize &&
         cc > 0 && cc <= std::numeric_limits<int>::max() &&
         ggml_cuda_should_use_mmvq(static_cast<ggml_type>(type),
                                   static_cast<int>(cc), batch);
}

Tensor ggml_moe_a8_upstream(Tensor X, Tensor W, Tensor topk_ids, int64_t type,
                            int64_t row, int64_t top_k, int64_t tokens) {
  check_moe_inputs(X, W, topk_ids, type, row, top_k, tokens);
  const int64_t k = logical_k_from_moe_weight(W, type, "ggml_moe_a8_upstream");
  STD_TORCH_CHECK(X.size(1) == k, kMoeNotEligibleMarker,
                  ": X K dimension "
                  "does not match the packed expert row");
  STD_TORCH_CHECK(has_weight_padding(W, k, type, "ggml_moe_a8_upstream"),
                  kMoeNotEligibleMarker,
                  ": W lacks required "
                  "MATRIX_ROW_PADDING storage");
  const int64_t routes = checked_moe_product(tokens, top_k, "route count");
  checked_moe_product(routes, k, "routed input element count");
  checked_moe_product(routes, row, "routed output element count");
  return run_upstream_moe_projection(W, X, topk_ids, type, row, top_k, tokens,
                                     k);
}

namespace {

Tensor run_dense_entry(Tensor W, Tensor X, int64_t type, int64_t row,
                       bool vector_entry, const char* op_name) {
  check_common_inputs(W, X, row, op_name);
  if (X.size(0) == 0) {
    return torch::stable::new_empty(W, {0, row}, X.scalar_type());
  }

  const KernelMode mode = kernel_mode();
  if (mode == KernelMode::kTriton) {
    throw std::runtime_error(
        std::string(op_name) +
        ": Triton backend selected; use the Python dispatcher");
  }
  const bool legacy_available =
      vector_entry ? is_legacy_mmvq_type(type) : is_legacy_mmq_type(type);
  const char* legacy_name = vector_entry ? "MMVQ" : "MMQ";
  if (mode == KernelMode::kLegacy) {
    STD_TORCH_CHECK(legacy_available, op_name, ": no legacy ", legacy_name,
                    " kernel for quantization type ", type);
    return vector_entry ? ggml_mul_mat_vec_a8_legacy(W, X, type, row)
                        : ggml_mul_mat_a8_legacy(W, X, type, row);
  }

  if (is_upstream_type(type)) {
    const int64_t k = logical_k_from_weight(W, type, op_name);
    STD_TORCH_CHECK(X.size(1) == k, op_name,
                    ": X K dimension does not match W");
    return run_upstream_dense(W, X, type, row, k);
  }
  if (mode == KernelMode::kUpstream) {
    STD_TORCH_CHECK(false, op_name,
                    ": no upstream CUDA path for quantization type ", type);
  }
  if (legacy_available) {
    return vector_entry ? ggml_mul_mat_vec_a8_legacy(W, X, type, row)
                        : ggml_mul_mat_a8_legacy(W, X, type, row);
  }
  throw std::runtime_error(std::string(op_name) +
                           ": no eligible CUDA kernel for quantization type " +
                           std::to_string(type));
}

}  // namespace

Tensor ggml_mul_mat_vec_a8(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_dense_entry(W, X, type, row, true, "ggml_mul_mat_vec_a8");
}

Tensor ggml_mul_mat_a8(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run_dense_entry(W, X, type, row, false, "ggml_mul_mat_a8");
}
