// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/c/shim.h>

#include "mmvq.cuh"
#include "quantize.cuh"
#include "upstream_dispatch.cuh"
#include "upstream_type_list.cuh"
#include "convert.cuh"
#include "dtype_convert.cuh"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;
using torch::stable::accelerator::DeviceGuard;

Tensor ggml_mul_mat_vec_a8_legacy(Tensor W, Tensor X, int64_t type,
                                  int64_t row);
Tensor ggml_mul_mat_a8_legacy(Tensor W, Tensor X, int64_t type, int64_t row);

namespace {

// Authoritative value comes from the upstream common.cuh macro; keep this
// alias so a future upstream change only needs this line updated (guarded
// against drift by the static_assert below).
constexpr int64_t kMatrixRowPadding = MATRIX_ROW_PADDING;
// Python kernel_support.py hard-codes 512 (see _MATRIX_ROW_PADDING) and the
// test helpers mirror it. A multiple-of-512 assertion would let an upstream
// bump to e.g. 1024 pass here while Python under-allocates weight storage,
// so require exact equality and force both sides to move together.
static_assert(MATRIX_ROW_PADDING == 512,
              "Python kernel_support.py assumes a 512-value row padding; "
              "update _MATRIX_ROW_PADDING and tests/helpers_upstream.py "
              "together with any upstream MATRIX_ROW_PADDING change");
constexpr int64_t kQK8_1 = 32;
// Kernel launch limit, distinct from the architecture-tuned dispatch policy.
constexpr int64_t kMmvqMaxBatchSize = MMVQ_MAX_BATCH_SIZE;

// Stable marker embedded in every "cannot run the upstream MoE kernel"
// error message. kernel_support.py exposes this constant and fused_moe.py
// matches on it to decide whether auto mode may fall back. Do not reword
// the marker without updating both sides.
constexpr const char* kMoeNotEligibleMarker = "VLLM_GGUF_MOE_NOT_ELIGIBLE";

enum class KernelMode { kAuto, kUpstream, kLegacy, kTriton };

KernelMode kernel_mode() {
  const char* value = std::getenv("VLLM_GGUF_CUDA_DENSE_KERNEL");
  if (value == nullptr) {
    value = std::getenv("VLLM_GGUF_CUDA_KERNEL");
  }
  if (value == nullptr || std::string(value) == "auto") {
    return KernelMode::kAuto;
  }
  if (std::string(value) == "upstream") {
    return KernelMode::kUpstream;
  }
  if (std::string(value) == "legacy") {
    return KernelMode::kLegacy;
  }
  if (std::string(value) == "triton") {
    return KernelMode::kTriton;
  }
  throw std::runtime_error(
      "VLLM_GGUF_CUDA_DENSE_KERNEL must be one of auto|upstream|legacy|triton");
}

void check_common_inputs(const Tensor& W, const Tensor& X, int64_t row,
                         const char* op_name) {
  STD_TORCH_CHECK(W.is_cuda() && X.is_cuda(), op_name,
                  ": W and X must be CUDA tensors");
  STD_TORCH_CHECK(W.get_device_index() == X.get_device_index(), op_name,
                  ": W and X must be on the same CUDA device");
  STD_TORCH_CHECK(W.dim() == 2 && X.dim() == 2, op_name,
                  ": W and X must be rank-2 tensors");
  STD_TORCH_CHECK(W.is_contiguous() && X.is_contiguous(), op_name,
                  ": W and X must be contiguous");
  STD_TORCH_CHECK(X.scalar_type() == ScalarType::Float ||
                      X.scalar_type() == ScalarType::Half ||
                      X.scalar_type() == ScalarType::BFloat16,
                  op_name, ": X must have dtype fp32, fp16, or bf16");
  STD_TORCH_CHECK(W.element_size() == 1, op_name,
                  ": W must contain packed byte data");
  STD_TORCH_CHECK(row > 0 && row <= W.size(0), op_name,
                  ": row must be in (0, W.size(0)]");
}

// The full set of quantization types the upstream CUDA kernels handle. Named
// "upstream" (not "upstream MMVQ") because dequantize/MMQ/MoE eligibility all
// derive from it.
bool upstream_mmq_type_supported(int64_t type) {
  switch (type) {
#define GGUF_TYPE_CASE(value) case value:
    GGUF_UPSTREAM_MMQ_TYPES(GGUF_TYPE_CASE)
#undef GGUF_TYPE_CASE
    return true;
    default:
      return false;
  }
}

bool is_upstream_type(int64_t type) {
  return upstream_mmq_type_supported(type) || type == GGML_TYPE_IQ1_M;
}

bool is_legacy_mmvq_type(int64_t type) {
  // Explicit whitelist mirroring kernel_support.py's LEGACY MMVQ set. Do NOT
  // derive this from is_upstream_type by exclusion: a new upstream type
  // would then be wrongly reported as legacy-capable while gguf_kernel.cu's
  // fixed switch has no instance for it.
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
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ1_S:
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_IQ1_M:
      return true;
    default:
      return false;
  }
}

int64_t block_size_for_type(int64_t type, const char* op_name) {
  STD_TORCH_CHECK(is_upstream_type(type), op_name,
                  ": unsupported upstream MMVQ quantization type: ", type);
  return ggml_blck_size(static_cast<ggml_type>(type));
}

size_t type_size_for_type(int64_t type, const char* op_name) {
  STD_TORCH_CHECK(is_upstream_type(type), op_name,
                  ": unsupported upstream MMVQ quantization type: ", type);
  return ggml_type_size(static_cast<ggml_type>(type));
}

size_t storage_padding_bytes(int64_t k, int64_t type, const char* op_name) {
  const int64_t remainder = k % kMatrixRowPadding;
  if (remainder == 0) {
    return 0;
  }
  const int64_t missing = kMatrixRowPadding - remainder;
  const int64_t block_size = block_size_for_type(type, op_name);
  const size_t type_size = type_size_for_type(type, op_name);
  STD_TORCH_CHECK(missing % block_size == 0, op_name,
                  ": upstream row padding is not aligned to a quantization "
                  "block");
  return static_cast<size_t>(missing / block_size) * type_size;
}

bool has_weight_padding(const Tensor& W, int64_t k, int64_t type,
                        const char* op_name) {
  int64_t storage_size = 0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(W.get(), &storage_size));
  const size_t element_size = W.element_size();
  const size_t offset_bytes =
      static_cast<size_t>(W.storage_offset()) * element_size;
  const size_t logical_bytes = static_cast<size_t>(W.numel()) * element_size;
  if (storage_size < 0 || static_cast<uint64_t>(storage_size) < offset_bytes) {
    return false;
  }
  const size_t available_bytes =
      static_cast<size_t>(storage_size) - offset_bytes;
  return available_bytes >=
         logical_bytes + storage_padding_bytes(k, type, op_name);
}

// Shared body of the packed-row -> logical-k derivation. packed_row_bytes is
// the per-row packed byte count (W.size(1) for dense, W.size(2) for MoE); the
// wrappers keep their operation-specific error text.
int64_t logical_k_from_packed_row_bytes(int64_t packed_row_bytes, int64_t type,
                                        const char* op_name,
                                        const char* row_desc) {
  const size_t type_size = type_size_for_type(type, op_name);
  const int64_t block_size = block_size_for_type(type, op_name);
  STD_TORCH_CHECK(packed_row_bytes > 0 && packed_row_bytes % type_size == 0,
                  op_name, ": packed ", row_desc,
                  " size is not a multiple of the quantization type size");
  return packed_row_bytes / static_cast<int64_t>(type_size) * block_size;
}

int64_t logical_k_from_weight(const Tensor& W, int64_t type,
                              const char* op_name) {
  return logical_k_from_packed_row_bytes(W.size(1), type, op_name, "row");
}

int64_t padded_k(int64_t k) {
  return (k + kMatrixRowPadding - 1) / kMatrixRowPadding * kMatrixRowPadding;
}

}  // namespace
