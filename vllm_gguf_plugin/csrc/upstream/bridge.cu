// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
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

#include "mmvq.cuh"
#include "quantize.cuh"
#include "upstream_dispatch.cuh"
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
bool is_upstream_type(int64_t type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q1_0:
    case GGML_TYPE_Q2_0:
    case GGML_TYPE_MXFP4:
    case GGML_TYPE_NVFP4:
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

// The upstream context's stream() helper lazily creates a private
// cudaStreamNonBlocking stream whenever it observes a null slot. The bridge
// must never let that happen: all bridge-owned work (dtype casts, quantize,
// zeroing) runs on the Torch current stream, so upstream kernels launched
// through ctx.stream() have to observe the same underlying stream or reads
// and writes race with no ordering dependency.
//
// Torch's default stream reports a null handle. Upstream treats null as
// "not created", so a raw assignment would silently keep the private-stream
// behavior. Normalize instead to CUDA's special non-null handles, which map
// to the same underlying default stream the null handle denotes.
// setup.py does not compile with --default-stream per-thread, so the legacy
// default-stream semantics apply for this build; keep both mappings in one
// place in case that flag ever changes.
cudaStream_t normalize_borrowed_stream(cudaStream_t stream) {
  return stream != nullptr ? stream : cudaStreamLegacy;
}

// Install the borrowed Torch current stream into the per-call GGML context.
// The context destructor does not destroy borrowed streams (see
// runtime_adapter.cu), and the pool/scratch owner relies on the Torch
// caching allocator's same-stream ordering, so all allocations and kernels
// must stay on this one stream.
cudaStream_t current_stream(int32_t device_index);

void bind_torch_stream_to_context(ggml_backend_cuda_context& context,
                                  int32_t device_index) {
  const cudaStream_t stream = current_stream(device_index);
  context.streams[device_index][0] = stream;
  context.curr_stream_no = 0;
}

cudaStream_t current_stream(int32_t device_index) {
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
  return normalize_borrowed_stream(static_cast<cudaStream_t>(raw_stream));
}

void check_launch(const char* op_name) {
  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, op_name,
                  ": CUDA launch failed: ", cudaGetErrorString(error));
}

// Extra bytes handed out with every pool allocation. The upstream MMQ kernels
// assume the ggml_cuda_pool hands back chunks carved from larger aligned
// blocks, so tile loads can read a few bytes past the requested size (the
// caller-side J_max guard undercounts when src1 is a broadcast view with
// ne11 == 1, which is exactly how the bridge builds the MoE activation
// tensor). Returning exactly `size` bytes made those reads out-of-bounds,
// surfacing as NaN outputs or illegal memory accesses depending on where the
// trailing tile landed. The tail must be zeroed, not merely allocated:
// mul_mat_q reads full J-row tiles whose trailing rows lie past the logical
// data (write-back masks them out by j_max), and garbage scale bytes in that
// tail occasionally poisoned results non-deterministically.
//
// kMmqTileColumnsMax mirrors mul_mat_q_switch_J, which scans J from 8 to 128
// (its tile-column upper bound), so a tail of one row of the largest selected
// tile covers any J_best guard. This is deliberately NOT derived from
// MATRIX_ROW_PADDING (that guards the K dimension, not J). sizeof stays a
// run-time sizeof(block_q8_1_mmq): upstream static-asserts block_fp4_mmq has
// the same size, so one tail formula covers the Q8 and native-FP4 layouts.
constexpr int kMmqTileColumnsMax = 128;
constexpr size_t kPoolGuardTailBytes =
    static_cast<size_t>(kMmqTileColumnsMax) * sizeof(block_q8_1_mmq);

// The pool exposes two entry points with different guard policies:
//
//  - alloc() (the ggml_cuda_pool virtual override): every upstream
//    ggml_cuda_pool_alloc goes through here, and the interface carries only a
//    byte count - no buffer purpose. Upstream callers (mmvq.cu / mmq.cu
//    quantized activations, ids_src1/ids_dst/expert_bounds, NVFP4 src1_scale,
//    stream-k tmp_fixup) have heterogeneous tile-read patterns and their own
//    J_max guard undercounts for broadcast views (ne11 == 1), so the
//    conservative zeroed tail stays. Compatibility-layer policy for the
//    pinned upstream (002a12ad); do not shrink without per-buffer proofs.
//
//  - alloc_workspace(): bridge-owned workspace allocations only (dense
//    buffers). The caller knows the exact layout and already reserves every
//    needed guard explicitly (the Q8 region includes the j_max tile guard in
//    q8_bytes), so no extra tail is appended - protection lives at the
//    correct interior offset instead of being duplicated at the block end.
//    The dense workspace must never contain ids/scale/fixup data; those are
//    allocated by upstream itself via the virtual entry, where the tail
//    still applies.
class TorchScratchPool final : public ggml_cuda_pool {
 public:
  explicit TorchScratchPool(const Tensor& prototype) : prototype_(prototype) {}

  void* alloc(size_t size, size_t* actual_size) override {
    return alloc_impl(std::max<size_t>(size, 1) + kPoolGuardTailBytes,
                      actual_size);
  }

  void* alloc_workspace(size_t bytes, size_t* actual_size) {
    return alloc_impl(bytes, actual_size);
  }

  void free(void* /*ptr*/, size_t /*size*/) override {}

 private:
  void* alloc_impl(size_t bytes, size_t* actual_size) {
    const int64_t int_count = static_cast<int64_t>((bytes + 3) / 4);
    owners_.push_back(torch::stable::new_zeros(
        prototype_, {int_count}, std::optional<ScalarType>(ScalarType::Int)));
    *actual_size = static_cast<size_t>(int_count) * 4;
    return owners_.back().data_ptr();
  }

  const Tensor& prototype_;
  std::vector<Tensor> owners_;
};

struct DenseBuffers {
  Tensor output;
  void* q8;
  const float* input;
  float* result;
};

// Hand the scratch pool to the context's pool slot. Ownership transfers to
// the context's unique_ptr<ggml_cuda_pool>: the pool lives exactly as long
// as the per-call context, which outlives every allocation and kernel in the
// runner (the pool is destroyed only after the context is). This keeps
// ctx.pool() resolving the live object without any double-free risk - the
// runner must not keep a second owning pointer to the same pool.
void install_scratch_pool(ggml_backend_cuda_context& context,
                          int32_t device_index,
                          std::unique_ptr<TorchScratchPool>& pool) {
  context.pools[device_index][0] = std::move(pool);
}

DenseBuffers dense_buffers(const Tensor& X, int64_t output_rows, int64_t row,
                           size_t q8_bytes, TorchScratchPool& scratch_pool,
                           cudaStream_t stream) {
  Tensor output =
      torch::stable::new_empty(X, {output_rows, row}, X.scalar_type());
  const bool cast = X.scalar_type() != ScalarType::Float;
  const auto align = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
  const size_t input_offset = align(q8_bytes);
  const size_t input_bytes = cast ? X.numel() * sizeof(float) : 0;
  const size_t result_offset = input_offset + align(input_bytes);
  const size_t result_bytes = cast ? output_rows * row * sizeof(float) : 0;
  size_t actual_size = 0;
  // fp32 input with no bridge-owned quantized region needs no scratch at all:
  // input comes straight from X and the result goes straight into the output
  // tensor. Skip the allocation instead of handing back a pointer that is
  // only used for pointer arithmetic that never dereferences.
  const bool need_scratch = cast || q8_bytes > 0;
  // All temporary regions have one per-call Torch owner. The workspace entry
  // adds no tail: every guard the layout needs is already part of q8_bytes.
  // 256-byte alignment preserves the upstream quantizer's vector loads and
  // graph/stream safety.
  char* scratch = nullptr;
  if (need_scratch) {
    scratch = static_cast<char*>(scratch_pool.alloc_workspace(
        result_offset + result_bytes, &actual_size));
  }
  const float* input = cast ? reinterpret_cast<float*>(scratch + input_offset)
                            : static_cast<const float*>(X.data_ptr());
  float* result = cast ? reinterpret_cast<float*>(scratch + result_offset)
                       : static_cast<float*>(output.data_ptr());
  if (cast) {
    if (X.scalar_type() == ScalarType::Half) {
      gguf_cast_async<float, half>(scratch + input_offset, X.data_ptr(),
                                   X.numel(), stream);
    } else {
      gguf_cast_async<float, __nv_bfloat16>(scratch + input_offset,
                                            X.data_ptr(), X.numel(), stream);
    }
  }
  return {output, scratch, input, result};
}

Tensor finish_output(const DenseBuffers& buffers, cudaStream_t stream) {
  const auto dtype = buffers.output.scalar_type();
  if (dtype == ScalarType::Half) {
    gguf_cast_async<half, float>(buffers.output.data_ptr(), buffers.result,
                                 buffers.output.numel(), stream);
  } else if (dtype == ScalarType::BFloat16) {
    gguf_cast_async<__nv_bfloat16, float>(buffers.output.data_ptr(),
                                          buffers.result,
                                          buffers.output.numel(), stream);
  }
  check_launch("upstream dense output");
  return buffers.output;
}

ggml_tensor make_quant_tensor(const Tensor& W, int64_t type, int64_t k,
                              int64_t row) {
  ggml_tensor tensor{};
  tensor.type = static_cast<ggml_type>(type);
  tensor.ne[0] = k;
  tensor.ne[1] = row;
  tensor.ne[2] = 1;
  tensor.ne[3] = 1;
  tensor.nb[0] = type_size_for_type(type, "upstream MMVQ");
  tensor.nb[1] = static_cast<size_t>(W.size(1));
  tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(row);
  tensor.nb[3] = tensor.nb[2];
  tensor.data = W.data_ptr();
  return tensor;
}

// Contiguous two-dimensional F32 descriptor shared by the activation (ne[0]
// = row length) and output (ne[0] = row count) constructions; the thin
// wrappers keep their call-site semantics readable.
ggml_tensor make_f32_tensor_2d(const void* data, int64_t ne0, int64_t ne1) {
  ggml_tensor tensor{};
  tensor.type = GGML_TYPE_F32;
  tensor.ne[0] = ne0;
  tensor.ne[1] = ne1;
  tensor.ne[2] = 1;
  tensor.ne[3] = 1;
  tensor.nb[0] = sizeof(float);
  tensor.nb[1] = static_cast<size_t>(ne0) * sizeof(float);
  tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(ne1);
  tensor.nb[3] = tensor.nb[2];
  tensor.data = const_cast<void*>(data);
  return tensor;
}

ggml_tensor make_f32_tensor(const float* data, int64_t k, int64_t batch) {
  return make_f32_tensor_2d(data, k, batch);
}

ggml_tensor make_output_tensor(float* data, int64_t row, int64_t batch) {
  return make_f32_tensor_2d(data, row, batch);
}

ggml_tensor make_moe_weight_tensor(const Tensor& W, int64_t type, int64_t k,
                                   int64_t row) {
  ggml_tensor tensor{};
  tensor.type = static_cast<ggml_type>(type);
  tensor.ne[0] = k;
  tensor.ne[1] = row;
  tensor.ne[2] = W.size(0);
  tensor.ne[3] = 1;
  tensor.nb[0] = type_size_for_type(type, "upstream MoE");
  tensor.nb[1] = static_cast<size_t>(W.size(2));
  tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(W.size(1));
  tensor.nb[3] = tensor.nb[2] * static_cast<size_t>(W.size(0));
  tensor.data = W.data_ptr();
  return tensor;
}

ggml_tensor make_moe_input_tensor(const float* data, int64_t k,
                                  int64_t tokens) {
  ggml_tensor tensor{};
  tensor.type = GGML_TYPE_F32;
  tensor.ne[0] = k;
  tensor.ne[1] = 1;
  tensor.ne[2] = tokens;
  tensor.ne[3] = 1;
  tensor.nb[0] = sizeof(float);
  tensor.nb[1] = static_cast<size_t>(k) * sizeof(float);
  tensor.nb[2] = tensor.nb[1];
  tensor.nb[3] = tensor.nb[2] * static_cast<size_t>(tokens);
  tensor.data = const_cast<float*>(data);
  return tensor;
}

ggml_tensor make_moe_ids_tensor(const int32_t* data, int64_t top_k,
                                int64_t tokens) {
  ggml_tensor tensor{};
  tensor.type = GGML_TYPE_I32;
  tensor.ne[0] = top_k;
  tensor.ne[1] = tokens;
  tensor.ne[2] = 1;
  tensor.ne[3] = 1;
  tensor.nb[0] = sizeof(int32_t);
  tensor.nb[1] = static_cast<size_t>(top_k) * sizeof(int32_t);
  tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(tokens);
  tensor.nb[3] = tensor.nb[2];
  tensor.data = const_cast<int32_t*>(data);
  return tensor;
}

ggml_tensor make_moe_output_tensor(float* data, int64_t row, int64_t top_k,
                                   int64_t tokens) {
  ggml_tensor tensor{};
  tensor.type = GGML_TYPE_F32;
  tensor.ne[0] = row;
  tensor.ne[1] = top_k;
  tensor.ne[2] = tokens;
  tensor.ne[3] = 1;
  tensor.nb[0] = sizeof(float);
  tensor.nb[1] = static_cast<size_t>(row) * sizeof(float);
  tensor.nb[2] = tensor.nb[1] * static_cast<size_t>(top_k);
  tensor.nb[3] = tensor.nb[2] * static_cast<size_t>(tokens);
  tensor.data = data;
  return tensor;
}

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

bool is_upstream_mmq_type(int64_t type);

int64_t logical_k_from_moe_weight(const Tensor& W, int64_t type,
                                  const char* op_name) {
  return logical_k_from_packed_row_bytes(W.size(2), type, op_name,
                                         "expert row");
}

Tensor run_upstream_moe_projection(const Tensor& W, const Tensor& X,
                                   const Tensor& topk_ids, int64_t type,
                                   int64_t row, int64_t top_k, int64_t tokens,
                                   int64_t k) {
  const int32_t device_index = X.get_device_index();
  const DeviceGuard device_guard(device_index);
  const cudaStream_t stream = current_stream(device_index);
  const int64_t output_rows = tokens * top_k;

  ggml_backend_cuda_context context(device_index);
  bind_torch_stream_to_context(context, device_index);
  auto scratch_pool = std::make_unique<TorchScratchPool>(X);
  TorchScratchPool& pool_ref = *scratch_pool;
  install_scratch_pool(context, device_index, scratch_pool);
  const DenseBuffers buffers =
      dense_buffers(X, output_rows, row, 0, pool_ref, stream);

  ggml_tensor src0 = make_moe_weight_tensor(W, type, k, row);
  ggml_tensor src1 = make_moe_input_tensor(buffers.input, k, tokens);
  ggml_tensor ids_tensor = make_moe_ids_tensor(
      static_cast<const int32_t*>(topk_ids.data_ptr()), top_k, tokens);
  ggml_tensor dst = make_moe_output_tensor(buffers.result, row, top_k, tokens);

  const int cc = ggml_cuda_info().devices[device_index].cc;
  const int mmvq_max =
      get_mmvq_mmid_max_batch(static_cast<ggml_type>(type), cc);
  if (tokens <= kMmvqMaxBatchSize && tokens <= mmvq_max) {
    ggml_cuda_mul_mat_vec_q(context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMVQ");
  } else {
    const bool fallback = row % 128 != 0;
    const int j_max = ggml_cuda_mmq_get_J_max(static_cast<ggml_type>(type),
                                              fallback, cc, tokens);
    STD_TORCH_CHECK(is_upstream_mmq_type(type) && j_max > 0,
                    kMoeNotEligibleMarker,
                    ": no MMQ "
                    "configuration for type/shape/device");
    ggml_cuda_mul_mat_q(context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMQ");
  }
  return finish_output(buffers, stream);
}

Tensor run_upstream_mmvq(const Tensor& W, const Tensor& X, int64_t type,
                         int64_t row, int64_t k) {
  const int32_t device_index = X.get_device_index();
  const DeviceGuard device_guard(device_index);
  const cudaStream_t stream = current_stream(device_index);
  const int64_t batch = X.size(0);
  const int64_t k_padded = padded_k(k);

  ggml_backend_cuda_context context(device_index);
  bind_torch_stream_to_context(context, device_index);
  auto scratch_pool = std::make_unique<TorchScratchPool>(X);
  TorchScratchPool& pool_ref = *scratch_pool;
  install_scratch_pool(context, device_index, scratch_pool);
  const size_t q8_bytes = static_cast<size_t>(batch) *
                          static_cast<size_t>(k_padded) * sizeof(block_q8_1) /
                          kQK8_1;
  const DenseBuffers buffers =
      dense_buffers(X, X.size(0), row, q8_bytes, pool_ref, stream);
  void* q8_data = buffers.q8;

  quantize_row_q8_1_cuda(buffers.input, nullptr, q8_data,
                         static_cast<ggml_type>(type), k, k, 0, 0, k_padded,
                         batch, 1, 1, stream);

  ggml_tensor src0 = make_quant_tensor(W, type, k, row);
  ggml_tensor src1 = make_f32_tensor(buffers.input, k_padded, batch);
  ggml_tensor dst = make_output_tensor(buffers.result, row, batch);
  ggml_cuda_op_mul_mat_vec_q(context, &src0, &src1, &dst,
                             static_cast<const char*>(W.data_ptr()),
                             buffers.input, static_cast<const char*>(q8_data),
                             buffers.result, 0, row, batch, k_padded, stream);
  check_launch("upstream MMVQ");
  return finish_output(buffers, stream);
}

Tensor run_upstream_mmq(const Tensor& W, const Tensor& X, int64_t type,
                        int64_t row, int64_t k) {
  const int32_t device_index = X.get_device_index();
  const DeviceGuard device_guard(device_index);
  const cudaStream_t stream = current_stream(device_index);
  const int64_t batch = X.size(0);
  const int64_t k_padded = padded_k(k);
  const int cc = ggml_cuda_info().devices[device_index].cc;
  const bool fallback = row % 128 != 0;
  const int j_max = ggml_cuda_mmq_get_J_max(static_cast<ggml_type>(type),
                                            fallback, cc, batch);
  if (j_max <= 0 && batch <= kMmvqMaxBatchSize) {
    // MMVQ is still safe when MMQ has no configuration, even if upstream's
    // performance policy would normally prefer MMQ on this architecture.
    return run_upstream_mmvq(W, X, type, row, k);
  }
  STD_TORCH_CHECK(
      j_max > 0,
      "ggml_mul_mat_a8: no upstream MMQ configuration for type/shape/device");

  // Native FP4 (Blackwell MMA path): upstream swaps the Q8_1_MMQ activation
  // format for block_fp4_mmq and needs a separate per-column scale buffer for
  // NVFP4, with different block sizes, strides and kernel-side ne_block. The
  // bridge's merged Q8 workspace cannot express that layout, so for these
  // types hand the whole operator to the upstream wrapper: build plain F32
  // descriptors for src1/dst and let ggml_cuda_mul_mat_q quantize, allocate
  // (via the installed TorchScratchPool), scale and launch on ctx.stream()
  // itself. Describing src1 with the logical k (never k_padded, which would
  // claim padding we did not allocate) keeps upstream's own
  // MATRIX_ROW_PADDING handling authoritative.
  const bool native_fp4 = blackwell_mma_available(cc) &&
                          (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
  if (native_fp4) {
    ggml_backend_cuda_context context(device_index);
    bind_torch_stream_to_context(context, device_index);
    auto scratch_pool = std::make_unique<TorchScratchPool>(X);
    TorchScratchPool& pool_ref = *scratch_pool;
    install_scratch_pool(context, device_index, scratch_pool);
    // No bridge quantized workspace is needed; output conversion still runs
    // through the shared dense-buffers conversion path (zero q8 region).
    const DenseBuffers buffers =
        dense_buffers(X, X.size(0), row, 0, pool_ref, stream);
    ggml_tensor src0 = make_quant_tensor(W, type, k, row);
    ggml_tensor src1 = make_f32_tensor(buffers.input, k, batch);
    ggml_tensor dst = make_output_tensor(buffers.result, row, batch);
    ggml_cuda_mul_mat_q(context, &src0, &src1, /*ids=*/nullptr, &dst);
    check_launch("upstream MMQ (native FP4)");
    return finish_output(buffers, stream);
  }

  ggml_backend_cuda_context context(device_index);
  bind_torch_stream_to_context(context, device_index);
  auto scratch_pool = std::make_unique<TorchScratchPool>(X);
  TorchScratchPool& pool_ref = *scratch_pool;
  install_scratch_pool(context, device_index, scratch_pool);
  const size_t q8_bytes = static_cast<size_t>(batch) *
                              static_cast<size_t>(k_padded) *
                              sizeof(block_q8_1_mmq) / QK8_1_MMQ +
                          static_cast<size_t>(j_max) * sizeof(block_q8_1_mmq);
  const DenseBuffers buffers =
      dense_buffers(X, X.size(0), row, q8_bytes, pool_ref, stream);
  void* q8_data = buffers.q8;

  quantize_mmq_q8_1_cuda(buffers.input, nullptr, q8_data,
                         static_cast<ggml_type>(type), k, k, 0, 0, k_padded,
                         batch, 1, 1, stream);

  // Named-field initialization (C++17: no designated initializers) so an
  // upstream mmq_args field addition fails to compile here instead of
  // silently shifting all subsequent positional values. Every field is
  // annotated with its source; stride_channel/sample entries use 1 because
  // the bridge builds a single-channel, single-sample dense descriptor.
  mmq_args args{};
  args.x = static_cast<const char*>(W.data_ptr());
  args.type_x = static_cast<ggml_type>(type);
  args.y = static_cast<const int*>(q8_data);
  args.ids_dst = nullptr;        // dense path: no expert routing
  args.expert_bounds = nullptr;  // dense path: no expert bounds
  args.dst = buffers.result;
  args.y_scale = nullptr;  // Q8 activation path: no NVFP4 scale
  args.ncols_x = k;
  args.nrows_x = row;
  args.ncols_dst = batch;
  args.stride_row_x = static_cast<int64_t>(
      W.size(1) / type_size_for_type(type, "upstream MMQ"));
  args.ncols_y = batch;
  args.nrows_dst = row;
  args.nchannels_x = 1;
  args.nchannels_y = 1;
  args.stride_channel_x = 1;
  args.stride_channel_y = 1;
  args.stride_channel_dst = 1;
  args.nsamples_x = 1;
  args.nsamples_y = 1;
  args.stride_sample_x = 1;
  args.stride_sample_y = 1;
  args.stride_sample_dst = 1;
  args.ncols_max = batch;
  args.ncols_opt = batch;
  ggml_upstream_mul_mat_q(context, args, stream);
  check_launch("upstream MMQ");
  return finish_output(buffers, stream);
}

template <typename scalar_t>
void run_upstream_dequantize(const Tensor& W, Tensor& output, int64_t type,
                             int64_t total, cudaStream_t stream) {
  auto to_cuda = [&]() {
    if constexpr (std::is_same_v<scalar_t, float>) {
      return ggml_get_to_fp32_cuda(static_cast<ggml_type>(type));
    } else if constexpr (std::is_same_v<scalar_t, half>) {
      return ggml_get_to_fp16_cuda(static_cast<ggml_type>(type));
    } else {
      return ggml_get_to_bf16_cuda(static_cast<ggml_type>(type));
    }
  }();
  STD_TORCH_CHECK(to_cuda != nullptr,
                  "ggml_dequantize_upstream: no upstream dequantize kernel "
                  "for quantization type ",
                  type);
  to_cuda(W.data_ptr(), static_cast<scalar_t*>(output.data_ptr()), total,
          stream);
}

bool upstream_mmvq_runnable(const Tensor& W, const Tensor& X, int64_t type,
                            int64_t k) {
  return is_upstream_type(type) && X.size(0) <= kMmvqMaxBatchSize &&
         has_weight_padding(W, k, type, "ggml_mul_mat_vec_a8");
}

bool is_upstream_mmq_type(int64_t type) {
  return is_upstream_type(type) && type != GGML_TYPE_IQ1_M;
}

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

bool upstream_mmq_eligible(const Tensor& W, int64_t type, int64_t k) {
  return is_upstream_mmq_type(type) &&
         has_weight_padding(W, k, type, "ggml_mul_mat_a8");
}

Tensor run_upstream_mmvq_or_mmq(const Tensor& W, const Tensor& X, int64_t type,
                                int64_t row, int64_t k) {
  const int cc = ggml_cuda_info().devices[X.get_device_index()].cc;
  if (!ggml_cuda_should_use_mmvq(static_cast<ggml_type>(type), cc, X.size(0)) &&
      is_upstream_mmq_type(type)) {
    return run_upstream_mmq(W, X, type, row, k);
  }
  return run_upstream_mmvq(W, X, type, row, k);
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

Tensor ggml_dequantize_upstream(Tensor W, int64_t type, int64_t m, int64_t n,
                                std::optional<ScalarType> dtype) {
  STD_TORCH_CHECK(W.is_cuda(),
                  "ggml_dequantize_upstream: W must be a CUDA tensor");
  STD_TORCH_CHECK(W.dim() == 2 && W.is_contiguous(),
                  "ggml_dequantize_upstream: W must be a contiguous rank-2 "
                  "tensor");
  STD_TORCH_CHECK(W.element_size() == 1,
                  "ggml_dequantize_upstream: W must contain packed byte data");
  STD_TORCH_CHECK(m >= 0 && n >= 0,
                  "ggml_dequantize_upstream: output dimensions must be "
                  "non-negative");
  STD_TORCH_CHECK(m == 0 || n <= std::numeric_limits<int64_t>::max() / m,
                  "ggml_dequantize_upstream: output dimensions overflow");
  STD_TORCH_CHECK(is_upstream_type(type),
                  "ggml_dequantize_upstream: unsupported quantization type ",
                  type);

  const int64_t total = m * n;
  const auto quant_type = static_cast<ggml_type>(type);
  const int64_t block_size = ggml_blck_size(quant_type);
  const size_t type_size = ggml_type_size(quant_type);
  STD_TORCH_CHECK(n == 0 || n % block_size == 0,
                  "ggml_dequantize_upstream: n must be aligned to the "
                  "quantization block size");
  STD_TORCH_CHECK(
      W.size(1) == (n / block_size) * static_cast<int64_t>(type_size),
      "ggml_dequantize_upstream: packed row size does not match n and "
      "quantization type");
  if (quant_type == GGML_TYPE_MXFP4) {
    // convert.cu's MXFP4 row kernel consumes QK_K (256) values per launch.
    STD_TORCH_CHECK(
        n == 0 || n % 256 == 0,
        "ggml_dequantize_upstream: MXFP4 n must be aligned to 256 values");
  }
  STD_TORCH_CHECK(total == 0 || total % block_size == 0,
                  "ggml_dequantize_upstream: output element count must be "
                  "aligned to the quantization block size");
  // The convert kernels read rows[0..m) from the packed weight, so reject a
  // row count the input cannot back before launching anything. Callers may
  // legally decode a prefix (m < W.size(0)); they may not ask for more rows
  // than the input provides.
  STD_TORCH_CHECK(m <= W.size(0),
                  "ggml_dequantize_upstream: requested row count ", m,
                  " exceeds the packed input capacity ", W.size(0));

  // Validate the dtype before allocating so an unsupported request fails
  // deterministically regardless of the output shape (an empty output used
  // to return before dtype dispatch ever ran).
  const auto dtype_ = dtype.value_or(ScalarType::Half);
  STD_TORCH_CHECK(dtype_ == ScalarType::Float || dtype_ == ScalarType::Half ||
                      dtype_ == ScalarType::BFloat16,
                  "ggml_dequantize_upstream: output dtype must be fp32, fp16, "
                  "or bf16");

  Tensor output = torch::stable::new_empty(W, {m, n}, dtype_);
  if (total == 0) {
    return output;
  }

  const int32_t device_idx = W.get_device_index();
  const DeviceGuard device_guard(device_idx);
  cudaStream_t stream = current_stream(device_idx);
  if (dtype_ == ScalarType::Float) {
    run_upstream_dequantize<float>(W, output, type, total, stream);
  } else if (dtype_ == ScalarType::Half) {
    run_upstream_dequantize<half>(W, output, type, total, stream);
  } else {
    run_upstream_dequantize<nv_bfloat16>(W, output, type, total, stream);
  }
  check_launch("upstream dequantize");
  return output;
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
  return run_upstream_moe_projection(W, X, topk_ids, type, row, top_k, tokens,
                                     k);
}

Tensor ggml_mul_mat_vec_a8(Tensor W, Tensor X, int64_t type, int64_t row) {
  check_common_inputs(W, X, row, "ggml_mul_mat_vec_a8");
  if (X.size(0) == 0) {
    return torch::stable::new_empty(W, {0, row}, X.scalar_type());
  }
  const KernelMode mode = kernel_mode();
  if (mode == KernelMode::kTriton) {
    throw std::runtime_error(
        "ggml_mul_mat_vec_a8: Triton backend selected; use the Python "
        "dispatcher");
  }
  if (mode == KernelMode::kLegacy) {
    STD_TORCH_CHECK(
        is_legacy_mmvq_type(type),
        "ggml_mul_mat_vec_a8: no legacy MMVQ kernel for quantization type ",
        type);
    return ggml_mul_mat_vec_a8_legacy(W, X, type, row);
  }
  if (mode == KernelMode::kUpstream) {
    STD_TORCH_CHECK(
        is_upstream_type(type),
        "ggml_mul_mat_vec_a8: no upstream MMVQ kernel for quantization type ",
        type);
    const int64_t k = logical_k_from_weight(W, type, "ggml_mul_mat_vec_a8");
    STD_TORCH_CHECK(X.size(1) == k,
                    "ggml_mul_mat_vec_a8: X K dimension does not match W");
    STD_TORCH_CHECK(upstream_mmvq_runnable(W, X, type, k),
                    "ggml_mul_mat_vec_a8: upstream MMVQ requires batch <= ",
                    kMmvqMaxBatchSize,
                    " and "
                    "MATRIX_ROW_PADDING storage");
    return run_upstream_mmvq_or_mmq(W, X, type, row, k);
  }

  if (is_upstream_type(type)) {
    const int64_t k = logical_k_from_weight(W, type, "ggml_mul_mat_vec_a8");
    STD_TORCH_CHECK(X.size(1) == k,
                    "ggml_mul_mat_vec_a8: X K dimension does not match W");
    if (upstream_mmvq_runnable(W, X, type, k)) {
      return run_upstream_mmvq_or_mmq(W, X, type, row, k);
    }
  }
  if (is_legacy_mmvq_type(type)) {
    return ggml_mul_mat_vec_a8_legacy(W, X, type, row);
  }
  throw std::runtime_error(
      "ggml_mul_mat_vec_a8: no eligible CUDA kernel for quantization type " +
      std::to_string(type));
}

Tensor ggml_mul_mat_a8(Tensor W, Tensor X, int64_t type, int64_t row) {
  check_common_inputs(W, X, row, "ggml_mul_mat_a8");
  if (X.size(0) == 0) {
    return torch::stable::new_empty(W, {0, row}, X.scalar_type());
  }
  const KernelMode mode = kernel_mode();
  if (mode == KernelMode::kTriton) {
    throw std::runtime_error(
        "ggml_mul_mat_a8: Triton backend selected; use the Python "
        "dispatcher");
  }
  if (mode == KernelMode::kLegacy) {
    STD_TORCH_CHECK(
        is_legacy_mmq_type(type),
        "ggml_mul_mat_a8: no legacy MMQ kernel for quantization type ", type);
    return ggml_mul_mat_a8_legacy(W, X, type, row);
  }
  if (mode == KernelMode::kUpstream) {
    STD_TORCH_CHECK(
        is_upstream_mmq_type(type),
        "ggml_mul_mat_a8: no upstream MMQ kernel for quantization type ", type);
    const int64_t k = logical_k_from_weight(W, type, "ggml_mul_mat_a8");
    STD_TORCH_CHECK(X.size(1) == k,
                    "ggml_mul_mat_a8: X K dimension does not match W");
    STD_TORCH_CHECK(
        upstream_mmq_eligible(W, type, k),
        "ggml_mul_mat_a8: upstream MMQ requires MATRIX_ROW_PADDING storage");
    return run_upstream_mmq(W, X, type, row, k);
  }

  if (is_upstream_mmq_type(type)) {
    const int64_t k = logical_k_from_weight(W, type, "ggml_mul_mat_a8");
    STD_TORCH_CHECK(X.size(1) == k,
                    "ggml_mul_mat_a8: X K dimension does not match W");
    if (upstream_mmq_eligible(W, type, k)) {
      return run_upstream_mmq(W, X, type, row, k);
    }
  }
  if (is_legacy_mmq_type(type)) {
    return ggml_mul_mat_a8_legacy(W, X, type, row);
  }
  throw std::runtime_error(
      "ggml_mul_mat_a8: no eligible CUDA kernel for quantization type " +
      std::to_string(type));
}
