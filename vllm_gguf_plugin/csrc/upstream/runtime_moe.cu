// SPDX-License-Identifier: Apache-2.0
#include "ggml_dypes.cuh"
#include "torch_context.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"
#include "mmvf.cuh"
#include "mmf.cuh"

#include <algorithm>
#include <climits>
#include <limits>
#include <torch/csrc/stable/ops.h>

int64_t ggml_dense_upstream_capabilities(Tensor W, Tensor X, int64_t type,
                                         int64_t row);
Tensor ggml_dense_mmvq(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_mmq(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_dense_blas(Tensor W, Tensor X, int64_t type, int64_t row);

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

template <typename T>
__global__ void gather_moe_routes(T* sorted, const T* input,
                                  const int32_t* routes, int64_t total,
                                  int64_t top_k, int64_t k) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total * k) {
    const int64_t route = index / k;
    sorted[index] = input[(routes[route] / top_k) * k + index % k];
  }
}

template <typename T>
__global__ void scatter_moe_routes(T* output, const T* sorted,
                                   const int32_t* routes, int64_t total,
                                   int64_t row) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total * row) {
    output[static_cast<int64_t>(routes[index / row]) * row + index % row] =
        sorted[index];
  }
}

Tensor run_grouped_dense(const Tensor& W, const Tensor& X, int64_t type,
                         int64_t row) {
  const int64_t caps = ggml_dense_upstream_capabilities(W, X, type, row);
  if (X.size(0) >= MMQ_DP4A_MAX_BATCH_SIZE && (caps & 16)) {
    return ggml_dense_blas(W, X, type, row);
  }
  // Match the dense dispatch order used by the public Python selector.
  if (caps & 4) {
    return ggml_dense_mmvq(W, X, type, row);
  }
  if (caps & 8) {
    return ggml_dense_mmq(W, X, type, row);
  }
  STD_TORCH_CHECK(caps & 16, kMoeNotEligibleMarker,
                  ": grouped expert has no dense CUDA route");
  return ggml_dense_blas(W, X, type, row);
}

Tensor run_upstream_moe_grouped(const Tensor& W, const Tensor& X,
                                const Tensor& topk_ids, int64_t type,
                                int64_t row, int64_t top_k, int64_t tokens,
                                int64_t k, cudaStream_t stream) {
  const int64_t total = tokens * top_k;
  STD_TORCH_CHECK(total <= INT_MAX, kMoeNotEligibleMarker,
                  ": too many routed tokens for int32 indices");
  const int64_t experts = W.size(0);
  std::vector<int32_t> ids(total);
  STD_TORCH_CHECK(
      cudaMemcpyAsync(ids.data(), topk_ids.data_ptr(), total * sizeof(int32_t),
                      cudaMemcpyDeviceToHost, stream) == cudaSuccess,
      "upstream MoE could not read expert indices");
  STD_TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
                  "upstream MoE could not synchronize expert indices");

  std::vector<std::vector<int32_t>> by_expert(experts);
  for (int64_t route = 0; route < total; ++route) {
    STD_TORCH_CHECK(ids[route] >= 0 && ids[route] < experts,
                    "upstream MoE expert index out of range");
    by_expert[ids[route]].push_back(static_cast<int32_t>(route));
  }
  std::vector<int32_t> routes;
  routes.reserve(total);
  for (const auto& expert_routes : by_expert) {
    routes.insert(routes.end(), expert_routes.begin(), expert_routes.end());
  }
  Tensor device_routes =
      torch::stable::new_empty(topk_ids, {total}, ScalarType::Int);
  STD_TORCH_CHECK(
      cudaMemcpyAsync(device_routes.data_ptr(), routes.data(),
                      total * sizeof(int32_t), cudaMemcpyHostToDevice,
                      stream) == cudaSuccess,
      "upstream MoE could not upload sorted expert indices");
  STD_TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
                  "upstream MoE could not finish uploading expert indices");

  Tensor sorted_input =
      torch::stable::new_empty(X, {total, k}, X.scalar_type());
  const int input_grid = static_cast<int>((total * k + 255) / 256);
  if (X.element_size() == sizeof(float)) {
    gather_moe_routes<<<input_grid, 256, 0, stream>>>(
        static_cast<uint32_t*>(sorted_input.data_ptr()),
        static_cast<const uint32_t*>(X.data_ptr()),
        static_cast<const int32_t*>(device_routes.data_ptr()), total, top_k, k);
  } else {
    gather_moe_routes<<<input_grid, 256, 0, stream>>>(
        static_cast<uint16_t*>(sorted_input.data_ptr()),
        static_cast<const uint16_t*>(X.data_ptr()),
        static_cast<const int32_t*>(device_routes.data_ptr()), total, top_k, k);
  }
  check_launch("upstream MoE gather");

  Tensor sorted_output =
      torch::stable::new_empty(X, {total, row}, X.scalar_type());
  int64_t first = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    const int64_t count = by_expert[expert].size();
    if (count == 0) {
      continue;
    }
    Tensor expert_weight = torch::stable::select(W, 0, expert);
    Tensor expert_input = torch::stable::narrow(sorted_input, 0, first, count);
    Tensor expert_output =
        run_grouped_dense(expert_weight, expert_input, type, row);
    auto* destination = static_cast<char*>(sorted_output.data_ptr()) +
                        first * row * X.element_size();
    STD_TORCH_CHECK(
        cudaMemcpyAsync(destination, expert_output.data_ptr(),
                        count * row * X.element_size(),
                        cudaMemcpyDeviceToDevice, stream) == cudaSuccess,
        "upstream MoE could not copy expert output");
    first += count;
  }

  Tensor output = torch::stable::new_empty(X, {total, row}, X.scalar_type());
  const int output_grid = static_cast<int>((total * row + 255) / 256);
  if (X.element_size() == sizeof(float)) {
    scatter_moe_routes<<<output_grid, 256, 0, stream>>>(
        static_cast<uint32_t*>(output.data_ptr()),
        static_cast<const uint32_t*>(sorted_output.data_ptr()),
        static_cast<const int32_t*>(device_routes.data_ptr()), total, row);
  } else {
    scatter_moe_routes<<<output_grid, 256, 0, stream>>>(
        static_cast<uint16_t*>(output.data_ptr()),
        static_cast<const uint16_t*>(sorted_output.data_ptr()),
        static_cast<const int32_t*>(device_routes.data_ptr()), total, row);
  }
  check_launch("upstream MoE scatter");
  return output;
}

// mmid.cu stores one 4-byte mm_ids_helper_store per token in dynamic shared
// memory. Its token index also has a 22-bit limit.
int64_t mmid_max_tokens(size_t smpbo) {
  return static_cast<int64_t>(
      std::min<size_t>(smpbo / sizeof(int32_t), (1u << 22) - 1));
}

enum class MoeKernel {
  kMmvq,
  kMmq,
  kMmvf,
  kMmf,
  kMmfChunks,
};

constexpr int64_t kMmvqMaxRoutes = 1024;
constexpr int64_t kGroupedTokenThreshold = 8192;

MoeKernel select_moe_kernel(const Tensor& W, int64_t type, int64_t row,
                            int64_t top_k, int64_t tokens, int cc,
                            int warp_size, size_t smpbo, int64_t* chunk_size) {
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
    const int64_t max_tokens = std::min(tokens, mmid_max_tokens(smpbo));
    const auto mmf_eligible = [&](int64_t count) {
      return ggml_cuda_should_use_mmf(ggml_type, cc, warp_size, weight.ne,
                                      weight.nb, count, /*mul_mat_id=*/true);
    };
    if (tokens <= max_tokens && mmf_eligible(tokens)) {
      return MoeKernel::kMmf;
    }
    if (max_tokens > 0 && mmf_eligible(1)) {
      // The upstream MoE MMF predicate has a monotone token limit. Find its
      // largest safe batch without duplicating the upstream shape thresholds.
      int64_t low = 1;
      int64_t high = max_tokens;
      while (low < high) {
        const int64_t mid = low + (high - low + 1) / 2;
        if (mmf_eligible(mid)) {
          low = mid;
        } else {
          high = mid - 1;
        }
      }
      *chunk_size = low;
      return MoeKernel::kMmfChunks;
    }
    STD_TORCH_CHECK(false, kMoeNotEligibleMarker,
                    ": neither MMVF nor MMF supports this type/shape/device");
  }

  const int64_t mmvq_max = std::min<int>(
      MMVQ_MAX_BATCH_SIZE, get_mmvq_mmid_max_batch(ggml_type, cc));
  const int64_t mmq_max = mmid_max_tokens(smpbo);
  // Route count is the common workload measure for quantized MoE calls.
  // The caller already checked tokens * top_k for int64 overflow.
  if (tokens * top_k <= kMmvqMaxRoutes && mmvq_max > 0) {
    *chunk_size = mmvq_max;
    return MoeKernel::kMmvq;
  }
  // Kernel availability and shared-memory limits still take precedence over
  // the performance threshold.
  const bool fallback = row % 128 != 0;
  if (upstream_mmq_type_supported(type) && smpbo >= 48 * 1024 && mmq_max > 0 &&
      ggml_cuda_mmq_get_J_max(ggml_type, fallback, cc,
                              std::min(tokens, mmq_max)) > 0) {
    *chunk_size = mmq_max;
    return MoeKernel::kMmq;
  }
  if (mmvq_max > 0) {
    *chunk_size = mmvq_max;
    return MoeKernel::kMmvq;
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
  // Grouped dense sorts expert IDs on the host and cannot run during CUDA
  // graph capture. Check the dense BLAS shape constraints before selecting it.
  const bool grouped_eligible =
      !is_upstream_float_type(type) && tokens > kGroupedTokenThreshold &&
      tokens * top_k <= INT_MAX && row <= INT_MAX && k <= INT_MAX &&
      (type != GGML_TYPE_MXFP4 || k % 256 == 0);
  if (grouped_eligible) {
    const cudaStream_t stream = current_stream(device_index);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    STD_TORCH_CHECK(
        cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess,
        "upstream MoE could not query CUDA graph capture status");
    if (capture_status == cudaStreamCaptureStatusNone) {
      return run_upstream_moe_grouped(W, X, topk_ids, type, row, top_k, tokens,
                                      k, stream);
    }
  }
  int64_t chunk_size = 0;
  const MoeKernel kernel =
      select_moe_kernel(W, type, row, top_k, tokens, device.cc,
                        device.warp_size, device.smpbo, &chunk_size);

  UpstreamCall call(X);
  const int64_t output_rows = tokens * top_k;
  const ProjectionBuffers buffers = projection_buffers(
      X, output_rows, row, 0, *call.scratch_pool, call.stream);

  ggml_tensor src0 = kernel == MoeKernel::kMmvf || kernel == MoeKernel::kMmf ||
                             kernel == MoeKernel::kMmfChunks
                         ? make_moe_float_weight_tensor(W, type)
                         : make_moe_weight_tensor(W, type, k, row);
  ggml_tensor src1 = make_moe_input_tensor(buffers.input, k, tokens);
  ggml_tensor ids_tensor = make_moe_ids_tensor(
      static_cast<const int32_t*>(topk_ids.data_ptr()), top_k, tokens);
  ggml_tensor dst = make_moe_output_tensor(buffers.result, row, top_k, tokens);

  if (kernel == MoeKernel::kMmvq && tokens <= chunk_size) {
    ggml_cuda_mul_mat_vec_q(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMVQ");
  } else if (kernel == MoeKernel::kMmq && tokens <= chunk_size) {
    ggml_cuda_mul_mat_q(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMQ");
  } else if (kernel == MoeKernel::kMmvf) {
    ggml_cuda_mul_mat_vec_f(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMVF");
  } else if (kernel == MoeKernel::kMmf) {
    ggml_cuda_mul_mat_f(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMF");
  } else {
    // Each chunk uses the selected kernel's token limit and writes directly
    // into its slice of the shared projection output.
    const bool chunked_mmq = kernel == MoeKernel::kMmq;
    for (int64_t start = 0; start < tokens;) {
      int64_t count = std::min<int64_t>(chunk_size, tokens - start);
      if (chunked_mmq && tokens - start - count > 0 &&
          tokens - start - count < gguf_constants::kMmqTileStep) {
        // ggml_cuda_mmq_get_J_max rounds down to one J tile.
        count -= gguf_constants::kMmqTileStep - (tokens - start - count);
      }
      ggml_tensor chunk_input =
          make_moe_input_tensor(buffers.input + start * k, k, count);
      ggml_tensor chunk_ids = make_moe_ids_tensor(
          static_cast<const int32_t*>(topk_ids.data_ptr()) + start * top_k,
          top_k, count);
      ggml_tensor chunk_output = make_moe_output_tensor(
          buffers.result + start * top_k * row, row, top_k, count);
      if (chunked_mmq) {
        ggml_cuda_mul_mat_q(call.context, &src0, &chunk_input, &chunk_ids,
                            &chunk_output);
        check_launch("upstream MoE chunked MMQ");
      } else if (kernel == MoeKernel::kMmfChunks) {
        ggml_cuda_mul_mat_f(call.context, &src0, &chunk_input, &chunk_ids,
                            &chunk_output);
        check_launch("upstream MoE chunked MMF");
      } else {
        ggml_cuda_mul_mat_vec_q(call.context, &src0, &chunk_input, &chunk_ids,
                                &chunk_output);
        check_launch("upstream MoE chunked MMVQ");
      }
      start += count;
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
