// SPDX-License-Identifier: Apache-2.0
#include "bridge_common.cuh"

namespace {
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

  const int64_t input_count = total * k;
  const int64_t input_grid_count = (input_count - 1) / 256 + 1;
  STD_TORCH_CHECK(input_grid_count <= INT_MAX, kMoeNotEligibleMarker,
                  ": routed input exceeds CUDA grid limit");
  const int input_grid = static_cast<int>(input_grid_count);
  Tensor sorted_input =
      torch::stable::new_empty(X, {total, k}, X.scalar_type());
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
        run_upstream_dense(expert_weight, expert_input, type, row, k);
    auto* destination = static_cast<char*>(sorted_output.data_ptr()) +
                        first * row * X.element_size();
    STD_TORCH_CHECK(
        cudaMemcpyAsync(destination, expert_output.data_ptr(),
                        count * row * X.element_size(),
                        cudaMemcpyDeviceToDevice, stream) == cudaSuccess,
        "upstream MoE could not copy expert output");
    first += count;
  }

  const int64_t output_count = total * row;
  const int64_t output_grid_count = (output_count - 1) / 256 + 1;
  STD_TORCH_CHECK(output_grid_count <= INT_MAX, kMoeNotEligibleMarker,
                  ": routed output exceeds CUDA grid limit");
  const int output_grid = static_cast<int>(output_grid_count);
  Tensor output = torch::stable::new_empty(X, {total, row}, X.scalar_type());
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

void run_upstream_moe_mmvq_chunks(ggml_backend_cuda_context& context,
                                  const Tensor& W, const Tensor& topk_ids,
                                  const DenseBuffers& buffers, int64_t type,
                                  int64_t row, int64_t top_k, int64_t tokens,
                                  int64_t k, int64_t chunk_size) {
  ggml_tensor src0 = make_moe_weight_tensor(W, type, k, row);
  for (int64_t start = 0; start < tokens; start += chunk_size) {
    const int64_t count = std::min(chunk_size, tokens - start);
    ggml_tensor src1 =
        make_moe_input_tensor(buffers.input + start * k, k, count);
    ggml_tensor ids = make_moe_ids_tensor(
        static_cast<const int32_t*>(topk_ids.data_ptr()) + start * top_k, top_k,
        count);
    ggml_tensor dst = make_moe_output_tensor(
        buffers.result + start * top_k * row, row, top_k, count);
    ggml_cuda_mul_mat_vec_q(context, &src0, &src1, &ids, &dst);
    check_launch("upstream MoE chunked MMVQ");
  }
}

}  // namespace

Tensor run_upstream_moe_projection(const Tensor& W, const Tensor& X,
                                   const Tensor& topk_ids, int64_t type,
                                   int64_t row, int64_t top_k, int64_t tokens,
                                   int64_t k) {
  const int32_t device_index = X.get_device_index();
  const DeviceGuard device_guard(device_index);
  const cudaStream_t stream = current_stream(device_index);
  const int64_t output_rows = tokens * top_k;
  const int cc = ggml_cuda_info().devices[device_index].cc;
  const int mmvq_max =
      std::min<int>(kMmvqMaxBatchSize,
                    get_mmvq_mmid_max_batch(static_cast<ggml_type>(type), cc));
  const bool use_mmvq = mmvq_max > 0 && tokens <= mmvq_max;
  const bool fallback = row % 128 != 0;
  const bool use_mmq = !use_mmvq && is_upstream_mmq_type(type) &&
                       ggml_cuda_should_use_mmq(static_cast<ggml_type>(type),
                                                cc, tokens, W.size(0)) &&
                       ggml_cuda_mmq_get_J_max(static_cast<ggml_type>(type),
                                               fallback, cc, tokens) > 0;
  if (!use_mmvq && !use_mmq) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    STD_TORCH_CHECK(
        cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess,
        "upstream MoE could not query CUDA graph capture status");
    if (capture_status == cudaStreamCaptureStatusNone) {
      return run_upstream_moe_grouped(W, X, topk_ids, type, row, top_k, tokens,
                                      k, stream);
    }
    STD_TORCH_CHECK(mmvq_max > 0, kMoeNotEligibleMarker,
                    ": no graph-safe MMVQ configuration for type/device");
  }

  UpstreamCall call(X);
  const DenseBuffers buffers =
      dense_buffers(X, output_rows, row, 0, *call.scratch_pool, call.stream);

  ggml_tensor src0 = make_moe_weight_tensor(W, type, k, row);
  ggml_tensor src1 = make_moe_input_tensor(buffers.input, k, tokens);
  ggml_tensor ids_tensor = make_moe_ids_tensor(
      static_cast<const int32_t*>(topk_ids.data_ptr()), top_k, tokens);
  ggml_tensor dst = make_moe_output_tensor(buffers.result, row, top_k, tokens);

  if (use_mmvq) {
    ggml_cuda_mul_mat_vec_q(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMVQ");
  } else if (use_mmq) {
    ggml_cuda_mul_mat_q(call.context, &src0, &src1, &ids_tensor, &dst);
    check_launch("upstream MoE MMQ");
  } else {
    run_upstream_moe_mmvq_chunks(call.context, W, topk_ids, buffers, type, row,
                                 top_k, tokens, k, mmvq_max);
  }
  return finish_output(buffers, call.stream);
}
