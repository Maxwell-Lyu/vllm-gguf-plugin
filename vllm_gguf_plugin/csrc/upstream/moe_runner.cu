// SPDX-License-Identifier: Apache-2.0
#include "bridge_common.cuh"

Tensor run_upstream_moe_projection(const Tensor& W, const Tensor& X,
                                   const Tensor& topk_ids, int64_t type,
                                   int64_t row, int64_t top_k, int64_t tokens,
                                   int64_t k) {
  const int32_t device_index = X.get_device_index();
  const DeviceGuard device_guard(device_index);
  const int cc = ggml_cuda_info().devices[device_index].cc;
  const int mmvq_max =
      std::min<int>(kMmvqMaxBatchSize,
                    get_mmvq_mmid_max_batch(static_cast<ggml_type>(type), cc));
  const bool use_mmvq = mmvq_max > 0 && tokens <= mmvq_max;
  const bool fallback = row % 128 != 0;
  const bool use_mmq =
      !use_mmvq && is_upstream_mmq_type(type) &&
      ggml_cuda_info().devices[device_index].smpbo >= 48 * 1024 &&
      ggml_cuda_mmq_get_J_max(static_cast<ggml_type>(type), fallback, cc,
                              tokens) > 0;
  const bool use_iq1_m_chunks =
      type == GGML_TYPE_IQ1_M && !use_mmvq && mmvq_max > 0;
  STD_TORCH_CHECK(use_mmvq || use_mmq || use_iq1_m_chunks,
                  kMoeNotEligibleMarker,
                  ": neither MMVQ nor MMQ supports this type/shape/device");

  UpstreamCall call(X);
  const int64_t output_rows = tokens * top_k;
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
