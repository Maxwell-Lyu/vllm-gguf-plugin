// SPDX-License-Identifier: Apache-2.0
#include "bridge_common.cuh"

namespace {
Tensor run_upstream_mmvq(const Tensor& W, const Tensor& X, int64_t type,
                         int64_t row, int64_t k) {
  UpstreamCall call(X);
  const cudaStream_t stream = call.stream;
  const int64_t batch = X.size(0);
  const int64_t k_padded = padded_k(k);

  const size_t q8_bytes = static_cast<size_t>(batch) *
                          static_cast<size_t>(k_padded) * sizeof(block_q8_1) /
                          kQK8_1;
  const DenseBuffers buffers =
      dense_buffers(X, X.size(0), row, q8_bytes, *call.scratch_pool, stream);
  void* q8_data = buffers.q8;

  quantize_row_q8_1_cuda(buffers.input, nullptr, q8_data,
                         static_cast<ggml_type>(type), k, k, 0, 0, k_padded,
                         batch, 1, 1, stream);

  ggml_tensor src0 = make_quant_tensor(W, type, k, row);
  ggml_tensor src1 = make_f32_tensor(buffers.input, k_padded, batch);
  ggml_tensor dst = make_output_tensor(buffers.result, row, batch);
  ggml_cuda_op_mul_mat_vec_q(call.context, &src0, &src1, &dst,
                             static_cast<const char*>(W.data_ptr()),
                             buffers.input, static_cast<const char*>(q8_data),
                             buffers.result, 0, row, batch, k_padded, stream);
  check_launch("upstream MMVQ");
  return finish_output(buffers, stream);
}

Tensor run_upstream_mmq(const Tensor& W, const Tensor& X, int64_t type,
                        int64_t row, int64_t k) {
  const int32_t device_index = X.get_device_index();
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
  UpstreamCall call(X);
  const cudaStream_t stream = call.stream;

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
    // No bridge quantized workspace is needed; output conversion still runs
    // through the shared dense-buffers conversion path (zero q8 region).
    const DenseBuffers buffers =
        dense_buffers(X, X.size(0), row, 0, *call.scratch_pool, stream);
    ggml_tensor src0 = make_quant_tensor(W, type, k, row);
    ggml_tensor src1 = make_f32_tensor(buffers.input, k, batch);
    ggml_tensor dst = make_output_tensor(buffers.result, row, batch);
    ggml_cuda_mul_mat_q(call.context, &src0, &src1, /*ids=*/nullptr, &dst);
    check_launch("upstream MMQ (native FP4)");
    return finish_output(buffers, stream);
  }

  const size_t q8_bytes = static_cast<size_t>(batch) *
                              static_cast<size_t>(k_padded) *
                              sizeof(block_q8_1_mmq) / QK8_1_MMQ +
                          static_cast<size_t>(j_max) * sizeof(block_q8_1_mmq);
  const DenseBuffers buffers =
      dense_buffers(X, X.size(0), row, q8_bytes, *call.scratch_pool, stream);
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
  ggml_upstream_mul_mat_q(call.context, args, stream);
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

ScalarType blas_compute_dtype(int cc) {
  ScalarType dtype =
      fast_fp16_hardware_available(cc) ? ScalarType::Half : ScalarType::Float;
  const char* setting = std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
  if (setting != nullptr) {
    std::string name(setting);
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (name == "f32" || name == "fp32") {
      dtype = ScalarType::Float;
    } else if (name == "f16" || name == "fp16") {
      dtype = ScalarType::Half;
    } else if (name == "bf16") {
      dtype = ScalarType::BFloat16;
    } else {
      STD_TORCH_CHECK(name == "auto",
                      "GGML_CUDA_CUBLAS_COMPUTE_TYPE must be auto, f32, f16, "
                      "or bf16");
    }
  }
  STD_TORCH_CHECK(dtype != ScalarType::BFloat16 || cc >= GGML_CUDA_CC_AMPERE,
                  "BF16 cuBLAS requires Ampere or newer CUDA hardware");
  return dtype;
}

void cast_blas_input(const Tensor& X, Tensor& converted, int64_t count,
                     cudaStream_t stream) {
  const auto source = X.scalar_type();
  const auto target = converted.scalar_type();
  if (target == ScalarType::Float) {
    if (source == ScalarType::Half) {
      gguf_cast_async<float, half>(converted.data_ptr(), X.data_ptr(), count,
                                   stream);
    } else {
      gguf_cast_async<float, nv_bfloat16>(converted.data_ptr(), X.data_ptr(),
                                          count, stream);
    }
  } else if (target == ScalarType::Half) {
    if (source == ScalarType::Float) {
      gguf_cast_async<half, float>(converted.data_ptr(), X.data_ptr(), count,
                                   stream);
    } else {
      gguf_cast_async<half, nv_bfloat16>(converted.data_ptr(), X.data_ptr(),
                                         count, stream);
    }
  } else if (source == ScalarType::Float) {
    gguf_cast_async<nv_bfloat16, float>(converted.data_ptr(), X.data_ptr(),
                                        count, stream);
  } else {
    gguf_cast_async<nv_bfloat16, half>(converted.data_ptr(), X.data_ptr(),
                                       count, stream);
  }
}

Tensor run_upstream_blas(const Tensor& W, const Tensor& X, int64_t type,
                         int64_t row, int64_t k) {
  const int32_t device_index = X.get_device_index();
  const DeviceGuard device_guard(device_index);
  const cudaStream_t stream = current_stream(device_index);
  const int cc = ggml_cuda_info().devices[device_index].cc;
  const ScalarType compute_dtype = blas_compute_dtype(cc);
  const int64_t batch = X.size(0);
  STD_TORCH_CHECK(type != GGML_TYPE_MXFP4 || k % 256 == 0,
                  "upstream MXFP4 cuBLAS requires K aligned to 256 values");
  STD_TORCH_CHECK(row <= INT_MAX && k <= INT_MAX && batch <= INT_MAX,
                  "upstream cuBLAS dimensions exceed int32 limits");

  Tensor weights = torch::stable::new_empty(W, {row, k}, compute_dtype);
  if (compute_dtype == ScalarType::Float) {
    run_upstream_dequantize<float>(W, weights, type, row * k, stream);
  } else if (compute_dtype == ScalarType::Half) {
    run_upstream_dequantize<half>(W, weights, type, row * k, stream);
  } else {
    run_upstream_dequantize<nv_bfloat16>(W, weights, type, row * k, stream);
  }

  Tensor converted;
  const void* activation = X.data_ptr();
  if (X.scalar_type() != compute_dtype) {
    converted = torch::stable::new_empty(X, {batch, k}, compute_dtype);
    cast_blas_input(X, converted, batch * k, stream);
    activation = converted.data_ptr();
  }
  const bool half_result =
      compute_dtype == ScalarType::Half && cc != GGML_CUDA_CC_VOLTA;
  const ScalarType result_dtype =
      half_result ? ScalarType::Half : ScalarType::Float;
  Tensor result = torch::stable::new_empty(X, {batch, row}, result_dtype);

  void* raw_handle = nullptr;
  TORCH_ERROR_CODE_CHECK(torch_get_current_cuda_blas_handle(&raw_handle));
  auto handle = static_cast<cublasHandle_t>(raw_handle);
  STD_TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
                  "upstream cuBLAS could not bind the Torch current stream");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const half alpha_half = __float2half(1.0f);
  const half beta_half = __float2half(0.0f);
  const cudaDataType_t input_type =
      compute_dtype == ScalarType::Float  ? CUDA_R_32F
      : compute_dtype == ScalarType::Half ? CUDA_R_16F
                                          : CUDA_R_16BF;
  const cublasStatus_t status =
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(row),
                   static_cast<int>(batch), static_cast<int>(k),
                   half_result ? static_cast<const void*>(&alpha_half)
                               : static_cast<const void*>(&alpha),
                   weights.data_ptr(), input_type, static_cast<int>(k),
                   activation, input_type, static_cast<int>(k),
                   half_result ? static_cast<const void*>(&beta_half)
                               : static_cast<const void*>(&beta),
                   result.data_ptr(), half_result ? CUDA_R_16F : CUDA_R_32F,
                   static_cast<int>(row),
                   half_result ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  STD_TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                  "upstream cuBLAS GEMM failed with status ",
                  static_cast<int>(status));

  if (X.scalar_type() == result_dtype) {
    return result;
  }
  Tensor output = torch::stable::new_empty(X, {batch, row}, X.scalar_type());
  if (result_dtype == ScalarType::Half &&
      X.scalar_type() == ScalarType::Float) {
    gguf_cast_async<float, half>(output.data_ptr(), result.data_ptr(),
                                 batch * row, stream);
  } else if (result_dtype == ScalarType::Half) {
    gguf_cast_async<nv_bfloat16, half>(output.data_ptr(), result.data_ptr(),
                                       batch * row, stream);
  } else if (X.scalar_type() == ScalarType::Half) {
    gguf_cast_async<half, float>(output.data_ptr(), result.data_ptr(),
                                 batch * row, stream);
  } else {
    gguf_cast_async<nv_bfloat16, float>(output.data_ptr(), result.data_ptr(),
                                        batch * row, stream);
  }
  check_launch("upstream cuBLAS output");
  return output;
}

bool upstream_mmvq_runnable(const Tensor& W, const Tensor& X, int64_t type,
                            int64_t k) {
  return is_upstream_type(type) && X.size(0) <= kMmvqMaxBatchSize &&
         has_weight_padding(W, k, type, "ggml_mul_mat_vec_a8");
}

bool upstream_mmq_eligible(const Tensor& W, int64_t type, int64_t k) {
  return is_upstream_mmq_type(type) &&
         has_weight_padding(W, k, type, "ggml_mul_mat_a8");
}

}  // namespace

bool is_upstream_mmq_type(int64_t type) {
  return upstream_mmq_type_supported(type);
}

Tensor run_upstream_dense(const Tensor& W, const Tensor& X, int64_t type,
                          int64_t row, int64_t k) {
  const DeviceGuard device_guard(X.get_device_index());
  const int cc = ggml_cuda_info().devices[X.get_device_index()].cc;
  const auto quant_type = static_cast<ggml_type>(type);
  const int64_t batch = X.size(0);
  if (ggml_cuda_should_use_mmvq(quant_type, cc, batch) &&
      upstream_mmvq_runnable(W, X, type, k)) {
    return run_upstream_mmvq(W, X, type, row, k);
  }
  if (is_upstream_mmq_type(type) &&
      ggml_cuda_should_use_mmq(quant_type, cc, batch, 0) &&
      upstream_mmq_eligible(W, type, k) &&
      ggml_cuda_mmq_get_J_max(quant_type, row % 128 != 0, cc, batch) > 0) {
    return run_upstream_mmq(W, X, type, row, k);
  }
  return run_upstream_blas(W, X, type, row, k);
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
