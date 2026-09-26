// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "bridge_support.cuh"

namespace {
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

}  // namespace
