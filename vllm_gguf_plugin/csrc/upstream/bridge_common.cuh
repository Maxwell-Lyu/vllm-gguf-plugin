// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "bridge_runtime.cuh"
#include "ggml_tensor_views.cuh"
#include "bridge_validation.cuh"

// Internal runner interface. Only bridge.cu exposes Torch operators.
bool is_upstream_mmq_type(int64_t type);
Tensor run_upstream_dense(const Tensor& W, const Tensor& X, int64_t type,
                          int64_t row, int64_t k);
Tensor run_upstream_moe_projection(const Tensor& W, const Tensor& X,
                                   const Tensor& topk_ids, int64_t type,
                                   int64_t row, int64_t top_k, int64_t tokens,
                                   int64_t k);
