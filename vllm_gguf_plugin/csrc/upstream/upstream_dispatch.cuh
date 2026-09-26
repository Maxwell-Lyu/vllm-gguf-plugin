// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "mmq.cuh"
#include "upstream_type_list.cuh"

// Explicit instances are emitted by the upstream template-instances sources;
// this wrapper does not copy or specialize the upstream kernel body.
inline void ggml_upstream_mul_mat_q(ggml_backend_cuda_context& context,
                                    const mmq_args& args, cudaStream_t stream) {
  switch (args.type_x) {
#define GGUF_MMQ_CASE(type)                      \
  case type:                                     \
    mul_mat_q_case<type>(context, args, stream); \
    break;
    GGUF_UPSTREAM_MMQ_TYPES(GGUF_MMQ_CASE)
#undef GGUF_MMQ_CASE
    default:
      GGML_ABORT("unsupported upstream MMQ type");
  }
}
