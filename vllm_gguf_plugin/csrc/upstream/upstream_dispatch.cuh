// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "mmq.cuh"

// Explicit instances are emitted by the upstream template-instances sources;
// this wrapper does not copy or specialize the upstream kernel body.
inline void ggml_upstream_mul_mat_q(ggml_backend_cuda_context& context,
                                    const mmq_args& args, cudaStream_t stream) {
  switch (args.type_x) {
    case GGML_TYPE_Q1_0:
      mul_mat_q_case<GGML_TYPE_Q1_0>(context, args, stream);
      break;
    case GGML_TYPE_Q2_0:
      mul_mat_q_case<GGML_TYPE_Q2_0>(context, args, stream);
      break;
    case GGML_TYPE_Q4_0:
      mul_mat_q_case<GGML_TYPE_Q4_0>(context, args, stream);
      break;
    case GGML_TYPE_Q4_1:
      mul_mat_q_case<GGML_TYPE_Q4_1>(context, args, stream);
      break;
    case GGML_TYPE_Q5_0:
      mul_mat_q_case<GGML_TYPE_Q5_0>(context, args, stream);
      break;
    case GGML_TYPE_Q5_1:
      mul_mat_q_case<GGML_TYPE_Q5_1>(context, args, stream);
      break;
    case GGML_TYPE_Q8_0:
      mul_mat_q_case<GGML_TYPE_Q8_0>(context, args, stream);
      break;
    case GGML_TYPE_MXFP4:
      mul_mat_q_case<GGML_TYPE_MXFP4>(context, args, stream);
      break;
    case GGML_TYPE_NVFP4:
      mul_mat_q_case<GGML_TYPE_NVFP4>(context, args, stream);
      break;
    case GGML_TYPE_Q2_K:
      mul_mat_q_case<GGML_TYPE_Q2_K>(context, args, stream);
      break;
    case GGML_TYPE_Q3_K:
      mul_mat_q_case<GGML_TYPE_Q3_K>(context, args, stream);
      break;
    case GGML_TYPE_Q4_K:
      mul_mat_q_case<GGML_TYPE_Q4_K>(context, args, stream);
      break;
    case GGML_TYPE_Q5_K:
      mul_mat_q_case<GGML_TYPE_Q5_K>(context, args, stream);
      break;
    case GGML_TYPE_Q6_K:
      mul_mat_q_case<GGML_TYPE_Q6_K>(context, args, stream);
      break;
    case GGML_TYPE_IQ1_S:
      mul_mat_q_case<GGML_TYPE_IQ1_S>(context, args, stream);
      break;
    case GGML_TYPE_IQ2_XXS:
      mul_mat_q_case<GGML_TYPE_IQ2_XXS>(context, args, stream);
      break;
    case GGML_TYPE_IQ2_XS:
      mul_mat_q_case<GGML_TYPE_IQ2_XS>(context, args, stream);
      break;
    case GGML_TYPE_IQ2_S:
      mul_mat_q_case<GGML_TYPE_IQ2_S>(context, args, stream);
      break;
    case GGML_TYPE_IQ3_XXS:
      mul_mat_q_case<GGML_TYPE_IQ3_XXS>(context, args, stream);
      break;
    case GGML_TYPE_IQ3_S:
      mul_mat_q_case<GGML_TYPE_IQ3_S>(context, args, stream);
      break;
    case GGML_TYPE_IQ4_NL:
      mul_mat_q_case<GGML_TYPE_IQ4_NL>(context, args, stream);
      break;
    case GGML_TYPE_IQ4_XS:
      mul_mat_q_case<GGML_TYPE_IQ4_XS>(context, args, stream);
      break;
    default:
      GGML_ABORT("unsupported upstream MMQ type");
  }
}
