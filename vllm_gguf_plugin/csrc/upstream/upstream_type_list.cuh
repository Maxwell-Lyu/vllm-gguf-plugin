// SPDX-License-Identifier: Apache-2.0
#pragma once

// Types with an instantiated upstream MMQ kernel. Keep dispatch and capability
// checks on this single list; IQ1_M has only MMVQ and dequantization.
#define GGUF_UPSTREAM_MMQ_TYPES(X) \
  X(GGML_TYPE_Q1_0)                \
  X(GGML_TYPE_Q2_0)                \
  X(GGML_TYPE_Q4_0)                \
  X(GGML_TYPE_Q4_1)                \
  X(GGML_TYPE_Q5_0)                \
  X(GGML_TYPE_Q5_1)                \
  X(GGML_TYPE_Q8_0)                \
  X(GGML_TYPE_MXFP4)               \
  X(GGML_TYPE_NVFP4)               \
  X(GGML_TYPE_Q2_K)                \
  X(GGML_TYPE_Q3_K)                \
  X(GGML_TYPE_Q4_K)                \
  X(GGML_TYPE_Q5_K)                \
  X(GGML_TYPE_Q6_K)                \
  X(GGML_TYPE_IQ1_S)               \
  X(GGML_TYPE_IQ2_XXS)             \
  X(GGML_TYPE_IQ2_XS)              \
  X(GGML_TYPE_IQ2_S)               \
  X(GGML_TYPE_IQ3_XXS)             \
  X(GGML_TYPE_IQ3_S)               \
  X(GGML_TYPE_IQ4_NL)              \
  X(GGML_TYPE_IQ4_XS)
