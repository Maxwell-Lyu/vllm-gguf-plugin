// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Dense bridge tensors are contiguous and the stream/device are already set.
// Avoid the general aten::to dispatcher for these simple linear conversions.
// Quantization and matrix multiplication stay in the unmodified upstream TUs.
template <typename Dst, typename Src>
static __global__ void gguf_cast(Dst* dst, const Src* src, int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    dst[i] = static_cast<Dst>(static_cast<float>(src[i]));
  }
}

template <typename Dst, typename Src>
static void gguf_cast_async(void* dst, const void* src, int64_t count,
                            cudaStream_t stream) {
  if (count > 0) {
    gguf_cast<<<(count + 255) / 256, 256, 0, stream>>>(
        static_cast<Dst*>(dst), static_cast<const Src*>(src), count);
  }
}
