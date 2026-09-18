"""Upstream CUDA kernel robustness / regression tests.

Consolidates the former ``test_upstream_performance_regressions.py``.  These
are end-to-end CUDA tests that exercise conditions seen in real serving but
not covered by the basic implementation suite:

- dtype casting (fp16/bf16/fp32 inputs) with nonzero storage offsets and
  irregular tail tiles,
- CUDA graph capture and concurrent multi-stream execution,
- default/auto mode routing actually selects the upstream backend,
- public API fallback for IQ types and strict rejection in explicit legacy
  mode.
"""

from concurrent.futures import ThreadPoolExecutor

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as Q

from tests.helpers_upstream import make_padded_weight

cuda_mark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


# ---------------------------------------------------------------------------
# dtype casting / storage offsets
# ---------------------------------------------------------------------------


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("batch", [1, 8, 16, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_casts_match_float32_path_on_offset_irregular_weights(
    monkeypatch, batch, dtype
):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    n, k = 37, 768
    packed = gguf.quantize(
        np.random.default_rng(42).standard_normal((n, k), dtype=np.float32) * 0.1,
        Q.Q4_0,
    )
    base = make_padded_weight(packed, Q.Q4_0, k)
    weight = base[1:]  # Exercise nonzero storage offset and an irregular last tile.
    x = torch.randn(batch, k, device="cuda", dtype=dtype)
    name = "ggml_mul_mat_vec_a8" if batch <= 8 else "ggml_mul_mat_a8"
    op = getattr(torch.ops._C_gguf, name)
    actual = op(weight, x, int(Q.Q4_0), n - 1)
    reference = op(weight, x.float(), int(Q.Q4_0), n - 1).to(dtype)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    dense = torch.from_numpy(gguf.dequantize(packed[1:], Q.Q4_0)).cuda()
    torch.testing.assert_close(
        actual.float(), x.float() @ dense.T, atol=0.15, rtol=0.05
    )


# ---------------------------------------------------------------------------
# CUDA graphs and concurrent streams
# ---------------------------------------------------------------------------


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize("batch", [2, 16])
def test_cast_graphs_and_concurrent_streams(monkeypatch, batch):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    n, k = 37, 768
    packed = gguf.quantize(
        np.random.default_rng(9).standard_normal((n, k), dtype=np.float32), Q.Q4_0
    )
    weight = make_padded_weight(packed, Q.Q4_0, k)
    op = getattr(
        torch.ops._C_gguf, "ggml_mul_mat_vec_a8" if batch <= 8 else "ggml_mul_mat_a8"
    )
    inputs = [
        torch.randn(batch, k, device="cuda", dtype=torch.float16) for _ in range(2)
    ]
    expected = [op(weight, x, int(Q.Q4_0), n) for x in inputs]
    streams = [torch.cuda.Stream() for _ in inputs]
    torch.cuda.synchronize()

    def run(i):
        with torch.inference_mode(), torch.cuda.stream(streams[i]):
            return [op(weight, inputs[i], int(Q.Q4_0), n) for _ in range(10)]

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, range(2)))
    torch.cuda.synchronize()
    for i, outputs in enumerate(results):
        for output in outputs:
            torch.testing.assert_close(output, expected[i], rtol=0, atol=0)

    graphs, outputs = [], []
    for i in range(2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=streams[i]):
            outputs.append(op(weight, inputs[i], int(Q.Q4_0), n))
        graphs.append(graph)
    for _ in range(3):
        for i, graph in enumerate(graphs):
            with torch.cuda.stream(streams[i]):
                graph.replay()
    torch.cuda.synchronize()
    for i, output in enumerate(outputs):
        torch.testing.assert_close(output, expected[i], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Mode routing end-to-end
# ---------------------------------------------------------------------------


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize(
    "batch,dtype",
    [
        (1, torch.float16),
        (16, torch.float16),
        (16, torch.float32),
        (128, torch.float16),
    ],
)
def test_default_and_auto_use_upstream(monkeypatch, batch, dtype):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k = 512, 1024
    packed = gguf.quantize(
        np.random.default_rng(71).standard_normal((n, k), dtype=np.float32),
        Q.Q4_0,
    )
    weight = make_padded_weight(packed, Q.Q4_0, k)
    x = torch.randn(batch, k, device="cuda", dtype=dtype)
    op_name = "ggml_mul_mat_vec_a8" if batch == 1 else "ggml_mul_mat_a8"
    op = getattr(torch.ops._C_gguf, op_name)

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    expected = op(weight, x, int(Q.Q4_0), n)
    monkeypatch.delenv("VLLM_GGUF_CUDA_KERNEL")
    default = op(weight, x, int(Q.Q4_0), n)
    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "auto")
    auto = op(weight, x, int(Q.Q4_0), n)

    torch.testing.assert_close(default, expected, rtol=0, atol=0)
    torch.testing.assert_close(auto, expected, rtol=0, atol=0)


@cuda_mark
@torch.inference_mode()
def test_iq_public_fallback_is_nonzero_and_direct_legacy_rejected(monkeypatch):
    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "auto")
    n, k = 37, 256
    # gguf implements IQ4_NL dequantization, but not its Python quantizer.
    # Every nibble is a valid codebook index; use finite, nonzero block scales.
    packed = np.random.default_rng(23).integers(
        0, 256, (n, k // 32, 18), dtype=np.uint8
    )
    packed[:, :, :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    packed = packed.reshape(n, -1)
    weight = make_padded_weight(packed, Q.IQ4_NL, k)
    x = torch.randn(16, k, device="cuda", dtype=torch.float16)
    actual = ops.ggml_mul_mat_a8(weight, x, int(Q.IQ4_NL), n)
    reference = x.float() @ torch.from_numpy(gguf.dequantize(packed, Q.IQ4_NL)).cuda().T
    assert torch.count_nonzero(actual) > 0
    torch.testing.assert_close(actual.float(), reference, atol=0.25, rtol=0.10)
    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "legacy")
    with pytest.raises(RuntimeError, match="no legacy MMQ kernel"):
        torch.ops._C_gguf.ggml_mul_mat_a8(weight, x, int(Q.IQ4_NL), n)
