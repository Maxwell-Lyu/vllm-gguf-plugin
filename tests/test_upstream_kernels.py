"""Upstream CUDA kernel basic-implementation tests.

Consolidates the former ``test_upstream_phase0.py`` / ``test_upstream_phase2_3.py``
correctness suites into one file organized by kernel surface:

- **Storage padding**: Python-side materialization of the MATRIX_ROW_PADDING
  (512) tail required by upstream kernels (dense 2D, loader preallocation,
  mixed shards, MoE 3D).
- **Dense (MMVQ/MMQ)**: upstream vs. legacy and dequantized-reference checks,
  full type dispatch matrix, and eligibility edges (IQ1_M MMQ rejection,
  missing-padding rejection).
- **Dequantize / embedding**: template-only types vs. ``gguf.dequantize``.
- **MoE**: upstream MoE kernel template types and a non-512-multiple k
  end-to-end reference check.

All CUDA tests run in ``upstream`` mode and tolerate the absence of the
template-only types (Q1_0/Q2_0/MXFP4/NVFP4) on older gguf-python releases.
"""

from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as Q

from tests.helpers_upstream import (
    MMQ_TYPES,
    TEMPLATE_EXTRA_TYPES,
    make_padded_weight,
    make_template_raw,
    upstream_padding_bytes,
)

cuda_mark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


# ---------------------------------------------------------------------------
# Storage padding (CPU, no CUDA required)
# ---------------------------------------------------------------------------


def test_dense_storage_padding_materialization():
    from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

    n, packed_k = 3, 144
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randint(0, 256, (n, packed_k), dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_type = SimpleNamespace(weight_type=Q.Q4_0)

    GGUFLinearMethod(None)._materialize_upstream_storage_padding(layer)

    assert tuple(layer.weight.shape) == (n, packed_k)
    assert layer.weight.untyped_storage().nbytes() >= n * packed_k + 144


def test_weight_loader_preallocates_upstream_storage_padding():
    from vllm_gguf_plugin.quantization.params import GGUFUninitializedWeightParameter

    weight = GGUFUninitializedWeightParameter(requires_grad=False)
    weight.gguf_weight_type_parameter = SimpleNamespace(
        weight_type=Q.Q4_0, shard_weight_type={}
    )

    weight._store(torch.ones((3, 144), dtype=torch.uint8))

    assert tuple(weight.shape) == (3, 144)
    assert weight.untyped_storage().nbytes() >= weight.numel() + 144


def test_mixed_shards_keep_individual_upstream_storage_padding(monkeypatch):
    from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod
    from vllm_gguf_plugin.quantization.params import GGUFWeightParameter

    monkeypatch.setattr(
        "vllm_gguf_plugin.quantization.params.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm_gguf_plugin.quantization.params.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    sources = {
        0: torch.ones((2, 144), dtype=torch.uint8),
        1: torch.full((3, 84), 2, dtype=torch.uint8),
    }
    weight = GGUFWeightParameter(
        data=torch.empty(0, dtype=torch.uint8),
        weight_loader=lambda *args: None,
        input_dim=1,
        output_dim=0,
        tensor_shape=(5, 256),
    )
    weight.data_container = [sources[0], sources[1]]
    weight.shard_id = [0, 1]
    weight.shard_id_map = {0: 0, 1: 1}
    layer = torch.nn.Module()
    layer.register_parameter("weight", weight)
    layer.weight_type = SimpleNamespace(
        weight_type=Q.Q4_0,
        shard_weight_type={0: Q.Q4_0, 1: Q.Q2_K},
    )

    GGUFLinearMethod(None)._create_padded_weight_param(layer)

    assert layer.weight.ndim == 1
    for shard_id, source in sources.items():
        offset, rows, packed_row_size = layer.weight.shard_storage_map[shard_id]
        shard = layer.weight.narrow(0, offset, rows * packed_row_size).view(
            rows, packed_row_size
        )
        torch.testing.assert_close(shard, source)
        block_size, type_size = gguf.GGML_QUANT_SIZES[
            layer.weight_type.shard_weight_type[shard_id]
        ]
        logical_k = packed_row_size // type_size * block_size
        padding_bytes = ((-logical_k) % 512) // block_size * type_size
        available = layer.weight.untyped_storage().nbytes() - offset
        assert available >= shard.numel() + padding_bytes


def test_moe_process_weights_after_loading_pads_3d_weights():
    from vllm_gguf_plugin.quantization.fused_moe import GGUFMoEMethod

    # k=1184 is not a multiple of the 512-value MATRIX_ROW_PADDING, so a
    # storage tail must be reserved behind both 3D weights.
    n, k, experts = 37, 1184, 3
    quant_type = Q.Q4_0
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    packed_row = k // block_size * type_size
    layer = torch.nn.Module()
    for name in ("w13_weight", "w2_weight"):
        param = torch.nn.Parameter(
            torch.randint(0, 256, (experts, n, packed_row), dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter(name, param)
    layer.w13_weight_type = SimpleNamespace(weight_type=int(Q.Q4_0))
    layer.w2_weight_type = SimpleNamespace(weight_type=int(Q.Q4_0))

    GGUFMoEMethod(None, None).process_weights_after_loading(layer)

    logical_bytes = layer.w13_weight.numel() * layer.w13_weight.element_size()
    padding_bytes = upstream_padding_bytes(quant_type, k)
    assert padding_bytes > 0
    assert layer.w13_weight.untyped_storage().nbytes() >= logical_bytes + padding_bytes
    assert layer.w2_weight.untyped_storage().nbytes() >= logical_bytes + padding_bytes
    assert tuple(layer.w13_weight.shape) == (experts, n, packed_row)


# ---------------------------------------------------------------------------
# Dense kernels: MMVQ and MMQ
# ---------------------------------------------------------------------------


@cuda_mark
@torch.inference_mode()
def test_dense_upstream_matches_legacy_and_reference(monkeypatch):
    """Both MMVQ (batch=1) and MMQ (batch=16) agree with legacy and dense ref."""
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k = 37, 256
    source = np.random.default_rng(0).standard_normal((n, k), dtype=np.float32)
    packed = gguf.quantize(source, Q.Q4_0)
    weight = make_padded_weight(packed, Q.Q4_0, k)
    dense = torch.from_numpy(gguf.dequantize(packed, Q.Q4_0)).cuda()

    for op_name, batch in (
        ("ggml_mul_mat_vec_a8", 1),
        ("ggml_mul_mat_a8", 16),
    ):
        x = torch.randn((batch, k), device="cuda", dtype=torch.float32)
        monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
        upstream = getattr(torch.ops._C_gguf, op_name)(weight, x, int(Q.Q4_0), n)
        monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "legacy")
        legacy = getattr(torch.ops._C_gguf, op_name)(weight, x, int(Q.Q4_0), n)
        reference = x @ dense.T

        torch.testing.assert_close(upstream, reference, atol=1.5, rtol=0.2)
        torch.testing.assert_close(upstream, legacy, atol=1.5, rtol=0.2)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize(
    "quant_type", (Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0), ids=lambda q: q.name
)
def test_dense_mmvq_upstream_reference(monkeypatch, quant_type):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k = 7, 256
    source = np.random.default_rng(int(quant_type)).standard_normal(
        (n, k), dtype=np.float32
    )
    packed = gguf.quantize(source, quant_type)
    weight = make_padded_weight(packed, quant_type, k)
    x = torch.randn((2, k), device="cuda", dtype=torch.float32)

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    actual = torch.ops._C_gguf.ggml_mul_mat_vec_a8(weight, x, int(quant_type), n)
    reference = x @ torch.from_numpy(gguf.dequantize(packed, quant_type)).cuda().T
    torch.testing.assert_close(actual, reference, atol=1.5, rtol=0.2)


@cuda_mark
@torch.inference_mode()
@pytest.mark.parametrize(
    "quant_type", (Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0), ids=lambda q: q.name
)
def test_dense_mmq_upstream_reference(monkeypatch, quant_type):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k, batch = 37, 256, 16
    source = np.random.default_rng(int(quant_type) + 200).standard_normal(
        (n, k), dtype=np.float32
    )
    packed = gguf.quantize(source, quant_type)
    weight = make_padded_weight(packed, quant_type, k)
    x = torch.randn((batch, k), device="cuda", dtype=torch.float32)

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    actual = torch.ops._C_gguf.ggml_mul_mat_a8(weight, x, int(quant_type), n)
    reference = x @ torch.from_numpy(gguf.dequantize(packed, quant_type)).cuda().T
    torch.testing.assert_close(actual, reference, atol=1.5, rtol=0.2)


@cuda_mark
@torch.inference_mode()
def test_dense_mmq_zero_dispatch_matrix(monkeypatch):
    """Every MMQ type dispatches and returns finite zeros on zero inputs."""
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k, batch = 37, 768, 16
    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    for quant_type in MMQ_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        packed = np.zeros((n, k // block_size * type_size), dtype=np.uint8)
        weight = make_padded_weight(packed, quant_type, k)
        x = torch.zeros((batch, k), device="cuda", dtype=torch.float32)
        output = torch.ops._C_gguf.ggml_mul_mat_a8(weight, x, int(quant_type), n)
        assert output.shape == (batch, n)
        assert output.dtype == x.dtype
        assert bool(torch.isfinite(output).all())
        torch.testing.assert_close(output, torch.zeros_like(output))


@cuda_mark
@torch.inference_mode()
def test_dense_upstream_requires_storage_padding(monkeypatch):
    """Upstream mode must reject weights without the padding tail."""
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    n, k = 9, 256
    source = np.random.default_rng(1).standard_normal((n, k), dtype=np.float32)
    packed = gguf.quantize(source, Q.Q4_1)
    weight = torch.from_numpy(packed).cuda()
    x = torch.randn((2, k), device="cuda")

    for variable in ("VLLM_GGUF_CUDA_KERNEL", "VLLM_GGUF_CUDA_DENSE_KERNEL"):
        monkeypatch.delenv(variable, raising=False)

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "upstream")
    with pytest.raises(RuntimeError, match="MATRIX_ROW_PADDING storage"):
        torch.ops._C_gguf.ggml_mul_mat_vec_a8(weight, x, int(Q.Q4_1), n)

    # auto/legacy fall back and succeed.
    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "auto")
    output = torch.ops._C_gguf.ggml_mul_mat_vec_a8(weight, x, int(Q.Q4_1), n)
    assert output.shape == (x.shape[0], n)

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "legacy")
    output = torch.ops._C_gguf.ggml_mul_mat_vec_a8(weight, x, int(Q.Q4_1), n)
    assert output.shape == (x.shape[0], n)


@cuda_mark
@torch.inference_mode()
def test_dense_iq1_m_remains_mmvq_only(monkeypatch):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    quant_type = Q.IQ1_M
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    n, k = 9, block_size
    packed = np.zeros((n, type_size), dtype=np.uint8)
    weight = make_padded_weight(packed, quant_type, k)
    x = torch.zeros((2, k), device="cuda", dtype=torch.float32)
    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    with pytest.raises(RuntimeError, match="no upstream MMQ kernel"):
        torch.ops._C_gguf.ggml_mul_mat_a8(weight, x, int(quant_type), n)


# ---------------------------------------------------------------------------
# Dequantize / embedding
# ---------------------------------------------------------------------------


def _template_n(quant_type: Q) -> int:
    block_size, _ = gguf.GGML_QUANT_SIZES[quant_type]
    # convert.cu's MXFP4 row kernel consumes one 256-value super-block.
    return 256 if quant_type.name == "MXFP4" else block_size * 2


@cuda_mark
@torch.inference_mode()
def test_dequantize_template_instance_types(monkeypatch):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    rows = 3
    for quant_type in TEMPLATE_EXTRA_TYPES:
        n = _template_n(quant_type)
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        raw = np.zeros((rows, n // block_size * type_size), dtype=np.uint8)
        weight = torch.from_numpy(raw).cuda()
        output = ops.ggml_dequantize(weight, quant_type, rows, n, torch.float32)
        assert output.shape == (rows, n)
        assert bool(torch.isfinite(output).all())
        torch.testing.assert_close(output, torch.zeros_like(output))


@cuda_mark
@torch.inference_mode()
def test_dequantize_and_embedding_template_reference(monkeypatch):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization.vocal_embeds import apply_gguf_embedding

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "upstream")
    for quant_type in TEMPLATE_EXTRA_TYPES:
        try:
            probe = gguf.dequantize(
                np.zeros(gguf.GGML_QUANT_SIZES[quant_type][1], dtype=np.uint8),
                quant_type,
            )
        except NotImplementedError:
            # gguf-python has no Python dequantizer for this type yet; the
            # kernel-side smoke test still runs above.
            continue
        del probe
        n = _template_n(quant_type)
        raw = make_template_raw(quant_type, rows=4, n=n)
        weight = torch.from_numpy(raw).cuda()
        reference = torch.from_numpy(gguf.dequantize(raw, quant_type)).cuda()

        output = ops.ggml_dequantize(weight, quant_type, 4, n, torch.float32)
        torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)

        ids = torch.tensor([[0, 2], [3, 1]], dtype=torch.long, device="cuda")
        embedding = apply_gguf_embedding(
            ids, weight, quant_type, n, dtype=torch.float32
        )
        torch.testing.assert_close(embedding, reference[ids], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


@cuda_mark
@torch.inference_mode()
def test_moe_template_instance_types(monkeypatch):
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "upstream")
    n, k, experts, top_k = 37, 512, 3, 2
    ids = torch.tensor([[0, 1]] * 16, dtype=torch.int32, device="cuda")
    for quant_type in TEMPLATE_EXTRA_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        packed_row = k // block_size * type_size
        weight = torch.zeros((experts, n, packed_row), dtype=torch.uint8, device="cuda")
        for tokens in (2, 16):
            x = torch.zeros((tokens, k), dtype=torch.float32, device="cuda")
            output = torch.ops._C_gguf.ggml_moe_a8_upstream(
                x, weight, ids[:tokens], int(quant_type), n, top_k, tokens
            )
            assert output.shape == (tokens * top_k, n)
            assert bool(torch.isfinite(output).all())
            torch.testing.assert_close(output, torch.zeros_like(output))


@cuda_mark
@torch.inference_mode()
def test_moe_upstream_non_padded_k_reference(monkeypatch):
    """End-to-end: k that is not a multiple of 512 still reaches the kernel."""
    import vllm_gguf_plugin._C_gguf  # noqa: F401

    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "upstream")
    # k=1184 is 2*512+160, so the row needs a non-trivial padding tail.
    n, k, experts, top_k = 37, 1184, 3, 2
    quant_type = Q.Q4_0
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    packed_row = k // block_size * type_size
    source = np.random.default_rng(7).standard_normal((experts, n, k), dtype=np.float32)
    packed = gguf.quantize(source.reshape(-1, k), quant_type).reshape(
        experts, n, packed_row
    )
    weight = make_padded_weight(packed, quant_type, k)

    tokens = 4
    x = torch.randn((tokens, k), device="cuda", dtype=torch.float32)
    ids = torch.tensor([[0, 2]] * tokens, dtype=torch.int32, device="cuda")
    output = torch.ops._C_gguf.ggml_moe_a8_upstream(
        x, weight, ids, int(quant_type), n, top_k, tokens
    )
    assert output.shape == (tokens * top_k, n)
    assert bool(torch.isfinite(output).all())

    # Expert 0 for every token: route rows 0..top_k-1 of the flat output.
    dense0 = torch.from_numpy(gguf.dequantize(packed[0], quant_type)).cuda()
    reference = x @ dense0.T
    torch.testing.assert_close(
        output.view(tokens, top_k, n)[:, 0], reference, atol=1.5, rtol=0.2
    )
