"""Kernel selector tests: environment variables and Python-side routing.

These tests never touch CUDA.  They verify that

1. the ``VLLM_GGUF_CUDA_{KERNEL,DENSE_KERNEL,MOE_KERNEL,DEQUANTIZE_KERNEL}``
   environment variables are parsed with the documented precedence and
   defaults,
2. the support matrix published by ``kernel_support`` / ``ops`` is explicit
   and internally consistent, and
3. the Python-side dispatch picks the intended backend for a given mode and
   quant type (mocked, no actual kernel launch).
"""

import pytest
import torch
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as Q

KERNEL_ENV_VARS = (
    "VLLM_GGUF_CUDA_KERNEL",
    "VLLM_GGUF_CUDA_DENSE_KERNEL",
    "VLLM_GGUF_CUDA_MOE_KERNEL",
    "VLLM_GGUF_CUDA_DEQUANTIZE_KERNEL",
)

TEMPLATE_EXTRA_TYPES = tuple(
    getattr(Q, name) for name in ("Q1_0", "Q2_0", "MXFP4", "NVFP4") if hasattr(Q, name)
)


def _clear_kernel_environment(monkeypatch):
    for name in KERNEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Environment variable parsing
# ---------------------------------------------------------------------------


def test_kernel_selector_defaults_and_global_override(monkeypatch):
    from vllm_gguf_plugin import ops

    _clear_kernel_environment(monkeypatch)
    assert ops.cuda_dense_kernel_mode() == "auto"
    assert ops.cuda_moe_kernel_mode() == "auto"
    assert ops.cuda_dequantize_kernel_mode() == "auto"
    assert ops.cuda_kernel_mode() == "auto"

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "legacy")
    assert ops.cuda_dense_kernel_mode() == "legacy"
    assert ops.cuda_moe_kernel_mode() == "legacy"
    assert ops.cuda_dequantize_kernel_mode() == "legacy"

    monkeypatch.setenv("VLLM_GGUF_CUDA_DENSE_KERNEL", "upstream")
    monkeypatch.setenv("VLLM_GGUF_CUDA_MOE_KERNEL", "triton")
    assert ops.cuda_dense_kernel_mode() == "upstream"
    assert ops.cuda_moe_kernel_mode() == "triton"
    assert ops.cuda_dequantize_kernel_mode() == "legacy"

    monkeypatch.setenv("VLLM_GGUF_CUDA_DEQUANTIZE_KERNEL", "upstream")
    assert ops.cuda_dequantize_kernel_mode() == "upstream"


def test_kernel_selector_rejects_invalid_values(monkeypatch):
    from vllm_gguf_plugin import ops

    _clear_kernel_environment(monkeypatch)
    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", "invalid")
    with pytest.raises(ValueError, match="auto\\|upstream\\|legacy\\|triton"):
        ops.cuda_dense_kernel_mode()


# ---------------------------------------------------------------------------
# Support matrix
# ---------------------------------------------------------------------------


def test_support_matrix_is_explicit():
    from vllm_gguf_plugin import ops

    extra_types = {int(q) for q in TEMPLATE_EXTRA_TYPES}
    standard_mmvq = {
        ops.GGML_TYPE_Q4_0,
        ops.GGML_TYPE_Q4_1,
        ops.GGML_TYPE_Q5_0,
        ops.GGML_TYPE_Q5_1,
        ops.GGML_TYPE_Q8_0,
        ops.GGML_TYPE_Q2_K,
        ops.GGML_TYPE_Q3_K,
        ops.GGML_TYPE_Q4_K,
        ops.GGML_TYPE_Q5_K,
        ops.GGML_TYPE_Q6_K,
        ops.GGML_TYPE_IQ2_XXS,
        ops.GGML_TYPE_IQ2_XS,
        ops.GGML_TYPE_IQ3_XXS,
        ops.GGML_TYPE_IQ1_S,
        ops.GGML_TYPE_IQ4_NL,
        ops.GGML_TYPE_IQ3_S,
        ops.GGML_TYPE_IQ2_S,
        ops.GGML_TYPE_IQ4_XS,
        ops.GGML_TYPE_IQ1_M,
    }
    mmq_subset = standard_mmvq - {ops.GGML_TYPE_IQ1_M}

    assert standard_mmvq | extra_types == ops.CUDA_UPSTREAM_MMVQ_TYPES
    assert mmq_subset | extra_types == ops.CUDA_UPSTREAM_MMQ_TYPES
    assert ops.CUDA_LEGACY_MMVQ_TYPES < ops.CUDA_UPSTREAM_MMVQ_TYPES
    assert ops.CUDA_LEGACY_MMQ_TYPES < ops.CUDA_UPSTREAM_MMQ_TYPES
    assert ops.CUDA_UPSTREAM_DEQUANT_TYPES == ops.CUDA_UPSTREAM_MMVQ_TYPES
    assert ops.CUDA_UPSTREAM_MOE_TYPES == ops.CUDA_UPSTREAM_MMVQ_TYPES


def test_all_upstream_template_instances_are_listed():
    """Every mmq-instance-*.cu in the submodule is registered in the manifest."""
    from pathlib import Path

    import tomllib

    repo_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads(
        (repo_root / "vllm_gguf_plugin" / "llama_cpp_upstream.toml").read_text()
    )
    actual = {
        path.relative_to(repo_root / "third_party" / "llama.cpp").as_posix()
        for path in (
            repo_root
            / "third_party"
            / "llama.cpp"
            / "ggml"
            / "src"
            / "ggml-cuda"
            / "template-instances"
        ).glob("mmq-instance-*.cu")
    }
    listed = set(metadata["sources"])
    assert actual <= listed


# ---------------------------------------------------------------------------
# Python-side routing (mocked backends, no CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "legacy", "upstream"])
def test_iq_mmq_routes_to_an_implemented_backend(monkeypatch, mode):
    from vllm_gguf_plugin import ops

    monkeypatch.setenv("VLLM_GGUF_CUDA_KERNEL", mode)
    monkeypatch.setattr(ops, "_cuda_kernel_available", lambda *args: True)
    for quant in ops.CUDA_UPSTREAM_MMQ_TYPES - ops.CUDA_LEGACY_MMQ_TYPES:
        assert ops._cuda_gemm_kernel_available("ggml_mul_mat_a8", quant) == (
            mode in {"auto", "upstream"} and torch.version.hip is None
        )


def test_linear_dispatch_uses_upstream_mmq_above_mmvq_limit(monkeypatch):
    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    calls = []

    def mmvq(weight, x, weight_type, row):
        calls.append("mmvq")
        return torch.empty((x.shape[0], row), dtype=x.dtype)

    def mmq(weight, x, weight_type, row):
        calls.append("mmq")
        return torch.empty((x.shape[0], row), dtype=x.dtype)

    monkeypatch.setattr(ops, "cuda_dense_upstream_enabled", lambda: True)
    monkeypatch.setattr(ops, "ggml_mul_mat_vec_a8", mmvq)
    monkeypatch.setattr(ops, "ggml_mul_mat_a8", mmq)
    quant_type = Q.IQ4_NL
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    weight = torch.zeros((37, 256 // block_size * type_size), dtype=torch.uint8)

    _fused_mul_mat_gguf(torch.zeros((8, 256)), weight, int(quant_type))
    _fused_mul_mat_gguf(torch.zeros((9, 256)), weight, int(quant_type))

    assert calls == ["mmvq", "mmq"]
