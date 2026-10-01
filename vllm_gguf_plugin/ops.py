# SPDX-License-Identifier: Apache-2.0

import os
import typing

import torch

from .kernel_support import (  # noqa: F401
    CUDA_LEGACY_MMQ_TYPES,
    CUDA_LEGACY_MMVQ_TYPES,
    CUDA_UPSTREAM_DEQUANT_TYPES,
    CUDA_UPSTREAM_MMQ_TYPES,
    CUDA_UPSTREAM_MMVQ_TYPES,
    CUDA_UPSTREAM_MOE_TYPES,
    QuantizationBackend,
    QuantizationOperation,
    supports,
    supports_moe,
)
from .triton.dequantize.interface import ggml_dequantize_triton
from .triton.fused_moe.interface import ggml_moe_a8_triton
from .triton.fused_moe.utils import get_triton_moe_block_m
from .triton.gemm.interface import ggml_mul_mat_a8_triton

# Public re-exports: tests and external consumers reference
# `ops.GGML_TYPE_*` as stable numeric constants, so keep them importable
# from this module even though backend capability tables now live in
# kernel_support.py.
from .triton.gemm.utils import (  # noqa: F401
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
)

try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

# Backend selection: use CUDA kernels by default, unless explicitly disabled.
_USE_CUDA = os.environ.get("VLLM_GGUF_USE_CUDA", "1") == "1"

# Try importing CUDA extension
try:
    from . import _C_gguf  # noqa: F401

    _CUDA_AVAILABLE = True
except ImportError:
    _C_gguf = None
    _CUDA_AVAILABLE = False


# Effective CUDA usage: only when enabled AND available.
_CUDA_ENABLED = _USE_CUDA and _CUDA_AVAILABLE

_GLOBAL_KERNEL_ENV = "VLLM_GGUF_CUDA_KERNEL"
_VALID_KERNEL_MODES = {"auto", "upstream", "legacy", "triton"}


def _kernel_mode(env_name: str) -> str:
    mode = os.environ.get(env_name)
    if mode is None:
        mode = os.environ.get(_GLOBAL_KERNEL_ENV, "auto")
    if mode not in _VALID_KERNEL_MODES:
        raise ValueError(f"{env_name} must be one of auto|upstream|legacy|triton")
    return mode


def cuda_dense_kernel_mode() -> str:
    return _kernel_mode("VLLM_GGUF_CUDA_DENSE_KERNEL")


def cuda_moe_kernel_mode() -> str:
    return _kernel_mode("VLLM_GGUF_CUDA_MOE_KERNEL")


def cuda_dequantize_kernel_mode() -> str:
    return _kernel_mode("VLLM_GGUF_CUDA_DEQUANTIZE_KERNEL")


def cuda_kernel_mode() -> str:
    """Alias for :func:`cuda_dense_kernel_mode` (public selector API)."""
    return cuda_dense_kernel_mode()


def cuda_dense_upstream_enabled() -> bool:
    return (
        _CUDA_ENABLED
        and torch.version.hip is None
        and cuda_dense_kernel_mode() in {"upstream", "auto"}
    )


def cuda_upstream_enabled() -> bool:
    """Alias for :func:`cuda_dense_upstream_enabled` (public selector API)."""
    return cuda_dense_upstream_enabled()


def should_use_upstream_mmvq(X: torch.Tensor, quant_type: int) -> bool:
    """Query the pinned llama.cpp policy for the input tensor's device.

    This is a host-only query; it neither launches a kernel nor synchronizes.
    Keep the architecture/type thresholds in upstream rather than copying them
    into the Python dispatcher. Storage eligibility is checked by the bridge.
    """
    major, minor = torch.cuda.get_device_capability(X.device)
    cc = major * 100 + minor * 10
    return torch.ops._C_gguf.ggml_should_use_mmvq(quant_type, cc, X.shape[0])


DENSE_MMVF = 1
DENSE_MMF = 2
DENSE_MMVQ = 4
DENSE_MMQ = 8
DENSE_BLAS = 16


def dense_upstream_capabilities(
    W: torch.Tensor, X: torch.Tensor, quant_type: int, row: int
) -> int:
    return torch.ops._C_gguf.ggml_dense_upstream_capabilities(W, X, quant_type, row)


def ggml_dense_mmvq(W, X, quant_type, row):
    return torch.ops._C_gguf.ggml_dense_mmvq(W, X, quant_type, row)


def ggml_dense_mmq(W, X, quant_type, row):
    return torch.ops._C_gguf.ggml_dense_mmq(W, X, quant_type, row)


def ggml_dense_mmvf(W, X, quant_type, row):
    return torch.ops._C_gguf.ggml_dense_mmvf(W, X, quant_type, row)


def ggml_dense_mmf(W, X, quant_type, row):
    return torch.ops._C_gguf.ggml_dense_mmf(W, X, quant_type, row)


def ggml_dense_blas(W, X, quant_type, row):
    return torch.ops._C_gguf.ggml_dense_blas(W, X, quant_type, row)


def cuda_dequantize_upstream_enabled() -> bool:
    return (
        _CUDA_ENABLED
        and torch.version.hip is None
        and cuda_dequantize_kernel_mode() in {"upstream", "auto"}
    )


def _cuda_kernel_available(op_name: str, quant_type: int | None = None) -> bool:
    if not _CUDA_ENABLED:
        return False
    namespace = getattr(torch.ops, "_C_gguf", None)
    if namespace is None or not hasattr(namespace, op_name):
        return False
    if quant_type is None:
        return True
    return supports(
        quant_type,
        QuantizationBackend.LEGACY,
        QuantizationOperation.MMVQ,
    )


def _cuda_legacy_moe_available(
    op_name: str, quant_type: int, operation: QuantizationOperation
) -> bool:
    return _cuda_kernel_available(op_name) and supports(
        quant_type, QuantizationBackend.LEGACY, operation
    )


def _cuda_moe_upstream_kernel_available(op_name: str, quant_type: int) -> bool:
    return (
        _CUDA_ENABLED
        and torch.version.hip is None
        and _cuda_kernel_available(op_name)
        and supports_moe(quant_type, QuantizationBackend.UPSTREAM)
    )


# --- Fake implementations for CUDA custom ops (needed for torch.compile) ---

if (
    _CUDA_AVAILABLE
    and hasattr(torch.ops, "_C_gguf")
    and hasattr(torch.ops._C_gguf, "ggml_dequantize")
):

    @register_fake("_C_gguf::ggml_dequantize")
    def _ggml_dequantize_fake(
        W: torch.Tensor,
        quant_type: int,
        m: torch.SymInt,
        n: torch.SymInt,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.empty((m, n), dtype=dtype or torch.float16, device=W.device)

    @register_fake("_C_gguf::ggml_dense_mmvq")
    @register_fake("_C_gguf::ggml_dense_mmq")
    @register_fake("_C_gguf::ggml_dense_mmvf")
    @register_fake("_C_gguf::ggml_dense_mmf")
    @register_fake("_C_gguf::ggml_dense_blas")
    def _ggml_dense_explicit_fake(W, X, quant_type, row):
        return torch.empty((X.size(0), row), dtype=X.dtype, device=X.device)

    @register_fake("_C_gguf::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)

    @register_fake("_C_gguf::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.size(0), row), dtype=X.dtype, device=W.device)

    @register_fake("_C_gguf::ggml_moe_a8")
    def _ggml_moe_a8_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
        top_k: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.size(0) * top_k, row), dtype=X.dtype, device=W.device)


if (
    _CUDA_AVAILABLE
    and hasattr(torch.ops, "_C_gguf")
    and hasattr(torch.ops._C_gguf, "ggml_moe_a8_upstream")
):

    @register_fake("_C_gguf::ggml_moe_a8_upstream")
    def _ggml_moe_a8_upstream_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
        top_k: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        del topk_ids, quant_type, tokens
        return torch.empty((X.size(0) * top_k, row), dtype=X.dtype, device=W.device)

    for _moe_op in ("ggml_moe_mmvq", "ggml_moe_mmq", "ggml_moe_grouped_dense"):
        if hasattr(torch.ops._C_gguf, _moe_op):
            register_fake(f"_C_gguf::{_moe_op}")(_ggml_moe_a8_upstream_fake)

    @register_fake("_C_gguf::ggml_moe_upstream")
    def _ggml_moe_upstream_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_type: int,
        row: torch.SymInt,
        top_k: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        return _ggml_moe_a8_upstream_fake(
            X, W, topk_ids, quant_type, row, top_k, tokens
        )


if (
    _CUDA_AVAILABLE
    and hasattr(torch.ops, "_C_gguf")
    and hasattr(torch.ops._C_gguf, "ggml_dequantize_upstream")
):

    @register_fake("_C_gguf::ggml_dequantize_upstream")
    def _ggml_dequantize_upstream_fake(
        W: torch.Tensor,
        quant_type: int,
        m: torch.SymInt,
        n: torch.SymInt,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del quant_type
        return torch.empty((m, n), dtype=dtype or torch.float16, device=W.device)


if (
    _CUDA_AVAILABLE
    and hasattr(torch.ops, "_C_gguf")
    and hasattr(torch.ops._C_gguf, "ggml_moe_a8_vec")
):

    @register_fake("_C_gguf::ggml_moe_a8_vec")
    def _ggml_moe_a8_vec_fake(
        X: torch.Tensor,
        W: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        quant_type: int,
        row: torch.SymInt,
        tokens: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.size(0) * top_k, row), dtype=X.dtype, device=W.device)


# --- Public API ---


def _cuda_upstream_supports(
    op_name: str, quant_type: int, operation: QuantizationOperation
) -> bool:
    return (
        _CUDA_ENABLED
        and torch.version.hip is None
        and _cuda_kernel_available(op_name)
        and supports(quant_type, QuantizationBackend.UPSTREAM, operation)
    )


def _raise_backend_unavailable(
    backend: str, operation: str, quant_type: int
) -> typing.NoReturn:
    raise RuntimeError(
        f"{backend} {operation} backend is unavailable for quantization type "
        f"{quant_type}"
    )


def ggml_dequantize(
    W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    mode = cuda_dequantize_kernel_mode()
    upstream_available = _cuda_upstream_supports(
        "ggml_dequantize_upstream",
        quant_type,
        QuantizationOperation.DEQUANTIZE,
    )
    legacy_available = _cuda_kernel_available("ggml_dequantize", quant_type)
    if mode == "upstream":
        if not upstream_available:
            _raise_backend_unavailable("upstream", "dequantize", quant_type)
        return torch.ops._C_gguf.ggml_dequantize_upstream(W, quant_type, m, n, dtype)
    if mode == "legacy":
        if not legacy_available:
            _raise_backend_unavailable("legacy", "dequantize", quant_type)
        return torch.ops._C_gguf.ggml_dequantize(W, quant_type, m, n, dtype)
    if mode == "triton":
        if not supports(
            quant_type, QuantizationBackend.TRITON, QuantizationOperation.DEQUANTIZE
        ):
            _raise_backend_unavailable("triton", "dequantize", quant_type)
        return ggml_dequantize_triton(W, quant_type, m, n, dtype)
    if upstream_available:
        return torch.ops._C_gguf.ggml_dequantize_upstream(W, quant_type, m, n, dtype)
    if legacy_available:
        return torch.ops._C_gguf.ggml_dequantize(W, quant_type, m, n, dtype)
    if supports(
        quant_type, QuantizationBackend.TRITON, QuantizationOperation.DEQUANTIZE
    ):
        return ggml_dequantize_triton(W, quant_type, m, n, dtype)
    _raise_backend_unavailable("auto", "dequantize", quant_type)


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    # The raw Torch op with this name is a fixed legacy entry point. Choose
    # the upstream op here so C++ never performs a second backend selection.
    mode = cuda_dense_kernel_mode()
    if mode in {"upstream", "auto"}:
        if _cuda_upstream_supports(
            "ggml_dense_mmvq", quant_type, QuantizationOperation.MMVQ
        ):
            return ggml_dense_mmvq(W, X, quant_type, row)
        if mode == "upstream":
            _raise_backend_unavailable(mode, "MMVQ", quant_type)

    if mode in {"legacy", "auto"}:
        if _cuda_kernel_available("ggml_mul_mat_vec_a8", quant_type):
            return torch.ops._C_gguf.ggml_mul_mat_vec_a8(W, X, quant_type, row)
        if mode == "legacy":
            _raise_backend_unavailable(mode, "MMVQ", quant_type)

    if mode in {"triton", "auto"} and supports(
        quant_type, QuantizationBackend.TRITON, QuantizationOperation.MMVQ
    ):
        return ggml_mul_mat_a8_triton(W, X, quant_type, row)
    _raise_backend_unavailable(mode, "MMVQ", quant_type)


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    # Keep MMQ versus BLAS selection in Python; the raw Torch op is legacy.
    mode = cuda_dense_kernel_mode()
    if mode in {"upstream", "auto"}:
        if _cuda_upstream_supports(
            "ggml_dense_upstream_capabilities",
            quant_type,
            QuantizationOperation.DEQUANTIZE,
        ):
            caps = dense_upstream_capabilities(W, X, quant_type, row)
            if caps & DENSE_MMQ:
                return ggml_dense_mmq(W, X, quant_type, row)
            if caps & DENSE_BLAS:
                return ggml_dense_blas(W, X, quant_type, row)
            # The upstream backend exists, but these inputs have no valid route.
            _raise_backend_unavailable("upstream", "MMQ/BLAS", quant_type)
        if mode == "upstream":
            _raise_backend_unavailable(mode, "MMQ", quant_type)

    if mode in {"legacy", "auto"}:
        if supports(
            quant_type, QuantizationBackend.LEGACY, QuantizationOperation.MMQ
        ) and _cuda_kernel_available("ggml_mul_mat_a8"):
            return torch.ops._C_gguf.ggml_mul_mat_a8(W, X, quant_type, row)
        if mode == "legacy":
            _raise_backend_unavailable(mode, "MMQ", quant_type)

    if mode in {"triton", "auto"} and (
        supports(quant_type, QuantizationBackend.TRITON, QuantizationOperation.MMQ)
        or supports(
            quant_type,
            QuantizationBackend.TRITON,
            QuantizationOperation.DEQUANTIZE,
        )
    ):
        return ggml_mul_mat_a8_triton(W, X, quant_type, row)
    _raise_backend_unavailable(mode, "MMQ", quant_type)


def cuda_moe_upstream_kernel_available(quant_type: int) -> bool:
    return _cuda_moe_upstream_kernel_available("ggml_moe_upstream", quant_type)


def ggml_moe_mmvq(X, W, topk_ids, quant_type, row, top_k, tokens):
    """Force upstream MoE MMVQ, with internal token chunking and no fallback."""
    return torch.ops._C_gguf.ggml_moe_mmvq(
        X, W, topk_ids, quant_type, row, top_k, tokens
    )


def ggml_moe_mmq(X, W, topk_ids, quant_type, row, top_k, tokens):
    """Force upstream MoE MMQ, with internal token chunking and no fallback."""
    return torch.ops._C_gguf.ggml_moe_mmq(
        X, W, topk_ids, quant_type, row, top_k, tokens
    )


def ggml_moe_grouped_dense(X, W, topk_ids, quant_type, row, top_k, tokens):
    """Group routes by expert and use dense dispatch; CUDA graphs are unsupported."""
    return torch.ops._C_gguf.ggml_moe_grouped_dense(
        X, W, topk_ids, quant_type, row, top_k, tokens
    )


def ggml_moe_upstream(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    if _cuda_moe_upstream_kernel_available("ggml_moe_upstream", quant_type):
        return torch.ops._C_gguf.ggml_moe_upstream(
            X, W, topk_ids, quant_type, row, top_k, tokens
        )
    raise RuntimeError(
        f"upstream MoE CUDA kernel is unavailable for quantization type {quant_type}"
    )


def ggml_moe_a8_upstream(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """Compatibility alias for the quantized MoE entry point."""
    return ggml_moe_upstream(X, W, topk_ids, quant_type, row, top_k, tokens)


def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    mode = cuda_moe_kernel_mode()
    if mode == "upstream":
        _raise_backend_unavailable(mode, "MoE MMQ", quant_type)

    if mode in {"legacy", "auto"}:
        if _cuda_legacy_moe_available(
            "ggml_moe_a8", quant_type, QuantizationOperation.MMQ
        ):
            return torch.ops._C_gguf.ggml_moe_a8(
                X,
                W,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                quant_type,
                row,
                top_k,
                tokens,
            )
        if mode == "legacy":
            _raise_backend_unavailable(mode, "MoE MMQ", quant_type)

    if supports(quant_type, QuantizationBackend.TRITON, QuantizationOperation.MMQ):
        return ggml_moe_a8_triton(
            X,
            W,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            quant_type,
            row,
            top_k,
            tokens,
        )
    _raise_backend_unavailable(mode, "MoE MMQ", quant_type)


def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    mode = cuda_moe_kernel_mode()
    if mode == "upstream":
        _raise_backend_unavailable(mode, "MoE MMVQ", quant_type)

    if mode in {"legacy", "auto"}:
        if _cuda_legacy_moe_available(
            "ggml_moe_a8_vec", quant_type, QuantizationOperation.MMVQ
        ):
            return torch.ops._C_gguf.ggml_moe_a8_vec(
                X, W, topk_ids, top_k, quant_type, row, tokens
            )
        if mode == "legacy":
            _raise_backend_unavailable(mode, "MoE MMVQ", quant_type)

    if supports(quant_type, QuantizationBackend.TRITON, QuantizationOperation.MMVQ):
        from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

        E = W.shape[0]
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, get_triton_moe_block_m(quant_type), E
        )
        return ggml_moe_a8_triton(
            X,
            W,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            quant_type,
            row,
            top_k,
            tokens,
        )
    _raise_backend_unavailable(mode, "MoE MMVQ", quant_type)


def ggml_moe_get_block_size(quant_type: int) -> int:
    mode = cuda_moe_kernel_mode()
    if mode == "upstream":
        _raise_backend_unavailable(mode, "MoE block-size", quant_type)
    if mode in {"legacy", "auto"}:
        if _cuda_legacy_moe_available(
            "ggml_moe_get_block_size", quant_type, QuantizationOperation.MMQ
        ):
            return torch.ops._C_gguf.ggml_moe_get_block_size(quant_type)
        if mode == "legacy":
            _raise_backend_unavailable(mode, "MoE block-size", quant_type)
    # This helper describes Triton's layout; the caller checks kernel support.
    return get_triton_moe_block_m(quant_type)


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    torch.ops._moe_C.moe_sum(input, output)
