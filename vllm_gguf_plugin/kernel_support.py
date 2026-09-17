# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import gguf
from gguf import GGMLQuantizationType as WeightType


class QuantizationBackend(str, Enum):
    TRITON = "triton"
    LEGACY = "legacy"
    UPSTREAM = "upstream"


class QuantizationOperation(str, Enum):
    DEQUANTIZE = "dequantize"
    MMVQ = "mmvq"
    MMQ = "mmq"


@dataclass(frozen=True, slots=True)
class QuantizationSupport:
    """Kernel capabilities for one GGML quantization type."""

    triton: frozenset[QuantizationOperation] = frozenset()
    legacy: frozenset[QuantizationOperation] = frozenset()
    upstream: frozenset[QuantizationOperation] = frozenset()

    def supports(
        self, backend: QuantizationBackend, operation: QuantizationOperation
    ) -> bool:
        return operation in getattr(self, backend.value)


def _types(*names: str) -> tuple[WeightType, ...]:
    """Resolve optional GGUF Python enum members in one place."""
    return tuple(
        value
        for name in names
        if (value := getattr(WeightType, name, None)) is not None
    )


_TRITON_DEQUANT_MMVQ = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
)
_TRITON_ALL = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
    QuantizationOperation.MMQ,
)
_LEGACY_DEQUANT_MMVQ = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
)
_LEGACY_ALL = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
    QuantizationOperation.MMQ,
)
_UPSTREAM_DEQUANT_MMVQ = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
)
_UPSTREAM_ALL = (
    QuantizationOperation.DEQUANTIZE,
    QuantizationOperation.MMVQ,
    QuantizationOperation.MMQ,
)


def _support(
    *,
    triton: Iterable[QuantizationOperation] = (),
    legacy: Iterable[QuantizationOperation] = (),
    upstream: Iterable[QuantizationOperation] = (),
) -> QuantizationSupport:
    return QuantizationSupport(
        triton=frozenset(triton),
        legacy=frozenset(legacy),
        upstream=frozenset(upstream),
    )


# This is the single Python-side capability table. Keep the groups disjoint so
# every row states the complete support contract for its quantization family.
_STANDARD_K_TYPES = _types(
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
)
_IQ_TYPES = _types(
    "IQ1_M",
    "IQ1_S",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ3_XXS",
    "IQ3_S",
    "IQ4_XS",
    "IQ4_NL",
)
_UPSTREAM_EXTRA_TYPES = _types("Q1_0", "Q2_0", "MXFP4", "NVFP4")

_SUPPORT: dict[WeightType, QuantizationSupport] = {}


def _register(types: Iterable[WeightType], support: QuantizationSupport) -> None:
    for weight_type in types:
        if weight_type in _SUPPORT:
            raise AssertionError(f"duplicate support row for {weight_type}")
        _SUPPORT[weight_type] = support


# Q8_1 is available through Triton's generic quantized path only.
_register(_types("Q8_1"), _support(triton=_TRITON_ALL))

# Standard and K-quant formats have all three operations in every CUDA path.
_register(
    _STANDARD_K_TYPES,
    _support(
        triton=_TRITON_ALL,
        legacy=_LEGACY_ALL,
        upstream=_UPSTREAM_ALL,
    ),
)

# IQ1_M has no upstream MMQ instance; the other IQ formats do.
_register(
    _types("IQ1_M"),
    _support(
        triton=_TRITON_DEQUANT_MMVQ,
        legacy=_LEGACY_DEQUANT_MMVQ,
        upstream=_UPSTREAM_DEQUANT_MMVQ,
    ),
)
_register(
    _types(
        "IQ1_S",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_S",
        "IQ4_XS",
        "IQ4_NL",
    ),
    _support(
        triton=_TRITON_DEQUANT_MMVQ,
        legacy=_LEGACY_DEQUANT_MMVQ,
        upstream=_UPSTREAM_ALL,
    ),
)

# These formats are provided by the upstream template/conversion instances;
# they are intentionally not advertised as legacy or Triton capabilities.
_register(_UPSTREAM_EXTRA_TYPES, _support(upstream=_UPSTREAM_ALL))


def get_quantization_support(weight_type: int) -> QuantizationSupport:
    try:
        return _SUPPORT.get(WeightType(weight_type), QuantizationSupport())
    except (TypeError, ValueError):
        return QuantizationSupport()


def supports(
    weight_type: int,
    backend: QuantizationBackend,
    operation: QuantizationOperation,
) -> bool:
    return get_quantization_support(weight_type).supports(backend, operation)


def supports_moe(weight_type: int, backend: QuantizationBackend) -> bool:
    """MoE projections use either the MMVQ or MMQ kernel by token count."""
    support = get_quantization_support(weight_type)
    return support.supports(backend, QuantizationOperation.MMVQ) or support.supports(
        backend, QuantizationOperation.MMQ
    )


def _types_for(
    backend: QuantizationBackend, operation: QuantizationOperation
) -> frozenset[WeightType]:
    return frozenset(
        weight_type
        for weight_type, support in _SUPPORT.items()
        if support.supports(backend, operation)
    )


# Compatibility exports. New code should use supports()/supports_moe() so the
# operation and backend remain explicit at the call site.
CUDA_LEGACY_MMVQ_TYPES = _types_for(
    QuantizationBackend.LEGACY, QuantizationOperation.MMVQ
)
CUDA_LEGACY_MMQ_TYPES = _types_for(
    QuantizationBackend.LEGACY, QuantizationOperation.MMQ
)
CUDA_UPSTREAM_MMVQ_TYPES = _types_for(
    QuantizationBackend.UPSTREAM, QuantizationOperation.MMVQ
)
CUDA_UPSTREAM_MMQ_TYPES = _types_for(
    QuantizationBackend.UPSTREAM, QuantizationOperation.MMQ
)
CUDA_UPSTREAM_DEQUANT_TYPES = _types_for(
    QuantizationBackend.UPSTREAM, QuantizationOperation.DEQUANTIZE
)
CUDA_UPSTREAM_MOE_TYPES = frozenset(
    weight_type
    for weight_type in _SUPPORT
    if supports_moe(weight_type, QuantizationBackend.UPSTREAM)
)

TRITON_DEQUANT_TYPES = _types_for(
    QuantizationBackend.TRITON, QuantizationOperation.DEQUANTIZE
)
TRITON_MMVQ_TYPES = _types_for(QuantizationBackend.TRITON, QuantizationOperation.MMVQ)
TRITON_MMQ_TYPES = _types_for(QuantizationBackend.TRITON, QuantizationOperation.MMQ)

UNQUANTIZED_TYPES = frozenset(_types("F32", "F16", "BF16"))
STANDARD_QUANT_TYPES = frozenset(_types("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1"))
KQUANT_TYPES = frozenset(_types("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"))
IMATRIX_QUANT_TYPES = frozenset(_IQ_TYPES)

_UPSTREAM_STORAGE_TYPES = CUDA_UPSTREAM_MMVQ_TYPES | CUDA_UPSTREAM_MMQ_TYPES
_MATRIX_ROW_PADDING = 512


def upstream_storage_padding_bytes(weight_type: int, packed_row_size: int) -> int:
    """Return the extra byte storage required by upstream dense CUDA kernels."""
    if weight_type not in _UPSTREAM_STORAGE_TYPES or packed_row_size <= 0:
        return 0
    block_size, type_size = gguf.GGML_QUANT_SIZES[WeightType(weight_type)]
    if packed_row_size % type_size:
        return 0
    logical_k = packed_row_size // type_size * block_size
    padding_k = (-logical_k) % _MATRIX_ROW_PADDING
    return padding_k // block_size * type_size
