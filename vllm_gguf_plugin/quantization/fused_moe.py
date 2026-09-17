# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import torch
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from .. import ops
from ..kernel_support import (
    MOE_NOT_ELIGIBLE_MARKER,
    QuantizationBackend,
    QuantizationOperation,
    supports,
    supports_moe,
)
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_moe_weight_loader,
    _gguf_moe_weight_type_loader,
    _materialize_upstream_moe_storage_padding,
)
from .utils import logger


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)

    def act(inp: torch.Tensor):
        d = inp.shape[-1] // 2
        output_shape = inp.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
        apply_moe_activation(activation_enum, out, inp)
        return out

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    out_hidden_states = torch.empty_like(x)
    moe_mode = ops.cuda_moe_kernel_mode()
    upstream_types = supports_moe(
        weight_type, QuantizationBackend.UPSTREAM
    ) and supports_moe(weight_type2, QuantizationBackend.UPSTREAM)
    upstream_available = (
        upstream_types
        and ops.cuda_moe_upstream_kernel_available(weight_type)
        and ops.cuda_moe_upstream_kernel_available(weight_type2)
    )
    upstream_only_types = (
        supports(
            weight_type,
            QuantizationBackend.UPSTREAM,
            QuantizationOperation.DEQUANTIZE,
        )
        and not supports(
            weight_type, QuantizationBackend.LEGACY, QuantizationOperation.MMVQ
        )
    ) or (
        supports(
            weight_type2,
            QuantizationBackend.UPSTREAM,
            QuantizationOperation.DEQUANTIZE,
        )
        and not supports(
            weight_type2, QuantizationBackend.LEGACY, QuantizationOperation.MMVQ
        )
    )

    def fallback_supports(quant_type: int, operation: QuantizationOperation) -> bool:
        if moe_mode == "legacy":
            return supports(quant_type, QuantizationBackend.LEGACY, operation)
        if moe_mode == "triton":
            return supports(quant_type, QuantizationBackend.TRITON, operation)
        return supports(quant_type, QuantizationBackend.LEGACY, operation) or supports(
            quant_type, QuantizationBackend.TRITON, operation
        )

    if moe_mode == "upstream" and not upstream_available:
        raise RuntimeError(
            "upstream MoE CUDA kernel is unavailable for the selected "
            "quantization types or build"
        )
    if moe_mode == "legacy" and upstream_only_types:
        raise RuntimeError(
            "legacy MoE mode cannot handle upstream-only quantization types"
        )
    if (
        moe_mode in {"auto", "triton"}
        and not upstream_available
        and upstream_only_types
    ):
        raise RuntimeError(
            "upstream-only quantization types require the upstream MoE CUDA "
            "kernel; no legacy/Triton fallback is available"
        )
    if moe_mode == "triton" and not all(
        fallback_supports(quant_type, QuantizationOperation.MMVQ)
        or fallback_supports(quant_type, QuantizationOperation.MMQ)
        for quant_type in (weight_type, weight_type2)
    ):
        raise RuntimeError(
            "triton MoE backend is unavailable for the selected quantization types"
        )
    if moe_mode in {"upstream", "auto"} and upstream_available:
        num_tokens, _ = x.shape
        _, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        try:
            out = ops.ggml_moe_a8_upstream(
                x, w1, topk_ids, weight_type, N, top_k, num_tokens
            )
            out = act(out)
            flat_topk_ids = topk_ids.reshape(-1, 1)
            out = ops.ggml_moe_a8_upstream(
                out,
                w2,
                flat_topk_ids,
                weight_type2,
                w2.shape[1],
                1,
                num_tokens * top_k,
            )
            out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
                topk_weights.view(num_tokens, top_k, 1)
            )
            ops.moe_sum(out, out_hidden_states)
            return out_hidden_states
        except RuntimeError as error:
            # The upstream op embeds MOE_NOT_ELIGIBLE_MARKER (defined in both
            # kernel_support.py and bridge.cu) when the inputs cannot run on
            # the upstream MoE kernel; auto mode may then fall back.
            if moe_mode != "auto" or MOE_NOT_ELIGIBLE_MARKER not in str(error):
                raise
            if upstream_only_types:
                raise RuntimeError(
                    "upstream-only quantization types cannot fall back from "
                    "the upstream MoE CUDA kernel"
                ) from error

    if (
        fallback_supports(weight_type2, QuantizationOperation.MMQ)
        and fallback_supports(weight_type, QuantizationOperation.MMQ)
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        block_size = ops.ggml_moe_get_block_size(weight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, E
        )
        out = ops.ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            weight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ops.ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            weight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    elif fallback_supports(
        weight_type2, QuantizationOperation.MMVQ
    ) and fallback_supports(weight_type, QuantizationOperation.MMVQ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ops.ggml_moe_a8_vec(x, w1, topk_ids, top_k, weight_type, N, num_tokens)
        out = act(out)

        out = ops.ggml_moe_a8_vec(
            out, w2, topk_ids, 1, weight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    else:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                out = fused_mul_mat_gguf_op(inp, w1[ii], weight_type)
                out = act(out)
                current_state = fused_mul_mat_gguf_op(out, w2[ii], weight_type2).mul_(
                    ww
                )
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weight_type: int,
    weight_type2: int,
    activation: str,
) -> torch.Tensor:
    del w1, w2, topk_weights, topk_ids, weight_type, weight_type2, activation
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf
except AttributeError as error:
    raise error


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF."""

    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        base_weight_loader = extra_weight_attrs.pop("weight_loader")
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w13_weight", w13_weight)

        w13_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w13_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w13_weight_type, extra_weight_attrs)
        layer.register_parameter("w13_weight_type", w13_weight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_weight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)

        w2_weight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w2_weight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w2_weight_type, extra_weight_attrs)
        layer.register_parameter("w2_weight_type", w2_weight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Reserve the upstream MATRIX_ROW_PADDING storage tail on MoE weights.

        Runs after every expert shard is loaded. The upstream MoE kernels
        validate the trailing storage at launch time, and the per-shard
        loader only pads 2D dense weights, so the 3D MoE parameters are
        padded here using each parameter's quantization type.
        """
        _materialize_upstream_moe_storage_padding(
            layer.w13_weight, layer.w13_weight_type.weight_type
        )
        _materialize_upstream_moe_storage_padding(
            layer.w2_weight, layer.w2_weight_type.weight_type
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        from . import fused_moe_gguf as fused_moe_gguf_op

        return fused_moe_gguf_op(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            layer.w13_weight_type.weight_type,
            layer.w2_weight_type.weight_type,
            layer.activation.value,
        )
