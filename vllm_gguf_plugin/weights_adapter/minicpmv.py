# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GGUF weights adapter for MiniCPM-V (e.g. MiniCPM-V 4.5).

A MiniCPM-V GGUF checkpoint is split into two files:
  - backbone (``MiniCPM-V-4_5-*.gguf``): the text Qwen3 trunk, with
    qwen3/llama-style tensor names (``token_embd`` / ``blk.{i}.attn_q`` / ...)
    and no vision tensors.
  - mmproj (``mmproj-model-f16.gguf``): the vision tower (``v.*``) plus the
    resampler (``resampler.*``).

vLLM's ``MiniCPMV4_5`` expects the original HF layout instead:
  - text under ``llm.*`` (Qwen3ForCausalLM)
  - vision under ``vpm.*`` (Idefics2VisionTransformer)
  - ``resampler.*`` (Resampler4_5, whose attention is ``nn.MultiheadAttention``)

This adapter maps the GGUF tensor names onto that HF layout and reconciles two
structural differences:
  1. the resampler's ``attn.q/k/v`` are separate GGUF tensors and must be
     merged into ``in_proj_{weight,bias}``;
  2. ``resampler.pos_embed_k`` is recomputed at runtime by vLLM, so it is
     dropped.

Text norm weights are numerically identical to the HF checkpoint (no +/-1
offset needed) and quantized matrices are transposed automatically by the
plugin's dequantize path, so neither is handled here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import get_gguf_tensor_names
from .base import BaseGGUFWeightsAdapter, GGUFWeight

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

# The resampler's q/k/v are separate GGUF tensors; vLLM's
# nn.MultiheadAttention expects a single in_proj_{weight,bias}.
_RESAMPLER_QKV_RE = re.compile(r"^resampler\.attn\.([qkv])\.(weight|bias)$")

# Text backbone (Qwen3 style) plus the resampler, which lives in the
# backbone GGUF file. Quantized backbone matrices are transposed
# automatically by the dequantize path, so only name changes are listed here.
MINICPMV_TEXT_SUBSTR: dict[str, str] = {
    # backbone
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_output.": "self_attn.o_proj.",
    "attn_norm.": "input_layernorm.",
    "ffn_norm.": "post_attention_layernorm.",
    "ffn_gate.": "mlp.gate_proj.",
    "ffn_up.": "mlp.up_proj.",
    "ffn_down.": "mlp.down_proj.",
    # resampler
    "resampler.kv.": "resampler.kv_proj.",
    "resampler.attn.out.": "resampler.attn.out_proj.",
}

# Vision tower (v.* -> vpm.*). The mmproj file is F16 (read directly, not
# dequantized), so its values already use the HF layout; only names change.
# Numeric comparison (relerr ~1e-8) confirmed ffn_down -> fc1, ffn_up -> fc2.
MINICPMV_VISION_SUBSTR: dict[str, str] = {
    "ln1.": "layer_norm1.",
    "ln2.": "layer_norm2.",
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_out.": "self_attn.out_proj.",
    "ffn_down.": "mlp.fc1.",
    "ffn_up.": "mlp.fc2.",
}


def build_minicpmv_text_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": "llm.model.embed_tokens.",
            "blk.": "llm.model.layers.",
            "output_norm.": "llm.model.norm.",
            "output.": "llm.lm_head.",
            # resampler.proj is a bare Parameter in vLLM (no .weight suffix).
            "resampler.proj.weight": "resampler.proj",
            # Recomputed at runtime by vLLM; drop it.
            "resampler.pos_embed_k": None,
        },
        orig_to_new_substr=MINICPMV_TEXT_SUBSTR,
    )


def build_minicpmv_vision_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "v.patch_embd.": "vpm.embeddings.patch_embedding.",
            "v.position_embd.": "vpm.embeddings.position_embedding.",
            "v.post_ln.": "vpm.post_layernorm.",
            "v.blk.": "vpm.encoder.layers.",
        },
        orig_to_new_substr=MINICPMV_VISION_SUBSTR,
    )


def _map_tensor_name(mapper: WeightsMapper, name: str) -> str | None:
    # Unlike Gemma4, MiniCPM-V has legitimate identity mappings (e.g.
    # resampler.query / resampler.ln_*), so an unchanged name is kept rather
    # than treated as unmapped. Only an explicit drop (pos_embed_k) -> None.
    result = mapper.apply_list([name])
    return result[0] if result else None


class MiniCPMVGGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for MiniCPM-V text and multimodal GGUF models."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type == "minicpmv"

    @classmethod
    def architecture(cls, config) -> str | None:
        # vLLM already knows MiniCPMV (that is the original architecture name),
        # so just surface it to the config parser.
        return "MiniCPMV"

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        return maybe_patch_hf_config_from_gguf(
            files.primary_backbone,
            hf_config,
            mmproj_path=files.mm_proj,
        )

    @staticmethod
    def map_name(name: str) -> str | None:
        mapper = (
            build_minicpmv_vision_mapper()
            if name.startswith("v.")
            else build_minicpmv_text_mapper()
        )
        return _map_tensor_name(mapper, name)

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        del model_config
        text_mapper = build_minicpmv_text_mapper()
        vision_mapper = build_minicpmv_vision_mapper()
        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(files.all_files)):
            mapper = vision_mapper if name.startswith("v.") else text_mapper
            if mapped := _map_tensor_name(mapper, name):
                name_map[name] = mapped
            else:
                unmapped.append(name)
        if unmapped:
            logger.warning(
                "No HF name for %d MiniCPM-V GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        return name_map

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        """Transform mapped GGUF weights to the MiniCPM-V representation.

        1. ``resampler.proj`` is transposed: in vLLM it is a bare Parameter
           consumed as ``x @ proj`` (right-multiply), expecting [in, out].
        2. The resampler's separate ``attn.q/k/v`` tensors are merged into
           ``in_proj_{weight,bias}`` (concatenated in q, k, v order).

        Everything else already uses the HF layout (the F16 mmproj is read
        directly and the quantized backbone is dequantized with transposition),
        so no further transposition is applied.
        """
        del model_config
        pending: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in weights:
            if name == "resampler.proj":
                yield name, tensor.transpose(0, 1).contiguous()
                continue
            m = _RESAMPLER_QKV_RE.match(name)
            if m:
                qkv, suffix = m.groups()
                group = pending.setdefault(suffix, {})
                group[qkv] = tensor
                if set(group) == {"q", "k", "v"}:
                    del pending[suffix]
                    merged = torch.cat([group["q"], group["k"], group["v"]], dim=0)
                    yield f"resampler.attn.in_proj_{suffix}", merged
                continue
            yield name, tensor
        # Flush any incomplete group (should not happen in practice).
        for suffix, group in pending.items():
            if set(group) == {"q", "k", "v"}:
                yield (
                    f"resampler.attn.in_proj_{suffix}",
                    torch.cat([group["q"], group["k"], group["v"]], dim=0),
                )
