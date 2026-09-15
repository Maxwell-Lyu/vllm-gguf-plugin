# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GGUF weights adapter for MiniCPM-V 4.6 (MiniCPMV4_6ForConditionalGeneration).

A MiniCPM-V 4.6 GGUF checkpoint is split into two files:
  - backbone (``MiniCPM-V-4_6-*.gguf``): a Qwen3.5 hybrid text trunk
    (GDN linear-attention + full-attention), internal arch ``qwen35``, tensor
    names ``blk.*`` / ``token_embd.*`` / ``output_norm.*``. It has no
    ``output.weight`` (lm_head is tied to the embedding).
  - mmproj (``mmproj-model-f16.gguf``): a SigLIP-style vision tower (``v.*``),
    a window-attention ``vit_merger`` (``v.vit_merger.*``) and a final
    DownsampleMLP projector (``mm.*``).

vLLM's ``MiniCPMV4_6ForConditionalGeneration`` expects canonical HF names:
  - text under ``model.language_model.*`` (Qwen3_5ForCausalLM)
  - vision under ``model.vision_tower.*`` (Idefics2-style)
  - vit_merger under ``model.vision_tower.vit_merger.*``
  - final merger under ``model.merger.mlp.0.*``

The text backbone is layout-identical to Qwen3.5, so it reuses the Qwen3.5 name
mapper and the GDN weight reordering. The vision side uses a dedicated mapper;
numeric verification (relerr == 0.0 against the F16 HF checkpoint) confirmed
the directions below, and that text RMSNorm needs ``weight - 1`` while the
vision LayerNorms must NOT be shifted.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..quantization.layout import GGUFHeadTilingLayout
from ..weight_utils import get_gguf_tensor_names
from .base import BaseGGUFWeightsAdapter, GGUFWeight
from .qwen3_5 import (
    Qwen35GGUFAdapter,
    _gdn_value_head_layout,
    _map_tensor_name,
    build_qwen35_text_mapper,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

MINICPMV4_6_ARCH = "MiniCPMV4_6ForConditionalGeneration"

# Vision tower (v.blk.*), vit_merger (v.vit_merger.*) and final merger (mm.*).
# Substring order matters: more specific keys come first so e.g. ``ds_ffn_up.``
# and ``ffn_up.`` are matched before the generic ``up.``. Directions confirmed
# by numeric comparison (relerr == 0.0): ffn_up->fc1, ffn_down->fc2,
# ds_ffn_up->linear_1, ds_ffn_down->linear_2, up->linear_1, down->linear_2.
MINICPMV4_6_VISION_SUBSTR: dict[str, str] = {
    # vit_merger (specific)
    "ds_ffn_up.": "linear_1.",
    "ds_ffn_down.": "linear_2.",
    "ds_ln.": "pre_norm.",
    # shared attention (vision tower + vit_merger)
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_out.": "self_attn.out_proj.",
    "ln1.": "layer_norm1.",
    "ln2.": "layer_norm2.",
    # vision tower MLP
    "ffn_up.": "mlp.fc1.",
    "ffn_down.": "mlp.fc2.",
    # final merger (mm.*)
    "up.": "linear_1.",
    "down.": "linear_2.",
    "input_norm.": "pre_norm.",
}


def build_minicpmv46_vision_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "v.patch_embd.": "model.vision_tower.embeddings.patch_embedding.",
            "v.position_embd.": "model.vision_tower.embeddings.position_embedding.",
            "v.post_ln.": "model.vision_tower.post_layernorm.",
            "v.vit_merger.": "model.vision_tower.vit_merger.",
            "v.blk.": "model.vision_tower.encoder.layers.",
            "mm.": "model.merger.mlp.0.",
        },
        orig_to_new_substr=MINICPMV4_6_VISION_SUBSTR,
    )


class MiniCPMV4_6GGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for MiniCPM-V 4.6 multimodal GGUF models."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type == "minicpmv4_6"

    @classmethod
    def architecture(cls, config) -> str | None:
        return MINICPMV4_6_ARCH

    @staticmethod
    def map_name(name: str) -> str | None:
        """Map a single raw GGUF tensor name to its canonical HF name."""
        mapper = (
            build_minicpmv46_vision_mapper()
            if name.startswith(("v.", "mm."))
            else build_qwen35_text_mapper(is_multimodal=True, is_moe=False)
        )
        return _map_tensor_name(mapper, name)

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        if files.mm_proj is None:
            raise RuntimeError(
                "MiniCPM-V 4.6 is multimodal; could not find its mmproj GGUF. "
                "Place *mmproj*.gguf beside the backbone or pass "
                "model_loader_extra_config={'mm_proj': ...}."
            )
        patched = maybe_patch_hf_config_from_gguf(
            files.primary_backbone,
            hf_config,
            mmproj_path=files.mm_proj,
        )
        patched.architectures = [MINICPMV4_6_ARCH]
        return patched

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        # Text backbone is Qwen3.5 (multimodal prefix model.language_model.).
        text_mapper = build_qwen35_text_mapper(is_multimodal=True, is_moe=False)
        vision_mapper = build_minicpmv46_vision_mapper()
        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(files.all_files)):
            mapper = vision_mapper if name.startswith(("v.", "mm.")) else text_mapper
            if mapped := _map_tensor_name(mapper, name):
                name_map[name] = mapped
            else:
                unmapped.append(name)
        if unmapped:
            logger.warning(
                "No HF name for %d MiniCPM-V 4.6 GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        return name_map

    def get_linear_layouts(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
        name_map: dict[str, str],
    ) -> dict[str, GGUFHeadTilingLayout]:
        del files
        text_config = model_config.hf_config.get_text_config()
        layout = _gdn_value_head_layout(text_config)
        if layout is None:
            return {}
        return {
            mapped_name.removesuffix(".weight"): layout
            for mapped_name in name_map.values()
            if mapped_name.endswith("linear_attn.out_proj.weight")
        }

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        """Reorder GDN text weights and shift text RMSNorm by -1.

        The text backbone shares the Qwen3.5 layout, so its GDN (linear_attn)
        weights need the same head-tiling reordering as the Qwen3.5 adapter.
        Only the text RMSNorms (``model.language_model.*``) are shifted by -1;
        the vision tower / mergers use plain LayerNorms and must NOT be shifted.
        """
        text_config = model_config.hf_config.get_text_config()
        layout = _gdn_value_head_layout(text_config)
        # The Qwen3.5 adapter's GDN reordering is stateless; borrow it.
        qwen35 = Qwen35GGUFAdapter()
        quantized_bases: set[str] = set()
        for name, weight in weights:
            if name.endswith(".weight_type"):
                quantized_bases.add(name.removesuffix(".weight_type"))
            if layout is not None:
                reordered = qwen35._restore_gdn_weight(
                    name, weight, text_config, layout, quantized_bases
                )
                if reordered is not None:
                    yield name, reordered
                    continue
            if name.endswith(".A_log"):
                yield name, torch.log(-weight)
                continue
            # Text RMSNorm: GGUF stores weight+1, vLLM expects weight.
            if (
                name.startswith("model.language_model.")
                and name.endswith("norm.weight")
                and not name.endswith("linear_attn.norm.weight")
            ):
                yield name, weight - 1
                continue
            # GDN conv1d arrives flattened; restore its channel dim.
            if "conv1d.weight" in name and weight.dim() == 2:
                weight = weight.unsqueeze(1)
            elif (
                name.endswith(".weight")
                and weight.dim() == 1
                and "norm" not in name
                and name.removesuffix(".weight") not in quantized_bases
            ):
                weight = weight.unsqueeze(0)
            yield name, weight
