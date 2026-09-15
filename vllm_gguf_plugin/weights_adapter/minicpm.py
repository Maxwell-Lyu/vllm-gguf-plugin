# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GGUF weights adapter for MiniCPM text models (MiniCPMForCausalLM).

MiniCPM text checkpoints (MiniCPM-1B/2B/4/4.1, MiniCPM-S-1B, ...) are standard
dense LLaMA-style transformers. The GGUF (llama.cpp arch ``minicpm``) stores
tensor names in llama convention (``token_embd`` / ``blk.{i}.attn_q`` / ...) and,
crucially, the attention ``q``/``k`` weights are stored with llama.cpp's GQA
RoPE permutation applied. vLLM's ``MiniCPMForCausalLM`` expects the original HF
layout (``model.layers.{i}.self_attn.{q,k,v,o}_proj``), so:

  1. names are remapped llama -> HF (see :meth:`build_name_map`);
  2. the ``q_proj``/``k_proj`` weights are inverse-permuted back to the HF RoPE
     layout (see :meth:`transform_weights`). v/o need no transform.

Verified numerically against an F16 GGUF produced by llama.cpp's own converter
and the original HF weights (relerr ~1e-8, see test/verify_minicpm.py):
  - q: ``permute_inv(gguf_q, n_head, n_head) == hf_q``
  - k: ``permute_inv(gguf_k, n_head, n_kv)   == hf_k``
  - RMSNorm weights are stored as-is (NO -1 offset; unlike Qwen3.5).
  - ``rope_factors_long/short`` are recomputed at runtime by vLLM and dropped.
"""

from __future__ import annotations

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

MINICPM_ARCH = "MiniCPMForCausalLM"

# llama.cpp arch "minicpm" uses standard llama tensor names.
MINICPM_TEXT_SUBSTR: dict[str, str] = {
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_output.": "self_attn.o_proj.",
    "attn_norm.": "input_layernorm.",
    "ffn_norm.": "post_attention_layernorm.",
    "ffn_gate.": "mlp.gate_proj.",
    "ffn_up.": "mlp.up_proj.",
    "ffn_down.": "mlp.down_proj.",
}


def build_minicpm_text_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": "model.embed_tokens.",
            "blk.": "model.layers.",
            "output_norm.": "model.norm.",
            "output.": "lm_head.",
        },
        orig_to_new_substr=MINICPM_TEXT_SUBSTR,
    )


def _map_tensor_name(mapper: WeightsMapper, name: str) -> str | None:
    result = mapper.apply_list([name])
    return result[0] if result else None


def _permute_inv(w: torch.Tensor, n_head: int, n_kv: int | None) -> torch.Tensor:
    """Inverse of llama.cpp's ``LlamaModel.permute`` (which is not self-inverse).

    llama.cpp stores ``gguf = permute(hf, n_head, n_kv)``; this recovers ``hf``
    so the weight matches vLLM's expected RoPE layout.
    """
    if n_kv is not None and n_head != n_kv:
        n_head = n_kv
    rows, cols = w.shape[0], w.shape[1]
    m = rows // n_head // 2
    return w.reshape(n_head, m, 2, cols).swapaxes(1, 2).reshape(rows, cols)


class MiniCPMGGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for MiniCPM text (MiniCPMForCausalLM) GGUF models."""

    @classmethod
    def matches(cls, config) -> bool:
        # Exact match: minicpmv / minicpm3 / minicpmo / minicpmv4_6 are distinct.
        return config.model_type == "minicpm"

    @classmethod
    def architecture(cls, config) -> str | None:
        return MINICPM_ARCH

    @staticmethod
    def map_name(name: str) -> str | None:
        return _map_tensor_name(build_minicpm_text_mapper(), name)

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        patched = maybe_patch_hf_config_from_gguf(
            files.primary_backbone,
            hf_config,
            mmproj_path=files.mm_proj,
        )
        patched.architectures = [MINICPM_ARCH]
        return patched

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        del model_config
        mapper = build_minicpm_text_mapper()
        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(files.all_files)):
            # Recomputed at runtime by vLLM; drop silently (no warning).
            if name.startswith(("rope_factors_long.", "rope_factors_short.")):
                continue
            if mapped := _map_tensor_name(mapper, name):
                name_map[name] = mapped
            else:
                unmapped.append(name)
        if unmapped:
            logger.warning(
                "No HF name for %d MiniCPM GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        return name_map

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        """Inverse-permute the GQA ``q_proj``/``k_proj`` weights back to HF layout.

        llama.cpp permutes q/k on convert; vLLM expects the original HF RoPE
        layout, so we apply the inverse. v/o and everything else pass through
        unchanged (RMSNorm needs no -1 offset for MiniCPM).
        """
        cfg = model_config.hf_config
        n_head = getattr(cfg, "num_attention_heads", None) or getattr(
            cfg, "num_heads", None
        )
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_head
        for name, weight in weights:
            if name.endswith(".self_attn.q_proj.weight"):
                yield name, _permute_inv(weight, n_head, n_head)
                continue
            if name.endswith(".self_attn.k_proj.weight"):
                yield name, _permute_inv(weight, n_head, n_kv)
                continue
            yield name, weight
