# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the MiniCPM-V 4.6 GGUF adapter.

Pure name-mapping and transform tests (no model loading, no GPU). Directions
were confirmed against the F16 HF checkpoint with relerr == 0.0 (see
test/verify_minicpmv46.py and test/verify_minicpmv46_norm.py).
"""

import torch

from vllm_gguf_plugin.weights_adapter.minicpmv4_6 import MiniCPMV4_6GGUFAdapter


def test_minicpmv46_text_name_mapping():
    # Text backbone is Qwen3.5 (multimodal prefix model.language_model.).
    assert MiniCPMV4_6GGUFAdapter.map_name("token_embd.weight") == (
        "model.language_model.embed_tokens.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("blk.0.attn_q.weight") == (
        "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("blk.0.ffn_gate.weight") == (
        "model.language_model.layers.0.mlp.gate_proj.weight"
    )
    # GDN / linear-attention layers
    assert MiniCPMV4_6GGUFAdapter.map_name("blk.0.attn_qkv.weight") == (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("blk.0.ssm_a") == (
        "model.language_model.layers.0.linear_attn.A_log"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("blk.0.ssm_dt.bias") == (
        "model.language_model.layers.0.linear_attn.dt_bias"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("output_norm.weight") == (
        "model.language_model.norm.weight"
    )


def test_minicpmv46_vision_tower_name_mapping():
    assert MiniCPMV4_6GGUFAdapter.map_name("v.patch_embd.weight") == (
        "model.vision_tower.embeddings.patch_embedding.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("v.position_embd.weight") == (
        "model.vision_tower.embeddings.position_embedding.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("v.post_ln.bias") == (
        "model.vision_tower.post_layernorm.bias"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("v.blk.3.attn_q.weight") == (
        "model.vision_tower.encoder.layers.3.self_attn.q_proj.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("v.blk.3.ln1.weight") == (
        "model.vision_tower.encoder.layers.3.layer_norm1.weight"
    )
    # ffn_up -> fc1, ffn_down -> fc2 (4.6 direction; opposite of 4.5!)
    assert MiniCPMV4_6GGUFAdapter.map_name("v.blk.3.ffn_up.weight") == (
        "model.vision_tower.encoder.layers.3.mlp.fc1.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("v.blk.3.ffn_down.weight") == (
        "model.vision_tower.encoder.layers.3.mlp.fc2.weight"
    )


def test_minicpmv46_vit_merger_name_mapping():
    # vit_merger lives under the vision tower.
    assert MiniCPMV4_6GGUFAdapter.map_name("v.vit_merger.attn_q.weight") == (
        "model.vision_tower.vit_merger.self_attn.q_proj.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("v.vit_merger.ln1.weight") == (
        "model.vision_tower.vit_merger.layer_norm1.weight"
    )
    # ds_ffn_up -> linear_1, ds_ffn_down -> linear_2, ds_ln -> pre_norm
    assert MiniCPMV4_6GGUFAdapter.map_name("v.vit_merger.ds_ffn_up.weight") == (
        "model.vision_tower.vit_merger.linear_1.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("v.vit_merger.ds_ffn_down.weight") == (
        "model.vision_tower.vit_merger.linear_2.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("v.vit_merger.ds_ln.weight") == (
        "model.vision_tower.vit_merger.pre_norm.weight"
    )


def test_minicpmv46_final_merger_name_mapping():
    # mm.* maps to the final DownsampleMLP projector (merger_times == 1).
    assert MiniCPMV4_6GGUFAdapter.map_name("mm.up.weight") == (
        "model.merger.mlp.0.linear_1.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("mm.down.weight") == (
        "model.merger.mlp.0.linear_2.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("mm.input_norm.weight") == (
        "model.merger.mlp.0.pre_norm.weight"
    )
    assert MiniCPMV4_6GGUFAdapter.map_name("mm.up.bias") == (
        "model.merger.mlp.0.linear_1.bias"
    )


def test_minicpmv46_matches_and_architecture():
    class _Cfg:
        model_type = "minicpmv4_6"

    assert MiniCPMV4_6GGUFAdapter.matches(_Cfg())
    assert MiniCPMV4_6GGUFAdapter.architecture(_Cfg()) == (
        "MiniCPMV4_6ForConditionalGeneration"
    )

    class _Other:
        model_type = "minicpmv"

    assert not MiniCPMV4_6GGUFAdapter.matches(_Other())


def test_minicpmv46_transform_text_rmsnorm_minus_one():
    """Text RMSNorm weights are stored as weight+1 in GGUF -> shift by -1."""
    adapter = MiniCPMV4_6GGUFAdapter()

    class _MC:
        class _TC:
            linear_num_key_heads = 0
            linear_num_value_heads = 0

        hf_config = type("C", (), {"get_text_config": lambda self: _MC._TC()})()

    norm = torch.full((4,), 2.0)  # stored weight+1 -> real weight 1.0
    out = list(
        adapter.transform_weights(
            [("model.language_model.layers.0.input_layernorm.weight", norm)], _MC
        )
    )
    assert out[0][0] == "model.language_model.layers.0.input_layernorm.weight"
    assert torch.allclose(out[0][1], torch.full((4,), 1.0))


def test_minicpmv46_transform_vision_layernorm_untouched():
    """Vision LayerNorm weights must NOT be shifted (plain LayerNorm)."""
    adapter = MiniCPMV4_6GGUFAdapter()

    class _MC:
        class _TC:
            linear_num_key_heads = 0
            linear_num_value_heads = 0

        hf_config = type("C", (), {"get_text_config": lambda self: _MC._TC()})()

    ln = torch.full((4,), 3.0)
    out = list(
        adapter.transform_weights(
            [("model.vision_tower.encoder.layers.0.layer_norm1.weight", ln)], _MC
        )
    )
    assert torch.allclose(out[0][1], torch.full((4,), 3.0))
