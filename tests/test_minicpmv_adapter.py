# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_gguf_plugin.weights_adapter.minicpmv import MiniCPMVGGUFAdapter


def test_minicpmv_text_name_mapping():
    assert MiniCPMVGGUFAdapter.map_name("token_embd.weight") == (
        "llm.model.embed_tokens.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("blk.2.attn_q.weight") == (
        "llm.model.layers.2.self_attn.q_proj.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("blk.2.attn_k_norm.weight") == (
        "llm.model.layers.2.self_attn.k_norm.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("blk.2.attn_norm.weight") == (
        "llm.model.layers.2.input_layernorm.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("blk.2.ffn_gate.weight") == (
        "llm.model.layers.2.mlp.gate_proj.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("blk.2.ffn_down.weight") == (
        "llm.model.layers.2.mlp.down_proj.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("output_norm.weight") == (
        "llm.model.norm.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("output.weight") == ("llm.lm_head.weight")


def test_minicpmv_vision_name_mapping():
    assert MiniCPMVGGUFAdapter.map_name("v.patch_embd.weight") == (
        "vpm.embeddings.patch_embedding.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("v.position_embd.weight") == (
        "vpm.embeddings.position_embedding.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("v.post_ln.bias") == ("vpm.post_layernorm.bias")
    assert MiniCPMVGGUFAdapter.map_name("v.blk.3.attn_q.weight") == (
        "vpm.encoder.layers.3.self_attn.q_proj.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("v.blk.3.ln1.weight") == (
        "vpm.encoder.layers.3.layer_norm1.weight"
    )
    # ffn_down -> fc1, ffn_up -> fc2 (confirmed by numeric comparison)
    assert MiniCPMVGGUFAdapter.map_name("v.blk.3.ffn_down.weight") == (
        "vpm.encoder.layers.3.mlp.fc1.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("v.blk.3.ffn_up.weight") == (
        "vpm.encoder.layers.3.mlp.fc2.weight"
    )


def test_minicpmv_resampler_name_mapping():
    # Identity mappings are kept (unlike Gemma4's identity->None rule).
    assert MiniCPMVGGUFAdapter.map_name("resampler.query") == "resampler.query"
    assert MiniCPMVGGUFAdapter.map_name("resampler.ln_q.weight") == (
        "resampler.ln_q.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("resampler.attn.q.weight") == (
        "resampler.attn.q.weight"
    )
    # Renamed mappings.
    assert MiniCPMVGGUFAdapter.map_name("resampler.proj.weight") == ("resampler.proj")
    assert MiniCPMVGGUFAdapter.map_name("resampler.kv.weight") == (
        "resampler.kv_proj.weight"
    )
    assert MiniCPMVGGUFAdapter.map_name("resampler.attn.out.weight") == (
        "resampler.attn.out_proj.weight"
    )
    # Recomputed at runtime by vLLM, so dropped.
    assert MiniCPMVGGUFAdapter.map_name("resampler.pos_embed_k") is None


def test_minicpmv_transposes_resampler_proj():
    weight = torch.arange(6).reshape(2, 3)  # [out, in]
    mapped = list(
        MiniCPMVGGUFAdapter().transform_weights([("resampler.proj", weight)], None)
    )
    assert [name for name, _ in mapped] == ["resampler.proj"]
    assert torch.equal(mapped[0][1], weight.transpose(0, 1))


def test_minicpmv_merges_resampler_qkv():
    q = torch.arange(4).reshape(2, 2)
    k = torch.arange(4, 8).reshape(2, 2)
    v = torch.arange(8, 12).reshape(2, 2)
    weights = [
        ("resampler.attn.q.weight", q),
        ("resampler.attn.k.weight", k),
        ("resampler.attn.v.weight", v),
    ]
    mapped = list(MiniCPMVGGUFAdapter().transform_weights(weights, None))
    assert [name for name, _ in mapped] == ["resampler.attn.in_proj_weight"]
    assert torch.equal(mapped[0][1], torch.cat([q, k, v], dim=0))


def test_minicpmv_merges_resampler_qkv_bias():
    q = torch.arange(3).reshape(1, 3)
    k = torch.arange(3, 6).reshape(1, 3)
    v = torch.arange(6, 9).reshape(1, 3)
    weights = [
        ("resampler.attn.q.bias", q),
        ("resampler.attn.k.bias", k),
        ("resampler.attn.v.bias", v),
    ]
    mapped = list(MiniCPMVGGUFAdapter().transform_weights(weights, None))
    assert [name for name, _ in mapped] == ["resampler.attn.in_proj_bias"]
    assert torch.equal(mapped[0][1], torch.cat([q, k, v], dim=0))


def test_minicpmv_passthrough_for_other_weights():
    weight = torch.arange(4).reshape(2, 2)
    weights = [
        ("llm.model.layers.0.self_attn.q_proj.weight", weight),
        ("vpm.encoder.layers.0.layer_norm1.weight", weight),
    ]
    mapped = list(MiniCPMVGGUFAdapter().transform_weights(weights, None))
    assert [name for name, _ in mapped] == [
        "llm.model.layers.0.self_attn.q_proj.weight",
        "vpm.encoder.layers.0.layer_norm1.weight",
    ]
    assert all(torch.equal(w, weight) for _, w in mapped)
