# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the MiniCPM text (MiniCPMForCausalLM) GGUF adapter.

Pure name-mapping and transform tests (no model loading, no GPU). The q/k
inverse-permute and the "no -1 on RMSNorm" behavior were confirmed against an
F16 GGUF produced by llama.cpp's own converter (test/verify_minicpm.py and
test/verify_minicpm_pipeline.py).
"""

import torch

from vllm_gguf_plugin.weights_adapter.minicpm import (
    MiniCPMGGUFAdapter,
    _permute_inv,
)


def test_minicpm_name_mapping():
    assert MiniCPMGGUFAdapter.map_name("token_embd.weight") == (
        "model.embed_tokens.weight"
    )
    assert MiniCPMGGUFAdapter.map_name("blk.3.attn_q.weight") == (
        "model.layers.3.self_attn.q_proj.weight"
    )
    assert MiniCPMGGUFAdapter.map_name("blk.3.attn_k.weight") == (
        "model.layers.3.self_attn.k_proj.weight"
    )
    assert MiniCPMGGUFAdapter.map_name("blk.3.attn_v.weight") == (
        "model.layers.3.self_attn.v_proj.weight"
    )
    assert MiniCPMGGUFAdapter.map_name("blk.3.attn_output.weight") == (
        "model.layers.3.self_attn.o_proj.weight"
    )
    assert MiniCPMGGUFAdapter.map_name("blk.3.attn_norm.weight") == (
        "model.layers.3.input_layernorm.weight"
    )
    assert MiniCPMGGUFAdapter.map_name("blk.3.ffn_norm.weight") == (
        "model.layers.3.post_attention_layernorm.weight"
    )
    assert MiniCPMGGUFAdapter.map_name("blk.3.ffn_gate.weight") == (
        "model.layers.3.mlp.gate_proj.weight"
    )
    assert MiniCPMGGUFAdapter.map_name("blk.3.ffn_up.weight") == (
        "model.layers.3.mlp.up_proj.weight"
    )
    assert MiniCPMGGUFAdapter.map_name("blk.3.ffn_down.weight") == (
        "model.layers.3.mlp.down_proj.weight"
    )
    assert MiniCPMGGUFAdapter.map_name("output_norm.weight") == ("model.norm.weight")
    assert MiniCPMGGUFAdapter.map_name("output.weight") == ("lm_head.weight")


def test_minicpm_matches_and_architecture():
    class _Cfg:
        model_type = "minicpm"

    assert MiniCPMGGUFAdapter.matches(_Cfg())
    assert MiniCPMGGUFAdapter.architecture(_Cfg()) == "MiniCPMForCausalLM"

    # Must NOT collide with the other MiniCPM families.
    for other in ("minicpmv", "minicpm3", "minicpmo", "minicpmv4_6", "qwen3_5"):

        class _C:
            model_type = other

        assert not MiniCPMGGUFAdapter.matches(_C())


def test_permute_inv_is_inverse_of_llama_permute():
    """_permute_inv must recover the HF layout from llama.cpp's permute."""

    def llama_permute(w, n_head, n_kv=None):
        if n_kv is not None and n_head != n_kv:
            n_head = n_kv
        return (
            w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
            .swapaxes(1, 2)
            .reshape(w.shape)
        )

    torch.manual_seed(0)
    # MHA q: n_head == n_kv
    q = torch.randn(1024, 1024)
    assert torch.allclose(_permute_inv(llama_permute(q, 16, 16), 16, 16), q)
    # GQA k: n_head != n_kv
    k = torch.randn(128, 1024)
    assert torch.allclose(_permute_inv(llama_permute(k, 16, 2), 16, 2), k)


def test_transform_qk_only_vo_passthrough():
    """q/k get inverse-permuted; v/o and norms pass through unchanged."""
    adapter = MiniCPMGGUFAdapter()

    class _Cfg:
        num_attention_heads = 16
        num_key_value_heads = 2

        def get_text_config(self):
            return self

    class _MC:
        hf_config = _Cfg()

    torch.manual_seed(1)
    q = torch.randn(1024, 1024)
    k = torch.randn(128, 1024)
    v = torch.randn(128, 1024)
    norm = torch.randn(1024)

    stream = [
        ("model.layers.0.self_attn.q_proj.weight", q),
        ("model.layers.0.self_attn.k_proj.weight", k),
        ("model.layers.0.self_attn.v_proj.weight", v),
        ("model.layers.0.input_layernorm.weight", norm),
    ]
    out = dict(adapter.transform_weights(stream, _MC()))

    # v and norm are untouched.
    assert torch.equal(out["model.layers.0.self_attn.v_proj.weight"], v)
    assert torch.equal(out["model.layers.0.input_layernorm.weight"], norm)

    # q/k are actually changed (permute applied) and have the right shape.
    assert out["model.layers.0.self_attn.q_proj.weight"].shape == q.shape
    assert out["model.layers.0.self_attn.k_proj.weight"].shape == k.shape
    assert not torch.equal(out["model.layers.0.self_attn.q_proj.weight"], q)
    assert not torch.equal(out["model.layers.0.self_attn.k_proj.weight"], k)


def test_transform_norm_not_shifted():
    """MiniCPM RMSNorm weights must NOT be shifted by -1 (unlike Qwen3.5)."""
    adapter = MiniCPMGGUFAdapter()

    class _Cfg:
        num_attention_heads = 16
        num_key_value_heads = 2

    class _MC:
        hf_config = _Cfg()

    norm = torch.full((4,), 3.0)
    out = dict(
        adapter.transform_weights(
            [("model.layers.0.input_layernorm.weight", norm)], _MC
        )
    )
    assert torch.equal(out["model.layers.0.input_layernorm.weight"], norm)
