"""
Extract decoder layer components from any HuggingFace causal LM for TELLER MoT.
Supports Qwen3/LLaMA-style: .model.layers[i].{self_attn, input_layernorm, post_attention_layernorm, mlp}.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def get_base_model(backbone: nn.Module) -> nn.Module:
    """Return the inner transformer (e.g. backbone.model for ForCausalLM)."""
    if hasattr(backbone, "model"):
        return backbone.model
    return backbone


def get_config(backbone: nn.Module) -> Any:
    """Return the backbone config (hidden_size, intermediate_size, etc.)."""
    base = get_base_model(backbone)
    if hasattr(base, "config"):
        return base.config
    if hasattr(backbone, "config"):
        return backbone.config
    raise AttributeError("Backbone has no config")


def get_hidden_size(backbone: nn.Module) -> int:
    config = get_config(backbone)
    return getattr(config, "hidden_size", None) or getattr(config, "d_model", None)


def get_intermediate_size(backbone: nn.Module) -> int:
    config = get_config(backbone)
    return getattr(config, "intermediate_size", None) or getattr(config, "ffn_dim", None) or 4 * get_hidden_size(backbone)


def get_decoder_layers(backbone: nn.Module) -> list[Any]:
    """Return list of decoder layers (each with self_attn, input_layernorm, post_attention_layernorm, mlp)."""
    base = get_base_model(backbone)
    if hasattr(base, "layers"):
        return list(base.layers)
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        return list(base.decoder.layers)
    if hasattr(base, "h"):  # GPT-2 style
        return list(base.h)
    raise AttributeError("Backbone has no .model.layers / .decoder.layers / .h")


def get_embed_tokens(backbone: nn.Module) -> nn.Module:
    base = get_base_model(backbone)
    if hasattr(base, "embed_tokens"):
        return base.embed_tokens
    if hasattr(base, "wte"):  # GPT-2
        return base.wte
    if hasattr(backbone, "get_input_embeddings"):
        return backbone.get_input_embeddings()
    raise AttributeError("Backbone has no embed_tokens / wte / get_input_embeddings")


def get_final_norm(backbone: nn.Module) -> nn.Module | None:
    base = get_base_model(backbone)
    if hasattr(base, "norm"):
        return base.norm
    if hasattr(base, "ln_f"):  # GPT-2
        return base.ln_f
    return None


def get_lm_head(backbone: nn.Module) -> nn.Module | None:
    if hasattr(backbone, "lm_head"):
        return backbone.lm_head
    if hasattr(backbone, "output_projection"):
        return backbone.output_projection
    return None


def get_rotary_emb(backbone: nn.Module) -> Any | None:
    """Return rotary embedding module if present (Qwen/LLaMA)."""
    base = get_base_model(backbone)
    if hasattr(base, "rotary_emb"):
        return base.rotary_emb
    return None


def layer_has_components(layer: Any) -> bool:
    """Check if layer has self_attn, input_layernorm, and mlp (or equivalent)."""
    has_attn = hasattr(layer, "self_attn") or hasattr(layer, "attention")
    has_input_norm = hasattr(layer, "input_layernorm") or hasattr(layer, "ln_1")
    has_mlp = hasattr(layer, "mlp") or hasattr(layer, "feed_forward") or hasattr(layer, "mlp")
    has_post_norm = hasattr(layer, "post_attention_layernorm") or hasattr(layer, "ln_2")
    return has_attn and (has_input_norm or has_mlp)


def get_layer_self_attn(layer: Any) -> nn.Module:
    return getattr(layer, "self_attn", None) or getattr(layer, "attention")


def get_layer_input_layernorm(layer: Any) -> nn.Module | None:
    return getattr(layer, "input_layernorm", None) or getattr(layer, "ln_1", None)


def get_layer_post_attention_layernorm(layer: Any) -> nn.Module | None:
    return getattr(layer, "post_attention_layernorm", None) or getattr(layer, "ln_2", None)


def get_layer_mlp(layer: Any) -> nn.Module:
    return getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)


def get_layer_mlp_intermediate_size(layer: Any, default: int) -> int:
    """Infer intermediate_size from backbone MLP if possible."""
    mlp = get_layer_mlp(layer)
    if hasattr(mlp, "gate_proj"):
        return mlp.gate_proj.out_features
    if hasattr(mlp, "c_fc"):  # GPT-2
        return mlp.c_fc.out_features
    if isinstance(mlp, nn.Sequential):
        for m in mlp:
            if isinstance(m, nn.Linear) and m.in_features != m.out_features:
                return m.out_features
    return default
