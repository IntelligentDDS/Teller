# TellerMoTDecoderLayer: mixed attn + dual FFN
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from teller.mot.backbone_adapter import (
    get_layer_input_layernorm,
    get_layer_self_attn,
)

def _mlp(hidden_size: int, intermediate_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, intermediate_size),
        nn.GELU(),
        nn.Linear(intermediate_size, hidden_size),
    )

class TellerMoTDecoderLayer(nn.Module):
    def __init__(self, backbone_layer: Any, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.self_attn = get_layer_self_attn(backbone_layer)
        self.input_layernorm = get_layer_input_layernorm(backbone_layer)
        self.log_post_norm = nn.LayerNorm(hidden_size)
        self.trace_post_norm = nn.LayerNorm(hidden_size)
        self.log_ffn = _mlp(hidden_size, intermediate_size)
        self.trace_ffn = _mlp(hidden_size, intermediate_size)

    def forward(
        self,
        h_log: torch.Tensor,
        h_trace: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        L_log = h_log.size(1)
        L_trace = h_trace.size(1)
        h_all = torch.cat([h_log, h_trace], dim=1)
        residual = h_all
        h_all = self.input_layernorm(h_all)
        attn_out, _ = self.self_attn(
            hidden_states=h_all,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        h_all = residual + attn_out
        out_log = h_all[:, :L_log] + self.log_ffn(self.log_post_norm(h_all[:, :L_log]))
        out_trace = h_all[:, L_log:] + self.trace_ffn(self.trace_post_norm(h_all[:, L_log:]))
        return out_log, out_trace
