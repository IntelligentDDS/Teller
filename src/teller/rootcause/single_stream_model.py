"""
Single-stream root cause model: log + trace (embed) + label generation.

The entire label (including <fault_type>, the type text, </fault_type>, and root cause text)
lives in the LLM text vocabulary and is decoded by the backbone's lm_head only. Fault type
is LLM text semantics—no separate fault head or extra vocab.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class TellerSingleStreamForRC(nn.Module):
    """
    Single-stream: [log_emb, trace_emb, label_emb] -> backbone (same lm_head) -> loss on label.

    <fault_type>, </fault_type> and the type string are normal tokens in the LLM vocab;
    the backbone's lm_head decodes every label position, including the fault-type span.
    """

    def __init__(
        self,
        backbone: nn.Module,
        trace_vocab_size: int,
        trace_pad_token_id: int = 0,
        pad_token_id: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self._hidden_size = self._get_hidden_size(backbone)
        self.trace_pad_token_id = trace_pad_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.trace_embed = nn.Embedding(
            trace_vocab_size,
            self._hidden_size,
            padding_idx=trace_pad_token_id,
        )

    @staticmethod
    def _get_hidden_size(backbone: nn.Module) -> int:
        if hasattr(backbone, "config"):
            return getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "d_model", 768)
        model = getattr(backbone, "model", backbone)
        if hasattr(model, "config"):
            return getattr(model.config, "hidden_size", 768)
        return 768

    def _get_embed(self) -> nn.Module:
        if hasattr(self.backbone, "get_input_embeddings"):
            return self.backbone.get_input_embeddings()
        model = getattr(self.backbone, "model", self.backbone)
        return getattr(model, "embed_tokens", model.embed_tokens)

    def _trace_to_hidden(
        self,
        step_token_ids: torch.Tensor,
        step_token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """(B, N_steps, L_tok) -> (B, N_steps, d)."""
        B, N_steps, L_tok = step_token_ids.shape
        h = self.trace_embed(step_token_ids)
        if step_token_mask is None:
            step_token_mask = (step_token_ids != self.trace_pad_token_id).to(h.dtype)
        else:
            step_token_mask = step_token_mask.to(h.dtype)
        mask = step_token_mask.unsqueeze(-1)
        h = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1e-9)
        return h

    def forward(
        self,
        log_token_ids: torch.Tensor,
        step_token_ids: torch.Tensor,
        label_token_ids: torch.Tensor | None = None,
        step_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        log_token_ids: (B, L_log)
        step_token_ids: (B, N_steps, L_tok) TPE
        label_token_ids: (B, L_label) target sequence; loss computed on this part only.
        """
        embed = self._get_embed()
        h_log = embed(log_token_ids)
        h_trace = self._trace_to_hidden(step_token_ids, step_token_mask)
        dtype = h_log.dtype
        h_trace = h_trace.to(dtype)
        L_log = h_log.size(1)
        N_trace = h_trace.size(1)

        if label_token_ids is not None:
            h_label = embed(label_token_ids)
            L_label = h_label.size(1)
            inputs_embeds = torch.cat([h_log, h_trace, h_label], dim=1)
        else:
            L_label = 0
            inputs_embeds = torch.cat([h_log, h_trace], dim=1)

        B, L, D = inputs_embeds.shape
        device = inputs_embeds.device
        position_ids = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        attention_mask = torch.ones(B, L, device=device, dtype=torch.long)

        backbone_out = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = backbone_out.logits

        out: dict[str, torch.Tensor] = {"logits": logits}

        if label_token_ids is not None and logits is not None and L_label > 1:
            label_start = L_log + N_trace
            label_logits = logits[:, label_start - 1 : label_start + L_label - 2]
            label_labels = label_token_ids[:, 1:]
            valid = label_labels != self.pad_token_id
            if valid.any():
                loss = F.cross_entropy(
                    label_logits.reshape(-1, label_logits.size(-1)),
                    label_labels.reshape(-1),
                    ignore_index=self.pad_token_id,
                )
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            else:
                loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            out["loss"] = loss
        return out

    def save_pretrained(self, save_directory: str | Path, config_extra: dict | None = None, **kwargs: Any) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(save_dir / "backbone")
        state = {"trace_embed": self.trace_embed.state_dict()}
        torch.save(state, save_dir / "single_stream_adapter.pt")
        config = {
            "trace_vocab_size": self.trace_embed.num_embeddings,
            "trace_pad_token_id": self.trace_pad_token_id,
            "pad_token_id": self.pad_token_id,
        }
        if config_extra:
            config.update(config_extra)
        with open(save_dir / "single_stream_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, save_directory: str | Path, **kwargs: Any) -> "TellerSingleStreamForRC":
        from transformers import AutoModelForCausalLM
        save_dir = Path(save_directory)
        with open(save_dir / "single_stream_config.json", encoding="utf-8") as f:
            config = json.load(f)
        backbone = AutoModelForCausalLM.from_pretrained(save_dir / "backbone", **kwargs)
        model = cls(
            backbone=backbone,
            trace_vocab_size=config["trace_vocab_size"],
            trace_pad_token_id=config.get("trace_pad_token_id", 0),
            pad_token_id=config.get("pad_token_id"),
        )
        state = torch.load(save_dir / "single_stream_adapter.pt", map_location="cpu", weights_only=True)
        model.trace_embed.load_state_dict(state["trace_embed"], strict=True)
        return model
