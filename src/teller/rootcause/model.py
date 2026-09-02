"""
TellerMoTForRootCauseGenerating: log + trace (embed only) + root cause generation.
No TraceEncoder; trace tokens embedded and mean-pooled per step. Teacher forcing for eval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class TellerMoTForRootCauseGenerating(nn.Module):
    """
    Root cause generation + fault_type classification.
    Two vocabs: (1) trace = TPE token ids, pad typically 0; (2) LLM = backbone text ids, pad from tokenizer.
    Sequence: [log_emb, trace_emb (one per step), rc_emb]. Single-stream backbone; no TraceEncoder.
    """

    def __init__(
        self,
        backbone: nn.Module,
        trace_vocab_size: int,
        num_fault_types: int,
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
        self.fault_head = nn.Linear(self._hidden_size, num_fault_types)

    @staticmethod
    def _get_hidden_size(backbone: nn.Module) -> int:
        if hasattr(backbone, "config"):
            return getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "d_model", 768)
        model = getattr(backbone, "model", backbone)
        if hasattr(model, "config"):
            return getattr(model.config, "hidden_size", 768)
        return 768

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def _get_embed(self) -> nn.Module:
        if hasattr(self.backbone, "get_input_embeddings"):
            return self.backbone.get_input_embeddings()
        model = getattr(self.backbone, "model", self.backbone)
        return getattr(model, "embed_tokens", model.embed_tokens)

    def _get_layers(self):
        model = getattr(self.backbone, "model", self.backbone)
        return getattr(model, "layers", None) or getattr(model, "decoder", None) and getattr(model.decoder, "layers", None) or getattr(model, "h", None)

    def _get_norm(self):
        model = getattr(self.backbone, "model", self.backbone)
        return getattr(model, "norm", None) or getattr(model, "ln_f", None)

    def _get_lm_head(self):
        return getattr(self.backbone, "lm_head", None) or getattr(self.backbone, "output_projection", None)

    def _trace_to_hidden(
        self,
        step_token_ids: torch.Tensor,
        step_token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """(B, N_steps, L_tok) -> (B, N_steps, d) via embed + mean pool over tokens. Uses trace vocab (TPE) pad."""
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
        root_cause_token_ids: torch.Tensor | None = None,
        step_token_mask: torch.Tensor | None = None,
        fault_type_labels: torch.Tensor | None = None,
        log_attention_mask: torch.Tensor | None = None,
        trace_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        log_token_ids: (B, L_log)
        step_token_ids: (B, N_steps, L_tok) TPE token ids
        root_cause_token_ids: (B, L_rc) for teacher forcing; optional at inference
        Returns: dict with loss (if labels given), logits, fault_type_logits, etc.
        """
        embed = self._get_embed()
        h_log = embed(log_token_ids)
        h_trace = self._trace_to_hidden(step_token_ids, step_token_mask)
        dtype = h_log.dtype
        h_trace = h_trace.to(dtype)
        L_log = h_log.size(1)
        N_trace = h_trace.size(1)

        if root_cause_token_ids is not None:
            h_rc = embed(root_cause_token_ids)
            L_rc = h_rc.size(1)
            inputs_embeds = torch.cat([h_log, h_trace, h_rc], dim=1)
        else:
            L_rc = 0
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
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = backbone_out.hidden_states[-1]
        logits = backbone_out.logits

        pool_len = L_log + N_trace
        pool_mask = torch.ones(B, pool_len, device=device, dtype=hidden_states.dtype)
        if log_attention_mask is not None:
            pool_mask[:, :L_log] = log_attention_mask
        if trace_attention_mask is not None:
            pool_mask[:, L_log:pool_len] = trace_attention_mask
        pooled = (hidden_states[:, :pool_len] * pool_mask.unsqueeze(-1)).sum(1) / pool_mask.sum(1, keepdim=True).clamp(min=1e-9)
        fault_type_logits = self.fault_head(pooled)

        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "fault_type_logits": fault_type_logits,
            "hidden_states": hidden_states,
        }

        loss = None
        if root_cause_token_ids is not None and logits is not None and L_rc > 1:
            rc_start = L_log + N_trace
            rc_logits = logits[:, rc_start - 1 : rc_start + L_rc - 2]
            rc_labels = root_cause_token_ids[:, 1:]
            valid = rc_labels != self.pad_token_id
            if valid.any():
                loss_rc = F.cross_entropy(
                    rc_logits.reshape(-1, rc_logits.size(-1)),
                    rc_labels.reshape(-1),
                    ignore_index=self.pad_token_id,
                )
                if not (torch.isnan(loss_rc).item() or torch.isinf(loss_rc).item()):
                    out["loss_rc"] = loss_rc
                    loss = loss_rc
                else:
                    out["loss_rc"] = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            else:
                out["loss_rc"] = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if fault_type_labels is not None:
            loss_fault = F.cross_entropy(fault_type_logits, fault_type_labels)
            out["loss_fault"] = loss_fault
            loss = loss_fault if loss is None else loss + loss_fault
        if loss is not None:
            out["loss"] = loss
        return out

    def save_pretrained(self, save_directory: str | Path, config_extra: dict | None = None, **kwargs: Any) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(save_dir / "backbone")
        state = {
            "trace_embed": self.trace_embed.state_dict(),
            "fault_head": self.fault_head.state_dict(),
        }
        torch.save(state, save_dir / "rootcause_adapter.pt")
        config = {
            "trace_vocab_size": self.trace_embed.num_embeddings,
            "num_fault_types": self.fault_head.out_features,
            "trace_pad_token_id": self.trace_pad_token_id,
            "pad_token_id": self.pad_token_id,
        }
        if config_extra:
            config.update(config_extra)
        with open(save_dir / "rootcause_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str | Path,
        fault_type_labels: list[str] | None = None,
        **kwargs: Any,
    ) -> "TellerMoTForRootCauseGenerating":
        from transformers import AutoModelForCausalLM
        save_dir = Path(save_directory)
        with open(save_dir / "rootcause_config.json", encoding="utf-8") as f:
            config = json.load(f)
        backbone = AutoModelForCausalLM.from_pretrained(save_dir / "backbone", **kwargs)
        model = cls(
            backbone=backbone,
            trace_vocab_size=config["trace_vocab_size"],
            num_fault_types=config["num_fault_types"],
            trace_pad_token_id=config.get("trace_pad_token_id", 0),
            pad_token_id=config.get("pad_token_id"),
        )
        state = torch.load(save_dir / "rootcause_adapter.pt", map_location="cpu", weights_only=True)
        model.trace_embed.load_state_dict(state["trace_embed"], strict=True)
        model.fault_head.load_state_dict(state["fault_head"], strict=True)
        return model
