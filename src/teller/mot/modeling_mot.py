"""
TELLER MoT: dual-stream (log + trace) with mixed attention and separate FFNs per stream.
TellerMoTDecoderLayer = attention(concat(h_log, h_trace)) + log_ffn(h_log) + trace_ffn(h_trace).
Built from any HF causal LM via backbone_adapter. save_pretrained / from_pretrained supported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from teller.mot.backbone_adapter import (
    get_decoder_layers,
    get_embed_tokens,
    get_final_norm,
    get_hidden_size,
    get_intermediate_size,
    get_layer_mlp_intermediate_size,
    get_lm_head,
    get_rotary_emb,
)
from teller.mot.decoder_layer import TellerMoTDecoderLayer
from teller.mot.trace_encoder import TracePairEncoder


def _get_backbone_hidden_size(backbone: nn.Module) -> int:
    return get_hidden_size(backbone)


def _causal_mask(L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    return torch.triu(
        torch.full((L, L), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)


class TellerMoTForDiagnosis(nn.Module):
    """
    MoT: dual stream (log + trace). Each layer = mixed attention (concat) + separate Log FFN and Trace FFN.
    Built from any HF causal LM; backbone provides embed, self_attn/input_layernorm per layer, rotary_emb, norm, lm_head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        trace_encoder: TracePairEncoder,
        num_fault_reasons: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.trace_encoder = trace_encoder
        d = get_hidden_size(backbone)
        self._hidden_size = d

        decoder_layers = get_decoder_layers(backbone)
        inter_size = get_intermediate_size(backbone)
        self.mot_layers = nn.ModuleList()
        for layer in decoder_layers:
            inter = get_layer_mlp_intermediate_size(layer, inter_size)
            self.mot_layers.append(TellerMoTDecoderLayer(layer, d, inter))

        self.final_norm = get_final_norm(backbone)
        self.lm_head = get_lm_head(backbone)
        self.fault_reason_head = nn.Linear(d, num_fault_reasons)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(
        self,
        log_token_ids: torch.Tensor,
        step_token_ids: torch.Tensor,
        step_token_mask: torch.Tensor | None = None,
        step_parent_idx: torch.Tensor | None = None,
        step_duration: torch.Tensor | None = None,
        step_depth: torch.Tensor | None = None,
        log_attention_mask: torch.Tensor | None = None,
        trace_attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        log_token_ids: (B, L_log)
        step_token_ids: (B, L_trace, L_tok)
        step_token_mask: (B, L_trace, L_tok) optional
        step_parent_idx: (B, L_trace, L_tok) optional, for GCN call graph
        step_duration / step_depth: (B, L_trace, L_tok) optional, for node encoding
        log_attention_mask: (B, L_log) optional, 1=valid
        trace_attention_mask: (B, L_trace) optional, 1=valid
        position_ids: (B, L_log+L_trace) optional
        Returns: dict with logits (lm logits on log positions), fault_reason_logits (B, num_reasons), hidden_states for decoding.
        """
        B, L_log = log_token_ids.shape
        L_trace = step_token_ids.size(1)

        embed = get_embed_tokens(self.backbone)
        h_log = embed(log_token_ids)
        h_trace = self.trace_encoder(
            step_token_ids,
            step_token_mask,
            step_parent_idx=step_parent_idx,
            step_duration=step_duration,
            step_depth=step_depth,
        )
        L = L_log + L_trace
        device = log_token_ids.device
        if position_ids is None:
            position_ids = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        rotary_emb = get_rotary_emb(self.backbone)
        position_embeddings = rotary_emb(h_log, position_ids) if rotary_emb is not None else None
        causal_mask = _causal_mask(L, device, h_log.dtype)
        for mot_layer in self.mot_layers:
            h_log, h_trace = mot_layer(
                h_log, h_trace,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
        if self.final_norm is not None:
            h_log = self.final_norm(h_log)

        if trace_attention_mask is not None:
            pool_mask = trace_attention_mask.unsqueeze(-1).float()
            trace_pooled = (h_trace * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1e-9)
        else:
            trace_pooled = h_trace.mean(dim=1)

        fault_reason_logits = self.fault_reason_head(trace_pooled)

        lm_logits = self.lm_head(h_log)

        return {
            "logits": lm_logits,
            "fault_reason_logits": fault_reason_logits,
            "h_log": h_log,
            "h_trace": h_trace,
        }

    def get_fault_reason_probs(self, fault_reason_logits: torch.Tensor) -> torch.Tensor:
        """Softmax for probability ranking over fault reasons."""
        return F.softmax(fault_reason_logits, dim=-1)

    def decode_root_cause(
        self,
        log_token_ids: torch.Tensor,
        step_token_ids: torch.Tensor,
        tokenizer: Any,
        *,
        step_token_mask: torch.Tensor | None = None,
        step_parent_idx: torch.Tensor | None = None,
        step_duration: torch.Tensor | None = None,
        step_depth: torch.Tensor | None = None,
        trace_attention_mask: torch.Tensor | None = None,
        prefix: str = "Based on the log and trace, the root cause is: ",
        max_new_tokens: int = 128,
        do_sample: bool = False,
        eos_token_id: int | None = None,
    ) -> list[str]:
        """
        Autoregressively decode root cause text for each sample in the batch.
        Prompt = tokenizer BOS (if any) + prefix; then generate until EOS or max_new_tokens.
        """
        B = log_token_ids.size(0)
        device = log_token_ids.device
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if bos_token_id is None and getattr(tokenizer, "bos_token", None):
            bos_ids = tokenizer.encode(tokenizer.bos_token, add_special_tokens=False)
            bos_token_id = bos_ids[0] if bos_ids else None
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        if bos_token_id is not None:
            prefix_ids = [bos_token_id] + prefix_ids
        if not prefix_ids:
            prefix_ids = [bos_token_id] if bos_token_id is not None else []

        results: list[str] = []
        for b in range(B):
            log_b = log_token_ids[b : b + 1]
            step_b = step_token_ids[b : b + 1]
            step_mask_b = step_token_mask[b : b + 1] if step_token_mask is not None else None
            step_parent_b = step_parent_idx[b : b + 1] if step_parent_idx is not None else None
            step_dur_b = step_duration[b : b + 1] if step_duration is not None else None
            step_dep_b = step_depth[b : b + 1] if step_depth is not None else None
            trace_mask_b = trace_attention_mask[b : b + 1] if trace_attention_mask is not None else None

            prefix_t = torch.tensor([prefix_ids], device=device, dtype=log_b.dtype)
            current_log = torch.cat([log_b, prefix_t], dim=1)
            generated: list[int] = []
            for _ in range(max_new_tokens):
                out = self.forward(
                    log_token_ids=current_log,
                    step_token_ids=step_b,
                    step_token_mask=step_mask_b,
                    step_parent_idx=step_parent_b,
                    step_duration=step_dur_b,
                    step_depth=step_dep_b,
                    trace_attention_mask=trace_mask_b,
                )
                logits = out["logits"][:, -1, :]
                if do_sample:
                    next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
                else:
                    next_tok = logits.argmax(dim=-1)
                next_id = next_tok.item()
                if eos_token_id is not None and next_id == eos_token_id:
                    break
                generated.append(next_id)
                current_log = torch.cat([current_log, next_tok.unsqueeze(-1)], dim=1)
            results.append(tokenizer.decode(generated, skip_special_tokens=True))
        return results

    def save_mot_weights(self, path: str | Path) -> None:
        """Save only MoT-specific weights (trace_encoder, mot_layers, fault_reason_head)."""
        state = {
            "trace_encoder": self.trace_encoder.state_dict(),
            "mot_layers": self.mot_layers.state_dict(),
            "fault_reason_head": self.fault_reason_head.state_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load_mot_weights(self, path: str | Path) -> None:
        """Load MoT-specific weights (trace_encoder, mot_layers, fault_reason_head)."""
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.trace_encoder.load_state_dict(state["trace_encoder"], strict=True)
        self.mot_layers.load_state_dict(state["mot_layers"], strict=True)
        self.fault_reason_head.load_state_dict(state["fault_reason_head"], strict=True)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        tokenizer: Any = None,
        trace_tokenizer: Any = None,
        mot_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Save model and tokenizers to a directory (Transformers-style).
        Dir layout: backbone/, tokenizer/, trace_tokenizer/, config.json, mot_config.json, pytorch_model.bin.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.backbone.save_pretrained(save_dir / "backbone")
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir / "tokenizer")
        if trace_tokenizer is not None:
            trace_tokenizer.save_pretrained(save_dir / "trace_tokenizer")

        if mot_config is None:
            mot_config = {}
        with open(save_dir / "mot_config.json", "w", encoding="utf-8") as f:
            json.dump(mot_config, f, ensure_ascii=False, indent=2)

        config = {
            "model_type": "teller_mot",
            "backbone_path": "backbone",
            "tokenizer_path": "tokenizer",
            "trace_tokenizer_path": "trace_tokenizer",
            "mot_config_path": "mot_config.json",
        }
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        state_dict = self.state_dict()
        torch.save(state_dict, save_dir / "pytorch_model.bin")

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str | Path,
        **kwargs: Any,
    ) -> tuple[TellerMoTForDiagnosis, Any, Any]:
        """
        Load model and tokenizers from a directory saved by save_pretrained.
        Returns: (model, tokenizer, trace_tokenizer).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from teller.trace.trace_pair_tokenizer import TracePairTokenizer

        save_dir = Path(save_directory)
        with open(save_dir / "mot_config.json", encoding="utf-8") as f:
            mot_config = json.load(f)

        backbone = AutoModelForCausalLM.from_pretrained(save_dir / "backbone", **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(save_dir / "tokenizer")
        trace_tokenizer = TracePairTokenizer.from_pretrained(save_dir / "trace_tokenizer")
        tpe_vocab_size = len(trace_tokenizer.vocab)

        trace_cfg = mot_config.get("trace", {})
        fault_labels = mot_config.get("fault_reason", {}).get("labels", [])
        num_fault_reasons = len(fault_labels)
        d = _get_backbone_hidden_size(backbone)
        encoder_hidden_size = trace_cfg.get("encoder_hidden_size", 256)

        trace_encoder = TracePairEncoder(
            vocab_size=tpe_vocab_size,
            encoder_hidden_size=encoder_hidden_size,
            d=d,
            pad_token_id=0,
            num_gcn_layers=trace_cfg.get("num_gcn_layers", 2),
            max_depth=trace_cfg.get("max_depth", 64),
            duration_log_scale=trace_cfg.get("duration_log_scale", 1e-9),
        )
        model = cls(
            backbone=backbone,
            trace_encoder=trace_encoder,
            num_fault_reasons=num_fault_reasons,
        )
        state_dict = torch.load(save_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model, tokenizer, trace_tokenizer


def build_teller_mot_from_config(
    config: dict[str, Any],
    tpe_vocab_size: int,
) -> tuple[TellerMoTForDiagnosis, Any]:
    """
    Build TellerMoTForDiagnosis from config dict. Requires transformers and torch.
    config: from load_teller_mot_config()
    tpe_vocab_size: len(TracePairTokenizer.from_pretrained(...).vocab)
    Returns: (model, backbone_tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config.get("model", {})
    name_or_path = model_cfg.get("name_or_path", "Qwen/Qwen3-0.6B")
    cache_dir = model_cfg.get("cache_dir", "data/models")

    backbone = AutoModelForCausalLM.from_pretrained(name_or_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, cache_dir=cache_dir)

    d = _get_backbone_hidden_size(backbone)
    trace_cfg = config.get("trace", {})
    encoder_hidden_size = trace_cfg.get("encoder_hidden_size", 256)
    fault_labels = config.get("fault_reason", {}).get("labels", [])
    num_fault_reasons = len(fault_labels)

    trace_encoder = TracePairEncoder(
        vocab_size=tpe_vocab_size,
        encoder_hidden_size=encoder_hidden_size,
        d=d,
        pad_token_id=0,
        num_gcn_layers=trace_cfg.get("num_gcn_layers", 2),
        max_depth=trace_cfg.get("max_depth", 64),
        duration_log_scale=trace_cfg.get("duration_log_scale", 1e-9),
    )

    model = TellerMoTForDiagnosis(
        backbone=backbone,
        trace_encoder=trace_encoder,
        num_fault_reasons=num_fault_reasons,
    )
    return model, tokenizer
