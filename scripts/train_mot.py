#!/usr/bin/env python3
"""
Train MoT with Hugging Face Trainer. Supports single-GPU, multi-GPU (DDP), and CPU.

Requires: pip install -e ".[mot]" (torch, transformers, torch-geometric, accelerate).

Usage:
  # Single GPU or CPU
  python scripts/train_mot.py --config configs/mot/train.yaml

  # Multi-GPU (e.g. 4 GPUs) via accelerate
  accelerate launch --num_processes 4 scripts/train_mot.py --config configs/mot/train.yaml

  # FSDP: in config set training.fsdp and training.fsdp_config, then e.g.:
  accelerate launch --num_processes 8 scripts/train_mot.py --config configs/mot/train.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

from teller.mot.config import load_teller_mot_config
from teller.mot.dataset import TellerMoTDataset, teller_mot_collate_fn
from teller.mot.modeling_mot import build_teller_mot_from_config
from teller.trace.trace_pair_tokenizer import TracePairTokenizer


def _get_prefix_token_ids(tokenizer, prefix: str):
    """BOS (if any) + prefix token ids, list."""
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None and getattr(tokenizer, "bos_token", None):
        bos_ids = tokenizer.encode(tokenizer.bos_token, add_special_tokens=False)
        bos_token_id = bos_ids[0] if bos_ids else None
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if bos_token_id is not None:
        prefix_ids = [bos_token_id] + prefix_ids
    if not prefix_ids and bos_token_id is not None:
        prefix_ids = [bos_token_id]
    return prefix_ids


def _build_root_cause_labels(
    batch_size: int,
    L_log: int,
    L_prefix: int,
    max_rc: int,
    root_cause_token_ids: torch.Tensor,
    root_cause_lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Labels for CE on log stream: shape (B, L_log + L_prefix + max_rc - 1).
    -100 for log + prefix positions; for root-cause span use root_cause_token_ids[:, 1:]; padding -100.
    """
    L_total = L_log + L_prefix + max_rc - 1
    labels = torch.full((batch_size, L_total), -100, dtype=torch.long, device=device)
    for b in range(batch_size):
        rc_len = root_cause_lengths[b].item()
        if rc_len <= 1:
            continue
        start = L_log + L_prefix
        end = start + (rc_len - 1)
        labels[b, start:end] = root_cause_token_ids[b, 1:rc_len]
    return labels


class MotTrainer(Trainer):
    """Trainer with custom loss: L_fault + root_cause_weight * L_rc."""

    def __init__(self, prefix_ids: list[int] | None = None, fault_reason_weight: float = 1.0, root_cause_weight: float = 1.0, **kwargs: Any):
        self._prefix_ids = kwargs.pop("prefix_ids", prefix_ids or [])
        self._fault_reason_weight = kwargs.pop("fault_reason_weight", fault_reason_weight)
        self._root_cause_weight = kwargs.pop("root_cause_weight", root_cause_weight)
        super().__init__(**kwargs)

    def compute_loss(self, model: torch.nn.Module, inputs: dict[str, Any], return_outputs: bool = False, num_items_in_batch: int | None = None):
        device = inputs["log_token_ids"].device
        prefix_tensor = torch.tensor(
            [self._prefix_ids], dtype=torch.long, device=device
        )
        log_token_ids = inputs["log_token_ids"]
        root_cause_token_ids = inputs["root_cause_token_ids"]
        root_cause_lengths = inputs["root_cause_lengths"]
        B, L_log = log_token_ids.shape
        max_rc = root_cause_token_ids.size(1)
        L_prefix = len(self._prefix_ids)
        prefix_b = prefix_tensor.expand(B, -1)
        log_with_rc = torch.cat(
            [log_token_ids, prefix_b, root_cause_token_ids[:, :-1]],
            dim=1,
        )
        out = model(
            log_token_ids=log_with_rc,
            step_token_ids=inputs["step_token_ids"],
            step_token_mask=inputs.get("step_token_mask"),
            step_parent_idx=inputs.get("step_parent_idx"),
            step_duration=inputs.get("step_duration"),
            step_depth=inputs.get("step_depth"),
            log_attention_mask=inputs.get("log_attention_mask"),
            trace_attention_mask=inputs.get("trace_attention_mask"),
        )
        L_fault = F.cross_entropy(out["fault_reason_logits"], inputs["fault_reason_id"])
        labels_rc = _build_root_cause_labels(
            B, L_log, L_prefix, max_rc,
            root_cause_token_ids, root_cause_lengths, device,
        )
        logits_rc = out["logits"]
        has_rc_labels = (labels_rc != -100).any().item()
        if has_rc_labels:
            L_rc = F.cross_entropy(
                logits_rc.reshape(-1, logits_rc.size(-1)),
                labels_rc.reshape(-1),
                ignore_index=-100,
            )
            if torch.isnan(L_rc).item():
                L_rc = torch.tensor(0.0, device=device, dtype=logits_rc.dtype)
        else:
            L_rc = torch.tensor(0.0, device=device, dtype=logits_rc.dtype)
        loss = self._fault_reason_weight * L_fault + self._root_cause_weight * L_rc
        if torch.isnan(loss).item():
            loss = L_fault.clone()
        if torch.isnan(loss).item():
            loss = torch.tensor(0.0, device=device, dtype=logits_rc.dtype, requires_grad=True)
        return (loss, out) if return_outputs else loss


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MoT (fault reason + root cause)")
    parser.add_argument("--config", type=str, default="configs/mot/train.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--tpe-dir", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda / cpu). Default: cuda if available.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    config = load_teller_mot_config(args.config)
    data_cfg = config.get("data", {})
    tokenize_dir = Path(args.data_dir or data_cfg.get("tokenize_datasets_dir", "data/tokenize_datasets/256"))
    if not tokenize_dir.is_absolute():
        tokenize_dir = repo / tokenize_dir

    trace_cfg = config.get("trace", {})
    tpe_dir = args.tpe_dir or trace_cfg.get("tokenizer_dir", "data/tpe_tokenizer/256")
    if not Path(tpe_dir).is_absolute():
        tpe_dir = str(repo / tpe_dir)
    if not Path(tpe_dir).exists() and (tokenize_dir / "summary.json").exists():
        with open(tokenize_dir / "summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        tpe_dir = summary.get("tokenizer_dir", tpe_dir)
    tpe_path = Path(tpe_dir)
    if not (tpe_path / "vocab.json").exists():
        raise SystemExit(f"TPE tokenizer not found at {tpe_path}. Run TPE training first.")

    tpe_tokenizer = TracePairTokenizer.from_pretrained(tpe_dir)
    tpe_vocab_size = len(tpe_tokenizer.vocab)
    model, backbone_tokenizer = build_teller_mot_from_config(config, tpe_vocab_size)
    if args.device == "cpu":
        model = model.float()

    train_cfg = config.get("training", {})
    if args.max_steps is not None:
        train_cfg = {**train_cfg, "max_steps": args.max_steps}
    if args.batch_size is not None:
        train_cfg = {**train_cfg, "batch_size": args.batch_size}

    max_steps = train_cfg.get("max_steps", 8)
    batch_size = train_cfg.get("batch_size", 1)
    fault_reason_weight = train_cfg.get("fault_reason_weight", 1.0)
    root_cause_weight = train_cfg.get("root_cause_weight", 1.0)
    freeze_backbone = train_cfg.get("freeze_backbone", True)
    max_root_cause_tokens = train_cfg.get("max_root_cause_tokens", 32)
    root_cause_prefix = train_cfg.get(
        "root_cause_prefix", "Based on the log and trace, the root cause is: "
    )

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    prefix_ids = _get_prefix_token_ids(backbone_tokenizer, root_cause_prefix)

    fault_labels = config.get("fault_reason", {}).get("labels", [])
    default_root_cause = train_cfg.get(
        "default_root_cause", "No error in current trace and log."
    )
    dataset = TellerMoTDataset(
        tokenize_datasets_dir=tokenize_dir,
        backbone_tokenizer=backbone_tokenizer,
        fault_reason_labels=fault_labels,
        project_root=repo,
        max_steps_per_trace=data_cfg.get("max_steps_per_trace"),
        max_log_tokens=data_cfg.get("max_log_tokens", 512),
        max_trace_tokens_per_step=data_cfg.get("max_trace_tokens_per_step", 256),
        max_root_cause_tokens=max_root_cause_tokens,
        default_root_cause=default_root_cause,
    )

    out_cfg = config.get("output", {})
    save_dir = Path(out_cfg.get("save_dir", "output/test/model/mot_trained"))
    if not save_dir.is_absolute():
        save_dir = repo / save_dir
    save_steps = out_cfg.get("save_steps", 4)
    log_steps = out_cfg.get("log_steps", 1)

    fsdp_cfg = train_cfg.get("fsdp")
    fsdp_config = train_cfg.get("fsdp_config")
    if fsdp_cfg is not None and isinstance(fsdp_cfg, str):
        fsdp_cfg = [fsdp_cfg]
    if isinstance(fsdp_config, dict):
        fsdp_config = dict(fsdp_config)
        for key in ("fsdp_min_num_params", "min_num_params"):
            if key in fsdp_config:
                v = fsdp_config[key]
                fsdp_config[key] = int(float(v)) if v is not None else 0
        for key in ("fsdp_offload_params", "offload_params"):
            if key in fsdp_config:
                v = fsdp_config[key]
                fsdp_config[key] = bool(v) if not isinstance(v, str) else (v.lower() in ("true", "1", "yes"))
    training_args = TrainingArguments(
        output_dir=str(save_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        logging_steps=log_steps,
        save_steps=save_steps,
        save_total_limit=out_cfg.get("save_total_limit"),
        eval_strategy="no",
        bf16=train_cfg.get("bf16", False),
        fp16=train_cfg.get("fp16", False),
        remove_unused_columns=False,
        dataloader_num_workers=out_cfg.get("dataloader_num_workers", 0),
        use_cpu=(args.device == "cpu"),
        report_to="none",
        fsdp=fsdp_cfg if fsdp_cfg else [],
        fsdp_config=fsdp_config if isinstance(fsdp_config, dict) else None,
    )

    callbacks: list[TrainerCallback] = []

    trainer = MotTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=teller_mot_collate_fn,
        callbacks=callbacks,
        prefix_ids=prefix_ids,
        fault_reason_weight=fault_reason_weight,
        root_cause_weight=root_cause_weight,
    )

    trainer.train()

    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    print("Training done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
