#!/usr/bin/env python3
"""
Train TellerMoTForRootCauseGenerating with Hugging Face Trainer.
Logs loss_rc and loss_fault via callback (no periodic eval / accuracy metrics).

Usage:
  python scripts/train_rc.py --config configs/rootcause/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Subset
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

from teller.rootcause import (
    TellerMoTForRootCauseGenerating,
    RootCauseDataset,
    rootcause_collate_fn,
    build_fault_type_labels,
    split_dataset_indices,
)

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class RootCauseTrainerCallback(TrainerCallback):
    """Inject loss_rc and loss_fault into logs when Trainer logs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        if getattr(state, "_last_loss_rc", None) is not None:
            logs["loss_rc"] = state._last_loss_rc
            state._last_loss_rc = None
        if getattr(state, "_last_loss_fault", None) is not None:
            logs["loss_fault"] = state._last_loss_fault
            state._last_loss_fault = None
        return control


class RootCauseTrainer(Trainer):
    """Trainer that passes batch to our model and logs loss_rc/loss_fault."""

    def __init__(self, pad_token_id: int, fault_type_labels: list[str] | None = None, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self._pad_token_id = pad_token_id
        self._fault_type_labels = fault_type_labels or []
        self._tokenizer = tokenizer

    _MODEL_INPUT_KEYS = frozenset({
        "log_token_ids", "step_token_ids", "root_cause_token_ids", "step_token_mask",
        "fault_type_labels", "log_attention_mask", "trace_attention_mask",
    })

    def _model_inputs(self, inputs):
        out = {k: inputs[k] for k in self._MODEL_INPUT_KEYS if k in inputs}
        if "fault_type_labels" not in out and "fault_type_id" in inputs:
            out["fault_type_labels"] = inputs["fault_type_id"]
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = self._model_inputs(inputs)
        outputs = model(**model_inputs)
        loss = outputs.get("loss")
        if loss is None:
            loss = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        if isinstance(loss, torch.Tensor) and loss.numel() > 1:
            loss = loss.mean()
        if self.state is not None:
            lr = outputs.get("loss_rc")
            lf = outputs.get("loss_fault")

            def _to_scalar(t):
                if t is None or not isinstance(t, torch.Tensor):
                    return None
                return t.mean().item() if t.numel() > 1 else t.item()

            self.state._last_loss_rc = _to_scalar(lr)
            self.state._last_loss_fault = _to_scalar(lf)
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        """Save via model.save_pretrained (backbone + adapter) to avoid safetensors tied-weights error."""
        import os

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.args.should_log:
            import logging

            logging.getLogger("transformers.trainer").info(f"Saving model checkpoint to {output_dir}")
        unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        unwrapped.save_pretrained(output_dir, config_extra={"fault_type_labels": self._fault_type_labels})
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(Path(output_dir) / "tokenizer")
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rootcause/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    if args.output_dir:
        config.setdefault("training", {})["output_dir"] = args.output_dir
    if args.epochs is not None:
        config.setdefault("training", {})["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["per_device_train_batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.data_dir:
        config.setdefault("data", {})["tokenize_datasets_dir"] = args.data_dir

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    project_root = ROOT
    tokenize_dir = Path(data_cfg.get("tokenize_datasets_dir", "data/tokenize_datasets/256"))
    if not tokenize_dir.is_absolute():
        tokenize_dir = project_root / tokenize_dir

    summary_path = tokenize_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        fault_types_from_summary = summary.get("fault_types", [])
    else:
        fault_types_from_summary = []
    binary_fault_type = config.get("fault_type_binary", False)
    fault_type_labels = config.get("fault_type_labels") or build_fault_type_labels(
        fault_types_from_summary, binary=binary_fault_type
    )
    num_fault_types = len(fault_type_labels)

    backbone_name = model_cfg.get("backbone", "Qwen/Qwen2.5-0.5B")
    cache_dir = model_cfg.get("cache_dir", "data/models")
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = str(project_root / cache_dir)
    trace_vocab_size = model_cfg.get("trace_vocab_size", 256)
    dtype_str = model_cfg.get("dtype", "bf16")
    _dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = _dt.get(dtype_str.strip().lower(), torch.bfloat16)

    backbone = AutoModelForCausalLM.from_pretrained(
        backbone_name, cache_dir=cache_dir, torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(backbone_name, cache_dir=cache_dir)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = TellerMoTForRootCauseGenerating(
        backbone=backbone,
        trace_vocab_size=trace_vocab_size,
        num_fault_types=num_fault_types,
        trace_pad_token_id=0,
        pad_token_id=pad_token_id,
    )
    model = model.to(torch_dtype)

    dataset = RootCauseDataset(
        tokenize_datasets_dir=tokenize_dir,
        backbone_tokenizer=tokenizer,
        fault_type_labels=fault_type_labels,
        project_root=project_root,
        max_steps_per_trace=data_cfg.get("max_steps_per_trace"),
        max_log_tokens=data_cfg.get("max_log_tokens", 512),
        max_trace_tokens_per_step=data_cfg.get("max_trace_tokens_per_step", 256),
        max_root_cause_tokens=data_cfg.get("max_root_cause_tokens", 256),
        default_root_cause=data_cfg.get("default_root_cause", "No error in current trace and log."),
        root_cause_prefix=data_cfg.get("root_cause_prefix", "Based on the log and trace, the root cause is: "),
        binary_fault_type=binary_fault_type,
    )

    def data_collator(examples):
        return rootcause_collate_fn(examples, pad_token_id=pad_token_id)

    train_ratio = data_cfg.get("train_ratio", 0.8)
    valid_ratio = data_cfg.get("valid_ratio", 0.1)
    test_ratio = data_cfg.get("test_ratio", 0.1)
    split_seed = data_cfg.get("split_seed", 42)
    train_idx, valid_idx, test_idx = split_dataset_indices(
        len(dataset), train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, seed=split_seed
    )
    train_dataset = Subset(dataset, train_idx)

    output_dir = train_cfg.get("output_dir", "output/rootcause/default")
    if not Path(output_dir).is_absolute():
        output_dir = str(project_root / output_dir)

    logging_steps = train_cfg.get("logging_steps", 10)
    save_steps = train_cfg.get("save_steps", 200)
    save_strategy = train_cfg.get("save_strategy", "steps")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 5e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        logging_steps=logging_steps,
        logging_first_step=True,
        eval_strategy="no",
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_total_limit=train_cfg.get("save_total_limit", 2),
        bf16=train_cfg.get("bf16", True),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", True),
        remove_unused_columns=False,
        report_to=train_cfg.get("report_to", "none"),
        ddp_find_unused_parameters=False,
    )

    trainer = RootCauseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        pad_token_id=pad_token_id,
        fault_type_labels=fault_type_labels,
        tokenizer=tokenizer,
        callbacks=[RootCauseTrainerCallback()],
    )

    trainer.train()

    final_dir = Path(output_dir) / "final"
    if trainer.is_world_process_zero():
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = trainer.accelerator.unwrap_model(trainer.model, keep_torch_compile=False)
        unwrapped.save_pretrained(final_dir, config_extra={"fault_type_labels": fault_type_labels})
        tokenizer.save_pretrained(final_dir / "tokenizer")
        with open(final_dir / "fault_type_labels.json", "w", encoding="utf-8") as f:
            json.dump(fault_type_labels, f, indent=2)
        split_info = {
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
            "split_seed": split_seed,
            "n_train": len(train_idx),
            "n_valid": len(valid_idx),
            "n_test": len(test_idx),
        }
        with open(final_dir / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2)
        print(f"Saved final model to {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
