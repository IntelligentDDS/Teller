"""
Single-stream root cause dataset: fault_type inside label as plain text.

All of <fault_type>, </fault_type>, and the fault type string (e.g. "none") are part of the
LLM text vocabulary. They are tokenized by the backbone tokenizer and decoded by the LLM's
own lm_head—fault type is treated as normal text semantics, not a separate head or vocab.
Label format: "Based on the log and trace, the root cause is: <fault_type> {type} </fault_type> {rc_text}"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from teller.rootcause.dataset import (
    FAULT_TYPE_NO_ERROR,
    RootCauseDataset,
    build_fault_type_labels,
    fault_type_to_index,
)

# Tags in the label string; added to LLM vocab as normal tokens, decoded by lm_head
FAULT_TYPE_OPEN = "<fault_type>"
FAULT_TYPE_CLOSE = "</fault_type>"

SINGLE_STREAM_PREFIX = "Based on the log and trace, the root cause is: "
DEFAULT_RC_NO_ERROR = "No error in current trace and log."


def fault_type_to_label_str(fault_type: str, fault_label: bool, binary: bool = False) -> str:
    """Map (fault_type, fault_label) to string for inside <fault_type> </fault_type>. Returns 'none' or 'error' (when binary) or specific type."""
    if not fault_label or (fault_type or "").strip().lower() in ("none", ""):
        return "none"
    if binary:
        return "error"
    return (fault_type or "").strip().lower()


def build_single_stream_label(
    fault_type: str,
    fault_label: bool,
    root_cause: str,
    default_root_cause: str = DEFAULT_RC_NO_ERROR,
    binary: bool = False,
) -> str:
    """Build full label: prefix + <fault_type> type </fault_type> + rc_text. When binary=True, type is 'none' or 'error'."""
    type_str = fault_type_to_label_str(fault_type, fault_label, binary=binary)
    rc_text = (root_cause or "").strip() or default_root_cause
    return f"{SINGLE_STREAM_PREFIX}{FAULT_TYPE_OPEN} {type_str} {FAULT_TYPE_CLOSE} {rc_text}"


def get_fault_type_special_tokens() -> list[str]:
    return [FAULT_TYPE_OPEN, FAULT_TYPE_CLOSE]


class SingleStreamRootCauseDataset(Dataset):
    """
    Same data as RootCauseDataset but label = single string with <fault_type>...</fault_type> + root cause.
    Each sample: log_token_ids, step_token_ids (TPE), label_token_ids (full label), fault_type_id.
    """

    def __init__(
        self,
        tokenize_datasets_dir: str | Path,
        backbone_tokenizer: Any,
        fault_type_labels: list[str],
        project_root: str | Path | None = None,
        max_steps_per_trace: int | None = None,
        max_log_tokens: int = 512,
        max_trace_tokens_per_step: int = 256,
        max_label_tokens: int = 384,
        default_root_cause: str = DEFAULT_RC_NO_ERROR,
        binary_fault_type: bool = False,
    ):
        self.tokenize_datasets_dir = Path(tokenize_datasets_dir)
        self.backbone_tokenizer = backbone_tokenizer
        self.fault_type_labels = list(fault_type_labels)
        self.project_root = Path(project_root) if project_root else self.tokenize_datasets_dir.resolve().parents[2]
        self.max_steps_per_trace = max_steps_per_trace
        self.max_log_tokens = max_log_tokens
        self.max_trace_tokens_per_step = max_trace_tokens_per_step
        self.max_label_tokens = max_label_tokens
        self.default_root_cause = default_root_cause or DEFAULT_RC_NO_ERROR
        self.binary_fault_type = binary_fault_type

        self._base = RootCauseDataset(
            tokenize_datasets_dir=tokenize_datasets_dir,
            backbone_tokenizer=backbone_tokenizer,
            fault_type_labels=fault_type_labels,
            project_root=project_root,
            max_steps_per_trace=max_steps_per_trace,
            max_log_tokens=max_log_tokens,
            max_trace_tokens_per_step=max_trace_tokens_per_step,
            max_root_cause_tokens=max_label_tokens,
            default_root_cause=default_root_cause,
            root_cause_prefix="",
            binary_fault_type=binary_fault_type,
        )
        self._samples = self._base._samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._samples[idx]
        log_text = self._base._load_log_text(sample["log_path"])
        log_enc = self.backbone_tokenizer(
            log_text,
            max_length=self.max_log_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        log_token_ids = log_enc["input_ids"].squeeze(0)
        log_attention_mask = log_enc.get("attention_mask")
        if log_attention_mask is not None:
            log_attention_mask = log_attention_mask.squeeze(0)

        step_input_ids = sample["step_input_ids"]
        step_masks: list[list[int]] = []
        padded_steps: list[list[int]] = []
        for ids in step_input_ids:
            orig_len = len(ids)
            if orig_len > self.max_trace_tokens_per_step:
                ids = ids[: self.max_trace_tokens_per_step]
                orig_len = self.max_trace_tokens_per_step
            pad_len = self.max_trace_tokens_per_step - len(ids)
            padded_steps.append(ids + [0] * pad_len)
            step_masks.append([1] * orig_len + [0] * pad_len)

        fault_type = sample.get("fault_type", "none")
        fault_label = sample.get("fault_label", False)
        root_cause = sample.get("root_cause", "")
        full_label = build_single_stream_label(
            fault_type, fault_label, root_cause, self.default_root_cause, binary=self.binary_fault_type
        )
        label_enc = self.backbone_tokenizer(
            full_label,
            max_length=self.max_label_tokens,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        label_token_ids = label_enc.get("input_ids", [])
        if not label_token_ids and getattr(self.backbone_tokenizer, "eos_token_id", None) is not None:
            label_token_ids = [self.backbone_tokenizer.eos_token_id]

        return {
            "log_token_ids": log_token_ids,
            "log_attention_mask": log_attention_mask,
            "step_token_ids": padded_steps,
            "step_token_mask": step_masks,
            "fault_type_id": sample["fault_type_id"],
            "label_token_ids": label_token_ids,
            "trace_id": sample["trace_id"],
            "num_steps": sample["num_steps"],
        }


def single_stream_collate_fn(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    """Pad step_token_ids and label_token_ids."""
    B = len(batch)
    max_trace = max(s["num_steps"] for s in batch)
    L_tok = len(batch[0]["step_token_ids"][0])
    L_log = batch[0]["log_token_ids"].shape[0]

    log_token_ids = torch.stack([s["log_token_ids"] for s in batch])
    step_token_ids = torch.zeros(B, max_trace, L_tok, dtype=torch.long)
    step_token_mask = torch.zeros(B, max_trace, L_tok, dtype=torch.float)

    fault_type_ids = []
    label_list = [s["label_token_ids"] for s in batch]
    max_label = max(len(l) for l in label_list)
    label_token_ids = torch.zeros(B, max_label, dtype=torch.long)
    pad_id = pad_token_id

    for i, s in enumerate(batch):
        n = s["num_steps"]
        step_token_ids[i, :n] = torch.tensor(s["step_token_ids"], dtype=torch.long)
        step_token_mask[i, :n] = torch.tensor(s["step_token_mask"], dtype=torch.float)
        fault_type_ids.append(s["fault_type_id"])
        lbl = s["label_token_ids"]
        label_token_ids[i, : len(lbl)] = torch.tensor(lbl, dtype=torch.long)
        if len(lbl) < max_label:
            label_token_ids[i, len(lbl) :] = pad_id

    log_attention_mask = None
    if batch[0].get("log_attention_mask") is not None:
        log_attention_mask = torch.stack([s["log_attention_mask"] for s in batch])
    trace_attention_mask = torch.zeros(B, max_trace, dtype=torch.float)
    for i, s in enumerate(batch):
        trace_attention_mask[i, : s["num_steps"]] = 1.0

    return {
        "log_token_ids": log_token_ids,
        "step_token_ids": step_token_ids,
        "step_token_mask": step_token_mask,
        "fault_type_id": torch.tensor(fault_type_ids, dtype=torch.long),
        "label_token_ids": label_token_ids,
        "label_lengths": torch.tensor([len(l) for l in label_list], dtype=torch.long),
        "log_attention_mask": log_attention_mask,
        "trace_attention_mask": trace_attention_mask,
        "trace_ids": [s["trace_id"] for s in batch],
    }
