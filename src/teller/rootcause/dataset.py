"""
Root cause dataset: load tokenize_datasets; map fault_type to index (no_error as first class).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


FAULT_TYPE_NO_ERROR = "no_error"


FAULT_TYPE_ERROR = "error"


def fault_type_to_index(
    fault_type: str,
    fault_label: bool,
    labels: list[str],
    binary: bool = False,
) -> int:
    """
    Map (fault_type, fault_label) to fault_type index.
    When binary=True: 0 = no_error (none), 1 = error (any fault).
    Otherwise: no_error when fault_label is False or fault_type in ("none", "").
    """
    if not fault_label or (fault_type or "").strip().lower() in ("none", ""):
        if FAULT_TYPE_NO_ERROR in labels:
            return labels.index(FAULT_TYPE_NO_ERROR)
        return 0
    if binary:
        if FAULT_TYPE_ERROR in labels:
            return labels.index(FAULT_TYPE_ERROR)
        return 1 if len(labels) > 1 else 0
    ft = (fault_type or "").strip().lower()
    if ft in labels:
        return labels.index(ft)
    if FAULT_TYPE_NO_ERROR in labels:
        return labels.index(FAULT_TYPE_NO_ERROR)
    return 0


def build_fault_type_labels(summary_fault_types: list[str], binary: bool = False) -> list[str]:
    """Build ordered labels with no_error first. When binary=True, only [no_error, error]."""
    if binary:
        return [FAULT_TYPE_NO_ERROR, FAULT_TYPE_ERROR]
    out = [FAULT_TYPE_NO_ERROR]
    for ft in summary_fault_types:
        ft = (ft or "").strip().lower()
        if ft and ft != "none" and ft not in out:
            out.append(ft)
    return out


def split_dataset_indices(
    n: int,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Return (train_indices, valid_indices, test_indices) with deterministic shuffle. Ratios should sum to 1."""
    import random
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = max(0, n - n_train - n_valid)
    train_idx = indices[:n_train]
    valid_idx = indices[n_train : n_train + n_valid]
    test_idx = indices[n_train + n_valid : n_train + n_valid + n_test]
    return train_idx, valid_idx, test_idx


class RootCauseDataset(Dataset):
    """
    Load traces from data/tokenize_datasets/<run>/steps/*.json.
    Each sample: log_token_ids, step_token_ids (TPE), fault_type_id, root_cause_token_ids.
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
        max_root_cause_tokens: int = 256,
        default_root_cause: str = "No error in current trace and log.",
        root_cause_prefix: str = "Based on the log and trace, the root cause is: ",
        binary_fault_type: bool = False,
    ):
        self.tokenize_datasets_dir = Path(tokenize_datasets_dir)
        self.backbone_tokenizer = backbone_tokenizer
        self.fault_type_labels = list(fault_type_labels)
        self.project_root = Path(project_root) if project_root else self.tokenize_datasets_dir.resolve().parents[2]
        self.max_steps_per_trace = max_steps_per_trace
        self.max_log_tokens = max_log_tokens
        self.max_trace_tokens_per_step = max_trace_tokens_per_step
        self.max_root_cause_tokens = max_root_cause_tokens
        self.default_root_cause = default_root_cause or "No error in current trace and log."
        self.root_cause_prefix = root_cause_prefix or "Based on the log and trace, the root cause is: "
        self.binary_fault_type = binary_fault_type

        self.steps_dir = self.tokenize_datasets_dir / "steps"
        if not self.steps_dir.exists():
            raise FileNotFoundError(f"Steps dir not found: {self.steps_dir}")

        self._trace_files = sorted(self.steps_dir.glob("*.json"))
        self._samples: list[dict[str, Any]] = []

        for path in self._trace_files:
            with open(path, encoding="utf-8") as f:
                steps = json.load(f)
            if not steps:
                continue
            first = steps[0]
            trace_id = first.get("trace_id", path.stem)
            log_path = first.get("log_path", "")
            fault_type = first.get("fault_type", "none")
            fault_label = first.get("fault_label", False)
            root_cause = first.get("root_cause", "")

            if self.max_steps_per_trace is not None:
                steps = steps[: self.max_steps_per_trace]

            step_input_ids = [s["input_ids"] for s in steps]
            step_parent_idx = [s.get("parent_idx", [-1] * len(s["input_ids"])) for s in steps]
            step_duration = [s.get("duration", [0] * len(s["input_ids"])) for s in steps]
            step_depth = [s.get("depth", [0] * len(s["input_ids"])) for s in steps]

            fault_type_id = fault_type_to_index(
                fault_type, fault_label, self.fault_type_labels, binary=self.binary_fault_type
            )

            self._samples.append({
                "trace_id": trace_id,
                "log_path": log_path,
                "step_input_ids": step_input_ids,
                "step_parent_idx": step_parent_idx,
                "step_duration": step_duration,
                "step_depth": step_depth,
                "fault_type_id": fault_type_id,
                "fault_type": fault_type,
                "fault_label": fault_label,
                "root_cause": root_cause,
                "num_steps": len(steps),
            })

    def __len__(self) -> int:
        return len(self._samples)

    def _load_log_text(self, log_path: str) -> str:
        p = Path(log_path)
        if not p.is_absolute():
            p = self.project_root / log_path
        if not p.exists():
            return ""
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._samples[idx]
        log_text = self._load_log_text(sample["log_path"])
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

        rc_text = (sample["root_cause"] or "").strip() or self.default_root_cause
        rc_full = self.root_cause_prefix + rc_text
        rc_enc = self.backbone_tokenizer(
            rc_full,
            max_length=self.max_root_cause_tokens,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        root_cause_token_ids = rc_enc.get("input_ids", [])
        if not root_cause_token_ids and getattr(self.backbone_tokenizer, "eos_token_id", None) is not None:
            root_cause_token_ids = [self.backbone_tokenizer.eos_token_id]

        return {
            "log_token_ids": log_token_ids,
            "log_attention_mask": log_attention_mask,
            "step_token_ids": padded_steps,
            "step_token_mask": step_masks,
            "fault_type_id": sample["fault_type_id"],
            "root_cause": sample["root_cause"],
            "root_cause_token_ids": root_cause_token_ids,
            "trace_id": sample["trace_id"],
            "num_steps": sample["num_steps"],
        }


def rootcause_collate_fn(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    """Pad step_token_ids to max L_trace in batch; pad root_cause_token_ids with pad_token_id."""
    B = len(batch)
    max_trace = max(s["num_steps"] for s in batch)
    L_tok = len(batch[0]["step_token_ids"][0])
    L_log = batch[0]["log_token_ids"].shape[0]

    log_token_ids = torch.stack([s["log_token_ids"] for s in batch])
    step_token_ids = torch.zeros(B, max_trace, L_tok, dtype=torch.long)
    step_token_mask = torch.zeros(B, max_trace, L_tok, dtype=torch.float)

    fault_type_ids = []
    root_cause_token_ids_list = [s["root_cause_token_ids"] for s in batch]
    max_rc = max(len(rc) for rc in root_cause_token_ids_list)
    root_cause_token_ids = torch.zeros(B, max_rc, dtype=torch.long)
    pad_id = pad_token_id

    for i, s in enumerate(batch):
        n = s["num_steps"]
        step_token_ids[i, :n] = torch.tensor(s["step_token_ids"], dtype=torch.long)
        step_token_mask[i, :n] = torch.tensor(s["step_token_mask"], dtype=torch.float)
        fault_type_ids.append(s["fault_type_id"])
        rc = s["root_cause_token_ids"]
        root_cause_token_ids[i, : len(rc)] = torch.tensor(rc, dtype=torch.long)
        if len(rc) < max_rc:
            root_cause_token_ids[i, len(rc) :] = pad_id

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
        "fault_type_labels": torch.tensor(fault_type_ids, dtype=torch.long),
        "root_cause_token_ids": root_cause_token_ids,
        "root_cause_lengths": torch.tensor([len(rc) for rc in root_cause_token_ids_list], dtype=torch.long),
        "log_attention_mask": log_attention_mask,
        "trace_attention_mask": trace_attention_mask,
        "trace_ids": [s["trace_id"] for s in batch],
    }