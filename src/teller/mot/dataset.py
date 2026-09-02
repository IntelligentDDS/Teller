"""
TellerMoTDataset: load tokenize_datasets/256 (steps/*.json, summary.json); group by trace; map fault_type/fault_label to fault_reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def fault_type_to_reason_index(
    fault_type: str,
    fault_label: bool,
    labels: list[str],
) -> int:
    """
    Map (fault_type, fault_label) to single fault_reason index.
    no_error when fault_label is False or fault_type in ("none", "").
    """
    if not fault_label or (fault_type or "").strip().lower() in ("none", ""):
        if "no_error" in labels:
            return labels.index("no_error")
        return 0
    ft = (fault_type or "").strip().lower()
    if ft in labels:
        return labels.index(ft)
    if "no_error" in labels:
        return labels.index("no_error")
    return 0


class TellerMoTDataset(Dataset):
    """
    Load traces from data/tokenize_datasets/<run>/steps/*.json.
    Each sample: log_token_ids (from log_path with backbone tokenizer), step_token_ids (from step input_ids),
    fault_reason_id, root_cause text, trace_id.
    """

    def __init__(
        self,
        tokenize_datasets_dir: str | Path,
        backbone_tokenizer: Any,
        fault_reason_labels: list[str],
        project_root: str | Path | None = None,
        max_steps_per_trace: int | None = None,
        max_log_tokens: int = 512,
        max_trace_tokens_per_step: int = 256,
        max_root_cause_tokens: int = 64,
        default_root_cause: str = "No error in current trace and log.",
    ):
        self.tokenize_datasets_dir = Path(tokenize_datasets_dir)
        self.backbone_tokenizer = backbone_tokenizer
        self.fault_reason_labels = list(fault_reason_labels)
        self.project_root = Path(project_root) if project_root else self.tokenize_datasets_dir.resolve().parents[2]
        self.max_steps_per_trace = max_steps_per_trace
        self.max_log_tokens = max_log_tokens
        self.max_trace_tokens_per_step = max_trace_tokens_per_step
        self.max_root_cause_tokens = max_root_cause_tokens
        self.default_root_cause = default_root_cause or "No error in current trace and log."

        self.steps_dir = self.tokenize_datasets_dir / "steps"
        if not self.steps_dir.exists():
            raise FileNotFoundError(f"Steps dir not found: {self.steps_dir}")

        self._trace_files: list[Path] = sorted(self.steps_dir.glob("*.json"))
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

            fault_reason_id = fault_type_to_reason_index(
                fault_type, fault_label, self.fault_reason_labels
            )

            self._samples.append({
                "trace_id": trace_id,
                "log_path": log_path,
                "step_input_ids": step_input_ids,
                "step_parent_idx": step_parent_idx,
                "step_duration": step_duration,
                "step_depth": step_depth,
                "fault_reason_id": fault_reason_id,
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
        step_parent_idx = sample.get("step_parent_idx", [])
        step_duration = sample.get("step_duration", [])
        step_depth = sample.get("step_depth", [])

        padded_steps: list[list[int]] = []
        step_masks: list[list[int]] = []
        padded_parent: list[list[int]] = []
        padded_duration: list[list[float]] = []
        padded_depth: list[list[int]] = []
        for i, ids in enumerate(step_input_ids):
            orig_len = len(ids)
            if orig_len > self.max_trace_tokens_per_step:
                ids = ids[: self.max_trace_tokens_per_step]
                orig_len = self.max_trace_tokens_per_step
            pad_len = self.max_trace_tokens_per_step - len(ids)
            padded_steps.append(ids + [0] * pad_len)
            step_masks.append([1] * orig_len + [0] * pad_len)

            p = step_parent_idx[i] if i < len(step_parent_idx) else [-1] * orig_len
            d = step_duration[i] if i < len(step_duration) else [0] * orig_len
            dep = step_depth[i] if i < len(step_depth) else [0] * orig_len
            if len(p) > orig_len:
                p, d, dep = p[:orig_len], d[:orig_len], dep[:orig_len]
            elif len(p) < orig_len:
                p = p + [-1] * (orig_len - len(p))
                d = d + [0] * (orig_len - len(d))
                dep = dep + [0] * (orig_len - len(dep))
            padded_parent.append(p + [-1] * pad_len)
            padded_duration.append(d + [0.0] * pad_len)
            padded_depth.append(dep + [0] * pad_len)

        rc_text = (sample["root_cause"] or "").strip() or self.default_root_cause
        rc_enc = self.backbone_tokenizer(
            rc_text,
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
            "step_parent_idx": padded_parent,
            "step_duration": padded_duration,
            "step_depth": padded_depth,
            "fault_reason_id": sample["fault_reason_id"],
            "root_cause": sample["root_cause"],
            "root_cause_token_ids": root_cause_token_ids,
            "trace_id": sample["trace_id"],
            "num_steps": sample["num_steps"],
        }


def teller_mot_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad step_token_ids to max L_trace in batch; stack to tensors."""
    B = len(batch)
    max_trace = max(s["num_steps"] for s in batch)
    L_tok = len(batch[0]["step_token_ids"][0])
    L_log = batch[0]["log_token_ids"].shape[0]

    log_token_ids = torch.stack([s["log_token_ids"] for s in batch])
    step_token_ids = torch.zeros(B, max_trace, L_tok, dtype=torch.long)
    step_token_mask = torch.zeros(B, max_trace, L_tok, dtype=torch.float)
    step_parent_idx = torch.zeros(B, max_trace, L_tok, dtype=torch.long)
    step_duration = torch.zeros(B, max_trace, L_tok, dtype=torch.float)
    step_depth = torch.zeros(B, max_trace, L_tok, dtype=torch.long)

    fault_reason_ids = []
    root_causes = []
    root_cause_token_ids_list = [s["root_cause_token_ids"] for s in batch]
    max_rc = max(len(rc) for rc in root_cause_token_ids_list)
    root_cause_token_ids = torch.zeros(B, max_rc, dtype=torch.long)
    for i, rc in enumerate(root_cause_token_ids_list):
        root_cause_token_ids[i, : len(rc)] = torch.tensor(rc, dtype=torch.long)

    trace_ids = []
    log_attention_mask = None
    if batch[0].get("log_attention_mask") is not None:
        log_attention_mask = torch.stack([s["log_attention_mask"] for s in batch])
    trace_attention_mask = torch.zeros(B, max_trace, dtype=torch.float)

    for i, s in enumerate(batch):
        n = s["num_steps"]
        step_token_ids[i, :n] = torch.tensor(s["step_token_ids"], dtype=torch.long)
        step_token_mask[i, :n] = torch.tensor(s["step_token_mask"], dtype=torch.float)
        step_parent_idx[i, :n] = torch.tensor(s["step_parent_idx"], dtype=torch.long)
        step_duration[i, :n] = torch.tensor(s["step_duration"], dtype=torch.float)
        step_depth[i, :n] = torch.tensor(s["step_depth"], dtype=torch.long)
        trace_attention_mask[i, :n] = 1.0
        fault_reason_ids.append(s["fault_reason_id"])
        root_causes.append(s["root_cause"])
        trace_ids.append(s["trace_id"])

    out: dict[str, Any] = {
        "log_token_ids": log_token_ids,
        "step_token_ids": step_token_ids,
        "step_token_mask": step_token_mask,
        "step_parent_idx": step_parent_idx,
        "step_duration": step_duration,
        "step_depth": step_depth,
        "fault_reason_id": torch.tensor(fault_reason_ids, dtype=torch.long),
        "root_cause": root_causes,
        "root_cause_token_ids": root_cause_token_ids,
        "root_cause_lengths": torch.tensor([len(rc) for rc in root_cause_token_ids_list], dtype=torch.long),
        "trace_id": trace_ids,
        "trace_attention_mask": trace_attention_mask,
    }
    if log_attention_mask is not None:
        out["log_attention_mask"] = log_attention_mask
    return out
