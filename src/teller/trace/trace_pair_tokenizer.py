"""
TracePairTokenizer: encode trace step strings to token ids with call relations and mask.
Uses vocab + merges from TPE training; supports encode_step, encode_step_batch, save/load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SPECIAL = ["[PAD]", "[UNK]", "[STEP]", "[FE]", "[BE]", "[RT]", "[K]", "[DRIVER]"]


def _expand_node_to_token_structure(
    num_nodes: int,
    parent_idx_node: list[int],
    duration_node: list[int],
    depth_node: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Per-node -> per-token (2 tokens per node). parent_idx in token indices."""
    parent_idx: list[int] = []
    duration: list[int] = []
    depth: list[int] = []
    for i in range(num_nodes):
        p = 2 * parent_idx_node[i] if parent_idx_node[i] >= 0 else -1
        parent_idx.append(p)
        parent_idx.append(p)
        duration.append(0)
        duration.append(duration_node[i])
        depth.append(depth_node[i])
        depth.append(depth_node[i])
    return parent_idx, duration, depth


class TracePairTokenizer:
    """
    Tokenizer for trace step strings (scheme B). Outputs token ids, call relations (parent_idx),
    duration, depth, and mask. Compatible with batch step input (padding with [PAD] id 0).
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        pad_token_id: int = 0,
        unk_token_id: int = 1,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self._id_to_token = {v: k for k, v in self.vocab.items()}

    def _tokenize_with_merges(
        self,
        tokens: list[str],
        parent_idx: list[int],
        duration: list[int],
        depth: list[int],
    ) -> None:
        """Apply merges greedily until no merge applies."""
        changed = True
        while changed:
            changed = False
            for (a, b) in self.merges:
                new_token = f"{a} {b}"
                if new_token not in self.vocab:
                    continue
                for i in range(len(tokens) - 1):
                    if tokens[i] == a and tokens[i + 1] == b:
                        duration[i] = duration[i] + duration[i + 1]
                        depth[i] = depth[i]
                        tokens[i] = new_token
                        del tokens[i + 1]
                        del parent_idx[i + 1]
                        del duration[i + 1]
                        del depth[i + 1]
                        for j in range(len(parent_idx)):
                            if parent_idx[j] == i + 1:
                                parent_idx[j] = i
                            elif parent_idx[j] > i + 1:
                                parent_idx[j] -= 1
                        changed = True
                        break
                if changed:
                    break

    def encode_step(
        self,
        step_string: str,
        parent_idx_node: list[int] | None = None,
        duration_node: list[int] | None = None,
        depth_node: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Encode one step. Structure (parent_idx_node, duration_node, depth_node) must have
        length = num_nodes (one per node; 2 tokens per node in step_string).
        """
        tokens = step_string.split()
        num_nodes = len(tokens) // 2
        if parent_idx_node is None:
            parent_idx_node = [-1] * num_nodes
        if duration_node is None:
            duration_node = [0] * num_nodes
        if depth_node is None:
            depth_node = [0] * num_nodes
        if len(parent_idx_node) != num_nodes:
            raise ValueError("Structure length must equal num_nodes")
        parent_idx, duration, depth = _expand_node_to_token_structure(
            num_nodes, parent_idx_node, duration_node, depth_node
        )
        self._tokenize_with_merges(tokens, parent_idx, duration, depth)
        input_ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "parent_idx": parent_idx,
            "duration": duration,
            "depth": depth,
        }

    def encode_step_batch(
        self,
        step_strings: list[str],
        parent_idx_list: list[list[int]] | None = None,
        duration_list: list[list[int]] | None = None,
        depth_list: list[list[int]] | None = None,
        padding: bool = True,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        """Encode batch of steps. Pad with pad_token_id=0; parent_idx pad = -1."""
        encoded = []
        for i, s in enumerate(step_strings):
            p = parent_idx_list[i] if parent_idx_list is not None else None
            d = duration_list[i] if duration_list is not None else None
            dep = depth_list[i] if depth_list is not None else None
            encoded.append(self.encode_step(s, p, d, dep))
        max_len = max(len(e["input_ids"]) for e in encoded)
        if max_length is not None:
            max_len = min(max_len, max_length)
        input_ids_b = []
        attention_mask_b = []
        parent_idx_b = []
        duration_b = []
        depth_b = []
        for e in encoded:
            L = len(e["input_ids"])
            pad_len = max(0, max_len - L)
            input_ids_b.append(e["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask_b.append([1] * L + [0] * pad_len)
            parent_idx_b.append(e["parent_idx"] + [-1] * pad_len)
            duration_b.append(e["duration"] + [0] * pad_len)
            depth_b.append(e["depth"] + [0] * pad_len)
        return {
            "input_ids": input_ids_b,
            "attention_mask": attention_mask_b,
            "parent_idx": parent_idx_b,
            "duration": duration_b,
            "depth": depth_b,
        }

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        from teller.trace.trace_parser import SPACE_PLACEHOLDER

        tokens = []
        for i in ids:
            if i == self.pad_token_id and skip_special_tokens:
                continue
            tokens.append(self._id_to_token.get(i, "[UNK]"))
        return " ".join(tokens).replace(SPACE_PLACEHOLDER, " ")

    def save_pretrained(self, save_directory: str | Path) -> None:
        out = Path(save_directory)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(out / "merges.json", "w", encoding="utf-8") as f:
            json.dump(self.merges, f, ensure_ascii=False)
        config = {"pad_token_id": self.pad_token_id, "unk_token_id": self.unk_token_id}
        with open(out / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str | Path) -> TracePairTokenizer:
        out = Path(save_directory)
        with open(out / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(out / "merges.json", encoding="utf-8") as f:
            merges = [tuple(p) for p in json.load(f)]
        pad_token_id = 0
        unk_token_id = 1
        if (out / "tokenizer_config.json").exists():
            with open(out / "tokenizer_config.json", encoding="utf-8") as f:
                cfg = json.load(f)
                pad_token_id = cfg.get("pad_token_id", 0)
                unk_token_id = cfg.get("unk_token_id", 1)
        return cls(vocab=vocab, merges=merges, pad_token_id=pad_token_id, unk_token_id=unk_token_id)
