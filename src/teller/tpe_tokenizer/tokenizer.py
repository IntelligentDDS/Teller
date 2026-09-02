"""
TPETokenizer: Hugging Face–compatible Trace Pair Encoding tokenizer.
Encodes trace step strings to token ids with parent_idx, duration, depth.
Uses vocab + merges from TPE training; supports encode_step, encode_step_batch, decode, save/load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import AddedToken
from transformers.utils import PaddingStrategy, TensorType


# Token strings may contain spaces; decode replaces this with actual space.
SPACE_PLACEHOLDER = "\u2423"  # ␣ (word joiner)


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


class TPETokenizer(PreTrainedTokenizerBase):
    """
    Tokenizer for trace step strings. Outputs token ids, call relations (parent_idx),
    duration, depth, and mask. Compatible with Hugging Face (save_pretrained, from_pretrained,
    decode, __call__). Padding uses pad_token_id=0; parent pad = -1.
    """

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.json"}

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        vocab_file: str | Path | None = None,
        merges_file: str | Path | None = None,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        added_tokens_decoder: dict[int, AddedToken] | None = None,
        **kwargs: Any,
    ):
        if vocab is None and vocab_file is not None:
            with open(vocab_file, encoding="utf-8") as f:
                vocab = json.load(f)
        if merges is None and merges_file is not None:
            with open(merges_file, encoding="utf-8") as f:
                merges = [tuple(p) for p in json.load(f)]
        vocab = vocab or {"[PAD]": 0, "[UNK]": 1}
        merges = merges or []
        self._vocab = dict(vocab)
        self._merges = list(merges)
        self._unk_token_id = unk_token_id
        self._pad_token_id = pad_token_id
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self._added_tokens_decoder = dict(added_tokens_decoder) if added_tokens_decoder else {}

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            **kwargs,
        )

    @property
    def added_tokens_decoder(self) -> dict[int, AddedToken]:
        return self._added_tokens_decoder

    @property
    def added_tokens_encoder(self) -> dict[str, int]:
        """Token string -> id for added tokens (beyond main vocab). We use main vocab only."""
        return {}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        """Space-split tokenization (no BPE merge). For full step encoding use encode_step."""
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, "[UNK]")

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
            for (a, b) in self._merges:
                new_token = f"{a} {b}"
                if new_token not in self._vocab:
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
        Encode one trace step. Structure (parent_idx_node, duration_node, depth_node)
        must have length = num_nodes (one per node; 2 tokens per node in step_string).
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
        input_ids = [self._vocab.get(t, self._unk_token_id) for t in tokens]
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
            input_ids_b.append(e["input_ids"] + [self._pad_token_id] * pad_len)
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

    def decode(
        self,
        token_ids: list[int] | list[list[int]],
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        """Decode token ids to string. Replaces SPACE_PLACEHOLDER with space."""
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        pad_id = self._pad_token_id
        tokens = []
        for i in token_ids:
            if i == pad_id and skip_special_tokens:
                continue
            tokens.append(self._id_to_token.get(i, "[UNK]"))
        return " ".join(tokens).replace(SPACE_PLACEHOLDER, " ")

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[str, ...]:
        out = Path(save_directory)
        out.mkdir(parents=True, exist_ok=True)
        prefix = filename_prefix or ""
        vocab_path = out / f"{prefix}vocab.json"
        merges_path = out / f"{prefix}merges.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(self._merges, f, ensure_ascii=False)
        return (str(vocab_path), str(merges_path))

    def save_pretrained(
        self,
        save_directory: str | Path,
        **kwargs: Any,
    ) -> None:
        super().save_pretrained(save_directory, **kwargs)
        out = Path(save_directory)
        config = {
            "pad_token_id": self._pad_token_id,
            "unk_token_id": self._unk_token_id,
        }
        with open(out / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
