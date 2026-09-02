"""
Trace Pair Encode (TPE) trainer: build vocab from special tokens + names,
then merge adjacent pairs (with duration/depth/parent rules) until target vocab size.

Why not use an existing BPE library (e.g. HuggingFace tokenizers)?
- TPE is **word-level** BPE: the atoms are whole tokens like "[STEP]", "aten::copy_",
  and we merge adjacent such tokens. We also maintain parallel structure (parent_idx,
  duration, depth) when merging.
- Libraries like tokenizers/sentencepiece do **character/subword-level** BPE: they
  split text into characters (or bytes) and merge within that space. Even with a
  custom pre-tokenizer that splits on a delimiter, the trained model still
  tokenizes each "word" into subwords (characters then merged), so we do not get
  "merge two whole words" semantics.
- So we keep this custom trainer. The output (vocab + list of merges) is compatible
  with our TracePairTokenizer, which applies merges while updating parent/duration/depth.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from teller.trace.trace_parser import (
    collect_trace_paths,
    iter_all_steps_from_exp_datasets,
    load_one_trace_steps,
)


def _expand_step_to_per_token(
    tokens: list[str],
    parent_idx_node: list[int],
    duration_node: list[int],
    depth_node: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Expand per-node structure to per-token. len(tokens) == 2 * len(parent_idx_node)."""
    n = len(parent_idx_node)
    parent_idx = []
    duration = []
    depth = []
    for i in range(n):
        p = parent_idx_node[i]
        if p >= 0:
            token_parent = 2 * p
        else:
            token_parent = -1
        parent_idx.append(token_parent)
        parent_idx.append(token_parent)
        duration.append(0)
        duration.append(duration_node[i])
        depth.append(depth_node[i])
        depth.append(depth_node[i])
    return parent_idx, duration, depth


def _load_one_trace_steps_args(
    args: tuple[str, int | None, bool, int, bool],
) -> list[tuple[str, list[int], list[int], list[int]]]:
    """Unpack args and call load_one_trace_steps (for ProcessPoolExecutor)."""
    path, max_steps, k_demangle, k_max_len, k_filter = args
    return load_one_trace_steps(
        path,
        max_steps=max_steps,
        kernel_demangle=k_demangle,
        kernel_max_length=k_max_len,
        kernel_filter_template=k_filter,
    )


def load_corpus(
    exp_datasets_dir: str | Path,
    config: dict[str, Any],
) -> list[tuple[list[str], list[int], list[int], list[int]]]:
    """
    Load corpus: list of (tokens, parent_idx, duration, depth) per step.
    All four lists have the same length (per-token).
    Uses parallel workers when data.num_workers > 1.
    """
    kernel_cfg = config.get("kernel", {})
    data_cfg = config.get("data", {})
    num_workers = data_cfg.get("num_workers")
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    max_traces_per_engine = data_cfg.get("max_traces_per_engine")
    if num_workers <= 1:
        steps_iter = iter_all_steps_from_exp_datasets(
            exp_datasets_dir,
            max_traces=data_cfg.get("max_traces"),
            max_traces_per_engine=max_traces_per_engine,
            max_steps_per_trace=data_cfg.get("max_steps_per_trace"),
            engine_whitelist=data_cfg.get("engine_whitelist"),
            kernel_demangle=kernel_cfg.get("demangle", True),
            kernel_max_length=kernel_cfg.get("demangle_max_length", 128),
            kernel_filter_template=kernel_cfg.get("filter_template_args", True),
        )
        corpus = []
        for scheme_b, parent_node, duration_node, depth_node in tqdm(
            steps_iter, desc="Load corpus", unit=" step"
        ):
            tokens = scheme_b.split()
            if len(tokens) != 2 * len(parent_node):
                continue
            parent_idx, duration, depth = _expand_step_to_per_token(
                tokens, parent_node, duration_node, depth_node
            )
            corpus.append((tokens, parent_idx, duration, depth))
        return corpus

    paths = collect_trace_paths(
        exp_datasets_dir,
        max_traces=data_cfg.get("max_traces"),
        max_traces_per_engine=max_traces_per_engine,
        engine_whitelist=data_cfg.get("engine_whitelist"),
    )
    if not paths:
        return []
    max_steps = data_cfg.get("max_steps_per_trace")
    k_d = kernel_cfg.get("demangle", True)
    k_ml = kernel_cfg.get("demangle_max_length", 128)
    k_ft = kernel_cfg.get("filter_template_args", True)
    arg_list = [
        (str(p), max_steps, k_d, k_ml, k_ft)
        for p in paths
    ]
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        step_lists = list(
            tqdm(
                ex.map(_load_one_trace_steps_args, arg_list),
                total=len(arg_list),
                desc="Load traces",
                unit=" trace",
            )
        )
    corpus = []
    for step_list in step_lists:
        for scheme_b, parent_node, duration_node, depth_node in step_list:
            tokens = scheme_b.split()
            if len(tokens) != 2 * len(parent_node):
                continue
            parent_idx, duration, depth = _expand_step_to_per_token(
                tokens, parent_node, duration_node, depth_node
            )
            corpus.append((tokens, parent_idx, duration, depth))
    return corpus


def build_initial_vocab(
    corpus: list[tuple[list[str], list[int], list[int], list[int]]],
    special_tokens: list[str],
) -> dict[str, int]:
    """Build vocab: special tokens first, then all unique token strings from corpus."""
    vocab: dict[str, int] = {}
    for i, t in enumerate(special_tokens):
        vocab[t] = i
    next_id = len(vocab)
    for tokens, _, _, _ in corpus:
        for t in tokens:
            if t not in vocab:
                vocab[t] = next_id
                next_id += 1
    return vocab


# Int-corpus type: (token_ids, parent_idx, duration, depth) per step
IntCorpus = list[tuple[list[int], list[int], list[int], list[int]]]


def _corpus_to_int(
    corpus: list[tuple[list[str], list[int], list[int], list[int]]],
    special_tokens: list[str],
) -> tuple[IntCorpus, list[str]]:
    """
    Replace every event name with a numeric id. Returns (corpus_int, id_to_name).
    id_to_name[i] is the string for id i; special tokens first, then unique names in sorted order.
    """
    name_to_id: dict[str, int] = {}
    for i, t in enumerate(special_tokens):
        name_to_id[t] = i
    other: set[str] = set()
    for tokens, _, _, _ in corpus:
        for t in tokens:
            if t not in name_to_id:
                other.add(t)
    id_to_name = list(special_tokens) + sorted(other)
    for i, s in enumerate(id_to_name):
        name_to_id[s] = i
    corpus_int: IntCorpus = []
    for tokens, parent_idx, duration, depth in corpus:
        ids = [name_to_id[t] for t in tokens]
        corpus_int.append((ids, parent_idx, duration, depth))
    return corpus_int, id_to_name


def count_pairs_int(
    corpus_int: IntCorpus,
    *,
    num_workers: int | None = None,
) -> Counter[tuple[int, int]]:
    """Count adjacent (id_i, id_i+1) pairs. Optional parallel by chunk."""
    if not corpus_int:
        return Counter()

    if num_workers is None or num_workers <= 1:
        counter: Counter[tuple[int, int]] = Counter()
        for ids, _, _, _ in corpus_int:
            for i in range(len(ids) - 1):
                counter[(ids[i], ids[i + 1])] += 1
        return counter

    n = min(num_workers, len(corpus_int))
    chunk_size = (len(corpus_int) + n - 1) // n
    chunks = [
        corpus_int[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n)
    ]
    with ProcessPoolExecutor(max_workers=n) as ex:
        counters = list(ex.map(_count_pairs_int_chunk, chunks))
    total: Counter[tuple[int, int]] = Counter()
    for c in counters:
        total.update(c)
    return total


def _count_pairs_int_chunk(chunk: IntCorpus) -> Counter[tuple[int, int]]:
    counter: Counter[tuple[int, int]] = Counter()
    for ids, _, _, _ in chunk:
        for i in range(len(ids) - 1):
            counter[(ids[i], ids[i + 1])] += 1
    return counter


def apply_merge_int(
    corpus_int: IntCorpus,
    pair: tuple[int, int],
    new_id: int,
) -> None:
    """Replace adjacent pair (a_id, b_id) with new_id; update structure. In-place."""
    a_id, b_id = pair
    for (ids, parent_idx, duration, depth) in corpus_int:
        i = 0
        while i < len(ids) - 1:
            if ids[i] == a_id and ids[i + 1] == b_id:
                new_parent = parent_idx[i]
                new_duration = duration[i] + duration[i + 1]
                new_depth = depth[i]
                ids[i] = new_id
                del ids[i + 1]
                parent_idx[i] = new_parent
                duration[i] = new_duration
                depth[i] = new_depth
                del parent_idx[i + 1]
                del duration[i + 1]
                del depth[i + 1]
                for j in range(len(parent_idx)):
                    if parent_idx[j] == i + 1:
                        parent_idx[j] = i
                    elif parent_idx[j] > i + 1:
                        parent_idx[j] -= 1
                continue
            i += 1


def count_pairs(
    corpus: list[tuple[list[str], list[int], list[int], list[int]]],
    *,
    num_workers: int | None = None,
) -> Counter[tuple[str, str]]:
    """Count adjacent (token_i, token_i+1) pairs in corpus. Optional parallel by chunk."""
    if not corpus:
        return Counter()

    if num_workers is None or num_workers <= 1:
        counter: Counter[tuple[str, str]] = Counter()
        for tokens, _, _, _ in corpus:
            for i in range(len(tokens) - 1):
                counter[(tokens[i], tokens[i + 1])] += 1
        return counter

    n = min(num_workers, len(corpus))
    chunk_size = (len(corpus) + n - 1) // n
    chunks = [
        corpus[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n)
    ]
    with ProcessPoolExecutor(max_workers=n) as ex:
        counters = list(ex.map(_count_pairs_chunk, chunks))
    total: Counter[tuple[str, str]] = Counter()
    for c in counters:
        total.update(c)
    return total


def _count_pairs_chunk(
    chunk: list[tuple[list[str], list[int], list[int], list[int]]],
) -> Counter[tuple[str, str]]:
    """Count pairs in a corpus chunk (for ProcessPoolExecutor)."""
    counter: Counter[tuple[str, str]] = Counter()
    for tokens, _, _, _ in chunk:
        for i in range(len(tokens) - 1):
            counter[(tokens[i], tokens[i + 1])] += 1
    return counter


def apply_merge(
    corpus: list[tuple[list[str], list[int], list[int], list[int]]],
    pair: tuple[str, str],
    new_token: str,
) -> None:
    """
    Replace all adjacent occurrences of pair with new_token and update structure.
    Merge (A, B) -> C: C.parent = A.parent, C.duration = A.duration + B.duration, C.depth = A.depth.
    Reindex parent_idx: pointers to i+1 -> i; pointers > i+1 -> decrement by 1.
    """
    a, b = pair
    for (tokens, parent_idx, duration, depth) in corpus:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == a and tokens[i + 1] == b:
                new_parent = parent_idx[i]
                new_duration = duration[i] + duration[i + 1]
                new_depth = depth[i]
                tokens[i] = new_token
                del tokens[i + 1]
                parent_idx[i] = new_parent
                duration[i] = new_duration
                depth[i] = new_depth
                del parent_idx[i + 1]
                del duration[i + 1]
                del depth[i + 1]
                for j in range(len(parent_idx)):
                    if parent_idx[j] == i + 1:
                        parent_idx[j] = i
                    elif parent_idx[j] > i + 1:
                        parent_idx[j] -= 1
                continue
            i += 1


def train_tpe(
    config: dict[str, Any],
    exp_datasets_dir: str | Path,
) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """
    Run TPE training. Returns (vocab, merges).
    - vocab: token string -> id
    - merges: list of (token_a, token_b) in merge order (new token = a + " " + b for decoding)
    """
    special_tokens = config.get("special_tokens", ["[PAD]", "[UNK]", "[STEP]", "[FE]", "[BE]", "[RT]", "[K]"])
    merge_cfg = config.get("merge", {})
    target_vocab_size = merge_cfg.get("target_vocab_size", 8192)
    max_merge_rounds = merge_cfg.get("max_merge_rounds")
    min_pair_count = merge_cfg.get("min_pair_count", 2)

    corpus = load_corpus(exp_datasets_dir, config)
    if not corpus:
        raise ValueError(
            "TPE corpus is empty: no steps were loaded or all steps were filtered out. "
            "Check that trace.json files exist under exp_datasets_dir and that step "
            "sequences have exactly 2*num_nodes tokens (names with spaces are now escaped)."
        )
    # Replace event names with numeric ids for faster pair counting and merging
    corpus_int, id_to_name = _corpus_to_int(corpus, special_tokens)
    del corpus  # free memory
    num_workers = config.get("data", {}).get("num_workers")
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    merges_int: list[tuple[int, int]] = []
    round_ = 0
    current_vocab_size = len(id_to_name)
    remaining = target_vocab_size - current_vocab_size
    if max_merge_rounds is not None:
        remaining = min(remaining, max_merge_rounds)
    pbar = tqdm(
        total=remaining,
        initial=0,
        desc="Merge",
        unit=" merge",
        dynamic_ncols=True,
    )
    try:
        while current_vocab_size < target_vocab_size:
            if max_merge_rounds is not None and round_ >= max_merge_rounds:
                break
            counter = count_pairs_int(corpus_int, num_workers=num_workers)
            best_pair = None
            best_count = 0
            for (a_id, b_id), count in counter.most_common():
                if count < min_pair_count:
                    break
                new_name = f"{id_to_name[a_id]} {id_to_name[b_id]}"
                if new_name in id_to_name:
                    continue
                if count > best_count:
                    best_count = count
                    best_pair = (a_id, b_id)
            if best_pair is None:
                break
            a_id, b_id = best_pair
            new_id = len(id_to_name)
            id_to_name.append(f"{id_to_name[a_id]} {id_to_name[b_id]}")
            merges_int.append((a_id, b_id))
            apply_merge_int(corpus_int, (a_id, b_id), new_id)
            current_vocab_size = len(id_to_name)
            round_ += 1
            pbar.update(1)
            pbar.set_postfix(vocab=current_vocab_size, round=round_)
    finally:
        pbar.close()
    vocab = {id_to_name[i]: i for i in range(len(id_to_name))}
    merges = [(id_to_name[a], id_to_name[b]) for a, b in merges_int]
    return vocab, merges


def save_tpe(
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
    output_dir: str | Path,
) -> None:
    """Save vocab.json and merges.json to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(out / "merges.json", "w", encoding="utf-8") as f:
        json.dump(merges, f, ensure_ascii=False)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config."""
    import yaml
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
