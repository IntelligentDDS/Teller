"""
Parse trace.json into per-step sequences (scheme B string + parent_idx, duration, depth)
for TPE vocab training and tokenizer input. Supports C++ kernel name demangle.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Iterator

# Placeholder for space inside token names so scheme_b.split() yields exactly 2*num_nodes
SPACE_PLACEHOLDER = "\u2423"  # ␣ (word joiner); unescape in tokenizer decode

# Type tags for scheme B (must match config special_tokens)
TAG_STEP = "[STEP]"
TAG_FE = "[FE]"
TAG_BE = "[BE]"
TAG_RT = "[RT]"
TAG_K = "[K]"
TAG_DRIVER = "[DRIVER]"


def _demangle_kernel(name: str, max_length: int = 128, filter_template: bool = True) -> str:
    """Demangle C++ kernel name via c++filt; optionally truncate and strip template args."""
    if not name or (name.startswith("?") or "(" in name):
        return name
    try:
        out = subprocess.run(
            ["c++filt", "-n", name],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            s = out.stdout.strip()
        else:
            s = name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        s = name
    if filter_template:
        idx = s.find("<")
        if idx != -1:
            s = s[:idx]
    if len(s) > max_length:
        s = s[:max_length]
    return s


def _node_duration(node: dict[str, Any]) -> int:
    """Duration in ns from start/end or gpu_start + duration."""
    if "start" in node and "end" in node:
        return node["end"] - node["start"]
    if "gpu_start" in node and "duration" in node:
        return node["duration"]
    return 0


def _dfs_step(
    step: dict[str, Any],
    step_index: int,
    parent_idx: int,
    depth: int,
    nodes: list[tuple[str, str, int, int]],
    kernel_demangle: bool = True,
    kernel_max_length: int = 128,
    kernel_filter_template: bool = True,
) -> None:
    """DFS one step; append (type_tag, name, duration_ns, depth) to nodes. parent_idx is the index in nodes of the parent (before this node is appended)."""
    step_name = step.get("step_name") or "step"
    step_dur = _node_duration(step)
    nodes.append((TAG_STEP, step_name, step_dur, depth))
    my_idx = len(nodes) - 1

    for fe in step.get("torch_frontend_ops") or []:
        fe_name = fe.get("op_name") or "unknown"
        fe_dur = _node_duration(fe)
        nodes.append((TAG_FE, fe_name, fe_dur, depth + 1))
        fe_idx = len(nodes) - 1

        for be in fe.get("torch_backend_ops") or []:
            be_name = be.get("op_name") or "unknown"
            be_dur = _node_duration(be)
            nodes.append((TAG_BE, be_name, be_dur, depth + 2))
            be_idx = len(nodes) - 1

            for rt in be.get("runtime_calls") or []:
                rt_name = rt.get("name") or "unknown"
                rt_dur = _node_duration(rt)
                nodes.append((TAG_RT, rt_name, rt_dur, depth + 3))
                rt_idx = len(nodes) - 1
                for k in rt.get("kernels") or []:
                    k_name = k.get("name") or "unknown"
                    if kernel_demangle:
                        k_name = _demangle_kernel(
                            k_name,
                            max_length=kernel_max_length,
                            filter_template=kernel_filter_template,
                        )
                    k_dur = _node_duration(k)
                    nodes.append((TAG_K, k_name, k_dur, depth + 4))

            for dr in be.get("driver_calls") or []:
                dr_name = dr.get("name") or "unknown"
                dr_dur = _node_duration(dr)
                nodes.append((TAG_DRIVER, dr_name, dr_dur, depth + 3))
                for k in dr.get("kernels") or []:
                    k_name = k.get("name") or "unknown"
                    if kernel_demangle:
                        k_name = _demangle_kernel(
                            k_name,
                            max_length=kernel_max_length,
                            filter_template=kernel_filter_template,
                        )
                    k_dur = _node_duration(k)
                    nodes.append((TAG_K, k_name, k_dur, depth + 4))


def _nodes_to_scheme_b_and_structure(
    nodes: list[tuple[str, str, int, int]],
) -> tuple[list[str], list[int], list[int], list[int]]:
    """
    Convert list of (type_tag, name, duration_ns, depth) to scheme B token sequence
    and parallel arrays parent_idx, duration, depth (one per node).
    parent_idx[i] = node index of parent (-1 for root). DFS order => parent is last j<i with depth[j]<depth[i].
    """
    depth_arr = [d for (_, _, _, d) in nodes]
    duration = [dur for (_, _, dur, _) in nodes]
    parent_idx: list[int] = []
    for i in range(len(nodes)):
        p = -1
        for j in range(i - 1, -1, -1):
            if depth_arr[j] < depth_arr[i]:
                p = j
                break
        parent_idx.append(p)
    tokens: list[str] = []
    for tag, name, _, _ in nodes:
        tokens.append(tag)
        tokens.append(name)
    return tokens, parent_idx, duration, depth_arr


def parse_step(
    step: dict[str, Any],
    step_index: int = 0,
    kernel_demangle: bool = True,
    kernel_max_length: int = 128,
    kernel_filter_template: bool = True,
) -> tuple[str, list[int], list[int], list[int]]:
    """
    Parse one step dict from trace.json.
    Returns (scheme_b_string, parent_idx, duration_ns, depth).
    - scheme_b_string: space-separated "[STEP] step_name [FE] op_name ..."
    - parent_idx, duration_ns, depth: length = num_nodes (one per node, not per token).
    """
    nodes: list[tuple[str, str, int, int]] = []
    _dfs_step(
        step,
        step_index,
        parent_idx=-1,
        depth=0,
        nodes=nodes,
        kernel_demangle=kernel_demangle,
        kernel_max_length=kernel_max_length,
        kernel_filter_template=kernel_filter_template,
    )
    tokens, parent_idx, duration_ns, depth = _nodes_to_scheme_b_and_structure(nodes)
    # Escape spaces in names so " ".join(tokens).split() recovers exactly 2*num_nodes
    tokens_esc = [t.replace(" ", SPACE_PLACEHOLDER) for t in tokens]
    scheme_b = " ".join(tokens_esc)
    return scheme_b, parent_idx, duration_ns, depth


def iter_steps_from_trace(
    trace_path: str | Path,
    max_steps: int | None = None,
    kernel_demangle: bool = True,
    kernel_max_length: int = 128,
    kernel_filter_template: bool = True,
) -> Iterator[tuple[str, list[int], list[int], list[int]]]:
    """Yield (scheme_b_string, parent_idx, duration_ns, depth) for each step in trace.json."""
    import json

    path = Path(trace_path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps") or []
    if max_steps is not None:
        steps = steps[:max_steps]
    for i, step in enumerate(steps):
        yield parse_step(
            step,
            step_index=i,
            kernel_demangle=kernel_demangle,
            kernel_max_length=kernel_max_length,
            kernel_filter_template=kernel_filter_template,
        )


def load_one_trace_steps(
    trace_path: str | Path,
    max_steps: int | None = None,
    kernel_demangle: bool = True,
    kernel_max_length: int = 128,
    kernel_filter_template: bool = True,
) -> list[tuple[str, list[int], list[int], list[int]]]:
    """Load one trace.json and return all steps (scheme_b, parent_idx, duration, depth). For parallel use."""
    return list(
        iter_steps_from_trace(
            trace_path,
            max_steps=max_steps,
            kernel_demangle=kernel_demangle,
            kernel_max_length=kernel_max_length,
            kernel_filter_template=kernel_filter_template,
        )
    )


def collect_trace_paths(
    exp_datasets_dir: str | Path,
    max_traces: int | None = None,
    max_traces_per_engine: int | None = None,
    engine_whitelist: list[str] | None = None,
) -> list[Path]:
    """Collect trace.json paths under exp_datasets_dir for parallel loading.
    If max_traces_per_engine is set, take at most that many traces per engine, then
    optionally cap total by max_traces."""
    root = Path(exp_datasets_dir)
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for engine_dir in sorted(root.iterdir()):
        if not engine_dir.is_dir():
            continue
        if engine_whitelist is not None and engine_dir.name not in engine_whitelist:
            continue
        engine_count = 0
        for sample_dir in sorted(engine_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            trace_file = sample_dir / "trace.json"
            if not trace_file.is_file():
                continue
            if max_traces_per_engine is not None and engine_count >= max_traces_per_engine:
                break
            if max_traces is not None and len(paths) >= max_traces:
                return paths
            paths.append(trace_file)
            engine_count += 1
    return paths


def iter_all_steps_from_exp_datasets(
    exp_datasets_dir: str | Path,
    max_traces: int | None = None,
    max_traces_per_engine: int | None = None,
    max_steps_per_trace: int | None = None,
    engine_whitelist: list[str] | None = None,
    kernel_demangle: bool = True,
    kernel_max_length: int = 128,
    kernel_filter_template: bool = True,
) -> Iterator[tuple[str, list[int], list[int], list[int]]]:
    """Scan exp_datasets_dir for trace.json and yield (scheme_b, parent_idx, duration, depth) per step.
    If max_traces_per_engine is set, take at most that many traces per engine."""
    root = Path(exp_datasets_dir)
    if not root.is_dir():
        return
    total_count = 0
    for engine_dir in sorted(root.iterdir()):
        if not engine_dir.is_dir():
            continue
        if engine_whitelist is not None and engine_dir.name not in engine_whitelist:
            continue
        engine_count = 0
        for sample_dir in sorted(engine_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            trace_file = sample_dir / "trace.json"
            if not trace_file.is_file():
                continue
            if max_traces_per_engine is not None and engine_count >= max_traces_per_engine:
                break
            if max_traces is not None and total_count >= max_traces:
                return
            total_count += 1
            engine_count += 1
            for out in iter_steps_from_trace(
                trace_file,
                max_steps=max_steps_per_trace,
                kernel_demangle=kernel_demangle,
                kernel_max_length=kernel_max_length,
                kernel_filter_template=kernel_filter_template,
            ):
                yield out
