#!/usr/bin/env python3
"""
Tokenize datasets: load exp_datasets, encode each step with TPE tokenizer, save to data/tokenize_datasets.
Per-step records include: engine, fault_type, root_cause, fault_label, log (path + optional preview),
encoded trace (input_ids, parent_idx, duration, depth), and event counts. Saves overall summary and
per-trace step files. Uses multiprocessing for parallel trace processing.

Usage (from project root):
  python scripts/tokenize_datasets.py
  python scripts/tokenize_datasets.py --config configs/tokenize_datasets/default.yaml --num-workers 8
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_annotation(sample_dir: Path) -> dict:
    p = sample_dir / "annotation.json"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _load_trace_meta(trace_path: Path) -> dict:
    with open(trace_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "engine": data.get("engine"),
        "is_fault": data.get("is_fault"),
        "fault_type": data.get("fault_type"),
        "trace_id": data.get("trace_id"),
        "request_id": data.get("request_id"),
    }


def _log_path_and_preview(sample_dir: Path, project_root: Path, preview_chars: int = 500) -> tuple[str, str]:
    log_file = sample_dir / "log.txt"
    if log_file.is_file():
        try:
            log_path = str(log_file.relative_to(project_root))
        except ValueError:
            # Keep an absolute path when the published dataset is outside the code tree.
            log_path = str(log_file.resolve())
    else:
        log_path = ""
    preview = ""
    if log_file.is_file():
        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                preview = f.read(preview_chars)
            if len(preview) >= preview_chars:
                preview += "..."
        except Exception:
            preview = ""
    return log_path, preview


def _process_one_trace(
    args: tuple[str, str, str, str, dict[str, Any], int | None],
) -> tuple[str, int, int, str, str, int]:
    """
    Worker: process one trace, write steps to steps_subdir immediately, return only metadata.
    Returns (trace_id, total_before, total_after, engine, fault_type, num_steps).
    """
    trace_path_str, project_root_str, tokenizer_dir_str, steps_subdir_str, kernel_cfg, max_steps_per_trace = args
    from pathlib import Path

    from teller.trace.trace_pair_tokenizer import TracePairTokenizer
    from teller.trace.trace_parser import iter_steps_from_trace

    trace_path = Path(trace_path_str)
    project_root = Path(project_root_str)
    tokenizer_dir = Path(tokenizer_dir_str)
    if not tokenizer_dir.is_absolute():
        tokenizer_dir = project_root / tokenizer_dir
    steps_subdir = Path(steps_subdir_str)
    tokenizer = TracePairTokenizer.from_pretrained(tokenizer_dir)

    sample_dir = trace_path.parent
    engine = sample_dir.parent.name
    trace_id = f"{engine}/{sample_dir.name}"

    meta_trace = _load_trace_meta(trace_path)
    annotation = _load_annotation(sample_dir)
    log_path, log_preview = _log_path_and_preview(sample_dir, project_root)

    engine_type = meta_trace.get("engine") or engine
    fault_type = annotation.get("fault_type") or meta_trace.get("fault_type") or ""
    root_cause = annotation.get("root_cause") or ""
    fault_label = annotation.get("is_anomaly", meta_trace.get("is_fault", False))

    k_demangle = kernel_cfg.get("demangle", True)
    k_max_len = kernel_cfg.get("demangle_max_length", 128)
    k_filter = kernel_cfg.get("filter_template_args", True)

    total_before = 0
    total_after = 0
    steps_in_trace: list[dict] = []
    for step_idx, (scheme_b, parent_node, duration_node, depth_node) in enumerate(
        iter_steps_from_trace(
            trace_path,
            max_steps=max_steps_per_trace,
            kernel_demangle=k_demangle,
            kernel_max_length=k_max_len,
            kernel_filter_template=k_filter,
        )
    ):
        tokens = scheme_b.split()
        num_nodes = len(parent_node)
        if len(tokens) != 2 * num_nodes:
            continue
        n_before = num_nodes
        enc = tokenizer.encode_step(scheme_b, parent_node, duration_node, depth_node)
        n_after = len(enc["input_ids"])
        total_before += n_before
        total_after += n_after
        step_record = {
            "trace_id": trace_id,
            "step_idx": step_idx,
            "engine": engine_type,
            "fault_type": fault_type,
            "root_cause": root_cause,
            "fault_label": fault_label,
            "log_path": log_path,
            "log_preview": log_preview[:500] if log_preview else "",
            "n_events_before": n_before,
            "n_events_after": n_after,
            "input_ids": enc["input_ids"],
            "parent_idx": enc["parent_idx"],
            "duration": enc["duration"],
            "depth": enc["depth"],
        }
        steps_in_trace.append(step_record)
    # Write this trace's steps to disk immediately (streaming save)
    steps_subdir.mkdir(parents=True, exist_ok=True)
    safe_name = trace_id.replace("/", "_")
    with open(steps_subdir / f"{safe_name}.json", "w", encoding="utf-8") as f:
        json.dump(steps_in_trace, f, ensure_ascii=False, indent=2)
    return (trace_id, total_before, total_after, engine, fault_type, len(steps_in_trace))


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize exp_datasets with TPE tokenizer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "tokenize_datasets" / "default.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--tokenizer-dir", type=str, default=None, help="Override tokenizer dir from config")
    parser.add_argument("--data-dir", type=str, default=None, help="Override exp_datasets dir from config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output base dir from config")
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--max-traces-per-engine", type=int, default=None)
    parser.add_argument("--max-steps-per-trace", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers for trace processing (default: from config or os.cpu_count())",
    )
    args = parser.parse_args()

    from teller.trace.trace_pair_tokenizer import TracePairTokenizer
    from teller.trace.trace_parser import collect_trace_paths, iter_steps_from_trace
    from teller.trace.tpe_trainer import load_config

    config = load_config(args.config)
    tok_cfg = config.get("tokenizer", {})
    data_cfg = config.get("data", {})
    kernel_cfg = config.get("kernel", {})
    out_cfg = config.get("output", {})

    tokenizer_path = Path(args.tokenizer_dir or tok_cfg.get("dir", ""))
    if not tokenizer_path.is_absolute():
        tokenizer_path = PROJECT_ROOT / tokenizer_path
    if not tokenizer_path.is_dir():
        print(f"[tokenize_datasets] Tokenizer dir not found: {tokenizer_path}")
        return 1
    tokenizer = TracePairTokenizer.from_pretrained(tokenizer_path)
    run_name = out_cfg.get("run_name") or tokenizer_path.name

    exp_dir = Path(args.data_dir or data_cfg.get("exp_datasets_dir", "data/exp_datasets"))
    if not exp_dir.is_absolute():
        exp_dir = PROJECT_ROOT / exp_dir
    if not exp_dir.is_dir():
        print(f"[tokenize_datasets] exp_datasets dir not found: {exp_dir}")
        return 1

    out_base = Path(args.output_dir or out_cfg.get("dir", "data/tokenize_datasets"))
    if not out_base.is_absolute():
        out_base = PROJECT_ROOT / out_base
    out_dir = out_base / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_subdir = out_dir / "steps"
    steps_subdir.mkdir(exist_ok=True)

    max_traces = args.max_traces or data_cfg.get("max_traces")
    max_traces_per_engine = args.max_traces_per_engine or data_cfg.get("max_traces_per_engine")
    max_steps_per_trace = args.max_steps_per_trace or data_cfg.get("max_steps_per_trace")
    num_workers = args.num_workers if args.num_workers is not None else data_cfg.get("num_workers")
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    k_demangle = kernel_cfg.get("demangle", True)
    k_max_len = kernel_cfg.get("demangle_max_length", 128)
    k_filter = kernel_cfg.get("filter_template_args", True)

    paths = collect_trace_paths(
        exp_dir,
        max_traces=max_traces,
        max_traces_per_engine=max_traces_per_engine,
        engine_whitelist=data_cfg.get("engine_whitelist"),
    )
    if not paths:
        print("[tokenize_datasets] No trace paths found")
        return 1

    total_events_before = 0
    total_events_after = 0
    num_steps = 0
    steps_by_trace: list[dict] = []
    all_engines: set[str] = set()
    all_fault_types: set[str] = set()

    tokenizer_dir_for_worker = str(tokenizer_path)
    project_root_str = str(PROJECT_ROOT)
    kernel_cfg_copy = {"demangle": k_demangle, "demangle_max_length": k_max_len, "filter_template_args": k_filter}

    if num_workers <= 1:
        # Sequential
        for trace_path in tqdm(paths, desc="Traces", unit=" trace"):
            sample_dir = trace_path.parent
            engine = sample_dir.parent.name
            trace_id = f"{engine}/{sample_dir.name}"
            all_engines.add(engine)

            meta_trace = _load_trace_meta(trace_path)
            annotation = _load_annotation(sample_dir)
            log_path, log_preview = _log_path_and_preview(sample_dir, PROJECT_ROOT)

            engine_type = meta_trace.get("engine") or engine
            fault_type = annotation.get("fault_type") or meta_trace.get("fault_type") or ""
            root_cause = annotation.get("root_cause") or ""
            fault_label = annotation.get("is_anomaly", meta_trace.get("is_fault", False))
            if fault_type:
                all_fault_types.add(fault_type)

            steps_in_trace = []
            for step_idx, (scheme_b, parent_node, duration_node, depth_node) in enumerate(
                iter_steps_from_trace(
                    trace_path,
                    max_steps=max_steps_per_trace,
                    kernel_demangle=k_demangle,
                    kernel_max_length=k_max_len,
                    kernel_filter_template=k_filter,
                )
            ):
                tokens = scheme_b.split()
                num_nodes = len(parent_node)
                if len(tokens) != 2 * num_nodes:
                    continue
                n_before = num_nodes
                enc = tokenizer.encode_step(scheme_b, parent_node, duration_node, depth_node)
                n_after = len(enc["input_ids"])
                total_events_before += n_before
                total_events_after += n_after
                num_steps += 1
                step_record = {
                    "trace_id": trace_id,
                    "step_idx": step_idx,
                    "engine": engine_type,
                    "fault_type": fault_type,
                    "root_cause": root_cause,
                    "fault_label": fault_label,
                    "log_path": log_path,
                    "log_preview": log_preview[:500] if log_preview else "",
                    "n_events_before": n_before,
                    "n_events_after": n_after,
                    "input_ids": enc["input_ids"],
                    "parent_idx": enc["parent_idx"],
                    "duration": enc["duration"],
                    "depth": enc["depth"],
                }
                steps_in_trace.append(step_record)
            steps_by_trace.append({"trace_id": trace_id, "steps": steps_in_trace})
            safe_name = trace_id.replace("/", "_")
            with open(steps_subdir / f"{safe_name}.json", "w", encoding="utf-8") as f:
                json.dump(steps_in_trace, f, ensure_ascii=False, indent=2)
    else:
        # Parallel: each worker writes its trace's step file immediately and returns only metadata
        arg_list = [
            (
                str(p),
                project_root_str,
                tokenizer_dir_for_worker,
                str(steps_subdir),
                kernel_cfg_copy,
                max_steps_per_trace,
            )
            for p in paths
        ]
        n_workers = min(num_workers, len(paths))
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for result in tqdm(
                ex.map(_process_one_trace, arg_list),
                total=len(arg_list),
                desc="Traces",
                unit=" trace",
            ):
                trace_id, total_before, total_after, engine, fault_type, n_steps = result
                total_events_before += total_before
                total_events_after += total_after
                num_steps += n_steps
                all_engines.add(engine)
                if fault_type:
                    all_fault_types.add(fault_type)

    compression = total_events_before / total_events_after if total_events_after else 0.0
    summary = {
        "tokenizer_dir": str(tokenizer_path),
        "data_dir": str(exp_dir),
        "output_dir": str(out_dir),
        "num_traces": len(paths),
        "num_steps": num_steps,
        "total_events_before": total_events_before,
        "total_events_after": total_events_after,
        "compression_ratio": round(compression, 4),
        "avg_events_per_step_before": round(total_events_before / num_steps, 2) if num_steps else 0,
        "avg_events_per_step_after": round(total_events_after / num_steps, 2) if num_steps else 0,
        "engines": sorted(all_engines),
        "fault_types": sorted(all_fault_types),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if steps_by_trace:
        with open(out_dir / "steps_by_trace.json", "w", encoding="utf-8") as f:
            json.dump(steps_by_trace, f, ensure_ascii=False, indent=2)

    print("[tokenize_datasets] Summary:")
    print(f"  Traces: {summary['num_traces']}, Steps: {summary['num_steps']}")
    print(f"  Events before: {summary['total_events_before']}, after: {summary['total_events_after']}")
    print(f"  Compression ratio: {summary['compression_ratio']}")
    print(f"  Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
