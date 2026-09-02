#!/usr/bin/env python3
"""Run GMM candidate localization on packaged TELLER traces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_paths(data_dir: Path, limit: int | None) -> list[Path]:
    summary_path = data_dir / "dataset_summary.json"
    paths: list[Path] = []
    if summary_path.is_file():
        summary = load_json(summary_path)
        for sample in summary.get("samples") or []:
            rel = sample.get("relative_path")
            if rel:
                path = data_dir / rel / "trace.json"
                if path.is_file():
                    paths.append(path)
                    if limit and len(paths) >= limit:
                        return paths
    else:
        for path in sorted(data_dir.glob("*/*/trace.json")):
            paths.append(path)
            if limit and len(paths) >= limit:
                break
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Locate suspicious operator families with GMM.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT.parent / "data")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "output" / "candidates.jsonl")
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sample-cap", type=int, default=5000)
    args = parser.parse_args()

    from teller.candidate.localize import CandidateCfg, fit_models, flatten_trace, localize_trace

    cfg = CandidateCfg(threshold=args.threshold, top_k=args.top_k, sample_cap=args.sample_cap)
    paths = collect_paths(args.data.resolve(), args.max_traces)
    if not paths:
        raise FileNotFoundError(f"no trace.json files found under {args.data}")

    all_nodes = []
    traces: list[tuple[Path, dict]] = []
    for path in tqdm(paths, desc="load traces"):
        trace = load_json(path)
        traces.append((path, trace))
        all_nodes.extend(flatten_trace(trace))

    models = fit_models(all_nodes, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_candidates = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for path, trace in tqdm(traces, desc="localize"):
            result = localize_trace(trace, models, cfg)
            result["path"] = str(path)
            total_candidates += len(result["candidates"])
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps({
        "traces": len(traces),
        "families": len(models),
        "candidates": total_candidates,
        "output": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
