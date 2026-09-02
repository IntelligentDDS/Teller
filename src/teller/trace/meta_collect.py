"""Collect system, env, software info and file paths for meta_data.json."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _get_gpu_info() -> list[dict] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        gpus = []
        for line in out.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({"index": parts[0], "name": parts[1], "driver": parts[2], "memory_mb": parts[3]})
            elif len(parts) >= 2:
                gpus.append({"index": parts[0], "name": parts[1]})
        return gpus
    except Exception:
        return None


def _get_software_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    versions["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        import torch
        versions["torch"] = getattr(torch, "__version__", "?")
    except ImportError:
        versions["torch"] = "not installed"
    for name, mod in [("vllm", "vllm"), ("sglang", "sglang"), ("transformers", "transformers")]:
        try:
            m = __import__(mod)
            versions[name] = getattr(m, "__version__", "?")
        except ImportError:
            versions[name] = "not installed"
    return versions


def collect_meta(
    run_dir: str | Path,
    run_id: str,
    command: list[str] | None = None,
    task_type: str | None = None,
    tag: str | None = None,
    summary: dict[str, Any] | None = None,
    env_filter: tuple[str, ...] = (
        "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PATH", "LD_LIBRARY_PATH",
        "LD_PRELOAD", "NVTX_INJECTION64_PATH", "TELLER_DATA_DIR", "TELLER_TRACE_DIR",
    ),
) -> dict[str, Any]:
    """
    Build meta_data dict: system, env, software, paths; optional summary merged in.
    run_dir: data/trace/xxx; paths are relative to run_dir.
    """
    run_dir = Path(run_dir)
    paths = {"trace": [], "log": []}
    trace_dir = run_dir / "trace"
    if trace_dir.is_dir():
        paths["trace"] = [f.name for f in trace_dir.glob("*.jsonl")]
    log_dir = run_dir / "log"
    if log_dir.is_dir():
        paths["log"] = []
        if (log_dir / "stdout.log").is_file():
            paths["log"].append("stdout.log")
        if (log_dir / "stderr.log").is_file():
            paths["log"].append("stderr.log")

    system = {
        "machine": platform.machine(),
        "processor": platform.processor() or "",
        "system": platform.system(),
        "release": platform.release(),
        "gpus": _get_gpu_info(),
    }
    env = {k: os.environ.get(k, "") for k in env_filter if os.environ.get(k)}
    software = _get_software_versions()

    out = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "command": command or [],
        "task_type": task_type or "",
        "tag": tag or "",
        "system": system,
        "env": env,
        "software": software,
        "paths": paths,
    }
    if summary is not None:
        out["summary"] = summary
    return out


def write_meta_data(run_dir: str | Path, run_id: str, **kwargs: Any) -> Path:
    """Collect meta and write run_dir/meta_data.json."""
    meta = collect_meta(run_dir, run_id, **kwargs)
    out_path = Path(run_dir) / "meta_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out_path
