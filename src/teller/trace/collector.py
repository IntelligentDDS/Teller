"""
Single-stage trace collection: run with LD_PRELOAD, then write
data/trace/<run_id>/stdout.log (stdout+stderr) and teller_trace.json (meta + merged trace).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from teller.config import get_data_dir, get_default_nvtx_config_path, get_trace_subdir
from teller.trace.parser import TellerTraceParser
from teller.trace.utils import pid_from_trace_filename


def _find_cupti_so(so_path: Optional[str]) -> Path:
    so = os.environ.get("TELLER_CUPTI_SO") or so_path
    if so and Path(so).exists():
        return Path(so)
    pkg_dir = Path(__file__).resolve().parents[1]
    pkg_lib = pkg_dir / "lib" / "libcupti_trace_injection_with_nvtx.so"
    if pkg_lib.exists():
        return pkg_lib
    repo_root = pkg_dir.parent.parent
    build_so = repo_root / "build" / "libcupti_trace_injection_with_nvtx.so"
    if build_so.exists():
        return build_so
    raise FileNotFoundError(
        "CUPTI .so not found. Set TELLER_CUPTI_SO or install the package from source to build CUPTI."
    )


def _find_nvtx_injection_libcupti(cuda_home: str) -> Optional[str]:
    env_path = os.environ.get("NVTX_INJECTION64_PATH", "").strip()
    if env_path and Path(env_path).exists():
        return env_path
    cuda = Path(cuda_home)
    for p in [cuda / "extras" / "CUPTI" / "lib64" / "libcupti.so", cuda / "lib64" / "libcupti.so"]:
        if p.exists():
            return str(p)
    return None


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_with_trace(
    cmd: list[str],
    *,
    run_id: Optional[str] = None,
    task_type: Optional[str] = None,
    tag: Optional[str] = None,
    so_path: Optional[str] = None,
    cuda_home: Optional[str] = None,
    nvtx_json_path: Optional[str] = None,
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> int:
    """
    Run cmd with TELLER tracing. Raw *.jsonl are deleted after parse.
    Output: <run_id>/stdout.log (stdout+stderr combined), <run_id>/teller_trace.json (meta + trace).
    """
    if run_id is None:
        run_id = _make_run_id()
    data_dir = get_data_dir()
    trace_dir = get_trace_subdir(run_id)
    trace_dir.mkdir(parents=True, exist_ok=True)

    so = _find_cupti_so(so_path)
    cuda = cuda_home or os.environ.get("CUDA_HOME", "/usr/local/cuda")
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if str(so) not in ld_preload:
        ld_preload = f"{so}:{ld_preload}" if ld_preload else str(so)

    run_env = dict(os.environ)
    run_env["LD_PRELOAD"] = ld_preload
    run_env["TELLER_ENABLE"] = "1"
    run_env["TELLER_DATA_DIR"] = str(data_dir)
    run_env["TELLER_TRACE_DIR"] = str(trace_dir)
    nvtx_injection = _find_nvtx_injection_libcupti(cuda)
    if nvtx_injection:
        run_env["NVTX_INJECTION64_PATH"] = nvtx_injection
    else:
        run_env.pop("NVTX_INJECTION64_PATH", None)
        print("[teller] warn: NVTX injection libcupti.so not found.", file=sys.stderr)
    run_env["LD_LIBRARY_PATH"] = ":".join(
        filter(
            None,
            [
                os.environ.get("LD_LIBRARY_PATH"),
                str(Path(cuda) / "lib64"),
                str(Path(cuda) / "extras" / "CUPTI" / "lib64"),
            ],
        )
    )
    nvtx_cfg = nvtx_json_path or os.environ.get("NVTX_JSON_PATH")
    if not nvtx_cfg and get_default_nvtx_config_path() is not None:
        nvtx_cfg = str(get_default_nvtx_config_path())
    if nvtx_cfg:
        run_env["NVTX_JSON_PATH"] = nvtx_cfg

    src_dir = Path(__file__).resolve().parents[1].parent
    run_env["PYTHONPATH"] = ":".join(
        list(dict.fromkeys([str(src_dir), run_env.get("PYTHONPATH", "")]))
    )
    run_env["VLLM_WORKER_MULTIPROC_METHOD"] = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    run_env["TORCHDYNAMO_DISABLE"] = "1"
    if "SGLANG_DISABLE_REQUEST_LOGGING" not in run_env:
        run_env["SGLANG_DISABLE_REQUEST_LOGGING"] = "1"
    if env:
        run_env.update(env)

    # Single stdout.log at task root: stdout + stderr combined
    stdout_log = trace_dir / "stdout.log"
    _no_cap = os.environ.get("TELLER_NO_CAPTURE_LOG", "")
    capture_log = _no_cap.strip().lower() not in ("1", "true", "yes")
    if capture_log:
        try:
            with open(stdout_log, "w", encoding="utf-8", errors="replace") as log_f:
                proc = subprocess.run(
                    cmd,
                    cwd=cwd or os.getcwd(),
                    env=run_env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )
        except TypeError:
            proc = subprocess.run(cmd, cwd=cwd or os.getcwd(), env=run_env)
    else:
        proc = subprocess.run(cmd, cwd=cwd or os.getcwd(), env=run_env)
    ret = proc.returncode

    # Merge all PIDs into one trace array; then delete raw .jsonl
    raw_files = [(pid_from_trace_filename(f.name), f) for f in Path(trace_dir).glob("*.jsonl") if f.is_file()]
    raw_files = [(pid, f) for pid, f in raw_files if pid is not None]
    raw_files.sort(key=lambda x: x[0])

    parser = TellerTraceParser()
    all_events = []
    by_cat = {}
    time_ranges = []
    for pid, path in raw_files:
        try:
            events, stats = parser.parse_file(path)
            all_events.extend(events)
            for k, v in stats.get("by_category", {}).items():
                by_cat[k] = by_cat.get(k, 0) + v
            if stats.get("time_range_ns"):
                time_ranges.append(stats["time_range_ns"])
            path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[teller] error: trace for pid {pid} failed: {e}", file=sys.stderr)

    all_events.sort(key=lambda e: (e["start"], e["dur"]))
    time_range_ns = None
    if time_ranges:
        time_range_ns = [min(r[0] for r in time_ranges), max(r[1] for r in time_ranges)]

    log_lines = log_size = 0
    if stdout_log.exists():
        t = stdout_log.read_text(encoding="utf-8", errors="replace")
        log_lines = len([ln for ln in t.splitlines() if ln.strip()])
        log_size = stdout_log.stat().st_size

    meta = {
        "run_id": run_id,
        "command": cmd,
        "task_type": task_type or os.environ.get("TELLER_TASK_TYPE", ""),
        "tag": tag or os.environ.get("TELLER_TAG", ""),
        "pids": [p for p, _ in raw_files],
        "total_events": len(all_events),
        "by_category": by_cat,
        "time_range_ns": time_range_ns,
        "log_lines": log_lines,
        "log_size_bytes": log_size,
    }

    trace_path = trace_dir / "teller_trace.json"
    try:
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "trace": all_events}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[teller] error: teller_trace.json write failed: {e}", file=sys.stderr)

    return ret
