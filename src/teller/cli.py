"""CLI: teller run."""

from __future__ import annotations

import argparse
import sys

from teller.trace.collector import run_with_trace


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="teller",
        description="TELLER: non-intrusive trace collection for LLM inference",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run command with tracing; per-rank trace + log")
    p_run.add_argument(
        "command_args",
        nargs="+",
        help="Command to run (e.g. python example/vllm_offline.py)",
    )
    p_run.add_argument("--run-id", type=str, default=None, help="Subdir under data/trace")
    p_run.add_argument("--task-type", type=str, default="", help="Task type (saved in teller_trace.json meta)")
    p_run.add_argument("--tag", type=str, default=None, help="Optional tag (saved in teller_trace.json meta)")
    p_run.add_argument("--so", type=str, default=None, help="Path to libcupti_trace_injection_with_nvtx.so")
    p_run.add_argument("--cuda-home", type=str, default=None, help="CUDA installation path")
    p_run.add_argument("--nvtx-json", type=str, default=None, help="Path to pytorch.json for NVTX hook")
    p_run.add_argument("-C", "--cwd", type=str, default=None, help="Working directory")

    args = parser.parse_args()

    if args.command == "run":
        return run_with_trace(
            args.command_args,
            run_id=args.run_id,
            task_type=args.task_type,
            tag=args.tag,
            so_path=args.so,
            cuda_home=args.cuda_home,
            nvtx_json_path=args.nvtx_json,
            cwd=args.cwd,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
