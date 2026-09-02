"""Split combined stdout/stderr log by rank and write per-rank .log files."""

from __future__ import annotations

import re
from pathlib import Path

# Match vLLM-style: (EngineCore_DP0 pid=801245) or [rank0]: or rank 0
_RANK_PID = re.compile(r"\([^)]*pid=(\d+)\)|\[rank\s*(\d+)\]|rank\s*(\d+)", re.IGNORECASE)


def _detect_rank_pid(line: str) -> tuple[int | None, int | None]:
    """Return (rank_index, pid) if line is from a known rank/pid, else (None, None)."""
    m = _RANK_PID.search(line)
    if not m:
        return None, None
    pid = m.group(1)
    rank_from_bracket = m.group(2)
    rank_from_word = m.group(3)
    if pid:
        try:
            pid_int = int(pid)
            # pid known, rank unknown -> use pid as rank key; rank index assigned by sorted(pid)
            return None, pid_int
        except ValueError:
            pass
    if rank_from_bracket is not None:
        try:
            return int(rank_from_bracket), None
        except ValueError:
            pass
    if rank_from_word is not None:
        try:
            return int(rank_from_word), None
        except ValueError:
            pass
    return None, None


def split_log_by_rank(
    combined_log_path: str | Path,
    output_log_dir: str | Path,
    pid_to_rank: dict[int, int] | None = None,
) -> dict[int, list[str]]:
    """
    Read combined log file, split lines by rank/pid into per-rank lists of log entries.
    Writes output_log_dir/rank_X_pid_XXX.json for each rank.
    pid_to_rank: optional mapping pid -> rank_index (e.g. from trace merge); used when line only has pid.
    Returns dict rank_index -> list of {"line", "pid", "rank"} entries.
    """
    combined_log_path = Path(combined_log_path)
    output_log_dir = Path(output_log_dir)
    output_log_dir.mkdir(parents=True, exist_ok=True)
    pid_to_rank = pid_to_rank or {}

    # Collect lines by (rank_index, pid). If we only have pid, use pid_to_rank to get rank.
    by_key: dict[tuple[int, int], list[str]] = {}

    def add_entry(rank_idx: int | None, pid_val: int | None, line: str) -> None:
        if rank_idx is None and pid_val is not None:
            rank_idx = pid_to_rank.get(pid_val, len(pid_to_rank))
        if rank_idx is None:
            rank_idx = 0
        if pid_val is None:
            pid_val = 0
        key = (rank_idx, pid_val)
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(line.rstrip("\n"))

    if not combined_log_path.exists():
        return {}

    current_rank: int | None = None
    current_pid: int | None = None
    with open(combined_log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            r, p = _detect_rank_pid(line)
            if r is not None:
                current_rank = r
            if p is not None:
                current_pid = p
            add_entry(current_rank, current_pid, line)

    # If no rank/pid ever detected, put all in rank_0_pid_0
    if not by_key:
        with open(combined_log_path, "r", encoding="utf-8", errors="replace") as f:
            by_key[(0, 0)] = [ln.rstrip("\n") for ln in f]

    result = {}
    for (rank_idx, pid_val), entries in sorted(by_key.items()):
        out_path = output_log_dir / f"rank_{rank_idx}_pid_{pid_val}.log"
        with open(out_path, "w", encoding="utf-8", errors="replace") as out:
            out.write("\n".join(entries))
            if entries:
                out.write("\n")
        result[rank_idx] = entries

    return result
