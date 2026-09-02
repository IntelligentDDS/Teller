"""Event schema: normalize short keys from C++ writer to long keys for Python pipeline."""

from __future__ import annotations


def get_event_time(ev: dict) -> float:
    """Timestamp for ordering events (nanoseconds)."""
    if ev.get("type") == "NVTX_MARKER":
        return ev.get("timestamp") or float("inf")
    if ev.get("type") in ("RUNTIME", "DRIVER"):
        return ev.get("start") or float("inf")
    if ev.get("type") == "KERNEL":
        return ev.get("gpu_start") or float("inf")
    return float("inf")


# Short key -> long key for JSONL events (C++ compression)
_NVTX_SHORT = {"t": "type", "n": "name", "ts": "timestamp", "id": "id", "pid": "process_id", "tid": "thread_id"}
_KERNEL_SHORT = {"t": "type", "n": "name", "gs": "gpu_start", "ge": "gpu_end", "d": "duration", "c": "correlation_id"}
_API_SHORT = {"t": "type", "cbid": "cbid", "n": "name", "s": "start", "e": "end", "d": "duration", "c": "correlation_id", "pid": "process_id", "tid": "thread_id"}


def normalize_event(raw: dict) -> dict:
    """Convert short keys to long keys if present; leave long keys unchanged."""
    t = raw.get("t") or raw.get("type")
    if t is None:
        return raw
    out = dict(raw)
    if "t" in out:
        out["type"] = out.pop("t")
    if "n" in out:
        out["name"] = out.pop("n")
    if "ts" in out:
        out["timestamp"] = out.pop("ts")
    if "id" in out and "id" not in _API_SHORT:
        pass  # keep id
    if "pid" in out:
        out["process_id"] = out.pop("pid")
    if "tid" in out:
        out["thread_id"] = out.pop("tid")
    if "gs" in out:
        out["gpu_start"] = out.pop("gs")
    if "ge" in out:
        out["gpu_end"] = out.pop("ge")
    if "s" in out:
        out["start"] = out.pop("s")
    if "e" in out:
        out["end"] = out.pop("e")
    if "d" in out:
        out["duration"] = out.pop("d")
    if "c" in out:
        out["correlation_id"] = out.pop("c")
    return out
