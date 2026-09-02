"""
Single TellerTraceParser: raw trace events -> compact JSONL (name, type, start, dur, pid, tid).
Unified rules: build NVTX ranges from push/pop; filter/parse event names.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from teller.trace.events import get_event_time, normalize_event

# Name filtering: drop noisy suffixes; keep full names (no shortening of torch.nn etc.)
MAX_NAME_LEN = 120
_MAX_KERNEL_NAME = 80
# "func at /path/file.py:123" -> func (we drop the location part per user request)
_NVTX_AT = re.compile(r"^(.+?)\s+at\s+(.+)$")
# "step_execute_model[request_ids:...]"
_STEP_PREFIX = re.compile(r"^step_execute_model\[.*\]$")
# Strip ", op_id = 123", ", seq = 16993", ", key = value" from NVTX/aten names
_COMMA_KV = re.compile(r",\s*\w+\s*=\s*[^,\[]+")
# cudaLaunchKernel_v7000 -> cudaLaunchKernel
_RUNTIME_SUFFIX = re.compile(r"_v\d+$")
# Trailing " (file:line)" or " (:0)" -> drop so "linear (:0)" -> "linear", "Module._call_impl (module.py:1779)" -> "Module._call_impl"
_STRIP_LOCATION = re.compile(r"\s*\([^)]*\)\s*$")

# NVTX subtypes: PYTORCH, LIBTORCH, ENGINE, and NVTX.
_LIBTORCH_PREFIXES = ("aten::", "c10::", "caffe2::", "torch::")


def _nvtx_subtype(raw_name: str) -> str:
    """Classify an NVTX marker into PYTORCH, LIBTORCH, ENGINE, or NVTX."""
    if not raw_name or not isinstance(raw_name, str):
        return "NVTX"
    s = raw_name.strip()
    if not s:
        return "NVTX"
    if _STEP_PREFIX.match(s) or "step_execute_model" in s:
        return "ENGINE"
    if s.startswith("torch."):
        return "PYTORCH"
    if any(s.startswith(p) for p in _LIBTORCH_PREFIXES):
        return "LIBTORCH"
    low = s.lower()
    if any(k in low for k in ("vllm", "sglang", "transformers", "engine_core", "execute_model")):
        return "ENGINE"
    return "NVTX"


def _filter_event_name(name: str, cat: str) -> str:
    """Filter event name: strip location suffix and comma key=value; keep full names (no torch prefix shortening)."""
    if not name or not isinstance(name, str):
        return cat.lower()
    s = name.strip()
    if not s:
        return cat.lower()
    if cat == "NVTX":
        m = _NVTX_AT.match(s)
        if m:
            # Keep full func name (no shortening); drop " at path:line" -> just func
            s = m.group(1).strip()
        elif _STEP_PREFIX.match(s):
            s = "step_execute_model"
        else:
            s = _COMMA_KV.sub("", s).strip()
        # Drop trailing " (file:line)" or " (:0)" so "linear (:0)" -> "linear", "Module._call_impl (module.py:1779)" -> "Module._call_impl"
        s = _STRIP_LOCATION.sub("", s).strip()
    elif cat == "RUNTIME":
        s = _RUNTIME_SUFFIX.sub("", s)
    elif cat == "DRIVER":
        s = _RUNTIME_SUFFIX.sub("", s)
    elif cat == "KERNEL":
        if len(s) > _MAX_KERNEL_NAME:
            s = s[: _MAX_KERNEL_NAME - 2] + ".."
    if len(s) > MAX_NAME_LEN:
        s = s[: MAX_NAME_LEN - 2] + ".."
    return s


def _compact_event(name: str, type_: str, start_ns: int | float, dur_ns: int | float, pid: int, tid: int) -> dict:
    """One trace event: name, type, start, dur, pid, tid (all scalar, start/dur in ns)."""
    return {
        "name": name,
        "type": type_,
        "start": int(start_ns),
        "dur": int(dur_ns),
        "pid": pid,
        "tid": tid,
    }


class TellerTraceParser:
    """Convert raw trace JSONL events to compact JSONL (name, type, start, dur, pid, tid)."""

    def parse_events(self, events: list[dict]) -> tuple[list[dict], dict]:
        """
        Build compact trace from normalized events.
        Returns (list of {name, type, start, dur, pid, tid}, stats dict). start/dur in ns.
        """
        events = sorted(events, key=get_event_time)
        trace_events = []

        # NVTX ranges: push (name not null) -> stack; pop (null or [range_end]) -> close by id
        process_stacks: dict[int, list[dict]] = defaultdict(list)
        for ev in events:
            if ev.get("type") != "NVTX_MARKER":
                continue
            name = ev.get("name")
            ts = ev.get("timestamp")
            marker_id = ev.get("id")
            process_id = ev.get("process_id") or 0
            thread_id = ev.get("thread_id") or 0
            if ts is None:
                continue
            if name in ("null", "[range_end]", None) or name == "":
                if marker_id is None:
                    continue
                stack = process_stacks[process_id]
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i]["id"] == marker_id:
                        r = stack.pop(i)
                        dur = ts - r["start"]
                        nvtx_type = _nvtx_subtype(r["name"])
                        trace_events.append(_compact_event(
                            _filter_event_name(r["name"], "NVTX"),
                            nvtx_type,
                            r["start"],
                            dur,
                            process_id,
                            r.get("tid", thread_id),
                        ))
                        break
                continue
            if marker_id is None:
                continue
            process_stacks[process_id].append({
                "name": name,
                "start": ts,
                "id": marker_id,
                "tid": thread_id,
            })

        # RUNTIME
        for ev in events:
            if ev.get("type") != "RUNTIME":
                continue
            name = ev.get("name") or "runtime"
            start = ev.get("start", 0)
            end = ev.get("end") or start
            dur = max(end - start, 0)
            trace_events.append(_compact_event(
                _filter_event_name(name, "RUNTIME"),
                "RUNTIME",
                start,
                dur,
                ev.get("process_id") or 0,
                ev.get("thread_id") or 0,
            ))

        # DRIVER
        for ev in events:
            if ev.get("type") != "DRIVER":
                continue
            name = ev.get("name") or "driver"
            start = ev.get("start", 0)
            end = ev.get("end") or start
            dur = max(end - start, 0)
            trace_events.append(_compact_event(
                _filter_event_name(name, "DRIVER"),
                "DRIVER",
                start,
                dur,
                ev.get("process_id") or 0,
                ev.get("thread_id") or 0,
            ))

        # KERNEL (GPU time: gpu_start, gpu_end)
        for ev in events:
            if ev.get("type") != "KERNEL":
                continue
            name = ev.get("name") or "kernel"
            gs = ev.get("gpu_start", 0)
            ge = ev.get("gpu_end")
            if ge is None:
                d = ev.get("duration", 0)
                ge = gs + d if d else gs
            dur = max(ge - gs, 0)
            trace_events.append(_compact_event(
                _filter_event_name(name, "KERNEL"),
                "KERNEL",
                gs,
                dur,
                ev.get("process_id") or 0,
                0,
            ))

        trace_events.sort(key=lambda e: (e["start"], e["dur"]))
        by_cat = defaultdict(int)
        min_ts = max_ts = None
        for e in trace_events:
            by_cat[e["type"]] += 1
            st, du = e["start"], e["dur"]
            end_ts = st + du
            if min_ts is None or st < min_ts:
                min_ts = st
            if max_ts is None or end_ts > max_ts:
                max_ts = end_ts
        stats = {
            "total_events": len(trace_events),
            "by_category": dict(by_cat),
            "time_range_ns": [min_ts, max_ts] if min_ts is not None else None,
        }
        return trace_events, stats

    def parse_file(self, path: str | Path) -> tuple[list[dict], dict]:
        """Read JSONL, normalize, parse; return (list of compact events, stats dict)."""
        path = Path(path)
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(normalize_event(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        return self.parse_events(events)
