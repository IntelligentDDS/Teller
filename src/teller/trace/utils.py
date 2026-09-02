import re

_PID_IN_FILENAME = re.compile(r"pid(\d+)")


def pid_from_trace_filename(name):
    """Extract process id from trace filename like output_pid952379.tmp.jsonl."""
    m = _PID_IN_FILENAME.search(name)
    return int(m.group(1)) if m else None
