"""Trace collection: run with tracing; output stdout.log + teller_trace.json (meta + trace)."""

from teller.trace.collector import run_with_trace
from teller.trace.parser import TellerTraceParser
from teller.trace.trace_pair_tokenizer import TracePairTokenizer
from teller.trace.trace_parser import parse_step, iter_steps_from_trace

__all__ = [
    "run_with_trace",
    "TellerTraceParser",
    "TracePairTokenizer",
    "parse_step",
    "iter_steps_from_trace",
]
