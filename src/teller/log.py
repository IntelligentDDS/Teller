"""Unified logging for TELLER. No emoji; prefix [teller]."""

from __future__ import annotations

import sys

_PREFIX = "[teller]"


def info(msg: str) -> None:
    print(f"{_PREFIX} {msg}", file=sys.stdout)


def warn(msg: str) -> None:
    print(f"{_PREFIX} warn: {msg}", file=sys.stderr)


def error(msg: str) -> None:
    print(f"{_PREFIX} error: {msg}", file=sys.stderr)
