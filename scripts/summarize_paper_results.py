#!/usr/bin/env python3
"""Summarize the stored paper main-result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


FIELDS = [
    "step_accuracy",
    "step_precision",
    "step_f1",
    "operator_macro_precision",
    "operator_macro_f1",
    "operator_macro_jaccard",
    "operator_macro+_precision",
    "operator_macro+_f1",
    "operator_macro+_jaccard",
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
]


def load_result(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    missing = [name for name in FIELDS if name not in data]
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    return data


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize paper main-result JSON files")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "output")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.results_dir.glob("trace_vocab_*_*_result.json")):
        rows.append(load_result(path))

    if not rows:
        print(f"No paper result JSON files found under {args.results_dir}")
        return 1

    rows.sort(key=lambda r: (str(r["trace_type"]), int(r["trace_vocab_size"])))
    header = [
        "View",
        "Vocab",
        "Acc.",
        "Prec.",
        "F1",
        "Op Prec.",
        "Op F1",
        "Op Jac.",
        "Op+ Prec.",
        "Op+ F1",
        "Op+ Jac.",
        "BLEU-1",
        "BLEU-2",
        "BLEU-3",
        "BLEU-4",
    ]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["trace_type"]),
                    str(row["trace_vocab_size"]),
                    fmt(row["step_accuracy"]),
                    fmt(row["step_precision"]),
                    fmt(row["step_f1"]),
                    fmt(row["operator_macro_precision"]),
                    fmt(row["operator_macro_f1"]),
                    fmt(row["operator_macro_jaccard"]),
                    fmt(row["operator_macro+_precision"]),
                    fmt(row["operator_macro+_f1"]),
                    fmt(row["operator_macro+_jaccard"]),
                    fmt(row["bleu_1"]),
                    fmt(row["bleu_2"]),
                    fmt(row["bleu_3"]),
                    fmt(row["bleu_4"]),
                ]
            )
            + " |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
