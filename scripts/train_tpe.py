#!/usr/bin/env python3
"""
Train Trace Pair Encode (TPE) and save vocab, merges, and TracePairTokenizer.

Usage (from project root):
  python scripts/train_tpe.py
  python scripts/train_tpe.py --config configs/train_tpe/default.yaml --output-dir output/train_tpe/my_run

Config: ./configs/train_tpe/*.yaml
Output: ./output/train_tpe/<run_name>/  (vocab.json, merges.json, tokenizer_config.json)
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

# Run from project root so that teller and config paths resolve
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TPE and save tokenizer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_tpe" / "default.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output dir (default: output/train_tpe/<run_name>)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override exp_datasets path (default: from config data.exp_datasets_dir)",
    )
    args = parser.parse_args()

    from teller.trace.tpe_trainer import load_config, train_tpe, save_tpe
    from teller.trace.trace_pair_tokenizer import TracePairTokenizer

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    exp_dir = Path(args.data_dir or data_cfg.get("exp_datasets_dir", "data/exp_datasets"))
    if not exp_dir.is_absolute():
        exp_dir = PROJECT_ROOT / exp_dir
    if not exp_dir.is_dir():
        print(f"[train_tpe] exp_datasets dir not found: {exp_dir}")
        return 1

    out_cfg = config.get("output", {})
    out_dir = args.output_dir or out_cfg.get("dir", "output/train_tpe")
    run_name = out_cfg.get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / run_name
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    print("[train_tpe] Training TPE...")
    vocab, merges = train_tpe(config, exp_dir)
    print(f"[train_tpe] Vocab size: {len(vocab)}, merges: {len(merges)}")

    out_path.mkdir(parents=True, exist_ok=True)
    if out_cfg.get("save_vocab", True):
        save_tpe(vocab, merges, out_path)
        print(f"[train_tpe] Saved vocab and merges to {out_path}")
    if out_cfg.get("save_tokenizer", True):
        tokenizer = TracePairTokenizer(vocab=vocab, merges=merges)
        tokenizer.save_pretrained(out_path)
        print(f"[train_tpe] Saved tokenizer to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
