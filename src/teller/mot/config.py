"""Load TELLER MoT YAML config (model, trace, fault_reason, data, output)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_teller_mot_config(path: str | Path = "configs/mot/default.yaml") -> dict[str, Any]:
    """Load TELLER MoT config from YAML. Path can be relative to cwd or absolute."""
    p = Path(path)
    if not p.is_absolute():
        # Prefer repo root (parent of configs/)
        repo = Path(__file__).resolve().parents[2]
        if (repo / "configs" / "mot" / "default.yaml").exists():
            p = repo / path
        else:
            p = Path.cwd() / path
    if not p.exists():
        raise FileNotFoundError(f"TELLER MoT config not found: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)
