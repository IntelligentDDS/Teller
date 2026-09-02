"""Paths: ./data and ./processed_data relative to project root."""

import os
from pathlib import Path

ENV_DATA_DIR = "TELLER_DATA_DIR"
ENV_PROCESSED_DIR = "TELLER_PROCESSED_DIR"
ENV_PROJECT_ROOT = "TELLER_PROJECT_ROOT"
ENV_NVTX_JSON_PATH = "NVTX_JSON_PATH"

_DATA_SUBDIR = "data"
_PROCESSED_SUBDIR = "processed_data"


def _env_first(*keys: str) -> str | None:
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return None


def get_project_root() -> Path:
    """Project root (directory containing pyproject.toml)."""
    env = _env_first(ENV_PROJECT_ROOT)
    if env:
        return Path(env).resolve()
    # From this file: src/teller/config.py -> repo root is parents[2]
    start = Path(__file__).resolve()
    for p in [start.parents[2], start.parents[3], Path.cwd()]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def get_data_dir() -> Path:
    """Raw collected trace/log; default ./data (relative to project root)."""
    raw = _env_first(ENV_DATA_DIR)
    if raw:
        return Path(raw).resolve()
    return (get_project_root() / _DATA_SUBDIR).resolve()


def get_processed_dir() -> Path:
    """Processed/annotated data; default ./processed_data."""
    raw = _env_first(ENV_PROCESSED_DIR)
    if raw:
        return Path(raw).resolve()
    return (get_project_root() / _PROCESSED_SUBDIR).resolve()


def get_trace_subdir(run_id: str | None = None) -> Path:
    base = get_data_dir() / "trace"
    if run_id:
        return base / run_id
    return base


def get_log_subdir(run_id: str | None = None) -> Path:
    base = get_data_dir() / "log"
    if run_id:
        return base / run_id
    return base


def get_nvtx_json_path() -> str | None:
    return os.environ.get(ENV_NVTX_JSON_PATH)


def get_default_nvtx_config_path() -> Path | None:
    """Default NVTX config: package config/pytorch.json or project src/config."""
    # Package layout: teller/config/pytorch.json
    pkg_config = Path(__file__).resolve().parent / "config" / "pytorch.json"
    if pkg_config.exists():
        return pkg_config
    root = get_project_root()
    for sub in ["src/config", "config"]:
        p = root / sub / "pytorch.json"
        if p.exists():
            return p
    return None
