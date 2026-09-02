"""CUDA and Torch version compatibility; avoid hard deps on torch."""

from __future__ import annotations

import sys
from typing import Any

# Optional torch
try:
    import torch
    _TORCH_AVAILABLE = True
    _TORCH_VERSION: str | None = getattr(torch, "__version__", None)
except ImportError:
    _TORCH_AVAILABLE = False
    _TORCH_VERSION = None


def get_torch_version() -> str | None:
    """Return torch version string or None if not installed."""
    return _TORCH_VERSION


def get_cuda_version_from_torch() -> tuple[int, int] | None:
    """Return (major, minor) CUDA version from torch if available."""
    if not _TORCH_AVAILABLE:
        return None
    try:
        v = torch.version.cuda
        if not v:
            return None
        parts = v.split(".")
        major = int(parts[0]) if parts else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except Exception:
        return None


def is_cuda_available() -> bool:
    """True if torch sees CUDA (no import if torch not installed)."""
    if not _TORCH_AVAILABLE:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def nvtx_hook_supported() -> bool:
    """True if NVTX hook is likely to work (Linux + torch/cuda)."""
    if sys.platform != "linux":
        return False
    if not _TORCH_AVAILABLE:
        return False
    try:
        import torch.cuda.nvtx  # noqa: F401
        return True
    except Exception:
        return False
