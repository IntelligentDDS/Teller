"""NVTX hook: install import-time tracing for torch/vllm."""

from __future__ import annotations

import atexit
import os
import sys

from teller.config import get_default_nvtx_config_path, get_nvtx_json_path


def install_nvtx_hook(json_path: str | None = None) -> None:
    """Install the NVTX meta path finder. Call from sitecustomize or early in process."""
    if "torch_nvtx_initialized" in sys.modules:
        return
    json_path = json_path or get_nvtx_json_path()
    if not json_path and get_default_nvtx_config_path() is not None:
        json_path = str(get_default_nvtx_config_path())
    if not json_path:
        json_path = os.environ.get("NVTX_JSON_PATH", "")
    from teller.hook.impl import TorchNVTXTracer

    tracer = TorchNVTXTracer(json_path=json_path or "pytorch.json")
    sys.meta_path = [h for h in sys.meta_path if not isinstance(h, TorchNVTXTracer)]
    sys.meta_path.insert(0, tracer)
    sys.modules["torch_nvtx_initialized"] = True

    def _cleanup():
        if getattr(tracer, "loader_instance", None) and getattr(
            tracer.loader_instance, "emit_nvtx_cm", None
        ):
            try:
                tracer.loader_instance.emit_nvtx_cm.__exit__(None, None, None)
            except Exception:
                pass

    atexit.register(_cleanup)
