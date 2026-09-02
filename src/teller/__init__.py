"""TELLER: non-intrusive cross-layer tracing and diagnosis for LLM inference."""

from teller.config import get_data_dir, get_processed_dir


def install_nvtx_hook(json_path: str | None = None) -> None:
    """Install NVTX hook for torch/vllm. Called automatically when PYTHONPATH includes src/."""
    from teller.hook import install_nvtx_hook as _install
    _install(json_path)

__all__ = ["get_data_dir", "get_processed_dir", "install_nvtx_hook"]
