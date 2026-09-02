"""Hatch build hook: build CUPTI .so before wheel and include it in the package.

CUPTI compilation runs automatically when the package is installed from source.
Requires: cmake, make, CUDA (CUDA_HOME or /usr/local/cuda).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _build_cupti(root: Path) -> bool:
    """Build CUPTI injection .so into src/teller/lib/. Returns True if .so was built."""
    csrc = root / "src" / "csrc"
    build_dir = root / "build_cupti"
    lib_dest = root / "src" / "teller" / "lib"
    so_name = "libcupti_trace_injection_with_nvtx.so"

    if not (csrc / "cupti_injection.cpp").exists():
        print("[teller] src/csrc/cupti_injection.cpp not found, skipping CUPTI build", file=sys.stderr)
        return False

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if not Path(cuda_home).exists():
        print(
            f"[teller] CUDA_HOME={cuda_home} not found, skipping CUPTI build (set CUDA_HOME for tracing)",
            file=sys.stderr,
        )
        return False

    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["cmake", str(csrc), f"-DCUDA_HOME={cuda_home}"],
            cwd=build_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(["make", "-j"], cwd=build_dir, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[teller] CUPTI build failed: {e} (cmake/make and CUDA required for tracing)", file=sys.stderr)
        return False

    so_src = build_dir / so_name
    if not so_src.exists():
        print(f"[teller] {so_name} not produced", file=sys.stderr)
        return False

    lib_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(so_src, lib_dest / so_name)
    print(f"[teller] Built and installed {so_name} -> {lib_dest}")
    return True


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        # Build CUPTI .so into src/teller/lib/ (runs on every pip install / editable install)
        _build_cupti(root)
        # Include built .so in wheel if present
        lib_so = root / "src" / "teller" / "lib" / "libcupti_trace_injection_with_nvtx.so"
        if lib_so.exists():
            build_data.setdefault("force_include", {})[
                str(lib_so)
            ] = "teller/lib/libcupti_trace_injection_with_nvtx.so"
