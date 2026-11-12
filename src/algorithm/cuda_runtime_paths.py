"""Windows CUDA DLL search-path helpers for pip-installed NVIDIA wheels."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_ADDED = False
_HANDLES: list[object] = []


def add_cuda_dll_directories() -> None:
    """Add common pip CUDA runtime bin directories before importing CuPy.

    On Windows, packages such as ``nvidia-cublas-cu12`` install DLLs under
    ``site-packages/nvidia/<lib>/bin``. Some launch contexts do not put those
    folders on the DLL search path, causing imports such as ``cupy`` or
    ``nvmath.bindings.cublas`` to fail with Windows error 126 even though the
    wheel files are present.
    """
    global _ADDED
    if _ADDED or os.name != "nt":
        return
    _ADDED = True

    site_roots = [
        Path(prefix) / "Lib" / "site-packages"
        for prefix in {sys.prefix, sys.base_prefix}
    ]
    names = (
        "nvidia/cublas/bin",
        "nvidia/cuda_runtime/bin",
        "nvidia/cuda_nvrtc/bin",
        "nvidia/cusolver/bin",
        "nvidia/cusparse/bin",
        "nvidia/cufft/bin",
        "nvidia/curand/bin",
        "nvidia/nvjitlink/bin",
        "nvidia/cudss/bin",
        "cutensor/bin",
        "cuda_bindings.libs",
        "cuda_core.libs",
    )
    for root in site_roots:
        for name in names:
            directory = root / name
            if not directory.is_dir():
                continue
            try:
                _HANDLES.append(os.add_dll_directory(str(directory)))
            except (FileNotFoundError, OSError):
                pass
