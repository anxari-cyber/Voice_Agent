from __future__ import annotations

import os
import sys
from pathlib import Path

_dll_directories: list[object] = []


def configure_cuda_dlls() -> None:
    """Register pip-installed CUDA DLL directories for the current Python process."""
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    if not site_packages.exists():
        return

    package_directories = (
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
    )
    for directory in package_directories:
        if directory.exists() and hasattr(os, "add_dll_directory"):
            _dll_directories.append(os.add_dll_directory(str(directory)))
            os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
