# ort_runtime_setup.py
from __future__ import annotations

import ctypes
import os
import platform
import site
import sys
from pathlib import Path
from typing import Iterable

# Keep these alive. On Windows, os.add_dll_directory handles are removed
# when the handle object is garbage-collected.
_DLL_DIR_HANDLES = []

# Keep ctypes-loaded libraries alive for the process lifetime.
_PRELOADED_LIBS = []

_NVIDIA_COMPONENTS_IN_LOAD_ORDER = [
    "cuda_runtime",
    "cuda_nvrtc",
    "nvjitlink",
    "cublas",
    "cufft",
    "curand",
    "cudnn",
]


def _site_packages_dirs() -> list[Path]:
    dirs: list[Path] = []

    try:
        dirs.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass

    try:
        dirs.append(Path(site.getusersitepackages()))
    except Exception:
        pass

    # Useful for venv/conda environments where getsitepackages can vary.
    dirs.extend(Path(p) for p in sys.path if "site-packages" in p)

    # Preserve order, remove duplicates.
    seen = set()
    result = []
    for d in dirs:
        d = d.resolve()
        if d not in seen and d.exists():
            seen.add(d)
            result.append(d)

    return result


def _nvidia_component_dirs() -> list[Path]:
    system = platform.system()
    result: list[Path] = []

    for sp in _site_packages_dirs():
        nvidia_root = sp / "nvidia"
        if not nvidia_root.exists():
            continue

        for component in _NVIDIA_COMPONENTS_IN_LOAD_ORDER:
            if system == "Windows":
                candidate = nvidia_root / component / "bin"
            elif system == "Linux":
                candidate = nvidia_root / component / "lib"
            else:
                continue

            if candidate.exists():
                result.append(candidate)

    # Preserve order, remove duplicates.
    seen = set()
    unique = []
    for d in result:
        d = d.resolve()
        if d not in seen:
            seen.add(d)
            unique.append(d)

    return unique


def _prepend_env_path(var_name: str, dirs: Iterable[Path]) -> None:
    existing = os.environ.get(var_name, "")
    parts = [str(d) for d in dirs]
    if existing:
        parts.append(existing)
    os.environ[var_name] = os.pathsep.join(parts)


def _setup_windows_nvidia_paths(verbose: bool) -> list[Path]:
    dirs = _nvidia_component_dirs()

    for d in dirs:
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(d)))
            if verbose:
                print(f"Added Windows DLL directory: {d}")
        except Exception as exc:
            if verbose:
                print(f"Could not add Windows DLL directory {d}: {exc}")

    # Helps libraries that use PATH-based lookup internally.
    _prepend_env_path("PATH", dirs)

    return dirs


def _linux_library_files(dirs: Iterable[Path]) -> list[Path]:
    libs: list[Path] = []

    # Load lower-level CUDA libs before cuDNN.
    for d in dirs:
        if not d.exists():
            continue

        # Prefer versioned shared objects, but include plain .so too.
        for pattern in ("*.so", "*.so.*"):
            libs.extend(sorted(d.glob(pattern)))

    # Avoid loading the same resolved file more than once.
    seen = set()
    unique = []
    for lib in libs:
        try:
            resolved = lib.resolve()
        except Exception:
            resolved = lib

        if resolved not in seen and resolved.is_file():
            seen.add(resolved)
            unique.append(resolved)

    return unique


def _setup_linux_nvidia_paths_and_preload(verbose: bool) -> list[Path]:
    dirs = _nvidia_component_dirs()

    # Useful for subprocesses and sometimes for libraries that inspect env.
    # Note: on Linux, changing LD_LIBRARY_PATH after process start is not
    # always enough by itself, so we also ctypes-preload absolute paths below.
    _prepend_env_path("LD_LIBRARY_PATH", dirs)

    rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)
    loaded = 0

    for lib in _linux_library_files(dirs):
        try:
            _PRELOADED_LIBS.append(ctypes.CDLL(str(lib), mode=rtld_global))
            loaded += 1
            if verbose:
                print(f"Preloaded Linux shared library: {lib}")
        except OSError as exc:
            # Some libs may depend on optional system pieces or be loaded later.
            # Do not fail the whole setup unless CUDA provider creation fails.
            if verbose:
                print(f"Skipped {lib}: {exc}")

    if verbose:
        print(f"Linux NVIDIA library directories: {[str(d) for d in dirs]}")
        print(f"Preloaded {loaded} NVIDIA shared libraries")

    return dirs


def configure_onnxruntime_acceleration(
    *,
    prefer_cuda: bool = True,
    allow_coreml_on_macos: bool = True,
    verbose: bool = True,
) -> list:
    """
    Configure ONNX Runtime library loading as early as possible.

    Call this before importing libraries that create ONNX Runtime sessions,
    e.g. before importing forensicface / insightface wrappers.

    Returns a providers list suitable for ort.InferenceSession(...).

    Windows:
        Uses os.add_dll_directory + PATH + ort.preload_dlls(directory="").

    Linux:
        Adds NVIDIA wheel lib dirs to LD_LIBRARY_PATH for child processes,
        preloads shared libraries by absolute path with ctypes, then calls
        ort.preload_dlls(directory="") where available.

    macOS:
        CUDA is not available through onnxruntime-gpu. Optionally returns
        CoreMLExecutionProvider if present, otherwise CPUExecutionProvider.
    """
    system = platform.system()

    if prefer_cuda and system == "Windows":
        _setup_windows_nvidia_paths(verbose)

    elif prefer_cuda and system == "Linux":
        _setup_linux_nvidia_paths_and_preload(verbose)

    elif prefer_cuda and system == "Darwin":
        if verbose:
            print(
                "macOS detected: CUDAExecutionProvider is not available via "
                "onnxruntime-gpu. Will try CoreMLExecutionProvider if available."
            )

    import onnxruntime as ort

    # ONNX Runtime documents this for loading CUDA/cuDNN from NVIDIA site-packages.
    # directory="" means NVIDIA site packages.
    if prefer_cuda and system in {"Windows", "Linux"} and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(directory="")
            if verbose:
                print("Called onnxruntime.preload_dlls(directory='')")
        except Exception as exc:
            if verbose:
                print(f"onnxruntime.preload_dlls(directory='') failed: {exc}")

    available = ort.get_available_providers()

    if verbose:
        print("Available ONNX Runtime providers:", available)
        try:
            ort.print_debug_info()
        except Exception:
            pass

    if prefer_cuda and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if system == "Darwin" and allow_coreml_on_macos and "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]