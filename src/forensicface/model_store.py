"""Model layout resolution helpers."""

from __future__ import annotations

from glob import glob
import os.path as osp
from pathlib import Path


__all__ = [
    "collect_backend_model_files",
    "resolve_component_model",
    "resolve_quality_model",
    "resolve_recognition_model",
]


def resolve_component_model(
    models_root: str,
    *,
    task: str,
    alias: str,
    filenames: tuple[str, ...] = (),
    legacy_paths: tuple[str, ...] = (),
) -> str:
    """Resolve one built-in component model from namespaced/legacy layouts.

    A known filename is preferred. If none is present, a directory containing
    exactly one ONNX file is accepted; this supports versioned offline files
    such as ``centerface20260722.onnx`` without hard-coding release dates.
    """
    directory = Path(models_root) / task / alias
    searched: list[Path] = []
    for filename in filenames:
        candidate = directory / filename
        searched.append(candidate)
        if candidate.is_file():
            return str(candidate)

    if directory.is_dir():
        candidates = sorted(directory.glob("*.onnx"))
        if len(candidates) == 1:
            return str(candidates[0])
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple ONNX files found for {task} model '{alias}' in "
                f"{directory}: {[str(path) for path in candidates]}. "
                "Use ModelSpec(path=...) to select one explicitly."
            )
    searched.append(directory / "*.onnx")

    for relative_path in legacy_paths:
        candidate = Path(models_root) / relative_path
        searched.append(candidate)
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"No ONNX file found for {task} model '{alias}'. Searched: "
        + ", ".join(str(path) for path in searched)
    )


def collect_backend_model_files(models_root: str, model_name: str) -> list[str]:
    """Collect candidate ONNX files for backend probing.

    The task/alias layout is preferred, followed by the flat shared layout and
    the original per-model layout. Duplicate filenames are skipped so a model
    present in more than one layout is loaded from the newest location only.
    """
    sources = [
        osp.join(models_root, "detection", "scrfd"),
        osp.join(models_root, "pose", "insightface-3d68"),
        osp.join(models_root, "attributes", "insightface-genderage"),
        osp.join(models_root, "detection"),
        osp.join(models_root, "attributes"),
        osp.join(models_root, model_name),
    ]
    files: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if not osp.isdir(source):
            continue
        for path in sorted(glob(osp.join(source, "*.onnx"))):
            filename = osp.basename(path)
            if filename in seen:
                continue
            seen.add(filename)
            files.append(path)
    return files


def resolve_recognition_model(models_root: str, model_name: str) -> str:
    """Resolve a recognition ONNX model path from new or legacy layouts."""
    new_pattern = osp.join(models_root, "recognition", model_name, "*face*.onnx")
    legacy_pattern = osp.join(models_root, model_name, "*", "*face*.onnx")

    model_path = glob(new_pattern)
    searched_pattern = new_pattern
    if len(model_path) == 0:
        model_path = glob(legacy_pattern)
        searched_pattern = legacy_pattern

    if len(model_path) == 0:
        raise Exception(
            f"No face embedding model found for '{model_name}'. "
            f"Searched: {new_pattern} and {legacy_pattern}"
        )
    if len(model_path) > 1:
        raise Exception(
            f"Multiple face embedding models found at {searched_pattern}: {model_path}\n"
            f"Please ensure there is only one ONNX file for face embedding in the model directory."
        )
    return model_path[0]


def resolve_quality_model(models_root: str, model_name: str) -> str:
    """Resolve the CR-FIQA model from task/alias, flat, or legacy layouts."""
    namespaced_path = osp.join(
        models_root,
        "quality",
        "cr-fiqa",
        "cr_fiqa_l.onnx",
    )
    if osp.isfile(namespaced_path):
        return namespaced_path
    flat_path = osp.join(models_root, "quality", "cr_fiqa_l.onnx")
    if osp.isfile(flat_path):
        return flat_path
    legacy_path = osp.join(models_root, model_name, "cr_fiqa", "cr_fiqa_l.onnx")
    if osp.isfile(legacy_path):
        return legacy_path
    raise FileNotFoundError(
        "CR-FIQA quality model not found. Searched: "
        f"{namespaced_path}, {flat_path}, and {legacy_path}"
    )
