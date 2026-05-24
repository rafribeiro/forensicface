"""Model layout resolution helpers."""

from __future__ import annotations

from glob import glob
import os.path as osp


__all__ = [
    "collect_backend_model_files",
    "resolve_quality_model",
    "resolve_recognition_model",
]


def collect_backend_model_files(models_root: str, model_name: str) -> list[str]:
    """Collect candidate ONNX files for backend probing.

    The new shared layout is preferred over the legacy per-model layout:
    ``detection/`` and ``attributes/`` under ``models_root`` are searched
    first, followed by ``<models_root>/<model_name>/``. Duplicate filenames are
    skipped so a model that exists in both layouts is loaded from the new
    shared location only.
    """
    sources = [
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
    """Resolve the CR-FIQA quality model from new or legacy layouts."""
    new_path = osp.join(models_root, "quality", "cr_fiqa_l.onnx")
    if osp.isfile(new_path):
        return new_path
    legacy_path = osp.join(models_root, model_name, "cr_fiqa", "cr_fiqa_l.onnx")
    if osp.isfile(legacy_path):
        return legacy_path
    raise FileNotFoundError(
        f"CR-FIQA quality model not found. Searched: {new_path} and {legacy_path}"
    )
