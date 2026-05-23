"""Model layout resolution helpers."""

from __future__ import annotations

from glob import glob
import os.path as osp


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

