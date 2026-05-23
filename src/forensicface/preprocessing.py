"""Preprocessing helpers for recognition ONNX inputs."""

from __future__ import annotations

import numpy as np


def to_ada_input(
    aligned_bgr_img: np.ndarray,
    *,
    image_size: tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Convert aligned BGR face crop(s) to AdaFace-style NCHW input."""
    arr = aligned_bgr_img.astype(np.float32)
    arr = ((arr / 255.0) - 0.5) / 0.5
    if arr.ndim == 3:
        return arr.transpose(2, 0, 1).reshape(1, 3, *image_size)
    if arr.ndim == 4:
        return arr.transpose(0, 3, 1, 2).copy()
    raise ValueError(
        f"Expected ndim 3 (H, W, 3) or 4 (N, H, W, 3); got {arr.ndim}."
    )


def normalize_aligned_keypoints(
    aligned_keypoints: np.ndarray | None,
    *,
    model_name: str,
    image_size: tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Normalize aligned 5-point keypoints to the [0, 1] ONNX input frame."""
    if aligned_keypoints is None:
        raise ValueError(
            f"Model '{model_name}' requires aligned 5-point keypoints in "
            "the 112x112 face image coordinate system."
        )

    aligned_keypoints = np.asarray(aligned_keypoints, dtype=np.float32)
    if aligned_keypoints.shape == (5, 2):
        normalized_keypoints = aligned_keypoints.copy()
        normalized_keypoints[:, 0] /= image_size[1]
        normalized_keypoints[:, 1] /= image_size[0]
        return normalized_keypoints.reshape(1, 5, 2)
    if aligned_keypoints.ndim == 3 and aligned_keypoints.shape[1:] == (5, 2):
        normalized_keypoints = aligned_keypoints.copy()
        normalized_keypoints[:, :, 0] /= image_size[1]
        normalized_keypoints[:, :, 1] /= image_size[0]
        return normalized_keypoints

    raise ValueError(
        f"Model '{model_name}' requires keypoints with shape "
        f"(5, 2) or (N, 5, 2); received {aligned_keypoints.shape}."
    )

