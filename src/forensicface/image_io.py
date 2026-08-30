"""OpenCV image I/O helpers with Unicode path support on Windows."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


__all__ = ["read_image", "write_image"]


def read_image(path: str | os.PathLike[str]) -> np.ndarray | None:
    """Read an image without passing a potentially non-ASCII path to OpenCV."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None

    if not data:
        return None

    return cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def write_image(path: str | os.PathLike[str], image: np.ndarray) -> None:
    """Encode and write an image while preserving Unicode filesystem paths."""
    extension = Path(path).suffix
    if not extension:
        raise ValueError(f"Image output path has no file extension: {path!s}")

    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"OpenCV could not encode image for output path: {path!s}")
    Path(path).write_bytes(encoded.tobytes())
