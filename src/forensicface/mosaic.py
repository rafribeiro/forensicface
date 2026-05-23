"""Mosaic-building workflow for aligned faces."""

from __future__ import annotations

import warnings

import cv2
from imutils import build_montages
import numpy as np


__all__ = ["build_aligned_face_mosaic"]


def build_aligned_face_mosaic(
    processor,
    img_path_list: list[str | np.ndarray],
    mosaic_shape: tuple[int, int],
    border: float = 0.03,
    save_to: str | None = None,
    draw_keypoints: bool = False,
) -> np.ndarray:
    """Build a rectangular mosaic of aligned faces using a processor object."""
    if mosaic_shape is None:
        raise ValueError("mosaic_shape is required.")
    top = int(border * processor.IMG_SIZE[0])
    bottom = top
    left = int(border * processor.IMG_SIZE[1])
    right = left

    imgs = []
    list_of_arrays = False
    for img in img_path_list:
        if type(img) != str:
            list_of_arrays = True
        ret = processor.process_image(
            img, draw_keypoints=draw_keypoints, single_face=True
        )
        if len(ret) > 0:
            img = cv2.cvtColor(ret["aligned_face"], cv2.COLOR_RGB2BGR)
            img = cv2.copyMakeBorder(
                img,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            imgs.append(img)
    mosaic = build_montages(
        imgs,
        image_shape=(
            int(processor.IMG_SIZE[0] * (1 + 2 * border)),
            int(processor.IMG_SIZE[1] * (1 + 2 * border)),
        ),
        montage_shape=mosaic_shape,
    )[0]
    if list_of_arrays:
        warnings.warn(
            "A list of arrays was passed as argument. Make sure image arrays are in BGR format.",
            Warning,
        )
    if save_to is not None:
        cv2.imwrite(save_to, mosaic)
    return mosaic
