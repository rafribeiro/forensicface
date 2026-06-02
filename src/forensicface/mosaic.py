"""Mosaic-building workflow for aligned faces."""

from __future__ import annotations

import warnings

import cv2
from imutils import build_montages
import numpy as np
from .utils import annotate_img_with_kps


__all__ = ["build_aligned_face_mosaic"]


def build_aligned_face_mosaic(
    processor,
    img_path_list: list[str | np.ndarray] | None = None,
    mosaic_shape: tuple[int, int] | None = None,
    border: float = 0.03,
    save_to: str | None = None,
    draw_keypoints: bool = False,
    aligned_faces: list | np.ndarray | None = None,
    keypoints: list | np.ndarray | None = None,
) -> np.ndarray:
    """Build a rectangular mosaic of aligned faces using a processor object."""
    if mosaic_shape is None:
        raise ValueError("mosaic_shape is required.")
    if img_path_list is None and aligned_faces is None:
        raise ValueError("Either img_path_list or aligned_faces must be provided.")
    if aligned_faces is not None and draw_keypoints and keypoints is None:
            raise ValueError("draw_keypoints=True requires keypoints when using aligned_faces.")
    top = int(border * processor.IMG_SIZE[0])
    bottom = top
    left = int(border * processor.IMG_SIZE[1])
    right = left

    imgs = []
    list_of_arrays = False
    
    if aligned_faces is not None:
        aligned_faces = list(aligned_faces)
        if keypoints is not None:
            keypoints = list(keypoints)
            if len(keypoints) != len(aligned_faces):
                raise ValueError("keypoints must have the same length as aligned_faces. ")
        for i, aligned_rgb in enumerate(aligned_faces):
            aligned_rgb = np.asarray(aligned_rgb)
            if aligned_rgb.ndim != 3 or aligned_rgb.shape[2] != 3:
                raise ValueError(f"aligned_faces[{i}] must have shape (H, W, 3); ")
            if aligned_rgb.dtype != np.uint8:
                aligned_rgb = aligned_rgb.astype(np.uint8)
            if aligned_rgb.shape[:2] != processor.IMG_SIZE:
                aligned_rgb = cv2.resize(
                    aligned_rgb,
                    processor.IMG_SIZE,
                    interpolation=cv2.INTER_AREA,
                )
            img = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
            if draw_keypoints:
                kps = np.asarray(keypoints[i], dtype=np.float32)
                if kps.shape != (5, 2):
                    raise ValueError(f"keypoints[{i}] must have shape (5, 2).")
                img = annotate_img_with_kps(
                    img,
                    kps=kps,
                    color="green",
                )
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
    else:
        
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
