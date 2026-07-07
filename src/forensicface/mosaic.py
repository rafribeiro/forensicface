"""Mosaic-building utilities for original images and aligned faces."""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from .utils import DEFAULT_KEYPOINT_COLORS, annotate_img_with_kps


__all__ = [
    "build_mosaic_from_images",
    "build_mosaic_from_aligned_faces",
]


DEFAULT_FACE_SIZE = (112, 112)


def _validate_pair(name: str, value: tuple[int, int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must be a tuple of two integers.")
    first, second = int(value[0]), int(value[1])
    if first <= 0 or second <= 0:
        raise ValueError(f"{name} values must be positive.")
    return first, second


def _border_widths(image_size: tuple[int, int], border: float) -> tuple[int, int]:
    if border < 0:
        raise ValueError("border must be non-negative.")
    height, width = image_size
    return int(border * height), int(border * width)


def _add_border(img: np.ndarray, image_size: tuple[int, int], border: float) -> np.ndarray:
    top, left = _border_widths(image_size, border)
    return cv2.copyMakeBorder(
        img,
        top=top,
        bottom=top,
        left=left,
        right=left,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def _tile_images(
    image_list: list[np.ndarray],
    tile_size: tuple[int, int],
    mosaic_shape: tuple[int, int],
) -> np.ndarray:
    """Tile BGR images row-major into one mosaic.

    Adapted from imutils.build_montages, which is MIT licensed:
    copyright (c) 2015-2016 Adrian Rosebrock, PyImageSearch.
    The original build_montages docstring credits Kyle Hounslow.
    """
    tile_height, tile_width = tile_size
    n_cols, n_rows = mosaic_shape
    mosaic = np.zeros((tile_height * n_rows, tile_width * n_cols, 3), dtype=np.uint8)

    for idx, img in enumerate(image_list[: n_cols * n_rows]):
        if not isinstance(img, np.ndarray):
            raise TypeError(f"image_list[{idx}] must be a numpy array.")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"image_list[{idx}] must have shape (H, W, 3).")
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.shape[:2] != tile_size:
            img = cv2.resize(img, (tile_width, tile_height), interpolation=cv2.INTER_AREA)

        row = idx // n_cols
        col = idx % n_cols
        y0 = row * tile_height
        x0 = col * tile_width
        mosaic[y0 : y0 + tile_height, x0 : x0 + tile_width] = img

    return mosaic


def _build_mosaic(
    imgs: list[np.ndarray],
    *,
    image_size: tuple[int, int],
    mosaic_shape: tuple[int, int],
    border: float,
    save_to: str | None,
) -> np.ndarray:
    height, width = image_size
    border_y, border_x = _border_widths(image_size, border)
    tile_size = (height + 2 * border_y, width + 2 * border_x)
    mosaic = _tile_images(imgs, tile_size=tile_size, mosaic_shape=mosaic_shape)
    if save_to is not None:
        cv2.imwrite(save_to, mosaic)
    return mosaic


def build_mosaic_from_images(
    processor,
    img_path_list: list[str | np.ndarray],
    mosaic_shape: tuple[int, int],
    border: float = 0.03,
    save_to: str | None = None,
    draw_keypoints: bool = False,
    keypoint_colors: tuple[str, str, str, str, str] = DEFAULT_KEYPOINT_COLORS,
) -> np.ndarray:
    """Detect, align, and build a rectangular BGR mosaic from original images."""
    mosaic_shape = _validate_pair("mosaic_shape", mosaic_shape)
    image_size = _validate_pair("image_size", processor.IMG_SIZE)

    imgs = []
    list_of_arrays = False
    for img in img_path_list:
        if not isinstance(img, str):
            list_of_arrays = True

        ret = processor.process_image(
            img,
            draw_keypoints=draw_keypoints,
            keypoint_colors=keypoint_colors,
            single_face=True,
        )
        if ret is None or len(ret) == 0:
            continue
        if isinstance(ret, list):
            ret = ret[0]

        aligned_bgr = cv2.cvtColor(ret["aligned_face"], cv2.COLOR_RGB2BGR)
        imgs.append(_add_border(aligned_bgr, image_size, border))

    if list_of_arrays:
        warnings.warn(
            "A list of arrays was passed as argument. Make sure image arrays are in BGR format.",
            Warning,
        )

    return _build_mosaic(
        imgs,
        image_size=image_size,
        mosaic_shape=mosaic_shape,
        border=border,
        save_to=save_to,
    )


def build_mosaic_from_aligned_faces(
    aligned_faces: list[np.ndarray] | np.ndarray,
    mosaic_shape: tuple[int, int],
    border: float = 0.03,
    save_to: str | None = None,
    draw_keypoints: bool = False,
    keypoints: list[np.ndarray] | np.ndarray | None = None,
    keypoint_colors: tuple[str, str, str, str, str] = DEFAULT_KEYPOINT_COLORS,
    image_size: tuple[int, int] = DEFAULT_FACE_SIZE,
) -> np.ndarray:
    """Build a rectangular BGR mosaic from already aligned RGB face images."""
    mosaic_shape = _validate_pair("mosaic_shape", mosaic_shape)
    image_size = _validate_pair("image_size", image_size)

    aligned_faces = list(aligned_faces)
    if draw_keypoints and keypoints is None:
        raise ValueError("draw_keypoints=True requires keypoints.")
    if keypoints is not None:
        keypoints = list(keypoints)
        if len(keypoints) != len(aligned_faces):
            raise ValueError("keypoints must have the same length as aligned_faces.")

    imgs = []
    for idx, aligned_rgb in enumerate(aligned_faces):
        aligned_rgb = np.asarray(aligned_rgb)
        if aligned_rgb.ndim != 3 or aligned_rgb.shape[2] != 3:
            raise ValueError(f"aligned_faces[{idx}] must have shape (H, W, 3).")
        if aligned_rgb.dtype != np.uint8:
            aligned_rgb = aligned_rgb.astype(np.uint8)
        if aligned_rgb.shape[:2] != image_size:
            height, width = image_size
            aligned_rgb = cv2.resize(
                aligned_rgb,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )

        aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
        if draw_keypoints:
            kps = np.asarray(keypoints[idx], dtype=np.float32)
            if kps.shape != (5, 2):
                raise ValueError(f"keypoints[{idx}] must have shape (5, 2).")
            aligned_bgr = annotate_img_with_kps(
                aligned_bgr, kps=kps, colors=keypoint_colors
            )

        imgs.append(_add_border(aligned_bgr, image_size, border))

    return _build_mosaic(
        imgs,
        image_size=image_size,
        mosaic_shape=mosaic_shape,
        border=border,
        save_to=save_to,
    )
