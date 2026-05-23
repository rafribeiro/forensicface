"""Geometry helpers used by image and video workflows."""

from __future__ import annotations

import numpy as np


def select_best_face(img_shape, faces, criterion: str = "size"):
    """Select one face by bbox size or centrality."""
    if criterion not in {"centrality", "size"}:
        raise ValueError("criterion must be either 'centrality' or 'size'.")
    if faces is None or len(faces) == 0:
        raise ValueError("faces must contain at least one face.")

    boxes = [face.bbox.astype("int").flatten() for face in faces]
    if criterion == "centrality":
        img_center = np.array([img_shape[0] // 2, img_shape[1] // 2])
        scores = [
            np.linalg.norm(
                img_center - np.array([(box[0] + box[2]) // 2, (box[1] + box[3]) // 2])
            )
            for box in boxes
        ]
        return faces[scores.index(min(scores))]

    scores = [abs((box[2] - box[0]) * (box[3] - box[1])) for box in boxes]
    return faces[scores.index(max(scores))]


def extend_bbox(bbox, frame_shape, margin_factor: float) -> list[int]:
    """Return bbox coordinates expanded by a margin and clipped to the frame."""
    start_x, start_y, end_x, end_y = bbox.astype("int")
    h, w = frame_shape[:2]
    out_width = (end_x - start_x) * margin_factor
    out_height = (end_y - start_y) * margin_factor

    start_x_out = int((start_x + end_x) / 2 - out_width / 2)
    end_x_out = int((start_x + end_x) / 2 + out_width / 2)
    start_y_out = int((start_y + end_y) / 2 - out_height / 2)
    end_y_out = int((start_y + end_y) / 2 + out_height / 2)

    return [
        max(start_x_out, 0),
        max(start_y_out, 0),
        min(end_x_out, int(w)),
        min(end_y_out, int(h)),
    ]

