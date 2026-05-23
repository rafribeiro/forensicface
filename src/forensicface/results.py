"""Result containers and assembly helpers."""

from __future__ import annotations

import numpy as np


__all__ = ["FaceResult"]


class FaceResult(dict):
    """Face processing result with both mapping and attribute access.

    Existing code can keep using ``ret["bbox"]`` and ``isinstance(ret, dict)``;
    new code can use ``ret.bbox`` for keys that do not collide with dict
    methods such as ``keys`` or ``items``.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __dir__(self):
        return sorted(set(super().__dir__()) | {str(key) for key in self.keys()})


def gender_label(gender) -> str | None:
    if gender is None:
        return None
    if isinstance(gender, str):
        return gender
    return "M" if int(gender) == 1 else "F"


def pose_angles(pose) -> tuple[float | None, float | None, float | None]:
    """Return yaw, pitch, roll from a pose vector stored as pitch, yaw, roll."""
    if pose is None:
        return None, None, None
    return pose[1], pose[0], pose[2]


def build_align_result(
    *,
    aligned_face: np.ndarray,
    bbox: np.ndarray,
    keypoints: np.ndarray,
    aligned_keypoints: np.ndarray,
    det_score: float,
    extended: bool,
    gender=None,
    age=None,
    pose=None,
) -> FaceResult:
    result = FaceResult(
        {
            "aligned_face": aligned_face,
            "bbox": bbox.astype("int"),
            "keypoints": keypoints,
            "aligned_keypoints": aligned_keypoints,
            "det_score": float(det_score),
        }
    )
    if extended:
        result["gender"] = gender_label(gender)
        result["age"] = int(age) if age is not None else None
        result["pose"] = pose.copy() if pose is not None else None
    return result


def build_face_result(
    *,
    aligned_face: np.ndarray,
    bbox: np.ndarray,
    keypoints: np.ndarray,
    det_score: float,
    embeddings,
    fiqa_score,
    models: list[str],
    extended: bool,
    concat_embeddings: bool,
    gender=None,
    age=None,
    pose=None,
) -> FaceResult:
    result = FaceResult({"ipd": np.linalg.norm(keypoints[0] - keypoints[1])})

    if extended:
        yaw, pitch, roll = pose_angles(pose)
        result.update(
            {
                "fiqa_score": fiqa_score,
                "gender": gender_label(gender),
                "age": age,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
            }
        )

    result.update(
        {
            "det_score": det_score,
            "keypoints": keypoints,
            "bbox": bbox.astype("int"),
        }
    )
    if concat_embeddings:
        result["embedding"] = embeddings
    else:
        for model_name, embedding in zip(models, embeddings):
            result[f"embedding_{model_name}"] = embedding

    result["aligned_face"] = aligned_face
    return result


def build_face_result_from_align_result(
    *,
    align_item: dict,
    embeddings,
    fiqa_score,
    models: list[str],
    extended: bool,
    concat_embeddings: bool,
) -> FaceResult:
    return build_face_result(
        aligned_face=align_item["aligned_face"],
        bbox=align_item["bbox"],
        keypoints=align_item["keypoints"],
        det_score=align_item["det_score"],
        embeddings=embeddings,
        fiqa_score=fiqa_score,
        models=models,
        extended=extended,
        concat_embeddings=concat_embeddings,
        gender=align_item.get("gender"),
        age=align_item.get("age"),
        pose=align_item.get("pose"),
    )
