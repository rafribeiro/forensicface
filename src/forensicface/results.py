"""Result containers and assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends import PoseAngles, normalize_pose


__all__ = [
    "AlignedFace",
    "FaceResult",
    "assemble_face_result",
    "build_align_result",
    "build_embedding_result",
    "build_face_result",
    "build_face_result_from_align_result",
]


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


@dataclass
class AlignedFace:
    """Internal normalized representation of an aligned face.

    ``aligned_face`` is RGB because public ``FaceResult`` outputs expose aligned
    faces in RGB order. Recognition helpers may still convert it to BGR at the
    ONNX boundary.
    """

    aligned_face: np.ndarray
    bbox: np.ndarray
    keypoints: np.ndarray
    aligned_keypoints: np.ndarray | None
    det_score: float
    gender: str | int | None = None
    age: int | None = None
    pose: PoseAngles | np.ndarray | None = None

    @classmethod
    def from_align_result(cls, align_item: dict):
        return cls(
            aligned_face=align_item["aligned_face"],
            bbox=align_item["bbox"],
            keypoints=align_item["keypoints"],
            aligned_keypoints=align_item.get("aligned_keypoints"),
            det_score=align_item["det_score"],
            gender=align_item.get("gender"),
            age=align_item.get("age"),
            pose=align_item.get("pose"),
        )


def gender_label(gender) -> str | None:
    if gender is None:
        return None
    if isinstance(gender, str):
        return gender
    return "M" if int(gender) == 1 else "F"


def pose_angles(pose) -> tuple[float | None, float | None, float | None]:
    """Return yaw, pitch, roll from the normalized internal representation."""
    normalized = normalize_pose(pose)
    if normalized is None:
        return None, None, None
    return normalized.yaw, normalized.pitch, normalized.roll


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
    enabled_tasks: frozenset[str] | set[str] | None = None,
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
    if enabled_tasks is None:
        enabled_tasks = {"gender", "age", "pose"} if extended else set()
    if "gender" in enabled_tasks:
        result["gender"] = gender_label(gender)
    if "age" in enabled_tasks:
        result["age"] = int(age) if age is not None else None
    if "pose" in enabled_tasks:
        normalized_pose = normalize_pose(pose)
        result["pose"] = (
            normalized_pose.as_array() if normalized_pose is not None else None
        )
    return result


def build_embedding_result(
    *,
    embeddings,
    fiqa_score,
    models: list[str],
    extended: bool,
    concat_embeddings: bool,
    index: int | None = None,
    enabled_tasks: frozenset[str] | set[str] | None = None,
) -> FaceResult:
    """Build a result for recognition-only APIs that receive aligned faces."""
    result = FaceResult()
    if embeddings is not None and concat_embeddings:
        result["embedding"] = embeddings if index is None else embeddings[index]
    elif embeddings is not None:
        for model_name, embedding in zip(models, embeddings):
            result[f"embedding_{model_name}"] = (
                embedding if index is None else embedding[index]
            )
    if enabled_tasks is None:
        enabled_tasks = {"quality"} if extended else set()
    if "quality" in enabled_tasks:
        if index is None:
            result["fiqa_score"] = fiqa_score
        else:
            result["fiqa_score"] = (
                float(fiqa_score[index]) if fiqa_score is not None else None
            )
    return result


def assemble_face_result(
    *,
    aligned_face: AlignedFace,
    embeddings,
    fiqa_score,
    models: list[str],
    extended: bool,
    concat_embeddings: bool,
    enabled_tasks: frozenset[str] | set[str] | None = None,
) -> FaceResult:
    result = FaceResult(
        {"ipd": np.linalg.norm(aligned_face.keypoints[0] - aligned_face.keypoints[1])}
    )

    if enabled_tasks is None:
        enabled_tasks = (
            {"gender", "age", "pose", "quality", "embedding"}
            if extended
            else {"embedding"}
        )
    if "quality" in enabled_tasks:
        result["fiqa_score"] = fiqa_score
    if "gender" in enabled_tasks:
        result["gender"] = gender_label(aligned_face.gender)
    if "age" in enabled_tasks:
        result["age"] = aligned_face.age
    if "pose" in enabled_tasks:
        yaw, pitch, roll = pose_angles(aligned_face.pose)
        result.update({"yaw": yaw, "pitch": pitch, "roll": roll})

    result.update(
        {
            "det_score": aligned_face.det_score,
            "keypoints": aligned_face.keypoints,
            "bbox": aligned_face.bbox.astype("int"),
        }
    )
    if embeddings is not None and concat_embeddings:
        result["embedding"] = embeddings
    elif embeddings is not None:
        for model_name, embedding in zip(models, embeddings):
            result[f"embedding_{model_name}"] = embedding

    result["aligned_face"] = aligned_face.aligned_face
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
    aligned_keypoints=None,
    enabled_tasks: frozenset[str] | set[str] | None = None,
) -> FaceResult:
    return assemble_face_result(
        aligned_face=AlignedFace(
            aligned_face=aligned_face,
            bbox=bbox,
            keypoints=keypoints,
            aligned_keypoints=aligned_keypoints,
            det_score=det_score,
            gender=gender,
            age=age,
            pose=pose,
        ),
        embeddings=embeddings,
        fiqa_score=fiqa_score,
        models=models,
        extended=extended,
        concat_embeddings=concat_embeddings,
        enabled_tasks=enabled_tasks,
    )


def build_face_result_from_align_result(
    *,
    align_item: dict,
    embeddings,
    fiqa_score,
    models: list[str],
    extended: bool,
    concat_embeddings: bool,
    enabled_tasks: frozenset[str] | set[str] | None = None,
) -> FaceResult:
    return assemble_face_result(
        aligned_face=AlignedFace.from_align_result(align_item),
        embeddings=embeddings,
        fiqa_score=fiqa_score,
        models=models,
        extended=extended,
        concat_embeddings=concat_embeddings,
        enabled_tasks=enabled_tasks,
    )
