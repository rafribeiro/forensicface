"""Contracts and configuration objects for extensible face-model components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import cv2
import numpy as np

from .backends import FaceBackend, FaceData, normalize_pose
from .insightface import face_align
from .utils import transform_keypoints


__all__ = [
    "DEFAULT",
    "ComponentBackend",
    "ComponentMetadata",
    "DefaultSelection",
    "EmbeddingEstimator",
    "FaceContext",
    "FaceEstimator",
    "FaceDetector",
    "ModelSpec",
    "QualityEstimator",
]


class DefaultSelection:
    """Sentinel representing an omitted task selector."""

    def __repr__(self) -> str:
        return "DEFAULT"


DEFAULT = DefaultSelection()


@dataclass(frozen=True, init=False)
class ModelSpec:
    """Select a built-in alias with a path and model-specific options."""

    alias: str
    path: Path | None
    options: Mapping[str, Any]

    def __init__(
        self,
        alias: str,
        *,
        path: str | Path | None = None,
        **options: Any,
    ):
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("ModelSpec alias must be a non-empty string.")
        object.__setattr__(self, "alias", alias.strip())
        object.__setattr__(
            self,
            "path",
            Path(path).expanduser() if path is not None else None,
        )
        object.__setattr__(self, "options", MappingProxyType(dict(options)))


@dataclass(frozen=True)
class ComponentMetadata:
    component_id: str
    tasks: frozenset[str]
    implementation: str
    runtime: str
    model_path: Path | None
    input_space: str
    supports_batch: bool
    providers: tuple[str, ...] = ()


@dataclass
class FaceContext:
    original_bgr: np.ndarray
    face: FaceData
    aligned_bgr: np.ndarray | None = None
    aligned_keypoints: np.ndarray | None = None
    alignment_transform: np.ndarray | None = None


@runtime_checkable
class FaceDetector(Protocol):
    metadata: ComponentMetadata

    def detect(self, bgr_img: np.ndarray) -> list[FaceData]: ...


@runtime_checkable
class FaceEstimator(Protocol):
    metadata: ComponentMetadata
    capabilities: frozenset[str]

    def estimate(self, context: FaceContext) -> Mapping[str, Any]: ...


@runtime_checkable
class QualityEstimator(Protocol):
    metadata: ComponentMetadata

    def score_one(self, aligned_bgr: np.ndarray) -> float: ...

    def score_batch(self, aligned_bgr_batch: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class EmbeddingEstimator(Protocol):
    metadata: ComponentMetadata
    name: str
    requires_aligned_keypoints: bool

    def embed_one(
        self,
        aligned_bgr: np.ndarray,
        *,
        aligned_keypoints: np.ndarray | None = None,
    ) -> np.ndarray: ...

    def embed_batch(
        self,
        aligned_bgr_batch: np.ndarray,
        *,
        aligned_keypoints_batch: np.ndarray | None = None,
    ) -> np.ndarray: ...


class ComponentBackend(FaceBackend):
    """Compose a detector, standard five-point alignment, and estimators."""

    name = "components"

    def __init__(
        self,
        *,
        detector: FaceDetector,
        estimators: list[FaceEstimator],
        enabled_tasks: set[str],
    ):
        self.detector = detector
        # Joint estimators (for example gender+age) may be selected for more
        # than one task. Deduplicate by identity without requiring custom
        # component objects to be hashable.
        self.estimators = []
        seen_ids: set[int] = set()
        for estimator in estimators:
            if id(estimator) not in seen_ids:
                self.estimators.append(estimator)
                seen_ids.add(id(estimator))
        self.enabled_tasks = frozenset(enabled_tasks)

        # Compatibility and runtime-summary seams used by existing code.
        self.det_model = getattr(detector, "model", detector)
        self.landmark_model = next(
            (
                getattr(estimator, "model", estimator)
                for estimator in self.estimators
                if "pose" in estimator.capabilities
            ),
            None,
        )
        self.genderage_model = next(
            (
                getattr(estimator, "model", estimator)
                for estimator in self.estimators
                if estimator.capabilities & {"gender", "age"}
            ),
            None,
        )

    def detect_faces(self, bgr_img: np.ndarray) -> list[FaceData]:
        faces = self.detector.detect(bgr_img)
        for face in faces:
            self._validate_face(face)
            transform = face_align.estimate_norm(face.kps)
            aligned_bgr = cv2.warpAffine(
                bgr_img, transform, (112, 112), borderValue=0.0
            )
            aligned_keypoints = transform_keypoints(face.kps, transform)
            face.aligned_bgr = aligned_bgr
            face.aligned_keypoints = aligned_keypoints
            face.alignment_transform = transform
            context = FaceContext(
                original_bgr=bgr_img,
                face=face,
                aligned_bgr=aligned_bgr,
                aligned_keypoints=aligned_keypoints,
                alignment_transform=transform,
            )
            for estimator in self.estimators:
                values = estimator.estimate(context)
                if not isinstance(values, Mapping):
                    raise TypeError(
                        f"Estimator '{estimator.metadata.component_id}' must return "
                        "a mapping of task names to values."
                    )
                for task, value in values.items():
                    if task not in estimator.capabilities:
                        raise ValueError(
                            f"Estimator '{estimator.metadata.component_id}' returned "
                            f"undeclared task '{task}'."
                        )
                    if task in self.enabled_tasks:
                        normalized_value = normalize_pose(value) if task == "pose" else value
                        setattr(face, task, normalized_value)
        return faces

    @staticmethod
    def _validate_face(face) -> None:
        missing = [
            name for name in ("bbox", "kps", "det_score") if not hasattr(face, name)
        ]
        if missing:
            raise ValueError(
                f"Detector output is missing required fields: {sorted(missing)}."
            )
        bbox = np.asarray(face.bbox)
        keypoints = np.asarray(face.kps)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError(
                f"Detector bbox must be finite with shape (4,); got {bbox.shape}."
            )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(
                "Detector bbox must have x2 > x1 and y2 > y1; "
                f"received {bbox.tolist()}."
            )
        if keypoints.shape != (5, 2) or not np.isfinite(keypoints).all():
            raise ValueError(
                "Detector keypoints must be finite with shape (5, 2); "
                f"got {keypoints.shape}."
            )
        if not np.isfinite(float(face.det_score)):
            raise ValueError("Detector det_score must be finite.")

    def norm_crop(self, bgr_img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        return face_align.norm_crop(bgr_img, keypoints)

    def estimate_norm(self, keypoints: np.ndarray) -> np.ndarray:
        return face_align.estimate_norm(keypoints)
