from abc import ABC, abstractmethod
from dataclasses import dataclass
import glob
import os.path as osp

import numpy as np

from .insightface import face_align
from .insightface.attribute import AttributeONNX
from .insightface.landmark import LandmarkONNX
from .insightface.scrfd import SCRFD


@dataclass
class FaceData:
    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    gender: int | None = None
    age: int | None = None
    pose: np.ndarray | None = None


class FaceBackend(ABC):
    @abstractmethod
    def detect_faces(self, bgr_img: np.ndarray) -> list[FaceData]:
        raise NotImplementedError

    @abstractmethod
    def norm_crop(self, bgr_img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def estimate_norm(self, keypoints: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ONNXOnlyBackend(FaceBackend):
    """ONNX-only face pipeline using insightface model_zoo components.

    Attribution:
        This implementation is adapted from InsightFace's MIT-licensed code,
        primarily:
        - ``insightface.app.face_analysis.FaceAnalysis`` (pipeline pattern)
        - ``insightface.model_zoo.scrfd.SCRFD`` (detector)
        - ``insightface.utils.face_align`` (alignment)
    """

    def __init__(
        self,
        model_name: str,
        allowed_modules: list[str],
        providers: list[str] | list[tuple[str, dict]],
        ctx_id: int,
        det_size: tuple[int, int],
        det_thresh: float,
        models_root: str,
    ):
        self.name = "onnx"
        model_dir = osp.join(models_root, model_name)
        if not osp.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. "
                f"Download models into {models_root} or provide a valid model name."
            )

        onnx_files = sorted(glob.glob(osp.join(model_dir, "*.onnx")))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX model files found under {model_dir}")

        self.det_model = None
        self.landmark_model = None
        self.genderage_model = None
        self._detection_file = None
        wants_landmark = any(mod.startswith("landmark") for mod in allowed_modules)
        wants_landmark_3d68 = "landmark_3d_68" in allowed_modules
        for onnx_file in onnx_files:
            try:
                candidate = SCRFD(model_file=onnx_file, providers=providers)
            except Exception:
                candidate = None

            if candidate is not None and candidate.taskname == "detection":
                self.det_model = candidate
                self._detection_file = onnx_file
                continue

            low_name = osp.basename(onnx_file).lower()
            is_landmark_candidate = (
                (wants_landmark_3d68 and "3d68" in low_name)
                or (
                    not wants_landmark_3d68
                    and (
                        "3d68" in low_name
                        or "2d106" in low_name
                        or "landmark" in low_name
                    )
                )
            )
            if (
                self.landmark_model is None
                and wants_landmark
                and is_landmark_candidate
            ):
                try:
                    self.landmark_model = LandmarkONNX(
                        model_file=onnx_file,
                        providers=providers,
                    )
                    continue
                except Exception:
                    pass

            if (
                self.genderage_model is None
                and "genderage" in allowed_modules
                and "genderage" in low_name
            ):
                try:
                    self.genderage_model = AttributeONNX(
                        model_file=onnx_file,
                        providers=providers,
                    )
                    continue
                except Exception:
                    pass

        if self.det_model is None:
            raise RuntimeError(
                f"Could not load a SCRFD detection model from {model_dir}."
            )

        self.det_model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
        if self.landmark_model is not None:
            self.landmark_model.prepare(ctx_id)
        if self.genderage_model is not None:
            self.genderage_model.prepare(ctx_id)

    def detect_faces(self, bgr_img: np.ndarray) -> list[FaceData]:
        bboxes, kpss = self.det_model.detect(bgr_img, max_num=0, metric="default")
        if bboxes.shape[0] == 0:
            return []

        faces_data: list[FaceData] = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None if kpss is None else kpss[i]
            if kps is None:
                # ONNX pipeline currently requires detector keypoints for alignment.
                continue

            face = _Face(bbox=bbox, kps=kps, det_score=float(det_score))
            if self.landmark_model is not None:
                self.landmark_model.get(bgr_img, face)
            if self.genderage_model is not None:
                self.genderage_model.get(bgr_img, face)

            faces_data.append(
                FaceData(
                    bbox=bbox.astype("int"),
                    kps=kps,
                    det_score=float(det_score),
                    gender=face.get("gender"),
                    age=face.get("age"),
                    pose=face.get("pose"),
                )
            )
        return faces_data

    def norm_crop(self, bgr_img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        return face_align.norm_crop(bgr_img, keypoints)

    def estimate_norm(self, keypoints: np.ndarray) -> np.ndarray:
        return face_align.estimate_norm(keypoints)


class _Face(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None


def create_backend(
    backend_name: str,
    model_name: str,
    allowed_modules: list[str],
    providers: list[str] | list[tuple[str, dict]],
    ctx_id: int,
    det_size: tuple[int, int],
    det_thresh: float,
    models_root: str,
) -> FaceBackend:
    if backend_name == "onnx":
        return ONNXOnlyBackend(
            model_name=model_name,
            allowed_modules=allowed_modules,
            providers=providers,
            ctx_id=ctx_id,
            det_size=det_size,
            det_thresh=det_thresh,
            models_root=models_root,
        )
    raise ValueError(
        f"Unknown backend_name '{backend_name}'. Supported values: 'onnx'."
    )
