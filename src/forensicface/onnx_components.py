"""Built-in ONNX component adapters.

Each adapter owns its preprocessing. In particular, face estimators receive
the original image and bounding box and therefore remain free to use different
native input sizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime

from .backends import FaceData, normalize_pose
from .components import ComponentMetadata, FaceContext
from .insightface.attribute import AttributeONNX
from .insightface.landmark import LandmarkONNX
from .insightface.scrfd import SCRFD
from .preprocessing import normalize_aligned_keypoints, to_ada_input


class _MutableFace(dict):
    """Small InsightFace-compatible record used by wrapped estimators."""

    def __init__(self, **values):
        super().__init__(values)
        self.__dict__.update(values)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        setattr(self, key, value)


def _providers_tuple(session) -> tuple[str, ...]:
    try:
        return tuple(session.get_providers())
    except Exception:
        return ()


class SCRFDDetector:
    capabilities = frozenset({"detection"})

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers,
        ctx_id: int,
        det_size: int | tuple[int, int] = 320,
        det_thresh: float = 0.5,
        nms_thresh: float = 0.4,
    ):
        self.model_path = Path(model_path)
        self.model = SCRFD(model_file=str(self.model_path), providers=providers)
        input_size = (det_size, det_size) if isinstance(det_size, int) else tuple(det_size)
        self.model.prepare(
            ctx_id,
            input_size=input_size,
            det_thresh=det_thresh,
            nms_thresh=nms_thresh,
        )
        self.input_size = tuple(self.model.input_size)
        self.input_size_summary = self.input_size
        self.metadata = ComponentMetadata(
            component_id="scrfd",
            tasks=self.capabilities,
            implementation=type(self).__name__,
            runtime="onnxruntime",
            model_path=self.model_path,
            input_space="BGR image; aspect-preserving resize and padding owned by SCRFD",
            supports_batch=False,
            providers=_providers_tuple(self.model.session),
        )

    def detect(self, bgr_img: np.ndarray) -> list[FaceData]:
        bboxes, keypoints = self.model.detect(bgr_img, max_num=0, metric="default")
        if bboxes.shape[0] == 0 or keypoints is None:
            return []
        return [
            FaceData(
                # Preserve detector precision for downstream face crops. Public
                # result builders retain the historical integer bbox schema.
                bbox=np.asarray(bboxes[index, :4], dtype=np.float32),
                kps=np.asarray(keypoints[index], dtype=np.float32),
                det_score=float(bboxes[index, 4]),
            )
            for index in range(bboxes.shape[0])
        ]


class CenterFaceDetector:
    """ONNX Runtime adapter for the corrected dynamic CenterFace export."""

    capabilities = frozenset({"detection"})

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.3,
        input_multiple: int = 32,
    ):
        if input_multiple <= 0:
            raise ValueError("CenterFace input_multiple must be positive.")
        self.model_path = Path(model_path)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_multiple = int(input_multiple)
        self.input_size = None
        self.input_size_summary = f"dynamic (multiple of {self.input_multiple})"
        self.session = onnxruntime.InferenceSession(
            str(self.model_path), providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        output_names = [output.name for output in self.session.get_outputs()]
        required = ["heatmap", "scale", "offset", "landmarks"]
        if all(name in output_names for name in required):
            self.output_names = required
        elif len(output_names) == 4:
            # Supports converted artifacts made before semantic names were
            # introduced, while preserving the reference output order.
            self.output_names = output_names
        else:
            raise ValueError(
                "CenterFace expects four outputs: heatmap, scale, offset, landmarks; "
                f"received {output_names}."
            )
        self.metadata = ComponentMetadata(
            component_id="centerface",
            tasks=self.capabilities,
            implementation=type(self).__name__,
            runtime="onnxruntime",
            model_path=self.model_path,
            input_space=(
                "BGR image resized independently to the next input_multiple and "
                "converted to RGB NCHW"
            ),
            supports_batch=False,
            providers=_providers_tuple(self.session),
        )

    def detect(self, bgr_img: np.ndarray) -> list[FaceData]:
        if bgr_img.ndim != 3 or bgr_img.shape[2] != 3:
            raise ValueError(f"CenterFace expects a BGR HxWx3 image; got {bgr_img.shape}.")
        height, width = bgr_img.shape[:2]
        input_height = int(np.ceil(height / self.input_multiple) * self.input_multiple)
        input_width = int(np.ceil(width / self.input_multiple) * self.input_multiple)
        blob = cv2.dnn.blobFromImage(
            bgr_img,
            scalefactor=1.0,
            size=(input_width, input_height),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        heatmap, scale, offset, landmarks = self.session.run(
            self.output_names, {self.input_name: blob}
        )
        boxes, keypoints, scores = self._decode(
            heatmap[0, 0], scale[0], offset[0], landmarks[0]
        )
        if not boxes:
            return []
        keep = self._nms(np.asarray(boxes, dtype=np.float32), np.asarray(scores))
        scale_x = width / input_width
        scale_y = height / input_height
        faces: list[FaceData] = []
        for index in keep:
            bbox = np.asarray(boxes[index], dtype=np.float32)
            bbox[[0, 2]] *= scale_x
            bbox[[1, 3]] *= scale_y
            kps = np.asarray(keypoints[index], dtype=np.float32)
            kps[:, 0] *= scale_x
            kps[:, 1] *= scale_y
            faces.append(
                FaceData(
                    bbox=bbox,
                    kps=kps,
                    det_score=float(scores[index]),
                )
            )
        return faces

    def _decode(self, heatmap, scale, offset, landmarks):
        ys, xs = np.where(heatmap > self.score_threshold)
        boxes: list[list[float]] = []
        keypoints: list[np.ndarray] = []
        scores: list[float] = []
        for y, x in zip(ys, xs):
            box_height = float(np.exp(scale[0, y, x]) * 4.0)
            box_width = float(np.exp(scale[1, y, x]) * 4.0)
            center_y = float((y + offset[0, y, x] + 0.5) * 4.0)
            center_x = float((x + offset[1, y, x] + 0.5) * 4.0)
            x1 = max(0.0, center_x - box_width / 2.0)
            y1 = max(0.0, center_y - box_height / 2.0)
            boxes.append([x1, y1, x1 + box_width, y1 + box_height])
            points = np.empty((5, 2), dtype=np.float32)
            for point_index in range(5):
                points[point_index, 0] = (
                    landmarks[point_index * 2 + 1, y, x] * box_width + x1
                )
                points[point_index, 1] = (
                    landmarks[point_index * 2, y, x] * box_height + y1
                )
            keypoints.append(points)
            scores.append(float(heatmap[y, x]))
        return boxes, keypoints, scores

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        if boxes.size == 0:
            return []
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size:
            current = int(order[0])
            keep.append(current)
            xx1 = np.maximum(x1[current], x1[order[1:]])
            yy1 = np.maximum(y1[current], y1[order[1:]])
            xx2 = np.minimum(x2[current], x2[order[1:]])
            yy2 = np.minimum(y2[current], y2[order[1:]])
            widths = np.maximum(0.0, xx2 - xx1 + 1)
            heights = np.maximum(0.0, yy2 - yy1 + 1)
            overlap = widths * heights
            iou = overlap / (areas[current] + areas[order[1:]] - overlap)
            order = order[np.where(iou <= self.nms_threshold)[0] + 1]
        return keep


class InsightFacePoseEstimator:
    capabilities = frozenset({"pose"})

    def __init__(self, model_path: str | Path, *, providers, ctx_id: int):
        self.model_path = Path(model_path)
        self.model = LandmarkONNX(model_file=str(self.model_path), providers=providers)
        self.model.prepare(ctx_id)
        self.metadata = ComponentMetadata(
            component_id="insightface-3d68",
            tasks=self.capabilities,
            implementation=type(self).__name__,
            runtime="onnxruntime",
            model_path=self.model_path,
            input_space=f"BGR face crop; adapter-owned native size {self.model.input_size}",
            supports_batch=False,
            providers=_providers_tuple(self.model.session),
        )

    def estimate(self, context: FaceContext) -> dict[str, Any]:
        record = _MutableFace(bbox=context.face.bbox, kps=context.face.kps)
        self.model.get(context.original_bgr, record)
        return {"pose": normalize_pose(record.get("pose"))}


class InsightFaceGenderAgeEstimator:
    capabilities = frozenset({"gender", "age"})

    def __init__(self, model_path: str | Path, *, providers, ctx_id: int):
        self.model_path = Path(model_path)
        self.model = AttributeONNX(model_file=str(self.model_path), providers=providers)
        self.model.prepare(ctx_id)
        self.metadata = ComponentMetadata(
            component_id="insightface-genderage",
            tasks=self.capabilities,
            implementation=type(self).__name__,
            runtime="onnxruntime",
            model_path=self.model_path,
            input_space=f"BGR face crop; adapter-owned native size {self.model.input_size}",
            supports_batch=False,
            providers=_providers_tuple(self.model.session),
        )

    def estimate(self, context: FaceContext) -> dict[str, Any]:
        record = _MutableFace(bbox=context.face.bbox, kps=context.face.kps)
        gender, age = self.model.get(context.original_bgr, record)
        return {"gender": gender, "age": age}


class CRFIQAQualityEstimator:
    capabilities = frozenset({"quality"})

    def __init__(self, model_path: str | Path, *, providers):
        self.model_path = Path(model_path)
        self.session = onnxruntime.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.metadata = ComponentMetadata(
            component_id="cr-fiqa",
            tasks=self.capabilities,
            implementation=type(self).__name__,
            runtime="onnxruntime",
            model_path=self.model_path,
            input_space="aligned BGR 112x112; AdaFace normalization",
            supports_batch=True,
            providers=_providers_tuple(self.session),
        )

    def score_one(self, aligned_bgr: np.ndarray) -> float:
        return float(self.score_batch(aligned_bgr[None, ...])[0])

    def score_batch(self, aligned_bgr_batch: np.ndarray) -> np.ndarray:
        model_input = to_ada_input(aligned_bgr_batch, image_size=(112, 112))
        output = self.session.run(None, {self.input_name: model_input})
        return np.asarray(output[-1]).reshape(-1)


class ONNXEmbeddingEstimator:
    capabilities = frozenset({"embedding"})

    def __init__(self, name: str, model_path: str | Path, *, providers):
        self.name = name
        self.model_path = Path(model_path)
        self.session = onnxruntime.InferenceSession(str(self.model_path), providers=providers)
        self.requires_aligned_keypoints = name == "sepaelv6"
        self.metadata = ComponentMetadata(
            component_id=name,
            tasks=self.capabilities,
            implementation=type(self).__name__,
            runtime="onnxruntime",
            model_path=self.model_path,
            input_space="aligned BGR 112x112; model-specific ONNX inputs",
            supports_batch=True,
            providers=_providers_tuple(self.session),
        )

    def embed_one(self, aligned_bgr, *, aligned_keypoints=None) -> np.ndarray:
        return self.embed_batch(
            aligned_bgr[None, ...],
            aligned_keypoints_batch=(
                None if aligned_keypoints is None else aligned_keypoints[None, ...]
            ),
        )[0]

    def embed_batch(self, aligned_bgr_batch, *, aligned_keypoints_batch=None):
        model_input = to_ada_input(aligned_bgr_batch, image_size=(112, 112))
        if self.requires_aligned_keypoints:
            if aligned_keypoints_batch is None:
                raise ValueError(
                    f"Model '{self.name}' requires aligned five-point keypoints."
                )
            inputs = {input_info.name for input_info in self.session.get_inputs()}
            required = {"input_images", "keypoints"}
            missing = required - inputs
            if missing:
                raise ValueError(
                    f"Model '{self.name}' is missing ONNX inputs {sorted(missing)}."
                )
            feed = {
                "input_images": model_input,
                "keypoints": normalize_aligned_keypoints(
                    aligned_keypoints_batch,
                    model_name=self.name,
                    image_size=(112, 112),
                ),
            }
        else:
            feed = {self.session.get_inputs()[0].name: model_input}
        outputs = self.session.run(None, feed)
        embedding = outputs[0]
        if len(outputs) == 2:
            embedding = embedding * outputs[1]
        return np.asarray(embedding)
