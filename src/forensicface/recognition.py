"""Recognition and FIQA inference helpers."""

from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np

from .preprocessing import normalize_aligned_keypoints, to_ada_input


__all__ = [
    "KEYPOINT_RECOGNITION_MODELS",
    "RecognitionRunner",
    "build_keypoint_model_inputs",
    "looks_like_cuda_oom",
    "try_compute_embeddings_batch",
]


KEYPOINT_RECOGNITION_MODELS = {"sepaelv6"}
SEPAELV6_IMAGE_INPUT = "input_images"
SEPAELV6_KEYPOINTS_INPUT = "keypoints"


def build_keypoint_model_inputs(
    *,
    model_name,
    rec_ort,
    img_to_input,
    aligned_keypoints,
    image_size: tuple[int, int] = (112, 112),
    image_input_name: str = SEPAELV6_IMAGE_INPUT,
    keypoints_input_name: str = SEPAELV6_KEYPOINTS_INPUT,
):
    input_names = [input_info.name for input_info in rec_ort.get_inputs()]
    required_input_names = {image_input_name, keypoints_input_name}
    missing_input_names = required_input_names.difference(input_names)
    if missing_input_names:
        raise ValueError(
            f"Model '{model_name}' requires ONNX inputs "
            f"{sorted(required_input_names)}, but the loaded session has "
            f"{input_names}. Missing: {sorted(missing_input_names)}."
        )

    return {
        image_input_name: img_to_input,
        keypoints_input_name: normalize_aligned_keypoints(
            aligned_keypoints,
            model_name=model_name,
            image_size=image_size,
        ),
    }


def looks_like_cuda_oom(exc: BaseException) -> bool:
    """Best-effort match for CUDA out-of-memory errors raised by ONNX Runtime."""
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cudaerrormemoryallocation" in msg
        or "cuda" in msg and "oom" in msg
        or "alloc" in msg and "memory" in msg
    )


def merge_batch_outputs(emb_a, emb_b):
    if isinstance(emb_a, np.ndarray):
        return np.concatenate([emb_a, emb_b], axis=0)
    return [np.concatenate([a, b], axis=0) for a, b in zip(emb_a, emb_b)]


def try_compute_embeddings_batch(
    compute_batch: Callable,
    bgr_aligned_batch,
    aligned_keypoints_batch=None,
):
    """Call a batch embedding function with recursive CUDA OOM retry."""
    try:
        return compute_batch(
            bgr_aligned_batch,
            aligned_keypoints_batch=aligned_keypoints_batch,
        )
    except Exception as exc:
        n = bgr_aligned_batch.shape[0]
        if n <= 1 or not looks_like_cuda_oom(exc):
            raise
        half = n // 2
        warnings.warn(
            f"CUDA OOM with batch_size={n}; falling back to {half}. "
            f"Pass a smaller `batch_size` to `process_images_batch` "
            f"to avoid this overhead.",
            stacklevel=2,
        )
        kps_a = (
            aligned_keypoints_batch[:half]
            if aligned_keypoints_batch is not None else None
        )
        kps_b = (
            aligned_keypoints_batch[half:]
            if aligned_keypoints_batch is not None else None
        )
        emb_a, fiqa_a = try_compute_embeddings_batch(
            compute_batch,
            bgr_aligned_batch[:half],
            aligned_keypoints_batch=kps_a,
        )
        emb_b, fiqa_b = try_compute_embeddings_batch(
            compute_batch,
            bgr_aligned_batch[half:],
            aligned_keypoints_batch=kps_b,
        )
        embeddings = merge_batch_outputs(emb_a, emb_b)
        fiqa_scores = None
        if fiqa_a is not None and fiqa_b is not None:
            fiqa_scores = np.concatenate([fiqa_a, fiqa_b], axis=0)
        return embeddings, fiqa_scores


class RecognitionRunner:
    """Runs recognition and optional FIQA sessions for aligned face crops."""

    def __init__(
        self,
        *,
        models: list[str],
        rec_inference_sessions: list,
        ort_fiqa=None,
        extended: bool,
        concat_embeddings: bool,
        image_size: tuple[int, int] = (112, 112),
        keypoint_recognition_models: set[str] | None = None,
        image_input_name: str = SEPAELV6_IMAGE_INPUT,
        keypoints_input_name: str = SEPAELV6_KEYPOINTS_INPUT,
    ):
        self.models = models
        self.rec_inference_sessions = rec_inference_sessions
        self.ort_fiqa = ort_fiqa
        self.extended = extended
        self.concat_embeddings = concat_embeddings
        self.image_size = image_size
        self.keypoint_recognition_models = (
            keypoint_recognition_models or KEYPOINT_RECOGNITION_MODELS
        )
        self.image_input_name = image_input_name
        self.keypoints_input_name = keypoints_input_name

    def to_input(self, aligned_bgr_img):
        return to_ada_input(aligned_bgr_img, image_size=self.image_size)

    def build_keypoint_model_inputs(
        self, model_name, rec_ort, img_to_input, aligned_keypoints
    ):
        return build_keypoint_model_inputs(
            model_name=model_name,
            rec_ort=rec_ort,
            img_to_input=img_to_input,
            aligned_keypoints=aligned_keypoints,
            image_size=self.image_size,
            image_input_name=self.image_input_name,
            keypoints_input_name=self.keypoints_input_name,
        )

    def compute_one(self, bgr_aligned_face, aligned_keypoints=None):
        """Compute embeddings and optional FIQA score for one aligned face."""
        img_to_input = self.to_input(bgr_aligned_face)
        embeddings = []
        for model_name, rec_ort in zip(self.models, self.rec_inference_sessions):
            if model_name in self.keypoint_recognition_models:
                model_inputs = self.build_keypoint_model_inputs(
                    model_name=model_name,
                    rec_ort=rec_ort,
                    img_to_input=img_to_input,
                    aligned_keypoints=aligned_keypoints,
                )
            else:
                model_inputs = {rec_ort.get_inputs()[0].name: img_to_input}
            model_output = rec_ort.run(None, model_inputs)
            if len(model_output) == 2:
                embedding = model_output[0].flatten() * model_output[1].flatten()[0]
            else:
                embedding = model_output[0].flatten()
            embeddings.append(embedding)

        fiqa_score = None
        if self.extended:
            _, fiqa_score = self.ort_fiqa.run(
                None, {self.ort_fiqa.get_inputs()[0].name: img_to_input}
            )

        return (
            np.concatenate(embeddings) if self.concat_embeddings else embeddings,
            fiqa_score[0][0] if fiqa_score is not None else None,
        )

    def compute_batch(
        self,
        bgr_aligned_batch: np.ndarray,
        aligned_keypoints_batch: np.ndarray = None,
    ):
        """Compute embeddings and optional FIQA scores for aligned face crops."""
        assert (
            bgr_aligned_batch.ndim == 4
            and bgr_aligned_batch.shape[1:] == (*self.image_size, 3)
        ), (
            f"Expected shape (N, {self.image_size[0]}, {self.image_size[1]}, 3); "
            f"got {bgr_aligned_batch.shape}."
        )

        batch_input = self.to_input(bgr_aligned_batch)

        keypoints_input = None
        if aligned_keypoints_batch is not None:
            kp = np.asarray(aligned_keypoints_batch, dtype=np.float32)
            if kp.ndim != 3 or kp.shape[1:] != (5, 2):
                raise ValueError(
                    "aligned_keypoints_batch must have shape (N, 5, 2); "
                    f"received {kp.shape}."
                )
            if kp.shape[0] != bgr_aligned_batch.shape[0]:
                raise ValueError(
                    f"aligned_keypoints_batch has N={kp.shape[0]} but "
                    f"bgr_aligned_batch has N={bgr_aligned_batch.shape[0]}."
                )
            keypoints_input = kp

        embeddings_per_model = []
        for rec_ort, model_name in zip(self.rec_inference_sessions, self.models):
            if model_name in self.keypoint_recognition_models:
                if keypoints_input is None:
                    raise ValueError(
                        f"Model '{model_name}' requires aligned_keypoints_batch "
                        "(5-point keypoints in the 112x112 coordinate system)."
                    )
                model_inputs = self.build_keypoint_model_inputs(
                    model_name=model_name,
                    rec_ort=rec_ort,
                    img_to_input=batch_input,
                    aligned_keypoints=keypoints_input,
                )
            else:
                model_inputs = {rec_ort.get_inputs()[0].name: batch_input}
            model_output = rec_ort.run(None, model_inputs)
            if len(model_output) == 2:
                emb = model_output[0] * model_output[1]
            else:
                emb = model_output[0]
            embeddings_per_model.append(emb)

        if self.concat_embeddings:
            embeddings = np.concatenate(embeddings_per_model, axis=1)
        else:
            embeddings = embeddings_per_model

        fiqa_scores = None
        if self.extended and self.ort_fiqa is not None:
            fiqa_output = self.ort_fiqa.run(
                None, {self.ort_fiqa.get_inputs()[0].name: batch_input}
            )
            fiqa_scores = np.asarray(fiqa_output[-1]).reshape(-1)

        return embeddings, fiqa_scores

    def try_compute_batch(self, bgr_aligned_batch, aligned_keypoints_batch=None):
        return try_compute_embeddings_batch(
            self.compute_batch,
            bgr_aligned_batch,
            aligned_keypoints_batch=aligned_keypoints_batch,
        )
