"""Quality inference isolated from recognition inference."""

from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np

from .preprocessing import to_ada_input
from .recognition import looks_like_cuda_oom


__all__ = ["QualityRunner", "try_compute_quality_batch"]


def try_compute_quality_batch(
    compute_batch: Callable[[np.ndarray], np.ndarray],
    bgr_aligned_batch: np.ndarray,
) -> np.ndarray:
    """Run quality inference with recursive CUDA-OOM splitting."""
    try:
        return compute_batch(bgr_aligned_batch)
    except Exception as exc:
        size = bgr_aligned_batch.shape[0]
        if size <= 1 or not looks_like_cuda_oom(exc):
            raise
        half = size // 2
        warnings.warn(
            f"CUDA OOM during quality inference with batch_size={size}; "
            f"falling back to {half}. Pass a smaller `batch_size` to "
            "`process_images_batch` to avoid this overhead.",
            stacklevel=2,
        )
        first = try_compute_quality_batch(compute_batch, bgr_aligned_batch[:half])
        second = try_compute_quality_batch(compute_batch, bgr_aligned_batch[half:])
        return np.concatenate([first, second], axis=0)


class QualityRunner:
    """Run a quality component or a legacy CR-FIQA ONNX session."""

    def __init__(
        self,
        *,
        estimator=None,
        legacy_session=None,
        image_size: tuple[int, int] = (112, 112),
    ):
        if estimator is not None and legacy_session is not None:
            raise ValueError("Provide either estimator or legacy_session, not both.")
        self.estimator = estimator
        self.legacy_session = legacy_session
        self.image_size = image_size

    @property
    def enabled(self) -> bool:
        return self.estimator is not None or self.legacy_session is not None

    def compute_one(self, bgr_aligned_face: np.ndarray) -> float | None:
        if self.estimator is not None:
            value = float(self.estimator.score_one(bgr_aligned_face))
            if not np.isfinite(value):
                raise ValueError("Quality estimator returned a non-finite score.")
            return value
        if self.legacy_session is None:
            return None
        values = self._compute_legacy_batch(bgr_aligned_face[None, ...])
        return float(values[0])

    def compute_batch(self, bgr_aligned_batch: np.ndarray) -> np.ndarray | None:
        if self.estimator is not None:
            scores = np.asarray(
                self.estimator.score_batch(bgr_aligned_batch)
            ).reshape(-1)
            return self._validate_scores(scores, len(bgr_aligned_batch))
        if self.legacy_session is None:
            return None
        return self._compute_legacy_batch(bgr_aligned_batch)

    def try_compute_batch(self, bgr_aligned_batch: np.ndarray) -> np.ndarray | None:
        if not self.enabled:
            return None
        return try_compute_quality_batch(self.compute_batch, bgr_aligned_batch)

    def _compute_legacy_batch(self, bgr_aligned_batch: np.ndarray) -> np.ndarray:
        model_input = to_ada_input(bgr_aligned_batch, image_size=self.image_size)
        output = self.legacy_session.run(
            None,
            {self.legacy_session.get_inputs()[0].name: model_input},
        )
        scores = np.asarray(output[-1]).reshape(-1)
        return self._validate_scores(scores, len(bgr_aligned_batch))

    @staticmethod
    def _validate_scores(scores: np.ndarray, expected: int) -> np.ndarray:
        if len(scores) != expected:
            raise ValueError(
                "Quality estimator must return one score per face; "
                f"expected {expected}, received {len(scores)}."
            )
        if not np.isfinite(scores).all():
            raise ValueError("Quality estimator returned non-finite scores.")
        return scores
