"""Orchestration for embedding and quality inference on aligned crops."""

from __future__ import annotations

import numpy as np

from .quality import QualityRunner
from .recognition import RecognitionRunner


class AlignedFaceRunner:
    """Coordinate independent recognition and quality runners."""

    def __init__(
        self,
        *,
        recognition: RecognitionRunner,
        quality: QualityRunner,
    ):
        self.recognition = recognition
        self.quality = quality

    def compute_one(self, bgr_aligned_face, aligned_keypoints=None):
        embeddings = self.recognition.compute_one(
            bgr_aligned_face,
            aligned_keypoints=aligned_keypoints,
        )
        quality_score = self.quality.compute_one(bgr_aligned_face)
        return embeddings, quality_score

    def compute_batch(
        self,
        bgr_aligned_batch: np.ndarray,
        aligned_keypoints_batch: np.ndarray | None = None,
    ):
        embeddings = self.recognition.compute_batch(
            bgr_aligned_batch,
            aligned_keypoints_batch=aligned_keypoints_batch,
        )
        quality_scores = self.quality.compute_batch(bgr_aligned_batch)
        return embeddings, quality_scores

    def try_compute_batch(
        self,
        bgr_aligned_batch: np.ndarray,
        aligned_keypoints_batch: np.ndarray | None = None,
    ):
        # Retry independently so a quality OOM does not rerun recognition and
        # an embedding OOM does not rerun quality inference.
        embeddings = self.recognition.try_compute_batch(
            bgr_aligned_batch,
            aligned_keypoints_batch=aligned_keypoints_batch,
        )
        quality_scores = self.quality.try_compute_batch(bgr_aligned_batch)
        return embeddings, quality_scores
