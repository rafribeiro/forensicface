"""Batch image processing workflow."""

from __future__ import annotations

import cv2
import numpy as np

from .results import build_face_result_from_align_result


__all__ = ["process_images_batch"]


def _chunked(items, batch_size: int):
    for chunk_start in range(0, len(items), batch_size):
        yield items[chunk_start : chunk_start + batch_size]


def _needs_aligned_keypoints(processor) -> bool:
    estimators = getattr(processor, "embedding_estimators", None)
    if estimators is not None:
        return any(
            estimator.requires_aligned_keypoints for estimator in estimators
        )
    return bool(set(processor.models) & processor.KEYPOINT_RECOGNITION_MODELS)


def _aligned_rgb_items_to_bgr_batch(items) -> np.ndarray:
    return np.stack(
        [
            cv2.cvtColor(item["aligned_face"], cv2.COLOR_RGB2BGR)
            for item in items
        ],
        axis=0,
    )


def _aligned_keypoints_batch(items) -> np.ndarray:
    return np.stack([item["aligned_keypoints"] for item in items], axis=0)


def _embedding_at(embeddings, index: int, concat_embeddings: bool):
    if embeddings is None:
        return None
    if concat_embeddings:
        return embeddings[index]
    return [per_model[index] for per_model in embeddings]


def _fiqa_at(fiqa_scores, index: int):
    return float(fiqa_scores[index]) if fiqa_scores is not None else None


def process_images_batch(
    processor,
    imgpaths: list[str | np.ndarray],
    *,
    single_face: bool = True,
    select_single_face_by: str = "size",
    batch_size: int = 16,
) -> list:
    """Run detection/alignment per image, then batch recognition/FIQA by crop."""
    imgpaths = list(imgpaths)
    if not imgpaths:
        return []

    if single_face:
        return _process_single_face_images_batch(
            processor,
            imgpaths,
            select_single_face_by=select_single_face_by,
            batch_size=batch_size,
        )

    return _process_multi_face_images_batch(
        processor,
        imgpaths,
        select_single_face_by=select_single_face_by,
        batch_size=batch_size,
    )


def _process_single_face_images_batch(
    processor,
    imgpaths,
    *,
    select_single_face_by: str,
    batch_size: int,
):
    aligned_items = [
        processor.detect_and_align(
            img,
            single_face=True,
            select_single_face_by=select_single_face_by,
        )
        for img in imgpaths
    ]
    results: list = [None] * len(imgpaths)
    valid_indices = [i for i, item in enumerate(aligned_items) if item is not None]
    needs_kps = _needs_aligned_keypoints(processor)

    for chunk_idx in _chunked(valid_indices, batch_size):
        chunk_items = [aligned_items[i] for i in chunk_idx]
        embeddings, fiqa_scores = _compute_batch_for_aligned_items(
            processor,
            chunk_items,
            needs_kps=needs_kps,
        )

        for k, idx in enumerate(chunk_idx):
            results[idx] = _assemble_result(
                processor,
                aligned_items[idx],
                embeddings,
                fiqa_scores,
                k,
            )

    return results


def _process_multi_face_images_batch(
    processor,
    imgpaths,
    *,
    select_single_face_by: str,
    batch_size: int,
):
    aligned_per_image = [
        processor.detect_and_align(
            img,
            single_face=False,
            select_single_face_by=select_single_face_by,
        )
        for img in imgpaths
    ]
    results_multi: list = [[] for _ in imgpaths]
    flat: list = []
    for img_idx, faces in enumerate(aligned_per_image):
        for face_item in faces:
            flat.append((img_idx, face_item))
    if not flat:
        return results_multi

    needs_kps = _needs_aligned_keypoints(processor)
    for chunk in _chunked(flat, batch_size):
        chunk_items = [face_item for _, face_item in chunk]
        embeddings, fiqa_scores = _compute_batch_for_aligned_items(
            processor,
            chunk_items,
            needs_kps=needs_kps,
        )

        for k, (img_idx, face_item) in enumerate(chunk):
            results_multi[img_idx].append(
                _assemble_result(
                    processor,
                    face_item,
                    embeddings,
                    fiqa_scores,
                    k,
                )
            )

    return results_multi


def _compute_batch_for_aligned_items(processor, aligned_items, *, needs_kps: bool):
    # `detect_and_align` returns RGB to match `process_image`, while recognition
    # sessions are fed BGR crops.
    crops = _aligned_rgb_items_to_bgr_batch(aligned_items)
    keypoints_batch = _aligned_keypoints_batch(aligned_items) if needs_kps else None
    return processor._aligned_face_runner().try_compute_batch(
        crops,
        aligned_keypoints_batch=keypoints_batch,
    )


def _assemble_result(processor, align_item, embeddings, fiqa_scores, index: int):
    return build_face_result_from_align_result(
        align_item=align_item,
        embeddings=_embedding_at(embeddings, index, processor.concat_embeddings),
        fiqa_score=_fiqa_at(fiqa_scores, index),
        models=processor.models,
        extended=processor.extended,
        concat_embeddings=processor.concat_embeddings,
        enabled_tasks=getattr(processor, "enabled_tasks", None),
    )
