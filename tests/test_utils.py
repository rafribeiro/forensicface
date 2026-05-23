import numpy as np
import pytest

from forensicface.utils import (
    aggregate_embeddings,
    annotate_img_with_kps,
    compute_ss_ds,
    cosine_score,
    cosine_similarity,
)
from forensicface.mosaic import build_aligned_face_mosaic


def test_cosine_score_matches_single_pair_from_matrix_similarity():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    z = np.array([3.0, 2.0, 1.0], dtype=np.float32)

    assert np.allclose(cosine_score(x, z), cosine_similarity(x[None], z[None])[0, 0])


def test_aggregate_embeddings_mean_with_weights():
    embeddings = np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32)
    weights = np.array([1.0, 3.0], dtype=np.float32)

    result = aggregate_embeddings(embeddings, weights=weights, method="mean")

    np.testing.assert_allclose(result, [2.5, 4.5])


def test_aggregate_embeddings_median_preserves_existing_weighted_behavior():
    embeddings = np.array([[1.0, 10.0], [3.0, 20.0]], dtype=np.float32)
    weights = np.array([2.0, 3.0], dtype=np.float32)

    result = aggregate_embeddings(embeddings, weights=weights, method="median")

    np.testing.assert_allclose(result, np.median(weights[:, None] * embeddings, axis=0))


def test_aggregate_embeddings_rejects_unknown_method():
    embeddings = np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="method"):
        aggregate_embeddings(embeddings, method="mode")


def test_aggregate_embeddings_rejects_weight_length_mismatch():
    embeddings = np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="weights"):
        aggregate_embeddings(embeddings, weights=weights)


def test_compute_ss_ds_rejects_label_length_mismatch():
    X = np.zeros((2, 4), dtype=np.float32)
    x_id = np.array([1])

    with pytest.raises(ValueError, match="x_id length"):
        compute_ss_ds(X, x_id)


def test_compute_ss_ds_rejects_missing_z_id():
    X = np.zeros((2, 4), dtype=np.float32)
    Z = np.zeros((2, 4), dtype=np.float32)
    x_id = np.array([1, 2])

    with pytest.raises(ValueError, match="z_id"):
        compute_ss_ds(X, x_id, Z=Z)


def test_annotate_img_with_kps_rejects_invalid_color_and_shape():
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    kps = np.zeros((5, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="color"):
        annotate_img_with_kps(img, kps, color="purple")

    with pytest.raises(ValueError, match="shape"):
        annotate_img_with_kps(img, np.zeros((4, 2), dtype=np.float32))


def test_build_aligned_face_mosaic_rejects_missing_shape():
    class _Processor:
        IMG_SIZE = (112, 112)

    with pytest.raises(ValueError, match="mosaic_shape"):
        build_aligned_face_mosaic(_Processor(), [], None)
