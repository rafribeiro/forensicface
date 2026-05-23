import numpy as np
import pytest

from forensicface.utils import aggregate_embeddings, cosine_score, cosine_similarity


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

    with pytest.raises(AssertionError):
        aggregate_embeddings(embeddings, method="mode")

