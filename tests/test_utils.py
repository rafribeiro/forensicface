import inspect

import numpy as np
import pytest

from forensicface.app import ForensicFace
from forensicface.mosaic import (
    build_mosaic_from_aligned_faces,
    build_mosaic_from_images,
)
from forensicface.utils import (
    aggregate_embeddings,
    annotate_img_with_kps,
    compute_ss_ds,
    cosine_score,
    cosine_similarity,
)


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


def test_annotate_img_with_kps_uses_default_keypoint_colors():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    kps = np.array(
        [[2, 2], [5, 2], [8, 2], [11, 2], [14, 2]],
        dtype=np.float32,
    )

    annotated = annotate_img_with_kps(img, kps, radius=0)

    np.testing.assert_array_equal(annotated[2, 2], [0, 255, 0])
    np.testing.assert_array_equal(annotated[2, 5], [0, 0, 255])
    np.testing.assert_array_equal(annotated[2, 8], [0, 255, 0])
    np.testing.assert_array_equal(annotated[2, 11], [0, 255, 0])
    np.testing.assert_array_equal(annotated[2, 14], [0, 255, 0])


def test_annotate_img_with_kps_accepts_per_keypoint_colors():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    kps = np.array(
        [[2, 2], [5, 2], [8, 2], [11, 2], [14, 2]],
        dtype=np.float32,
    )

    annotated = annotate_img_with_kps(
        img,
        kps,
        colors=("red", "blue", "green", "white", "black"),
        radius=0,
    )

    np.testing.assert_array_equal(annotated[2, 2], [0, 0, 255])
    np.testing.assert_array_equal(annotated[2, 5], [255, 0, 0])
    np.testing.assert_array_equal(annotated[2, 8], [0, 255, 0])
    np.testing.assert_array_equal(annotated[2, 11], [255, 255, 255])
    np.testing.assert_array_equal(annotated[2, 14], [0, 0, 0])


def test_annotate_img_with_kps_rejects_invalid_colors_and_shape():
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    kps = np.zeros((5, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="colors"):
        annotate_img_with_kps(
            img,
            kps,
            colors=("green", "red", "purple", "green", "green"),
        )

    with pytest.raises(ValueError, match="length 5"):
        annotate_img_with_kps(img, kps, colors=("green", "red"))

    with pytest.raises(ValueError, match="shape"):
        annotate_img_with_kps(img, np.zeros((4, 2), dtype=np.float32))


def test_build_mosaic_from_aligned_faces_tiles_rgb_faces_as_bgr():
    red_rgb = np.full((2, 2, 3), [255, 0, 0], dtype=np.uint8)
    green_rgb = np.full((2, 2, 3), [0, 255, 0], dtype=np.uint8)
    blue_rgb = np.full((2, 2, 3), [0, 0, 255], dtype=np.uint8)

    mosaic = build_mosaic_from_aligned_faces(
        [red_rgb, green_rgb, blue_rgb],
        mosaic_shape=(2, 2),
        border=0,
        image_size=(2, 2),
    )

    assert mosaic.shape == (4, 4, 3)
    np.testing.assert_array_equal(mosaic[0, 0], [0, 0, 255])
    np.testing.assert_array_equal(mosaic[0, 2], [0, 255, 0])
    np.testing.assert_array_equal(mosaic[2, 0], [255, 0, 0])
    np.testing.assert_array_equal(mosaic[2, 2], [0, 0, 0])


def test_build_mosaic_from_aligned_faces_validates_keypoints():
    aligned_faces = [np.zeros((112, 112, 3), dtype=np.uint8)]

    with pytest.raises(ValueError, match="requires keypoints"):
        build_mosaic_from_aligned_faces(
            aligned_faces,
            mosaic_shape=(1, 1),
            draw_keypoints=True,
        )

    with pytest.raises(ValueError, match="same length"):
        build_mosaic_from_aligned_faces(
            aligned_faces,
            mosaic_shape=(1, 1),
            keypoints=[],
        )


def test_build_mosaic_from_aligned_faces_uses_custom_keypoint_colors():
    aligned_faces = [np.zeros((20, 20, 3), dtype=np.uint8)]
    keypoints = [
        np.array(
            [[2, 2], [5, 2], [8, 2], [11, 2], [14, 2]],
            dtype=np.float32,
        )
    ]

    mosaic = build_mosaic_from_aligned_faces(
        aligned_faces,
        mosaic_shape=(1, 1),
        border=0,
        draw_keypoints=True,
        keypoints=keypoints,
        keypoint_colors=("blue", "white", "red", "green", "black"),
        image_size=(20, 20),
    )

    np.testing.assert_array_equal(mosaic[2, 2], [255, 0, 0])
    np.testing.assert_array_equal(mosaic[2, 5], [255, 255, 255])
    np.testing.assert_array_equal(mosaic[2, 8], [0, 0, 255])


def test_build_mosaic_from_images_processes_original_images():
    class _Processor:
        IMG_SIZE = (2, 2)

        def __init__(self):
            self.calls = []

        def process_image(
            self,
            img,
            draw_keypoints=False,
            keypoint_colors=None,
            single_face=True,
        ):
            self.calls.append((img, draw_keypoints, keypoint_colors, single_face))
            return {
                "aligned_face": np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8),
            }

    processor = _Processor()
    keypoint_colors = ("blue", "white", "red", "green", "black")
    mosaic = build_mosaic_from_images(
        processor,
        ["a.jpg", "b.jpg"],
        mosaic_shape=(2, 1),
        border=0,
        draw_keypoints=True,
        keypoint_colors=keypoint_colors,
    )

    assert processor.calls == [
        ("a.jpg", True, keypoint_colors, True),
        ("b.jpg", True, keypoint_colors, True),
    ]
    assert mosaic.shape == (2, 4, 3)
    np.testing.assert_array_equal(mosaic[0, 0], [30, 20, 10])


def test_forensicface_mosaic_api_separates_aligned_faces():
    assert "aligned_faces" not in inspect.signature(ForensicFace.build_mosaic).parameters

    ff = ForensicFace.__new__(ForensicFace)
    aligned_face = np.zeros((112, 112, 3), dtype=np.uint8)

    mosaic = ff.build_mosaic_from_aligned_faces([aligned_face], mosaic_shape=(1, 1), border=0)

    assert mosaic.shape == (112, 112, 3)
