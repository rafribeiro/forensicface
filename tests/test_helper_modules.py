from types import SimpleNamespace

import numpy as np
import pytest

from forensicface.comparison import aggregate_from_images, compare_faces
from forensicface.geometry import extend_bbox, select_best_face
from forensicface.preprocessing import normalize_aligned_keypoints, to_ada_input
from forensicface.recognition import (
    RecognitionRunner,
    build_keypoint_model_inputs,
    looks_like_cuda_oom,
    try_compute_embeddings_batch,
)
from forensicface.results import (
    AlignedFace,
    FaceResult,
    assemble_face_result,
    build_align_result,
    build_face_result,
)


class _Input:
    def __init__(self, name):
        self.name = name


class _Session:
    def __init__(self, input_names, outputs):
        self._inputs = [_Input(name) for name in input_names]
        self._outputs = outputs
        self.calls = []

    def get_inputs(self):
        return self._inputs

    def run(self, output_names, inputs):
        self.calls.append(inputs)
        if callable(self._outputs):
            return self._outputs(inputs)
        return self._outputs


def test_select_best_face_by_size_and_centrality():
    small_center = SimpleNamespace(bbox=np.array([40, 40, 60, 60], dtype=np.float32))
    large_corner = SimpleNamespace(bbox=np.array([0, 0, 40, 40], dtype=np.float32))

    assert (
        select_best_face((100, 100, 3), [small_center, large_corner], "size")
        is large_corner
    )
    assert (
        select_best_face((100, 100, 3), [small_center, large_corner], "centrality")
        is small_center
    )


def test_select_best_face_rejects_invalid_inputs():
    face = SimpleNamespace(bbox=np.array([0, 0, 10, 10], dtype=np.float32))

    with pytest.raises(ValueError, match="criterion"):
        select_best_face((100, 100, 3), [face], "confidence")

    with pytest.raises(ValueError, match="at least one"):
        select_best_face((100, 100, 3), [], "size")


def test_extend_bbox_expands_and_clips_to_frame():
    bbox = np.array([10, 20, 30, 40], dtype=np.float32)

    assert extend_bbox(bbox, (50, 60, 3), margin_factor=2.0) == [0, 10, 40, 50]


def test_to_ada_input_normalizes_single_and_batch_images():
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    img[:, :, 1] = 255

    single = to_ada_input(img)
    batch = to_ada_input(np.stack([img, img]))

    assert single.shape == (1, 3, 112, 112)
    assert batch.shape == (2, 3, 112, 112)
    assert single.dtype == np.float32
    np.testing.assert_allclose(single[0, 0, 0, 0], -1.0)
    np.testing.assert_allclose(single[0, 1, 0, 0], 1.0)
    np.testing.assert_allclose(batch[0], single[0])


def test_to_ada_input_rejects_invalid_rank():
    with pytest.raises(ValueError, match="ndim"):
        to_ada_input(np.zeros((112, 112), dtype=np.uint8))


def test_normalize_aligned_keypoints_accepts_single_and_batch():
    keypoints = np.array(
        [[0, 0], [56, 28], [112, 56], [28, 84], [84, 112]], dtype=np.float32
    )
    batch = np.stack([keypoints, keypoints + 1])

    single_normalized = normalize_aligned_keypoints(keypoints, model_name="sepaelv6")
    batch_normalized = normalize_aligned_keypoints(batch, model_name="sepaelv6")

    assert single_normalized.shape == (1, 5, 2)
    assert batch_normalized.shape == (2, 5, 2)
    np.testing.assert_allclose(single_normalized[0, 1], [0.5, 0.25])
    np.testing.assert_allclose(batch_normalized[0], single_normalized[0])


def test_normalize_aligned_keypoints_rejects_missing_or_wrong_shape():
    with pytest.raises(ValueError, match="requires aligned"):
        normalize_aligned_keypoints(None, model_name="sepaelv6")

    with pytest.raises(ValueError, match=r"\(5, 2\) or \(N, 5, 2\)"):
        normalize_aligned_keypoints(np.zeros((4, 2)), model_name="sepaelv6")


def test_face_result_supports_mapping_and_attribute_access():
    result = FaceResult({"bbox": [1, 2, 3, 4]})

    assert result.bbox == [1, 2, 3, 4]
    result.det_score = 0.8
    assert result["det_score"] == 0.8
    del result.det_score
    assert "det_score" not in result

    with pytest.raises(AttributeError, match="det_score"):
        _ = result.det_score


def test_build_align_result_converts_types_and_extended_fields():
    pose = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = build_align_result(
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        bbox=np.array([1.5, 2.5, 3.5, 4.5]),
        keypoints=np.zeros((5, 2), dtype=np.float32),
        aligned_keypoints=np.ones((5, 2), dtype=np.float32),
        det_score=np.float32(0.75),
        extended=True,
        gender=1,
        age=np.int64(42),
        pose=pose,
    )

    assert isinstance(result, FaceResult)
    assert result.gender == "M"
    assert result.age == 42
    assert result.det_score == 0.75
    np.testing.assert_array_equal(result.bbox, [1, 2, 3, 4])
    assert result.pose is not pose


def test_build_face_result_maps_pose_and_per_model_embeddings():
    keypoints = np.array([[0.0, 0.0], [3.0, 4.0], [0, 0], [0, 0], [0, 0]])
    embeddings = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

    result = build_face_result(
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        bbox=np.array([1.2, 2.2, 3.2, 4.2]),
        keypoints=keypoints,
        det_score=0.9,
        embeddings=embeddings,
        fiqa_score=0.6,
        models=["a", "b"],
        extended=True,
        concat_embeddings=False,
        gender=0,
        age=30,
        pose=np.array([10.0, 20.0, 30.0]),
    )

    assert result.ipd == 5.0
    assert result.gender == "F"
    assert result.yaw == 20.0
    assert result.pitch == 10.0
    assert result.roll == 30.0
    np.testing.assert_array_equal(result.embedding_a, embeddings[0])
    np.testing.assert_array_equal(result.embedding_b, embeddings[1])
    assert "embedding" not in result


def test_assemble_face_result_accepts_aligned_face_dataclass():
    aligned = AlignedFace(
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        bbox=np.array([1.2, 2.2, 3.2, 4.2]),
        keypoints=np.array([[0.0, 0.0], [6.0, 8.0], [0, 0], [0, 0], [0, 0]]),
        aligned_keypoints=np.ones((5, 2), dtype=np.float32),
        det_score=0.95,
        gender=1,
        age=44,
        pose=np.array([1.0, 2.0, 3.0]),
    )

    result = assemble_face_result(
        aligned_face=aligned,
        embeddings=np.array([1.0, 2.0], dtype=np.float32),
        fiqa_score=0.7,
        models=["dummy"],
        extended=True,
        concat_embeddings=True,
    )

    assert result.ipd == 10.0
    assert result.gender == "M"
    assert result.age == 44
    assert result.yaw == 2.0
    np.testing.assert_array_equal(result.bbox, [1, 2, 3, 4])
    np.testing.assert_array_equal(result.embedding, [1.0, 2.0])


def test_aligned_face_can_be_built_from_align_result_mapping():
    align_result = FaceResult(
        {
            "aligned_face": np.zeros((112, 112, 3), dtype=np.uint8),
            "bbox": np.array([1, 2, 3, 4]),
            "keypoints": np.zeros((5, 2), dtype=np.float32),
            "aligned_keypoints": np.ones((5, 2), dtype=np.float32),
            "det_score": 0.8,
            "gender": "F",
            "age": 36,
        }
    )

    aligned = AlignedFace.from_align_result(align_result)

    assert aligned.aligned_face is align_result.aligned_face
    assert aligned.gender == "F"
    assert aligned.age == 36
    np.testing.assert_array_equal(aligned.aligned_keypoints, np.ones((5, 2)))


def test_build_keypoint_model_inputs_validates_and_normalizes_keypoints():
    session = _Session(["input_images", "keypoints"], [np.zeros((1, 2))])
    img_to_input = np.zeros((1, 3, 112, 112), dtype=np.float32)
    keypoints = np.array(
        [[0, 0], [56, 56], [112, 112], [28, 84], [84, 28]], dtype=np.float32
    )

    inputs = build_keypoint_model_inputs(
        model_name="sepaelv6",
        rec_ort=session,
        img_to_input=img_to_input,
        aligned_keypoints=keypoints,
    )

    assert inputs["input_images"] is img_to_input
    np.testing.assert_allclose(inputs["keypoints"][0, 1], [0.5, 0.5])


def test_build_keypoint_model_inputs_reports_missing_onnx_inputs():
    session = _Session(["input_images"], [np.zeros((1, 2))])

    with pytest.raises(ValueError, match="Missing: \\['keypoints'\\]"):
        build_keypoint_model_inputs(
            model_name="sepaelv6",
            rec_ort=session,
            img_to_input=np.zeros((1, 3, 112, 112), dtype=np.float32),
            aligned_keypoints=np.zeros((5, 2), dtype=np.float32),
        )


def test_recognition_runner_compute_one_wires_inputs_and_fiqa():
    plain = _Session(["image"], [np.array([[1.0, 2.0]], dtype=np.float32)])
    keypoint = _Session(
        ["input_images", "keypoints"],
        [
            np.array([[3.0, 4.0]], dtype=np.float32),
            np.array([[0.5]], dtype=np.float32),
        ],
    )
    fiqa = _Session(
        ["image"],
        [np.zeros((1, 1), dtype=np.float32), np.array([[0.9]], dtype=np.float32)],
    )
    runner = RecognitionRunner(
        models=["plain", "sepaelv6"],
        rec_inference_sessions=[plain, keypoint],
        ort_fiqa=fiqa,
        extended=True,
        concat_embeddings=True,
    )
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    keypoints = np.full((5, 2), 56, dtype=np.float32)

    embedding, fiqa_score = runner.compute_one(img, aligned_keypoints=keypoints)

    np.testing.assert_allclose(embedding, [1.0, 2.0, 1.5, 2.0])
    assert fiqa_score == pytest.approx(0.9)
    assert set(keypoint.calls[0]) == {"input_images", "keypoints"}
    np.testing.assert_allclose(keypoint.calls[0]["keypoints"], 0.5)


def test_recognition_runner_compute_batch_concatenates_embeddings_and_fiqa():
    def plain_outputs(inputs):
        batch_size = inputs["image"].shape[0]
        return [np.ones((batch_size, 2), dtype=np.float32)]

    def scaled_outputs(inputs):
        batch_size = inputs["image"].shape[0]
        return [
            np.full((batch_size, 2), 4.0, dtype=np.float32),
            np.full((batch_size, 1), 0.25, dtype=np.float32),
        ]

    def fiqa_outputs(inputs):
        batch_size = inputs["image"].shape[0]
        return [
            np.zeros((batch_size, 1), dtype=np.float32),
            np.arange(batch_size, dtype=np.float32).reshape(batch_size, 1),
        ]

    runner = RecognitionRunner(
        models=["plain", "scaled"],
        rec_inference_sessions=[
            _Session(["image"], plain_outputs),
            _Session(["image"], scaled_outputs),
        ],
        ort_fiqa=_Session(["image"], fiqa_outputs),
        extended=True,
        concat_embeddings=True,
    )
    batch = np.zeros((3, 112, 112, 3), dtype=np.uint8)

    embeddings, fiqa_scores = runner.compute_batch(batch)

    assert embeddings.shape == (3, 4)
    np.testing.assert_allclose(embeddings, np.ones((3, 4), dtype=np.float32))
    np.testing.assert_allclose(fiqa_scores, [0.0, 1.0, 2.0])


def test_recognition_runner_compute_batch_rejects_invalid_shapes():
    runner = RecognitionRunner(
        models=[],
        rec_inference_sessions=[],
        extended=False,
        concat_embeddings=True,
    )

    with pytest.raises(ValueError, match="Expected shape"):
        runner.compute_batch(np.zeros((112, 112, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match=r"shape \(N, 5, 2\)"):
        runner.compute_batch(
            np.zeros((2, 112, 112, 3), dtype=np.uint8),
            aligned_keypoints_batch=np.zeros((2, 4, 2), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="N=1.*N=2"):
        runner.compute_batch(
            np.zeros((2, 112, 112, 3), dtype=np.uint8),
            aligned_keypoints_batch=np.zeros((1, 5, 2), dtype=np.float32),
        )


def test_cuda_oom_detection_matches_common_messages():
    assert looks_like_cuda_oom(RuntimeError("CUDAErrorMemoryAllocation"))
    assert looks_like_cuda_oom(RuntimeError("CUDA OOM during inference"))
    assert looks_like_cuda_oom(RuntimeError("failed to alloc memory"))
    assert not looks_like_cuda_oom(RuntimeError("invalid input shape"))


def test_try_compute_embeddings_batch_splits_cuda_oom_and_merges_results():
    seen_batches = []
    seen_keypoints = []

    def compute_batch(batch, aligned_keypoints_batch=None):
        seen_batches.append(batch.shape[0])
        seen_keypoints.append(
            None if aligned_keypoints_batch is None else aligned_keypoints_batch.copy()
        )
        if batch.shape[0] > 1:
            raise RuntimeError("CUDA out of memory")
        value = float(batch[0, 0, 0, 0])
        return np.array([[value, value + 1]], dtype=np.float32), np.array([value])

    batch = np.arange(3 * 112 * 112 * 3, dtype=np.float32).reshape(3, 112, 112, 3)
    keypoints = np.arange(3 * 5 * 2, dtype=np.float32).reshape(3, 5, 2)

    with pytest.warns(UserWarning, match="CUDA OOM"):
        embeddings, fiqa_scores = try_compute_embeddings_batch(
            compute_batch,
            batch,
            aligned_keypoints_batch=keypoints,
        )

    assert seen_batches == [3, 1, 2, 1, 1]
    np.testing.assert_array_equal(seen_keypoints[1], keypoints[:1])
    np.testing.assert_array_equal(seen_keypoints[3], keypoints[1:2])
    np.testing.assert_array_equal(seen_keypoints[4], keypoints[2:])
    assert embeddings.shape == (3, 2)
    np.testing.assert_allclose(fiqa_scores, batch[:, 0, 0, 0])


def test_compare_faces_uses_concatenated_embeddings():
    class _Processor:
        concat_embeddings = True

        def process_image(self, imgpath, single_face=True):
            assert single_face is True
            embeddings = {
                "a": np.array([1.0, 0.0], dtype=np.float32),
                "b": np.array([0.0, 1.0], dtype=np.float32),
            }
            return FaceResult({"embedding": embeddings[imgpath]})

    assert compare_faces(_Processor(), "a", "a") == pytest.approx(1.0)
    assert compare_faces(_Processor(), "a", "b") == pytest.approx(0.0)


def test_compare_faces_rejects_non_concatenated_processor():
    class _Processor:
        concat_embeddings = False

    with pytest.raises(ValueError, match="concat_embeddings=False"):
        compare_faces(_Processor(), "a", "b")


def test_aggregate_from_images_handles_per_model_embeddings_with_quality_weights():
    class _Processor:
        concat_embeddings = False
        extended = True
        models = ["a", "b"]

        def process_image(self, imgpath, single_face=True):
            assert single_face is True
            if imgpath == "missing":
                return []
            return FaceResult(
                {
                    "embedding_a": np.array([1.0, 3.0], dtype=np.float32)
                    if imgpath == "low"
                    else np.array([3.0, 5.0], dtype=np.float32),
                    "embedding_b": np.array([10.0, 30.0], dtype=np.float32)
                    if imgpath == "low"
                    else np.array([30.0, 50.0], dtype=np.float32),
                    "fiqa_score": 1.0 if imgpath == "low" else 3.0,
                }
            )

    result = aggregate_from_images(
        _Processor(),
        ["low", "missing", "high"],
        quality_weight=True,
    )

    np.testing.assert_allclose(result["embedding_a"], [2.5, 4.5])
    np.testing.assert_allclose(result["embedding_b"], [25.0, 45.0])


def test_aggregate_from_images_rejects_quality_weights_without_extended():
    class _Processor:
        extended = False

    with pytest.raises(ValueError, match="extended = True"):
        aggregate_from_images(_Processor(), ["img"], quality_weight=True)
