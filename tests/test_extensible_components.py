from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from forensicface import ComponentMetadata, ModelSpec
from forensicface.app import ForensicFace
from forensicface.backends import FaceData, ONNXOnlyBackend, PoseAngles
from forensicface.components import ComponentBackend, FaceContext
from forensicface.model_store import resolve_component_model
from forensicface.onnx_components import (
    CenterFaceDetector,
    InsightFaceGenderAgeEstimator,
    InsightFacePoseEstimator,
    SCRFDDetector,
)
from forensicface.runtime_summary import collect_session_provider_details


def _metadata(
    component_id: str,
    tasks: set[str],
    providers: tuple[str, ...] = (),
) -> ComponentMetadata:
    return ComponentMetadata(
        component_id=component_id,
        tasks=frozenset(tasks),
        implementation="test",
        runtime="test",
        model_path=None,
        input_space="test",
        supports_batch=True,
        providers=providers,
    )


class _Detector:
    metadata = _metadata("test-detector", {"detection"})

    def detect(self, image):
        return [
            FaceData(
                bbox=np.array([12, 12, 92, 92]),
                kps=np.array(
                    [[30, 38], [70, 38], [50, 56], [35, 75], [65, 75]],
                    dtype=np.float32,
                ),
                det_score=0.9,
            )
        ]


class _JointGenderAge:
    metadata = _metadata("test-gender-age", {"gender", "age"})
    capabilities = frozenset({"gender", "age"})

    def __init__(self):
        self.calls = 0
        self.seen_shape = None
        self.aligned_shape = None

    def estimate(self, context: FaceContext):
        self.calls += 1
        self.seen_shape = context.original_bgr.shape
        self.aligned_shape = context.aligned_bgr.shape
        return {"gender": 1, "age": 42}


class _Embedding:
    metadata = _metadata("test-embedding", {"embedding"})
    requires_aligned_keypoints = False

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def embed_one(self, aligned_bgr, *, aligned_keypoints=None):
        return np.full(3, self.value, dtype=np.float32)

    def embed_batch(self, aligned_bgr_batch, *, aligned_keypoints_batch=None):
        return np.full((len(aligned_bgr_batch), 3), self.value, dtype=np.float32)


class _Quality:
    metadata = _metadata("test-quality", {"quality"})

    def score_one(self, aligned_bgr):
        return 0.75

    def score_batch(self, aligned_bgr_batch):
        return np.full(len(aligned_bgr_batch), 0.75, dtype=np.float32)


def _component_processor(tmp_path, **overrides):
    options = {
        "detection": _Detector(),
        "pose": None,
        "gender": None,
        "age": None,
        "quality": _Quality(),
        "embedding": _Embedding("test", 1),
        "extended": False,
        "models_root": str(tmp_path),
        "use_gpu": False,
    }
    options.update(overrides)
    return ForensicFace(**options)


class _SingleAttribute:
    def __init__(self, task, value):
        self.metadata = _metadata(f"test-{task}", {task})
        self.capabilities = frozenset({task})
        self.task = task
        self.value = value
        self.calls = 0

    def estimate(self, context):
        self.calls += 1
        return {self.task: self.value}


class _PoseEstimator:
    metadata = _metadata("test-pose", {"pose"})
    capabilities = frozenset({"pose"})

    def estimate(self, context):
        return {"pose": {"pitch": 1.5, "yaw": -2.5, "roll": 3.5}}


def test_model_spec_is_generic_and_options_are_immutable():
    spec = ModelSpec("centerface", path="~/centerface.onnx", nms_threshold=0.2)

    assert spec.alias == "centerface"
    assert spec.path == Path("~/centerface.onnx").expanduser()
    assert spec.options["nms_threshold"] == 0.2
    with pytest.raises(TypeError):
        spec.options["nms_threshold"] = 0.3


def test_component_backend_runs_joint_estimator_once_and_preserves_native_input():
    estimator = _JointGenderAge()
    backend = ComponentBackend(
        detector=_Detector(),
        estimators=[estimator, estimator],
        enabled_tasks={"gender", "age"},
    )
    image = np.zeros((173, 241, 3), dtype=np.uint8)

    face = backend.detect_faces(image)[0]

    assert estimator.calls == 1
    assert estimator.seen_shape == (173, 241, 3)
    assert estimator.aligned_shape == (112, 112, 3)
    assert face.gender == 1
    assert face.age == 42


@pytest.mark.parametrize(
    "bad_face, message",
    [
        (
            FaceData(
                bbox=np.array([0, 0, 10, 10]),
                kps=np.zeros((4, 2)),
                det_score=0.5,
            ),
            "shape \\(5, 2\\)",
        ),
        (
            FaceData(
                bbox=np.zeros(3),
                kps=np.zeros((5, 2)),
                det_score=0.5,
            ),
            "shape \\(4,\\)",
        ),
    ],
)
def test_component_backend_validates_detector_contract(bad_face, message):
    class _BadDetector:
        metadata = _metadata("bad", {"detection"})

        def detect(self, image):
            return [bad_face]

    backend = ComponentBackend(
        detector=_BadDetector(), estimators=[], enabled_tasks=set()
    )

    with pytest.raises(ValueError, match=message):
        backend.detect_faces(np.zeros((20, 20, 3), dtype=np.uint8))


def test_explicit_components_allow_task_subset_and_omit_disabled_fields(tmp_path):
    estimator = _JointGenderAge()
    ff = ForensicFace(
        detection=_Detector(),
        pose=None,
        gender=estimator,
        age=estimator,
        quality=None,
        embedding=None,
        models_root=str(tmp_path),
        use_gpu=False,
    )

    with pytest.warns(FutureWarning):
        result = ff.process_image(
            np.zeros((120, 120, 3), dtype=np.uint8), single_face=True
        )

    assert estimator.calls == 1
    assert result["gender"] == "M"
    assert result["age"] == 42
    assert "embedding" not in result
    assert "fiqa_score" not in result
    assert "yaw" not in result
    with pytest.raises(ValueError, match="requires an embedding model"):
        ff.process_aligned_face_image(np.zeros((112, 112, 3), dtype=np.uint8))


def test_explicit_multiple_embedding_objects_and_quality(tmp_path):
    ff = ForensicFace(
        detection=_Detector(),
        embedding=[_Embedding("one", 1), _Embedding("two", 2)],
        quality=_Quality(),
        extended=False,
        models_root=str(tmp_path),
        use_gpu=False,
    )

    with pytest.warns(FutureWarning):
        result = ff.process_image(
            np.zeros((120, 120, 3), dtype=np.uint8), single_face=True
        )

    np.testing.assert_array_equal(result["embedding"], [1, 1, 1, 2, 2, 2])
    assert result["fiqa_score"] == pytest.approx(0.75)
    assert "gender" not in result
    assert ff.models == ["one", "two"]

    one = ff.process_aligned_face_image(np.zeros((112, 112, 3), dtype=np.uint8))
    batch = ff.process_aligned_faces_batch(
        np.zeros((2, 112, 112, 3), dtype=np.uint8)
    )
    np.testing.assert_array_equal(one["embedding"], batch[0]["embedding"])
    assert one["fiqa_score"] == pytest.approx(batch[0]["fiqa_score"])


def test_gender_and_age_can_use_separate_estimators(tmp_path):
    gender = _SingleAttribute("gender", 0)
    age = _SingleAttribute("age", 37)
    ff = ForensicFace(
        detection=_Detector(),
        pose=None,
        gender=gender,
        age=age,
        quality=None,
        embedding=None,
        models_root=str(tmp_path),
        use_gpu=False,
    )

    with pytest.warns(FutureWarning):
        result = ff.process_image(
            np.zeros((120, 120, 3), dtype=np.uint8), single_face=True
        )

    assert result["gender"] == "F"
    assert result["age"] == 37
    assert gender.calls == age.calls == 1


def test_pose_component_is_normalized_internally_and_keeps_public_fields(tmp_path):
    ff = ForensicFace(
        detection=_Detector(),
        pose=_PoseEstimator(),
        gender=None,
        age=None,
        quality=None,
        embedding=None,
        models_root=str(tmp_path),
        use_gpu=False,
    )

    face = ff.backend.detect_faces(np.zeros((120, 120, 3), dtype=np.uint8))[0]
    assert isinstance(face.pose, PoseAngles)
    assert face.pose.pitch == pytest.approx(1.5)

    with pytest.warns(FutureWarning):
        result = ff.process_image(
            np.zeros((120, 120, 3), dtype=np.uint8), single_face=True
        )
    assert result["pitch"] == pytest.approx(1.5)
    assert result["yaw"] == pytest.approx(-2.5)
    assert result["roll"] == pytest.approx(3.5)

    aligned = ff.detect_and_align(
        np.zeros((120, 120, 3), dtype=np.uint8), single_face=True
    )
    np.testing.assert_allclose(aligned["pose"], [1.5, -2.5, 3.5])


def test_component_pipeline_supports_compare_aggregation_and_mosaic(tmp_path):
    ff = _component_processor(tmp_path)
    images = [np.zeros((120, 120, 3), dtype=np.uint8) for _ in range(2)]

    with pytest.warns(FutureWarning):
        assert ff.compare(images[0], images[1]) == pytest.approx(1.0)
    with pytest.warns(FutureWarning):
        aggregated = ff.aggregate_from_images(images, quality_weight=True)
    np.testing.assert_allclose(aggregated, [1, 1, 1])

    with pytest.warns(Warning):
        mosaic = ff.build_mosaic(images, mosaic_shape=(2, 1), border=0)
    assert mosaic.shape == (112, 224, 3)


def test_component_pipeline_supports_video_extraction(tmp_path):
    ff = _component_processor(tmp_path, quality=None)
    video_path = tmp_path / "component_video.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (120, 120),
    )
    assert writer.isOpened()
    writer.write(np.zeros((120, 120, 3), dtype=np.uint8))
    writer.write(np.ones((120, 120, 3), dtype=np.uint8))
    writer.release()

    destination = tmp_path / "component_faces"
    assert ff.extract_faces(str(video_path), dest_folder=str(destination)) == 2
    assert len(list(destination.glob("frame_*_face_*.png"))) == 2


def test_runtime_summary_reads_component_metadata_providers(tmp_path):
    class _ProviderDetector(_Detector):
        metadata = _metadata(
            "provider-detector", {"detection"}, ("CPUExecutionProvider",)
        )

    ff = _component_processor(
        tmp_path,
        detection=_ProviderDetector(),
        embedding=None,
        quality=None,
    )

    assert collect_session_provider_details(ff) == [
        ("provider-detector", ["CPUExecutionProvider"])
    ]


@pytest.mark.parametrize("legacy_name", ["models", "model", "backend"])
def test_explicit_selectors_reject_legacy_configuration(tmp_path, legacy_name):
    kwargs = {
        "detection": _Detector(),
        "embedding": None,
        "models_root": str(tmp_path),
        "use_gpu": False,
    }
    kwargs[legacy_name] = {
        "models": ["sepaelv2"],
        "model": "sepaelv2",
        "backend": object(),
    }[legacy_name]

    with pytest.raises(ValueError, match=legacy_name):
        ForensicFace(**kwargs)


def test_detection_cannot_be_disabled(tmp_path):
    with pytest.raises(ValueError, match="detection=None"):
        ForensicFace(
            detection=None,
            embedding=None,
            models_root=str(tmp_path),
            use_gpu=False,
        )


def test_explicit_selectors_reject_nondefault_backend_name(tmp_path):
    with pytest.raises(ValueError, match="backend_name"):
        ForensicFace(
            detection=_Detector(),
            embedding=None,
            backend_name="custom",
            models_root=str(tmp_path),
            use_gpu=False,
        )


@pytest.mark.parametrize("legacy_parameter", ["det_size", "det_thresh"])
def test_non_scrfd_detector_rejects_explicit_scrfd_parameters(
    tmp_path, legacy_parameter
):
    kwargs = {
        "detection": _Detector(),
        "embedding": None,
        "extended": False,
        "models_root": str(tmp_path),
        "use_gpu": False,
        legacy_parameter: 320 if legacy_parameter == "det_size" else 0.5,
    }
    with pytest.raises(ValueError, match="apply only to SCRFD"):
        ForensicFace(**kwargs)


def test_scrfd_modelspec_det_size_updates_effective_state_and_summary(
    monkeypatch, tmp_path, capsys
):
    class _Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    class _SCRFD:
        def __init__(self, **kwargs):
            self.session = _Session()
            self.input_size = None

        def prepare(self, ctx_id, **kwargs):
            self.input_size = tuple(kwargs["input_size"])

        def detect(self, image, max_num=0, metric="default"):
            return np.empty((0, 5), dtype=np.float32), None

    monkeypatch.setattr("forensicface.onnx_components.SCRFD", _SCRFD)
    model_path = tmp_path / "scrfd.onnx"
    model_path.write_bytes(b"test")

    ff = ForensicFace(
        detection=ModelSpec("scrfd", path=model_path, det_size=128),
        pose=None,
        gender=None,
        age=None,
        quality=None,
        embedding=None,
        models_root=str(tmp_path),
        use_gpu=False,
    )

    assert ff.det_size == (128, 128)
    assert ff.backend.det_model.input_size == (128, 128)
    assert "det_size=(128, 128)" in capsys.readouterr().out


def test_component_model_resolver_accepts_one_versioned_onnx_file(tmp_path):
    model_dir = tmp_path / "detection" / "centerface"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "centerface20260722.onnx"
    model_file.write_bytes(b"model")

    assert resolve_component_model(
        str(tmp_path), task="detection", alias="centerface"
    ) == str(model_file)


def test_component_model_resolver_rejects_ambiguous_directory(tmp_path):
    model_dir = tmp_path / "detection" / "centerface"
    model_dir.mkdir(parents=True)
    (model_dir / "one.onnx").write_bytes(b"one")
    (model_dir / "two.onnx").write_bytes(b"two")

    with pytest.raises(RuntimeError, match="Multiple ONNX files"):
        resolve_component_model(
            str(tmp_path), task="detection", alias="centerface"
        )


def test_centerface_adapter_owns_dynamic_preprocessing_and_decoding(monkeypatch):
    class _Info:
        def __init__(self, name):
            self.name = name

    class _Session:
        def __init__(self):
            self.feed_shape = None

        def get_inputs(self):
            return [_Info("input")]

        def get_outputs(self):
            return [_Info(name) for name in ("heatmap", "scale", "offset", "landmarks")]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, output_names, feed):
            assert output_names == ["heatmap", "scale", "offset", "landmarks"]
            self.feed_shape = feed["input"].shape
            heatmap = np.zeros((1, 1, 16, 24), dtype=np.float32)
            heatmap[0, 0, 5, 7] = 0.9
            scale = np.zeros((1, 2, 16, 24), dtype=np.float32)
            offset = np.zeros((1, 2, 16, 24), dtype=np.float32)
            landmarks = np.full((1, 10, 16, 24), 0.5, dtype=np.float32)
            return heatmap, scale, offset, landmarks

    session = _Session()
    monkeypatch.setattr(
        "forensicface.onnx_components.onnxruntime.InferenceSession",
        lambda *_args, **_kwargs: session,
    )
    detector = CenterFaceDetector(
        "centerface.onnx", providers=["CPUExecutionProvider"]
    )

    faces = detector.detect(np.zeros((33, 65, 3), dtype=np.uint8))

    assert session.feed_shape == (1, 3, 64, 96)
    assert len(faces) == 1
    assert faces[0].det_score == pytest.approx(0.9)
    assert faces[0].kps.shape == (5, 2)
    np.testing.assert_allclose(
        faces[0].bbox,
        [28 * 65 / 96, 20 * 33 / 64, 32 * 65 / 96, 24 * 33 / 64],
    )


def test_current_component_adapters_match_legacy_backend(monkeypatch):
    class _Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    class _SCRFD:
        def __init__(self, **kwargs):
            self.session = _Session()
            self.input_size = None

        def prepare(self, *args, **kwargs):
            self.input_size = tuple(kwargs["input_size"])

        def detect(self, image, max_num=0, metric="default"):
            return (
                np.array([[12.25, 13.5, 92.75, 95.0, 0.9]], dtype=np.float32),
                np.array(
                    [[[30, 38], [70, 38], [50, 56], [35, 75], [65, 75]]],
                    dtype=np.float32,
                ),
            )

    class _Landmark:
        input_size = (192, 192)

        def __init__(self, **kwargs):
            self.session = _Session()

        def prepare(self, *args, **kwargs):
            pass

        def get(self, image, face):
            face["pose"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    class _Attribute:
        input_size = (96, 96)

        def __init__(self, **kwargs):
            self.session = _Session()

        def prepare(self, *args, **kwargs):
            pass

        def get(self, image, face):
            face["gender"] = 1
            face["age"] = 41
            return 1, 41

    monkeypatch.setattr("forensicface.onnx_components.SCRFD", _SCRFD)
    monkeypatch.setattr("forensicface.onnx_components.LandmarkONNX", _Landmark)
    monkeypatch.setattr("forensicface.onnx_components.AttributeONNX", _Attribute)

    detector = SCRFDDetector(
        "det.onnx",
        providers=["CPUExecutionProvider"],
        ctx_id=-1,
    )
    pose = InsightFacePoseEstimator(
        "pose.onnx", providers=["CPUExecutionProvider"], ctx_id=-1
    )
    gender_age = InsightFaceGenderAgeEstimator(
        "genderage.onnx", providers=["CPUExecutionProvider"], ctx_id=-1
    )
    component = ComponentBackend(
        detector=detector,
        estimators=[pose, gender_age],
        enabled_tasks={"pose", "gender", "age"},
    )

    legacy = object.__new__(ONNXOnlyBackend)
    legacy.det_model = detector.model
    legacy.landmark_model = pose.model
    legacy.genderage_model = gender_age.model
    image = np.zeros((120, 120, 3), dtype=np.uint8)

    old_face = legacy.detect_faces(image)[0]
    new_face = component.detect_faces(image)[0]

    # The component pipeline preserves detector precision for estimator crops;
    # public result assembly applies the same historical integer conversion.
    np.testing.assert_array_equal(new_face.bbox.astype("int"), old_face.bbox)
    np.testing.assert_array_equal(new_face.kps, old_face.kps)
    assert new_face.det_score == old_face.det_score
    assert new_face.pose == old_face.pose
    assert new_face.gender == old_face.gender
    assert new_face.age == old_face.age
