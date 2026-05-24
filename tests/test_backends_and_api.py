import os.path as osp

import numpy as np
import pytest
import cv2
from pathlib import Path

import forensicface.app as app_module
import forensicface.model_store as model_store
from forensicface.app import ForensicFace
from forensicface.backends import FaceBackend, FaceData
from forensicface.results import FaceResult


class _DummyInput:
    name = "input"


class _NamedInput:
    def __init__(self, name):
        self.name = name


class _DummyRecSession:
    def get_inputs(self):
        return [_DummyInput()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, _inputs):
        # One embedding output tensor.
        return [np.ones((1, 4), dtype=np.float32)]


class _CapturingRecSession:
    def __init__(self, input_names):
        self._inputs = [_NamedInput(name) for name in input_names]
        self.captured_inputs = None

    def get_inputs(self):
        return self._inputs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, inputs):
        self.captured_inputs = inputs
        return [np.ones((1, 4), dtype=np.float32)]


class DummyBackend(FaceBackend):
    def __init__(self, n_faces=1):
        self.n_faces = n_faces

    def detect_faces(self, _bgr_img: np.ndarray) -> list[FaceData]:
        faces = []
        for i in range(self.n_faces):
            faces.append(
                FaceData(
                    bbox=np.array([10 + i, 10 + i, 60 + i, 60 + i], dtype=np.int32),
                    kps=np.array(
                        [[20, 20], [40, 20], [30, 30], [22, 40], [38, 40]],
                        dtype=np.float32,
                    ),
                    det_score=0.99,
                )
            )
        return faces

    def norm_crop(self, _bgr_img: np.ndarray, _keypoints: np.ndarray) -> np.ndarray:
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def estimate_norm(self, _keypoints: np.ndarray) -> np.ndarray:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


class _DummyProviderSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return self._providers


class _DummyBackendWithProvider(DummyBackend):
    def __init__(self, provider_name: str):
        super().__init__(n_faces=1)
        self.det_model = type("_DetModel", (), {"session": _DummyProviderSession([provider_name])})()


def _patch_model_loading(monkeypatch):
    monkeypatch.setattr(ForensicFace, "_load_model", lambda *_args, **_kwargs: _DummyRecSession())


def test_backend_contract_enforced_for_abstract_base():
    class BrokenBackend(FaceBackend):
        pass

    with pytest.raises(TypeError):
        BrokenBackend()


def test_process_image_single_face_api_shape(monkeypatch):
    _patch_model_loading(monkeypatch)

    ff = ForensicFace(models=["dummy"], extended=False, backend=DummyBackend(n_faces=1))
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    with pytest.warns(FutureWarning):
        result = ff.process_image(img, single_face=True)

    assert isinstance(result, dict)
    assert isinstance(result, FaceResult)
    assert "embedding" in result
    assert result["embedding"].shape == (4,)
    assert result.embedding.shape == (4,)
    assert np.array_equal(result.bbox, result["bbox"])
    assert result["aligned_face"].shape == (112, 112, 3)


def test_process_image_multi_face_returns_list(monkeypatch):
    _patch_model_loading(monkeypatch)

    ff = ForensicFace(models=["dummy"], extended=False, backend=DummyBackend(n_faces=2))
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    result = ff.process_image(img, single_face=False)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all("embedding" in item for item in result)


def test_compare_rejects_non_concatenated_embeddings(monkeypatch):
    ff = ForensicFace.__new__(ForensicFace)
    ff.concat_embeddings = False
    ff.models = ["model_a", "model_b"]

    with pytest.raises(ValueError, match="concat_embeddings=False"):
        ff.compare("img1", "img2")


def test_compare_rejects_missing_faces(monkeypatch):
    ff = ForensicFace.__new__(ForensicFace)
    ff.concat_embeddings = True

    monkeypatch.setattr(ff, "process_image", lambda *_args, **_kwargs: [])

    with pytest.raises(ValueError, match="No face detected"):
        ff.compare("img1", "img2")


def test_aggregate_from_images_rejects_quality_weight_without_extended():
    ff = ForensicFace.__new__(ForensicFace)
    ff.extended = False

    with pytest.raises(ValueError, match="extended = True"):
        ff.aggregate_from_images(["img1"], quality_weight=True)


def test_aggregate_from_images_supports_non_concatenated_embeddings(monkeypatch):
    ff = ForensicFace.__new__(ForensicFace)
    ff.concat_embeddings = False
    ff.models = ["model_a", "model_b"]
    ff.extended = True

    outputs = {
        "img1": {
            "embedding_model_a": np.array([1.0, 3.0], dtype=np.float32),
            "embedding_model_b": np.array([10.0, 30.0], dtype=np.float32),
            "fiqa_score": 1.0,
        },
        "img2": {
            "embedding_model_a": np.array([3.0, 5.0], dtype=np.float32),
            "embedding_model_b": np.array([30.0, 50.0], dtype=np.float32),
            "fiqa_score": 3.0,
        },
    }

    def _fake_process_image(imgpath, single_face=True):
        assert single_face is True
        return outputs[imgpath]

    monkeypatch.setattr(ff, "process_image", _fake_process_image)

    aggregated = ff.aggregate_from_images(
        ["img1", "img2"],
        quality_weight=True,
    )

    assert set(aggregated) == {"embedding_model_a", "embedding_model_b"}
    np.testing.assert_allclose(aggregated["embedding_model_a"], [2.5, 4.5])
    np.testing.assert_allclose(aggregated["embedding_model_b"], [25.0, 45.0])


def test_sepaelv6_receives_aligned_normalized_keypoints(monkeypatch):
    rec_session = _CapturingRecSession(["input_images", "keypoints"])
    monkeypatch.setattr(
        ForensicFace, "_load_model", lambda *_args, **_kwargs: rec_session
    )

    ff = ForensicFace(models=["sepaelv6"], extended=False, backend=DummyBackend(n_faces=1))
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    with pytest.warns(FutureWarning):
        result = ff.process_image(img, single_face=True)

    assert result["embedding"].shape == (4,)
    assert set(rec_session.captured_inputs) == {"input_images", "keypoints"}
    assert rec_session.captured_inputs["input_images"].shape == (1, 3, 112, 112)
    assert rec_session.captured_inputs["input_images"].dtype == np.float32
    assert rec_session.captured_inputs["keypoints"].shape == (1, 5, 2)
    assert rec_session.captured_inputs["keypoints"].dtype == np.float32
    expected_keypoints = np.array(
        [[[20, 20], [40, 20], [30, 30], [22, 40], [38, 40]]],
        dtype=np.float32,
    ) / 112.0
    assert np.allclose(rec_session.captured_inputs["keypoints"], expected_keypoints)


def test_sepaelv6_validates_required_onnx_input_names(monkeypatch):
    rec_session = _CapturingRecSession(["input_images"])
    monkeypatch.setattr(
        ForensicFace, "_load_model", lambda *_args, **_kwargs: rec_session
    )

    ff = ForensicFace(models=["sepaelv6"], extended=False, backend=DummyBackend(n_faces=1))
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError, match="Missing: \\['keypoints'\\]"):
            ff.process_image(img, single_face=True)


def test_process_aligned_face_image_accepts_keypoints_for_sepaelv6(monkeypatch):
    rec_session = _CapturingRecSession(["input_images", "keypoints"])
    monkeypatch.setattr(
        ForensicFace, "_load_model", lambda *_args, **_kwargs: rec_session
    )

    ff = ForensicFace(models=["sepaelv6"], extended=False, backend=DummyBackend(n_faces=1))
    rgb_aligned_face = np.zeros((112, 112, 3), dtype=np.uint8)
    aligned_keypoints = np.array(
        [[20, 20], [40, 20], [30, 30], [22, 40], [38, 40]],
        dtype=np.float32,
    )

    result = ff.process_aligned_face_image(
        rgb_aligned_face, keypoints=aligned_keypoints
    )

    assert isinstance(result, FaceResult)
    assert result["embedding"].shape == (4,)
    assert result.embedding.shape == (4,)
    assert np.allclose(
        rec_session.captured_inputs["keypoints"],
        aligned_keypoints.reshape(1, 5, 2) / 112.0,
    )


def test_process_aligned_face_image_requires_keypoints_for_sepaelv6(monkeypatch):
    rec_session = _CapturingRecSession(["input_images", "keypoints"])
    monkeypatch.setattr(
        ForensicFace, "_load_model", lambda *_args, **_kwargs: rec_session
    )

    ff = ForensicFace(models=["sepaelv6"], extended=False, backend=DummyBackend(n_faces=1))
    rgb_aligned_face = np.zeros((112, 112, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="requires aligned 5-point keypoints"):
        ff.process_aligned_face_image(rgb_aligned_face)


def test_process_aligned_face_image_rejects_wrong_shape(monkeypatch):
    _patch_model_loading(monkeypatch)

    ff = ForensicFace(models=["dummy"], extended=False, backend=DummyBackend(n_faces=1))
    rgb_aligned_face = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="rgb_aligned_face"):
        ff.process_aligned_face_image(rgb_aligned_face)


def test_backend_name_is_forwarded_to_factory(monkeypatch):
    _patch_model_loading(monkeypatch)

    captured = {}

    def fake_create_backend(**kwargs):
        captured["backend_name"] = kwargs["backend_name"]
        return DummyBackend(n_faces=1)

    monkeypatch.setattr(app_module, "create_backend", fake_create_backend)

    ff = ForensicFace(models=["dummy"], extended=False, backend_name="onnx")
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    with pytest.warns(FutureWarning):
        result = ff.process_image(img, single_face=True)

    assert captured["backend_name"] == "onnx"
    assert isinstance(result, dict)


def test_initialization_prints_loaded_models_det_size_and_provider(monkeypatch, capsys):
    _patch_model_loading(monkeypatch)

    ff = ForensicFace(models=["dummy"], extended=False, backend=DummyBackend(n_faces=1))

    captured = capsys.readouterr()
    assert "[ForensicFace] Initialized" in captured.out
    assert "loaded_models=['dummy']" in captured.out
    assert "det_size=(320, 320)" in captured.out
    assert "session_providers=all models use CPUExecutionProvider" in captured.out
    assert ff is not None


def test_initialization_reports_effective_detector_provider(monkeypatch, capsys):
    _patch_model_loading(monkeypatch)

    ff = ForensicFace(
        models=["dummy"],
        extended=False,
        backend=_DummyBackendWithProvider("CUDAExecutionProvider"),
    )

    captured = capsys.readouterr()
    assert "session_providers=dummy: CPUExecutionProvider, detection: CUDAExecutionProvider" in captured.out
    assert ff is not None


def test_onnx_backend_propagates_optional_fields(monkeypatch):
    # Build backend instance without running its __init__ filesystem/model loading.
    from forensicface.backends import ONNXOnlyBackend

    backend = object.__new__(ONNXOnlyBackend)

    class _DummyDet:
        def detect(self, _img, max_num=0, metric="default"):
            bboxes = np.array([[10, 10, 60, 60, 0.98]], dtype=np.float32)
            kpss = np.array([[[20, 20], [40, 20], [30, 30], [22, 40], [38, 40]]], dtype=np.float32)
            return bboxes, kpss

    class _DummyLandmark:
        def get(self, _img, face):
            face["pose"] = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    class _DummyGenderAge:
        def get(self, _img, face):
            face["gender"] = 1
            face["age"] = 33

    backend.det_model = _DummyDet()
    backend.landmark_model = _DummyLandmark()
    backend.genderage_model = _DummyGenderAge()

    img = np.zeros((128, 128, 3), dtype=np.uint8)
    faces = backend.detect_faces(img)

    assert len(faces) == 1
    assert faces[0].gender == 1
    assert faces[0].age == 33
    assert np.allclose(faces[0].pose, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_process_image_draw_keypoints_saves_aligned_image_onnx_backend(
    monkeypatch, tmp_path
):
    _patch_model_loading(monkeypatch)
    image_path = Path("nbs/obama.png")
    if not image_path.exists():
        pytest.skip("Sample image nbs/obama.png not found")

    try:
        ff = ForensicFace(
            models=["sepaelv2"],
            extended=False,
            use_gpu=False,
            backend_name="onnx",
        )
    except Exception as exc:
        pytest.skip(f"ONNX backend unavailable in this environment: {exc}")

    with pytest.warns(FutureWarning):
        result = ff.process_image(
            str(image_path),
            single_face=True,
            draw_keypoints=True,
        )

    if not result:
        pytest.skip("No face detected with ONNX backend on sample image")

    out_file = tmp_path / "aligned_onnx_draw_keypoints.png"
    aligned_bgr = cv2.cvtColor(result["aligned_face"], cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(out_file), aligned_bgr)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_extract_faces_processes_small_video_and_exports_metadata(monkeypatch, tmp_path):
    _patch_model_loading(monkeypatch)

    video_path = tmp_path / "dummy_video.avi"
    frame_size = (128, 128)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        frame_size,
    )
    assert writer.isOpened()
    for i in range(3):
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"F{i}",
            (15, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()

    ff = ForensicFace(models=["dummy"], extended=False, backend=DummyBackend(n_faces=1))
    out_dir = tmp_path / "faces_out"
    nfaces = ff.extract_faces(
        str(video_path),
        dest_folder=str(out_dir),
        every_n_frames=1,
        export_metadata=True,
    )

    assert nfaces == 3
    extracted_faces = sorted(out_dir.glob("frame_*_face_*.png"))
    assert len(extracted_faces) == 3

    metadata_file = out_dir / f"{video_path.stem}.jsonl"
    assert metadata_file.exists()
    metadata_lines = metadata_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(metadata_lines) == 3


def test_onnx_backend_loads_landmark_3d68_when_requested(monkeypatch):
    import forensicface.backends as backends_module

    class _DummyDetModel:
        taskname = "detection"

        def prepare(self, *_args, **_kwargs):
            return None

    class _DummyLandmarkModel:
        def __init__(self, model_file=None, **_kwargs):
            self.model_file = model_file

        def prepare(self, *_args, **_kwargs):
            return None

    class _DummyGenderAgeModel:
        def __init__(self, model_file=None, **_kwargs):
            self.model_file = model_file

        def prepare(self, *_args, **_kwargs):
            return None

    def _fake_scrfd(model_file=None, **_kwargs):
        name = Path(model_file).name.lower()
        if "det" in name:
            return _DummyDetModel()
        raise RuntimeError("not a detection model")

    fake_files = [
        "/fake/model/det_10g.onnx",
        "/fake/model/1k3d68.onnx",
        "/fake/model/genderage.onnx",
    ]

    monkeypatch.setattr(model_store.osp, "isdir", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(model_store, "glob", lambda *_args, **_kwargs: fake_files)
    monkeypatch.setattr(backends_module, "SCRFD", _fake_scrfd)
    monkeypatch.setattr(backends_module, "LandmarkONNX", _DummyLandmarkModel)
    monkeypatch.setattr(backends_module, "AttributeONNX", _DummyGenderAgeModel)

    backend = backends_module.ONNXOnlyBackend(
        model_name="sepaelv2",
        allowed_modules=["detection", "landmark_3d_68", "genderage"],
        providers=["CPUExecutionProvider"],
        ctx_id=-1,
        det_size=(320, 320),
        det_thresh=0.5,
        models_root="/fake/model",
    )

    assert backend.det_model is not None
    assert backend.landmark_model is not None
    assert backend.genderage_model is not None


def test_load_model_prefers_new_recognition_layout(monkeypatch):
    captured = {"patterns": []}

    def fake_glob(pattern):
        captured["patterns"].append(pattern)
        return ["/fake/path/face.onnx"]

    class _DummySession:
        pass

    monkeypatch.setattr(model_store, "glob", fake_glob)
    monkeypatch.setattr(app_module.onnxruntime, "InferenceSession", lambda *_a, **_k: _DummySession())

    ff = object.__new__(ForensicFace)
    models_root = str(Path.home() / ".forensicface" / "models")
    ff._load_model(
        model_name="sepaelv2",
        providers=["CPUExecutionProvider"],
        gpu=0,
        models_root=models_root,
    )

    new_prefix = str(Path(models_root) / "recognition" / "sepaelv2")
    assert captured["patterns"][0].startswith(new_prefix)
    assert "*face*.onnx" in captured["patterns"][0]


def test_load_model_falls_back_to_legacy_layout(monkeypatch):
    captured = {"patterns": []}

    def fake_glob(pattern):
        captured["patterns"].append(pattern)
        # First call (new layout) returns empty; second call (legacy) returns a hit.
        return [] if len(captured["patterns"]) == 1 else ["/fake/path/face.onnx"]

    class _DummySession:
        pass

    monkeypatch.setattr(model_store, "glob", fake_glob)
    monkeypatch.setattr(app_module.onnxruntime, "InferenceSession", lambda *_a, **_k: _DummySession())

    ff = object.__new__(ForensicFace)
    models_root = str(Path.home() / ".forensicface" / "models")
    ff._load_model(
        model_name="sepaelv2",
        providers=["CPUExecutionProvider"],
        gpu=0,
        models_root=models_root,
    )

    assert len(captured["patterns"]) == 2
    legacy_prefix = str(Path(models_root) / "sepaelv2")
    assert captured["patterns"][1].startswith(legacy_prefix)
    assert "*face*.onnx" in captured["patterns"][1]


def test_resolve_quality_model_prefers_new_layout(monkeypatch, tmp_path):
    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()
    new_path = quality_dir / "cr_fiqa_l.onnx"
    new_path.write_bytes(b"fake-onnx")

    resolved = model_store.resolve_quality_model(str(tmp_path), "sepaelv2")
    assert Path(resolved) == new_path


def test_resolve_quality_model_falls_back_to_legacy_layout(tmp_path):
    legacy_dir = tmp_path / "sepaelv2" / "cr_fiqa"
    legacy_dir.mkdir(parents=True)
    legacy_path = legacy_dir / "cr_fiqa_l.onnx"
    legacy_path.write_bytes(b"fake-onnx")

    resolved = model_store.resolve_quality_model(str(tmp_path), "sepaelv2")
    assert Path(resolved) == legacy_path


def test_resolve_quality_model_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="cr_fiqa_l.onnx"):
        model_store.resolve_quality_model(str(tmp_path), "sepaelv2")


def test_onnx_backend_missing_directory_message_uses_forensicface_root(monkeypatch):
    import forensicface.backends as backends_module

    monkeypatch.setattr(backends_module.osp, "isdir", lambda *_args, **_kwargs: False)

    with pytest.raises(FileNotFoundError, match=r"/fake/model"):
        backends_module.ONNXOnlyBackend(
            model_name="sepaelv2",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
            ctx_id=-1,
            det_size=(320, 320),
            det_thresh=0.5,
            models_root="/fake/model",
        )


def test_backend_prefers_new_shared_structure_over_legacy(monkeypatch):
    import forensicface.backends as backends_module

    new_det = "/fake/model/detection/det_10g.onnx"
    legacy_det = "/fake/model/sepaelv2/det_10g.onnx"

    glob_calls = {"patterns": []}

    def fake_glob(pattern):
        glob_calls["patterns"].append(pattern)
        if pattern.endswith(osp.join("detection", "*.onnx")):
            return [new_det]
        if pattern.endswith(osp.join("attributes", "*.onnx")):
            return []
        if pattern.endswith(osp.join("sepaelv2", "*.onnx")):
            return [legacy_det]
        return []

    class _DummyDetModel:
        taskname = "detection"

        def __init__(self, model_file):
            self.model_file = model_file

        def prepare(self, *_args, **_kwargs):
            return None

    def fake_scrfd(model_file=None, **_kwargs):
        return _DummyDetModel(model_file)

    monkeypatch.setattr(model_store.osp, "isdir", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(model_store, "glob", fake_glob)
    monkeypatch.setattr(backends_module, "SCRFD", fake_scrfd)

    backend = backends_module.ONNXOnlyBackend(
        model_name="sepaelv2",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
        ctx_id=-1,
        det_size=(320, 320),
        det_thresh=0.5,
        models_root="/fake/model",
    )

    assert backend.det_model.model_file == new_det


def test_backend_falls_back_to_legacy_when_new_layout_missing(monkeypatch):
    import forensicface.backends as backends_module

    legacy_dir = osp.join("/fake/model", "sepaelv2")
    legacy_det = osp.join(legacy_dir, "det_10g.onnx")

    def fake_isdir(path):
        return osp.normpath(path) == osp.normpath(legacy_dir)

    def fake_glob(pattern):
        if pattern.endswith(osp.join("sepaelv2", "*.onnx")):
            return [legacy_det]
        return []

    class _DummyDetModel:
        taskname = "detection"

        def __init__(self, model_file):
            self.model_file = model_file

        def prepare(self, *_args, **_kwargs):
            return None

    def fake_scrfd(model_file=None, **_kwargs):
        return _DummyDetModel(model_file)

    monkeypatch.setattr(model_store.osp, "isdir", fake_isdir)
    monkeypatch.setattr(model_store, "glob", fake_glob)
    monkeypatch.setattr(backends_module, "SCRFD", fake_scrfd)

    backend = backends_module.ONNXOnlyBackend(
        model_name="sepaelv2",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
        ctx_id=-1,
        det_size=(320, 320),
        det_thresh=0.5,
        models_root="/fake/model",
    )

    assert backend.det_model.model_file == legacy_det


def test_collect_onnx_files_dedupes_across_sources(monkeypatch):
    duplicate = "/fake/model/detection/det_10g.onnx"

    monkeypatch.setattr(model_store.osp, "isdir", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(model_store, "glob", lambda *_args, **_kwargs: [duplicate])

    files = model_store.collect_backend_model_files("/fake/model", "sepaelv2")
    assert files == [duplicate]
