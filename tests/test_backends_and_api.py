import numpy as np
import pytest
import cv2
from pathlib import Path

import forensicface.app as app_module
from forensicface.app import ForensicFace
from forensicface.backends import FaceBackend, FaceData
from forensicface.forensicface import ForensicFace as LegacyForensicFace


class _DummyInput:
    name = "input"


class _DummyRecSession:
    def get_inputs(self):
        return [_DummyInput()]

    def run(self, _output_names, _inputs):
        # One embedding output tensor.
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
    assert "embedding" in result
    assert result["embedding"].shape == (4,)
    assert result["aligned_face"].shape == (112, 112, 3)


def test_process_image_multi_face_returns_list(monkeypatch):
    _patch_model_loading(monkeypatch)

    ff = ForensicFace(models=["dummy"], extended=False, backend=DummyBackend(n_faces=2))
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    result = ff.process_image(img, single_face=False)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all("embedding" in item for item in result)


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


def test_legacy_forensicface_alias_is_deprecated(monkeypatch):
    _patch_model_loading(monkeypatch)

    with pytest.warns(DeprecationWarning):
        ff = LegacyForensicFace(models=["dummy"], extended=False, backend=DummyBackend(n_faces=1))

    img = np.zeros((128, 128, 3), dtype=np.uint8)
    with pytest.warns(FutureWarning):
        result = ff.process_image(img, single_face=True)
    assert isinstance(result, dict)


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
        def __init__(self, model_file=None):
            self.model_file = model_file

        def prepare(self, *_args, **_kwargs):
            return None

    class _DummyGenderAgeModel:
        def __init__(self, model_file=None):
            self.model_file = model_file

        def prepare(self, *_args, **_kwargs):
            return None

    def _fake_scrfd(model_file=None):
        name = Path(model_file).name.lower()
        if "det" in name:
            return _DummyDetModel()
        raise RuntimeError("not a detection model")

    fake_files = [
        "/fake/model/det_10g.onnx",
        "/fake/model/1k3d68.onnx",
        "/fake/model/genderage.onnx",
    ]

    monkeypatch.setattr(backends_module.osp, "isdir", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(backends_module.glob, "glob", lambda *_args, **_kwargs: fake_files)
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
    )

    assert backend.det_model is not None
    assert backend.landmark_model is not None
    assert backend.genderage_model is not None


def test_load_model_uses_forensicface_model_root(monkeypatch):
    captured = {}

    def fake_glob(pattern):
        captured["pattern"] = pattern
        return ["/fake/path/face.onnx"]

    class _DummySession:
        pass

    monkeypatch.setattr(app_module, "glob", fake_glob)
    monkeypatch.setattr(app_module.onnxruntime, "InferenceSession", lambda *_a, **_k: _DummySession())

    ff = object.__new__(ForensicFace)
    ff._load_model(model_name="sepaelv2", use_gpu=False, gpu=0)

    assert captured["pattern"].startswith(str(Path.home() / ".forensicface" / "models" / "sepaelv2"))
    assert "*face*.onnx" in captured["pattern"]


def test_onnx_backend_missing_directory_message_uses_forensicface_root(monkeypatch):
    import forensicface.backends as backends_module

    monkeypatch.setattr(backends_module.osp, "isdir", lambda *_args, **_kwargs: False)

    with pytest.raises(FileNotFoundError, match=r"~/.forensicface/models"):
        backends_module.ONNXOnlyBackend(
            model_name="sepaelv2",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
            ctx_id=-1,
            det_size=(320, 320),
            det_thresh=0.5,
        )
