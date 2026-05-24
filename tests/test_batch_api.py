"""Tests for the optional batched API."""

import numpy as np
import pytest

import forensicface.app as app_module
from forensicface.app import ForensicFace
from forensicface.backends import FaceBackend, FaceData
from forensicface.results import FaceResult


class _DummyInput:
    name = "input"


class _BatchRecSession:
    """Mock recognition session that respects the batch dimension."""

    def __init__(self, dim: int = 4, two_outputs: bool = False):
        self.dim = dim
        self.two_outputs = two_outputs
        self.last_input_shape: tuple | None = None
        self.last_input_keys: tuple[str, ...] | None = None
        self.call_count = 0

    def get_inputs(self):
        return [_DummyInput()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, inputs):
        self.call_count += 1
        self.last_input_keys = tuple(inputs.keys())
        batch = next(iter(inputs.values()))
        self.last_input_shape = tuple(batch.shape)
        n = batch.shape[0]
        if self.two_outputs:
            normed = np.full((n, self.dim), 0.5, dtype=np.float32)
            norm = np.full((n, 1), 2.0, dtype=np.float32)
            return [normed, norm]
        return [np.full((n, self.dim), 0.5, dtype=np.float32)]


class _ImageInput:
    name = "input_images"


class _KeypointsInput:
    name = "keypoints"


class _BatchKPRPESession:
    """Mock recognition session that consumes both an image batch and a
    keypoints batch — matches the contract of `KEYPOINT_RECOGNITION_MODELS`
    (e.g. sepaelv6/KPRPE)."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.last_inputs: dict | None = None
        self.call_count = 0

    def get_inputs(self):
        return [_ImageInput(), _KeypointsInput()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, inputs):
        self.call_count += 1
        self.last_inputs = {k: v.shape for k, v in inputs.items()}
        batch = inputs["input_images"]
        n = batch.shape[0]
        return [np.full((n, self.dim), 0.5, dtype=np.float32)]


class _BatchFIQASession:
    """Mock CR-FIQA session — returns (logits, quality) with N rows each."""

    def __init__(self):
        self.last_input_shape: tuple | None = None
        self.call_count = 0

    def get_inputs(self):
        return [_DummyInput()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, inputs):
        self.call_count += 1
        batch = next(iter(inputs.values()))
        self.last_input_shape = tuple(batch.shape)
        n = batch.shape[0]
        logits = np.zeros((n, 2), dtype=np.float32)
        quality = np.linspace(0.1, 0.9, n, dtype=np.float32).reshape(-1, 1)
        return [logits, quality]


class _DummyBackend(FaceBackend):
    """Backend that returns N synthetic faces with stable bbox/kps."""

    def __init__(self, n_faces: int = 1, with_attributes: bool = False):
        self.n_faces = n_faces
        self.with_attributes = with_attributes

    def detect_faces(self, _bgr_img):
        faces = []
        for i in range(self.n_faces):
            face = FaceData(
                bbox=np.array([10 + i, 10 + i, 60 + i, 60 + i], dtype=np.int32),
                kps=np.array(
                    [[20, 20], [40, 20], [30, 30], [22, 40], [38, 40]],
                    dtype=np.float32,
                ),
                det_score=0.9 + 0.01 * i,
            )
            if self.with_attributes:
                face.gender = i % 2  # alternates 0/1
                face.age = 20 + i
                face.pose = np.array([0.1, 0.2, 0.3], dtype=np.float32) * (i + 1)
            faces.append(face)
        return faces

    def norm_crop(self, _bgr_img, _keypoints):
        return np.full((112, 112, 3), 128, dtype=np.uint8)

    def estimate_norm(self, _keypoints):
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


def _make_ff(
    monkeypatch,
    *,
    extended: bool = False,
    n_faces: int = 1,
    with_attributes: bool = False,
    dim: int = 4,
    two_outputs: bool = False,
    n_models: int = 1,
    concat_embeddings: bool = True,
):
    """Builds a ForensicFace instance wired to in-memory mocks."""
    rec_sessions = [
        _BatchRecSession(dim=dim, two_outputs=two_outputs) for _ in range(n_models)
    ]
    rec_iter = iter(rec_sessions)
    monkeypatch.setattr(
        ForensicFace,
        "_load_model",
        lambda *_args, **_kwargs: next(rec_iter),
    )
    if extended:
        monkeypatch.setattr(
            app_module,
            "resolve_quality_model",
            lambda *_args, **_kwargs: "dummy_cr_fiqa_l.onnx",
        )
        monkeypatch.setattr(
            app_module.onnxruntime,
            "InferenceSession",
            lambda *_a, **_k: _BatchFIQASession(),
        )

    ff = ForensicFace(
        models=[f"dummy{i}" for i in range(n_models)],
        extended=extended,
        concat_embeddings=concat_embeddings,
        backend=_DummyBackend(n_faces=n_faces, with_attributes=with_attributes),
    )
    return ff, rec_sessions


# ---------------------------------------------------------------------------
# detect_and_align
# ---------------------------------------------------------------------------

def test_old_detect_and_align_alias_is_not_kept():
    assert "align" + "_only" not in dir(ForensicFace)


def test_detect_and_align_single_face_returns_dict(monkeypatch):
    ff, _ = _make_ff(monkeypatch, n_faces=1)
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    result = ff.detect_and_align(img, single_face=True)

    assert isinstance(result, dict)
    assert isinstance(result, FaceResult)
    assert result["aligned_face"].shape == (112, 112, 3)
    assert result["aligned_face"].dtype == np.uint8
    assert result["bbox"].shape == (4,)
    assert np.array_equal(result.bbox, result["bbox"])
    assert result["keypoints"].shape == (5, 2)
    assert isinstance(result["det_score"], float)


def test_detect_and_align_multi_face_returns_list(monkeypatch):
    ff, _ = _make_ff(monkeypatch, n_faces=3)
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    result = ff.detect_and_align(img, single_face=False)

    assert isinstance(result, list)
    assert len(result) == 3
    for item in result:
        assert item["aligned_face"].shape == (112, 112, 3)


def test_detect_and_align_single_face_returns_none_when_no_face(monkeypatch):
    ff, _ = _make_ff(monkeypatch, n_faces=0)
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    assert ff.detect_and_align(img, single_face=True) is None


def test_detect_and_align_multi_face_returns_empty_list_when_no_face(monkeypatch):
    ff, _ = _make_ff(monkeypatch, n_faces=0)
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    assert ff.detect_and_align(img, single_face=False) == []


def test_detect_and_align_extended_includes_attribute_fields(monkeypatch):
    ff, _ = _make_ff(monkeypatch, n_faces=1, with_attributes=True, extended=True)
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    result = ff.detect_and_align(img, single_face=True)
    assert "gender" in result and result["gender"] in ("M", "F")
    assert "age" in result and isinstance(result["age"], int)
    assert result["pose"].shape == (3,)


def test_detect_and_align_non_extended_omits_attribute_fields(monkeypatch):
    ff, _ = _make_ff(
        monkeypatch, n_faces=1, with_attributes=True, extended=False
    )
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    result = ff.detect_and_align(img, single_face=True)
    assert "gender" not in result
    assert "age" not in result
    assert "pose" not in result


# ---------------------------------------------------------------------------
# process_aligned_faces_batch
# ---------------------------------------------------------------------------

def test_process_aligned_faces_batch_concat_shape(monkeypatch):
    ff, rec_sessions = _make_ff(monkeypatch, dim=4, two_outputs=False, n_models=2)
    batch = np.zeros((5, 112, 112, 3), dtype=np.uint8)

    results = ff.process_aligned_faces_batch(batch)
    embeddings = np.stack([item.embedding for item in results], axis=0)

    assert embeddings.shape == (5, 8)  # 2 models × 4 dims each, concatenated
    assert all("fiqa_score" not in item for item in results)
    for sess in rec_sessions:
        assert sess.last_input_shape == (5, 3, 112, 112)
        assert sess.call_count == 1


def test_process_aligned_faces_batch_without_concat_returns_per_model_results(monkeypatch):
    ff, _ = _make_ff(
        monkeypatch, dim=4, n_models=2, concat_embeddings=False
    )
    batch = np.zeros((3, 112, 112, 3), dtype=np.uint8)

    results = ff.process_aligned_faces_batch(batch)

    assert len(results) == 3
    for item in results:
        assert "embedding" not in item
        assert item["embedding_dummy0"].shape == (4,)
        assert item["embedding_dummy1"].shape == (4,)
        assert "fiqa_score" not in item


def test_process_aligned_faces_batch_two_outputs_multiplies(monkeypatch):
    ff, _ = _make_ff(monkeypatch, dim=4, two_outputs=True, n_models=1)
    batch = np.zeros((2, 112, 112, 3), dtype=np.uint8)

    results = ff.process_aligned_faces_batch(batch)
    embeddings = np.stack([item.embedding for item in results], axis=0)
    # normed=0.5, norm=2.0 → 0.5*2.0=1.0
    np.testing.assert_allclose(embeddings, np.full((2, 4), 1.0, dtype=np.float32))


def test_process_aligned_faces_batch_fiqa_shape_when_extended(monkeypatch):
    ff, _ = _make_ff(monkeypatch, extended=True, n_models=1, dim=4)
    batch = np.zeros((7, 112, 112, 3), dtype=np.uint8)

    results = ff.process_aligned_faces_batch(batch)
    embeddings = np.stack([item.embedding for item in results], axis=0)
    fiqa = np.array([item.fiqa_score for item in results])
    assert embeddings.shape == (7, 4)
    assert fiqa.shape == (7,)


def test_process_aligned_faces_batch_matches_single_path_with_fake_session(monkeypatch):
    """The batch and single aligned-face APIs expose matching fake-session outputs."""
    ff, _ = _make_ff(monkeypatch, dim=4, two_outputs=True, n_models=1)
    batch = np.random.RandomState(0).randint(
        0, 256, size=(3, 112, 112, 3), dtype=np.uint8
    )

    batch_results = ff.process_aligned_faces_batch(batch)
    batch_emb = np.stack([item.embedding for item in batch_results], axis=0)
    single_embs = np.stack(
        [ff.process_aligned_face_image(crop).embedding for crop in batch], axis=0
    )
    np.testing.assert_allclose(batch_emb, single_embs)


def test_process_aligned_faces_batch_rejects_wrong_shape(monkeypatch):
    ff, _ = _make_ff(monkeypatch)
    bad = np.zeros((3, 64, 64, 3), dtype=np.uint8)  # 64 ≠ 112

    with pytest.raises(ValueError, match="rgb_aligned_faces"):
        ff.process_aligned_faces_batch(bad)


# ---------------------------------------------------------------------------
# process_images_batch
# ---------------------------------------------------------------------------

def test_process_images_batch_single_face_returns_list_of_dicts(monkeypatch):
    ff, _ = _make_ff(monkeypatch, n_faces=1)
    imgs = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(3)]

    results = ff.process_images_batch(imgs, single_face=True)

    assert len(results) == 3
    for item in results:
        assert isinstance(item, dict)
        assert isinstance(item, FaceResult)
        assert item["embedding"].shape == (4,)
        assert item.embedding.shape == (4,)
        assert item["aligned_face"].shape == (112, 112, 3)


def test_process_images_batch_returns_none_for_no_face_slot(monkeypatch):
    """Mixes images with 1 face and images with no face; verifies that
    None lands in the exact slots where detection failed."""
    face_counts = iter([1, 0, 1])
    rec_session = _BatchRecSession(dim=4)
    monkeypatch.setattr(
        ForensicFace, "_load_model", lambda *_a, **_k: rec_session
    )

    class _VariableBackend(_DummyBackend):
        def detect_faces(self, bgr_img):
            count = next(face_counts)
            self.n_faces = count
            return super().detect_faces(bgr_img)

    ff = ForensicFace(
        models=["dummy"],
        extended=False,
        backend=_VariableBackend(n_faces=1),
    )

    imgs = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(3)]
    results = ff.process_images_batch(imgs, single_face=True)

    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None


def test_process_images_batch_uses_single_onnx_call_per_chunk(monkeypatch):
    ff, rec_sessions = _make_ff(monkeypatch, n_faces=1, dim=4)
    imgs = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(5)]

    ff.process_images_batch(imgs, single_face=True, batch_size=32)
    assert rec_sessions[0].call_count == 1  # one chunk covers all 5


def test_process_images_batch_respects_chunk_boundary(monkeypatch):
    ff, rec_sessions = _make_ff(monkeypatch, n_faces=1, dim=4)
    imgs = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(7)]

    ff.process_images_batch(imgs, single_face=True, batch_size=3)
    # 7 valid faces, batch=3 → 3 chunks (3, 3, 1).
    assert rec_sessions[0].call_count == 3


def test_process_images_batch_empty_list_returns_empty(monkeypatch):
    ff, _ = _make_ff(monkeypatch)
    assert ff.process_images_batch([], single_face=True) == []


def test_process_images_batch_multi_face_returns_nested_list(monkeypatch):
    ff, _ = _make_ff(monkeypatch, n_faces=2)
    imgs = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(2)]

    results = ff.process_images_batch(imgs, single_face=False)

    assert len(results) == 2
    for face_list in results:
        assert isinstance(face_list, list)
        assert len(face_list) == 2
        for face in face_list:
            assert face["embedding"].shape == (4,)


def test_process_images_batch_emits_keys_compatible_with_process_image(monkeypatch):
    """The dicts emitted by process_images_batch must carry the same
    keys that process_image emits — so callers can swap APIs."""
    ff, _ = _make_ff(monkeypatch, n_faces=1, with_attributes=True, extended=True)
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    with pytest.warns(FutureWarning):
        single = ff.process_image(img, single_face=True)
    batched = ff.process_images_batch([img], single_face=True)[0]

    assert set(single.keys()) == set(batched.keys())


def test_process_images_batch_without_concat_returns_per_model_embeddings(monkeypatch):
    ff, _ = _make_ff(
        monkeypatch,
        n_faces=1,
        dim=4,
        n_models=2,
        concat_embeddings=False,
    )
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    result = ff.process_images_batch([img], single_face=True)[0]
    assert "embedding" not in result
    assert "embedding_dummy0" in result
    assert "embedding_dummy1" in result
    assert result["embedding_dummy0"].shape == (4,)


# ---------------------------------------------------------------------------
# sepaelv6 / KPRPE keypoint-aware batched API
# ---------------------------------------------------------------------------

def test_detect_and_align_includes_aligned_keypoints(monkeypatch):
    """`detect_and_align` must emit `aligned_keypoints` in the 112×112 frame so
    `process_images_batch` can stack them and feed KPRPE models."""
    ff, _ = _make_ff(monkeypatch, n_faces=1)
    img = np.zeros((128, 128, 3), dtype=np.uint8)

    result = ff.detect_and_align(img, single_face=True)
    assert "aligned_keypoints" in result
    assert result["aligned_keypoints"].shape == (5, 2)


def test_process_aligned_faces_batch_with_kprpe_passes_keypoints(monkeypatch):
    """When a loaded model is in KEYPOINT_RECOGNITION_MODELS (sepaelv6),
    `process_aligned_faces_batch` must build inputs with both `input_images`
    and `keypoints` — and keypoints must be normalized to [0, 1] by IMG_SIZE."""
    kprpe = _BatchKPRPESession(dim=4)
    monkeypatch.setattr(
        ForensicFace, "_load_model", lambda *_a, **_k: kprpe
    )
    ff = ForensicFace(
        models=["sepaelv6"],
        extended=False,
        backend=_DummyBackend(n_faces=1),
    )

    batch = np.zeros((3, 112, 112, 3), dtype=np.uint8)
    kps_batch = np.array(
        [
            [[10, 20], [30, 20], [56, 56], [22, 90], [38, 90]],
            [[12, 22], [32, 22], [58, 58], [24, 92], [40, 92]],
            [[14, 24], [34, 24], [60, 60], [26, 94], [42, 94]],
        ],
        dtype=np.float32,
    )

    results = ff.process_aligned_faces_batch(
        batch, aligned_keypoints_batch=kps_batch
    )
    embeddings = np.stack([item.embedding for item in results], axis=0)
    assert embeddings.shape == (3, 4)
    assert kprpe.call_count == 1
    # Both ONNX inputs present with batch dim N=3.
    assert kprpe.last_inputs["input_images"] == (3, 3, 112, 112)
    assert kprpe.last_inputs["keypoints"] == (3, 5, 2)


def test_process_aligned_faces_batch_kprpe_raises_when_keypoints_missing(monkeypatch):
    """sepaelv6 without keypoints must fail explicitly — not silently
    feed only images and produce garbage embeddings."""
    kprpe = _BatchKPRPESession(dim=4)
    monkeypatch.setattr(
        ForensicFace, "_load_model", lambda *_a, **_k: kprpe
    )
    ff = ForensicFace(
        models=["sepaelv6"],
        extended=False,
        backend=_DummyBackend(n_faces=1),
    )

    batch = np.zeros((2, 112, 112, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="requires aligned_keypoints_batch"):
        ff.process_aligned_faces_batch(batch)


def test_process_aligned_faces_batch_keypoints_shape_validation(monkeypatch):
    """Wrong shape on keypoints batch must raise before reaching ONNX."""
    kprpe = _BatchKPRPESession(dim=4)
    monkeypatch.setattr(
        ForensicFace, "_load_model", lambda *_a, **_k: kprpe
    )
    ff = ForensicFace(
        models=["sepaelv6"],
        extended=False,
        backend=_DummyBackend(n_faces=1),
    )
    batch = np.zeros((2, 112, 112, 3), dtype=np.uint8)

    # Wrong shape: (N, 3, 2) instead of (N, 5, 2).
    bad_kps = np.zeros((2, 3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \(N, 5, 2\)"):
        ff.process_aligned_faces_batch(batch, aligned_keypoints_batch=bad_kps)

    # Mismatched N between batch and keypoints.
    mismatched_kps = np.zeros((3, 5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="N=3.*N=2"):
        ff.process_aligned_faces_batch(
            batch, aligned_keypoints_batch=mismatched_kps,
        )


def test_process_images_batch_with_kprpe_end_to_end(monkeypatch):
    """End-to-end: process_images_batch collects aligned_keypoints from
    each detect_and_align result and feeds them to the KPRPE ONNX session."""
    kprpe = _BatchKPRPESession(dim=4)
    monkeypatch.setattr(
        ForensicFace, "_load_model", lambda *_a, **_k: kprpe
    )
    ff = ForensicFace(
        models=["sepaelv6"],
        extended=False,
        backend=_DummyBackend(n_faces=1),
    )

    imgs = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(4)]
    results = ff.process_images_batch(imgs, single_face=True, batch_size=32)

    assert len(results) == 4
    for item in results:
        assert isinstance(item, dict)
        assert item["embedding"].shape == (4,)
    # One ONNX call covered all 4 faces; both inputs were passed.
    assert kprpe.call_count == 1
    assert kprpe.last_inputs["keypoints"] == (4, 5, 2)


def test_process_images_batch_without_kprpe_skips_keypoint_stacking(monkeypatch):
    """When no loaded model is KPRPE, the batched path must NOT pay the
    extra cost of stacking aligned_keypoints — the session shouldn't
    receive a `keypoints` input at all."""
    ff, rec_sessions = _make_ff(monkeypatch, n_faces=1, dim=4, n_models=1)
    imgs = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(2)]

    ff.process_images_batch(imgs, single_face=True)
    assert rec_sessions[0].last_input_keys == ("input",)
    assert rec_sessions[0].last_input_shape == (2, 3, 112, 112)
