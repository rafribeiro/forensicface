# Extending ForensicFace Models

ForensicFace supports built-in ONNX aliases, `ModelSpec` configuration, and
direct injection of constructed component objects. The built-in catalog is
intentionally immutable: custom models are integrated by implementing the
relevant contract and passing the object to `ForensicFace`.

## Shared conventions

- Public image inputs and component image inputs are OpenCV BGR arrays.
- Public `aligned_face` results are RGB arrays.
- Original-image bounding boxes use `(x1, y1, x2, y2)` coordinates and remain
  floating-point internally. Public result dictionaries retain integer boxes.
- Five-point landmarks have shape `(5, 2)` and are ordered left eye, right
  eye, nose, left mouth corner, right mouth corner in image coordinates.
- The canonical aligned crop is BGR with shape `(112, 112, 3)`.
- Aligned keypoints use the canonical 112-by-112 coordinate system.
- Pose is normalized internally as `PoseAngles(pitch, yaw, roll)`. A pose
  estimator may return `PoseAngles`, a mapping with those three names, or a
  legacy `[pitch, yaw, roll]` sequence.

Every component exposes `ComponentMetadata`. Operational metadata is required:
component ID, tasks, implementation, runtime, input-space description, and
batch support. Model path and providers should be supplied when available.

## Detector contract

A detector implements:

```python
class Detector:
    metadata: ComponentMetadata

    def detect(self, bgr_img: np.ndarray) -> list[FaceData]:
        ...
```

Every returned `FaceData` requires a finite four-value `bbox`, finite
`det_score`, and finite `(5, 2)` keypoints. The initial contract deliberately
does not accept box-only detectors because alignment and keypoint-aware
embeddings require five landmarks.

The detector owns all resizing, padding, normalization, decoding, NMS, and
coordinate rescaling. SCRFD and CenterFace therefore use independent input-size
policies without exposing detector tensors to the orchestrator.

## Face-estimator contract

Pose, gender, and age components implement:

```python
class Estimator:
    metadata: ComponentMetadata
    capabilities = frozenset({"gender", "age"})

    def estimate(self, context: FaceContext) -> Mapping[str, object]:
        ...
```

`FaceContext` provides:

- `original_bgr` and the normalized detection;
- canonical `aligned_bgr` and `aligned_keypoints`;
- the original-to-aligned affine transform.

An estimator chooses the appropriate source, crop, input dimensions, color
conversion, and normalization. There is no common estimator tensor shape.
Returned keys must be declared in `capabilities`. The same joint estimator
object or equivalent built-in specification is invoked once when it serves
both gender and age.

Face-estimator batching is intentionally not part of this implementation.
Detection, pose, gender, and age execute per image/face.

## Quality contract

A quality component implements `score_one(aligned_bgr)` and
`score_batch(aligned_bgr_batch)`. It owns preprocessing and returns scalar
scores using the existing public result name `fiqa_score`, including for
models other than CR-FIQA.

Quality batching is independent from embedding batching. CUDA out-of-memory
retry splits only the failing quality batch and does not rerun embeddings.

## Embedding contract

An embedding component exposes `name`, `metadata`,
`requires_aligned_keypoints`, `embed_one()`, and `embed_batch()`. Multiple
embedding objects can be supplied in a sequence. Names must be unique because
they identify per-model result keys when `concat_embeddings=False`.

## Configuration examples

Use aliases for built-in defaults:

```python
ff = ForensicFace(
    detection="centerface",
    pose="insightface-3d68",
    embedding=["sepaelv2", "sepaelv4"],
)
```

Use `ModelSpec` for a model path or adapter-specific options:

```python
ff = ForensicFace(
    detection=ModelSpec(
        "centerface",
        path="/models/centerface-dynamic.onnx",
        score_threshold=0.4,
        nms_threshold=0.3,
    ),
    embedding="sepaelv6",
)
```

Use constructed objects for custom implementations:

```python
ff = ForensicFace(
    detection=my_detector,
    pose=my_pose_estimator,
    gender=my_joint_gender_age,
    age=my_joint_gender_age,
    quality=my_quality_estimator,
    embedding=[my_embedding_a, my_embedding_b],
)
```

Any direct task selector is mutually exclusive with legacy `models`, `model`,
and `backend` configuration. `detection=None` is invalid. Optional tasks and
embeddings may be disabled explicitly with `None`.

## Providers, lifetime, and thread safety

Built-in components receive the provider selected during `ForensicFace`
initialization and own their ONNX Runtime sessions. Injected components own
their runtime and device configuration; the facade does not move them between
devices.

A `ForensicFace` instance owns its component instances. The library does not
promise that one instance is safe for simultaneous calls from multiple
threads. Extension authors should avoid mutable per-call buffers on component
objects, protect mutable runtime state when necessary, and document any
stronger concurrency guarantee their adapter provides.

## Model files and testing

Built-in files use `<models_root>/<task>/<alias>/*.onnx`. A `ModelSpec(path=...)`
may select a file directly. Model acquisition remains manual and offline.

Extension tests should cover empty detections, malformed outputs, coordinate
rescaling, single/batch equivalence where batching exists, model-native input
dimensions, result types, provider selection, and non-OOM error propagation.
Adapters replacing current models should also be compared against the legacy
pipeline on identical inputs.
