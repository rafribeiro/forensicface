# Codebase Simplification Plan

This document records a review of the current codebase with possible refactors.
It does not propose immediate behavior changes. The priority is to make the
library easier to understand and contribute to while keeping the user-facing API
stable.

## Executive Summary

The main source of complexity is that `ForensicFace` currently acts as both a
public API facade and a container for many stateless helper operations. That is
convenient for end users, but it makes `app.py` large and forces contributors
to reason about initialization, model lookup, preprocessing, result assembly,
batch chunking, comparison, aggregation, mosaics, and video extraction in one
class.

Update: the first implementation pass introduced `FaceResult(dict)` for public
face results and extracted initial helper modules for result assembly,
preprocessing, geometry, and model-store path resolution. The remaining items
below still describe the direction for follow-up refactors.

Recommended direction:

1. Keep `ForensicFace` as the end-user facade.
2. Move stateless data transformations into functions.
3. Move model discovery/loading into a small module.
4. Move result-shaping logic into typed helper functions.
5. Move video and mosaic workflows into task modules, while preserving
   convenience methods on `ForensicFace`.
6. Replace internal `assert` checks on user input with explicit exceptions.
7. Gradually standardize return types around lists for face results, while
   honoring existing deprecation warnings.

## What Is Already Working Well

- `FaceBackend` is a good boundary: `ForensicFace` does not depend directly on
  SCRFD internals for detection.
- Tests already use injected fake backends and fake ONNX sessions, which makes
  refactoring safer.
- The migration tool is already well separated into planning, applying,
  formatting, and CLI concerns.
- Runtime setup and initialization summary are already separate modules.
- `utils.py` contains simple stateless functions, which is the right pattern for
  reusable operations.

## Refactor Candidates

### 1. Move model path resolution out of `ForensicFace`

Current locations:

- `ForensicFace._load_model()` in `src/forensicface/app.py`
- `ForensicFace._resolve_quality_model()` in `src/forensicface/app.py`
- `ONNXOnlyBackend._collect_onnx_files()` in `src/forensicface/backends.py`

Problem:

Model lookup rules are duplicated conceptually across recognition, quality, and
backend model discovery. They are part of package configuration, not part of the
face-processing object itself.

Proposal:

Create `src/forensicface/model_store.py` with functions such as:

```python
resolve_recognition_model(models_root, model_name) -> Path
resolve_quality_model(models_root, model_name) -> Path
collect_backend_model_files(models_root, model_name) -> list[Path]
```

Keep `ForensicFace._load_model()` temporarily as a compatibility wrapper that
calls the new function and creates the ONNX session.

Benefit:

New contributors can understand model layout in one file. Tests for new and
legacy layout become focused and do not need a `ForensicFace` instance.

### 2. Move image preprocessing to pure functions

Current locations:

- `ForensicFace._to_input_ada()`
- `ForensicFace._to_keypoints_input()`
- some RGB/BGR conversion logic inside `process_images_batch()`

Problem:

Preprocessing is stateless except for constants and model names. Keeping it as
methods makes it harder to test independently and hides the BGR/RGB boundary.

Proposal:

Create `src/forensicface/preprocessing.py` with:

```python
to_ada_input(aligned_bgr, image_size=(112, 112)) -> np.ndarray
normalize_aligned_keypoints(keypoints, image_size=(112, 112)) -> np.ndarray
rgb_aligned_to_bgr_batch(items) -> np.ndarray
```

Keep thin methods on `ForensicFace` during a transition if downstream code may
use private methods.

Benefit:

Preprocessing becomes easier to test for shape, dtype, normalization, and color
ordering. It also prepares the codebase for recognition models with different
preprocessing requirements.

### 3. Introduce a recognition runner

Current locations:

- `ForensicFace._compute_embeddings()`
- `ForensicFace._compute_embeddings_batch()`
- `ForensicFace._build_keypoint_model_inputs()`
- `ForensicFace._try_compute_embeddings_batch()`

Problem:

Recognition inference has its own responsibilities: build model inputs, handle
single vs batch shapes, normalize keypoints, support two-output models, merge
multi-model embeddings, compute FIQA, and retry on CUDA OOM.

Proposal:

Create a small internal class or module, for example `recognition.py`:

```python
class RecognitionRunner:
    def compute_one(...)
    def compute_batch(...)
```

Use a class only if it owns sessions/configuration (`models`,
`rec_inference_sessions`, `ort_fiqa`, `concat_embeddings`, `extended`). Use
pure functions for smaller pieces such as output normalization.

Benefit:

`ForensicFace` remains the user facade, while recognition behavior gets a
single owner. Batch and single behavior can be tested against the same runner.

### 4. Deduplicate result assembly

Current locations:

- `ForensicFace._assemble_result()`
- `ForensicFace._assemble_result_from_align_only()`
- attribute formatting in `align_only()`

Problem:

There are two code paths that build process-compatible dictionaries. They share
the same fields but differ in source format: one starts from `FaceData` and BGR
aligned image; the other starts from an `align_only()` dictionary and RGB
aligned image.

Proposal:

Create an internal dataclass such as `AlignedFace`:

```python
@dataclass
class AlignedFace:
    aligned_rgb: np.ndarray
    bbox: np.ndarray
    keypoints: np.ndarray
    aligned_keypoints: np.ndarray
    det_score: float
    gender: str | None = None
    age: int | None = None
    pose: np.ndarray | None = None
```

Then build results through one function:

```python
assemble_face_result(aligned_face, embeddings, fiqa_score, models, extended, concat_embeddings)
```

Benefit:

This reduces duplicate logic for `ipd`, `gender`, `pose`, `embedding` keys, and
RGB conversion. It also makes output compatibility easier to enforce.

### 5. Move face selection and geometry helpers to functions

Current locations:

- `ForensicFace._get_best_face()`
- `ForensicFace._align_keypoints()`
- `ForensicFace._get_extended_bbox()`

Problem:

These are stateless calculations except for backend use in `_align_keypoints()`.
They can be named by intent and tested without constructing the full facade.

Proposal:

Create `src/forensicface/geometry.py`:

```python
select_best_face(img_shape, faces, criterion="size") -> FaceData
extend_bbox(bbox, frame_shape, margin_factor) -> list[int]
```

Keep keypoint alignment either as a small function that accepts an affine
matrix, or keep it on the backend boundary if `estimate_norm()` remains
backend-specific.

Benefit:

Video extraction and image processing can share geometry helpers without
expanding `ForensicFace`.

### 6. Move video extraction into a task module

Current location:

- `ForensicFace.extract_faces()`

Problem:

Video I/O, progress bars, crop saving, metadata serialization, and frame
skipping are a separate workflow. Keeping this inside `ForensicFace` increases
class size and imports `pandas` lazily inside a method.

Proposal:

Create `src/forensicface/video.py`:

```python
extract_faces_from_video(processor, video_path, dest_folder=None, ...)
```

Keep `ForensicFace.extract_faces()` as a wrapper:

```python
return extract_faces_from_video(self, video_path, ...)
```

Benefit:

The class remains easy for users, while contributors can reason about video
export in isolation. It also makes room for future streaming or callback-based
APIs.

### 7. Move mosaic construction into a task module

Current location:

- `ForensicFace.build_mosaic()`

Problem:

Mosaic creation is a convenience visualization task, not core recognition
state. It also mixes processing, image borders, warning behavior, and optional
saving.

Proposal:

Create `src/forensicface/mosaic.py` with:

```python
build_aligned_face_mosaic(processor, images, mosaic_shape, border=0.03, ...)
```

Keep the public method as a wrapper.

Benefit:

Visualization code becomes optional task code and can evolve without making
the core facade more complex.

### 8. Move aggregation and comparison math to functions

Current locations:

- `ForensicFace.compare()`
- `ForensicFace.aggregate_embeddings()`
- `ForensicFace.aggregate_from_images()`
- `utils.cosine_similarity()`

Problem:

The math itself is not tied to model sessions. `compare()` and
`aggregate_from_images()` need the facade to process images, but cosine and
aggregation functions do not.

Proposal:

Move or expose:

```python
cosine_score(a, b) -> float
aggregate_embeddings(embeddings, weights=None, method="mean") -> np.ndarray
```

Keep `ForensicFace.compare()` and `ForensicFace.aggregate_from_images()` as
workflow wrappers.

Benefit:

End users get both ergonomic workflow methods and reusable math utilities.
Tests for numerical behavior become simpler.

### 9. Replace internal `assert` validation with explicit exceptions

Current examples:

- `_get_best_face()` checks criterion and non-empty faces.
- `_compute_embeddings_batch()` checks crop shape.
- `aggregate_embeddings()` checks shape and method.
- `process_aligned_face_image()` checks aligned face shape.
- utility functions validate color and keypoint shape with `assert`.

Problem:

`assert` can be disabled with Python optimization flags and raises generic
`AssertionError`. For library input validation, explicit `ValueError` or
`TypeError` is clearer for users and more stable for tests.

Proposal:

Replace user-input assertions with explicit exceptions. Keep low-level
developer invariants as asserts only when they truly indicate impossible
internal states.

Benefit:

Errors become clearer and the library behaves consistently under optimized
Python.

### 10. Clean up compatibility return types deliberately

Current behavior:

- `process_image(single_face=True)` warns that its return will change from dict
  to list.
- No-face single-image processing returns `[]`, while `align_only(single_face=True)`
  returns `None`.
- Batch single-face processing returns `None` for images without detected faces.

Problem:

The mixed return shapes make downstream code more defensive and complicate
internal reuse.

Proposal:

Plan a versioned migration:

1. Current version: document the exact behavior and keep warnings.
2. Next minor version: add an opt-in flag or new method with list-only return.
3. Next major version: make `process_image()` always return a list of face
   dicts, with an empty list for no face.

Benefit:

Contributors can write simpler internals while end users get a predictable
migration path.

## Suggested Module Layout

This is one possible target structure:

```text
src/forensicface/
  app.py                 # public facade and backward-compatible methods
  backends.py            # backend interface and ONNX backend
  model_store.py         # model path resolution and layout rules
  preprocessing.py       # image/keypoint input normalization
  recognition.py         # recognition/FIQA session runner
  results.py             # AlignedFace/result assembly helpers
  geometry.py            # bbox, face selection, keypoint geometry
  video.py               # video extraction task
  mosaic.py              # mosaic task
  utils.py               # public numerical/evaluation utilities
  ort_runtime_setup.py
  runtime_summary.py
  insightface/
  tools/
```

The target is not to split files for its own sake. Each new module should own a
clear concept and have tests that are simpler than the original class tests.

## Refactor Order

1. Add characterization tests for current return shapes, color ordering, and
   no-face behavior before moving code.
2. Extract pure functions first: preprocessing, geometry, aggregation, cosine
   score.
3. Extract result assembly and introduce an `AlignedFace` dataclass.
4. Extract model path resolution into `model_store.py`.
5. Extract recognition runner, preserving current private method wrappers.
6. Move video and mosaic workflows behind wrapper methods.
7. Replace `assert` validation with explicit exceptions.
8. Revisit public return types only after internal boundaries are stable.

## Compatibility Principles

- Do not remove public methods in the same change that moves implementation.
- Keep `ForensicFace` ergonomic for end users; complexity should move behind
  small helpers, not into user code.
- Preserve current keys in result dictionaries unless a versioned deprecation
  says otherwise.
- Keep fake-session tests as the default for CI; use real-model smoke tests for
  numeric/model-adapter changes.
- Avoid changing vendored InsightFace logic unless the change is targeted and
  covered by a real-model smoke test.

## Concrete "Move Out of Class" Checklist

High-confidence candidates:

- `_to_input_ada()` -> `preprocessing.to_ada_input()`
- `_to_keypoints_input()` -> `preprocessing.normalize_aligned_keypoints()`
- `_get_best_face()` -> `geometry.select_best_face()`
- `_get_extended_bbox()` -> `geometry.extend_bbox()`
- `aggregate_embeddings()` -> `utils.aggregate_embeddings()`
- `_looks_like_cuda_oom()` -> `recognition.looks_like_cuda_oom()`
- `_assemble_result()` and `_assemble_result_from_align_only()` ->
  `results.assemble_face_result()`
- `_load_model()` and `_resolve_quality_model()` path logic ->
  `model_store.py`

Keep as methods or wrappers for end-user convenience:

- `process_image()`
- `align_only()`
- `process_images_batch()`
- `process_aligned_face_image()`
- `compare()`
- `aggregate_from_images()`
- `build_mosaic()`
- `extract_faces()`

Consider later:

- Move `process_images_batch()` orchestration into a batch task function if it
  remains large after recognition/result extraction.
- Add typed result dataclasses internally, but continue returning dictionaries
  publicly unless the public API is intentionally versioned.
