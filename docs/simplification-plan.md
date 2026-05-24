# Codebase Simplification Status

This document records the simplification work performed after the architecture
review. It is now a status document rather than a proposal. The main goal was
to keep `ForensicFace` as the end-user facade while moving stateless logic and
large workflows into focused modules.

## Summary

The original concern was that `ForensicFace` had become a large class that mixed
public API, model lookup, preprocessing, recognition inference, result
assembly, batch processing, comparison, aggregation, mosaics, and video export.

The current design keeps public convenience methods on `ForensicFace`, but the
implementation is split by responsibility:

- `model_store.py`: model layout and path resolution.
- `preprocessing.py`: image/keypoint input normalization.
- `recognition.py`: recognition and FIQA inference.
- `results.py`: `FaceResult`, `AlignedFace`, and result builders.
- `geometry.py`: face selection and bbox expansion.
- `batch.py`: batched image processing workflow.
- `comparison.py`: comparison and image-set aggregation workflows.
- `mosaic.py`: aligned-face mosaic workflow.
- `video.py`: video face extraction workflow.
- `utils.py`: public math/evaluation utilities.

## Completed Work

### Public Result Objects

Added `FaceResult(dict)` so existing code can continue using dict access:

```python
ret["bbox"]
```

while newer code can use dot access:

```python
ret.bbox
```

Public face-processing APIs now return `FaceResult` instances where they
previously returned plain dictionaries.

### Model Store

Moved model layout rules into `model_store.py`:

- `resolve_recognition_model()`
- `resolve_quality_model()`
- `collect_backend_model_files()`

`ForensicFace._load_model()` remains intentionally as the small seam that
creates an ONNX Runtime session for a resolved recognition model path.

### Preprocessing

Moved stateless recognition preprocessing into `preprocessing.py`:

- `to_ada_input()`
- `normalize_aligned_keypoints()`

These functions are directly unit tested and used by `RecognitionRunner`.

### Recognition Runner

Moved recognition/FIQA inference into `recognition.py`.

`RecognitionRunner` owns:

- single aligned-crop inference;
- batch aligned-crop inference;
- keypoint-aware input construction;
- two-output embedding scaling;
- optional FIQA;
- CUDA OOM batch splitting.

### Result Assembly

Centralized result creation in `results.py`.

Important pieces:

- `AlignedFace`: internal normalized face representation.
- `build_align_result()`: detection/alignment results.
- `assemble_face_result()` and `build_face_result()`: full face-processing
  results.
- `build_face_result_from_align_result()`: bridges `detect_and_align()` output to
  batched recognition output.
- `build_embedding_result()`: recognition-only results for aligned-face APIs.

This keeps public keys consistent across single-image and batch workflows.

### Geometry

Moved stateless geometry helpers into `geometry.py`:

- `select_best_face()`
- `extend_bbox()`

Video extraction now uses `extend_bbox()` directly rather than going through a
private facade wrapper.

### Batch Workflow

Moved `process_images_batch()` orchestration into `batch.py`.

`ForensicFace.process_images_batch()` remains the public method. The task
module handles:

- per-image `detect_and_align()` calls;
- valid-face collection;
- chunking;
- optional aligned-keypoint stacking;
- CUDA OOM retry through `RecognitionRunner.try_compute_batch()`;
- scattering results back to input order.

### Comparison and Aggregation

Moved image comparison and image-set aggregation orchestration into
`comparison.py`.

Math remains in `utils.py`:

- `cosine_score()`
- `aggregate_embeddings()`

`ForensicFace.compare()` and `ForensicFace.aggregate_from_images()` remain
public convenience wrappers.

### Mosaic and Video Workflows

Moved task workflows into:

- `mosaic.py`
- `video.py`

`ForensicFace.build_mosaic()` and `ForensicFace.extract_faces()` remain public
facade methods.

### Explicit Validation

Replaced user-facing `assert` validation with explicit exceptions in core
library code. Remaining `assert` statements are in vendored/adapted
InsightFace code and tests.

### Redundant Private Wrappers

Removed redundant pass-through private wrappers from `ForensicFace`, including:

- `_to_input_ada()`
- `_to_keypoints_input()`
- `_get_best_face()`
- `_get_extended_bbox()`
- `_build_keypoint_model_inputs()`
- `_looks_like_cuda_oom()`
- `_try_compute_embeddings_batch()`
- `_compute_embeddings()`
- `_assemble_result()`
- the old private align-result assembly wrapper
- `_resolve_quality_model()`

The code now calls the focused helper modules directly.

### Test Coverage

Added direct unit tests for helper modules and revised facade tests so they are
mostly smoke tests. Branch-heavy behavior is now tested closer to the owning
module.

Current test organization:

- `tests/test_backends_and_api.py`: facade, backend, initialization, model
  loading smoke tests.
- `tests/test_batch_api.py`: batch image and aligned-face APIs.
- `tests/test_helper_modules.py`: extracted helper modules and task workflows.
- `tests/test_utils.py`: utility math and validation.
- `tests/test_migrate_shared.py`: model-layout migration tool.

## Current Module Layout

```text
src/forensicface/
  app.py                 # public facade and backward-compatible methods
  backends.py            # backend interface and ONNX backend
  model_store.py         # model path resolution and layout rules
  preprocessing.py       # image/keypoint input normalization
  recognition.py         # recognition/FIQA session runner
  batch.py               # batch image processing task
  comparison.py          # comparison and image-set aggregation tasks
  results.py             # FaceResult, AlignedFace, result builders
  geometry.py            # bbox and face-selection helpers
  video.py               # video extraction task
  mosaic.py              # mosaic task
  utils.py               # public numerical/evaluation utilities
  ort_runtime_setup.py
  runtime_summary.py
  insightface/
  tools/
```

## Intentional Remaining Private Helpers

Some private methods remain in `ForensicFace` because they are small and tied to
the facade's runtime state:

- `_load_model()`: creates an ONNX Runtime session for a resolved recognition
  model path.
- `_recognition_runner()`: creates a `RecognitionRunner` from current sessions
  and configuration.
- `_align_keypoints()`: asks the backend for the affine transform and applies
  it.
- `_draw_keypoints_on_aligned_face()`: local drawing helper for
  `draw_keypoints=True`.
- `_load_image()`: accepts either a path or an already loaded BGR image.
- `_get_loaded_modules()`: supports the initialization summary.

These are not currently worth extracting unless their responsibilities grow.

## Remaining Design Decision

The main unresolved API decision is the return shape of
`process_image(single_face=True)`.

Current behavior:

- `process_image(single_face=True)` returns a single `FaceResult` when a face is
  detected, and `[]` when no face is detected.
- `process_image(single_face=False)` returns a list.
- `detect_and_align(single_face=True)` returns one `FaceResult` or `None`.
- `process_images_batch(single_face=True)` returns a list parallel to inputs,
  with `None` in no-face slots.

There is an existing `FutureWarning` stating that `process_image(single_face=True)`
may later return a list with one result instead of a single result object. This
should be handled as an intentional versioned API migration, not as an internal
cleanup.

Suggested migration options:

1. Keep current behavior and update/remove the warning.
2. Add an opt-in list-only mode or new method first.
3. Change `process_image()` in a future major version so it always returns a
   list of face results.

## Compatibility Principles

- Keep `ForensicFace` ergonomic for end users.
- Preserve public result keys unless a versioned deprecation says otherwise.
- Keep `FaceResult` compatible with `dict`.
- Keep fake-session tests as the default for CI.
- Use real-model smoke tests for changes that touch vendored InsightFace
  adapters or numeric model behavior.
- Avoid changing vendored InsightFace code unless the change is targeted and
  covered.

## Completion Criteria

The simplification plan is complete from a structural standpoint. Future work
should focus on:

- the public return-shape decision;
- keeping contributor documentation aligned with module boundaries;
- small opportunistic cleanups found during normal feature work.
