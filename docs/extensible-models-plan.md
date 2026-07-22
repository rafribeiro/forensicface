# Extensible Models Implementation Record

Status: **Completed for forensicface v0.8.0**  
Created: **2026-07-21**  
Completed: **2026-07-22**  
Scope: detection, pose, gender, age, and face-image quality models  
Compatibility target: preserve the existing `ForensicFace` public API and default behavior

## Purpose

This document began as the living plan for making forensicface extensible to
alternative detection, pose, gender, age, quality, and embedding models. The
planned implementation was completed for v0.8.0. This document is now a
historical record of:

1. the coupling that existed before v0.8.0;
2. the architecture and public API that were implemented;
3. compatibility decisions and tradeoffs;
4. implementation and verification outcomes; and
5. features intentionally excluded from v0.8.0 that may be reconsidered later.

For current extension-author instructions, see
[`extending-models.md`](extending-models.md). For current user-facing examples,
see the project [`README.md`](../README.md).

## Final Outcome

forensicface v0.8.0 completed the planned extensibility work while preserving
the legacy initialization path and public result behavior. The release:

- added independent `detection`, `pose`, `gender`, `age`, `quality`, and
  `embedding` selectors;
- added generic `ModelSpec` configuration and direct component-object
  injection for every task;
- introduced detector, face-estimator, embedding, and quality contracts with
  an internal immutable component catalog;
- wrapped the existing SCRFD, pose, joint gender-age, CR-FIQA, and SEPAEL
  implementations as components without changing legacy defaults;
- added CenterFace as the first alternative detector, using ONNX Runtime and a
  reproducibly corrected offline ONNX artifact;
- separated quality execution from recognition while retaining batching and
  independent CUDA out-of-memory fallback for embeddings and quality;
- added task/alias model directories, legacy-layout lookup fallbacks, an
  offline model-layout migration tool, and the CenterFace conversion tool;
- preserved legacy constructor forms, backend injection, public field names,
  result shapes, and default behavior; and
- documented extension authoring and added compatibility, component,
  workflow, batching, parity, and real-model validation coverage.

## Starting Point Before v0.8.0

The pre-v0.8.0 implementation had three important couplings:

- Detection, pose, gender, and age were bundled into `ONNXOnlyBackend`. A
  detector could not be selected independently from the attribute
  models. See [`backends.py`](../src/forensicface/backends.py).
- Gender and age were produced by one InsightFace model and stored directly on
  `FaceData`.
- CR-FIQA was owned by `RecognitionRunner`, even though quality estimation is
  conceptually independent from recognition. See
  [`recognition.py`](../src/forensicface/recognition.py).

The `extended` flag also enabled or disabled pose, gender, age, and quality as
one indivisible bundle in [`app.py`](../src/forensicface/app.py). v0.8.0 kept
this legacy path for compatibility and added independently selectable task
components alongside it.

## Implemented Architecture

The v0.8.0 component pipeline is:

```text
BGR image
   |
   v
Detector -> detected face: bbox + score + five alignment keypoints
   |
   +-- Face aligner -> 112x112 aligned face
   |
   +-- Pose estimator
   +-- Gender estimator
   +-- Age estimator
   +-- Quality estimator
   +-- Recognition model(s)
   |
   v
Existing FaceResult assembly
```

Each model adapter owns its own:

- Preprocessing
- Selection of the appropriate source image or face crop
- Resizing, padding, and normalization to its own input shape
- Runtime/session
- Input and output interpretation
- Single-face inference
- Batch inference where supported (embeddings and quality)
- Model metadata such as name and provider

The orchestration layer works with normalized forensicface data rather than
model-specific tensors.

### Core Contracts

The implementation introduced one dedicated detector contract and a common
face-estimator contract.

A detector must return:

- `bbox` as `(x1, y1, x2, y2)`
- `det_score` as a float
- Five keypoints with shape `(5, 2)` in the existing order

The five keypoints are essential: alignment and `sepaelv6` depend on them.
Box-only detectors were excluded because forensicface does not yet have a
configurable alignment-landmark stage.

Estimators declare capabilities:

```python
class FaceEstimator:
    capabilities = {"gender", "age"}

    def estimate(self, context) -> FaceAttributes:
        ...
```

`context` exposes the original BGR image, detection, aligned BGR face,
aligned keypoints, and the alignment transform. An adapter uses only what it
needs.

No common estimator image size was introduced. The context contains source
data with documented coordinate and color conventions, not a tensor already
resized for every model. Each adapter is responsible for producing its own
model input. For example:

- SCRFD consumes the original BGR image and applies its configured detection
  size and aspect-ratio policy.
- The current pose and gender-age models consume a crop derived from the
  original image and bounding box, resized to the input shape reported by
  their own sessions.
- CR-FIQA and the current recognition models consume an aligned face and apply
  their own tensor preprocessing.
- A future quality model may choose the original image and bounding box rather
  than the canonical aligned crop.

Face-estimator batching was considered but is not part of this implementation.
Pose, gender, and age run per face. Embedding and quality adapters retain their
specialized batch contracts and own the policy that makes their inputs
batchable.

The capability design matters because a single model may produce both gender
and age. If the same adapter is selected for both tasks, the pipeline groups
it and runs inference once. Separate gender and age models remain equally
possible.

Pose is normalized internally as `PoseAngles(pitch, yaw, roll)`. Public result
fields and the legacy `[pitch, yaw, roll]` array returned by alignment APIs were
preserved.

## Public Configuration

The meaning of the existing `models` parameter was not changed: it remains the
legacy way to select one or more recognition/embedding aliases. Existing calls
retain their previous meaning:

```python
ff = ForensicFace(
    models=["sepaelv2", "sepaelv4"],
)
```

This selects `sepaelv2` and `sepaelv4` for embeddings while selecting the
current defaults for detection, pose, gender, age, and quality. `ForensicFace()`
continues to select `sepaelv2` plus all current default components.

v0.8.0 added keyword-only selectors for each task:

```python
ff = ForensicFace(
    detection="scrfd",
    pose="insightface-3d68",
    embedding="sepaelv6",
)
```

Unspecified task selectors use the current defaults, so this example also
loads the current gender, age, and quality models. `embedding` accepts either one
alias or a sequence of aliases:

```python
ff = ForensicFace(
    detection="scrfd",
    embedding=["sepaelv2", "sepaelv4"],
)
```

The implemented task keywords are:

- `detection`
- `pose`
- `gender`
- `age`
- `quality`
- `embedding`

The new configuration keyword and existing public result key are both
`gender`. This deliberately preserves the existing public terminology in
v0.8.0. The limitations and interpretation of the selected
model are documented through component metadata.

If any new task selector is explicitly provided, `models` and the deprecated
`model` argument cannot also be provided; the implementation raises a clear
`ValueError`. An omitted-value sentinel distinguishes an omitted task from an
explicitly disabled task:

- Omitted selector: inherit the current default or the `extended` preset.
- `None`: explicitly disable the task where disabling is supported.
- Alias/specification/object: explicitly select and enable the task.

The existing `models` default is normalized internally rather than using a
mutable list default. This makes it possible to distinguish
`ForensicFace()` from a call that explicitly supplied `models`.

### Three Configuration Levels

Each task selector supports three levels of control:

- A registered model alias
- A configured model specification containing path/options
- An already constructed user implementation

`ModelSpec` is generic infrastructure, not a detector-specific type.
Detection, pose, gender, age, quality, and embedding selectors may all receive
a `ModelSpec`; each component factory validates the options relevant to its
own alias.

Common usage stays concise:

```python
ff = ForensicFace(detection="scrfd", embedding="sepaelv6")
```

Model-specific initialization parameters use a specification object:

```python
ff = ForensicFace(
    detection=ModelSpec(
        "scrfd",
        det_size=640,
        det_thresh=0.4,
    ),
    embedding=["sepaelv2", "sepaelv4"],
)
```

Advanced users can construct and inject an implementation:

```python
ff = ForensicFace(
    detection=my_detector,
    pose=my_pose_estimator,
    gender=None,
    age=my_age_estimator,
    quality=my_quality_estimator,
    embedding=[my_embedding_estimator, "sepaelv4"],
)
```

Users are not required to instantiate every task. Aliases are the ergonomic
default, `ModelSpec` handles model-specific parameters, and direct object
injection is the lowest-level extension mechanism.

The existing top-level `det_size` and `det_thresh` arguments remain functional
in legacy/default mode. For an explicitly supplied SCRFD `ModelSpec`, values
in the specification take precedence. v0.8.0 does not emit a warning when both
forms are explicitly supplied; adding such diagnostics can be reconsidered if
real-world ambiguity warrants it.

When an explicitly selected non-SCRFD detector is combined with explicitly
set legacy SCRFD-only parameters, raise a configuration error rather than
silently ignoring them. Omitted-value sentinels are required for `det_size`
and `det_thresh` so an explicit value can be distinguished from the historical
default appearing in the function signature.

`det_size` did not become a universal detector setting. Different detectors
have different input-size policies. For example, the CenterFace
implementation rounds the original height and width independently to
multiples of 32, whereas the current SCRFD path uses the configured square
detection size. Such settings belong in the selected detector's adapter or
`ModelSpec`.

### Alias Semantics

`sepaelv2`, `sepaelv3`, and the other existing names remain embedding aliases
and legacy configuration presets. They did not become aliases for
unrelated task implementations in the new task-specific namespaces.

In legacy mode, the translation is:

```text
models=["sepaelv2", "sepaelv4"]
    -> embedding=["sepaelv2", "sepaelv4"]
    -> detection=current default
    -> pose=current default when extended=True
    -> gender=current default when extended=True
    -> age=current default when extended=True
    -> quality=current default when extended=True
```

The internal component catalog is namespaced by task. This prevents alias
collisions from making it unclear whether `scrfd` is a detector or an embedding
model. The legacy directory fallback can use the first embedding alias to find
shared historical files, but this lookup detail does not define the
new component identities.

v0.8.0 uses an internal immutable catalog only.
There is no public registration API, process-global mutable registry, or
per-instance registry overlay. Advanced users extend the pipeline by passing a
constructed component object directly.

### Model Acquisition and Offline Artifacts

Model acquisition remains manual and offline. Initialization does not download
models, and model binaries are not bundled in the Python wheel. Users place
files in the documented task/alias directories or provide an explicit path in
a `ModelSpec`.

The corrected CenterFace binary is generated with
`python -m forensicface.tools.convert_centerface_onnx` and distributed offline
to users. The tool records the source checksum in model metadata and prints the
generated artifact checksum. `python -m forensicface.tools.migrate_shared`
migrates both the original per-model layout and the 0.7 flat shared layout to
the task/alias structure, using hashes to delete only identical duplicates and
blocking on conflicts.

### Disabled Tasks and Result Fields

An omitted selector inherits from `extended` and the library defaults. An
explicit `None` disables that task. For calls using the new explicit selector
API, result keys belonging only to a disabled task are omitted rather than
included with `None` values. Existing legacy calls retain their current result
key sets exactly.

Detection and embedding need explicit validation rules because existing full
processing workflows require them, while detection-only workflows may not
need embeddings. `detection=None` is invalid. `embedding=None` creates a
detection/alignment/attribute-only processor: `detect_and_align()` and
`process_image()` work, with embedding fields omitted, while comparison,
aggregation, and aligned-face embedding APIs raise clear configuration errors.

### Embedding Components

The `embedding` selector uses the same alias, `ModelSpec`, and constructed
object syntax as the other task selectors and continues to accept more than
one model. Internally, embeddings retain a specialized interface because they
return vectors, may require aligned keypoints, participate in concatenation,
comparison, and aggregation, and have batch-specific behavior.

## Compatibility Outcome

The following were kept compatible:

- Existing constructor calls and positional parameters
- `models`, `model`, `extended`, `backend`, and `backend_name`
- All public method names and return shapes
- BGR public inputs and RGB `aligned_face` outputs
- Existing `FaceResult` keys and value types
- Legacy and current model directory layouts
- Existing custom `FaceBackend` implementations

An injected legacy backend continues working as before when used alone. The
legacy `backend` argument is mutually exclusive with all new task selectors;
mixing both configuration systems raises a clear `ValueError`.

`fiqa_score` was retained as the public quality key, even for an alternative
quality model. It is not an ideal generic name, but renaming it would have
conflicted with strict compatibility. A generic `quality_score` remains a
possible future, versioned API decision.

The initialization summary in
[`runtime_summary.py`](../src/forensicface/runtime_summary.py) enumerates
explicitly selected components through their metadata while retaining the
legacy session-inspection path for legacy initialization.

## Implementation Outcome by Phase

### 1. Freeze compatibility behavior

Characterization tests were added for constructor defaults, result keys/types,
default model loading, single/batch parity, and existing backend injection.

### 2. Introduce contracts without changing execution

Detector/estimator protocols, normalized pose/attribute results, generic
`ModelSpec`, and an internal component catalog were added. The legacy backend
remained operational.

### 3. Wrap current models

Adapters were created for SCRFD, 3D68 pose, joint gender-age, CR-FIQA, and the
existing embedding models. Their outputs were compared with the legacy
implementation using the same images and sessions.

### 4. Add the pipeline orchestrator

The explicit component path moved task sequencing out of `ONNXOnlyBackend`.
Quality execution moved out of `RecognitionRunner` into `QualityRunner`, and
`AlignedFaceRunner` now combines independent embedding and quality results.
Embedding/quality batching and independent CUDA OOM fallback were preserved.

### 5. Connect the opt-in public configuration

Keyword-only task selectors were added. Calls without explicit selectors use
the exact legacy path, and `backend` remains a compatibility path.

### 6. Validate extensibility

Alternate fake adapters were tested for each task, and CenterFace was
integrated as the genuine alternative detector. These tests exercised crops,
native input dimensions, coordinates, output scales, batching boundaries, and
result schemas.

The first real alternative model was the ONNX CenterFace detector. v0.8.0 is
ONNX-only; other runtimes were deferred.

The supplied CenterFace reference repository indicates that the landmark
variant produces:

- bounding boxes and confidence scores;
- five facial landmarks;
- heatmap, scale, offset, and landmark ONNX outputs;
- image preprocessing that rounds height and width to multiples of 32 and
  rescales output coordinates to the original image.

CenterFace validated the detector contract and detector-specific input sizing.
The supplied ONNX graph was inspected for input/output shapes, five-landmark
semantics, and numerical parity with the reference decoder before its adapter
was implemented. The reference repository uses the MIT license.

#### CenterFace inference runtime

The reference implementation uses OpenCV DNN, which is already available
through forensicface's OpenCV dependency. OpenCV DNN accepts the supplied
CenterFace export at arbitrary image dimensions despite its incorrect fixed
shape metadata. On `nbs/obama.png`, its four outputs matched ONNX Runtime using
corrected in-memory shape metadata with maximum absolute differences between
approximately `7e-7` and `1.6e-5`.

OpenCV DNN is therefore useful as a numerical reference and could provide a
CPU implementation. It is not the primary runtime for the first
CenterFace adapter because the standard `opencv-python-headless` wheel used in
the inspected environment has no CUDA support, while forensicface already
configures ONNX Runtime CPU/CUDA providers for its other models. OpenCV's
`Net.setInput()` API also introduces mutable per-network state that needs extra
care under concurrent use.

**Implemented outcome:** the CenterFace adapter uses ONNX Runtime and the
corrected ONNX artifact. OpenCV DNN was used only as a numerical parity
reference. An OpenCV-specific fallback was not added.

### 7. Document extension authoring

Detector requirements, color order, coordinate systems, thread safety,
batching boundaries, provider/device behavior, and expected output types were
documented in [`extending-models.md`](extending-models.md).

## Verification Completed

The completed test suite covers:

- Replacing each task independently
- Using different models for gender and age
- A joint gender-age model being invoked only once
- Enabling only a subset of optional tasks
- Invalid detector output and missing five-point keypoints
- Single-image versus batch consistency
- Quality estimator batching and OOM splitting
- Existing `extended=True` and `extended=False` behavior
- Existing custom `FaceBackend` compatibility
- Identical default result keys and conversions
- Video, mosaic, comparison, and aggregation workflows remaining unaffected

At release preparation, 150 tests passed. Additional real-model CPU checks
confirmed exact legacy-versus-component SCRFD parity for ordered result keys,
detection, pose, gender, age, embeddings, and quality. CenterFace completed
single-image inference and aligned-face batch embedding/quality processing
with the corrected offline artifact.

## Implemented Recommendations

The initial recommendations were implemented as follows:

1. Use direct task keywords for the primary public API, with alias,
   `ModelSpec`, and object-injection forms.
2. Use capability-based estimators to support joint models efficiently.
3. Keep direct object injection as the fundamental extension mechanism.
4. Box-only detectors and automatic third-party discovery were excluded from
   v0.8.0 and recorded for possible future reconsideration.

## Decisions

The table below preserves the accepted design decisions made during planning
and implementation.

| Date | Decision | Status | Rationale |
| --- | --- | --- | --- |
| 2026-07-21 | Begin with task contracts and model adapters while preserving the `ForensicFace` facade. | Accepted | Separates model-specific behavior without forcing existing users to migrate. |
| 2026-07-21 | Require detectors to return five alignment keypoints in the first extensible detector contract. | Accepted | The current alignment and keypoint-aware recognition paths require them. |
| 2026-07-21 | Preserve `fiqa_score` during the first implementation. | Accepted | Avoids breaking consumers of the existing result schema. |
| 2026-07-21 | Let each adapter own its input shape, cropping, resizing, padding, and normalization. | Accepted | Current and future estimators do not share an input tensor shape or necessarily the same source crop. |
| 2026-07-21 | Preserve `models` as the legacy multi-embedding selector and make it mutually exclusive with explicit task selectors. | Accepted | Keeps existing calls valid while preventing two competing configuration styles in one call. |
| 2026-07-21 | Support aliases, generic `ModelSpec` values, and constructed objects in every task selector. | Accepted | Provides concise defaults, configurable built-ins, and unrestricted custom implementations without forcing object construction on every user. |
| 2026-07-21 | Expose direct task keywords before considering a public `FaceComponents` object. | Accepted | Keeps common configuration concise and discoverable. |
| 2026-07-21 | Resolve task configuration with precedence `explicit selector > extended preset > library default`. | Accepted | Preserves incremental overrides while retaining existing `extended` behavior. |
| 2026-07-21 | Require operational component metadata and keep full provenance metadata strongly recommended but optional initially. | Accepted | Supports diagnostics and reproducibility without making the first adapter contract unnecessarily burdensome. |
| 2026-07-21 | Use the ONNX CenterFace detector as the first real alternative model. | Accepted | It exercises detector-specific preprocessing and returns boxes, scores, and five landmarks required by the initial detector contract. |
| 2026-07-21 | Limit the initial model-extension scope to ONNX models. | Accepted | Keeps runtime and packaging concerns stable while the component interfaces are established. |
| 2026-07-22 | Keep `gender` as both the task-selector name and existing result-field name. | Accepted | Preserves the current public terminology and avoids adding a parallel `sex` keyword. |
| 2026-07-22 | Omit fields belonging to explicitly disabled tasks while preserving legacy result key sets. | Accepted | Distinguishes a disabled task from an estimator that ran but returned no estimate without changing existing calls. |
| 2026-07-22 | Give embeddings the common selector syntax but retain a specialized internal embedding interface. | Accepted | Supports multiple configurable embedding models while preserving vector, keypoint, comparison, aggregation, and batching semantics. |
| 2026-07-22 | Use capability-based joint estimators. | Accepted | Ensures a model such as the current joint gender-age model runs only once while serving both configured tasks. |
| 2026-07-22 | Namespace model directories by task and alias, with legacy-layout fallbacks. | Accepted | Allows multiple models per task without ambiguous flat-directory discovery. |
| 2026-07-22 | Use an internal immutable component catalog only. | Accepted | Built-in aliases remain deterministic; custom implementations use direct object injection without public registry state. |
| 2026-07-22 | Use ONNX Runtime for CenterFace and OpenCV DNN only as a parity reference. | Accepted | Preserves provider/GPU behavior while retaining validation against the reference implementation. |
| 2026-07-22 | Reject `detection=None` and allow `embedding=None` for detection/alignment/attribute-only processing. | Accepted | Detection is fundamental to image processing, while embedding-free analysis is a useful coherent mode. |
| 2026-07-22 | Keep model acquisition manual and offline. | Accepted | Avoids network activity during initialization and keeps model distribution explicit. |
| 2026-07-22 | Generate and distribute a corrected CenterFace artifact offline. | Accepted | The source export needs dynamic metadata and initializer cleanup for reliable ONNX Runtime use. |
| 2026-07-22 | Make legacy `backend` mutually exclusive with new task selectors. | Accepted | Avoids ambiguous ownership and override behavior while preserving existing backend-only calls. |
| 2026-07-22 | Reject explicitly supplied SCRFD-only legacy parameters with a non-SCRFD detector. | Accepted | Prevents silently ignored detector configuration. |
| 2026-07-22 | Keep batching only for embeddings and quality; do not add a batch contract for face estimators. | Accepted | Keeps the initial pose/gender/age extension contract small while preserving the existing performance-critical batch paths. |
| 2026-07-22 | Do not implement the previously deferred features in this implementation. | Accepted | Keeps this change focused on ONNX task replacement, compatibility, and CenterFace. |

## Possible Future Reconsideration

The following items were deliberately excluded from v0.8.0. They are not
commitments or an active roadmap, but future work may reconsider them if a
concrete use case justifies the added API, runtime, packaging, or maintenance
cost.

- **Face-estimator batching:** pose, gender, and age execute per face. Only
  embeddings and quality have batch contracts in v0.8.0. A future version may
  add face-estimator batching if profiling demonstrates a material benefit and
  there is a clear way to handle different native input sizes, joint
  capabilities, dynamic shapes, and per-estimator fallback without making the
  common contract unnecessarily complex.
- **Box-only detectors:** these would require a configurable landmark or
  alignment stage capable of producing the five points required by current
  alignment and `sepaelv6`.
- **Non-ONNX runtimes:** PyTorch or other runtimes may be considered after
  weighing optional dependencies, device selection, packaging, and provider
  diagnostics.
- **Public registration and discovery:** the current catalog is internal and
  immutable. A public registry, per-instance catalog overlay, or entry-point
  discovery could be reconsidered if direct object injection proves
  insufficient.
- **Automatic model acquisition or wheel bundling:** v0.8.0 remains manual and
  offline. Any future acquisition mechanism would need explicit integrity,
  provenance, licensing, caching, and offline-use policies.
- **A public `FaceComponents` configuration object:** direct task keywords were
  preferred for v0.8.0. A reusable/serializable configuration object may be
  useful if the constructor grows or complete configurations need to be shared.
- **A model-neutral `quality_score` result field:** v0.8.0 retains
  `fiqa_score`. A new name would require a separately versioned schema and
  migration decision.
- **OpenCV DNN as a CenterFace runtime fallback:** OpenCV remains a parity
  reference. A fallback may be reconsidered only for a concrete deployment
  need.
- **Conflicting SCRFD-setting diagnostics:** `ModelSpec` values take precedence
  over top-level SCRFD settings. A future version may add warnings when both
  forms are explicitly supplied with different values.
- **Required full provenance metadata:** checksum, license, citation, training
  source, and detailed model-card fields remain recommended rather than
  mandatory. Stronger requirements may be appropriate for future forensic
  reproducibility workflows.

## Historical Tradeoffs and Outcomes

### Public `FaceComponents` versus direct task keywords

**Public `FaceComponents` object**

Pros:

- Keeps the `ForensicFace` signature from growing as tasks are added.
- Can be built, copied, validated, serialized, and reused independently.
- Clearly represents a complete pipeline configuration.

Cons:

- Adds an import and nesting for common calls.
- Duplicates some constructor concepts and makes simple overrides verbose.
- Makes compatibility behavior around `extended` less obvious.

**Direct task keywords**

Pros:

- Concise and discoverable through the `ForensicFace` signature and IDE help.
- Closely matches how users describe the desired pipeline.
- Makes incremental overrides straightforward.

Cons:

- Expands the constructor signature.
- Requires a sentinel to distinguish omitted selectors from `None`.
- Is less convenient for storing or reusing a complete configuration.

**Implemented outcome:** direct task keywords were exposed. A public
`FaceComponents` object was deferred for possible future reconsideration.

### Interaction between explicit selectors and `extended`

**Explicit configuration completely supersedes `extended`**

Pros:

- One authoritative source of configuration.
- No hidden inheritance once explicit configuration is selected.
- Easy to serialize and reproduce.

Cons:

- Users must repeat every desired default just to change one task.
- `detection="scrfd"` could unexpectedly disable pose, gender, age, and quality.
- Does not match the proposed incremental-override examples.

**Unspecified tasks inherit from `extended`**

Pros:

- Small overrides remain concise.
- Existing default and `extended=False` behavior remain useful.
- Matches the expectation that omitted tasks keep current defaults.

Cons:

- The final pipeline depends on both old and new arguments.
- Explicit `None` and omitted values must be distinguished.
- Precedence must be documented carefully.

**Implemented outcome:** each task resolves through explicit selector, then the
`extended` preset, then the library default. Detection and embedding have
separate validation because they are not optional extended attributes. An
explicit selector enables a task even when `extended=False`; explicit `None`
disables supported tasks.

### Registry ownership

A registry maps a task-specific alias to a factory; it does not normally hold
an already loaded model instance. For example:

```text
(detection, scrfd)     -> factory that constructs the SCRFD adapter
(detection, centerface)-> factory that constructs the CenterFace adapter
(quality, cr-fiqa)     -> factory that constructs the CR-FIQA adapter
```

The ownership question is about where those alias-to-factory mappings live
and who is allowed to mutate them. It is separate from model lifetime: each
`ForensicFace` instance still owns the detector and estimator instances
constructed from the selected factories.

**Process-global mutable registry**

Pros:

- Convenient decorators and third-party auto-registration.
- Aliases are available everywhere without passing a registry.

Cons:

- Shared mutable state can leak between tests and applications.
- Alias collisions and import order can change behavior.
- Makes an initialized pipeline harder to reproduce.

**Per-instance registry**

Pros:

- Isolated, deterministic, and straightforward to test.
- Different `ForensicFace` instances can use different aliases safely.

Cons:

- More setup and object passing.
- Third-party packages cannot register once for normal application use.

**Built-in catalog plus per-instance overlay**

Pros:

- Built-in aliases remain convenient and stable.
- Applications can add or override aliases locally.
- Avoids most shared mutable-state problems.

Cons:

- More implementation and documentation complexity.
- Override and collision rules still need to be explicit.

**Implemented outcome:** the built-in catalog is internal and immutable.
Direct object injection is exposed for custom implementations. v0.8.0 did not
add a public registration API, process-global mutable registry, per-instance
overlay, or automatic entry-point discovery.

### Minimum component metadata

**Minimal metadata (`name` and `capabilities`)** is easy for extension authors
but insufficient for meaningful runtime summaries and reproducibility.

**Operational metadata** adds runtime, provider/device, model path or model ID,
input source/space, and batch support. It enables useful diagnostics without
requiring a full model card from every implementation.

**Full provenance metadata** could also require version, checksum, license,
citation, training source, and preprocessing details. This is valuable in a
forensic context, but some values are unavailable for programmatic or remote
estimators and the burden could discourage adapters.

**Implemented outcome:** component ID/name, capabilities,
implementation type, runtime/provider/device, model source/path when
available, input-space declaration, and batch-support declaration are
required. Version, checksum, license, citation, and detailed model-card
information remain strongly recommended but optional.

### Generic `quality_score` result name

**Add it immediately as an alias**

Pros:

- New integrations get model-neutral terminology immediately.
- Provides a path away from the CR-FIQA-specific name.

Cons:

- Changes exact result key sets.
- Two names for one value can confuse consumers.

**Add it only for explicitly configured quality models**

Pros:

- The legacy result remains untouched.

Cons:

- Result schemas differ based on how the same model was configured.
- Generic code becomes harder to write.

**Introduce it through a versioned migration**

Pros:

- Allows documentation, warnings, and a deliberate compatibility window.

Cons:

- `fiqa_score` remains misleading for non-CR-FIQA models in the meantime.

**Never add it**

Pros:

- Maximum result-schema compatibility.

Cons:

- Permanently couples the public name to one quality method.

**Implemented outcome:** `fiqa_score` was retained and the active quality model
is exposed through component metadata. `quality_score` was deferred as a
separate possible versioned API decision.

### First real alternative model

**Mocks only** are fast and deterministic but cannot reveal preprocessing,
runtime, or numerical integration problems.

**An alternative ONNX model** minimizes dependency and runtime changes and is
the safest first end-to-end validation, though it does not prove that the
contracts are runtime-neutral.

**A model using a different runtime** such as PyTorch is a stronger test of the
architecture, but introduces packaging, device, and optional-dependency work
at the same time as the interfaces are being stabilized.

**Implemented outcome:** each task was validated with fake adapters in CI and
the ONNX CenterFace detector became the first real integration. v0.8.0 remains
ONNX-only; different runtimes are future reconsideration candidates.

## Progress Log

This chronological log preserves planning, implementation, compatibility, and
verification milestones.

### 2026-07-21 - Planning document created

- Inspected the current facade, backend, recognition/FIQA runner, result
  assembly, model-store, batch workflow, tests, and contributor architecture
  documentation.
- Recorded the initial extension architecture and compatibility policy.
- No implementation code was changed.

### 2026-07-21 - Input-shape and public configuration discussion

- Clarified that every component adapter owns model-specific cropping,
  resizing, padding, normalization, and batching.
- Proposed keeping `models` as the legacy multi-embedding selector.
- Proposed mutually exclusive task-specific selectors: `detection`, `pose`,
  `sex`, `age`, `quality`, and `embedding`.
- Added alias, `ModelSpec`, and constructed-object configuration levels.
- Recorded pros, cons, and current recommendations for the open design
  questions.
- No implementation code was changed.

### 2026-07-21 - Initial decisions and CenterFace selection

- Accepted direct task keywords as the first public configuration API.
- Accepted selector precedence: explicit selector, then `extended`, then the
  library default.
- Accepted the recommended minimum operational component metadata.
- Accepted preserving `fiqa_score` during the first implementation.
- Selected the ONNX CenterFace detector as the first real alternative model.
- Limited the initial model-extension scope to ONNX models.
- Inspected the supplied CenterFace repository snapshot and recorded its
  bounding-box, confidence, five-landmark, preprocessing, decoding, and MIT
  license characteristics.
- Left registry ownership open for further discussion.
- No implementation code was changed.

### 2026-07-22 - CenterFace ONNX graph inspection

- Inspected the supplied `centerface.onnx` file without modifying it.
- Recorded SHA-256:
  `77e394b51108381b4c4f7b4baf1c64ca9f4aba73e5e803b2636419578913b5fe`.
- ONNX validation succeeded. The graph uses IR version 4, ONNX opset 9, and
  identifies its producer as PyTorch 1.2.
- The supplied export declares a fixed input shape of `[10, 3, 32, 32]` and
  fixed outputs of `[10, 1, 8, 8]`, `[10, 2, 8, 8]`, `[10, 2, 8, 8]`, and
  `[10, 10, 8, 8]` named `537`, `538`, `539`, and `540`.
- ONNX Runtime rejects other input shapes when loading the file as supplied.
- The graph is nevertheless fully convolutional. After changing only shape
  metadata in memory, ONNX Runtime successfully executed inputs of
  `[1, 3, 64, 64]`, `[2, 3, 64, 96]`, and `[1, 3, 320, 320]`, producing
  stride-4 output maps.
- The old export also exposes its initializers as graph inputs. ONNX Runtime
  accepts this but warns that it prevents some graph optimizations.
- Ran the dynamically described graph on `nbs/obama.png` using the reference
  preprocessing and decoder. It detected the face and produced five landmarks
  in the expected alignment-compatible pattern: two eyes, nose, and two mouth
  corners, with the paired points in left-to-right image order.
- Established the requirement for a reproducible corrected CenterFace
  artifact: dynamic batch/height/width metadata, initializers removed from
  graph inputs, semantic outputs, ONNX validation, and numerical comparison
  with the source artifact. This was subsequently implemented by the offline
  converter.
- No attached model or implementation code was changed.

### 2026-07-22 - API decisions and CenterFace runtime comparison

- Accepted generic `ModelSpec` and direct object injection for every task.
- Kept `gender` as the new selector name and existing result-field name; no
  `sex` selector was added in v0.8.0.
- Accepted omission of result fields for explicitly disabled tasks while
  preserving legacy result schemas.
- Accepted common embedding selector syntax with a specialized internal
  embedding interface.
- Accepted capability-based joint estimators, namespaced task/alias model
  directories with legacy fallbacks, and an internal immutable catalog only.
- Compared OpenCV DNN and ONNX Runtime outputs for CenterFace on the same input;
  all outputs agreed within `1e-4` tolerance.
- Recommended ONNX Runtime for CenterFace and OpenCV DNN as the parity
  reference because the installed OpenCV wheel does not provide CUDA.
- No implementation code or attached model was changed.

### 2026-07-22 - Offline converter and task/alias migration support

- Accepted ONNX Runtime as the CenterFace runtime.
- Added `forensicface.tools.convert_centerface_onnx`, which makes the old
  CenterFace export dynamic, removes initializers from graph inputs, assigns
  semantic output names, records source metadata, validates the graph, checks
  source numerical parity, checks dynamic execution, writes atomically, and
  refuses overwrites by default.
- Added synthetic ONNX conversion tests; no CenterFace binary is committed to
  the repository.
- Generated and verified a temporary corrected artifact from the supplied
  source. After removal of 59 unused initializers, its SHA-256 was
  `b53a0a79efe29186ad288a5e3ca5445c790f61927c8a714db3c9478c245c2261`.
- Adapted `forensicface.tools.migrate_shared` to migrate either historical
  model layout to task/alias directories while retaining dry-run, hashing,
  conflict blocking, and duplicate-removal safeguards.
- Extended model lookup to prefer task/alias paths and fall back to the 0.7
  flat shared and original per-model layouts, so migrated installations remain
  usable.
- Recorded manual/offline model acquisition, embedding-free processor
  behavior, legacy backend exclusivity, and detector-parameter conflict rules.

### 2026-07-22 - Component pipeline implementation

- Added the generic public `ModelSpec`, operational component metadata, task
  protocols, face context, and a compatibility backend that composes detector
  and estimator objects.
- Added an internal immutable, task-namespaced catalog with aliases for the
  existing SCRFD, InsightFace pose, InsightFace gender-age, CR-FIQA, and
  SEPAEL embedding models, plus CenterFace.
- Added ONNX adapters in which every model owns its preprocessing and native
  input size. The joint gender-age adapter is reused and invoked once when it
  serves both selected tasks.
- Added keyword-only `detection`, `pose`, `gender`, `age`, `quality`, and
  `embedding` selectors. Legacy-only constructor calls continue through the
  original initialization path.
- Implemented selector precedence, explicit task disabling, multi-embedding
  object/alias selection, legacy argument conflicts, and embedding-free image
  analysis behavior.
- Added the ONNX Runtime CenterFace detector and verified the corrected
  `centerface20260722.onnx` artifact with dynamic input and a repository sample
  image.
- Added focused contract, selector, result-schema, joint-estimator, model-path,
  and compatibility tests. The full suite passes.
- Compared the legacy pipeline and the explicit SCRFD component pipeline on
  the same sample image. Bounding boxes, keypoints, pose, gender, age,
  embedding, FIQA score, and ordered result keys matched exactly. This check
  also established that detector coordinates must retain floating-point
  precision internally and be converted to integers only at result assembly.

### 2026-07-22 - Remaining architecture and validation work

- Kept batching only for embeddings and quality by explicit decision; no
  `estimate_batch` contract was added to pose, gender, or age estimators.
- Removed quality inference from `RecognitionRunner`. `QualityRunner` now owns
  legacy/component quality execution and independent CUDA-OOM splitting, while
  `AlignedFaceRunner` combines embedding and quality results.
- Added the named internal `PoseAngles(pitch, yaw, roll)` representation while
  preserving legacy public pose arrays and result fields.
- Updated runtime summaries to enumerate explicit components through
  `ComponentMetadata` rather than compatibility backend attributes.
- Added component-mode tests for custom pose, comparison, quality-weighted
  aggregation, mosaics, video extraction, metadata-based provider reporting,
  and independent quality OOM retry.
- Added `docs/extending-models.md` with extension contracts, coordinate/color
  conventions, native input ownership, provider/device behavior, concurrency,
  batching boundaries, configuration, and testing guidance.
- Confirmed that all previously deferred features remain outside this
  implementation.
- Added automated current-adapter versus legacy-backend parity coverage and
  completed the suite with 150 passing tests.
- Repeated real-model CPU validation after the orchestration changes. The
  legacy and explicit SCRFD pipelines matched exactly for ordered keys,
  detection, pose, attributes, embeddings, and quality. CenterFace completed
  full single-image and aligned-face batch embedding/quality processing with
  the corrected offline artifact.
- Corrected initialization reporting for detector-specific input policies.
  SCRFD now exposes its effective prepared input size, so a
  `ModelSpec("scrfd", det_size=128)` updates both `ff.det_size` and the summary
  to `(128, 128)`. Dynamic CenterFace reports `dynamic (multiple of 32)` rather
  than the unrelated legacy SCRFD default.

### 2026-07-22 - v0.8.0 release preparation and historical closeout

- Updated the package version and release documentation for v0.8.0.
- Updated README usage examples to use the task-selector API and documented
  compatibility with legacy constructor forms.
- Validated the final release state with 150 passing tests and successful
  source-distribution and wheel builds.
- Reclassified this document from an active plan to a historical
  implementation record.
- Recorded face-estimator batching and all other deliberately excluded
  features as possible future reconsiderations rather than unfinished v0.8.0
  work.
