"""Immutable catalog and resolvers for built-in model components."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

from .components import (
    EmbeddingEstimator,
    FaceDetector,
    FaceEstimator,
    ModelSpec,
    QualityEstimator,
)
from .model_store import (
    resolve_component_model,
    resolve_quality_model,
    resolve_recognition_model,
)
from .onnx_components import (
    CRFIQAQualityEstimator,
    CenterFaceDetector,
    InsightFaceGenderAgeEstimator,
    InsightFacePoseEstimator,
    ONNXEmbeddingEstimator,
    SCRFDDetector,
)


DEFAULT_ALIASES = MappingProxyType(
    {
        "detection": "scrfd",
        "pose": "insightface-3d68",
        "gender": "insightface-genderage",
        "age": "insightface-genderage",
        "quality": "cr-fiqa",
        "embedding": "sepaelv2",
    }
)

# Alias membership is deliberately immutable and internal. Extension outside
# this list is supported through constructed-object injection.
BUILTIN_ALIASES = MappingProxyType(
    {
        "detection": frozenset({"scrfd", "centerface"}),
        "pose": frozenset({"insightface-3d68"}),
        "gender": frozenset({"insightface-genderage"}),
        "age": frozenset({"insightface-genderage"}),
        "quality": frozenset({"cr-fiqa"}),
        "embedding": frozenset(
            {"sepaelv2", "sepaelv3", "sepaelv4", "sepaelv5", "sepaelv6"}
        ),
    }
)


def _implements(value, *methods: str) -> bool:
    return all(callable(getattr(value, method, None)) for method in methods)


def _validate_metadata(component, task: str) -> None:
    metadata = getattr(component, "metadata", None)
    required = (
        "component_id",
        "tasks",
        "implementation",
        "runtime",
        "input_space",
        "supports_batch",
    )
    missing = [name for name in required if not hasattr(metadata, name)]
    if missing:
        raise TypeError(
            f"Injected {task} component metadata is missing: {sorted(missing)}."
        )
    if not metadata.component_id or task not in metadata.tasks:
        raise ValueError(
            f"Injected {task} component metadata must have a non-empty "
            f"component_id and declare task '{task}'."
        )


def _selection_spec(task: str, selection) -> ModelSpec | None:
    if isinstance(selection, ModelSpec):
        spec = selection
    elif isinstance(selection, str):
        spec = ModelSpec(selection)
    else:
        return None
    if spec.alias not in BUILTIN_ALIASES[task]:
        raise ValueError(
            f"Unknown {task} model alias '{spec.alias}'. Built-in aliases: "
            f"{sorted(BUILTIN_ALIASES[task])}. Pass a constructed component "
            "object to use a custom implementation."
        )
    return spec


def _validate_options(spec: ModelSpec, allowed: set[str]) -> dict:
    options = dict(spec.options)
    unexpected = set(options) - allowed
    if unexpected:
        raise TypeError(
            f"Unsupported options for '{spec.alias}': {sorted(unexpected)}. "
            f"Allowed options: {sorted(allowed)}."
        )
    return options


def _path_or_resolve(spec: ModelSpec, resolver) -> str:
    if spec.path is not None:
        if not spec.path.is_file():
            raise FileNotFoundError(f"ModelSpec path does not exist: {spec.path}")
        return str(spec.path)
    return resolver()


def build_detector(
    selection,
    *,
    models_root: str,
    providers,
    ctx_id: int,
    det_size: int | tuple[int, int],
    det_thresh: float,
    legacy_model_names: Sequence[str] = (),
) -> FaceDetector:
    spec = _selection_spec("detection", selection)
    if spec is None:
        if not _implements(selection, "detect") or not hasattr(selection, "metadata"):
            raise TypeError(
                "detection must be a built-in alias, ModelSpec, or FaceDetector object."
            )
        _validate_metadata(selection, "detection")
        return selection

    if spec.alias == "scrfd":
        options = _validate_options(
            spec, {"det_size", "det_thresh", "nms_threshold"}
        )
        path = _path_or_resolve(
            spec,
            lambda: resolve_component_model(
                models_root,
                task="detection",
                alias="scrfd",
                filenames=("det_10g.onnx",),
                legacy_paths=("detection/det_10g.onnx",) + tuple(
                    f"{name}/det_10g.onnx" for name in legacy_model_names
                ),
            ),
        )
        return SCRFDDetector(
            path,
            providers=providers,
            ctx_id=ctx_id,
            det_size=options.get("det_size", det_size),
            det_thresh=options.get("det_thresh", det_thresh),
            nms_thresh=options.get("nms_threshold", 0.4),
        )

    options = _validate_options(
        spec, {"score_threshold", "nms_threshold", "input_multiple"}
    )
    path = _path_or_resolve(
        spec,
        lambda: resolve_component_model(
            models_root,
            task="detection",
            alias="centerface",
        ),
    )
    return CenterFaceDetector(path, providers=providers, **options)


def build_face_estimator(
    task: str,
    selection,
    *,
    models_root: str,
    providers,
    ctx_id: int,
    legacy_model_names: Sequence[str] = (),
) -> FaceEstimator:
    spec = _selection_spec(task, selection)
    if spec is None:
        if not _implements(selection, "estimate") or not all(
            hasattr(selection, name) for name in ("metadata", "capabilities")
        ):
            raise TypeError(
                f"{task} must be a built-in alias, ModelSpec, or FaceEstimator object."
            )
        if task not in selection.capabilities:
            raise ValueError(
                f"Injected {task} estimator does not declare the '{task}' capability."
            )
        _validate_metadata(selection, task)
        return selection
    _validate_options(spec, set())

    if task == "pose":
        path = _path_or_resolve(
            spec,
            lambda: resolve_component_model(
                models_root,
                task="pose",
                alias="insightface-3d68",
                filenames=("1k3d68.onnx",),
                legacy_paths=("attributes/1k3d68.onnx",) + tuple(
                    f"{name}/1k3d68.onnx" for name in legacy_model_names
                ),
            ),
        )
        return InsightFacePoseEstimator(path, providers=providers, ctx_id=ctx_id)

    path = _path_or_resolve(
        spec,
        lambda: resolve_component_model(
            models_root,
            task="attributes",
            alias="insightface-genderage",
            filenames=("genderage.onnx",),
            legacy_paths=("attributes/genderage.onnx",) + tuple(
                f"{name}/genderage.onnx" for name in legacy_model_names
            ),
        ),
    )
    return InsightFaceGenderAgeEstimator(path, providers=providers, ctx_id=ctx_id)


def build_quality_estimator(
    selection,
    *,
    models_root: str,
    legacy_model_name: str,
    providers,
) -> QualityEstimator:
    spec = _selection_spec("quality", selection)
    if spec is None:
        if not _implements(selection, "score_one", "score_batch") or not hasattr(
            selection, "metadata"
        ):
            raise TypeError(
                "quality must be a built-in alias, ModelSpec, or QualityEstimator object."
            )
        _validate_metadata(selection, "quality")
        return selection
    _validate_options(spec, set())
    path = _path_or_resolve(
        spec, lambda: resolve_quality_model(models_root, legacy_model_name)
    )
    return CRFIQAQualityEstimator(path, providers=providers)


def normalize_embedding_selection(selection) -> list:
    if isinstance(selection, (str, ModelSpec)) or _implements(
        selection, "embed_one", "embed_batch"
    ):
        return [selection]
    if isinstance(selection, Sequence):
        return list(selection)
    raise TypeError(
        "embedding must be an alias, ModelSpec, embedding component, or sequence of them."
    )


def build_embedding_estimators(
    selections,
    *,
    models_root: str,
    providers,
) -> list[EmbeddingEstimator]:
    estimators: list[EmbeddingEstimator] = []
    for selection in normalize_embedding_selection(selections):
        spec = _selection_spec("embedding", selection)
        if spec is None:
            if not _implements(selection, "embed_one", "embed_batch") or not all(
                hasattr(selection, name)
                for name in ("metadata", "name", "requires_aligned_keypoints")
            ):
                raise TypeError(
                    "Each embedding selection must be an alias, ModelSpec, or "
                    "EmbeddingEstimator object."
                )
            _validate_metadata(selection, "embedding")
            estimators.append(selection)
            continue
        _validate_options(spec, set())
        path = _path_or_resolve(
            spec,
            lambda spec=spec: resolve_recognition_model(models_root, spec.alias),
        )
        estimators.append(
            ONNXEmbeddingEstimator(spec.alias, path, providers=providers)
        )
    names = [estimator.name for estimator in estimators]
    if len(names) != len(set(names)):
        raise ValueError(f"Embedding model names must be unique; received {names}.")
    return estimators


def selection_alias(selection) -> str:
    if isinstance(selection, ModelSpec):
        return selection.alias
    if isinstance(selection, str):
        return selection
    metadata = getattr(selection, "metadata", None)
    return getattr(
        selection,
        "name",
        getattr(metadata, "component_id", type(selection).__name__),
    )
