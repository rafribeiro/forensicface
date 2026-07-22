"""Initialization summary helpers for ForensicFace."""

from __future__ import annotations


def _session_providers(session) -> list[str]:
    if session is None or not hasattr(session, "get_providers"):
        return []

    try:
        providers = session.get_providers()
    except Exception:
        return []

    return [str(provider) for provider in providers if provider]


def collect_session_provider_details(ff) -> list[tuple[str, list[str]]]:
    details: list[tuple[str, list[str]]] = []

    backend = getattr(ff, "backend", None)
    if hasattr(backend, "detector"):
        components = []
        components.extend(getattr(ff, "embedding_estimators", None) or [])
        components.append(backend.detector)
        components.extend(getattr(backend, "estimators", []))
        quality = getattr(ff, "quality_estimator", None)
        if quality is not None:
            components.append(quality)

        seen: set[int] = set()
        for component in components:
            if id(component) in seen:
                continue
            seen.add(id(component))
            metadata = getattr(component, "metadata", None)
            providers = [str(value) for value in getattr(metadata, "providers", ())]
            if providers:
                details.append((metadata.component_id, providers))
        return details

    for model_name, session in zip(ff.models, ff.rec_inference_sessions):
        providers = _session_providers(session)
        if providers:
            details.append((model_name, providers))

    det_session = getattr(getattr(backend, "det_model", None), "session", None)
    det_providers = _session_providers(det_session)
    if det_providers:
        details.append(("detection", det_providers))

    headpose_session = getattr(getattr(backend, "landmark_model", None), "session", None)
    headpose_providers = _session_providers(headpose_session)
    if headpose_providers:
        details.append(("headpose", headpose_providers))

    genderage_session = getattr(getattr(backend, "genderage_model", None), "session", None)
    genderage_providers = _session_providers(genderage_session)
    if genderage_providers:
        details.append(("genderage", genderage_providers))

    fiqa_providers = _session_providers(getattr(ff, "ort_fiqa", None))
    if fiqa_providers:
        details.append(("cr_fiqa", fiqa_providers))

    return details


def _best_provider(providers: list[str]) -> str | None:
    if not providers:
        return None

    if "CUDAExecutionProvider" in providers:
        return "CUDAExecutionProvider"

    if "CoreMLExecutionProvider" in providers:
        return "CoreMLExecutionProvider"

    if "CPUExecutionProvider" in providers:
        return "CPUExecutionProvider"

    return providers[0]


def get_effective_provider(ff) -> str:
    candidate_providers = [
        provider
        for _, providers in collect_session_provider_details(ff)
        for provider in providers
    ]

    provider = _best_provider(candidate_providers)
    if provider is not None:
        return provider

    if getattr(ff, "providers", None):
        return str(ff.providers[0])

    return "unknown"


def build_initialization_summary(ff) -> str:
    session_provider_details = collect_session_provider_details(ff)
    session_primary_providers = [
        providers[0]
        for _, providers in session_provider_details
        if providers
    ]
    session_provider_summary = "unknown"
    if session_primary_providers:
        unique_providers = sorted(set(session_primary_providers))
        if len(unique_providers) == 1:
            session_provider_summary = f"all models use {unique_providers[0]}"
        else:
            parts = [f"{label}: {providers[0]}" for label, providers in session_provider_details]
            session_provider_summary = ", ".join(parts) if parts else "unknown"

    backend = getattr(ff, "backend", None)
    detector = getattr(backend, "detector", None)
    det_size_summary = getattr(
        detector,
        "input_size_summary",
        getattr(ff, "det_size", "unknown"),
    )

    return (
        "[ForensicFace] Initialized with configuration:\n"
        f"                  loaded_models={ff.models}\n"
        f"                  modules={ff._get_loaded_modules()}\n"
        f"                  det_size={det_size_summary}\n"
        f"                  session_providers={session_provider_summary}\n"
    )


def print_initialization_summary(ff) -> None:
    print(build_initialization_summary(ff))
