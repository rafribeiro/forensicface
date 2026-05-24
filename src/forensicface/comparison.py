"""Face comparison and image-set aggregation workflows."""

from __future__ import annotations

import numpy as np

from .utils import aggregate_embeddings, cosine_score


__all__ = ["aggregate_from_images", "compare_faces"]


def _aggregate_embeddings(processor, embeddings, *, method: str, weights):
    aggregate = getattr(processor, "aggregate_embeddings", aggregate_embeddings)
    return aggregate(embeddings, method=method, weights=weights)


def compare_faces(processor, img1path: str, img2path: str) -> float:
    """Compare two images through the processor facade."""
    if not processor.concat_embeddings:
        raise ValueError(
            "compare() is not compatible with concat_embeddings=False. "
            "Instantiate ForensicFace with concat_embeddings=True, or "
            "compare the model-specific embedding_<model_name> arrays manually."
        )

    img1data = processor.process_image(img1path, single_face=True)
    if len(img1data) == 0:
        raise ValueError(f"No face detected in {img1path}")
    img2data = processor.process_image(img2path, single_face=True)
    if len(img2data) == 0:
        raise ValueError(f"No face detected in {img2path}")

    return cosine_score(img1data["embedding"], img2data["embedding"])


def aggregate_from_images(
    processor,
    list_of_image_paths: list[str],
    method: str = "mean",
    quality_weight: bool = False,
) -> np.ndarray | dict[str, np.ndarray] | list:
    """Aggregate embeddings from all detected faces in a list of images."""
    if quality_weight and processor.extended is not True:
        raise ValueError("You must initialize ForensicFace with extended = True")

    if processor.concat_embeddings:
        embeddings = []
    else:
        embeddings = {model_name: [] for model_name in processor.models}
    weights = []

    for imgpath in list_of_image_paths:
        result = processor.process_image(imgpath, single_face=True)
        if len(result) == 0:
            continue

        if processor.concat_embeddings:
            embeddings.append(result["embedding"])
        else:
            for model_name in processor.models:
                embeddings[model_name].append(result[f"embedding_{model_name}"])
        weights.append(result["fiqa_score"] if quality_weight else 1.0)

    if len(weights) == 0:
        return []

    weights_array = np.array(weights)
    if not processor.concat_embeddings:
        return {
            f"embedding_{model_name}": _aggregate_embeddings(
                processor,
                np.array(model_embeddings),
                method=method,
                weights=weights_array,
            )
            for model_name, model_embeddings in embeddings.items()
        }
    return _aggregate_embeddings(
        processor,
        np.array(embeddings),
        method=method,
        weights=weights_array,
    )
