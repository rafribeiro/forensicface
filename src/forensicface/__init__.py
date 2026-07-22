from .components import (
    ComponentMetadata,
    EmbeddingEstimator,
    FaceDetector,
    FaceEstimator,
    ModelSpec,
    QualityEstimator,
)
from .backends import PoseAngles
from .results import FaceResult

__version__ = "0.8.0"
__all__ = [
    "ComponentMetadata",
    "EmbeddingEstimator",
    "FaceDetector",
    "FaceEstimator",
    "FaceResult",
    "ModelSpec",
    "PoseAngles",
    "QualityEstimator",
    "__version__",
]
