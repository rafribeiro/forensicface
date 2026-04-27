"""Legacy compatibility wrapper for forensicface.ForensicFace."""

import warnings

from .app import ForensicFace as _ForensicFace

__all__ = ["ForensicFace"]


class ForensicFace(_ForensicFace):
    """Deprecated alias kept for backward compatibility.

    Use ``forensicface.app.ForensicFace`` instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "forensicface.forensicface.ForensicFace is deprecated and will be removed in a future release. "
            "Use forensicface.app.ForensicFace instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
