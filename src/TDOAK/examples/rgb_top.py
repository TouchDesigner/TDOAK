from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from td import scriptTOP

# Reused RGBA buffer (alpha pre-filled) — allocating a fresh ~2 MB array per cook is
# pure GC churn. copyNumpyArray copies synchronously, so reuse across cooks is safe.
_rgba: np.ndarray | None = None


def cook(scriptOp: scriptTOP) -> None:
    global _rgba
    ext   = parent.TDOAK.ext.OAKManagerExt
    frame = ext.GetRgb()
    if frame is None:
        return

    if _rgba is None or _rgba.shape[:2] != frame.shape[:2]:
        _rgba = np.empty((*frame.shape[:2], 4), dtype=np.uint8)
        _rgba[:, :, 3] = 255
    _rgba[:, :, :3] = frame
    scriptOp.copyNumpyArray(_rgba)
