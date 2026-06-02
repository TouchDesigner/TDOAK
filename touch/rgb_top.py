from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from td import scriptTOP


def cook(scriptOp: scriptTOP) -> None:
    ext   = parent.TDOAK.ext.OAKManagerExt
    frame = ext.GetFrame('rgb')
    if frame is None:
        return

    rgba = np.dstack([frame, np.full(frame.shape[:2], 255, dtype=np.uint8)])
    scriptOp.copyNumpyArray(rgba)
