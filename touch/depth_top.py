from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from td import scriptTOP


def cook(scriptOp: scriptTOP) -> None:
    ext   = op('..').ext.OAKManagerExt
    frame = ext.GetFrame('depth')
    if frame is None:
        return

    h, w  = frame.shape
    out   = np.empty((h, w, 1), dtype=np.uint16)
    out[:, :, 0] = frame
    scriptOp.copyNumpyArray(out)
