from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from td import scriptTOP


def cook(scriptOp: scriptTOP) -> None:
    ext   = parent.TDOAK.ext.OAKManagerExt
    frame = ext.GetDepth()
    if frame is None:
        return

    h, w = frame.shape
    # The worker delivers a contiguous 2D array, so this reshape is a zero-copy view —
    # no per-cook allocation.
    scriptOp.copyNumpyArray(frame.reshape(h, w, 1))
