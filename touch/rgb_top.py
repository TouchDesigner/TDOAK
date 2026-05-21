# rgb Script TOP — cook callback
# Set par.stream = 'rgb' on this Script TOP.
import numpy as np


def cook(scriptOp):
    ext   = op('..').ext.OAKManagerExt
    frame = ext.GetRgb()
    if frame is None:
        return

    # DepthAI delivers BGR uint8 (H, W, 3) — add alpha channel for TD
    rgba = np.dstack([frame, np.full(frame.shape[:2], 255, dtype=np.uint8)])
    scriptOp.copyNumpyArray(rgba)
