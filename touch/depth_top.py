# depth Script TOP — cook callback
import numpy as np


def cook(scriptOp):
    ext   = op('..').ext.OAKManagerExt
    frame = ext.GetDepth()
    if frame is None:
        return

    # uint16 mm values — pass raw for downstream Level TOP / GLSL processing.
    # To normalise: frame / 10000.0 maps 0–10 m → 0.0–1.0 float.
    scriptOp.copyNumpyArray(frame[:, :, np.newaxis])
