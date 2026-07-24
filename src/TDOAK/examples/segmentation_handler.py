# segmentation_handler.py — optional Handler DAT for the 'segmentation' stream of
# segmentation_pipeline.py, for a POP silhouette. `np` is already in scope.
#
# fill-only: the built-in SegmentationMask decode (uint16 class map, TD-oriented) stays,
# this just shapes it into points. Bind the Streams block's Scriptop to a Script POP —
# POPs are pushed directly on each new frame, no cook callback needed.
#
# Output: one point per person pixel (strided), P normalized 0-1 on the image plane
# (z = 0) — displace/instance/emit from them downstream.

STRIDE = 3   # sample every Nth mask pixel; 256×256 / 3 ≈ 7k points worst case


def fill(scriptOp, mask) -> None:
    """MAIN THREAD — mask: uint16 (H, W) class indices, 0 = background, already flipped
    to TD's bottom-up orientation (so row index maps to y-up directly)."""
    stagingData = scriptOp.createGeometryStagingData()
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        scriptOp.buildGeometry(stagingData)
        return  # block bound to a non-mask stream — ignore rather than error

    sub = mask[::STRIDE, ::STRIDE]
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        scriptOp.buildGeometry(stagingData)
        return

    h, w = mask.shape
    pts = np.stack([
        xs.astype(np.float32) * STRIDE / w,
        ys.astype(np.float32) * STRIDE / h,
        np.zeros(len(xs), dtype=np.float32),
    ], axis=-1)

    stagingData.createPointAttrib(name='P')
    stagingData.appendPointAttribValues('P', pts.tolist())
    scriptOp.buildGeometry(stagingData)
