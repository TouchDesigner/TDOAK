# pointcloud_handler.py — Handler DAT for the 'depth' stream: metric point cloud into a
# Script POP. `np` is already in scope.
#
# fill-only: the built-in depth decode (uint16 mm, TD-oriented) stays; this reprojects it
# through the device calibration (GetIntrinsics). Works with the default RGB+depth
# pipeline (touch/pipeline.py). Add a Streams block: Stream 'depth', Scriptop → a Script
# POP, Handler → this DAT. Script POPs are pushed directly (no cook callback needed) —
# each new depth frame lands as geometry.
#
# Output: one point per valid depth pixel (strided), P in meters — x right, y up,
# z negative-forward (OpenGL/TD convention, so the cloud sits in front of a default camera).

STRIDE      = 4      # sample every Nth pixel — 640x480 / 4 ≈ 19k points max per frame
MAX_DEPTH_M = 4.0    # drop points further than this (stereo gets noisy with distance)

# Stereo depth is aligned to the rectified-RIGHT camera by default → CAM_C intrinsics.
# Use 'CAM_B' if your pipeline aligns depth to the left camera instead.
SOCKET = 'CAM_C'


def fill(scriptOp, depth) -> None:
    """MAIN THREAD — depth: uint16 (H, W) millimetres, already flipped to TD's
    bottom-up orientation (array row r = sensor row h-1-r)."""
    stagingData = scriptOp.createGeometryStagingData()
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        scriptOp.buildGeometry(stagingData)
        return  # block bound to a non-depth stream — ignore rather than error

    # Resolve the OAK component relative to the bound op (fill runs inside the extension,
    # so a bare op('TDOAK') would resolve inside the OAK component itself).
    oakComp = scriptOp.parent().op('TDOAK')
    k = oakComp.ext.OAKManagerExt.GetIntrinsics(SOCKET) if oakComp else None
    if k is None:
        scriptOp.buildGeometry(stagingData)
        return

    h, w = depth.shape
    # Calibration coefficients refer to the native sensor resolution — scale them to this
    # stream's resolution. (Approximate when the aspect ratios differ, e.g. after a
    # rectification crop; good enough for visualization.)
    sx = w / k['width']  if k['width']  else 1.0
    sy = h / k['height'] if k['height'] else 1.0
    fx, cx = k['fx'] * sx, k['cx'] * sx
    fy, cy = k['fy'] * sy, k['cy'] * sy

    z = depth[::STRIDE, ::STRIDE].astype(np.float32) * 0.001          # mm -> m
    r, u = np.meshgrid(np.arange(0, h, STRIDE, dtype=np.float32),
                       np.arange(0, w, STRIDE, dtype=np.float32), indexing='ij')
    valid = (z > 0.0) & (z < MAX_DEPTH_M)

    x   = (u - cx) * z / fx                      # camera x: right
    yTd = (cy - (h - 1.0 - r)) * z / fy          # camera y down -> TD y up, via sensor row
    pts = np.stack([x[valid], yTd[valid], -z[valid]], axis=-1)

    stagingData.createPointAttrib(name='P')
    stagingData.appendPointAttribValues('P', pts.tolist())
    scriptOp.buildGeometry(stagingData)
