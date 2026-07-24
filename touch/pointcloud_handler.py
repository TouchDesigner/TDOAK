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

# Which camera's intrinsics to reproject through. None = auto: stereo depth is aligned
# to the rectified-RIGHT camera (CAM_C); a ToF sensor carries its own intrinsics on
# CAM_A and has no CAM_C at all. Set explicitly (e.g. 'CAM_B' for left-aligned stereo)
# to override the auto order.
SOCKET = None


def fill(scriptOp, depth) -> None:
    """MAIN THREAD — depth: uint16 (H, W) millimetres, already flipped to TD's
    bottom-up orientation (array row r = sensor row h-1-r)."""
    stagingData = scriptOp.createGeometryStagingData()
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        scriptOp.buildGeometry(stagingData)
        return  # block bound to a non-depth stream — ignore rather than error

    # `oak` — the OAK component — is injected into every Handler DAT's namespace by the
    # extension (fill runs inside the extension, where a bare op('TDOAK') would resolve
    # inside the component itself and return None).
    sockets = (SOCKET,) if SOCKET else ('CAM_C', 'CAM_A', 'CAM_B')
    k = next((ki for s in sockets if (ki := oak.GetIntrinsics(s)) is not None), None)
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
    # copyNumpyArrayToPointAttrib bulk-copies the (N, 3) float32 buffer — far cheaper
    # than appendPointAttribValues(pts.tolist()), which builds N Python tuples per frame.
    stagingData.copyNumpyArrayToPointAttrib('P', np.ascontiguousarray(pts))

    # Per-point color from the companion rgb stream: with depth aligned to the RGB
    # camera (tofShowcase ALIGN_DEPTH_TO_RGB) both share one grid, so the same stride
    # + validity mask picks each point's pixel. Skipped when rgb isn't streaming yet
    # or lives on a different grid (alignment off) — the cloud just has no Color then.
    rgb = oak.GetFrame('rgb')
    if isinstance(rgb, np.ndarray) and rgb.ndim == 3 and rgb.shape[:2] == depth.shape:
        col = rgb[::STRIDE, ::STRIDE][valid][:, :3].astype(np.float32) / 255.0
        rgba = np.empty((col.shape[0], 4), dtype=np.float32)
        rgba[:, :3] = col
        rgba[:, 3]  = 1.0
        # 'Color' is TD's standard POP color attribute (RGBA, like 'P' it comes with
        # its component count built in).
        stagingData.createPointAttrib(name='Color')
        stagingData.copyNumpyArrayToPointAttrib('Color', rgba)

    scriptOp.buildGeometry(stagingData)