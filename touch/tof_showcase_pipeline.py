from __future__ import annotations

# tof_showcase_pipeline.py — showcases what an OAK ToF does that stereo can't.
# Load into the Pipeline DAT parameter of the OAK component.
# build(pipeline) must return ({name: out_queue}, [stream_names], {name: in_queue})
#
# What's unique to ToF (vs. a stereo OAK-D):
#   • A dense, metric depth map from a single sensor — no texture/matching needed,
#     so it works on blank walls and in the dark where stereo fails.
#   • An on-device point cloud generated directly from that depth.
#   • An 'amplitude' channel: the strength of the returned IR signal, which doubles
#     as a per-pixel confidence map and an IR-lit grayscale image. Amplitude is NOT
#     geometry — depth is the ToF distance measurement; amplitude tells you how much
#     to trust each pixel (mask `depth[amplitude < thresh] = 0` before reprojecting).
#
# Streams:
#   'rgb'        — color camera (auto-detected socket; absent on ToF-only boards)
#   'depth'      — uint16 depth in mm. With ALIGN_DEPTH_TO_RGB it is re-registered
#                  onto the RGB camera's grid: pixel (u, v) in depth == pixel (u, v)
#                  in rgb, so the rgb frame textures the reprojected cloud directly.
#                  Without alignment it stays on the ToF sensor grid.
#   'amplitude'  — ToF return strength (grayscale / confidence). Always on the ToF
#                  sensor grid — when depth is aligned to rgb, amplitude no longer
#                  overlays depth pixel-for-pixel (mask before alignment semantics
#                  don't survive the re-registration; disable alignment if you need
#                  amplitude-masked depth).
#   'pointcloud' — (H, W, 3) float32 XYZ in mm, organized, on the same grid as
#                  'depth'. Only streamed when STREAM_POINTCLOUD is True — see below.
#
# Control queue:
#   'tof_cfg'    — send a dai.ToFConfig to retune the decode at runtime
#                  (e.g. phase-unwrapping level for extended range)
#
# Host reprojection note: when depth is aligned to rgb, reproject through the RGB
# camera's intrinsics — set SOCKET in pointcloudHandler to the RGB socket if the
# auto order (CAM_C, CAM_A, CAM_B) doesn't already land on it.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai

# Re-register depth onto the RGB camera's viewpoint on-device (ImageAlign node).
# Costs a little device compute, buys pixel-for-pixel rgb/depth correspondence.
ALIGN_DEPTH_TO_RGB = True

# OV9782 native is 1280×800; half res keeps PoE headroom (BGR888 at 640×400 @ 30 fps
# ≈ 184 Mbit/s — fine next to depth + amplitude's ≈ 300 Mbit/s).
RGB_RESOLUTION = (640, 400)

# The on-device cloud is bandwidth-hungry: organized XYZ float32 is 12 bytes/px —
# 640×480 @ 30 fps ≈ 885 Mbit/s, which saturates a gigabit PoE link on its own and
# shows up as multi-second latency while every buffer in the chain fills. On PoE,
# leave this False and reproject 'depth' on the host instead (pointcloudHandler +
# GetIntrinsics — depth is 6× smaller on the wire for the same geometry).
# On USB3 (~5 Gbit/s) True is fine.
STREAM_POINTCLOUD = False


def build(pipeline: dai.Pipeline) -> tuple[
    dict[str, dai.DataOutputQueue],
    list[str],
    dict[str, dai.InputQueue],
]:
    import depthai as dai

    # Guard against the wrong device: this pipeline needs a ToF sensor on CAM_A.
    # Started on a stereo OAK-D — easy to do with Device on '-- Select Device --'
    # and several OAKs attached — fail here with a readable message instead of a
    # device-side error.
    device   = pipeline.getDefaultDevice()
    features = {f.socket: f for f in device.getConnectedCameraFeatures()}
    camA = features.get(dai.CameraBoardSocket.CAM_A)
    if camA is None or dai.CameraSensorType.TOF not in camA.supportedTypes:
        raise RuntimeError(
            'tofShowcase needs a ToF sensor on CAM_A, but this device does not have '
            'one — it looks like the stereo OAK-D. Select the ToF board in the Device '
            'menu, or load text_pipeline for this device.')

    # ── ToF sensor (CAM_A) ──────────────────────────────────────────────
    # TOF_MID_RANGE is a balanced preset; switch to TOF_LONG_RANGE / TOF_HIGH_DENSITY
    # to trade range against noise, or retune live via the 'tof_cfg' queue below.
    tof = pipeline.create(dai.node.ToF).build(
        dai.CameraBoardSocket.CAM_A,
        dai.ImageFiltersPresetMode.TOF_MID_RANGE,
    )

    # ── RGB camera: color-capable socket, CAM_C preferred ───────────────
    # Boards like the OAK-D SR carry TWO color cams (CAM_B + CAM_C). Prefer CAM_C
    # so the alignment target matches pointcloudHandler's auto intrinsics order
    # (CAM_C, CAM_A, CAM_B) — reprojecting CAM_B-aligned depth through CAM_C's
    # intrinsics puts a centimeters-scale lateral error in the cloud.
    rgbSocket = next(
        (s for s in (dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_B,
                     dai.CameraBoardSocket.CAM_D)
         if s in features and dai.CameraSensorType.COLOR in features[s].supportedTypes),
        None,
    )

    # maxSize=1, blocking=False — latest-frame semantics. The host drains one message
    # per poll pass, so the default 16-deep queue serves stale frames for seconds when
    # the host falls behind (16 × 3.7 MB of backlog on the pointcloud stream alone).
    queues: dict[str, dai.DataOutputQueue] = {}
    streams: list[str] = []

    rgbOut = None
    if rgbSocket is not None:
        cam    = pipeline.create(dai.node.Camera).build(rgbSocket)
        # NV12, not BGR888p: BGR conversion costs an on-device ImageManip stage, and
        # with the ToF decode already occupying CMX the RVC2 runs out of pool memory
        # ('ImageManip out of memory'). NV12 comes straight off the ISP; the host's
        # getCvFrame() converts to BGR for free on decode.
        rgbOut = cam.requestOutput(RGB_RESOLUTION, type=dai.ImgFrame.Type.NV12)
        queues['rgb'] = rgbOut.createOutputQueue(maxSize=1, blocking=False)
        streams.append('rgb')

    # ── Depth, optionally re-registered onto the RGB grid ───────────────
    depthOut = tof.depth
    if ALIGN_DEPTH_TO_RGB and rgbOut is not None:
        align = pipeline.create(dai.node.ImageAlign)
        # Run the warp on the HOST: on this RVC2 the ToF decode already occupies the
        # CMX pools, and the device-side ImageAlign dies with 'ImageManip out of
        # memory' (225472B vs 150528B available) no matter how its output is sized.
        # Host-side alignment costs a little CPU and zero extra bandwidth — depth
        # crosses the link once either way.
        align.setRunOnHost(True)
        align.setOutputSize(*RGB_RESOLUTION)
        tof.depth.link(align.input)
        rgbOut.link(align.inputAlignTo)
        depthOut = align.outputAligned

    queues['depth']     = depthOut.createOutputQueue(maxSize=1, blocking=False)
    queues['amplitude'] = tof.amplitude.createOutputQueue(maxSize=1, blocking=False)
    streams += ['depth', 'amplitude']

    if STREAM_POINTCLOUD:
        # ── On-device point cloud from the streamed depth ────────────────
        # Linked to depthOut, so the cloud lands on whichever grid 'depth' uses
        # (RGB grid when aligned — texture it with the rgb stream directly).
        # setOrganized(True) keeps every pixel (incl. invalid) so the cloud stays
        # a fixed H×W grid, reshape-able for a 32-bit float TOP / Script POP.
        pcl = pipeline.create(dai.node.PointCloud)
        pcl.initialConfig.setOrganized(True)
        depthOut.link(pcl.inputDepth)
        queues['pointcloud'] = pcl.outputPointCloud.createOutputQueue(maxSize=1, blocking=False)
        streams.append('pointcloud')

    controls: dict[str, dai.InputQueue] = {
        'tof_cfg': tof.tofBaseInputConfig.createInputQueue(maxSize=1, blocking=False),
    }

    return queues, streams, controls
