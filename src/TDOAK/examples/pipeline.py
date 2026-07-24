from __future__ import annotations

# pipeline.py — default OAK pipeline with synced RGB + depth (depthai v3)
# Load into the Pipeline DAT parameter of the OAK component.
# build(pipeline) must return ({name: out_queue}, [stream_names], {name: in_queue})

from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai

SYNC_THRESHOLD_MS = 50


def build(pipeline: dai.Pipeline) -> tuple[
    dict[str, dai.DataOutputQueue],
    list[str],
    dict[str, dai.InputQueue],
]:
    import depthai as dai

    # Guard against the wrong device: this pipeline assumes the stereo OAK-D layout
    # (color on CAM_A, stereo pair on CAM_B/C). Started on a ToF board — easy to do
    # with Device on '-- Select Device --' and several OAKs attached — the ToF sensor
    # can't act as a video camera and the device firmware eventually crashes, taking
    # every stream down with X_LINK_ERROR. Fail here with a readable message instead.
    feats = {f.socket: f for f in pipeline.getDefaultDevice().getConnectedCameraFeatures()}
    camA  = feats.get(dai.CameraBoardSocket.CAM_A)
    if camA is None or dai.CameraSensorType.COLOR not in camA.supportedTypes:
        raise RuntimeError(
            'pipeline.py expects a color camera on CAM_A (stereo OAK-D layout), but this '
            'device does not have one — it looks like the ToF board. Select the OAK-D in '
            'the Device menu, or load the ToF showcase pipeline for this device.')

    # ── RGB camera ────────────────────────────────────────────────────────
    cam     = pipeline.create(dai.node.Camera).build()
    rgb_out = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.NV12)

    # ── Stereo depth ──────────────────────────────────────────────────────
    left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setRectification(True)
    stereo.setLeftRightCheck(True)
    left.requestFullResolutionOutput().link(stereo.left)
    right.requestFullResolutionOutput().link(stereo.right)

    # ── Sync + demux ──────────────────────────────────────────────────────
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=SYNC_THRESHOLD_MS))
    rgb_out.link(sync.inputs['rgb'])
    stereo.depth.link(sync.inputs['depth'])

    demux = pipeline.create(dai.node.MessageDemux)
    sync.out.link(demux.input)

    # maxSize=1, blocking=False — latest-frame semantics. The host drains one message per
    # poll pass, so the default 16-deep queue would serve up to ~half a second of stale frames.
    queues: dict[str, dai.DataOutputQueue] = {
        'rgb':   demux.outputs['rgb'].createOutputQueue(maxSize=1, blocking=False),
        'depth': demux.outputs['depth'].createOutputQueue(maxSize=1, blocking=False),
    }

    controls: dict[str, dai.InputQueue] = {
        'cam_ctrl':   cam.inputControl.createInputQueue(maxSize=1, blocking=False),
        'stereo_cfg': stereo.inputConfig.createInputQueue(maxSize=1, blocking=False),
    }

    return queues, ['rgb', 'depth'], controls
