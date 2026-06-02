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

    # ── RGB camera ────────────────────────────────────────────────────────
    cam     = pipeline.create(dai.node.Camera).build()
    rgb_out = cam.requestOutput((1920, 1080), type=dai.ImgFrame.Type.BGR888p)

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

    queues: dict[str, dai.DataOutputQueue] = {
        'rgb':   demux.outputs['rgb'].createOutputQueue(),
        'depth': demux.outputs['depth'].createOutputQueue(),
    }

    controls: dict[str, dai.InputQueue] = {
        'cam_ctrl':   cam.inputControl.createInputQueue(maxSize=1, blocking=False),
        'stereo_cfg': stereo.inputConfig.createInputQueue(maxSize=1, blocking=False),
    }

    return queues, ['rgb', 'depth'], controls
