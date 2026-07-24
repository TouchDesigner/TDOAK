from __future__ import annotations

# hand_pipeline.py — MediaPipe two-stage hand tracking (depthai v3)
# Load into the Pipeline DAT parameter of the OAK component.
#
# Stage 1  palm detection      → dai.ImgDetections with ROTATED boxes (one per hand)
# Stage 2  hand landmarker     → 21 normalized keypoints per cropped hand
#
# Graph:
#   Camera ─► ParsingNeuralNetwork(palm) ─► FrameCropper(rotated crop per palm)
#                       │                              │
#                       │ passthrough ('rgb')          ▼
#                       │                    ParsingNeuralNetwork(hand landmarks)
#                       ▼                              │
#                  GatherData ◄────────────────────────┘   (pairs landmarks↔detection
#                       │                                    by timestamp + count)
#                       ▼
#                    'hands'  (dai GatheredData → decoded by hand_handler.py)
#
# REQUIRES `depthai_nodes` (provides ParsingNeuralNetwork, FrameCropper, GatherData).
#
# Streams:
#   'rgb'   — the 192×192 frame the palm detector ran on (palm.passthrough). Detection
#             and landmark coords are normalized to THIS frame, so overlays line up 1:1.
#   'hands' — a depthai_nodes GatheredData message. It is NOT one of the host's built-in
#             decoders, so bind this stream to a Streams block whose Handler points at
#             hand_handler.py — its parse() back-maps each hand's crop-local
#             keypoints onto the full frame using the palm's rotated rect, and fill()
#             draws the 21-point skeleton into a Script SOP.
#
# OAK-D is an RVC2 device → the :192x192 / :224x224 variants below are the RVC2 builds.
# For an OAK4 (RVC4) swap to the matching RVC4 variants.
#
# VERIFY ON DEVICE (can't be checked without the OAK):
#   - CROP_PADDING and the rotated-rect back-map (angle sign) in hand_handler.py — tune
#     until the skeleton sits on the hand. The value is shared via TDOAK.core.constants, so
#     tuning it there adjusts both sides at once.
#   - If the landmarker resolves to multiple output heads, ParsingNeuralNetwork.out emits a
#     MessageGroup; hand_handler.parse() already digs the Keypoints out of that case.
#   - Hands near the frame edge: the padded crop rect can extend outside [0, 1]. Depending
#     on firmware, ImageManip either clamps (shifting the effective crop, which makes the
#     handler's affine back-map approximate near edges) or errors per detection (the crop
#     Script now skips that detection and keeps running). Check both behaviors on device.

from typing import TYPE_CHECKING

from TDOAK.core.constants import HAND_CROP_PADDING

if TYPE_CHECKING:
    import depthai as dai

PALM_MODEL: str = 'luxonis/mediapipe-palm-detection:192x192'    # stage 1 (RVC2)
HAND_MODEL: str = 'luxonis/mediapipe-hand-landmarker:224x224'   # stage 2 (RVC2)
HAND_INPUT: tuple[int, int] = (224, 224)   # stage-2 model input == FrameCropper output
CROP_PADDING: float = HAND_CROP_PADDING    # shared with hand_handler.py via TDOAK.core.constants
FPS: int = 30                              # camera fps, used by GatherData for matching

# Declared so the host pre-fetches both models on a subprocess before the build runs.
MODELS: list[str] = [PALM_MODEL, HAND_MODEL]


def _landmarksOutput(pnn) -> dai.Node.Output:
    """The hand landmarker exposes 3 heads (keypoints + score + handedness), so its `.out`
    is unavailable — pick the KeypointParser head's output by type (it's head 0, but select
    on type so head reordering can't break us)."""
    i = 0
    while True:
        try:
            parser = pnn.getParser(index=i)
        except KeyError:
            break
        if type(parser).__name__ == 'KeypointParser':
            return pnn.getOutput(i)
        i += 1
    return pnn.getOutput(0)


def build(pipeline: dai.Pipeline) -> tuple[dict[str, dai.DataOutputQueue], list[str]]:
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork, GatherData

    cam = pipeline.create(dai.node.Camera).build()

    # ── Stage 1: palm detection (rotated boxes) ───────────────────────────────
    palm = pipeline.create(ParsingNeuralNetwork).build(cam, PALM_MODEL)

    # ── One rotated + padded crop per palm, resized to the landmarker input ────
    # NB: we DON'T use depthai_nodes' FrameCropper.fromImgDetections — its on-device
    # Script builds Size2f(width=…, height=…, normalized=True), but the firmware's
    # Size2f only accepts positional args, so it raises per detection. We emit the crop
    # configs ourselves and set the rect via attribute assignment (the same approach as
    # depthai_nodes' detection_config_generator), which avoids the Size2f constructor.
    # STRETCH maps crop coords linearly onto the rect → the handler back-map stays affine.
    #
    # The script re-pairs frame/detections by sequence number: passthrough and out normally
    # arrive in lockstep, but one dropped message would otherwise shift the pairing by one
    # PERMANENTLY — every later crop cut from the wrong frame with no error anywhere.
    # The try/except sits INSIDE the loops so one bad detection (e.g. a degenerate rect)
    # skips that detection instead of killing the whole 'hands' stream for the session.
    cropW, cropH = HAND_INPUT
    cropScript = pipeline.create(dai.node.Script)
    cropScript.setScript(f"""
while True:
    try:
        frame = node.inputs['frame'].get()
        dets  = node.inputs['dets'].get()
        while frame.getSequenceNum() < dets.getSequenceNum():
            frame = node.inputs['frame'].get()
        while dets.getSequenceNum() < frame.getSequenceNum():
            dets = node.inputs['dets'].get()
        for det in dets.detections:
            try:
                rr = det.getBoundingBox()
                rect = RotatedRect()
                rect.center.x   = rr.center.x
                rect.center.y   = rr.center.y
                rect.size.width  = rr.size.width  + 2 * {CROP_PADDING}
                rect.size.height = rr.size.height + 2 * {CROP_PADDING}
                rect.angle = rr.angle
                cfg = ImageManipConfig()
                cfg.addCropRotatedRect(rect, True)
                cfg.setOutputSize({cropW}, {cropH}, ImageManipConfig.ResizeMode.STRETCH)
                cfg.setFrameType(ImgFrame.Type.BGR888p)
                cfg.setTimestamp(dets.getTimestamp())
                cfg.setSequenceNum(dets.getSequenceNum())
                node.outputs['cfg'].send(cfg)
                node.outputs['img'].send(frame)
            except Exception as e:
                node.warn('crop skipped: ' + str(e))
    except Exception as e:
        node.error(str(e))
""")
    palm.passthrough.link(cropScript.inputs['frame'])
    palm.out.link(cropScript.inputs['dets'])

    crop = pipeline.create(dai.node.ImageManip)
    crop.setMaxOutputFrameSize(cropW * cropH * 3)
    crop.inputConfig.setWaitForMessage(True)
    cropScript.outputs['cfg'].link(crop.inputConfig)
    cropScript.outputs['img'].link(crop.inputImage)

    # ── Stage 2: 21-point hand landmarks on each crop ─────────────────────────
    landmarks = pipeline.create(ParsingNeuralNetwork).build(crop.out, HAND_MODEL)

    # ── Re-pair each hand's landmarks with its palm detection ─────────────────
    # waitCountFn defaults to len(reference.detections) → one landmark set per palm.
    # The landmarker is multi-head, so take the keypoints head specifically (see above).
    gather = pipeline.create(GatherData).build(
        cameraFps=FPS,
        inputData=_landmarksOutput(landmarks),
        inputReference=palm.out,
    )

    queues: dict[str, dai.DataOutputQueue] = {
        'rgb':   palm.passthrough.createOutputQueue(maxSize=1, blocking=False),
        'hands': gather.out.createOutputQueue(maxSize=4, blocking=False),
    }

    return queues, ['rgb', 'hands']
