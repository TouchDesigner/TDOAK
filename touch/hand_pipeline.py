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
#             touch/hand_handler.py — its parse() back-maps each hand's crop-local
#             keypoints onto the full frame using the palm's rotated rect, and fill()
#             draws the 21-point skeleton into a Script SOP.
#
# OAK-D is an RVC2 device → the :192x192 / :224x224 variants below are the RVC2 builds.
# For an OAK4 (RVC4) swap to the matching RVC4 variants.
#
# VERIFY ON DEVICE (can't be checked without the OAK):
#   - CROP_PADDING and the rotated-rect back-map (angle sign) in hand_handler.py — tune
#     until the skeleton sits on the hand. Keep CROP_PADDING here and in the handler equal.
#   - If the landmarker resolves to multiple output heads, ParsingNeuralNetwork.out emits a
#     MessageGroup; hand_handler.parse() already digs the Keypoints out of that case.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai

PALM_MODEL: str = 'luxonis/mediapipe-palm-detection:192x192'    # stage 1 (RVC2)
HAND_MODEL: str = 'luxonis/mediapipe-hand-landmarker:224x224'   # stage 2 (RVC2)
HAND_INPUT: tuple[int, int] = (224, 224)   # stage-2 model input == FrameCropper output
CROP_PADDING: float = 0.1                  # normalized padding around each palm rect
FPS: int = 30                              # camera fps, used by GatherData for matching

# Declared so the host pre-fetches both models on a subprocess before the build runs.
MODELS: list[str] = [PALM_MODEL, HAND_MODEL]


def build(pipeline: dai.Pipeline) -> tuple[dict[str, dai.DataOutputQueue], list[str]]:
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork, FrameCropper, GatherData

    cam = pipeline.create(dai.node.Camera).build()

    # ── Stage 1: palm detection (rotated boxes) ───────────────────────────────
    palm = pipeline.create(ParsingNeuralNetwork).build(cam, PALM_MODEL)

    # One rotated + padded crop per palm, resized to the landmarker's input.
    # fromImgDetections reads det.getBoundingBox() (a RotatedRect) so the crop follows
    # the hand's orientation; STRETCH maps crop coords linearly to that rect, which keeps
    # the landmark back-map in the handler a simple affine transform.
    cropper = (
        pipeline.create(FrameCropper)
        .fromImgDetections(
            inputImgDetections=palm.out,
            outputSize=HAND_INPUT,
            resizeMode=dai.ImageManipConfig.ResizeMode.STRETCH,
            padding=CROP_PADDING,
        )
        .build(inputImage=palm.passthrough)
    )

    # ── Stage 2: 21-point hand landmarks on each crop ─────────────────────────
    landmarks = pipeline.create(ParsingNeuralNetwork).build(cropper.out, HAND_MODEL)

    # ── Re-pair each hand's landmarks with its palm detection ─────────────────
    # waitCountFn defaults to len(reference.detections) → one landmark set per palm.
    gather = pipeline.create(GatherData).build(
        cameraFps=FPS,
        inputData=landmarks.out,
        inputReference=palm.out,
    )

    queues: dict[str, dai.DataOutputQueue] = {
        'rgb':   palm.passthrough.createOutputQueue(maxSize=1, blocking=False),
        'hands': gather.out.createOutputQueue(maxSize=4, blocking=False),
    }

    return queues, ['rgb', 'hands']
