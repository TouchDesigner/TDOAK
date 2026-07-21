from __future__ import annotations

# face_pipeline.py — face detection + 5 facial landmarks (YuNet, depthai v3)
# Load into the Pipeline DAT parameter of the OAK component.
#
# REQUIRES the `depthai_nodes` package (provides ParsingNeuralNetwork; its parser emits
# ImgDetectionsExtended messages carrying keypoints).
#
# Streams:
#   'rgb'   — BGR passthrough: the exact 640×480 frame inference ran on, 1:1 with the
#             coordinates below.
#   'faces' — a dai.ImgDetections whose detections carry keypoints (YuNet's parser uses
#             the same message shape as the pose models), so the host's built-in decoder
#             yields PoseEntry dicts, one per face:
#               confidence
#               box        [xmin, ymin, xmax, ymax]   normalized 0-1
#               keypoints  [[x, y], …] — 5 landmarks: right eye, left eye, nose,
#                          right mouth corner, left mouth corner (normalized 0-1)
#               edges      [] (YuNet ships no skeleton edges)
#             No Handler is needed for the data itself; touch/face_handler.py is a
#             fill-only Handler that draws boxes + landmarks into a Script POP.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai

# YuNet face detector, 640×480 variant (RVC2 build exists; ':320x240' for more speed).
MODEL: str = 'luxonis/yunet:640x480'

# Declared so the host pre-fetches the model on a subprocess before the build runs.
MODELS: list[str] = [MODEL]


def build(pipeline: dai.Pipeline) -> tuple[dict[str, dai.DataOutputQueue], list[str]]:
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork

    cam = pipeline.create(dai.node.Camera).build()
    nn  = pipeline.create(ParsingNeuralNetwork).build(cam, MODEL)

    queues: dict[str, dai.DataOutputQueue] = {
        'rgb':   nn.passthrough.createOutputQueue(maxSize=1, blocking=False),
        'faces': nn.out.createOutputQueue(maxSize=1, blocking=False),
    }

    return queues, ['rgb', 'faces']
