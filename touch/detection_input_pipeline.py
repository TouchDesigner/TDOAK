from __future__ import annotations

# detection_input_pipeline.py — YOLO object detection on HOST frames (depthai v3)
# Load into the Pipeline DAT parameter of the OAK component.
#
# No camera: the device runs inference on frames TouchDesigner sends it — any TOP
# (movie file, webcam, render) becomes the model input via SendFrame. An on-device
# ImageManip letterboxes/repacks to the model input, so the source can be any size.
#
# REQUIRES the `depthai_nodes` package (provides ParsingNeuralNetwork).
#
# Feed it once per frame from an Execute DAT:
#   def onFrameStart(frame):
#       op('TDOAK').ext.OAKManagerExt.SendFrame('image_in', op('moviefilein1'))
#
# Streams:
#   'rgb'        — the letterboxed frame inference ran on (loops back from the device),
#                  1:1 with the detection coordinates.
#   'detections' — dai.ImgDetections → DetectionEntry dicts (label, confidence, box),
#                  decoded automatically. COCO-80 class indices.
#
# Controls:
#   'image_in'   — the host-frame input queue SendFrame targets.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai

# Same detector as yolo_pipeline.py; its input is 512×288.
MODEL: str = 'luxonis/yolov6-nano:r2-coco-512x288'
NN_INPUT: tuple[int, int] = (512, 288)

# Declared so the host pre-fetches the model on a subprocess before the build runs.
MODELS: list[str] = [MODEL]


def build(pipeline: dai.Pipeline) -> tuple[
    dict[str, dai.DataOutputQueue],
    list[str],
    dict[str, dai.InputQueue],
]:
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork

    w, h = NN_INPUT
    manip = pipeline.create(dai.node.ImageManip)
    manip.setMaxOutputFrameSize(w * h * 3)
    manip.initialConfig.setOutputSize(w, h, dai.ImageManipConfig.ResizeMode.LETTERBOX)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    imageIn = manip.inputImage.createInputQueue(maxSize=1, blocking=False)

    nn = pipeline.create(ParsingNeuralNetwork).build(manip.out, MODEL)

    queues: dict[str, dai.DataOutputQueue] = {
        'rgb':        nn.passthrough.createOutputQueue(maxSize=1, blocking=False),
        'detections': nn.out.createOutputQueue(maxSize=1, blocking=False),
    }
    controls: dict[str, dai.InputQueue] = {
        'image_in': imageIn,
    }

    return queues, ['rgb', 'detections'], controls
