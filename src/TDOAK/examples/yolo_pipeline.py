from __future__ import annotations

# yolo_pipeline.py — YOLO object detection (depthai v3)
# Load into the Pipeline DAT parameter of the OAK component.
#
# REQUIRES the `depthai_nodes` package (already in this project's .venv):
#   it provides ParsingNeuralNetwork, which resolves the model from the Luxonis
#   Model Zoo, runs it, and parses the raw output into a standard dai.ImgDetections
#   message — no hand-written tensor decoding needed.
#
# Streams:
#   'rgb'        — passthrough frames (RGB on the host): the exact letterboxed frame inference ran
#                  on, so the normalized boxes below map straight onto it (1:1).
#   'detections' — a dai.ImgDetections message; one entry per object, each carrying:
#                    .label               int class index (see COCO-80 below)
#                    .confidence          float
#                    .xmin/.ymin/
#                    .xmax/.ymax          bounding box, normalized 0-1
#
# NOTE: OAKManagerExt decodes a `.detections` message with no keypoints into bbox-only
# `DetectionEntry` dicts (label, confidence, xmin, ymin, xmax, ymax) automatically —
# no Handler needed. (A detector that also carries keypoints decodes to `PoseEntry`.)
#
# COCO-80 class indices (label):
#   0 person        1 bicycle       2 car           3 motorcycle    4 airplane
#   5 bus           6 train         7 truck         8 boat          9 traffic light
#   10 fire hydrant 11 stop sign    12 parking meter 13 bench       14 bird
#   15 cat          16 dog          17 horse        18 sheep        19 cow
#   …see the Luxonis model card for the full list. Map indices to names in a Handler
#   DAT or downstream in TD if you need labels.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai

# Model Zoo slug + variant. yolov6-nano is the lightweight COCO default; swap to
# 'luxonis/yolov8-nano:coco-512x288' for YOLOv8, or a -small/-medium variant for
# accuracy over speed.
MODEL: str = 'luxonis/yolov6-nano:r2-coco-512x288'

# Declared so the host (OAKManagerExt) pre-fetches every model on a subprocess with a live
# progress counter before the in-process build runs. List every zoo slug build() resolves.
MODELS: list[str] = [MODEL]

def build(pipeline: dai.Pipeline) -> tuple[dict[str, dai.DataOutputQueue], list[str]]:
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork

    cam = pipeline.create(dai.node.Camera).build()
    nn  = pipeline.create(ParsingNeuralNetwork).build(cam, MODEL)

    queues: dict[str, dai.DataOutputQueue] = {
        'rgb':        nn.passthrough.createOutputQueue(maxSize=1, blocking=False),
        'detections': nn.out.createOutputQueue(maxSize=1, blocking=False),
    }

    return queues, ['rgb', 'detections']
