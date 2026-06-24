from __future__ import annotations
from datetime import timedelta

# pose_pipeline.py — YOLOv8-nano human pose / skeleton tracking (depthai v3)
# Load into the Pipeline DAT parameter of the OAK component.
#
# REQUIRES the `depthai_nodes` package (already in this project's .venv):
#   it provides ParsingNeuralNetwork, which resolves the model from the Luxonis
#   Model Zoo, runs it, and parses the raw output into a standard dai.ImgDetections
#   message — no hand-written tensor decoding needed.
#
# Streams:
#   'rgb'  — BGR passthrough frames: the exact letterboxed frame inference ran on,
#            so the normalized keypoints below map straight onto it (1:1).
#   'pose' — a dai.ImgDetections message; one detection per person, each carrying:
#              .confidence          float
#              .xmin/.ymin/
#              .xmax/.ymax          bounding box, normalized 0-1
#              .getKeypoints2f()    17 COCO joints as Point2f (.x/.y, normalized 0-1)
#              .getEdges()          skeleton bone connections as [i, j] keypoint-index
#                                   pairs (19 edges) — draw a line per pair to render
#                                   the skeleton.
#
# NOTE: OAKManagerExt decodes a `.detections` message whose detections carry keypoints into
# `PoseEntry` dicts (confidence, box, keypoints, edges) — skeletons are surfaced automatically,
# no Handler needed. Plain object detectors (no keypoints) decode to bbox-only `DetectionEntry`.
#
# COCO-17 keypoint order:
#   0 nose       1 lEye       2 rEye       3 lEar       4 rEar
#   5 lShoulder  6 rShoulder  7 lElbow     8 rElbow     9 lWrist    10 rWrist
#   11 lHip      12 rHip      13 lKnee     14 rKnee     15 lAnkle   16 rAnkle

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai

# Model Zoo slug + variant (288×512 input). Swap to the large variant
# 'luxonis/yolov8-large-pose-estimation:coco-640x352' for accuracy over speed.
MODEL: str = 'luxonis/yolov8-nano-pose-estimation:coco-512x384'

# Declared so the host (OAKManagerExt) pre-fetches every model on a subprocess with a live
# progress counter before the in-process build runs. List every zoo slug build() resolves.
MODELS: list[str] = [MODEL]

def build(pipeline: dai.Pipeline) -> tuple[dict[str, dai.DataOutputQueue], list[str]]:
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork

    cam = pipeline.create(dai.node.Camera).build()
    hd = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p, fps=30)
    nn  = pipeline.create(ParsingNeuralNetwork).build(cam, MODEL)
    
    # sync camera frame and detection frame
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=10))
    hd.link(sync.inputs['rgb'])
    nn.out.link(sync.inputs['pose'])
    
    demux = pipeline.create(dai.node.MessageDemux)
    sync.out.link(demux.input)

    queues: dict[str, dai.DataOutputQueue] = {
        'rgb': demux.outputs['rgb'].createOutputQueue(maxSize=1, blocking=False),
        'pose': demux.outputs['pose'].createOutputQueue(maxSize=1, blocking=False)
    }

    return queues, ['rgb', 'pose']
