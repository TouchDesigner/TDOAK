from __future__ import annotations

# segmentation_pipeline.py — person segmentation (DeepLabV3+, depthai v3)
# Load into the Pipeline DAT parameter of the OAK component.
#
# REQUIRES the `depthai_nodes` package (provides ParsingNeuralNetwork; its
# SegmentationParser emits SegmentationMask messages).
#
# Streams:
#   'rgb'          — passthrough (RGB on the host): the exact 256×256 frame inference ran on, 1:1
#                    with the mask below.
#   'segmentation' — a depthai-nodes SegmentationMask. The host's built-in decoder turns
#                    it into a uint16 (H, W) class-index map (0 = background, flipped to
#                    TD orientation), which drops straight into a Script TOP — threshold
#                    with a Level/Threshold TOP for a person matte. For a POP variant,
#                    put segmentation_handler.py in the Streams block's Handler and
#                    bind a Script POP: its fill() scatters one point per person pixel
#                    (strided) — feed particles, instancing, etc.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai

# Person-specific DeepLabV3+ variant (RVC2 build exists).
MODEL: str = 'luxonis/deeplab-v3-plus:person-256x256'

# Declared so the host pre-fetches the model on a subprocess before the build runs.
MODELS: list[str] = [MODEL]


def build(pipeline: dai.Pipeline) -> tuple[dict[str, dai.DataOutputQueue], list[str]]:
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork

    cam = pipeline.create(dai.node.Camera).build()
    nn  = pipeline.create(ParsingNeuralNetwork).build(cam, MODEL)

    queues: dict[str, dai.DataOutputQueue] = {
        'rgb':          nn.passthrough.createOutputQueue(maxSize=1, blocking=False),
        'segmentation': nn.out.createOutputQueue(maxSize=1, blocking=False),
    }

    return queues, ['rgb', 'segmentation']
