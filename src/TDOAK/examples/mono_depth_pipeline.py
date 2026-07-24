from __future__ import annotations

# mono_depth_pipeline.py — monocular (single-camera) depth via MiDaS v2.1 (depthai v3)
# Neural depth from ONE ordinary RGB camera — no stereo pair, no ToF.
# Load into the Pipeline DAT parameter of the OAK component.
#
# REQUIRES the `depthai_nodes` package (provides ParsingNeuralNetwork, which resolves the
# model from the Luxonis Model Zoo and parses its output for you).
#
# MiDaS v2.1 ships an RVC2 (Myriad X / OAK-D) build — unlike Depth-Anything, which is
# RVC4-only and won't run on these devices.
#
# Streams:
#   'rgb'   — BGR passthrough: the exact frame inference ran on (384x256), so the depth
#             map below is pixel-aligned with it.
#   'depth' — a depthai_nodes Map2D message. OAKManagerExt's built-in Map2D decoder turns
#             it into a float32 (H, W) array, so a Script TOP bound to 'depth' fills
#             automatically as a single-channel float TOP. Values are RELATIVE inverse
#             depth (larger = nearer), NOT metric — normalize for display, e.g. divide by
#             the frame's max in a Script TOP or with a Math/Normalize TOP downstream.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai

# Model Zoo slug + variant. Other RVC2 variants: 256x192 (faster), 288x512 / 384x512 (finer).
MODEL: str = 'luxonis/midas-v2-1:small-384x256'


def build(pipeline: dai.Pipeline) -> tuple[dict[str, dai.DataOutputQueue], list[str]]:
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork

    cam = pipeline.create(dai.node.Camera).build()
    nn  = pipeline.create(ParsingNeuralNetwork).build(cam, MODEL)

    queues: dict[str, dai.DataOutputQueue] = {
        'rgb':   nn.passthrough.createOutputQueue(maxSize=1, blocking=False),
        'depth': nn.out.createOutputQueue(maxSize=1, blocking=False),
    }

    return queues, ['rgb', 'depth']
