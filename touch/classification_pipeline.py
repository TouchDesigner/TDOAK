from __future__ import annotations

# classification_pipeline.py — image classification (EfficientNet-Lite0, depthai v3)
# Load into the Pipeline DAT parameter of the OAK component.
#
# REQUIRES the `depthai_nodes` package (provides ParsingNeuralNetwork; its
# ClassificationParser emits Classifications messages with class-name strings from the
# model archive — no label table needed on the host).
#
# Streams:
#   'rgb'     — BGR passthrough: the exact 224×224 frame inference ran on.
#   'classes' — a depthai-nodes Classifications message. The host's built-in decoder
#               turns it into ClassificationEntry dicts [{label: str, score: float}, …],
#               sorted by score. touch/classification_handler.py is a fill-only Handler
#               that writes the top entries into a Table DAT — render the text however
#               you like from there (the data stays data).

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import depthai as dai

# ImageNet-1k EfficientNet-Lite0 (RVC2 build exists).
MODEL: str = 'luxonis/efficientnet-lite:lite0-224x224'

# Declared so the host pre-fetches the model on a subprocess before the build runs.
MODELS: list[str] = [MODEL]


def build(pipeline: dai.Pipeline) -> tuple[dict[str, dai.DataOutputQueue], list[str]]:
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork

    cam = pipeline.create(dai.node.Camera).build()
    nn  = pipeline.create(ParsingNeuralNetwork).build(cam, MODEL)

    queues: dict[str, dai.DataOutputQueue] = {
        'rgb':     nn.passthrough.createOutputQueue(maxSize=1, blocking=False),
        'classes': nn.out.createOutputQueue(maxSize=1, blocking=False),
    }

    return queues, ['rgb', 'classes']
