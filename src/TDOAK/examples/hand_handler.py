# hand_handler.py — Handler DAT for the 'hands' stream of hand_pipeline.py
# Point a Streams block's Handler parameter at this DAT, bind its Scriptop to a Script SOP,
# and add the onCook callback below to that SOP. `dai` and `np` are already in scope.
#
#   parse(msg)             BACKGROUND THREAD — depthai GatheredData -> plain per-hand data.
#   fill(scriptOp, hands)  MAIN THREAD      — draws each hand's landmark skeleton into a Script SOP.
#
# Script SOP cook callback (paste into the bound SOP's Callbacks DAT):
#   def onCook(scriptOp):
#       op('OAK').ext.OAKManagerExt.FillScriptOp(scriptOp)

import math

from TDOAK.core.constants import HAND_CROP_PADDING

# Must match the padding hand_pipeline.py bakes into its on-device crop Script — the
# back-map here undoes exactly that padding. Shared via TDOAK.core.constants so it can't drift.
CROP_PADDING = HAND_CROP_PADDING

# Standard MediaPipe 21-keypoint hand connections — used only if the model archive
# doesn't ship its own skeleton edges (Keypoints.getEdges()).
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


def _keypoints_of(item):
    """The gathered item is a Keypoints message, or a MessageGroup wrapping one (when the
    landmarker exposes multiple heads). Return whichever carries getPoints2f()."""
    if hasattr(item, 'getPoints2f'):
        return item
    try:
        for _key, member in item:               # MessageGroup is iterable as (name, msg)
            if hasattr(member, 'getPoints2f'):
                return member
    except TypeError:
        pass
    return None


def parse(msg) -> list[dict]:
    """GatheredData -> list of hands. Each hand's landmarks come back normalized within its
    (rotated, padded) crop; map them onto the full frame via the palm's RotatedRect so they
    overlay the 'rgb' stream directly. items[i] corresponds to detections[i] (crop order) —
    only when nothing was dropped, which the length check below guarantees."""
    if not hasattr(msg, 'reference_data'):
        raise TypeError(
            f"hand_handler.parse received a {type(msg).__name__}, expected GatheredData. "
            "Bind this Handler to the 'hands' stream block, not an image stream.")
    detections = msg.reference_data.detections
    items = msg.items
    if len(items) != len(detections):
        # A crop/inference was dropped for one hand this frame — pairing by index would
        # put a skeleton on the wrong hand, so skip the whole frame instead.
        return []
    hands = []
    for det, item in zip(detections, items):
        kp = _keypoints_of(item)
        if kp is None:
            continue

        rect = det.getBoundingBox()                      # RotatedRect, normalized
        cx, cy = rect.center.x, rect.center.y
        w = rect.size.width + 2 * CROP_PADDING
        h = rect.size.height + 2 * CROP_PADDING
        ang = math.radians(det.getAngle())
        ca, sa = math.cos(ang), math.sin(ang)

        # Crop-local, centered coords -> rotate into frame space (vectorized):
        # fx = cx + lx*ca - ly*sa,  fy = cy + lx*sa + ly*ca
        pts = np.asarray([[p.x, p.y] for p in kp.getPoints2f()], dtype=np.float32)
        local = (pts - 0.5) * np.array([w, h], dtype=np.float32)
        framePts = local @ np.array([[ca, sa], [-sa, ca]], dtype=np.float32) \
            + np.array([cx, cy], dtype=np.float32)

        raw = kp.getEdges() if hasattr(kp, 'getEdges') else []
        edges = [[int(a), int(b)] for a, b in raw] or [list(e) for e in HAND_EDGES]
        hands.append({
            'score':     round(float(det.confidence), 3),
            'keypoints': framePts.tolist(),
            'edges':     edges,
        })
    return hands


def fill(scriptOp, hands) -> None:
    """MAIN THREAD — full TD access, runs inside the operator's cook.
    hands: the list of per-hand dicts produced by parse() above."""
    stagingData = scriptOp.createGeometryStagingData()

    if not hands:
        scriptOp.buildGeometry(stagingData)
        return

    stagingData.createPointAttrib(name='P')
    # Stack every hand's keypoints as samples -> (sum of per-hand counts, 2)
    kp = np.concatenate(
        [np.asarray(hand['keypoints'], dtype=np.float32) for hand in hands],
        axis=0,
    )
    kp = np.pad(kp, ((0, 0), (0, 1)))   # (N, 2) -> (N, 3), z = 0
    kp[:, 1] = 1.0 - kp[:, 1]           # y-flip: rgb is flipped for TD display
    stagingData.appendPointAttribValues('P', kp.tolist())

    # Edge indices are per-hand — offset each hand's edges by the number of points
    # appended before it. Derived from the actual keypoint counts, so a model with a
    # different landmark count per hand still connects the right points.
    offset = 0
    for hand in hands:
        for a, b in hand['edges']:
            stagingData.appendPrimLine(a + offset, b + offset)
        offset += len(hand['keypoints'])

    scriptOp.buildGeometry(stagingData)
