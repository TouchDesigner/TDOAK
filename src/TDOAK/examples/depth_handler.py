# depth_handler.py — Handler DAT for a 'depth' (or disparity) stream shown in a Script TOP.
# `np` is already in scope.
#
# Why: the built-in default pushes depth as it arrives — a 16-bit integer image in
# millimetres. A TOP viewer maps 16-bit integers 0..65535 to black..white, so a room that
# is 0.3-4 m deep (300..4000) looks almost black. The data is fine; only the display is
# dark. This handler turns it into a 32-bit float image you can read and see.
#
# fill-only: the built-in depth decode (uint16 mm, TD-oriented) stays. Add a Streams
# block: Stream 'depth', Scriptop -> a Script TOP, Handler -> this DAT.
#
# Output (single channel, 32-bit float):
#   DISPLAY_RANGE_M = None        -> metres. Exact values for further processing; anything
#                                    beyond 1 m shows white in a viewer (normalize with a
#                                    Math TOP / Level TOP downstream).
#   DISPLAY_RANGE_M = (near, far) -> 0..1 across that range, near = white, far = black,
#                                    no data = 0. Ready to look at.
#
# For a DISPARITY stream set UNIT_SCALE = 1.0 (its values are pixels, not millimetres) and
# DISPLAY_RANGE_M to the disparity range you expect, e.g. (0, 95).

UNIT_SCALE      = 0.001        # stream units -> metres (uint16 millimetres)
DISPLAY_RANGE_M = (0.3, 4.0)   # or None for plain metres


def fill(scriptOp, depth) -> None:
    """MAIN THREAD — depth: (H, W) array, already flipped to TD's bottom-up orientation."""
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        return  # block bound to a non-depth stream — ignore rather than error
    metres = depth.astype(np.float32) * UNIT_SCALE
    if DISPLAY_RANGE_M is not None:
        near, far = DISPLAY_RANGE_M
        shown = 1.0 - (metres - near) / max(far - near, 1e-6)
        metres = np.where(depth > 0, np.clip(shown, 0.0, 1.0), 0.0).astype(np.float32)
    h, w = metres.shape
    scriptOp.copyNumpyArray(metres.reshape(h, w, 1))
