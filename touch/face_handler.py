# face_handler.py — Handler DAT for the 'faces' stream of face_pipeline.py. `np` in scope.
#
# fill-only: the built-in decode (PoseEntry dicts — see face_pipeline.py) stays, this
# just draws it. Bind the Streams block's Scriptop to a Script POP — POPs are pushed
# directly on each new frame, no cook callback needed.
#
# Output per face: 5 landmark points + the detection box as a 4-line outline.
# All coords normalized 0-1 on the image plane (z = 0), y-flipped to overlay the
# TD-oriented 'rgb' stream 1:1.


def fill(scriptOp, faces) -> None:
    """MAIN THREAD — faces: list of PoseEntry dicts (see face_pipeline.py)."""
    stagingData = scriptOp.createGeometryStagingData()
    if not isinstance(faces, list) or not faces:
        scriptOp.buildGeometry(stagingData)
        return

    stagingData.createPointAttrib(name='P')
    pts:   list[list[float]]     = []
    lines: list[tuple[int, int]] = []

    for face in faces:
        # 5 landmarks (detection coords are top-down; TD display is bottom-up -> y-flip)
        for x, y in face['keypoints']:
            pts.append([x, 1.0 - y, 0.0])
        # box outline: 4 corners, 4 edges
        x0, y0, x1, y1 = face['box']
        base = len(pts)
        pts += [[x0, 1.0 - y0, 0.0], [x1, 1.0 - y0, 0.0],
                [x1, 1.0 - y1, 0.0], [x0, 1.0 - y1, 0.0]]
        lines += [(base + i, base + (i + 1) % 4) for i in range(4)]

    stagingData.appendPointAttribValues('P', pts)
    for a, b in lines:
        stagingData.appendPrimLine(a, b)
    scriptOp.buildGeometry(stagingData)
