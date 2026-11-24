"""Built-in decoders: depthai message -> plain data (pure Python + numpy, worker thread).

Everything here takes already-extracted values or duck-typed message objects, never a TD
object, so it runs on the camera worker and under plain pytest alike.
"""
from __future__ import annotations

import math

import numpy as np


def toTdImage(frame: np.ndarray) -> np.ndarray:
    """An OpenCV-style frame (top-down rows, BGR/BGRA colour order — what depthai's
    getCvFrame() returns) as a contiguous array in TouchDesigner's convention: bottom-up
    rows and RGB/RGBA order. Single-channel frames (mono, depth, disparity) are only
    flipped. Decided by the frame's shape, never by a stream name."""
    out = np.flipud(frame)
    if out.ndim == 3 and out.shape[2] == 3:
        out = out[..., ::-1]
    elif out.ndim == 3 and out.shape[2] == 4:
        out = out[..., [2, 1, 0, 3]]
    return np.ascontiguousarray(out)


def organizePointCloud(points: np.ndarray, width: int, height: int,
                       fallbackHW: tuple[int, int] | None = None) -> np.ndarray:
    """(N, 3) float32 points — or, when they form a complete sensor grid, an organized
    (H, W, 3) array flipped to TD's bottom-up origin so it overlays the depth image.
    getPoints() returns (N, 3) or a flat (N*3,) array depending on the depthai build;
    width/height can be unreliable, so the companion image's (H, W) is the fallback."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    w, h = int(width or 0), int(height or 0)
    if w * h != pts.shape[0] and fallbackHW is not None and fallbackHW[0] * fallbackHW[1] == pts.shape[0]:
        h, w = fallbackHW
    if w > 0 and h > 0 and w * h == pts.shape[0]:
        pts = np.flipud(pts.reshape(h, w, 3))
    return np.ascontiguousarray(pts)


def rotatedRectBox(rr: object) -> tuple[list[float], float]:
    """Axis-aligned [xmin, ymin, xmax, ymax] outer bounds (normalized) + rotation in degrees
    for a depthai RotatedRect. Prefers getPoints(); falls back to center/size trigonometry."""
    angle = float(getattr(rr, 'angle', 0.0))
    getPoints = getattr(rr, 'getPoints', None)
    if callable(getPoints):
        pts = getPoints()
        xs = [float(p.x) for p in pts]
        ys = [float(p.y) for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)], angle
    cx, cy = float(rr.center.x), float(rr.center.y)                    # type: ignore[attr-defined]
    hw, hh = float(rr.size.width) / 2.0, float(rr.size.height) / 2.0   # type: ignore[attr-defined]
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    corners = ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    xs = [cx + dx * ca - dy * sa for dx, dy in corners]
    ys = [cy + dx * sa + dy * ca for dx, dy in corners]
    return [min(xs), min(ys), max(xs), max(ys)], angle


def _keypoints(det: object) -> list:
    """A detection's 2D keypoints, or [] — plain and spatial detections of older depthai
    builds have no getKeypoints2f at all."""
    getter = getattr(det, 'getKeypoints2f', None)
    if not callable(getter):
        return []
    try:
        return list(getter() or [])
    except Exception:
        return []


def _spatial(det: object) -> dict[str, float]:
    """{'x', 'y', 'z'} in millimetres for a spatial detection, {} for a plain one."""
    coords = getattr(det, 'spatialCoordinates', None)
    if coords is None:
        return {}
    return {'x': float(coords.x), 'y': float(coords.y), 'z': float(coords.z)}


def decodeDetections(detections: list) -> list[dict]:
    """dai.ImgDetections / SpatialImgDetections .detections -> list of plain dicts.

    Detections that carry keypoints (pose models) become PoseEntry dicts — confidence, box,
    keypoints, edges; all others become DetectionEntry dicts — label, confidence, xmin..ymax.
    Spatial detections additionally carry x, y, z (millimetres, camera space)."""
    dets = list(detections or [])
    if not dets:
        return []
    if _keypoints(dets[0]):
        getEdges = getattr(dets[0], 'getEdges', None)
        edges = [list(e) for e in getEdges()] if callable(getEdges) else []
        out = []
        for d in dets:
            entry = {
                'confidence': round(float(d.confidence), 3),
                'box':        [d.xmin, d.ymin, d.xmax, d.ymax],
                'keypoints':  [[p.x, p.y] for p in _keypoints(d)],
                'edges':      edges,
            }
            entry.update(_spatial(d))
            out.append(entry)
        return out
    out = []
    for d in dets:
        entry = {
            'label':      d.label,
            'confidence': round(float(d.confidence), 3),
            'xmin':       d.xmin,
            'ymin':       d.ymin,
            'xmax':       d.xmax,
            'ymax':       d.ymax,
        }
        entry.update(_spatial(d))
        out.append(entry)
    return out


def _seconds(reading: object) -> float:
    """Device timestamp of an IMU reading in seconds (0.0 when it has none)."""
    for name in ('getTimestampDevice', 'getTimestamp'):
        getter = getattr(reading, name, None)
        if callable(getter):
            try:
                return float(getter().total_seconds())
            except Exception:
                continue
    return 0.0


def decodeImu(packets: list) -> dict | None:
    """dai.IMUData .packets -> {'accel', 'gyro', 'samples'}, or None for an empty message.

    accel / gyro are the LATEST reading ([x, y, z] in m/s^2 and rad/s). samples holds
    EVERY packet of the batch in arrival order as [ax, ay, az, gx, gy, gz, t] (t = device
    timestamp in seconds): the IMU reports far faster than TouchDesigner cooks, so the
    latest value alone drops most of the data."""
    packets = list(packets or [])
    if not packets:
        return None
    samples = []
    for p in packets:
        acc = p.acceleroMeter if hasattr(p, 'acceleroMeter') else p.accelerometer
        gyro = p.gyroscope
        samples.append([acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z, _seconds(acc)])
    last = samples[-1]
    return {'accel': last[0:3], 'gyro': last[3:6], 'samples': samples}


def fitResolution(wanted: tuple[int, int], sensor: tuple[int, int]) -> tuple[int, int]:
    """The requested output size, clamped to what the sensor can deliver (a 1280x800
    sensor can't serve 1920x1080). An unknown sensor size (0) leaves the request alone."""
    sw, sh = int(sensor[0] or 0), int(sensor[1] or 0)
    if sw <= 0 or sh <= 0:
        return (int(wanted[0]), int(wanted[1]))
    return (min(int(wanted[0]), sw), min(int(wanted[1]), sh))
