from datetime import timedelta
from types import SimpleNamespace as NS

import numpy as np

from TDOAK.core import decode


def test_td_image_colour_is_flipped_and_rgb():
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[0, 0] = (10, 20, 30)          # top-left pixel, B G R
    out = decode.toTdImage(bgr)
    assert out.shape == (2, 3, 3) and out.flags['C_CONTIGUOUS']
    assert tuple(out[1, 0]) == (30, 20, 10)   # now bottom row, R G B


def test_td_image_bgra_keeps_alpha_last():
    bgra = np.zeros((1, 1, 4), dtype=np.uint8)
    bgra[0, 0] = (1, 2, 3, 4)
    assert tuple(decode.toTdImage(bgra)[0, 0]) == (3, 2, 1, 4)


def test_td_image_mono_only_flips():
    depth = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    out = decode.toTdImage(depth)
    assert out.dtype == np.uint16 and out.tolist() == [[3, 4], [1, 2]]


def test_point_cloud_flat_and_organized():
    flat = np.arange(18, dtype=np.float64)           # 6 points, flat
    pts = decode.organizePointCloud(flat, 3, 2)
    assert pts.shape == (2, 3, 3) and pts.dtype == np.float32
    assert pts[1, 0].tolist() == [0, 1, 2]           # first sensor row is now the bottom row
    assert decode.organizePointCloud(flat, 0, 0).shape == (6, 3)
    assert decode.organizePointCloud(flat, 0, 0, fallbackHW=(2, 3)).shape == (2, 3, 3)
    assert decode.organizePointCloud(flat, 5, 5).shape == (6, 3)   # bogus size: stay unorganized


def test_rotated_rect_from_points_and_trig():
    rr = NS(angle=0.0, getPoints=lambda: [NS(x=.1, y=.2), NS(x=.5, y=.2), NS(x=.5, y=.6), NS(x=.1, y=.6)])
    assert decode.rotatedRectBox(rr) == ([.1, .2, .5, .6], 0.0)
    rr = NS(angle=90.0, center=NS(x=.5, y=.5), size=NS(width=.4, height=.2))
    box, angle = decode.rotatedRectBox(rr)
    assert angle == 90.0
    assert np.allclose(box, [.4, .3, .6, .7])


def det(**kw):
    base = dict(label=1, confidence=0.87654, xmin=.1, ymin=.2, xmax=.3, ymax=.4)
    base.update(kw)
    return NS(**base)


def test_plain_detections():
    out = decode.decodeDetections([det(getKeypoints2f=lambda: [])])
    assert out == [{'label': 1, 'confidence': 0.877, 'xmin': .1, 'ymin': .2, 'xmax': .3, 'ymax': .4}]


def test_detections_without_keypoint_api():
    assert decode.decodeDetections([det()])[0]['label'] == 1     # older builds: no getKeypoints2f


def test_spatial_detections_carry_xyz():
    out = decode.decodeDetections([det(spatialCoordinates=NS(x=10.0, y=-20.0, z=1500.0))])
    assert (out[0]['x'], out[0]['y'], out[0]['z']) == (10.0, -20.0, 1500.0)


def test_pose_detections():
    kp = lambda: [NS(x=.5, y=.5), NS(x=.6, y=.7)]
    out = decode.decodeDetections([det(getKeypoints2f=kp, getEdges=lambda: [(0, 1)])])
    assert out == [{'confidence': 0.877, 'box': [.1, .2, .3, .4],
                    'keypoints': [[.5, .5], [.6, .7]], 'edges': [[0, 1]]}]


def test_empty_detections():
    assert decode.decodeDetections([]) == [] and decode.decodeDetections(None) == []


def packet(a, g, t):
    stamp = lambda: timedelta(seconds=t)
    return NS(acceleroMeter=NS(x=a, y=a + 1, z=a + 2, getTimestampDevice=stamp),
              gyroscope=NS(x=g, y=g + 1, z=g + 2))


def test_imu_keeps_every_sample():
    out = decode.decodeImu([packet(1, 10, 0.5), packet(2, 20, 0.502)])
    assert out['accel'] == [2, 3, 4] and out['gyro'] == [20, 21, 22]
    assert out['samples'] == [[1, 2, 3, 10, 11, 12, 0.5], [2, 3, 4, 20, 21, 22, 0.502]]


def test_imu_empty_and_alternate_spelling():
    assert decode.decodeImu([]) is None
    p = NS(accelerometer=NS(x=1, y=2, z=3), gyroscope=NS(x=4, y=5, z=6))
    assert decode.decodeImu([p])['samples'] == [[1, 2, 3, 4, 5, 6, 0.0]]


def test_fit_resolution():
    assert decode.fitResolution((1920, 1080), (4208, 3120)) == (1920, 1080)
    assert decode.fitResolution((1920, 1080), (1280, 800)) == (1280, 800)
    assert decode.fitResolution((1920, 1080), (0, 0)) == (1920, 1080)
