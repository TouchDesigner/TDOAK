# pipeline.py — edit this to define your OAK pipeline
# Drop this text into the pipeline.py Text DAT inside the OAK Base COMP.
import depthai as dai


def build(pipeline):
    """
    Define your DepthAI node graph here.
    Return a list of XLinkOut stream names — these tell the extension
    which device queues to open on the host side.
    """

    # ── Color camera ─────────────────────────────────────────────────────
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    # ── Stereo depth ─────────────────────────────────────────────────────
    left  = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetType.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    # ── Optional: neural network ─────────────────────────────────────────
    # nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    # nn.setConfidenceThreshold(0.5)
    # nn.setBlobPath('/path/to/model.blob')
    # cam.preview.link(nn.input)

    # ── XLinkOut nodes ────────────────────────────────────────────────────
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName('rgb')
    cam.video.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName('depth')
    stereo.depth.link(xout_depth.input)

    return ['rgb', 'depth']
