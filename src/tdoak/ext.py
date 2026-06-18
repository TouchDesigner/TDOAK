from __future__ import annotations

import math
import threading
import time
from typing import TYPE_CHECKING, Callable, Protocol, TypedDict

import numpy as np

if TYPE_CHECKING:
    from td import COMP, OP, Par, TOP, scriptTOP


class DeviceInfo(TypedDict):
    mxid:     str
    name:     str
    state:    str
    protocol: str


class DetectionEntry(TypedDict):
    label:      int
    confidence: float
    xmin:       float
    ymin:       float
    xmax:       float
    ymax:       float


class PoseEntry(TypedDict):
    confidence: float
    box:        list[float]        # [xmin, ymin, xmax, ymax], normalized 0-1
    keypoints:  list[list[float]]  # COCO-17 joints [[x, y], ...], normalized 0-1
    edges:      list[list[int]]    # skeleton bone pairs [[i, j], ...] indexing keypoints


class ExtendedDetectionEntry(TypedDict):
    label:      int
    confidence: float
    box:        list[float]        # axis-aligned [xmin, ymin, xmax, ymax], normalized 0-1
    angle:      float              # rotated_rect rotation in degrees (0 for upright boxes)
    keypoints:  list[list[float]]  # [[x, y, score], ...], normalized 0-1; may be empty


class ClassificationEntry(TypedDict):
    label: str
    score: float


class ImuData(TypedDict):
    accel: list[float]
    gyro:  list[float]


class StatusInfo(TypedDict):
    connected:  bool
    fps:        float
    latency_ms: float
    error:      str
    streams:    list[str]


class PipelineConfig(TypedDict):
    device:         str
    fps:            int
    enableDepth:    bool
    enableImu:      bool
    pipelineScript: str | None
    parsers:        dict[str, ParseFn]


# Per-stream handler hooks (defined in a Streams-block Handler DAT):
#   parse(msg)            — WORKER thread, NO TD access. depthai message -> plain data.
#   fill(scriptOp, data)  — MAIN thread, full TD access. plain data -> Script Operator.
ParseFn = Callable[[object], object]
FillFn  = Callable[['OP', object], None]


class _OutputQueue(Protocol):
    def tryGet(self) -> object: ...


class _InputQueue(Protocol):
    def send(self, msg: object) -> None: ...


class _Pipeline(Protocol):
    def start(self)     -> None: ...
    def stop(self)      -> None: ...
    def isRunning(self) -> bool: ...


class _Device(Protocol):
    def close(self) -> None: ...


FrameData = (np.ndarray | list[DetectionEntry] | list[PoseEntry]
             | list[ExtendedDetectionEntry] | list[ClassificationEntry] | ImuData
             | list[list[float]] | list[list[list[float]]])

# Sentinel: decodeNodeMessage returns this when no depthai-nodes type matched, so the
# caller falls through to the built-in (raw depthai) decode chain.
_UNHANDLED: object = object()

# Cached depthai-nodes message classes (or None if depthai-nodes isn't installed). depthai-nodes
# is optional at runtime, so the import is deferred and memoized — never imported at module load.
_NODE_TYPES: dict[str, type] | None = None


def _nodeMessageTypes() -> dict[str, type]:
    """Return {classname: class} for the depthai-nodes message types we decode, or {} if the
    package isn't importable. Imported once on the worker thread; cached for the process."""
    global _NODE_TYPES
    if _NODE_TYPES is None:
        names = ('ImgDetectionsExtended', 'Keypoints', 'SegmentationMask',
                 'Classifications', 'Lines', 'Clusters')
        mod = None
        for path in ('depthai_nodes.message', 'depthai_nodes'):
            try:
                mod = __import__(path, fromlist=['*'])
                break
            except Exception:
                continue
        # Map only the classes this build actually exposes; a missing one just skips its
        # decoder rather than disabling all of them.
        _NODE_TYPES = {n: getattr(mod, n) for n in names if hasattr(mod, n)} if mod else {}
    return _NODE_TYPES


def _rotatedRectBox(rr: object) -> tuple[list[float], float]:
    """Axis-aligned [xmin, ymin, xmax, ymax] outer bounds (normalized) + rotation in degrees
    for a depthai RotatedRect. Prefers getPoints(); falls back to center/size trigonometry."""
    angle = float(getattr(rr, 'angle', 0.0))
    getPoints = getattr(rr, 'getPoints', None)
    if callable(getPoints):
        pts = getPoints()
        xs  = [float(p.x) for p in pts]
        ys  = [float(p.y) for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)], angle
    cx, cy = float(rr.center.x), float(rr.center.y)        # type: ignore[attr-defined]
    hw, hh = float(rr.size.width) / 2.0, float(rr.size.height) / 2.0  # type: ignore[attr-defined]
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    xs = [cx + dx * ca - dy * sa for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
    ys = [cy + dx * sa + dy * ca for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
    return [min(xs), min(ys), max(xs), max(ys)], angle


class OAKManagerExt:

    def __init__(self, ownerComp: COMP) -> None:
        self.ownerComp: COMP = ownerComp
        self.ThreadManager: COMP = op.TDResources.ThreadManager

        self._lock:          threading.Lock       = threading.Lock()
        self._frames:        dict[str, FrameData] = {}
        self._connected:     bool                 = False
        self._fps:           float                = 0.0
        self._latencyMs:     float                = 0.0
        self._lastError:     str                  = ''
        self._activeStreams: list[str]            = []

        self._reconnectDelay:    float                                      = 10.0
        self._retryCount:        int                                        = 0
        self._stopFlag:          threading.Event                            = threading.Event()
        self._datVersion:        int                                        = -1
        self._latestDevices:     list[DeviceInfo]                          = []
        self._onDevicesCallback: Callable[[list[DeviceInfo]], None] | None = None
        self._activePipeline:    _Pipeline | None                           = None
        self._activeDevice:      _Device | None                             = None
        self._intrinsics:        dict[str, dict[str, float]]               = {}
        self._controlQueues:     dict[str, _InputQueue]                    = {}
        self._fillers:           dict[str, FillFn]                         = {}  # keyed by fillerId '{stream}#{i}'
        self._frameSeq:          int                                       = 0

        self._scriptopRegistry: dict[str, list[OP]] = {}
        self._lastStreamNames:  list[str]            = []
        self._lastImageHW:      tuple[int, int] | None = None  # (h, w) of last image frame, for organizing point clouds

        # reset parameters to known state
        self.ownerComp.par.Connected = 0
        self.ownerComp.par.Fps = 0
        self.ownerComp.par.Activestreams = ''
        self.ownerComp.par.Lasterror = ''
        self.ownerComp.par.Device.menuNames = ['None']
        self.ownerComp.par.Device.menuLabels = ['-- Select Device --']

    # ── Public API (main thread only) ────────────────────────────────────

    def Start(self) -> None:
        self._retryCount    = 0
        self._reconnectDelay = 10.0
        self.doStart()

    def doStart(self) -> None:
        self._stopFlag.clear()
        config = self.readConfig()
        # Seed the Autoreload baseline to the DAT we're starting with. Otherwise _datVersion
        # stays at its -1 sentinel and the first onRefresh tick reads changeCount != -1 as an
        # edit, firing a spurious Rebuild() a moment after every Start.
        datRef = self.ownerComp.par.Pipelinedat.eval()
        dat    = op(datRef) if datRef else None
        self._datVersion = dat.changeCount if dat else -1
        # Load per-stream Handler DATs on the main thread: parsers ride along in config
        # for the worker, fillers stay here for the main-thread cook side.
        parsers, fillers = self.loadStreamHandlers()
        config['parsers'] = parsers
        self._fillers     = fillers
        task = self.ThreadManager.TDTask(
            target=self.cameraPump,
            args=(config, self._stopFlag),
            SuccessHook=self.onSuccess,
            ExceptHook=self.onExcept,
            RefreshHook=self.onRefresh,
        )
        self.ThreadManager.EnqueueTask(task, standalone=True)

    def Stop(self) -> None:
        self._stopFlag.set()

    def Rebuild(self) -> None:
        self.Stop()
        run('args[0].Start()', self, delayFrames=5, fromOP=self.ownerComp)

    def ListDevices(self, callback: Callable[[list[DeviceInfo]], None] | None = None) -> None:
        self._onDevicesCallback = callback
        task = self.ThreadManager.TDTask(
            target=self.enumerateDevices,
            SuccessHook=self.onEnumerateSuccess,
            ExceptHook=self.onEnumerateExcept,
        )
        self.ThreadManager.EnqueueTask(task)

    def GetDevices(self) -> list[DeviceInfo]:
        with self._lock: return list(self._latestDevices)

    def OnPulseStart(self, par: Par) -> None:       self.Start()
    def OnPulseStop(self, par: Par) -> None:        self.Stop()
    def OnPulseRebuild(self, par: Par) -> None:     self.Rebuild()
    def OnPulseListdevices(self, par: Par) -> None: self.ListDevices()

    def RegisterScriptop(self, streamName: str, scriptop: OP) -> None:
        """Bind a Script TOP or Script CHOP to receive data for streamName.
        Call this at runtime, e.g. from a project-level script or Execute DAT."""
        if streamName not in self._scriptopRegistry:
            self._scriptopRegistry[streamName] = []
        if scriptop not in self._scriptopRegistry[streamName]:
            self._scriptopRegistry[streamName].append(scriptop)

    def UnregisterScriptop(self, streamName: str, scriptop: OP | None = None) -> None:
        """Remove a registered scriptop. Pass scriptop=None to remove all for the stream."""
        if scriptop is None:
            self._scriptopRegistry.pop(streamName, None)
        elif streamName in self._scriptopRegistry:
            try:
                self._scriptopRegistry[streamName].remove(scriptop)
            except ValueError:
                pass

    def bindingForOp(self, scriptop: OP) -> tuple[str, str | None] | None:
        """Resolve a Script Operator's binding: (streamName, fillerId).
        Streams blocks carry a per-block fillerId ({stream}#{i}); runtime-registered
        scriptops have no block, so fillerId is None. Returns None if not bound."""
        for streamName, ops in self._scriptopRegistry.items():
            if scriptop in ops:
                return streamName, None
        seq = getattr(self.ownerComp.seq, 'Streams', None)
        if seq is not None:
            for i in range(len(seq)):
                regOp = seq.blockPars.Scriptop[i].eval()
                if regOp is not None and regOp.path == scriptop.path:
                    streamName = seq.blockPars.Stream[i].eval()
                    return streamName, self.fillerId(streamName, i)
        return None

    def GetFrameForScriptop(self, scriptop: OP) -> FrameData | None:
        """Return the frame data for this scriptop via registry or sequential block.
        Call from a Script CHOP cook callback to pull data."""
        binding = self.bindingForOp(scriptop)
        return self.GetFrame(binding[0]) if binding is not None else None

    def FillScriptOp(self, scriptop: OP) -> bool:
        """Pull entry point for Script SOPs (which can't be written outside their cook):
        call this from the SOP's onCook to run the block's fill(scriptOp, data) inside it.
        TOPs/CHOPs/DATs are auto-pushed by autoPush and don't need this, though calling it
        is harmless. Falls back to copyNumpyArray for a Script TOP. Returns True if filled."""
        binding = self.bindingForOp(scriptop)
        if binding is None:
            return False
        name, fillerId = binding
        data = self.GetFrame(name)
        if data is None:
            return False
        filler = self._fillers.get(fillerId) if fillerId is not None else None
        if filler is not None:
            filler(scriptop, data)
            return True
        if isinstance(scriptop, scriptTOP) and isinstance(data, np.ndarray):
            scriptop.copyNumpyArray(self.prepareTopFrame(data))
            return True
        return False

    # ── Getters (safe from Script TOP/CHOP cook callbacks) ───────────────

    def GetFrame(self, name: str) -> FrameData | None:
        with self._lock: return self._frames.get(name)

    def GetFrameNames(self) -> list[str]:
        with self._lock: return list(self._frames.keys())

    def GetRgb(self)     -> FrameData | None: return self.GetFrame('rgb')
    def GetDepth(self)   -> FrameData | None: return self.GetFrame('depth')
    def GetIrLeft(self)  -> FrameData | None: return self.GetFrame('ir_left')
    def GetIrRight(self) -> FrameData | None: return self.GetFrame('ir_right')
    def GetImu(self)     -> FrameData | None: return self.GetFrame('imu')
    def GetPose(self)    -> FrameData | None: return self.GetFrame('pose')

    def GetIntrinsics(self, socket: str = 'CAM_A') -> dict[str, float] | None:
        """Return {'fx', 'fy', 'cx', 'cy'} for the given socket at native resolution, or None if not available."""
        with self._lock:
            entry = self._intrinsics.get(socket)
            return dict(entry) if entry is not None else None

    def SendControl(self, name: str, message: object) -> bool:
        """Send a control/config message to a named device input queue.
        Safe to call from TD's main thread. Returns False if the queue is not registered."""
        with self._lock:
            q = self._controlQueues.get(name)
        if q is None:
            return False
        try:
            q.send(message)
            return True
        except Exception:
            return False

    def SendFrame(self, name: str, source: TOP | np.ndarray,
                  imgType: object | None = None) -> bool:
        """Send a host image into a named device input queue, so the device runs a pipeline
        on TD-supplied frames instead of an OAK camera (see touch/mono_depth_input_pipeline.py).
        `source` is any TOP (read via .numpyArray()) or an (H, W, 3|4) numpy array. Call from
        the main thread — it reads the TOP. Returns False if the queue is not registered or
        the frame is unusable. Typically called once per frame from an Execute/CHOP callback."""
        import depthai as dai
        with self._lock:
            q = self._controlQueues.get(name)
        if q is None:
            return False
        bgr = self.toBgrUint8(source)
        if bgr is None:
            return False
        frame = dai.ImgFrame()
        frame.setCvFrame(bgr, imgType or dai.ImgFrame.Type.BGR888i)  # type: ignore[arg-type]
        frame.setTimestamp(dai.Clock.now())
        frame.setSequenceNum(self._frameSeq)
        self._frameSeq += 1
        try:
            q.send(frame)
            return True
        except Exception:
            return False

    def toBgrUint8(self, source: TOP | np.ndarray) -> np.ndarray | None:
        """Coerce a TOP texture or numpy array to a contiguous (H, W, 3) uint8 BGR frame,
        top-down (TOPs are bottom-up). Float inputs are assumed normalized 0..1, which is
        TD's TOP.numpyArray() default; RGB(A) channel order is reversed to BGR."""
        if isinstance(source, np.ndarray):
            arr = source
        else:
            try:
                arr = source.numpyArray()  # (H, W, 4) float32 0..1, origin bottom-left
            except Exception:
                return None
        if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
            return None
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        bgr = np.flipud(arr[:, :, 2::-1])   # RGB(A) -> BGR, flip to top-down
        return np.ascontiguousarray(bgr)

    def GetControlNames(self) -> list[str]:
        """Return the list of registered control queue names for the active pipeline."""
        with self._lock:
            return list(self._controlQueues.keys())

    def GetStatus(self) -> StatusInfo:
        with self._lock:
            return StatusInfo(
                connected=  self._connected,
                fps=        self._fps,
                latency_ms= self._latencyMs,
                error=      self._lastError,
                streams=    list(self._activeStreams),
            )

    # ── Internal push methods (background thread → shared state) ─────────

    def pushFrame(self, name: str, data: FrameData) -> None:
        with self._lock: self._frames[name] = data

    def pushDevices(self, devices: list[DeviceInfo]) -> None:
        with self._lock: self._latestDevices = devices

    def pushStatus(self, connected: bool, fps: float = 0.0, latency: float = 0.0,
                    error: str = '', streams: list[str] | None = None) -> None:
        with self._lock:
            self._connected = connected
            self._fps       = fps
            self._latencyMs = latency
            self._lastError = error
            if streams is not None:
                self._activeStreams = streams

    # ── Background thread — NO TD object access below this line ──────────

    def enumerateDevices(self) -> None:
        import depthai as dai
        found: list[DeviceInfo] = []
        for info in dai.Device.getAllAvailableDevices():
            found.append(DeviceInfo(
                mxid=     info.deviceId,
                name=     info.name,
                state=    str(info.state).split('X_LINK_')[-1],
                protocol= str(info.protocol).split('X_LINK_')[-1],
            ))
        self.pushDevices(found)

    def onEnumerateSuccess(self) -> None:
        devices = self.GetDevices()
        names:  list[str] = [d['mxid']                                      for d in devices]
        labels: list[str] = [f"{d['name'] or d['mxid']}  [{d['protocol']}]" for d in devices]
        names.insert(0, 'None')
        labels.insert(0, '-- Select Device --')
        self.ownerComp.par.Device.menuNames  = names  or [' ']
        self.ownerComp.par.Device.menuLabels = labels or ['—']
        if self._onDevicesCallback:
            self._onDevicesCallback(devices)

    def onEnumerateExcept(self, *args: object) -> None:
        errorMsg: str = str(args[0]) if args else 'Unknown error'
        self.pushDevices([])
        self.ownerComp.par.Lasterror = errorMsg

    def cameraPump(self, config: PipelineConfig, stopFlag: threading.Event) -> None:
        # Worker entry point. Tear the device down HERE (worker thread) on every exit path:
        # device.close()/pipeline.stop() are blocking depthai calls, and the SuccessHook/
        # ExceptHook run on TD's MAIN thread — closing there freezes the UI for several
        # frames on each stop/retry. The finally keeps that cost off the main thread.
        try:
            self.runPipeline(config, stopFlag)
        finally:
            self.closeActiveDevice()

    def runPipeline(self, config: PipelineConfig, stopFlag: threading.Event) -> None:
        import depthai as dai

        if config['device']:
            self._activeDevice = dai.Device(dai.DeviceInfo(config['device']))
            pipeline = dai.Pipeline(self._activeDevice)
        else:
            self._activeDevice = None
            pipeline = dai.Pipeline()

        queues, streamNames, controlQueues = self.buildPipeline(pipeline, config)
        self.pushStatus(False, streams=streamNames)

        self._activePipeline = pipeline
        pipeline.start()
        self.cacheIntrinsics(pipeline.getDefaultDevice())  # type: ignore[attr-defined]
        with self._lock:
            self._controlQueues = controlQueues
        self.pushStatus(True, streams=streamNames)

        parsers: dict[str, ParseFn] = config['parsers']
        frameCount: int   = 0
        fpsTimer:   float = time.monotonic()
        tdThread          = threading.current_thread()

        while not stopFlag.is_set() and pipeline.isRunning():
            for name, q in queues.items():
                msg = q.tryGet()
                if msg is None:
                    continue

                hostTs:    float = time.monotonic()
                deviceTs:  float = (msg.getTimestampDevice().total_seconds()  # type: ignore[union-attr]
                                    if hasattr(msg, 'getTimestampDevice') else hostTs)
                latencyMs: float = (hostTs - deviceTs) * 1000.0

                # Custom per-stream decode wins: turn the (model-specific) message into
                # plain, thread-safe data. Anything not handled here falls through to the
                # built-in decoders below (the QoL path for rgb/depth/tof/detections/imu).
                parseFn = parsers.get(name)
                if parseFn is not None:
                    try:
                        data = parseFn(msg)
                    except Exception as exc:
                        with self._lock:
                            self._lastError = f'parse[{name}]: {exc}'
                        continue
                    if data is not None:
                        self.pushFrame(name, data)
                    continue

                # depthai-nodes typed messages decode first: ImgDetectionsExtended also exposes
                # .detections, so it must be caught here before the raw dai.ImgDetections branch.
                nodeData = self.decodeNodeMessage(msg)
                if nodeData is not _UNHANDLED:
                    if nodeData is not None:
                        self.pushFrame(name, nodeData)
                    continue

                if hasattr(msg, 'detections'):
                    dets = msg.detections  # type: ignore[union-attr]
                    # Pose models (YOLOv8-pose) carry per-detection keypoints; plain
                    # object detectors return empty keypoints. Route on that so one
                    # branch serves both without depending on the stream name.
                    if dets and dets[0].getKeypoints2f():
                        edges: list[list[int]] = [list(e) for e in dets[0].getEdges()]
                        self.pushFrame(name, [
                            PoseEntry(
                                confidence= round(d.confidence, 3),
                                box=        [d.xmin, d.ymin, d.xmax, d.ymax],
                                keypoints=  [[p.x, p.y] for p in d.getKeypoints2f()],
                                edges=      edges,
                            )
                            for d in dets
                        ])
                    else:
                        self.pushFrame(name, [
                            DetectionEntry(
                                label=      d.label,
                                confidence= round(d.confidence, 3),
                                xmin=       d.xmin,
                                ymin=       d.ymin,
                                xmax=       d.xmax,
                                ymax=       d.ymax,
                            )
                            for d in dets
                        ])

                elif hasattr(msg, 'packets'):
                    if msg.packets:  # type: ignore[union-attr]
                        p   = msg.packets[-1]  # type: ignore[union-attr]
                        acc = p.acceleroMeter if hasattr(p, 'acceleroMeter') else p.accelerometer
                        self.pushFrame(name, ImuData(
                            accel= [acc.x, acc.y, acc.z],
                            gyro=  [p.gyroscope.x, p.gyroscope.y, p.gyroscope.z],
                        ))

                elif hasattr(msg, 'getPoints'):
                    # getPoints() may return (N, 3) or a flat (N*3,) array depending on the
                    # depthai build (3.6.1 returns flat) — normalise to (N, 3) first.
                    pts: np.ndarray = np.asarray(msg.getPoints(),  # type: ignore[union-attr]
                                                 dtype=np.float32).reshape(-1, 3)
                    # Organized cloud → reshape to the sensor grid so it's TOP-ready
                    # and pixel-aligned with depth/amplitude (flipud to match their origin).
                    w: int = int(msg.getWidth())  if hasattr(msg, 'getWidth')  else 0  # type: ignore[union-attr]
                    h: int = int(msg.getHeight()) if hasattr(msg, 'getHeight') else 0  # type: ignore[union-attr]
                    if w * h != pts.shape[0] and self._lastImageHW is not None:
                        # getWidth/getHeight unreliable on some builds → fall back to the
                        # companion ToF image grid (depth/amplitude share the sensor res).
                        ih, iw = self._lastImageHW
                        if ih * iw == pts.shape[0]:
                            h, w = ih, iw
                    if w > 0 and h > 0 and w * h == pts.shape[0]:
                        pts = np.flipud(pts.reshape(h, w, 3))
                    self.pushFrame(name, np.ascontiguousarray(pts))

                elif hasattr(msg, 'getCvFrame'):
                    frame: np.ndarray = np.flipud(msg.getCvFrame())  # type: ignore[union-attr]
                    if name == 'rgb':
                        frame = frame[..., ::-1]
                    self._lastImageHW = (frame.shape[0], frame.shape[1])
                    self.pushFrame(name, np.ascontiguousarray(frame))
                    if name == 'rgb':
                        frameCount += 1
                        elapsed: float = hostTs - fpsTimer
                        if elapsed >= 1.0:
                            fps: float = frameCount / elapsed
                            self.pushStatus(True, fps=fps,
                                             latency=latencyMs, streams=streamNames)
                            frameCount = 0
                            fpsTimer   = hostTs
                            tdThread.SetProgressSafe(  # type: ignore[attr-defined]
                                min(fps / config['fps'], 1.0) if config['fps'] else 0.5)

                elif hasattr(msg, 'getFrame'):
                    self.pushFrame(name, np.ascontiguousarray(msg.getFrame()))  # type: ignore[union-attr]

                elif hasattr(msg, 'map'):
                    # depthai_nodes Map2D (monocular depth, heatmaps, etc.) → float32 (H, W).
                    # flipud to match the image streams' TD origin so it overlays the rgb.
                    arr: np.ndarray = np.asarray(msg.map, dtype=np.float32)  # type: ignore[union-attr]
                    if arr.ndim >= 2:
                        self._lastImageHW = (arr.shape[0], arr.shape[1])
                    self.pushFrame(name, np.ascontiguousarray(np.flipud(arr)))

            time.sleep(0.0002)

    def decodeNodeMessage(self, msg: object) -> object:
        """WORKER thread. Decode a depthai-nodes typed message into plain, thread-safe data.
        Returns the payload, None to skip the frame (decode failed — logged to Lasterror), or
        the _UNHANDLED sentinel when msg isn't a recognised depthai-nodes type (so the caller
        falls through to the built-in raw-depthai decoders). A per-stream Handler `parse` still
        wins over this — these are zero-config defaults, not a replacement for custom decode."""
        types = _nodeMessageTypes()
        if not types:
            return _UNHANDLED

        def isType(name: str) -> bool:
            cls = types.get(name)
            return cls is not None and isinstance(msg, cls)

        try:
            if isType('ImgDetectionsExtended'):
                out: list[ExtendedDetectionEntry] = []
                for d in msg.detections:                       # type: ignore[attr-defined]
                    box, angle = _rotatedRectBox(d.rotated_rect)
                    out.append(ExtendedDetectionEntry(
                        label=      int(d.label),
                        confidence= round(float(d.confidence), 3),
                        box=        box,
                        angle=      angle,
                        keypoints=  [[float(k.x), float(k.y), float(getattr(k, 'confidence', 1.0))]
                                     for k in getattr(d, 'keypoints', [])],
                    ))
                return out

            if isType('Keypoints'):
                return [[float(k.x), float(k.y), float(getattr(k, 'confidence', 1.0))]
                        for k in msg.keypoints]                 # type: ignore[attr-defined]

            if isType('SegmentationMask'):
                # int class-index map, -1 = background → clamp to 0 and coerce to a TOP-friendly
                # uint. flipud to match the image streams' TD origin so it overlays the rgb.
                mask = np.asarray(msg.mask)                     # type: ignore[attr-defined]
                mask = np.where(mask < 0, 0, mask).astype(np.uint16)
                if mask.ndim >= 2:
                    self._lastImageHW = (mask.shape[0], mask.shape[1])
                return np.ascontiguousarray(np.flipud(mask))

            if isType('Classifications'):
                return [ClassificationEntry(label=str(c), score=round(float(s), 4))
                        for c, s in zip(msg.classes, msg.scores)]  # type: ignore[attr-defined]

            if isType('Lines'):
                return [[float(ln.start_point.x), float(ln.start_point.y),
                         float(ln.end_point.x),   float(ln.end_point.y),
                         float(getattr(ln, 'confidence', 1.0))]
                        for ln in msg.lines]                    # type: ignore[attr-defined]

            if isType('Clusters'):
                return [[[float(p.x), float(p.y)] for p in cl.points]
                        for cl in msg.clusters]                 # type: ignore[attr-defined]
        except Exception as exc:
            with self._lock:
                self._lastError = f'decode[{type(msg).__name__}]: {exc}'
            return None
        return _UNHANDLED

    def cacheIntrinsics(self, device: object) -> None:
        import depthai as dai
        result: dict[str, dict[str, float]] = {}
        try:
            calib = device.readCalibration()  # type: ignore[union-attr]
            for feat in device.getConnectedCameraFeatures():  # type: ignore[union-attr]
                key = str(feat.socket).split('.')[-1]
                try:
                    mat: list[list[float]] = calib.getCameraIntrinsics(feat.socket)
                    result[key] = {
                        'fx': mat[0][0],
                        'fy': mat[1][1],
                        'cx': mat[0][2],
                        'cy': mat[1][2],
                    }
                except Exception:
                    pass
        except Exception:
            pass
        with self._lock:
            self._intrinsics = result

    def buildPipeline(self, pipeline: _Pipeline, config: PipelineConfig,
                       ) -> tuple[dict[str, _OutputQueue], list[str], dict[str, _InputQueue]]:
        """No TD objects. Populates pipeline with nodes, returns ({name: out_queue}, [stream_names], {name: in_queue})."""
        import depthai as dai

        if config['pipelineScript']:
            namespace: dict[str, object] = {'dai': dai}
            exec(config['pipelineScript'], namespace)
            build_fn = namespace.get('build')
            if not build_fn:
                raise RuntimeError(
                    "pipeline.py must define: def build(pipeline) -> ({name: queue}, [stream_names])")
            result = build_fn(pipeline)  # type: ignore[operator]
            if len(result) == 3:
                queues, streams, controls = result
            else:
                queues, streams = result
                controls: dict[str, _InputQueue] = {}
            return queues, streams, controls

        queues:   dict[str, _OutputQueue] = {}
        streams:  list[str]               = []
        controls: dict[str, _InputQueue]  = {}

        device        = pipeline.getDefaultDevice()  # type: ignore[attr-defined]
        cam_features  = device.getConnectedCameraFeatures()
        primary_types = [
            f.supportedTypes for f in cam_features
            if f.socket == dai.CameraBoardSocket.CAM_A
        ]
        is_tof = bool(primary_types and dai.CameraSensorType.TOF in primary_types[0])

        if is_tof:
            tof_node = pipeline.create(dai.node.ToF).build(dai.CameraBoardSocket.CAM_A)  # type: ignore[attr-defined,union-attr]
            queues['depth'] = tof_node.depth.createOutputQueue()
            streams.append('depth')
            controls['tof_cfg'] = tof_node.tofBaseInputConfig.createInputQueue(maxSize=1, blocking=False)  # type: ignore[attr-defined]
        else:
            cam     = pipeline.create(dai.node.Camera).build()
            rgb_out = cam.requestOutput((1920, 1080), type=dai.ImgFrame.Type.BGR888p)
            queues['rgb'] = rgb_out.createOutputQueue()
            streams.append('rgb')
            controls['cam_ctrl'] = cam.inputControl.createInputQueue(maxSize=1, blocking=False)  # type: ignore[attr-defined]

            if config['enableDepth']:
                left   = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
                right  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
                stereo = pipeline.create(dai.node.StereoDepth)
                stereo.setRectification(True)
                stereo.setLeftRightCheck(True)
                left.requestFullResolutionOutput().link(stereo.left)
                right.requestFullResolutionOutput().link(stereo.right)
                queues['depth'] = stereo.depth.createOutputQueue()
                streams.append('depth')
                controls['stereo_cfg'] = stereo.inputConfig.createInputQueue(maxSize=1, blocking=False)  # type: ignore[attr-defined]

            if config['enableImu']:
                imu = pipeline.create(dai.node.IMU)
                imu.enableIMUSensor([
                    dai.IMUSensor.ACCELEROMETER_RAW,
                    dai.IMUSensor.GYROSCOPE_RAW,
                ], reportRate=500)
                imu.setBatchReportThreshold(1)
                imu.setMaxBatchReports(10)
                queues['imu'] = imu.out.createOutputQueue()
                streams.append('imu')

        return queues, streams, controls

    # ── Stream menu + scriptop dispatch (main thread) ────────────────────

    def updateStreamMenus(self, streamNames: list[str]) -> None:
        """Update the Stream menu on block[0]; TD propagates to all other blocks."""
        if streamNames == self._lastStreamNames:
            return
        self._lastStreamNames = list(streamNames)
        seq = getattr(self.ownerComp.seq, 'Streams', None)
        if seq is None or len(seq) == 0:
            return
        seq.blockPars.Stream[0].menuNames  = streamNames
        seq.blockPars.Stream[0].menuLabels = streamNames

    def pushToScriptops(self) -> None:
        """Auto-push the latest frame to every bound scriptop (sequential block + registry).
        Script TOPs and Script CHOPs support 'Modify Outside of Cook', so we push into them
        from here. Script SOPs have no such toggle — geometry must be built inside the cook,
        so a SOP pulls instead: call FillScriptOp(me) from its onCook."""
        seq = getattr(self.ownerComp.seq, 'Streams', None)
        if seq is not None:
            for i in range(len(seq)):
                streamName: str     = seq.blockPars.Stream[i].eval()
                targetOp:   OP | None = seq.blockPars.Scriptop[i].eval()
                if not streamName or targetOp is None:
                    continue
                self.autoPush(streamName, targetOp,
                               self._fillers.get(self.fillerId(streamName, i)))

        # Runtime-registered scriptops have no Streams block, hence no per-block fill.
        for streamName, opList in list(self._scriptopRegistry.items()):
            for targetOp in opList:
                self.autoPush(streamName, targetOp, None)

    def autoPush(self, streamName: str, targetOp: OP, filler: FillFn | None) -> None:
        data = self.GetFrame(streamName)
        if data is None:
            return
        if filler is not None:
            # Script TOP/CHOP ('Modify Outside of Cook') and DATs can be written from here.
            # Script SOPs cannot — geometry must be built in the cook, so a SOP pulls via
            # FillScriptOp(me) from its onCook instead.
            if targetOp.family == 'SOP':
                return
            modPar = getattr(targetOp.par, 'modoutsidecook', None)
            if modPar is not None:
                modPar.val = True
            try:
                filler(targetOp, data)
            except Exception as exc:
                self.ownerComp.par.Lasterror = f'fill[{streamName}]: {exc}'
            return
        # No custom fill: default numpy -> Script TOP only. Non-array defaults
        # (detections/pose/imu) and Script CHOP/SOPs are pulled, not pushed.
        if isinstance(targetOp, scriptTOP) and isinstance(data, np.ndarray):
            targetOp.par.modoutsidecook = True
            targetOp.copyNumpyArray(self.prepareTopFrame(data))

    def prepareTopFrame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            h, w = frame.shape
            out = np.empty((h, w, 1), dtype=frame.dtype)
            out[:, :, 0] = frame
            return out
        if frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8:
            alpha = np.full(frame.shape[:2], 255, dtype=np.uint8)
            return np.ascontiguousarray(np.dstack([frame, alpha]))
        return np.ascontiguousarray(frame)

    # ── Thread Manager hooks (main thread) ────────────────────────────────

    def onRefresh(self) -> None:
        if self.ownerComp.par.Autoreload.eval():
            datRef: str = self.ownerComp.par.Pipelinedat.eval()
            if datRef:
                dat = op(datRef)
                if dat and dat.changeCount != self._datVersion:
                    self._datVersion = dat.changeCount
                    self.Rebuild()
                    return

        status = self.GetStatus()
        self.ownerComp.par.Connected     = 1 if status['connected'] else 0
        self.ownerComp.par.Fps           = round(status['fps'], 1)
        self.ownerComp.par.Activestreams = ', '.join(status['streams'])
        self.ownerComp.par.Lasterror     = status['error']
        self.updateStreamMenus(status['streams'])
        self.pushToScriptops()

    def closeActiveDevice(self) -> None:
        """Release the active pipeline + device handle. Safe to call on any path
        (success or failure); guarded so a half-built pipeline can't crash teardown."""
        if self._activePipeline is not None:
            try:
                self._activePipeline.stop()
            except Exception:
                pass
            self._activePipeline = None
        if self._activeDevice is not None:
            try:
                self._activeDevice.close()
            except Exception:
                pass
            self._activeDevice = None
        with self._lock:
            self._intrinsics    = {}
            self._controlQueues = {}

    def onSuccess(self) -> None:
        # Device already released on the worker (cameraPump's finally) — this hook runs on
        # the main thread, so it must not call the blocking close itself.
        self.pushStatus(False)
        self.ownerComp.par.Connected = 0

    def onExcept(self, *args: object) -> None:
        # Device already released on the worker (cameraPump's finally), so the retry can't
        # reopen a still-open (PoE) handle. Don't close here — this hook is on the main thread.
        errorMsg: str = str(args[0]) if args else 'Unknown error'
        self._retryCount += 1

        if self._retryCount >= 3:
            finalMsg = f'Pipeline failed after 3 attempts — press Start to retry. Last error: {errorMsg}'
            self.pushStatus(False, error=finalMsg)
            self.ownerComp.par.Lasterror = finalMsg
            return

        self.pushStatus(False, error=errorMsg)
        self.ownerComp.par.Lasterror = errorMsg
        delayFrames: int = int(self._reconnectDelay * me.time.rate)
        run('args[0].doStart()', self, delayFrames=delayFrames, fromOP=self.ownerComp)

    # ── Helpers ───────────────────────────────────────────────────────────

    def readConfig(self) -> PipelineConfig:
        par    = self.ownerComp.par
        datRef: str = par.Pipelinedat.eval()
        raw_device: str = par.Device.eval().strip()
        device: str = '' if raw_device in ('', 'None') else raw_device
        return PipelineConfig(
            device=         device,
            fps=            int(par.Fps.eval()),
            enableDepth=    bool(par.Enabledepth.eval()),
            enableImu=      bool(par.Enableimu.eval()),
            pipelineScript= op(datRef).text if datRef else None,
            parsers=        {},  # filled in doStart via loadStreamHandlers()
        )

    @staticmethod
    def fillerId(streamName: str, index: int) -> str:
        """Unique id for a Streams block's fill, so the same stream bound in several
        blocks doesn't collide. Frames/parse stay keyed by stream (one physical queue),
        but each block fills its own scriptop, so fill is keyed per block."""
        return f'{streamName}#{index}'

    def loadStreamHandlers(self) -> tuple[dict[str, ParseFn], dict[str, FillFn]]:
        """Main thread. Exec each Streams block's Handler DAT (if the param exists) and
        collect its optional parse(msg) — run later on the worker — and fill(scriptOp, data)
        — run on the main thread. Streams without a Handler use the built-in defaults.
        The DAT text is read here (main thread) because the worker can't touch ops.

        parse is keyed by stream name: the worker reads one queue per physical stream, so
        a stream has a single parse (if two blocks on the same stream both define one, the
        last loaded wins). fill is keyed by a per-block id ({stream}#{i}) so the same
        stream bound in multiple blocks can fill each block's scriptop differently."""
        import depthai as dai
        parsers: dict[str, ParseFn] = {}
        fillers: dict[str, FillFn]  = {}
        seq = getattr(self.ownerComp.seq, 'Streams', None)
        if seq is None:
            return parsers, fillers
        for i in range(len(seq)):
            streamName: str = seq.blockPars.Stream[i].eval()
            try:
                handlerDat = seq.blockPars.Handler[i].eval()
            except Exception:
                return parsers, fillers  # Handler param not added yet → defaults only
            if not streamName or handlerDat is None:
                continue
            namespace: dict[str, object] = {'dai': dai, 'np': np}
            try:
                exec(handlerDat.text, namespace)
            except Exception as exc:
                self.ownerComp.par.Lasterror = f'handler[{self.fillerId(streamName, i)}] load error: {exc}'
                continue
            parseFn = namespace.get('parse')
            fillFn  = namespace.get('fill')
            if callable(parseFn):
                parsers[streamName] = parseFn  # type: ignore[assignment]
            if callable(fillFn):
                fillers[self.fillerId(streamName, i)] = fillFn  # type: ignore[assignment]
        return parsers, fillers
