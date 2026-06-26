from __future__ import annotations

import math
import os
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
    models:         list[str]   # everything the script declares (module-level MODELS): slugs + local paths
    localModels:    list[str]   # subset of models that resolve to a local file → load directly, no zoo prefetch
    cacheDir:       str         # DEPTHAI_ZOO_CACHE_PATH shared by the prefetch subprocess + in-proc build
    venvPython:     str         # interpreter to run the prefetch subprocess with


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
        self._datVersion:        int | None                                 = None
        self._latestDevices:     list[DeviceInfo]                          = []
        self._onDevicesCallback: Callable[[list[DeviceInfo]], None] | None = None
        self._activePipeline:    _Pipeline | None                           = None
        self._activeDevice:      _Device | None                             = None
        self._intrinsics:        dict[str, dict[str, float]]               = {}
        self._controlQueues:     dict[str, _InputQueue]                    = {}
        self._fillers:           dict[str, FillFn]                         = {}  # live, keyed by bound op path
        # Parsers are read by the worker (runPipeline) per message and swapped atomically by the
        # main thread on a handler reload — keyed by stream name, lock-free (whole-dict reassign).
        self._parsers:           dict[str, ParseFn]                        = {}
        # Component-registered handlers (RegisterHandler) live apart from the seq-derived ones so a
        # handler reload (which rebuilds from the Streams sequence) can't clobber them. fillers are
        # keyed by the consumer op's path (one fill per op), parsers by stream (one per queue).
        self._componentFillers:  dict[str, FillFn]                         = {}
        self._componentParsers:  dict[str, ParseFn]                        = {}
        self._handlersVersion:   int | None                                = None  # baseline for auto-reload
        self._frameSeq:          int                                       = 0

        self._scriptopRegistry: dict[str, list[OP]] = {}
        self._modelProgress:    dict[str, float]     = {}  # zoo slug -> download fraction 0..1
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
        # stays at its None sentinel and the first onRefresh tick reads a real hash != None as
        # an edit, firing a spurious Rebuild() a moment after every Start.
        datRef = self.ownerComp.par.Pipelinedat.eval()
        dat    = op(datRef) if datRef else None
        self._datVersion = self._datContentHash(dat)
        # Load per-stream Handler DATs on the main thread (the worker can't touch ops): parsers
        # are read by the worker, fillers by the main-thread cook side. Both live on the instance
        # so ReloadHandlers() can hot-swap them without restarting the pipeline. Seed the
        # handlers-version baseline too, so the first onRefresh tick doesn't fire a spurious reload.
        self.applyHandlers(*self.loadStreamHandlers())
        self._handlersVersion = self.handlersContentHash()
        # Models the pipeline script declares (module-level MODELS = [...]) get pre-fetched on a
        # worker subprocess so the download never holds TD's GIL. Mirror them into the readonly
        # Models sequence here (main thread, the only place TD ops are touchable) so each model
        # gets a live Progress readout that onRefresh updates while the subprocess runs.
        models = self.readPipelineModels(config)
        local  = self.localModels(models)
        config['models']      = models
        config['localModels'] = list(local)
        # Local NNArchive/blob models need no download — seed them at 1.0 so the sequence shows
        # them ready immediately; zoo slugs start at 0.0 and fill as the prefetch streams.
        with self._lock:
            self._modelProgress = {m: (1.0 if m in local else 0.0) for m in models}
        self.setupModelBlocks(models)
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

    def ReloadHandlers(self) -> None:
        """Re-load the Streams blocks' Handler DATs and hot-swap parse/fill into the running
        pipeline — no device/pipeline restart. Use after adding or editing a Streams block,
        its Scriptop, or its Handler DAT. Adding a brand-new *physical* stream still needs a
        Rebuild (that's a pipeline build() change). Safe to call when stopped — it just primes
        state for the next Start."""
        self.applyHandlers(*self.loadStreamHandlers())
        self._handlersVersion = self.handlersContentHash()

    def applyHandlers(self, seqParsers: dict[str, ParseFn],
                      seqFillers: dict[str, FillFn]) -> None:
        """Compose the live parse/fill dicts from the two sources — the Streams sequence
        (seq*, rebuilt on every reload) and the components (RegisterHandler, persistent).
        Component registrations win on conflict and survive seq reloads. _fillers is read on
        the main thread (cook side); _parsers is swapped whole so the worker reads it lock-free."""
        self._fillers = {**seqFillers, **self._componentFillers}
        with self._lock:
            self._parsers = {**seqParsers, **self._componentParsers}

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

    def OnPulseStart(self, par: Par) -> None:          self.Start()
    def OnPulseStop(self, par: Par) -> None:           self.Stop()
    def OnPulseRebuild(self, par: Par) -> None:        self.Rebuild()
    def OnPulseListdevices(self, par: Par) -> None:    self.ListDevices()
    def OnPulseReloadhandlers(self, par: Par) -> None: self.ReloadHandlers()

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

    def RegisterHandler(self, scriptop: OP, streamName: str, handlerDat: OP | None) -> None:
        """Ingest a consumer component's Handler DAT: exec it once and store its optional
        parse(msg) — keyed by stream, run on the worker at decode time — and fill(scriptOp, data)
        — keyed by this op's path, run inside the op's cook. Call when the component's
        Stream/Handler par or the DAT's contents change (a parexec + DAT-execute), not per cook.
        Pass handlerDat=None (or an empty DAT) to drop this op's fill — see UnregisterHandler for
        full removal. Parse is shared per stream (one physical queue): the last component to
        register a parse for a stream wins, and clearing one op's handler leaves it in place."""
        import depthai as dai
        key = scriptop.path
        if handlerDat is None or not handlerDat.text.strip():
            self._componentFillers.pop(key, None)
            self._fillers.pop(key, None)
            return
        namespace: dict[str, object] = {'dai': dai, 'np': np}
        try:
            exec(handlerDat.text, namespace)
        except Exception as exc:
            self.ownerComp.par.Lasterror = f'handler[{key}] load error: {exc}'
            return
        fillFn  = namespace.get('fill')
        parseFn = namespace.get('parse')
        if callable(fillFn):
            self._componentFillers[key] = fillFn        # type: ignore[assignment]
            self._fillers[key]          = fillFn         # type: ignore[assignment]
        else:
            self._componentFillers.pop(key, None)
            self._fillers.pop(key, None)
        if callable(parseFn):
            self._componentParsers[streamName] = parseFn  # type: ignore[assignment]
            with self._lock:                               # atomic swap → worker reads safely
                self._parsers = {**self._parsers, streamName: parseFn}  # type: ignore[dict-item]

    def UnregisterHandler(self, scriptop: OP, streamName: str | None = None) -> None:
        """Drop a component's handler. Always removes this op's fill (keyed by op path). If
        streamName is given, also removes that stream's component parse — only do this when no
        other component still relies on it, since parse is shared per physical queue."""
        key = scriptop.path
        self._componentFillers.pop(key, None)
        self._fillers.pop(key, None)
        if streamName is not None:
            self._componentParsers.pop(streamName, None)
            with self._lock:
                self._parsers = {**self._parsers}
                self._parsers.pop(streamName, None)

    def RegisterComponent(self, scriptop: OP, streamName: str | None = None,
                          handlerDat: OP | None = None) -> None:
        """One-call wiring for a stream component: bind the scriptop to its stream and ingest its
        Handler DAT. streamName/handlerDat default to the wrapper component's (scriptop.parent())
        Stream and Handler pars, so the component wires itself with RegisterComponent(op('script1')).
        Call on init and whenever the component's Stream/Handler par or the handler text changes —
        not per cook. Re-binds cleanly across a Stream change: the op is cleared from any prior
        stream first, so the registry never keeps a stale binding."""
        comp = scriptop.parent()
        if streamName is None:
            streamPar  = getattr(comp.par, 'Stream', None)
            streamName = streamPar.eval() if streamPar is not None else None
        if handlerDat is None:
            handlerPar = getattr(comp.par, 'Handler', None)
            handlerDat = handlerPar.eval() if handlerPar is not None else None
        if not streamName:
            self.UnregisterComponent(scriptop)  # no stream yet → make sure nothing stale lingers
            return
        for prior in list(self._scriptopRegistry):  # drop any binding under a previous stream
            if prior != streamName:
                self.UnregisterScriptop(prior, scriptop)
        self.RegisterScriptop(streamName, scriptop)
        self.RegisterHandler(scriptop, streamName, handlerDat)

    def UnregisterComponent(self, scriptop: OP) -> None:
        """Tear down a stream component (e.g. on delete): unbind it from every stream and drop
        its fill. Parse is left in place — it's shared per physical queue, so removing it is the
        manager's call (a Rebuild/ReloadHandlers re-derives it), not a single component's."""
        for streamName in list(self._scriptopRegistry):
            self.UnregisterScriptop(streamName, scriptop)
        self.UnregisterHandler(scriptop)

    def bindingForOp(self, scriptop: OP) -> str | None:
        """Resolve which stream a Script Operator is bound to, via the runtime registry
        (RegisterScriptop, used by the components) or a Streams block. Fill is keyed by the
        op's own path, so only the stream name is needed here. Returns None if not bound."""
        for streamName, ops in self._scriptopRegistry.items():
            if scriptop in ops:
                return streamName
        seq = getattr(self.ownerComp.seq, 'Streams', None)
        if seq is not None:
            for i in range(len(seq)):
                regOp = seq.blockPars.Scriptop[i].eval()
                if regOp is not None and regOp.path == scriptop.path:
                    return seq.blockPars.Stream[i].eval()
        return None

    def GetFrameForScriptop(self, scriptop: OP) -> FrameData | None:
        """Return the frame data for this scriptop via registry or sequential block.
        Call from a Script CHOP cook callback to pull data."""
        streamName = self.bindingForOp(scriptop)
        return self.GetFrame(streamName) if streamName is not None else None

    def FillScriptOp(self, scriptop: OP) -> bool:
        """Pull entry point: call from a Script Operator's onCook to fill it inside its cook.
        Runs the op's bound fill(scriptOp, data) — registered via RegisterHandler or a Streams
        block, both keyed by the op's path — against the latest frame for its stream. Falls back
        to copyNumpyArray for a Script TOP with no custom fill. Returns True if it filled."""
        name = self.bindingForOp(scriptop)
        if name is None:
            return False
        data = self.GetFrame(name)
        if data is None:
            return False
        filler = self._fillers.get(scriptop.path)
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

    def setModelProgress(self, model: str, frac: float) -> None:
        """Worker thread: record a model's download fraction (0..1). onRefresh mirrors it
        into the matching Models block's Progress par on the main thread."""
        with self._lock:
            self._modelProgress[model] = min(max(frac, 0.0), 1.0)

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

    def downloadModels(self, config: PipelineConfig, platform: str,
                       stopFlag: threading.Event) -> bool:
        """WORKER thread. Pre-fetch each declared model on a child interpreter so the
        download holds its OWN GIL, not TD's. Parse depthai's JSON progress
        ({"download_now", "download_total", "model_name", "log_type": "download"}, emitted
        ~every 100ms) and push a 0..1 fraction per model. Returns True if every model exited
        cleanly (so the caller can trust the cache and skip the online check)."""
        import json
        import subprocess
        import sys

        # Local models are loaded straight from disk in the script's build() — nothing to fetch,
        # so just mark them complete. Only zoo slugs go through getModelFromZoo.
        local = set(config['localModels'])
        for model in local:
            self.setModelProgress(model, 1.0)
        zooModels = [m for m in config['models'] if m not in local]
        if not zooModels:
            return True

        python = config['venvPython'] if os.path.exists(config['venvPython']) else sys.executable
        env = dict(os.environ)
        env['DEPTHAI_ZOO_CACHE_PATH'] = config['cacheDir']
        env['DEPTHAI_ZOO_INTERNET_CHECK'] = '1'  # the subprocess is allowed to hit the network
        # NNModelDescription needs the device platform (RVC2/RVC4) to resolve the right artifact.
        code = (
            'import sys, depthai as dai\n'
            'd = dai.NNModelDescription()\n'
            'd.model = sys.argv[1]\n'
            'd.platform = sys.argv[2]\n'
            "dai.getModelFromZoo(d, useCached=True, progressFormat='json')\n"
        )
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0  # type: ignore[attr-defined]
        allOk = True

        for model in zooModels:
            if stopFlag.is_set():
                return False
            proc = subprocess.Popen(
                [python, '-c', code, model, platform],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env, creationflags=flags,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if stopFlag.is_set():
                    proc.terminate()
                    proc.wait()
                    return False
                line = line.strip()
                if not line.startswith('{'):
                    continue  # depthai's own [info] logs aren't JSON — ignore them
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get('log_type') != 'download':
                    continue
                total = msg.get('download_total') or 0
                if total:
                    self.setModelProgress(model, msg.get('download_now', 0) / total)
            proc.wait()
            if proc.returncode == 0:
                self.setModelProgress(model, 1.0)  # cached archives emit no progress; force full
            else:
                allOk = False
                with self._lock:
                    self._lastError = f'download[{model}] failed (exit {proc.returncode})'
        return allOk

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

        # Pre-fetch declared models on a subprocess (its own interpreter = its own GIL) so the
        # download stays off TD's main thread and we can stream a live progress counter. Then
        # point the in-process build at the same cache and skip the online freshness check, so
        # build() loads the cached archive instead of doing a GIL-holding network round-trip.
        platform = pipeline.getDefaultDevice().getPlatformAsString()  # type: ignore[attr-defined]
        prefetched = self.downloadModels(config, platform, stopFlag)
        # Only trust the cache (skip the online check) when we actually pre-fetched a zoo slug.
        # An all-local pipeline leaves the check on so any incidental zoo resolve still works.
        zooDeclared = any(m not in set(config['localModels']) for m in config['models'])
        os.environ['DEPTHAI_ZOO_CACHE_PATH']     = config['cacheDir']
        os.environ['DEPTHAI_ZOO_INTERNET_CHECK'] = '0' if (zooDeclared and prefetched) else '1'
        if stopFlag.is_set():
            return  # Stop pressed mid-download — don't build/start; cameraPump's finally closes the device

        queues, streamNames, controlQueues = self.buildPipeline(pipeline, config)
        self.pushStatus(False, streams=streamNames)

        self._activePipeline = pipeline
        pipeline.start()
        self.cacheIntrinsics(pipeline.getDefaultDevice())  # type: ignore[attr-defined]
        with self._lock:
            self._controlQueues = controlQueues
        self.pushStatus(True, streams=streamNames)

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
                # Read self._parsers fresh each message so ReloadHandlers() can hot-swap them
                # (whole-dict reassign on the main thread → atomic, no torn read) mid-run.
                parseFn = self._parsers.get(name)
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

    def setupModelBlocks(self, models: list[str]) -> None:
        """Main thread. Resize the readonly Models sequence to one block per declared model and
        fill in each Model name, resetting Progress to 0. Grow with numBlocks (the documented
        'press +' equivalent) and shrink with destroyBlock from the tail."""
        seq = getattr(self.ownerComp.seq, 'Models', None)
        if seq is None:
            return
        needed = len(models)
        while seq.numBlocks < needed:
            seq.numBlocks += 1
        while seq.numBlocks > needed:
            before = seq.numBlocks
            seq.destroyBlock(seq.numBlocks - 1)
            if seq.numBlocks >= before:
                break  # sequence hit its minimum block count and won't shrink further
        for i, model in enumerate(models):
            seq.blockPars.Model[i].val    = model
            seq.blockPars.Progress[i].val = 0.0

    def updateModelProgress(self) -> None:
        """Main thread. Mirror the worker's per-model download fractions into each Models
        block's Progress par, matched by the block's Model name."""
        seq = getattr(self.ownerComp.seq, 'Models', None)
        if seq is None or len(seq) == 0:
            return
        with self._lock:
            progress = dict(self._modelProgress)
        if not progress:
            return
        for i in range(len(seq)):
            frac = progress.get(seq.blockPars.Model[i].eval())
            if frac is not None:
                seq.blockPars.Progress[i].val = frac

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
                               self._fillers.get(targetOp.path))

        # Runtime-registered scriptops carry their fill (if any) under their op path, set by
        # RegisterHandler — so a registered TOP/CHOP with a component handler still pushes.
        for streamName, opList in list(self._scriptopRegistry.items()):
            for targetOp in opList:
                self.autoPush(streamName, targetOp, self._fillers.get(targetOp.path))

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

    def _datContentHash(self, dat: OP | None) -> int | None:
        """Hash of a DAT's contents, used to detect edits for Autoreload. ``dat.text``
        covers both Text and Table DATs (tables come back tab-delimited). Returns None
        when there's no DAT, which never compares equal to a real content hash."""
        return hash(dat.text) if dat else None

    def handlersContentHash(self) -> int:
        """Hash of the handler-relevant Streams state — block count plus each block's Stream,
        Scriptop path, and Handler DAT text. Changes when a block/binding/handler is added,
        removed, or edited, so Autoreload can hot-reload handlers without a full Rebuild."""
        seq = getattr(self.ownerComp.seq, 'Streams', None)
        if seq is None:
            return hash(())
        parts: list[object] = []
        for i in range(len(seq)):
            stream   = seq.blockPars.Stream[i].eval()
            scriptop = seq.blockPars.Scriptop[i].eval()
            try:
                handler = seq.blockPars.Handler[i].eval()
            except Exception:
                handler = None  # Handler param not added on this build
            parts.append((stream,
                          scriptop.path if scriptop is not None else None,
                          handler.text if handler is not None else None))
        return hash(tuple(parts))

    def onRefresh(self) -> None:
        if self.ownerComp.par.Autoreload.eval():
            datRef: str = self.ownerComp.par.Pipelinedat.eval()
            if datRef:
                dat = op(datRef)
                version = self._datContentHash(dat)
                if dat and version != self._datVersion:
                    self._datVersion = version
                    self.Rebuild()           # pipeline build() changed → full restart
                    return
            # Streams blocks / bindings / Handler DATs changed → hot-reload only, no device restart.
            handlersVersion = self.handlersContentHash()
            if handlersVersion != self._handlersVersion:
                self._handlersVersion = handlersVersion
                self.ReloadHandlers()

        status = self.GetStatus()
        self.ownerComp.par.Connected     = 1 if status['connected'] else 0
        self.ownerComp.par.Fps           = round(status['fps'], 1)
        self.ownerComp.par.Activestreams = ', '.join(status['streams'])
        self.ownerComp.par.Lasterror     = status['error']
        self.updateStreamMenus(status['streams'])
        self.updateModelProgress()
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
        # project.folder is the .toe directory — depthai's default cache already lands in its
        # .depthai_cached_models, and the venv sits beside it; both read on the main thread here
        # so the worker (which can't touch TD) gets them via config.
        projFolder: str = project.folder
        return PipelineConfig(
            device=         device,
            fps=            int(par.Fps.eval()),
            enableDepth=    bool(par.Enabledepth.eval()),
            enableImu=      bool(par.Enableimu.eval()),
            pipelineScript= op(datRef).text if datRef else None,
            models=         [],  # filled in doStart via readPipelineModels()
            localModels=    [],  # filled in doStart via localModels()
            cacheDir=       os.path.join(projFolder, '.depthai_cached_models'),
            venvPython=     os.path.join(projFolder, '.venv', 'Scripts', 'python.exe'),
        )

    def readPipelineModels(self, config: PipelineConfig) -> list[str]:
        """Main thread. Extract the zoo slugs a pipeline script declares via a module-level
        ``MODELS = [...]`` (a bare ``MODEL = '...'`` is also accepted). The script is exec'd in
        a throwaway namespace — by convention the heavy work lives in build(), so module level
        is cheap. Returns [] on any error (the build then just resolves models online itself)."""
        script = config['pipelineScript']
        if not script:
            return []
        namespace: dict[str, object] = {}
        try:
            exec(script, namespace)
        except Exception as exc:
            self.ownerComp.par.Lasterror = f'model scan error: {exc}'
            return []
        decl = namespace.get('MODELS', namespace.get('MODEL'))
        if isinstance(decl, str):
            decl = [decl]
        if not isinstance(decl, (list, tuple)):
            return []
        return [str(m) for m in decl if m]

    def localModels(self, entries: list[str]) -> set[str]:
        """Main thread. Entries that resolve to an existing file (absolute, or relative to the
        project) are local NNArchive/blob models loaded directly via dai.NNArchive — no zoo
        prefetch. Everything else is treated as a HubAI zoo slug. project.folder needs TD, so
        this runs here and the result rides to the worker in config['localModels']."""
        local: set[str] = set()
        for entry in entries:
            path = entry if os.path.isabs(entry) else os.path.join(project.folder, entry)
            if os.path.exists(path):
                local.add(entry)
        return local

    def loadStreamHandlers(self) -> tuple[dict[str, ParseFn], dict[str, FillFn]]:
        """Main thread. Exec each Streams block's Handler DAT (if the param exists) and
        collect its optional parse(msg) — run later on the worker — and fill(scriptOp, data)
        — run on the main thread. Streams without a Handler use the built-in defaults.
        The DAT text is read here (main thread) because the worker can't touch ops.

        parse is keyed by stream name: the worker reads one queue per physical stream, so
        a stream has a single parse (if two blocks on the same stream both define one, the
        last loaded wins). fill is keyed by the block's bound Scriptop path — the same key the
        component path (RegisterHandler) uses — so a block with no Scriptop set carries no fill."""
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
                self.ownerComp.par.Lasterror = f'handler[{streamName}#{i}] load error: {exc}'
                continue
            parseFn = namespace.get('parse')
            fillFn  = namespace.get('fill')
            if callable(parseFn):
                parsers[streamName] = parseFn  # type: ignore[assignment]
            regOp = seq.blockPars.Scriptop[i].eval()
            if callable(fillFn) and regOp is not None:
                fillers[regOp.path] = fillFn  # type: ignore[assignment]
        return parsers, fillers
