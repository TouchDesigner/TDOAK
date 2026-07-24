from __future__ import annotations

import os
os.environ.setdefault('DEPTHAI_INSTALL_SIGNAL_HANDLER', '0')
os.environ.setdefault('DEPTHAI_CRASHDUMP', '0')
os.environ.setdefault('DEPTHAI_RECONNECT_TIMEOUT', '0')  # XLink auto-reconnect races Stop teardown and aborts the process
#os.environ.setdefault('DEPTHAI_AUTOCALIBRATION', 'OFF')  # keep the 3.6.1+ auto-injected AutoCalibration node out of pipelines (engages at 640x480 despite documented 1280x800-only support)

import ast
import math
import threading
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Callable, Protocol, TypedDict

import numpy as np

if TYPE_CHECKING:
    from td import COMP, OP, Par, TOP, scriptTOP


RECONNECT_DELAY_S    = 10.0   # delay between automatic retry attempts after a pipeline error
POST_CLOSE_SETTLE_S  = 3.0    # hold the alive-gate after device.close() returns — XLink keeps
                              # tearing down in native code after the call, and reopening the
                              # device inside that window can hard-crash the process
MAX_RETRIES          = 3      # start attempts before giving up until the next Start
REBUILD_DELAY_FRAMES = 5      # frames between Stop and Start on a Rebuild
RESTART_POLL_FRAMES  = 15     # re-check cadence while waiting for the previous worker to exit
MAINTENANCE_PERIOD   = 30     # cooks between autoreload checks + binding-cache rebuilds
POLL_IDLE_SLEEP_S    = 0.003  # worker sleep after a poll pass that yielded no messages
DOWNLOAD_POLL_S      = 0.25   # subprocess wait timeout while pre-fetching models

DEFAULT_RGB_RESOLUTION = (1920, 1080)
IMU_REPORT_RATE_HZ     = 500
IMU_BATCH_THRESHOLD    = 1
IMU_MAX_BATCH_REPORTS  = 10


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
    logLevel:       str         # depthai verbosity: trace/debug/info/warn/error/critical
    enableDepth:    bool
    enableImu:      bool
    pipelineScript: str | None
    pipelinePath:   str         # DAT path of the pipeline script, for traceback attribution
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

# Cached depthai-nodes message classes (or None if depthai-nodes isn't installed). depthai-nodes
# is optional at runtime, so the import is deferred and memoized — never imported at module load.
_NODE_TYPES: dict[str, type] | None = None

# Set when depthai-nodes exists but fails to import (broken install ≠ not installed) — surfaced
# once via setError the first time a message is classified, then cleared.
_NODE_IMPORT_ERROR: str | None = None

# classifyMessage kinds for the depthai-nodes message family, in match order.
# ImgDetectionsExtended also exposes .detections, so it must be recognised before the raw
# dai.ImgDetections duck-type check.
_NODE_KINDS: tuple[tuple[str, str], ...] = (
    ('extdet',          'ImgDetectionsExtended'),
    ('keypoints',       'Keypoints'),
    ('segmask',         'SegmentationMask'),
    ('classifications', 'Classifications'),
    ('lines',           'Lines'),
    ('clusters',        'Clusters'),
)


def _nodeMessageTypes() -> dict[str, type]:
    """Return {classname: class} for the depthai-nodes message types we decode, or {} if the
    package isn't importable. Imported once on the worker thread; cached for the process."""
    global _NODE_TYPES, _NODE_IMPORT_ERROR
    if _NODE_TYPES is None:
        names = tuple(clsName for _, clsName in _NODE_KINDS)
        mod = None
        for path in ('depthai_nodes.message', 'depthai_nodes'):
            try:
                mod = __import__(path, fromlist=['*'])
                break
            except ImportError:
                continue  # not installed — the node decoders are simply disabled
            except Exception as exc:
                _NODE_IMPORT_ERROR = f'depthai_nodes[import]: {exc}'
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
        self._mainThread: threading.Thread = threading.current_thread()

        self._lock:          threading.Lock       = threading.Lock()
        self._frames:        dict[str, FrameData] = {}
        self._frameVersions: dict[str, int]       = {}  # bumped per push; lets autoPush skip identical re-pushes
        self._connected:     bool                 = False
        self._fps:           float                = 0.0
        self._latencyMs:     float                = 0.0
        self._lastError:     str                  = ''
        self._activeStreams: list[str]            = []

        self._retryCount: int = 0
        # Shared worker state ('alive' + current 'stop' Event), fetched from a process-
        # lifetime registry in the TDOAK package so it SURVIVES extension re-inits — a fresh
        # instance must still see (and be able to stop) a previous instance's still-running
        # worker, or a Start right after reinit races the old device teardown (native crash
        # territory). Falls back to private Events when the TDOAK package isn't importable.
        try:
            from TDOAK.core.state import workerState
            self._workerShared: dict[str, threading.Event] = workerState(ownerComp.path)
        except Exception:
            self._workerShared = {'alive': threading.Event(), 'stop': threading.Event()}
        self._workerAlive: threading.Event = self._workerShared['alive']
        self._startRequested:  bool            = False
        self._datVersion:      int | None      = None
        self._latestDevices:   list[DeviceInfo] = []
        self._intrinsics:      dict[str, dict[str, float]] = {}
        self._controlQueues:   dict[str, _InputQueue]      = {}
        self._fillers:         dict[str, FillFn]           = {}  # live, keyed by bound op path
        # Parsers are read lock-free by the worker (runPipeline) each message; the main thread
        # only ever swaps in a NEW dict under the lock (reference reassignment is atomic) —
        # published dicts are never mutated in place.
        self._parsers:         dict[str, ParseFn]          = {}
        # Component-registered handlers (RegisterHandler) live apart from the seq-derived ones so a
        # handler reload (which rebuilds from the Streams sequence) can't clobber them. fillers are
        # keyed by the consumer op's path (one fill per op), parsers by stream (one per queue).
        self._componentFillers: dict[str, FillFn]  = {}
        self._componentParsers: dict[str, ParseFn] = {}
        # Pull-mode handlers (FillScriptOp with an explicit handler DAT), cached per DAT
        # path as (content hash, parse, fill) — re-exec'd only when the DAT text changes.
        self._pullHandlers: dict[str, tuple[int, ParseFn | None, FillFn | None]] = {}
        self._handlersVersion:  int | None         = None  # baseline for auto-reload
        self._frameSeq:         int                = 0

        self._scriptopRegistry: dict[str, list[OP]] = {}
        self._modelProgress:    dict[str, float]     = {}  # zoo slug -> download fraction 0..1
        self._modelsDone:       bool                 = False
        self._lastStreamNames:  list[str]            = []
        self._lastImageHW:      tuple[int, int] | None = None  # (h, w) of last image frame, for organizing point clouds

        # Streams the worker should decode: bound streams + anything pulled via GetFrame.
        # None = decode everything (until the first binding rebuild computes a real set).
        # Swapped whole under the lock; the worker reads the reference lock-free.
        self._wantedStreams: set[str] | None = None
        self._pullMarks:     set[str]        = set()  # streams ever requested via GetFrame

        # Cached (stream, op) push targets — see rebuildBindings.
        self._bindings:       list[tuple[str, OP]] = []
        self._bindingsDirty:  bool                 = False
        self._pushedVersions: dict[tuple[str, str], int] = {}  # (op path, stream) -> last pushed version
        self._refreshTick:    int                  = 0
        self._topScratch:     dict[tuple[int, int], np.ndarray] = {}  # per-resolution RGBA buffers

        # Built-in decoders by classifyMessage kind — all run on the WORKER thread with a
        # uniform (msg, name) signature so the pump dispatches through one dict lookup.
        self._decoders: dict[str, Callable[[object, str], FrameData | None]] = {
            'extdet':          self.decodeExtendedDetections,
            'keypoints':       self.decodeKeypoints,
            'segmask':         self.decodeSegmentationMask,
            'classifications': self.decodeClassifications,
            'lines':           self.decodeLines,
            'clusters':        self.decodeClusters,
            'detections':      self.decodeDetections,
            'imu':             self.decodeImu,
            'pointcloud':      self.decodePointCloud,
            'image':           self.decodeImage,
            'raw':             self.decodeRawFrame,
            'map':             self.decodeMap,
            'unknown':         self.decodeUnknown,
        }

        # reset parameters to known state
        self.ownerComp.par.Connected = 0
        self.ownerComp.par.Fps = 0
        self.ownerComp.par.Activestreams = ''
        self.ownerComp.par.Lasterror = ''
        self.ownerComp.par.Device.menuNames = ['None']
        self.ownerComp.par.Device.menuLabels = ['-- Select Device --']

    # ── Public API (main thread only) ────────────────────────────────────

    def Start(self) -> None:
        self._retryCount = 0
        self.doStart()

    def doStart(self) -> None:
        """Request a start. Defers (re-polling) while a previous worker is still winding
        down — device.close() can block for seconds, and racing it with a second pump would
        let the dying run close the new run's device."""
        self._startRequested = True
        self.startWhenIdle()

    def startWhenIdle(self) -> None:
        if not self._startRequested:
            return  # Stop() cancelled the pending start while we waited
        if self._workerAlive.is_set():
            if not self._workerShared['stop'].is_set():
                # A pump is already running (or booting) and nobody asked it to stop —
                # Start is a no-op, NOT a restart (that's Rebuild). Critically, do NOT
                # touch its stop flag here: a concurrent start chain may have just booted
                # this pump, and killing it from the deferral path livelocks both chains.
                self._startRequested = False
                return
            run('args[0].startWhenIdle()', self,
                delayFrames=RESTART_POLL_FRAMES, fromOP=self.ownerComp)
            return
        self._startRequested = False
        # Fresh Event per run, published in the shared slot: the old worker keeps polling
        # ITS flag (which stays set), so a new Start can never un-stop a run that is still
        # winding down — and Stop() from ANY instance (even after reinit) reaches this run.
        self._workerShared['stop'] = threading.Event()
        self.setError('')
        self._topScratch.clear()
        config = self.readConfig()
        # Seed the Autoreload baseline to the DAT we're starting with. Otherwise _datVersion
        # stays at its None sentinel and the first maintenance tick reads a real hash != None
        # as an edit, firing a spurious Rebuild() a moment after every Start.
        datRef = self.ownerComp.par.Pipelinedat.eval()
        dat    = op(datRef) if datRef else None
        self._datVersion = self.datContentHash(dat)
        # Load per-stream Handler DATs on the main thread (the worker can't touch ops): parsers
        # are read by the worker, fillers by the main-thread cook side. Both live on the instance
        # so ReloadHandlers() can hot-swap them without restarting the pipeline. Seed the
        # handlers-version baseline too, so the first maintenance tick doesn't fire a spurious reload.
        self.applyHandlers(*self.loadStreamHandlers())
        self._handlersVersion = self.handlersContentHash()
        self.rebuildBindings()
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
        self._modelsDone = False
        self.setupModelBlocks(models)
        task = self.ThreadManager.TDTask(
            target=self.cameraPump,
            args=(config, self._workerShared['stop']),
            SuccessHook=self.onSuccess,
            ExceptHook=self.onExcept,
            RefreshHook=self.onRefresh,
        )
        self._workerAlive.set()  # cleared by the worker in cameraPump's finally
        try:
            self.ThreadManager.EnqueueTask(task, standalone=True)
        except Exception:
            self._workerAlive.clear()
            raise

    def Stop(self) -> None:
        self._startRequested = False  # cancel any start still waiting on the old worker
        self._workerShared['stop'].set()

    def Rebuild(self) -> None:
        self.Stop()
        run('args[0].Start()', self, delayFrames=REBUILD_DELAY_FRAMES, fromOP=self.ownerComp)

    def ReloadHandlers(self) -> None:
        """Re-load the Streams blocks' Handler DATs and hot-swap parse/fill into the running
        pipeline — no device/pipeline restart. Use after adding or editing a Streams block,
        its Scriptop, or its Handler DAT. Adding a brand-new *physical* stream still needs a
        Rebuild (that's a pipeline build() change). Safe to call when stopped — it just primes
        state for the next Start."""
        self.applyHandlers(*self.loadStreamHandlers())
        self._handlersVersion = self.handlersContentHash()
        self.rebuildBindings()

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
        task = self.ThreadManager.TDTask(
            target=self.enumerateDevices,
            # Closure, not instance state: overlapping ListDevices calls each keep their own
            # callback instead of the newest one hijacking every completion.
            SuccessHook=lambda: self.onEnumerateSuccess(callback),
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
        self.rebuildBindings()

    def UnregisterScriptop(self, streamName: str, scriptop: OP | None = None) -> None:
        """Remove a registered scriptop. Pass scriptop=None to remove all for the stream."""
        if scriptop is None:
            self._scriptopRegistry.pop(streamName, None)
        elif streamName in self._scriptopRegistry:
            with suppress(ValueError):
                self._scriptopRegistry[streamName].remove(scriptop)
        self.rebuildBindings()

    def RegisterHandler(self, scriptop: OP, streamName: str, handlerDat: OP | None) -> None:
        """Ingest a consumer component's Handler DAT: exec it once and store its optional
        parse(msg) — keyed by stream, run on the worker at decode time — and fill(scriptOp, data)
        — keyed by this op's path, run inside the op's cook. Call when the component's
        Stream/Handler par or the DAT's contents change (a parexec + DAT-execute), not per cook.
        Pass handlerDat=None (or an empty DAT) to drop this op's fill — see UnregisterHandler for
        full removal. Parse is shared per stream (one physical queue): the last component to
        register a parse for a stream wins, and clearing one op's handler leaves it in place."""
        key = scriptop.path
        if handlerDat is None or not handlerDat.text.strip():
            self._componentFillers.pop(key, None)
            self._fillers.pop(key, None)
            return
        loaded = self.execHandler(handlerDat, key)
        if loaded is None:
            return  # exec failed (reported) — keep whatever was registered before
        parseFn, fillFn = loaded
        if fillFn is not None:
            self._componentFillers[key] = fillFn
            self._fillers[key]          = fillFn
        else:
            self._componentFillers.pop(key, None)
            self._fillers.pop(key, None)
        if parseFn is not None:
            self._componentParsers[streamName] = parseFn
            with self._lock:                               # atomic swap → worker reads safely
                self._parsers = {**self._parsers, streamName: parseFn}

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
                # Build-then-swap: published dicts are never mutated in place — the worker
                # reads the reference lock-free.
                self._parsers = {k: v for k, v in self._parsers.items() if k != streamName}

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

    def FillScriptOp(self, scriptop: OP, stream: str | None = None,
                     handler: OP | str | None = None) -> bool:
        """Pull entry point: call from a Script Operator's onCook to fill it inside its cook.
        With no arguments the op's registered binding is used (Streams block / RegisterHandler,
        keyed by the op's path). Pass `stream` (and optionally `handler`, a Handler DAT or its
        path) to pull explicitly with zero registration:

            oak.FillScriptOp(scriptOp, stream='depth', handler=op('text_pointcloudHandler'))

        The handler's fill(scriptOp, data) runs against the stream's latest frame; a parse(msg)
        it defines is installed for the stream's worker-side decode. The DAT is re-exec'd only
        when its text changes, so calling this every cook is cheap. Falls back to the built-in
        defaults (numpy image -> Script TOP, float point array -> Script POP) when no fill is
        available. Returns True if it filled."""
        name = stream if stream is not None else self.bindingForOp(scriptop)
        if name is None:
            return False
        data = self.GetFrame(name)
        if data is None:
            return False
        filler = self.pullHandlerFill(handler, name) if handler is not None else None
        if filler is None:
            filler = self._fillers.get(scriptop.path)
        if filler is not None:
            try:
                filler(scriptop, data)
            except Exception as exc:
                self.setError(f'fill[{name}]: {exc}')
                return False
            return True
        if isinstance(scriptop, scriptTOP) and isinstance(data, np.ndarray):
            scriptop.copyNumpyArray(self.prepareTopFrame(data))
            return True
        if scriptop.family == 'POP' and self.isPointArray(data):
            try:
                self.fillPopPoints(scriptop, data)  # type: ignore[arg-type]
            except Exception as exc:
                self.setError(f'fill[{name}]: {exc}')
                return False
            return True
        return False

    # ── Getters (safe from Script TOP/CHOP cook callbacks) ───────────────

    def GetFrame(self, name: str) -> FrameData | None:
        with self._lock:
            if name not in self._pullMarks:
                # First pull for this stream: remember the interest so the worker keeps
                # decoding it even without a Streams-block/registry binding.
                self._pullMarks.add(name)
                if self._wantedStreams is not None:
                    self._wantedStreams = self._wantedStreams | {name}
            return self._frames.get(name)

    def GetFrameNames(self) -> list[str]:
        with self._lock: return list(self._frames.keys())

    def GetRgb(self)     -> np.ndarray | None:      return self.GetFrame('rgb')       # type: ignore[return-value]
    def GetDepth(self)   -> np.ndarray | None:      return self.GetFrame('depth')     # type: ignore[return-value]
    def GetIrLeft(self)  -> np.ndarray | None:      return self.GetFrame('ir_left')   # type: ignore[return-value]
    def GetIrRight(self) -> np.ndarray | None:      return self.GetFrame('ir_right')  # type: ignore[return-value]
    def GetImu(self)     -> ImuData | None:         return self.GetFrame('imu')       # type: ignore[return-value]
    def GetPose(self)    -> list[PoseEntry] | None: return self.GetFrame('pose')      # type: ignore[return-value]

    def GetIntrinsics(self, socket: str = 'CAM_A') -> dict[str, float] | None:
        """Return {'fx', 'fy', 'cx', 'cy', 'width', 'height'} for the given socket at its
        native sensor resolution, or None if not available. width/height are the native
        resolution the coefficients refer to — scale fx/cx by frameW/width (fy/cy by
        frameH/height) when reprojecting a stream at a different resolution."""
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
        on TD-supplied frames instead of an OAK camera (see examples/detection_input_pipeline.py).
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
        with self._lock:
            self._frames[name] = data
            self._frameVersions[name] = self._frameVersions.get(name, 0) + 1

    def pushDevices(self, devices: list[DeviceInfo]) -> None:
        with self._lock: self._latestDevices = devices

    def setModelProgress(self, model: str, frac: float) -> None:
        """Worker thread: record a model's download fraction (0..1). onRefresh mirrors it
        into the matching Models block's Progress par on the main thread."""
        with self._lock:
            self._modelProgress[model] = min(max(frac, 0.0), 1.0)

    def setError(self, msg: str) -> None:
        """Record an error — the single choke point every error path funnels through.
        Thread-safe: a worker just records state for onRefresh to mirror into Lasterror;
        on the main thread the par is also written immediately, because onRefresh only
        runs while a pump task is alive (errors while stopped must still surface)."""
        with self._lock:
            self._lastError = msg
        if threading.current_thread() is self._mainThread:
            self.ownerComp.par.Lasterror = msg

    def pushStatus(self, connected: bool, fps: float = 0.0, latency: float = 0.0,
                   streams: list[str] | None = None) -> None:
        """Update connection status. Errors are NOT part of status — they go through
        setError and persist until the next Start, so a once-a-second fps push can't
        wipe an error the user hasn't seen yet."""
        with self._lock:
            self._connected = connected
            self._fps       = fps
            self._latencyMs = latency
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

    def onEnumerateSuccess(self, callback: Callable[[list[DeviceInfo]], None] | None = None) -> None:
        devices = self.GetDevices()
        names:  list[str] = [d['mxid']                                      for d in devices]
        labels: list[str] = [f"{d['name'] or d['mxid']}  [{d['protocol']}]" for d in devices]
        names.insert(0, 'None')
        labels.insert(0, '-- Select Device --')
        self.ownerComp.par.Device.menuNames  = names
        self.ownerComp.par.Device.menuLabels = labels
        if callback:
            callback(devices)

    def onEnumerateExcept(self, *args: object) -> None:
        errorMsg: str = str(args[0]) if args else 'Unknown error'
        self.pushDevices([])
        self.setError(f'devices[enumerate]: {errorMsg}')

    def downloadModels(self, config: PipelineConfig, platform: str,
                       stopFlag: threading.Event) -> bool:
        """WORKER thread. Pre-fetch each declared model on a child interpreter so the
        download holds its OWN GIL, not TD's. Returns True if every model exited cleanly
        (so the caller can trust the cache and skip the online check)."""
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
            if not self.fetchModel(python, code, model, platform, env, flags, stopFlag):
                allOk = False
                if stopFlag.is_set():
                    return False
        return allOk

    def fetchModel(self, python: str, code: str, model: str, platform: str,
                   env: dict[str, str], flags: int, stopFlag: threading.Event) -> bool:
        """WORKER thread. Run one zoo prefetch on a child interpreter. Its stdout is drained
        on a side thread so a silently-stalled download can never hang the pump: the wait loop
        here polls the stop flag and kills the child instead of blocking on its pipe."""
        import subprocess
        proc = subprocess.Popen(
            [python, '-c', code, model, platform],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, creationflags=flags,
        )
        drain = threading.Thread(target=self.drainDownloadProgress,
                                 args=(proc.stdout, model), daemon=True)
        drain.start()
        try:
            while True:
                try:
                    proc.wait(timeout=DOWNLOAD_POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    if stopFlag.is_set():
                        proc.kill()
                        proc.wait()
                        return False
        finally:
            if proc.poll() is None:  # left the loop abnormally — never leak the child
                proc.kill()
                proc.wait()
            drain.join(timeout=1.0)
            if proc.stdout is not None:
                proc.stdout.close()
        if proc.returncode == 0:
            self.setModelProgress(model, 1.0)  # cached archives emit no progress; force full
            return True
        self.setError(f'download[{model}]: exit {proc.returncode}')
        return False

    def drainDownloadProgress(self, stream: object, model: str) -> None:
        """Prefetch helper thread. Parse depthai's JSON progress lines
        ({"download_now", "download_total", "log_type": "download"}, emitted ~every 100 ms)
        into a 0..1 fraction. Best-effort: exits on EOF or when fetchModel closes the pipe."""
        import json
        try:
            for line in stream:  # type: ignore[attr-defined]
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
        except Exception:
            pass  # reader raced the pipe close on kill — progress is best-effort

    def cameraPump(self, config: PipelineConfig, stopFlag: threading.Event) -> None:
        # Worker entry point. Tear the device down HERE (worker thread) on every exit path:
        # device.close()/pipeline.stop() are blocking depthai calls, and the SuccessHook/
        # ExceptHook run on TD's MAIN thread — closing there freezes the UI for several
        # frames on each stop/retry. The finally keeps that cost off the main thread.
        # Handles travel through this dict (per-run locals, not instance state) so a dying
        # run can only ever close its own device — see startWhenIdle.
        handles: dict[str, object] = {'pipeline': None, 'device': None}
        try:
            self.runPipeline(config, stopFlag, handles)
        finally:
            self.closeDevice(handles['pipeline'], handles['device'])  # type: ignore[arg-type]
            if handles['device'] is not None or handles['pipeline'] is not None:
                # device.close() returns before XLink finishes tearing down natively —
                # keep the alive-gate held so no deferred Start can reopen into that
                # window (observed to hard-crash TD).
                time.sleep(POST_CLOSE_SETTLE_S)
            self._workerAlive.clear()

    def runPipeline(self, config: PipelineConfig, stopFlag: threading.Event,
                    handles: dict[str, object]) -> None:
        import depthai as dai

        if config['device']:
            device = dai.Device(dai.DeviceInfo(config['device']))
            handles['device'] = device
            pipeline = dai.Pipeline(device)
        else:
            pipeline = dai.Pipeline()
        handles['pipeline'] = pipeline

        # Pre-fetch declared models on a subprocess (its own interpreter = its own GIL) so the
        # download stays off TD's main thread and we can stream a live progress counter. Then
        # point the in-process build at the same cache and skip the online freshness check, so
        # build() loads the cached archive instead of doing a GIL-holding network round-trip.
        platform = pipeline.getDefaultDevice().getPlatformAsString()  # type: ignore[attr-defined]
        prefetched = self.downloadModels(config, platform, stopFlag)
        # Only trust the cache (skip the online check) when we actually pre-fetched a zoo slug.
        # An all-local pipeline leaves the check on so any incidental zoo resolve still works.
        zooDeclared = any(m not in set(config['localModels']) for m in config['models'])
        # NOTE: process-global environment — this affects every depthai user in the TD process
        # and assumes a single OAK component per project (or identical settings across them).
        os.environ['DEPTHAI_ZOO_CACHE_PATH']     = config['cacheDir']
        os.environ['DEPTHAI_ZOO_INTERNET_CHECK'] = '0' if (zooDeclared and prefetched) else '1'
        if stopFlag.is_set():
            return  # Stop pressed mid-download — don't build/start; cameraPump's finally closes the device

        queues, streamNames, controlQueues = self.buildPipeline(pipeline, config)
        self.pushStatus(False, streams=streamNames)

        pipeline.start()
        # Cap the device's log chatter (temperatures, cpu usage, ...) to the configured
        # verbosity — setLogLevel gates device->host transfer, setLogOutputLevel printing.
        with suppress(Exception):
            levels = {'trace': dai.LogLevel.TRACE, 'debug': dai.LogLevel.DEBUG,
                      'info': dai.LogLevel.INFO, 'warn': dai.LogLevel.WARN,
                      'error': dai.LogLevel.ERR, 'critical': dai.LogLevel.CRITICAL}
            level = levels.get(config['logLevel'])
            if level is not None:
                dev = pipeline.getDefaultDevice()  # type: ignore[attr-defined]
                dev.setLogLevel(level)
                dev.setLogOutputLevel(level)
        self.cacheIntrinsics(pipeline.getDefaultDevice())  # type: ignore[attr-defined]
        with self._lock:
            self._controlQueues = controlQueues
        self.pushStatus(True, streams=streamNames)

        frameCount: int   = 0
        fpsTimer:   float = time.monotonic()
        tdThread          = threading.current_thread()
        kinds: dict[str, str] = {}  # stream -> decode kind, classified on the first message
        # Fps/latency are metered on one designated stream: rgb when present, otherwise
        # the pipeline's first stream (ToF/custom pipelines carry no rgb at all). Counted
        # before the wanted-filter so the readout ticks even when nothing is bound yet.
        clockStream: str | None = 'rgb' if 'rgb' in queues else next(iter(queues), None)

        while not stopFlag.is_set() and pipeline.isRunning():
            got = False
            for name, q in queues.items():
                msg = q.tryGet()
                if msg is None:
                    continue
                got = True

                if name == clockStream:
                    frameCount += 1
                    now     = time.monotonic()
                    elapsed = now - fpsTimer
                    if elapsed >= 1.0:
                        fps = frameCount / elapsed
                        # Latency against depthai's own clock — msg timestamps share an
                        # epoch with dai.Clock.now(), not with host time.monotonic().
                        try:
                            latencyMs = (dai.Clock.now()
                                         - msg.getTimestamp()).total_seconds() * 1000.0  # type: ignore[union-attr]
                        except Exception:
                            latencyMs = 0.0
                        self.pushStatus(True, fps=fps, latency=latencyMs,
                                        streams=streamNames)
                        frameCount = 0
                        fpsTimer   = now
                        tdThread.SetProgressSafe(  # type: ignore[attr-defined]
                            min(fps / config['fps'], 1.0) if config['fps'] else 0.5)

                # Skip decoding streams nobody consumes (no binding, never pulled via
                # GetFrame) — the queue is still drained so the device never backs up.
                wanted = self._wantedStreams  # whole-set reassign on the main thread → atomic read
                if wanted is not None and name not in wanted:
                    continue

                # Custom per-stream decode wins: turn the (model-specific) message into
                # plain, thread-safe data. Anything without a parse falls through to the
                # built-in decoders (the QoL path for rgb/depth/tof/detections/imu).
                # Read self._parsers fresh each message so ReloadHandlers() can hot-swap them
                # (whole-dict reassign on the main thread → atomic, no torn read) mid-run.
                parseFn = self._parsers.get(name)
                if parseFn is not None:
                    try:
                        data = parseFn(msg)
                    except Exception as exc:
                        self.setError(f'parse[{name}]: {exc}')
                        continue
                    if data is not None:
                        self.pushFrame(name, data)
                    continue

                # A queue's message type never changes in practice, so classify once per
                # stream and dispatch through a dict lookup thereafter.
                kind = kinds.get(name)
                if kind is None:
                    kind = self.classifyMessage(msg)
                    kinds[name] = kind
                try:
                    data = self._decoders[kind](msg, name)
                except Exception as exc:
                    self.setError(f'decode[{type(msg).__name__}]: {exc}')
                    continue
                if data is not None:
                    self.pushFrame(name, data)

            if not got:
                # Sleep only when a full pass found nothing: bursts drain at full speed,
                # while an idle pipeline doesn't spin thousands of wakeups per second.
                time.sleep(POLL_IDLE_SLEEP_S)

    # ── Built-in decoders (WORKER thread, uniform (msg, name) signature) ──
    # Zero-config defaults keyed off the message type, not the stream name. A per-stream
    # Handler `parse` always wins over these — they are defaults, not a fixed contract.

    def classifyMessage(self, msg: object) -> str:
        """WORKER thread. Pick the built-in decoder kind for a stream's messages. Called once
        per stream (on its first message) and memoized by the pump — a queue's message type
        never changes in practice. depthai-nodes types are checked first: ImgDetectionsExtended
        also exposes .detections, so it must be recognised before the raw duck-type checks."""
        global _NODE_IMPORT_ERROR
        types = _nodeMessageTypes()
        if _NODE_IMPORT_ERROR:
            self.setError(_NODE_IMPORT_ERROR)  # broken install ≠ not installed — surface it
            _NODE_IMPORT_ERROR = None
        for kind, clsName in _NODE_KINDS:
            cls = types.get(clsName)
            if cls is not None and isinstance(msg, cls):
                return kind
        if hasattr(msg, 'detections'):  return 'detections'
        if hasattr(msg, 'packets'):     return 'imu'
        if hasattr(msg, 'getPoints'):   return 'pointcloud'
        if hasattr(msg, 'getCvFrame'):  return 'image'
        if hasattr(msg, 'getFrame'):    return 'raw'
        if hasattr(msg, 'map'):         return 'map'
        return 'unknown'

    def decodeDetections(self, msg: object, name: str) -> list[PoseEntry] | list[DetectionEntry]:
        dets = msg.detections  # type: ignore[attr-defined]
        # Pose models (YOLOv8-pose) carry per-detection keypoints; plain object detectors
        # return empty keypoints. Route on that so one branch serves both without depending
        # on the stream name.
        if dets and dets[0].getKeypoints2f():
            edges: list[list[int]] = [list(e) for e in dets[0].getEdges()]
            return [
                PoseEntry(
                    confidence= round(d.confidence, 3),
                    box=        [d.xmin, d.ymin, d.xmax, d.ymax],
                    keypoints=  [[p.x, p.y] for p in d.getKeypoints2f()],
                    edges=      edges,
                )
                for d in dets
            ]
        return [
            DetectionEntry(
                label=      d.label,
                confidence= round(d.confidence, 3),
                xmin=       d.xmin,
                ymin=       d.ymin,
                xmax=       d.xmax,
                ymax=       d.ymax,
            )
            for d in dets
        ]

    def decodeImu(self, msg: object, name: str) -> ImuData | None:
        if not msg.packets:  # type: ignore[attr-defined]
            return None
        p   = msg.packets[-1]  # type: ignore[attr-defined]
        acc = p.acceleroMeter if hasattr(p, 'acceleroMeter') else p.accelerometer
        return ImuData(
            accel= [acc.x, acc.y, acc.z],
            gyro=  [p.gyroscope.x, p.gyroscope.y, p.gyroscope.z],
        )

    def decodePointCloud(self, msg: object, name: str) -> np.ndarray:
        # getPoints() may return (N, 3) or a flat (N*3,) array depending on the
        # depthai build (3.6.1 returns flat) — normalise to (N, 3) first.
        pts: np.ndarray = np.asarray(msg.getPoints(),  # type: ignore[attr-defined]
                                     dtype=np.float32).reshape(-1, 3)
        # Organized cloud → reshape to the sensor grid so it's TOP-ready
        # and pixel-aligned with depth/amplitude (flipud to match their origin).
        w: int = int(msg.getWidth())  if hasattr(msg, 'getWidth')  else 0  # type: ignore[attr-defined]
        h: int = int(msg.getHeight()) if hasattr(msg, 'getHeight') else 0  # type: ignore[attr-defined]
        if w * h != pts.shape[0] and self._lastImageHW is not None:
            # getWidth/getHeight unreliable on some builds → fall back to the
            # companion ToF image grid (depth/amplitude share the sensor res).
            ih, iw = self._lastImageHW
            if ih * iw == pts.shape[0]:
                h, w = ih, iw
        if w > 0 and h > 0 and w * h == pts.shape[0]:
            pts = np.flipud(pts.reshape(h, w, 3))
        return np.ascontiguousarray(pts)

    def decodeImage(self, msg: object, name: str) -> np.ndarray:
        frame: np.ndarray = np.flipud(msg.getCvFrame())  # type: ignore[attr-defined]
        if name == 'rgb':
            frame = frame[..., ::-1]
        self._lastImageHW = (frame.shape[0], frame.shape[1])
        return np.ascontiguousarray(frame)

    def decodeRawFrame(self, msg: object, name: str) -> np.ndarray:
        return np.ascontiguousarray(msg.getFrame())  # type: ignore[attr-defined]

    def decodeMap(self, msg: object, name: str) -> np.ndarray:
        # depthai_nodes Map2D (monocular depth, heatmaps, etc.) → float32 (H, W).
        # flipud to match the image streams' TD origin so it overlays the rgb.
        arr: np.ndarray = np.asarray(msg.map, dtype=np.float32)  # type: ignore[attr-defined]
        if arr.ndim >= 2:
            self._lastImageHW = (arr.shape[0], arr.shape[1])
        return np.ascontiguousarray(np.flipud(arr))

    def decodeUnknown(self, msg: object, name: str) -> None:
        return None  # unrecognised message type — drained but not decoded

    def decodeExtendedDetections(self, msg: object, name: str) -> list[ExtendedDetectionEntry]:
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

    def decodeKeypoints(self, msg: object, name: str) -> list[list[float]]:
        return [[float(k.x), float(k.y), float(getattr(k, 'confidence', 1.0))]
                for k in msg.keypoints]                 # type: ignore[attr-defined]

    def decodeSegmentationMask(self, msg: object, name: str) -> np.ndarray:
        # int class-index map, -1 = background → clamp to 0 and coerce to a TOP-friendly
        # uint. flipud to match the image streams' TD origin so it overlays the rgb.
        mask = np.asarray(msg.mask)                     # type: ignore[attr-defined]
        mask = np.where(mask < 0, 0, mask).astype(np.uint16)
        if mask.ndim >= 2:
            self._lastImageHW = (mask.shape[0], mask.shape[1])
        return np.ascontiguousarray(np.flipud(mask))

    def decodeClassifications(self, msg: object, name: str) -> list[ClassificationEntry]:
        return [ClassificationEntry(label=str(c), score=round(float(s), 4))
                for c, s in zip(msg.classes, msg.scores)]  # type: ignore[attr-defined]

    def decodeLines(self, msg: object, name: str) -> list[list[float]]:
        return [[float(ln.start_point.x), float(ln.start_point.y),
                 float(ln.end_point.x),   float(ln.end_point.y),
                 float(getattr(ln, 'confidence', 1.0))]
                for ln in msg.lines]                    # type: ignore[attr-defined]

    def decodeClusters(self, msg: object, name: str) -> list[list[list[float]]]:
        return [[[float(p.x), float(p.y)] for p in cl.points]
                for cl in msg.clusters]                 # type: ignore[attr-defined]

    def cacheIntrinsics(self, device: object) -> None:
        result: dict[str, dict[str, float]] = {}
        with suppress(Exception):
            calib = device.readCalibration()  # type: ignore[union-attr]
            for feat in device.getConnectedCameraFeatures():  # type: ignore[union-attr]
                key = str(feat.socket).split('.')[-1]
                with suppress(Exception):
                    mat: list[list[float]] = calib.getCameraIntrinsics(feat.socket)
                    result[key] = {
                        'fx': mat[0][0],
                        'fy': mat[1][1],
                        'cx': mat[0][2],
                        'cy': mat[1][2],
                        # native sensor resolution the coefficients refer to — consumers
                        # scale by streamRes/native when reprojecting other resolutions
                        'width':  float(getattr(feat, 'width', 0)),
                        'height': float(getattr(feat, 'height', 0)),
                    }
        with self._lock:
            self._intrinsics = result

    def buildPipeline(self, pipeline: _Pipeline, config: PipelineConfig,
                       ) -> tuple[dict[str, _OutputQueue], list[str], dict[str, _InputQueue]]:
        """No TD objects. Populates pipeline with nodes, returns ({name: out_queue}, [stream_names], {name: in_queue})."""
        import depthai as dai

        if config['pipelineScript']:
            namespace: dict[str, object] = {'dai': dai}
            # compile() with the DAT's path so a build() traceback points at the real DAT,
            # not '<string>'.
            exec(compile(config['pipelineScript'],
                         config['pipelinePath'] or '<pipeline>', 'exec'), namespace)
            buildFn = namespace.get('build')
            if not buildFn:
                raise RuntimeError(
                    "pipeline.py must define: def build(pipeline) -> ({name: queue}, [stream_names])")
            result = buildFn(pipeline)  # type: ignore[operator]
            if len(result) == 3:
                queues, streams, controls = result
            else:
                queues, streams = result
                controls = {}
            return queues, streams, controls

        queues:   dict[str, _OutputQueue] = {}
        streams:  list[str]               = []
        controls: dict[str, _InputQueue]  = {}

        device       = pipeline.getDefaultDevice()  # type: ignore[attr-defined]
        camFeatures  = device.getConnectedCameraFeatures()
        primaryTypes = [
            f.supportedTypes for f in camFeatures
            if f.socket == dai.CameraBoardSocket.CAM_A
        ]
        isTof = bool(primaryTypes and dai.CameraSensorType.TOF in primaryTypes[0])

        if isTof:
            tofNode = pipeline.create(dai.node.ToF).build(dai.CameraBoardSocket.CAM_A)  # type: ignore[attr-defined,union-attr]
            queues['depth'] = tofNode.depth.createOutputQueue()
            streams.append('depth')
            controls['tof_cfg'] = tofNode.tofBaseInputConfig.createInputQueue(maxSize=1, blocking=False)  # type: ignore[attr-defined]
        else:
            cam    = pipeline.create(dai.node.Camera).build()
            rgbOut = cam.requestOutput(DEFAULT_RGB_RESOLUTION, type=dai.ImgFrame.Type.NV12)
            queues['rgb'] = rgbOut.createOutputQueue()
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
                ], reportRate=IMU_REPORT_RATE_HZ)
                imu.setBatchReportThreshold(IMU_BATCH_THRESHOLD)
                imu.setMaxBatchReports(IMU_MAX_BATCH_REPORTS)
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
        block's Progress par, matched by the block's Model name. Stops scanning the sequence
        once every model reports complete."""
        if self._modelsDone:
            return
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
        if all(f >= 1.0 for f in progress.values()):
            self._modelsDone = True

    def updateStreamMenus(self, streamNames: list[str]) -> None:
        """Update the Stream menu on block[0]; TD propagates to all other blocks.

        Menu pars store their selection by INDEX, so rewriting the menu silently re-maps
        every block's selection (a block set to a stream of another pipeline would clamp
        to whatever now sits at that index). Guard against that: the menu is the union of
        the active streams and every block's current selection, and each block's value is
        re-assigned by name after the rewrite so no selection ever drifts."""
        if streamNames == self._lastStreamNames:
            return
        self._lastStreamNames = list(streamNames)
        seq = getattr(self.ownerComp.seq, 'Streams', None)
        if seq is None or len(seq) == 0:
            return
        current = [seq.blockPars.Stream[i].eval() for i in range(len(seq))]
        menu = list(dict.fromkeys(list(streamNames) + [c for c in current if c]))
        seq.blockPars.Stream[0].menuNames  = menu
        seq.blockPars.Stream[0].menuLabels = menu
        for i, name in enumerate(current):
            if name:
                seq.blockPars.Stream[i].val = name

    def rebuildBindings(self) -> None:
        """Main thread. Cache the (stream, op) push targets from the Streams sequence and the
        runtime registry — walking the sequence pars every cook is measurable, so pushToScriptops
        iterates this cache instead. Rebuilt immediately by register/reload calls and every
        MAINTENANCE_PERIOD cooks (catching direct par edits). Also derives the wanted-stream set
        the worker uses to skip decoding streams nobody consumes (the Fps readout is unaffected:
        the worker meters its clock stream before the wanted-filter)."""
        bindings: list[tuple[str, OP]] = []
        seq = getattr(self.ownerComp.seq, 'Streams', None)
        if seq is not None:
            for i in range(len(seq)):
                streamName: str       = seq.blockPars.Stream[i].eval()
                targetOp:   OP | None = seq.blockPars.Scriptop[i].eval()
                if streamName and targetOp is not None:
                    bindings.append((streamName, targetOp))
        for streamName, opList in self._scriptopRegistry.items():
            for targetOp in opList:
                if (streamName, targetOp) not in bindings:
                    bindings.append((streamName, targetOp))
        self._bindings = bindings
        wanted = {name for name, _ in bindings}
        with self._lock:
            # Streams with a registered parse count as consumed too — a handler was
            # deliberately attached, so its stream must keep decoding.
            wanted |= self._pullMarks | set(self._parsers)
            self._wantedStreams = wanted

    def pushToScriptops(self) -> None:
        """Auto-push the latest frame to every bound scriptop (cached — see rebuildBindings).
        Script TOPs and Script CHOPs support 'Modify Outside of Cook', so we push into them
        from here. Script SOPs have no such toggle — geometry must be built inside the cook,
        so a SOP pulls instead: call FillScriptOp(me) from its onCook."""
        for streamName, targetOp in self._bindings:
            if not targetOp.valid:
                self._bindingsDirty = True  # op deleted — rebuild the cache next tick
                continue
            self.autoPush(streamName, targetOp, self._fillers.get(targetOp.path))

    def autoPush(self, streamName: str, targetOp: OP, filler: FillFn | None) -> None:
        with self._lock:
            data    = self._frames.get(streamName)
            version = self._frameVersions.get(streamName, 0)
        if data is None:
            return
        # Skip identical re-pushes: only write when the worker has published a new frame
        # since this op last got one (camera fps below the cook rate makes that most cooks).
        verKey = (targetOp.path, streamName)
        if self._pushedVersions.get(verKey) == version:
            return
        self._pushedVersions[verKey] = version
        if filler is not None:
            # Script TOP/CHOP ('Modify Outside of Cook'), DATs, and Script POPs (their
            # staging API works outside the cook) can all be written from here. Script
            # SOPs cannot — geometry must be built inside the cook, so a SOP pulls via
            # FillScriptOp(me) from its onCook instead.
            if targetOp.family == 'SOP':
                return
            modPar = getattr(targetOp.par, 'modoutsidecook', None)
            if modPar is not None and not modPar.eval():
                modPar.val = True
            try:
                filler(targetOp, data)
            except Exception as exc:
                self.setError(f'fill[{streamName}]: {exc}')
            return
        # No custom fill: built-in defaults — numpy image -> Script TOP, float point
        # array -> Script POP (its staging API works outside the cook, no toggle needed).
        # Non-array defaults (detections/pose/imu) and Script CHOP/SOPs are pulled, not pushed.
        if isinstance(targetOp, scriptTOP) and isinstance(data, np.ndarray):
            if not targetOp.par.modoutsidecook.eval():
                targetOp.par.modoutsidecook = True
            targetOp.copyNumpyArray(self.prepareTopFrame(data))
            return
        if targetOp.family == 'POP' and self.isPointArray(data):
            try:
                self.fillPopPoints(targetOp, data)  # type: ignore[arg-type]
            except Exception as exc:
                self.setError(f'fill[{streamName}]: {exc}')

    def isPointArray(self, data: FrameData | None) -> bool:
        """A float (N, 3) or organized (H, W, 3) array — point positions, as produced by
        the built-in pointcloud decode. Image frames are uint8/uint16, so the float dtype
        keeps an rgb/depth stream bound to a POP from being misread as geometry."""
        return (isinstance(data, np.ndarray) and data.ndim in (2, 3)
                and data.shape[-1] == 3 and np.issubdtype(data.dtype, np.floating))

    def fillPopPoints(self, scriptop: OP, pts: np.ndarray) -> None:
        """Main thread. Default fill for a Script POP bound to a pointcloud stream:
        point positions -> P attribute. Organized (H, W, 3) clouds are flattened;
        invalid points ride along as zeros so the cloud stays pixel-aligned with its
        companion depth stream. For filtering/color, use a Handler DAT fill instead."""
        staging = scriptop.createGeometryStagingData()
        staging.createPointAttrib(name='P')
        flat = np.ascontiguousarray(pts.reshape(-1, 3), dtype=np.float32)
        # Bulk buffer copy — appendPointAttribValues(tolist()) walks 307k Python tuples
        # per frame on a full ToF cloud; the numpy copy is two orders of magnitude cheaper.
        copyFn = getattr(staging, 'copyNumpyArrayToPointAttrib', None)
        if copyFn is None or not copyFn('P', flat):
            staging.appendPointAttribValues('P', flat.tolist())  # pre-POP-API builds
        scriptop.buildGeometry(staging)

    def prepareTopFrame(self, frame: np.ndarray) -> np.ndarray:
        """Main thread. Shape a decoded array for scriptTOP.copyNumpyArray without avoidable
        allocations: 2D → (H, W, 1) view; uint8 RGB → RGBA via a cached per-resolution scratch
        buffer (copyNumpyArray copies synchronously, so reusing the buffer across pushes and
        streams is safe)."""
        if frame.ndim == 2:
            return frame.reshape(frame.shape[0], frame.shape[1], 1)
        if frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8:
            key = (frame.shape[0], frame.shape[1])
            buf = self._topScratch.get(key)
            if buf is None:
                buf = np.empty((key[0], key[1], 4), dtype=np.uint8)
                buf[:, :, 3] = 255
                self._topScratch[key] = buf
            buf[:, :, :3] = frame
            return buf
        return np.ascontiguousarray(frame)

    # ── Thread Manager hooks (main thread) ────────────────────────────────

    def datContentHash(self, dat: OP | None) -> int | None:
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
        # Throttled maintenance: hashing the pipeline DAT + every Handler DAT and walking the
        # Streams sequence are per-cook costs that exist only to catch rare edits — every
        # MAINTENANCE_PERIOD cooks (~0.5 s) is plenty, and register/reload calls rebuild the
        # binding cache immediately anyway.
        self._refreshTick += 1
        if self._refreshTick >= MAINTENANCE_PERIOD or self._bindingsDirty:
            self._refreshTick   = 0
            self._bindingsDirty = False
            if self.ownerComp.par.Autoreload.eval():
                datRef: str = self.ownerComp.par.Pipelinedat.eval()
                if datRef:
                    dat = op(datRef)
                    version = self.datContentHash(dat)
                    if dat and version != self._datVersion:
                        self._datVersion = version
                        self.Rebuild()           # pipeline build() changed → full restart
                        return
                # Streams blocks / bindings / Handler DATs changed → hot-reload only, no device restart.
                handlersVersion = self.handlersContentHash()
                if handlersVersion != self._handlersVersion:
                    self._handlersVersion = handlersVersion
                    self.ReloadHandlers()
            self.rebuildBindings()

        status = self.GetStatus()
        par    = self.ownerComp.par
        connected = 1 if status['connected'] else 0
        fps       = round(status['fps'], 1)
        streams   = ', '.join(status['streams'])
        # Compare before set — a par write per cook is wasted work when nothing changed.
        if par.Connected.eval() != connected:
            par.Connected = connected
        if par.Fps.eval() != fps:
            par.Fps = fps
        if par.Activestreams.eval() != streams:
            par.Activestreams = streams
        if par.Lasterror.eval() != status['error']:
            par.Lasterror = status['error']
        self.updateStreamMenus(status['streams'])
        self.updateModelProgress()
        self.pushToScriptops()

    def closeDevice(self, pipeline: _Pipeline | None, device: _Device | None) -> None:
        """WORKER thread (cameraPump's finally). Release one run's pipeline + device handles.
        They arrive as per-run locals — never read from instance state — so a dying run can
        only ever close its own device. Safe on a half-built pipeline."""
        if pipeline is not None:
            with suppress(Exception):
                pipeline.stop()
            # Join the pipeline's host-side threads BEFORE touching the device: closing
            # while its nodes are still processing is a native race that can hard-crash
            # the process on a plain Stop.
            with suppress(Exception):
                pipeline.wait()
        if device is not None:
            with suppress(Exception):
                device.close()
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
        # A cancelled task can hand the hook a bare None — that's not an error message.
        errorMsg: str = str(args[0]) if args and args[0] is not None else ''
        self.pushStatus(False)

        if self._workerShared['stop'].is_set():
            # Stop was pressed while the pipeline was failing — honor it, no retry.
            if errorMsg:
                self.setError(errorMsg)
            return
        if not errorMsg:
            errorMsg = 'Unknown error'

        self._retryCount += 1
        if self._retryCount >= MAX_RETRIES:
            self.setError(f'Pipeline failed after {MAX_RETRIES} attempts — '
                          f'press Start to retry. Last error: {errorMsg}')
            return

        self.setError(errorMsg)
        delayFrames: int = int(RECONNECT_DELAY_S * self.ownerComp.time.rate)
        run('args[0].doStart()', self, delayFrames=delayFrames, fromOP=self.ownerComp)

    # ── Helpers ───────────────────────────────────────────────────────────

    def pullHandlerFill(self, handler: OP | str, streamName: str) -> FillFn | None:
        """Main thread. Resolve and exec a pull-mode Handler DAT (FillScriptOp's `handler`
        argument), cached by content hash so a per-cook call costs one hash of the DAT text.
        A parse(msg) the handler defines is installed as the stream's worker-side decoder
        (idempotent, same slot component registrations use, so it survives seq reloads).
        Returns the handler's fill, or None (missing DAT / exec error / fill-less handler)."""
        dat = op(handler) if isinstance(handler, str) else handler
        if dat is None or not dat.valid:
            self.setError(f'pull[{streamName}]: handler DAT {handler!r} not found')
            return None
        version = hash(dat.text)
        cached = self._pullHandlers.get(dat.path)
        if cached is not None and cached[0] == version:
            parseFn, fillFn = cached[1], cached[2]
        else:
            loaded = self.execHandler(dat, f'pull:{dat.path}')
            if loaded is None:
                # Exec failed (reported via setError). Cache the failure under this hash
                # so a broken DAT isn't re-exec'd (and re-reported) every single cook.
                self._pullHandlers[dat.path] = (version, None, None)
                return None
            parseFn, fillFn = loaded
            self._pullHandlers[dat.path] = (version, parseFn, fillFn)
        if parseFn is not None and self._parsers.get(streamName) is not parseFn:
            self._componentParsers[streamName] = parseFn
            with self._lock:  # build-then-swap: the worker reads the dict lock-free
                self._parsers = {**self._parsers, streamName: parseFn}
        return fillFn

    def execHandler(self, handlerDat: OP, label: str) -> tuple[ParseFn | None, FillFn | None] | None:
        """Main thread. Exec a Handler DAT and return its optional (parse, fill) callables,
        or None when the DAT fails to exec (reported via setError). Compiled with the DAT's
        path as filename so a handler traceback points at the real DAT, not '<string>'.
        Besides dai/np, `oak` (the OAK component) is in scope so a fill can reach the
        promoted API (GetIntrinsics, GetStatus, ...) without guessing paths — fill runs
        inside the extension, where a bare op('...') would resolve inside the component."""
        import depthai as dai
        namespace: dict[str, object] = {'dai': dai, 'np': np, 'oak': self.ownerComp}
        try:
            exec(compile(handlerDat.text, handlerDat.path, 'exec'), namespace)
        except Exception as exc:
            self.setError(f'handler[{label}]: {exc}')
            return None
        parseFn = namespace.get('parse')
        fillFn  = namespace.get('fill')
        return (parseFn if callable(parseFn) else None,   # type: ignore[return-value]
                fillFn  if callable(fillFn)  else None)

    def readConfig(self) -> PipelineConfig:
        par = self.ownerComp.par
        datRef: str = par.Pipelinedat.eval()
        dat = op(datRef) if datRef else None
        rawDevice: str = par.Device.eval().strip()
        device: str = '' if rawDevice in ('', 'None') else rawDevice
        # Loglevel is an optional custom par (older comps may not have it) — default 'warn'
        # silences depthai's periodic [info] chatter (temperatures, cpu usage, ...).
        logPar = getattr(par, 'Loglevel', None)
        logLevel: str = str(logPar.eval()) if logPar is not None else 'warn'
        # All TD reads happen here on the main thread; the worker (which can't touch TD)
        # gets everything via this config dict.
        projFolder: str = project.folder
        venvSub = ('Scripts', 'python.exe') if os.name == 'nt' else ('bin', 'python')
        return PipelineConfig(
            device=         device,
            fps=            int(par.Fps.eval()),
            logLevel=       logLevel,
            enableDepth=    bool(par.Enabledepth.eval()),
            enableImu=      bool(par.Enableimu.eval()),
            pipelineScript= dat.text if dat else None,
            pipelinePath=   dat.path if dat else '',
            models=         [],  # filled in startWhenIdle via readPipelineModels()
            localModels=    [],  # filled in startWhenIdle via localModels()
            cacheDir=       os.path.join(projFolder, '.depthai_cached_models'),
            venvPython=     os.path.join(projFolder, '.venv', *venvSub),
        )

    def readPipelineModels(self, config: PipelineConfig) -> list[str]:
        """Main thread. Statically extract the models a pipeline script declares via a
        module-level ``MODELS = [...]`` (a bare ``MODEL = '...'`` is also accepted). The script
        is parsed with ast — never exec'd — so the scan has no side effects, can't run
        module-level code a second time, and needs no namespace seeding. Name references
        between module-level assignments (``MODELS = [MODEL_A, MODEL_B]``) are resolved;
        anything computed is skipped (build() then just resolves it inline)."""
        script = config['pipelineScript']
        if not script:
            return []
        try:
            tree = ast.parse(script)
        except SyntaxError as exc:
            self.setError(f'models[scan]: {exc}')
            return []
        env: dict[str, object] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                value   = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
                value   = node.value
            else:
                continue
            if not targets or value is None:
                continue
            try:
                resolved = self.resolveLiteral(value, env)
            except ValueError:
                continue  # computed value — not statically resolvable
            for name in targets:
                env[name] = resolved
        decl = env.get('MODELS', env.get('MODEL'))
        if isinstance(decl, str):
            decl = [decl]
        if not isinstance(decl, (list, tuple)):
            return []
        return [str(m) for m in decl if m]

    def resolveLiteral(self, node: ast.expr, env: dict[str, object]) -> object:
        """Literal value of an ast expression, resolving Name references from previously
        assigned module-level values. Raises ValueError for anything non-literal."""
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise ValueError(node.id)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.resolveLiteral(e, env) for e in node.elts]
        return ast.literal_eval(node)

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
        parsers: dict[str, ParseFn] = {}
        fillers: dict[str, FillFn]  = {}
        seq = getattr(self.ownerComp.seq, 'Streams', None)
        if seq is None:
            return parsers, fillers
        try:
            seq.blockPars.Handler
            hasHandler = True
        except Exception:
            hasHandler = False  # Handler param not added on this build → defaults only
        for i in range(len(seq)):
            streamName: str = seq.blockPars.Stream[i].eval()
            handlerDat = seq.blockPars.Handler[i].eval() if hasHandler else None
            if not streamName or handlerDat is None:
                continue
            loaded = self.execHandler(handlerDat, f'{streamName}#{i}')
            if loaded is None:
                continue  # exec failed (reported) — this block keeps the built-in defaults
            parseFn, fillFn = loaded
            if parseFn is not None:
                parsers[streamName] = parseFn
            regOp = seq.blockPars.Scriptop[i].eval()
            if fillFn is not None and regOp is not None:
                fillers[regOp.path] = fillFn
        return parsers, fillers
