from __future__ import annotations

import os
os.environ.setdefault('DEPTHAI_INSTALL_SIGNAL_HANDLER', '0')
os.environ.setdefault('DEPTHAI_CRASHDUMP', '0')
os.environ.setdefault('DEPTHAI_RECONNECT_TIMEOUT', '0')  # XLink auto-reconnect races Stop teardown and aborts the process
# The auto-injected AutoCalibration node is switched off per pipeline in runPipeline
# (pipeline.setAutoCalibrationMode) — DEPTHAI_AUTOCALIBRATION=OFF is only honored when set
# before depthai is first loaded, so it can't be relied on from here.

import math
import threading
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Callable, Protocol, TypedDict

import numpy as np
# TouchDesigner's own module — this file is a regular package module (not a DAT), so the
# TD globals a DAT gets for free are imported explicitly. Only importable inside TD.
from td import op, project, run, scriptTOP

from TDOAK.core import compat as coreCompat
from TDOAK.core import decode as coreDecode
from TDOAK.core import env as coreEnv
from TDOAK.core import device as coreDevice
from TDOAK.core import models as coreModels
from TDOAK.core import state as coreState
from TDOAK.core.errors import (ErrorLog, PipelineError, Tail, formatTaskError, hint,
                               summarizeCrashDump, withHint)
from TDOAK.core.retry import RetryPolicy
from TDOAK.core.watchdog import FrameWatchdog

if TYPE_CHECKING:
    from td import COMP, OP, Par, TOP


POST_CLOSE_SETTLE_S  = 3.0    # hold the alive-gate after device.close() returns — XLink keeps
                              # tearing down in native code after the call, and reopening the
                              # device inside that window can hard-crash the process
MAX_RETRIES          = 3      # start attempts in a row before giving up (Reconnect = limited)
HEALTHY_RUN_S        = 30.0   # a run that streamed this long resets the retry counter
RETRY_BACKOFF_S      = (5.0, 10.0, 20.0, 40.0, 60.0)  # delay before retry 1, 2, ... (last repeats)
FPS_IDLE_RESET_S     = 1.5    # clock stream silent this long -> Fps readout drops to 0
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


class DetectionEntry(TypedDict):   # spatial detections add x, y, z (millimetres)
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
    accel:   list[float]        # latest reading [x, y, z], m/s^2
    gyro:    list[float]        # latest reading [x, y, z], rad/s
    samples: list[list[float]]  # every packet of the batch: [ax, ay, az, gx, gy, gz, t]


class StatusInfo(TypedDict):
    connected:  bool
    fps:        float
    latency_ms: float
    error:      str               # why the pipeline is not running ('' = no error)
    warning:    str               # a problem the pipeline survives ('' = none)
    streams:    list[str]
    status:     str               # lifecycle text, same as the Status par
    stream_age: dict[str, float]  # seconds since each stream's last message


class PipelineConfig(TypedDict):
    device:         str
    firstFrameTimeout: float    # seconds from start to the first frame before the run fails (0 = off)
    stallTimeout:   float       # seconds without any frame before the run fails (0 = off)
    crashDump:      bool        # read a pending device crash dump on connect
    logLevel:       str         # depthai verbosity: trace/debug/info/warn/error/critical
    autoCalibration: str        # depthai AutoCalibration mode: off/onstart/continuous
    enableDepth:    bool
    enableImu:      bool
    pipelineScript: str | None
    pipelinePath:   str         # DAT path of the pipeline script, for traceback attribution
    models:         list[str]   # everything the script declares (module-level MODELS): slugs + local paths
    localModels:    list[str]   # subset of models that resolve to a local file → load directly, no zoo prefetch
    projectFolder:  str         # project.folder — handed to the pipeline script as PROJECT_FOLDER
    cacheDir:       str         # DEPTHAI_ZOO_CACHE_PATH shared by the prefetch subprocess + in-proc build


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


# Python host nodes (depthai-nodes' ParsingNeuralNetwork and its parsers, any
# dai.node.ThreadedHostNode / HostNode subclass) created during the current run, kept alive
# for the life of that run — see installHostNodeKeepAlive. Per worker thread.
_hostNodes = threading.local()


def installHostNodeKeepAlive(dai: object) -> bool:
    """WORKER thread. Work around Python host nodes losing their identity.

    Seen with depthai 3.12.2 inside TouchDesigner 2025.33230 (Python 3.11.15), not with the
    same depthai under a stock Python: pipeline.create(<Python subclass of ThreadedHostNode>)
    hands back a bare depthai.node.ThreadedHostNode. The Python instance is dropped before
    create() returns, so depthai re-wraps the C++ node as its base type — every
    depthai-nodes pipeline then dies with "'ThreadedHostNode' object has no attribute
    'build'".

    The breakage is PROBED first (one throw-away host-only pipeline); only when it is
    present is Pipeline.create wrapped, and the wrapper does one thing: hold a reference to
    each Python host node while create() runs and for the rest of the run, so depthai
    finds the live instance. Returns True when the workaround is active."""
    create = dai.Pipeline.create  # type: ignore[attr-defined]
    if getattr(create, '_tdoakKeepAlive', False):
        return True
    base = dai.node.ThreadedHostNode  # type: ignore[attr-defined]

    class _Probe(base):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__()

        def run(self) -> None:
            pass

    probe = dai.Pipeline(False)  # type: ignore[attr-defined]
    healthy = isinstance(probe.create(_Probe), _Probe)
    del probe
    if healthy:
        return False

    def keepAliveCreate(self: object, cls: object, *args: object, **kwargs: object) -> object:
        isPythonHostNode = (isinstance(cls, type) and issubclass(cls, base)
                            and not cls.__module__.startswith('depthai.')
                            and cls.__module__ != 'depthai')
        if not isPythonHostNode:
            return create(self, cls, *args, **kwargs)
        made: list[object] = []
        ownInit = cls.__dict__.get('__init__')
        inherited = cls.__init__

        def capturingInit(obj: object, *a: object, **k: object) -> None:
            made.append(obj)
            inherited(obj, *a, **k)

        cls.__init__ = capturingInit  # type: ignore[misc]
        try:
            node = create(self, cls, *args, **kwargs)
        finally:
            if ownInit is None:
                del cls.__init__
            else:
                cls.__init__ = ownInit  # type: ignore[misc]
        keep = getattr(_hostNodes, 'nodes', None)
        if keep is not None:
            keep.extend(made)
        return node if isinstance(node, cls) or not made else made[0]

    keepAliveCreate._tdoakKeepAlive = True  # type: ignore[attr-defined]
    dai.Pipeline.create = keepAliveCreate   # type: ignore[attr-defined]
    return True


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
        self._errors:        ErrorLog             = ErrorLog()  # guarded by self._lock
        self._lastTraceback: str                  = ''
        self._activeStreams: list[str]            = []
        self._status:        str                  = 'Idle'
        self._deviceInfo:    str                  = ''
        self._streamAges:    dict[str, float]     = {}
        self._streamedFor:   float                = 0.0  # seconds the last run delivered frames

        self._retry: RetryPolicy = RetryPolicy(
            'limited', MAX_RETRIES, HEALTHY_RUN_S, RETRY_BACKOFF_S)
        # Bumped by every Start/Stop. A delayed start (retry, Rebuild) carries the token it
        # was scheduled with and stands down when it no longer matches — so Stop during a
        # retry wait really cancels the retry.
        self._startToken: int = 0
        # Shared worker state ('alive', current 'stop' Event, pump thread, owner), fetched
        # from a process-lifetime registry in the TDOAK package so it SURVIVES extension
        # re-inits — a fresh instance must still see (and be able to stop) a previous
        # instance's still-running worker, or a Start right after reinit races the old
        # device teardown (native crash territory).
        self._workerShared: dict[str, object] = coreState.workerState(ownerComp.path)
        self._workerAlive: threading.Event = self._workerShared['alive']  # type: ignore[assignment]
        # A pump from a previous instance is still running (extension re-init mid-run, e.g.
        # the extension DAT was edited). Its frames and hooks belong to the old instance,
        # so the component would be half-connected: stop that run and start a fresh one
        # under this instance — a re-init behaves like Rebuild.
        # The restart intent is kept in the shared state ('resume'): TouchDesigner can
        # initialize the extension more than once for one DAT reload, and only the last
        # instance survives to do the restart.
        inherited: bool = coreState.isWorkerAlive(self._workerShared)
        if inherited and not self.stopEvent().is_set():
            coreState.requestResume(self._workerShared)
        coreState.claim(self._workerShared, self)
        if inherited:
            self.stopEvent().set()
        resume: bool = coreState.resumeRequested(self._workerShared)
        # True from Start until Stop: the user wants a pipeline running. Drives the
        # restart after a re-init and the Autoreload restart after a failed build.
        self._wantRunning: bool = resume
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
        self.ownerComp.par.Lastwarning = ''
        self.ownerComp.par.Deviceinfo = ''
        self.ownerComp.par.Status = 'Idle'
        self.setDeviceMenu([])
        self.updateAbout()
        if resume:
            self.setStatus('Extension reloaded - restarting the pipeline...')
            self._startRequested = True
            run('args[0].delayedStart(args[1])', self, self._startToken,
                delayFrames=1, fromOP=self.ownerComp)
        run('args[0].maintenanceTick()', self,
            delayFrames=MAINTENANCE_PERIOD, fromOP=self.ownerComp)

    def onDestroyTD(self) -> None:
        """TouchDesigner is destroying this instance: re-init, component deleted, project
        closing or TD quitting. Never leave a pump running behind a dead extension — it
        would keep the device open with nobody able to stop it. If the user wanted the
        pipeline running, a restart is requested for whichever instance initializes next
        (a re-init); the request expires on its own when none does."""
        if not self.isOwner():
            return  # a newer instance already took over and dealt with the run
        if self._wantRunning:
            coreState.requestResume(self._workerShared)
        self._startRequested = False
        self._startToken += 1
        self.stopEvent().set()

    def maintenanceTick(self) -> None:
        """Main thread, every MAINTENANCE_PERIOD frames for the life of this instance —
        independent of the pump, whose RefreshHook only runs while a run is alive. Keeps
        Autoreload working exactly when it is needed most: after a build error there is no
        pump, and fixing the Pipeline DAT must still restart the pipeline."""
        if not self.isOwner() or not self.ownerComp.valid:
            return  # superseded or destroyed — let the chain end
        par = self.ownerComp.par
        if (self._wantRunning and not self._startRequested and par.Autoreload.eval()
                and not coreState.isWorkerAlive(self._workerShared)):
            dat = par.Pipelinedat.eval()
            if dat is not None and self.datContentHash(dat) != self._datVersion:
                self._retry = self.newRetryPolicy()
                self._startToken += 1  # cancel a retry that is still waiting
                self.setError('')      # the script was edited: its old error is history
                self.setStatus('Pipeline DAT changed - restarting...')
                self.doStart()
        run('args[0].maintenanceTick()', self,
            delayFrames=MAINTENANCE_PERIOD, fromOP=self.ownerComp)

    def stopEvent(self) -> threading.Event:
        """The CURRENT run's stop flag (a fresh Event is published per run)."""
        return self._workerShared['stop']  # type: ignore[return-value]

    def isOwner(self) -> bool:
        """False once a newer extension instance has taken over the component. Delayed
        callbacks and ThreadManager hooks of a superseded instance must stand down."""
        return coreState.isOwner(self._workerShared, self)

    def setDeviceMenu(self, devices: list[DeviceInfo]) -> None:
        """Main thread. Rebuild the Device menu WITHOUT losing the selection. Menu pars
        store their selection by index, so a plain rewrite silently re-points Device at
        whatever lands on the old index (or at 'first available' when the menu shrinks).
        The current selection is kept in the menu even when the device isn't visible right
        now, and re-assigned by name afterwards."""
        par     = self.ownerComp.par.Device
        current = par.eval()
        names:  list[str] = ['None'] + [d['mxid'] for d in devices]
        labels: list[str] = ['-- Select Device --'] + [
            f"{d['name'] or d['mxid']}  [{d['protocol']}]" for d in devices]
        if current and current not in names:
            names.append(current)
            labels.append(f'{current}  [not found]')
        par.menuNames  = names
        par.menuLabels = labels
        par.val = current if current else 'None'

    def updateAbout(self) -> None:
        """Main thread. Fill the About page's read-only version fields. Version comes from
        the installed TDOAK package (the .tox ships inside it, so they match). The depthai
        version is read from sys.modules only — never imported here, so init stays cheap;
        it fills in once enumeration or a Start has loaded depthai."""
        import sys
        import TDOAK
        par = self.ownerComp.par
        par.Version = str(TDOAK.__version__)
        dai = sys.modules.get('depthai')
        if dai is not None:
            par.Depthaiversion = coreCompat.depthaiStatus(
                str(getattr(dai, '__version__', 'unknown')))

    # ── Public API (main thread only) ────────────────────────────────────

    def Start(self) -> None:
        """Start the pipeline. A no-op (reported in Status) while one is already running —
        use Rebuild to restart."""
        self._retry = self.newRetryPolicy()
        self._startToken += 1
        self._wantRunning = True
        self.setError('')
        self.clearWarning()
        self.doStart()

    def newRetryPolicy(self) -> RetryPolicy:
        """A fresh failure counter with the current Reconnect setting — every deliberate
        (re)start begins one: Start, Rebuild, and the Autoreload restart."""
        return RetryPolicy(self.ownerComp.par.Reconnect.eval(),
                           MAX_RETRIES, HEALTHY_RUN_S, RETRY_BACKOFF_S)

    def StartFromUi(self) -> None:
        """The Start pulse: look for devices first, so 'nothing plugged in' is reported
        instead of failing somewhere inside depthai. While a pump is alive the enumeration
        is skipped (the device is in use; Start reports 'already running' or waits)."""
        if coreState.isWorkerAlive(self._workerShared):
            self.Start()
            return
        self.setStatus('Looking for devices...')
        self.ListDevices(callback=self.onDevicesForStart)

    def onDevicesForStart(self, devices: list[DeviceInfo]) -> None:
        # A specific Device selection is always tried, found or not (a PoE camera on
        # another subnet is invisible to discovery but reachable) — depthai then reports
        # it. Only 'first available' with nothing visible is a certain dead end.
        if not devices and not self.selectedDevice():
            self.setError('No OAK device found - check the cable, power, or (PoE) that the '
                          'camera is on this subnet')
            self.setStatus('Idle')
            return
        self.Start()

    def doStart(self) -> None:
        """Request a start. Defers (re-polling) while a previous worker is still winding
        down — device.close() can block for seconds, and racing it with a second pump would
        let the dying run close the new run's device."""
        self._startRequested = True
        self.startWhenIdle()

    def delayedStart(self, token: int) -> None:
        """A start scheduled earlier with run() (retry, Rebuild, re-init restart). Stands
        down when a Start/Stop happened in between or a newer instance owns the component."""
        if token != self._startToken or not self.isOwner():
            return
        self.doStart()

    def startWhenIdle(self) -> None:
        if not self._startRequested or not self.isOwner():
            return  # Stop() cancelled the pending start while we waited
        if coreState.isWorkerAlive(self._workerShared):
            if not self.stopEvent().is_set():
                # A pump is already running (or booting) and nobody asked it to stop —
                # Start is a no-op, NOT a restart (that's Rebuild). Critically, do NOT
                # touch its stop flag here: a concurrent start chain may have just booted
                # this pump, and killing it from the deferral path livelocks both chains.
                self._startRequested = False
                self.setStatus('Already running - use Rebuild to restart')
                return
            self.setStatus('Waiting for the previous run to close...')
            run('args[0].startWhenIdle()', self,
                delayFrames=RESTART_POLL_FRAMES, fromOP=self.ownerComp)
            return
        self._startRequested = False
        coreState.cancelResume(self._workerShared)
        # Fresh Event per run, published in the shared slot: the old worker keeps polling
        # ITS flag (which stays set), so a new Start can never un-stop a run that is still
        # winding down — and Stop() from ANY instance (even after reinit) reaches this run.
        self._workerShared['stop'] = threading.Event()
        self.setStatus('Starting...')
        self._topScratch.clear()
        config = self.readConfig()
        if os.environ.get('DEPTHAI_ZOO_CACHE_PATH') != config['cacheDir']:
            os.environ['DEPTHAI_ZOO_CACHE_PATH'] = config['cacheDir']
        # Seed the Autoreload baseline to the DAT we're starting with. Otherwise _datVersion
        # stays at its None sentinel and the first maintenance tick reads a real hash != None
        # as an edit, firing a spurious Rebuild() a moment after every Start.
        self._datVersion = self.datContentHash(self.ownerComp.par.Pipelinedat.eval())
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
        models = config['models']
        local  = self.localModels(models)
        config['localModels'] = list(local)
        # Local NNArchive/blob models need no download — seed them at 1.0 so the sequence shows
        # them ready immediately; zoo slugs start at 0.0 and fill as the prefetch streams.
        with self._lock:
            self._modelProgress = {m: (1.0 if m in local else 0.0) for m in models}
            self._streamedFor   = 0.0
            self._streamAges    = {}
            self._deviceInfo    = ''
            self._lastTraceback = ''
        self._modelsDone = False
        self.setupModelBlocks(models)
        task = self.ThreadManager.TDTask(
            target=self.cameraPump,
            args=(config, self.stopEvent()),
            SuccessHook=self.onSuccess,
            ExceptHook=self.onExcept,
            RefreshHook=self.onRefresh,
        )
        # The gate: set here, cleared by the worker in cameraPump's finally. The thread is
        # recorded next to it so isWorkerAlive() can tell a live pump from a stale flag.
        self._workerShared['thread'] = None
        self._workerAlive.set()
        try:
            self._workerShared['thread'] = self.ThreadManager.EnqueueTask(task, standalone=True)
        except Exception:
            # EnqueueTask can raise AFTER it started the thread (it writes pars last), so
            # only open the gate again when no pump thread actually exists.
            if self._workerShared['thread'] is None and not any(
                    t.name == 'cameraPump' and t.is_alive() for t in threading.enumerate()):
                self._workerAlive.clear()
            raise

    def Stop(self) -> None:
        self._startRequested = False  # cancel any start still waiting on the old worker
        self._startToken += 1         # ... and any retry / Rebuild start scheduled with run()
        self._wantRunning = False
        coreState.cancelResume(self._workerShared)
        self.stopEvent().set()
        self.setStatus('Stopping...' if coreState.isWorkerAlive(self._workerShared) else 'Stopped')

    def Rebuild(self) -> None:
        """Stop and start again — the way to apply changed settings to a running pipeline."""
        self.Stop()
        self._retry = self.newRetryPolicy()
        self._wantRunning = True
        self.setError('')
        self.clearWarning()
        run('args[0].delayedStart(args[1])', self, self._startToken,
            delayFrames=REBUILD_DELAY_FRAMES, fromOP=self.ownerComp)

    def ReloadHandlers(self) -> None:
        """Re-load the Streams blocks' Handler DATs and hot-swap parse/fill into the running
        pipeline — no device/pipeline restart. Use after adding or editing a Streams block,
        its Scriptop, or its Handler DAT. Adding a brand-new *physical* stream still needs a
        Rebuild (that's a pipeline build() change). Safe to call when stopped — it just primes
        state for the next Start."""
        # A successful reload proves the handler problems are fixed — drop their warnings
        # first; a handler that still fails reports again right away.
        self.clearWarning(('handler[', 'fill[', 'parse[', 'pull['))
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
        # Never scan while a device is booting or tearing down (pump alive, not yet/no
        # longer streaming): XLink discovery inside that window races the same native
        # teardown the alive-gate protects. The last known list is served instead.
        with self._lock:
            connected = self._connected
        if coreState.isWorkerAlive(self._workerShared) and not connected:
            if callback:
                callback(self.GetDevices())
            return
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

    def CheckForUpdate(self) -> None:
        """Main thread. Check GitHub for a newer TDOAK release and report it in the About
        page's Update Status. Check-only — never installs. The HTTP request runs on a
        ThreadManager worker so a slow network can't stall TD's main thread."""
        import TDOAK
        from TDOAK.core import update
        statusPar = self.ownerComp.par.Updatestatus
        current = str(TDOAK.__version__)
        devCheckout = update.isDevCheckout(TDOAK.__file__)
        result: dict[str, str] = {}  # per-call closure state — overlapping checks can't clash

        def work() -> None:  # WORKER thread — no TD access
            try:
                result['status'] = update.describe(current, update.fetchLatest(), devCheckout)
            except Exception as exc:
                result['status'] = 'Check failed: ' + update.reason(exc)

        def done() -> None:  # main thread
            statusPar.val = result.get('status', 'Check failed: no result')

        statusPar.val = 'Checking for updates...'
        self.ThreadManager.EnqueueTask(self.ThreadManager.TDTask(
            target=work, SuccessHook=done, ExceptHook=lambda *a: done()))

    # Pulse routing: the component's parexec DAT dispatches every custom pulse to
    # OnPulse<Parname>, so a new pulse par only needs a method here — no parexec edit.
    def OnPulseStart(self, par: Par) -> None:          self.StartFromUi()
    def OnPulseStop(self, par: Par) -> None:           self.Stop()
    def OnPulseRebuild(self, par: Par) -> None:        self.Rebuild()
    def OnPulseRefresh(self, par: Par) -> None:        self.ListDevices()
    def OnPulseReloadhandlers(self, par: Par) -> None: self.ReloadHandlers()
    def OnPulseCheckupdate(self, par: Par) -> None:    self.CheckForUpdate()

    def OnPulseHelp(self, par: Par) -> None:
        import webbrowser
        webbrowser.open('https://github.com/TouchDesigner/TDOAK')

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
        seq = self.ownerComp.seq.Streams
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
                self.setWarning(f'fill[{name}]: {exc}')
                return False
            return True
        if isinstance(scriptop, scriptTOP) and isinstance(data, np.ndarray):
            scriptop.copyNumpyArray(self.prepareTopFrame(data))
            return True
        if scriptop.family == 'POP' and self.isPointArray(data):
            try:
                self.fillPopPoints(scriptop, data)  # type: ignore[arg-type]
            except Exception as exc:
                self.setWarning(f'fill[{name}]: {exc}')
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
        except Exception as exc:
            self.setWarning(f'control[{name}]: {exc}')
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
        except Exception as exc:
            self.setWarning(f'control[{name}]: {exc}')
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
                error=      self._errors.text('error'),
                warning=    self._errors.text('warning'),
                streams=    list(self._activeStreams),
                status=     self._status,
                stream_age= dict(self._streamAges),
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
        """Record why the pipeline is NOT running (failed start, lost device, stall) — shown
        in Lasterror with the time it happened. Kept until the next Start/Rebuild."""
        self.report('error', msg)

    def setWarning(self, msg: str) -> None:
        """Record a problem the pipeline survives (a handler raised, a stream went quiet, a
        USB2 link, a failed model download) — shown in Lastwarning. Latest one wins."""
        self.report('warning', msg)

    def clearWarning(self, prefixes: tuple[str, ...] = ()) -> None:
        """Clear Lastwarning — with prefixes, only if the current warning starts with one."""
        with self._lock:
            self._errors.clear('warning', prefixes)
        if threading.current_thread() is self._mainThread:
            self.mirrorErrors()

    def report(self, kind: str, msg: str) -> None:
        """The single choke point every error/warning funnels through. Thread-safe: a
        worker just records state for onRefresh to mirror; on the main thread the pars are
        also written immediately, because onRefresh only runs while a pump task is alive
        (errors while stopped must still surface)."""
        with self._lock:
            self._errors.set(kind, msg)
        if threading.current_thread() is self._mainThread:
            self.mirrorErrors()

    def mirrorErrors(self) -> None:
        """Main thread. Write both channels to their pars and print every message that has
        not been printed before (once per distinct text — a per-frame handler error
        neither floods the textport nor hides behind the parameter)."""
        with self._lock:
            error   = self._errors.display('error')
            warning = self._errors.display('warning')
            pending = self._errors.takePending()
        par = self.ownerComp.par
        if par.Lasterror.eval() != error:
            par.Lasterror = error
        if par.Lastwarning.eval() != warning:
            par.Lastwarning = warning
        for kind, message in pending:
            print(f'TDOAK {kind} ({self.ownerComp.path}): {message}')

    def GetLastTraceback(self) -> str:
        """Full Python traceback of the last failed run ('' if none) — the detail behind
        the one-line Lasterror."""
        with self._lock:
            return self._lastTraceback

    def setStatus(self, text: str) -> None:
        """Record the lifecycle state shown in the Status par ('Starting...', 'Streaming',
        'Retrying in 10 s', ...). Thread-safe, same split as setError: a worker records it
        for onRefresh to mirror; the main thread also writes the par immediately."""
        with self._lock:
            self._status = text
        if threading.current_thread() is self._mainThread:
            self.ownerComp.par.Status = text

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
        self.setDeviceMenu(devices)
        self.updateAbout()  # enumeration has loaded depthai by now
        if callback:
            callback(devices)

    def onEnumerateExcept(self, *args: object) -> None:
        errorMsg: str = (formatTaskError(args[0]) if args else '') or 'Unknown error'
        self.pushDevices([])
        self.setError(f'devices[enumerate]: {errorMsg}')
        if self._status == 'Looking for devices...':
            self.setStatus('Idle')

    def downloadModels(self, config: PipelineConfig, platform: str,
                       stopFlag: threading.Event) -> bool:
        """WORKER thread. Pre-fetch each declared model on a child interpreter so the
        download holds its OWN GIL, not TD's. Returns True if every model exited cleanly
        (so the caller can trust the cache and skip the online check)."""
        import subprocess

        # Local models are loaded straight from disk in the script's build() — nothing to fetch,
        # so just mark them complete. Only zoo slugs go through getModelFromZoo.
        local = set(config['localModels'])
        for model in local:
            self.setModelProgress(model, 1.0)
        zooModels = [m for m in config['models'] if m not in local]
        if not zooModels:
            return True

        # The interpreter of the environment depthai was loaded from (else <project>/.venv).
        # Never TouchDesigner's own sys.executable: that can be the application binary.
        import depthai as dai
        python = coreEnv.findPython(getattr(dai, '__file__', None), config['projectFolder'])
        if python is None:
            self.setWarning('download: no Python interpreter with depthai found for the model '
                            'pre-fetch (looked in the environment depthai was loaded from and '
                            'in <project>/.venv) - models are fetched in-process instead, '
                            'which can freeze the UI during the download')
            return False
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
        tail = Tail()  # last output lines of the child — the reason when it fails
        drain = threading.Thread(target=self.drainDownloadProgress,
                                 args=(proc.stdout, model, tail), daemon=True)
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
        reason = tail.text() or f'exit {proc.returncode}'
        advice = hint(tail.full())
        self.setWarning(f'download[{model}]: {reason} (interpreter: {python})'
                        + (f' - {advice}' if advice else ''))
        return False

    def drainDownloadProgress(self, stream: object, model: str, tail: Tail) -> None:
        """Prefetch helper thread. Parse depthai's JSON progress lines
        ({"download_now", "download_total", "log_type": "download"}, emitted ~every 100 ms)
        into a 0..1 fraction. Best-effort: exits on EOF or when fetchModel closes the pipe."""
        import json
        try:
            for line in stream:  # type: ignore[attr-defined]
                line = line.strip()
                if not line.startswith('{'):
                    tail.add(line)  # depthai logs / a Python traceback — kept for the error
                    continue
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
        # Everything the teardown needs is bound to LOCALS up front, and the gate is
        # cleared in the innermost finally: nothing that goes wrong during teardown — not
        # even this module's globals vanishing under a DAT reload — may leave 'alive' set,
        # because a stuck gate turns every later Start into a no-op.
        alive  = self._workerAlive
        sleep  = time.sleep
        settle = POST_CLOSE_SETTLE_S
        close  = self.closeDevice
        failure: str = ''
        try:
            self.runPipeline(config, stopFlag, handles)
        except Exception as exc:
            # Keep the MESSAGE, drop the traceback: ThreadManager stores the exception it
            # is handed, and a traceback pins runPipeline's frame — its pipeline, device and
            # queues — until the main thread lets go of it. Their native destructors would
            # then run on the main thread, possibly after a retry reopened the device.
            # A fresh PipelineError is raised below, after the teardown, instead.
            import traceback
            failure = formatTaskError(exc) or type(exc).__name__
            with self._lock:
                self._lastTraceback = traceback.format_exc()
            traceback.clear_frames(exc.__traceback__)
        finally:
            try:
                # The run is over the moment the loop exits: don't keep reporting
                # Connected/Fps through the seconds the teardown below takes.
                self.pushStatus(False)
                self.setStatus('Closing device...')
                opened = handles['device'] is not None or handles['pipeline'] is not None
                close(handles['pipeline'], handles['device'])  # type: ignore[arg-type]
                _hostNodes.nodes = None
                handles.clear()  # last references die HERE, on the worker
                if opened:
                    # device.close() returns before XLink finishes tearing down natively —
                    # keep the alive-gate held so no deferred Start can reopen into that
                    # window (observed to hard-crash TD).
                    sleep(settle)
            finally:
                alive.clear()
        if failure:
            raise PipelineError(failure)

    def runPipeline(self, config: PipelineConfig, stopFlag: threading.Event,
                    handles: dict[str, object]) -> None:
        import depthai as dai

        self.setStatus('Opening device...')
        if config['device']:
            device = dai.Device(dai.DeviceInfo(config['device']))
            handles['device'] = device
            pipeline = dai.Pipeline(device)
        else:
            pipeline = dai.Pipeline()
        handles['pipeline'] = pipeline

        if config['crashDump']:
            # Opt-in (Crashdump par): if the device firmware crashed during an earlier
            # connection, it kept a dump — read its summary now, on a fresh connection.
            # (depthai's own post-close dump check stays disabled via DEPTHAI_CRASHDUMP=0:
            # that one reconnects DURING teardown and has crashed the host process.)
            try:
                dev = pipeline.getDefaultDevice()  # type: ignore[attr-defined]
                if dev.hasCrashDump():
                    summary = summarizeCrashDump(dev.getCrashDump())
                    if summary:
                        self.setWarning(summary)
            except Exception as exc:
                self.setWarning(f'crashdump: {exc}')

        # Pre-fetch declared models on a subprocess (its own interpreter = its own GIL) so the
        # download stays off TD's main thread and we can stream a live progress counter. Then
        # point the in-process build at the same cache and skip the online freshness check, so
        # build() loads the cached archive instead of doing a GIL-holding network round-trip.
        platform = pipeline.getDefaultDevice().getPlatformAsString()  # type: ignore[attr-defined]
        if any(m not in set(config['localModels']) for m in config['models']):
            self.setStatus('Downloading models...')
        prefetched = self.downloadModels(config, platform, stopFlag)
        # Only trust the cache (skip the online check) when we actually pre-fetched a zoo slug.
        # An all-local pipeline leaves the check on so any incidental zoo resolve still works.
        zooDeclared = any(m not in set(config['localModels']) for m in config['models'])
        # NOTE: process-global environment — this affects every depthai user in the TD process
        # and assumes a single OAK component per project (or identical settings across them).
        # The cache path is set on the main thread before the run starts (startWhenIdle);
        # this one depends on the prefetch result, so it is written here — but only when
        # it actually changes: setenv is not thread-safe against native getenv calls.
        internetCheck = '0' if (zooDeclared and prefetched) else '1'
        if os.environ.get('DEPTHAI_ZOO_INTERNET_CHECK') != internetCheck:
            os.environ['DEPTHAI_ZOO_INTERNET_CHECK'] = internetCheck
        if stopFlag.is_set():
            return  # Stop pressed mid-download — don't build/start; cameraPump's finally closes the device

        # depthai auto-injects AutoCalibration/DynamicCalibration/Gate nodes into any pipeline
        # with a StereoDepth node (library default ON_START). On an OAK-D Lite (RVC2) with
        # depthai 3.12.2 that crashes the device firmware ~2 s after start() (RTEMS assert in
        # RefCnt::DecRef on a PlgXLinkIn thread) — every stream drops with 'MessageQueue was
        # closed' before the first frame. So the Autocalibration par defaults to Off. It is
        # applied before build(), so a pipeline script can still override it in build().
        modes = getattr(dai.Pipeline, 'AutoCalibrationMode', None)
        if modes is not None and hasattr(pipeline, 'setAutoCalibrationMode'):
            mode = {'off':        getattr(modes, 'OFF', None),
                    'onstart':    getattr(modes, 'ON_START', None),
                    'continuous': getattr(modes, 'CONTINUOUS', None),
                    }.get(config['autoCalibration'], getattr(modes, 'OFF', None))
            if mode is not None:
                try:
                    pipeline.setAutoCalibrationMode(mode)  # type: ignore[attr-defined]
                except Exception as exc:
                    self.setWarning(f"autocalibration: could not set '{config['autoCalibration']}' "
                                    f'({exc}) - depthai uses its default (On Start), which '
                                    'crashes some devices right after start')
        elif config['autoCalibration'] != 'onstart':
            self.setWarning('autocalibration: this depthai version has no per-pipeline switch - '
                            'the Auto Calibration parameter has no effect')

        self.setStatus('Starting pipeline...')
        # Python host nodes built by the script must outlive build(): they are parked in
        # the run's handles and released with them in the teardown.
        _hostNodes.nodes = handles.setdefault('hostNodes', [])
        keepAlive = installHostNodeKeepAlive(dai)
        queues, streamNames, controlQueues, startedFn = self.buildPipeline(pipeline, config)
        if keepAlive and handles['hostNodes']:
            self.setWarning('depthai host-node workaround active: this TouchDesigner/depthai '
                            'combination drops Python host nodes (depthai-nodes) - TDOAK keeps '
                            'them alive')
        self.pushStatus(False, streams=streamNames)

        pipeline.start()
        if startedFn is not None:
            startedFn(pipeline, pipeline.getDefaultDevice())  # type: ignore[attr-defined]
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
        self.readDeviceInfo(pipeline.getDefaultDevice())   # type: ignore[attr-defined]
        with self._lock:
            self._controlQueues = controlQueues
        # Connected means FRAMES ARE ARRIVING, not 'pipeline.start() returned': a device
        # that accepts the pipeline and then sends nothing must not look healthy. It turns
        # on with the first message below; until then the watchdog's first-frame timer runs.
        self.setStatus('Waiting for the first frame...')
        watchdog = FrameWatchdog(config['firstFrameTimeout'], config['stallTimeout'], queues)
        firstFrameAt: float | None = None
        lastClockAt:  float        = time.monotonic()
        lastAgesAt:   float        = lastClockAt
        fpsShown:     float        = 0.0

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
                watchdog.mark(name)
                if firstFrameAt is None:
                    firstFrameAt = time.monotonic()
                    self.pushStatus(True, streams=streamNames)
                    self.setStatus('Streaming')

                if name == clockStream:
                    frameCount += 1
                    now     = time.monotonic()
                    lastClockAt = now
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
                        fpsShown   = fps
                        frameCount = 0
                        fpsTimer   = now
                        tdThread.SetProgressSafe(1.0)  # type: ignore[attr-defined]

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
                        self.setWarning(f'parse[{name}]: {exc}')
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
                    if kind == 'unknown':
                        self.setWarning(f"stream '{name}': no built-in decoder for "
                                        f'{type(msg).__name__} messages - attach a Handler DAT '
                                        'with a parse(msg) to this stream')
                try:
                    data = self._decoders[kind](msg, name)
                except Exception as exc:
                    self.setWarning(f'decode[{type(msg).__name__}]: {exc}')
                    continue
                if data is not None:
                    self.pushFrame(name, data)

            if not got:
                now = time.monotonic()
                if firstFrameAt is not None:
                    with self._lock:
                        self._streamedFor = now - firstFrameAt
                # The Fps readout must not freeze at its last value when frames stop.
                if fpsShown and now - lastClockAt >= FPS_IDLE_RESET_S:
                    fpsShown = 0.0
                    self.pushStatus(firstFrameAt is not None, streams=streamNames)
                if now - lastAgesAt >= 1.0:
                    lastAgesAt = now
                    with self._lock:
                        self._streamAges = watchdog.ages()
                    for silent in watchdog.silentStreams():
                        self.setWarning(f"stream '{silent}' stopped delivering while others "
                                      'still flow - check its sensor/node (a starved input '
                                      'also blocks everything behind a Sync node)')
                # No frames at all (never, or not any more) fails the run: it goes through
                # onExcept, so it is reported and retried like any other pipeline error.
                problem = watchdog.check()
                if problem:
                    raise PipelineError(problem)
                # Sleep only when a full pass found nothing: bursts drain at full speed,
                # while an idle pipeline doesn't spin thousands of wakeups per second.
                time.sleep(POLL_IDLE_SLEEP_S)

        if firstFrameAt is not None:
            with self._lock:
                self._streamedFor = time.monotonic() - firstFrameAt
        if not stopFlag.is_set():
            # The loop ended although nobody pressed Stop: pipeline.isRunning() turned
            # False by itself. That is a lost device or a firmware crash, never a clean
            # stop — fail the run so it is reported and retried instead of ending silently.
            raise PipelineError('pipeline stopped by itself - the device was lost or its '
                                'firmware crashed')

    def readDeviceInfo(self, device: object) -> None:
        """WORKER thread. Name, platform and link speed of the connected device for the
        Deviceinfo readout. A USB device below SuperSpeed is called out: a USB2 cable or
        port is the most common cause of stalls that look like firmware crashes."""
        name = platform = protocol = ''
        usbSpeed: str | None = None
        with suppress(Exception):
            name = str(device.getDeviceName())  # type: ignore[attr-defined]
        with suppress(Exception):
            platform = str(device.getPlatformAsString())  # type: ignore[attr-defined]
        with suppress(Exception):
            protocol = coreDevice.enumName(device.getDeviceInfo().protocol)  # type: ignore[attr-defined]
        if 'USB' in protocol.upper():
            with suppress(Exception):
                usbSpeed = coreDevice.enumName(device.getUsbSpeed())  # type: ignore[attr-defined]
        text, warning = coreDevice.describeDevice(name, platform, protocol, usbSpeed)
        with self._lock:
            self._deviceInfo = text
        if warning:
            self.setWarning(warning)

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
            self.setWarning(_NODE_IMPORT_ERROR)  # broken install ≠ not installed — surface it
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
        # Pose models carry per-detection keypoints (-> PoseEntry), plain detectors don't
        # (-> DetectionEntry); spatial detections add x/y/z in millimetres. Routed on the
        # message content, never on the stream name.
        return coreDecode.decodeDetections(msg.detections)  # type: ignore[attr-defined,return-value]

    def decodeImu(self, msg: object, name: str) -> ImuData | None:
        return coreDecode.decodeImu(msg.packets)  # type: ignore[attr-defined,return-value]

    def decodePointCloud(self, msg: object, name: str) -> np.ndarray:
        # Organized cloud -> the sensor grid, TOP-ready and pixel-aligned with
        # depth/amplitude; the companion image grid is the fallback when the message's own
        # width/height are unreliable.
        w: int = int(msg.getWidth())  if hasattr(msg, 'getWidth')  else 0  # type: ignore[attr-defined]
        h: int = int(msg.getHeight()) if hasattr(msg, 'getHeight') else 0  # type: ignore[attr-defined]
        return coreDecode.organizePointCloud(msg.getPoints(), w, h, self._lastImageHW)  # type: ignore[attr-defined]

    def decodeImage(self, msg: object, name: str) -> np.ndarray:
        # getCvFrame() is OpenCV-style: top-down, BGR. TouchDesigner wants bottom-up RGB —
        # for EVERY colour stream, whatever it is called (mono/depth frames only flip).
        frame = coreDecode.toTdImage(msg.getCvFrame())  # type: ignore[attr-defined]
        self._lastImageHW = (frame.shape[0], frame.shape[1])
        return frame

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
            box, angle = coreDecode.rotatedRectBox(d.rotated_rect)
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
                       ) -> tuple[dict[str, _OutputQueue], list[str], dict[str, _InputQueue],
                                  Callable[[object, object], None] | None]:
        """No TD objects. Populates pipeline with nodes, returns ({name: out_queue},
        [stream_names], {name: in_queue}, started-hook or None)."""
        import depthai as dai

        if config['pipelineScript']:
            # In scope for the script: dai, and PROJECT_FOLDER (the .toe's folder) — the
            # script runs on the worker, where TouchDesigner's `project` doesn't exist.
            namespace: dict[str, object] = {'dai': dai,
                                            'PROJECT_FOLDER': config['projectFolder']}
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
            # Optional: def started(pipeline, device) — called on the worker right after
            # pipeline.start(), for device calls that only work on a running pipeline
            # (IR dot projector / flood light intensity, ...).
            startedFn = namespace.get('started')
            return queues, streams, controls, (startedFn if callable(startedFn) else None)  # type: ignore[return-value]

        queues:   dict[str, _OutputQueue] = {}
        streams:  list[str]               = []
        controls: dict[str, _InputQueue]  = {}

        device       = pipeline.getDefaultDevice()  # type: ignore[attr-defined]
        camFeatures  = device.getConnectedCameraFeatures()
        bySocket     = {f.socket: f for f in camFeatures}
        camA         = bySocket.get(dai.CameraBoardSocket.CAM_A)
        isTof = camA is not None and dai.CameraSensorType.TOF in camA.supportedTypes

        if isTof:
            tofNode = pipeline.create(dai.node.ToF).build(dai.CameraBoardSocket.CAM_A)  # type: ignore[attr-defined,union-attr]
            queues['depth'] = tofNode.depth.createOutputQueue()
            streams.append('depth')
            controls['tof_cfg'] = tofNode.tofBaseInputConfig.createInputQueue(maxSize=1, blocking=False)  # type: ignore[attr-defined]
        else:
            if not camFeatures:
                raise PipelineError('this device reports no cameras - the built-in pipeline '
                                    'has nothing to stream')
            cam = pipeline.create(dai.node.Camera).build()
            # Never ask for more than the sensor has: a 1280x800 colour sensor (OAK-D SR,
            # OAK-D Pro W) can't serve the 1920x1080 default.
            sensor = bySocket.get(cam.getBoardSocket()) if hasattr(cam, 'getBoardSocket') else camA
            size = coreDecode.fitResolution(
                DEFAULT_RGB_RESOLUTION,
                (getattr(sensor, 'width', 0), getattr(sensor, 'height', 0)))
            rgbOut = cam.requestOutput(size)
            queues['rgb'] = rgbOut.createOutputQueue()
            streams.append('rgb')
            controls['cam_ctrl'] = cam.inputControl.createInputQueue(maxSize=1, blocking=False)  # type: ignore[attr-defined]

            if config['enableDepth']:
                hasPair = (dai.CameraBoardSocket.CAM_B in bySocket
                           and dai.CameraBoardSocket.CAM_C in bySocket)
                if not hasPair:
                    self.setWarning('Enable Depth ignored: this device has no stereo pair on '
                                    'CAM_B/CAM_C')
                else:
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
                imuType = ''
                with suppress(Exception):
                    imuType = str(device.getConnectedIMU())
                if imuType.strip().upper() in ('', 'NONE'):
                    self.setWarning('Enable IMU ignored: this device has no IMU')
                else:
                    imu = pipeline.create(dai.node.IMU)
                    imu.enableIMUSensor([
                        dai.IMUSensor.ACCELEROMETER_RAW,
                        dai.IMUSensor.GYROSCOPE_RAW,
                    ], reportRate=IMU_REPORT_RATE_HZ)
                    imu.setBatchReportThreshold(IMU_BATCH_THRESHOLD)
                    imu.setMaxBatchReports(IMU_MAX_BATCH_REPORTS)
                    queues['imu'] = imu.out.createOutputQueue()
                    streams.append('imu')

        return queues, streams, controls, None

    # ── Stream menu + scriptop dispatch (main thread) ────────────────────

    def setupModelBlocks(self, models: list[str]) -> None:
        """Main thread. Resize the readonly Models sequence to one block per declared model and
        fill in each Model name, resetting Progress to 0. Grow with numBlocks (the documented
        'press +' equivalent) and shrink with destroyBlock from the tail."""
        seq = self.ownerComp.seq.Models
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
        seq = self.ownerComp.seq.Models
        if len(seq) == 0:
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
        seq = self.ownerComp.seq.Streams
        if len(seq) == 0:
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
        seq = self.ownerComp.seq.Streams
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
                self.setWarning(f'fill[{streamName}]: {exc}')
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
                self.setWarning(f'fill[{streamName}]: {exc}')

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
        seq = self.ownerComp.seq.Streams
        parts: list[object] = []
        for i in range(len(seq)):
            stream   = seq.blockPars.Stream[i].eval()
            scriptop = seq.blockPars.Scriptop[i].eval()
            handler  = seq.blockPars.Handler[i].eval()
            parts.append((stream,
                          scriptop.path if scriptop is not None else None,
                          handler.text if handler is not None else None))
        return hash(tuple(parts))

    def onRefresh(self) -> None:
        if not self.isOwner():
            return  # superseded by a re-init: the new instance drives the pars now
        # Throttled maintenance: hashing the pipeline DAT + every Handler DAT and walking the
        # Streams sequence are per-cook costs that exist only to catch rare edits — every
        # MAINTENANCE_PERIOD cooks (~0.5 s) is plenty, and register/reload calls rebuild the
        # binding cache immediately anyway.
        self._refreshTick += 1
        if self._refreshTick >= MAINTENANCE_PERIOD or self._bindingsDirty:
            self._refreshTick   = 0
            self._bindingsDirty = False
            if self.ownerComp.par.Autoreload.eval():
                dat = self.ownerComp.par.Pipelinedat.eval()
                if dat is not None:
                    version = self.datContentHash(dat)
                    if version != self._datVersion:
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
        self.mirrorErrors()
        if par.Status.eval() != status['status']:
            par.Status = status['status']
        with self._lock:
            info = self._deviceInfo
        if par.Deviceinfo.eval() != info:
            par.Deviceinfo = info
        self.updateStreamMenus(status['streams'])
        self.updateModelProgress()
        self.pushToScriptops()

    def closeDevice(self, pipeline: _Pipeline | None, device: _Device | None) -> None:
        """WORKER thread (cameraPump's finally). Release one run's pipeline + device handles.
        They arrive as per-run locals — never read from instance state — so a dying run can
        only ever close its own device. Safe on a half-built pipeline."""
        # Unpublish the control queues FIRST: SendControl/SendFrame run on the main thread
        # (SendFrame every frame) and must not send into a pipeline that is tearing down.
        with self._lock:
            self._controlQueues = {}
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
            self._intrinsics = {}

    def onSuccess(self) -> None:
        # Device already released on the worker (cameraPump's finally) — this hook runs on
        # the main thread, so it must not call the blocking close itself.
        self.pushStatus(False)
        if not self.isOwner():
            return
        self.ownerComp.par.Connected = 0
        self.ownerComp.par.Fps = 0
        self.mirrorErrors()  # a warning the worker set in its last moments
        self.setStatus('Stopped')

    def onExcept(self, *args: object) -> None:
        # Device already released on the worker (cameraPump's finally), so the retry can't
        # reopen a still-open (PoE) handle. Don't close here — this hook is on the main thread.
        # A cancelled task can hand the hook a bare None — that's not an error message.
        errorMsg: str = withHint(formatTaskError(args[0])) if args and args[0] else ''
        self.pushStatus(False)
        if not self.isOwner():
            return  # superseded by a re-init — the new instance decides about restarts
        self.mirrorErrors()
        self.ownerComp.par.Connected = 0
        self.ownerComp.par.Fps = 0

        if self.stopEvent().is_set():
            # Stop was pressed while the pipeline was failing — honor it, no retry.
            if errorMsg:
                self.setError(errorMsg)
            self.setStatus('Stopped')
            return
        if not errorMsg:
            errorMsg = 'Unknown error'

        with self._lock:
            ranFor = self._streamedFor
        delay = self._retry.onFailure(ranFor)
        if delay is None:
            if self._retry.mode == 'limited':
                self.setError(f'Pipeline failed {self._retry.failures} times in a row - '
                              f'press Start to retry. Last error: {errorMsg}')
            else:
                self.setError(errorMsg)
            self.setStatus('Failed - press Start')
            return

        self.setError(errorMsg)
        self.setStatus(f'Retrying in {delay:g} s ({self._retry.describe()})')
        delayFrames: int = int(delay * self.ownerComp.time.rate)
        run('args[0].delayedStart(args[1])', self, self._startToken,
            delayFrames=delayFrames, fromOP=self.ownerComp)

    # ── Helpers ───────────────────────────────────────────────────────────

    def pullHandlerFill(self, handler: OP | str, streamName: str) -> FillFn | None:
        """Main thread. Resolve and exec a pull-mode Handler DAT (FillScriptOp's `handler`
        argument), cached by content hash so a per-cook call costs one hash of the DAT text.
        A parse(msg) the handler defines is installed as the stream's worker-side decoder
        (idempotent, same slot component registrations use, so it survives seq reloads).
        Returns the handler's fill, or None (missing DAT / exec error / fill-less handler)."""
        dat = self.ownerComp.op(handler) if isinstance(handler, str) else handler
        if dat is None or not dat.valid:
            self.setWarning(f'pull[{streamName}]: handler DAT {handler!r} not found')
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
            self.setWarning(f'handler[{label}]: {exc}')
            return None
        parseFn = namespace.get('parse')
        fillFn  = namespace.get('fill')
        return (parseFn if callable(parseFn) else None,   # type: ignore[return-value]
                fillFn  if callable(fillFn)  else None)

    def selectedDevice(self) -> str:
        """Main thread. The device to open: the hand-typed Device Address (MXID or IP — for
        a PoE camera discovery can't see) when set, else the Device menu selection.
        '' means 'first available'."""
        address: str = self.ownerComp.par.Deviceaddress.eval().strip()
        if address:
            return address
        selected: str = self.ownerComp.par.Device.eval().strip()
        return '' if selected in ('', 'None') else selected

    def readConfig(self) -> PipelineConfig:
        par = self.ownerComp.par
        dat = par.Pipelinedat.eval()
        device: str = self.selectedDevice()
        script: str | None = dat.text if dat else None
        # Module-level literals of the pipeline script, read with ast (never exec'd): the
        # models to pre-fetch, and optional watchdog overrides — a host-frame pipeline sets
        # STALL_TIMEOUT_S = 0 because it legitimately sends nothing until it is fed.
        try:
            constants = coreModels.scanConstants(script)
        except SyntaxError as exc:
            self.setWarning(f'pipeline[scan]: {exc}')
            constants = {}
        # All TD reads happen here on the main thread; the worker (which can't touch TD)
        # gets everything via this config dict.
        projFolder: str = project.folder
        return PipelineConfig(
            device=         device,
            firstFrameTimeout= coreModels.timeoutOverride(
                constants, 'FIRST_FRAME_TIMEOUT_S', par.Firstframetimeout.eval()),
            stallTimeout=   coreModels.timeoutOverride(
                constants, 'STALL_TIMEOUT_S', par.Stalltimeout.eval()),
            crashDump=      bool(par.Crashdump.eval()),
            logLevel=       str(par.Loglevel.eval()),
            autoCalibration= str(par.Autocalibration.eval()),
            enableDepth=    bool(par.Enabledepth.eval()),
            enableImu=      bool(par.Enableimu.eval()),
            pipelineScript= script,
            pipelinePath=   dat.path if dat else '',
            models=         coreModels.declaredModels(constants),
            localModels=    [],  # filled in startWhenIdle via localModels()
            projectFolder=  projFolder,
            cacheDir=       os.path.join(projFolder, '.depthai_cached_models'),
        )

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
        seq = self.ownerComp.seq.Streams
        for i in range(len(seq)):
            streamName: str = seq.blockPars.Stream[i].eval()
            handlerDat = seq.blockPars.Handler[i].eval()
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
