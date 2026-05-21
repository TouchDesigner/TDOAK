import threading
import time
import numpy as np


class OAKManagerExt:

    def __init__(self, ownerComp):
        self.ownerComp     = ownerComp
        self.ThreadManager = op.TDResources.ThreadManager

        self._lock         = threading.Lock()
        self._latestRgb    = None   # numpy uint8 BGR (H, W, 3)
        self._latestDepth  = None   # numpy uint16   (H, W)
        self._latestIrL    = None   # numpy uint8    (H, W)
        self._latestIrR    = None
        self._latestImu    = None   # dict: {accel:[x,y,z], gyro:[x,y,z]}
        self._connected    = False
        self._fps          = 0.0
        self._latencyMs    = 0.0
        self._lastError    = ''
        self._activeStreams = []

        self._reconnectDelay = 2.0
        self._stopFlag     = threading.Event()
        self._datVersion   = -1

    # ── Public API (main thread only) ────────────────────────────────────

    def Start(self):
        self._stopFlag.clear()
        config = self._readConfig()
        task = self.ThreadManager.TDTask(
            target=self._cameraPump,
            args=(config, self._stopFlag),
            SuccessHook=self._onSuccess,
            ExceptHook=self._onExcept,
            RefreshHook=self._onRefresh,
        )
        self.ThreadManager.EnqueueTask(task, standalone=True)

    def Stop(self):
        self._stopFlag.set()

    def Rebuild(self):
        self.Stop()
        run('args[0].Start()', self, delayFrames=5, fromOP=self.ownerComp)

    def OnPulseStart(self, par):    self.Start()
    def OnPulseStop(self, par):     self.Stop()
    def OnPulseRebuild(self, par):  self.Rebuild()

    # ── Getters (safe from Script TOP/CHOP cook callbacks) ───────────────

    def GetRgb(self):
        with self._lock: return self._latestRgb

    def GetDepth(self):
        with self._lock: return self._latestDepth

    def GetIrLeft(self):
        with self._lock: return self._latestIrL

    def GetIrRight(self):
        with self._lock: return self._latestIrR

    def GetImu(self):
        with self._lock: return self._latestImu

    def GetStatus(self):
        with self._lock:
            return {
                'connected':  self._connected,
                'fps':        self._fps,
                'latency_ms': self._latencyMs,
                'error':      self._lastError,
                'streams':    list(self._activeStreams),
            }

    # ── Push methods (called from background thread) ──────────────────────

    def _pushRgb(self, arr):
        with self._lock: self._latestRgb = arr

    def _pushDepth(self, arr):
        with self._lock: self._latestDepth = arr

    def _pushIrLeft(self, arr):
        with self._lock: self._latestIrL = arr

    def _pushIrRight(self, arr):
        with self._lock: self._latestIrR = arr

    def _pushImu(self, data):
        with self._lock: self._latestImu = data

    def _pushStatus(self, connected, fps=0.0, latency=0.0, error='', streams=None):
        with self._lock:
            self._connected = connected
            self._fps       = fps
            self._latencyMs = latency
            self._lastError = error
            if streams is not None:
                self._activeStreams = streams

    # ── Background thread — NO TD object access below this line ──────────

    def _cameraPump(self, config, stopFlag):
        import depthai as dai

        pipeline, streamNames = self._buildPipeline(config)
        self._pushStatus(False, streams=streamNames)

        serial     = config.get('serial', '').strip()
        deviceInfo = dai.DeviceInfo(serial) if serial else None

        with (dai.Device(pipeline, deviceInfo) if deviceInfo
              else dai.Device(pipeline)) as device:

            queues = {
                name: device.getOutputQueue(name, maxSize=4, blocking=False)
                for name in streamNames
            }
            self._pushStatus(True, streams=streamNames)

            frameCount = 0
            fpsTimer   = time.monotonic()
            tdThread   = threading.current_thread()

            while not stopFlag.is_set():
                for name, q in queues.items():
                    msg = q.tryGet()
                    if msg is None:
                        continue

                    hostTs    = time.monotonic()
                    deviceTs  = (msg.getTimestampDevice().total_seconds()
                                 if hasattr(msg, 'getTimestampDevice') else hostTs)
                    latencyMs = (hostTs - deviceTs) * 1000.0

                    if name == 'rgb':
                        self._pushRgb(msg.getCvFrame())
                        frameCount += 1
                        elapsed = hostTs - fpsTimer
                        if elapsed >= 1.0:
                            fps = frameCount / elapsed
                            self._pushStatus(True, fps=fps,
                                             latency=latencyMs, streams=streamNames)
                            frameCount = 0
                            fpsTimer   = hostTs
                            tdThread.SetProgressSafe(
                                min(fps / config['fps'], 1.0) if 'fps' in config else 0.5)

                    elif name == 'depth':
                        self._pushDepth(msg.getFrame())

                    elif name in ('ir_left', 'left'):
                        self._pushIrLeft(msg.getCvFrame())

                    elif name in ('ir_right', 'right'):
                        self._pushIrRight(msg.getCvFrame())

                    elif name == 'imu':
                        if msg.packets:
                            p = msg.packets[-1]
                            self._pushImu({
                                'accel': [p.acceleroMeter.x,
                                          p.acceleroMeter.y,
                                          p.acceleroMeter.z],
                                'gyro':  [p.gyroscope.x,
                                          p.gyroscope.y,
                                          p.gyroscope.z],
                            })

                time.sleep(0.0002)

    def _buildPipeline(self, config):
        """No TD objects. Returns (dai.Pipeline, [stream_names])."""
        import depthai as dai

        pipeline = dai.Pipeline()

        if config.get('pipelineScript'):
            namespace = {'dai': dai}
            exec(config['pipelineScript'], namespace)
            build_fn = namespace.get('build')
            if not build_fn:
                raise RuntimeError(
                    "pipeline.py must define: def build(pipeline) -> [stream_names]")
            result = build_fn(pipeline)
            if not isinstance(result, list):
                raise RuntimeError("build() must return a list of XLinkOut stream names")
            return pipeline, result

        # Default preset pipeline
        streams = []

        cam = pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setFps(config.get('fps', 30))
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName('rgb')
        cam.video.link(xout_rgb.input)
        streams.append('rgb')

        if config.get('enableDepth', True):
            left  = pipeline.create(dai.node.MonoCamera)
            right = pipeline.create(dai.node.MonoCamera)
            left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetType.HIGH_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(False)
            left.out.link(stereo.left)
            right.out.link(stereo.right)

            xout_depth = pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName('depth')
            stereo.depth.link(xout_depth.input)
            streams.append('depth')

        if config.get('enableImu', False):
            imu = pipeline.create(dai.node.IMU)
            imu.enableIMUSensor([
                dai.IMUSensor.ACCELEROMETER_RAW,
                dai.IMUSensor.GYROSCOPE_RAW,
            ], reportRate=500)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)
            xout_imu = pipeline.create(dai.node.XLinkOut)
            xout_imu.setStreamName('imu')
            imu.out.link(xout_imu.input)
            streams.append('imu')

        return pipeline, streams

    # ── Thread Manager hooks (main thread) ────────────────────────────────

    def _onRefresh(self):
        if self.ownerComp.par.Autoreload.eval():
            datRef = self.ownerComp.par.Pipelinedat.eval()
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

    def _onSuccess(self):
        self._pushStatus(False)
        self.ownerComp.par.Connected = 0

    def _onExcept(self, *args):
        errorMsg = str(args[0]) if args else 'Unknown error'
        self._pushStatus(False, error=errorMsg)
        self.ownerComp.par.Lasterror = errorMsg

        delayFrames = int(self._reconnectDelay * me.time.rate)
        self._reconnectDelay = min(self._reconnectDelay * 1.5, 30.0)
        run('args[0].Start()', self, delayFrames=delayFrames, fromOP=self.ownerComp)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _readConfig(self):
        """Read all ownerComp parameters into a plain dict. Main thread only."""
        par    = self.ownerComp.par
        datRef = par.Pipelinedat.eval()
        return {
            'serial':         par.Serial.eval(),
            'mode':           par.Mode.eval(),
            'fps':            int(par.Fps.eval()),
            'enableDepth':    bool(par.Enabledepth.eval()),
            'enableImu':      bool(par.Enableimu.eval()),
            'pipelineScript': op(datRef).text if datRef else None,
        }
