# TDOAK

TouchDesigner extension for OAK cameras via [depthai](https://docs.luxonis.com/software-v3/) v3.

> Contributing or digging into the internals? Read **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — component map, threading model, data flow, lifecycle state machine, and the crash-safety invariants behind the device teardown path.

Supports USB OAK and PoE OAK including OAK-D SR PoE (ToF depth + aligned color) with automatic camera-type detection, plus neural pipelines (object detection, human-pose/skeleton tracking). Pipeline scripts are reloadable DATs. Runtime camera controls (exposure, stereo config, ToF config) can be sent from the main thread at any time while the pipeline is running.

Stream data is decoded off the main thread and delivered into Script Operators. Common streams (rgb, depth, amplitude, detections, pose, imu…) are handled out of the box; any other model's output can be decoded and shaped with a per-stream **Handler DAT** — see [Script OP Integration](#script-op-integration).

The component watches the stream: a camera that stops sending, never starts sending, or drops off is reported in **Status** / **Last Error** within seconds and — if you want — reconnected automatically. See [Troubleshooting](#troubleshooting).

---

## Requirements

- TouchDesigner 2025.33xxx
- depthai 3.12.x and depthai-nodes 0.8.x (installed as dependencies; the **About** page says when a different depthai version is loaded)
- A USB3 cable and port for USB cameras

---

## Quick Start

1. **Create a Python environment for your project.** Save your project (.toe) first, then add TouchDesigner's Python Environment Manager (`tdPyEnvManager` from the Palette), choose *Python vEnv*, and create the environment **in the project folder with the name `.venv`**. Keep it *active* so it loads with the project.
2. **Install TDOAK into that environment** (from the environment manager's terminal, or any shell with the venv activated). The package is not on a public index, so install straight from GitHub (or from a local clone with `pip install <path>`):
    ```
    pip install git+https://github.com/TouchDesigner/TDOAK
    ```
    depthai and depthai-nodes are installed as dependencies.
3. **Add the component.** Create a Base COMP and set its *External .tox* parameter to this expression:
    ```
    mod.TDOAK.toxPath('TDOAK')
    ```
    (Dragging `TDOAK.tox` into the network works too. The component's code is the installed package — if the environment from step 1 is not active, **Last Error** says so.)
4. **Add a viewer.** Create a second Base COMP with the External .tox expression `mod.TDOAK.toxPath('TDOAKStream')`, and drag your TDOAK component onto its **OAK Component** parameter. `Stream` is `rgb` by default.
5. Press **Start** on the TDOAK component.

**Status** walks through *Looking for devices → Opening device → Starting pipeline → Waiting for the first frame → Streaming*, and the image appears in the TDOAKStream component. With no Pipeline DAT set, a built-in pipeline runs with an `rgb` stream (plus `depth`/`imu` when enabled on the Pipeline page; a ToF camera gets a `depth` stream instead).

- Several cameras attached? Press **List Devices** and pick one in **Device**. `-- Select Device --` opens whichever is found first.
- Changed a setting while streaming? Press **Rebuild** — Start does nothing while a pipeline is running.
- Nothing happens, or it stops? Read **Status**, **Last Error** and **Last Warning**, then [Troubleshooting](#troubleshooting).

The examples folder (`mod.TDOAK.examplesPath()`) has a range of pipeline scripts and handler scripts. Other ways to get stream data into operators — the **Streams** sequence, `FillScriptOp()` in your own Script OP, Handler DATs for CHOPs/POPs/DATs — are described under [Script OP Integration](#script-op-integration).

---

## Component Parameters

**Device page**

| Parameter | Type | Description |
|---|---|---|
| `Device` | Menu | The camera to open. Filled by **List Devices** (Start lists automatically). `-- Select Device --` uses the first available device. The selection is kept across project loads, also while the device is unplugged (shown as *not found*). |
| `Device Address` | String | Optional MXID or IP typed by hand — for a PoE camera on another subnet that discovery can't see. When filled in it overrides the `Device` menu. |
| `List Devices` | Pulse | Enumerate all visible OAK devices and update the Device menu (parameter name `Refresh`). |
| `Start` | Pulse | Look for devices, then start the pipeline. Does nothing while a pipeline is already running — Status says so. |
| `Stop` | Pulse | Stop the pipeline and release the device. Also cancels a pending retry. |
| `Rebuild` | Pulse | Stop and start again — applies changed settings (Device, Pipeline DAT, Tune page) to a running pipeline. |
| `Status` | String (read-only) | What the component is doing: *Idle, Looking for devices, Waiting for the previous run to close, Opening device, Downloading models, Starting pipeline, Waiting for the first frame, Streaming, Closing device, Retrying in N s (attempt 2 of 3), Failed - press Start, Stopping, Stopped*. |
| `Connected` | Toggle (read-only) | On while **frames are arriving**. Turns on with the first frame (not when the pipeline starts) and off when the run ends, stalls or fails. |
| `Fps` | Int (read-only) | Measured frame rate, updated once per second; drops to 0 when frames stop. |
| `Device Info` | String (read-only) | Model, platform and link of the connected device, e.g. `OAK-D-LITE \| RVC2 \| USB SUPER`. |
| `Active Streams` | String (read-only) | Comma-separated list of active stream names. |
| `Last Error` | String (read-only) | Why the pipeline is **not running**, with the time it happened: a failed start, a lost device, a stall, a pipeline script error (`Type: message (DAT:line in function)`). Known depthai failures carry a hint. Kept until the next Start/Rebuild, so after an automatic reconnect it shows what the interruption was. `GetLastTraceback()` returns the full traceback. |
| `Last Warning` | String (read-only) | Latest problem the pipeline **survived**, with its time: a Handler `parse`/`fill` that raised, a stream that went quiet, a USB2 link, a failed model pre-fetch, an undecodable message type. Each distinct warning is also printed once to the textport. Cleared on Start/Rebuild; handler warnings also on Reload Handlers. |
| `Models` (seq) | Sequential block (read-only) | Status readout, one block per model the pipeline declares. `Model` is the slug or local path; `Progress` (0–1) tracks the pre-fetch. Populated automatically on Start — see [Model Pre-fetch](#model-pre-fetch). |

**Pipeline page**

| Parameter | Type | Description |
|---|---|---|
| `Pipeline DAT` | DAT ref | Optional custom pipeline script. See [Pipeline Scripts](#pipeline-scripts). Leave empty for the built-in auto-detect pipeline. |
| `Auto Reload` | Toggle | When enabled, the pipeline restarts when the Pipeline DAT changes (also after a failed build — fix the script and it starts again), and handler edits hot-reload. |
| `Enable Depth` | Toggle | Built-in pipeline only: stereo depth output (ignored with a warning on devices without a CAM_B/CAM_C pair; ToF devices always stream depth). |
| `Enable IMU` | Toggle | Built-in pipeline only: IMU output (ignored with a warning on devices without an IMU). |

**Streams page**

| Parameter | Type | Description |
|---|---|---|
| `Streams` (seq) | Sequential block | Bind stream names to Script Operators. Each block has a `Stream` menu, a `Scriptop` OP reference, and an optional `Handler` DAT reference (custom `parse`/`fill` — see [Script OP Integration](#script-op-integration)). |
| `Reload Handlers` | Pulse | Hot-reload the Streams blocks' Handler DATs (`parse`/`fill`) into the **running** pipeline — applies handler/binding edits without a restart. |

**Tune page** — settings; all take effect on the next Start or Rebuild.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `First Frame Timeout` | Float | 15 | Seconds to wait for the first frame after the pipeline starts. If none arrives the run fails with an error (and `Reconnect` applies) instead of sitting at "started, no image". `0` = off. A pipeline script can override it with a module-level `FIRST_FRAME_TIMEOUT_S`. |
| `Stall Timeout` | Float | 5 | Seconds without a frame on **any** stream before a running pipeline counts as stalled and the run fails. `0` = off. Script override: `STALL_TIMEOUT_S`. A single stream going quiet while others flow is a warning, not a failure. |
| `Reconnect` | Menu | Limited | What to do when a run fails. `Off`: stay stopped. `Limited`: up to 3 start attempts in a row, then stop until Start. `Forever`: keep retrying, waiting 5, 10, 20, 40, then 60 s — for unattended installations. A run that streamed for 30 s counts as healthy and resets the count. |
| `Auto Calibration` | Menu | Off | depthai's automatic stereo self-calibration (`Off` / `On Start` / `Continuous`), which depthai injects into any pipeline with `StereoDepth`. Default `Off`: on depthai 3.12.2 it crashes OAK-D Lite firmware ~2 s after start. A pipeline script can override it in `build()` with `pipeline.setAutoCalibrationMode()`. |
| `Read Crash Dump` | Toggle | Off | On each connect, ask the device whether its firmware crashed during an earlier connection and put a one-line summary (error source, processor, crashed thread) into Last Warning. For diagnosing a pipeline that keeps dying with *MessageQueue was closed*. |
| `Log Level` | Menu | Warn | depthai verbosity (`trace`…`critical`), applied to the device on Start. `warn` silences the periodic `[info]` chatter (temperatures, cpu usage). |

**About page**

| Parameter | Type | Description |
|---|---|---|
| `Help` | Pulse | Open the TDOAK GitHub page. |
| `Check for Update` | Pulse | Ask GitHub for the latest release and report it in `Update Status`. Check only, nothing is installed: TDOAK ships as a Python package, so updating means `pip install --upgrade git+https://github.com/TouchDesigner/TDOAK@v<version>` in the TouchDesigner venv, then restarting TouchDesigner (depthai is a native module and can't be reloaded in a running process). In a source checkout it suggests `git pull` instead. While the repo is private, set `TDOAK_GITHUB_TOKEN` in the environment. |
| `Update Status` | String (read-only) | Result of the last update check. |
| `Version` | String (read-only) | Version of the installed TDOAK package — the component's code *is* that package. |
| `Depthai Version` | String (read-only) | depthai version loaded in this TouchDesigner process, with `(tested)` or a note when it is outside the range this TDOAK release was tested with; fills in on List Devices or Start. |
| `.tox Save Build` | String (read-only) | TouchDesigner build the packaged `.tox` was last saved with — stamped at release, see [Releasing](#releasing). |

### The TDOAKStream component

`mod.TDOAK.toxPath('TDOAKStream')` — a Script TOP that shows one image stream. It pulls the frame inside its own cook, so there is nothing to wire up on the TDOAK component.

| Parameter | Type | Description |
|---|---|---|
| `OAK Component` | COMP | The TDOAK component to read from. |
| `Stream` | String | Stream name (`rgb`, `depth`, `amplitude`, …). |
| `Handler` | DAT | Optional Handler DAT whose `fill` shapes the frame — e.g. `depth_handler.py` to make a depth stream visible. |

---

## Troubleshooting

Read **Status**, **Last Error** and **Last Warning** first — every failure path reports there. To see what depthai itself prints (device logs, unsupported-format messages), start TouchDesigner with the environment variable `TOUCH_TEXT_CONSOLE=1`.

| Symptom | Likely cause | What to do |
|---|---|---|
| Last Error: *TDOAK package not importable* | The Python environment with TDOAK is not active in this project. | Quick Start step 1–2: environment in the project folder, active; reopen the project. |
| Last Error: *No OAK device found* | Nothing visible on USB or the local subnet. | Cable, power, port. A PoE camera on another subnet: type its IP into **Device Address**. Close other programs that hold the camera (OAK Viewer, another TouchDesigner). |
| Start does nothing | A pipeline is already running or closing — Status says which. | **Rebuild** to restart; or wait for *Waiting for the previous run to close*. |
| Last Error: *no frames within N s of start* | The device accepted the pipeline but sends nothing: an unsupported stream configuration (e.g. an explicit frame type the camera doesn't support), or a device that crashed right after start. | Check the pipeline script; run with `TOUCH_TEXT_CONSOLE=1`. A slow pipeline can raise `First Frame Timeout` or set `FIRST_FRAME_TIMEOUT_S`. |
| Last Error: *no frames for N s* / *pipeline stopped by itself* / *MessageQueue was closed* / *X_LINK_ERROR* | The device stalled or dropped off: power dip, cable, or a firmware crash. | USB3 cable and port, short and direct (or a powered hub) — **Device Info** shows the link speed and Last Warning flags USB2. With stereo depth keep `Auto Calibration` Off. Turn on `Read Crash Dump` and Rebuild to learn whether the firmware crashed. For installations set `Reconnect` to Forever. |
| Device Info shows `USB HIGH` | USB2 link (cable or port). | Use a USB3 cable and port; high-resolution or multi-stream pipelines will not fit through USB2. |
| A depth / disparity TOP looks black | The data is fine: 16-bit millimetres (0–65535 scale) shown as-is are dark. | Use `depth_handler.py` (Handler on the stream, or on a TDOAKStream), or a Math/Level TOP. |
| Last Warning: *fill[...]* / *parse[...]* | A Handler DAT raised. | Fix the DAT; with Auto Reload the warning clears on the next successful reload. |
| Last Warning: *download[model]: …* | The model pre-fetch subprocess failed; the reason is in the message. The model is then fetched in-process (works, but the UI can hitch). On macOS a stand-alone Python cannot load depthai from TouchDesigner's environment at all, so this warning is expected there. | Usually nothing. Offline: download once while online; the cache is `.depthai_cached_models/` in the project folder. |
| Last Warning: *depthai host-node workaround active* | Informational. In TouchDesigner 2025.33230 with depthai 3.12.2, depthai drops Python host nodes (everything from depthai-nodes); TDOAK keeps them alive. | Nothing. |
| Last Warning: *no built-in decoder for … messages* | The stream carries a message type TDOAK doesn't decode. | Attach a Handler DAT with a `parse(msg)`. |
| Colours of a stream look swapped (after upgrading) | Since 0.2.0 every colour stream is delivered as RGB, not only the one named `rgb`. | Remove channel swaps you added in your own handlers. |

The full Python traceback of the last failed run: `op('TDOAK').GetLastTraceback()`. The Thread Manager's log (Palette → TDResources → threadManager → logger) has the worker-side messages.

---

## `OAKManagerExt` API Reference

All methods listed here are safe to call from the **main thread** unless otherwise noted.

---

### Lifecycle

#### `Start() -> None`
Start the camera pipeline; clears `Lasterror`/`Lastwarning` and starts a fresh failure count with the current `Reconnect` setting. A no-op (reported in `Status`) while a pipeline is already running — use `Rebuild()`. If a run fails, the `Reconnect` policy decides: `off` stays stopped; `limited` allows 3 start attempts in a row (5 s, then 10 s apart) before `Lasterror` shows the final error; `forever` keeps retrying with backoff up to 60 s. A run that streamed for 30 s resets the count.

#### `StartFromUi() -> None`
What the **Start** pulse does: enumerate devices first, report *No OAK device found* when there is none (and no explicit device is selected), otherwise `Start()`.

#### `Stop() -> None`
Signal the background pump to exit cleanly. The pipeline and device are closed **on the background thread** (the pump's `finally`), not in a main-thread hook — `device.close()`/`pipeline.stop()` are blocking depthai calls, so keeping them off the main thread avoids freezing the UI for several frames on each stop or retry.

#### `Rebuild() -> None`
Stop the pipeline and restart it once the device has closed. The way to apply changed settings to a running pipeline. Triggered automatically when `Auto Reload` is on and the Pipeline DAT changes — also after a failed build, when no pipeline is running.

**Re-initializing the extension** (the *Re-Init Extensions* pulse, or TouchDesigner doing it) while a pipeline runs behaves like `Rebuild`: the old run is stopped and a new one starts under the new instance.

#### `ReloadHandlers() -> None`
Re-load the Streams blocks' Handler DATs and hot-swap their `parse`/`fill` into the **running** pipeline — no device or pipeline restart. Call after adding or editing a Streams block, its `Scriptop`, or its `Handler` DAT. The worker reads the new `parse` on its next message; the main thread picks up the new `fill` on the next cook. Adding a brand-new **physical stream** still needs a `Rebuild()` (that's a `build()` change). Safe to call while stopped — it just primes state for the next Start. Triggered automatically when `Auto Reload` is on and a Streams block/binding/Handler changes, or manually via the `Reload Handlers` pulse.

---

### Device Discovery

#### `ListDevices(callback=None) -> None`
Enumerate all visible OAK devices in a background thread and update the `Device` parameter menu. Optionally accepts a callback that receives the device list when enumeration completes. While a device is booting or closing, no scan is made and the last known list is served.

```python
def on_devices(devices):
    for d in devices:
        print(d['name'], d['mxid'], d['protocol'])

oak.ListDevices(callback=on_devices)
```

#### `GetDevices() -> list[DeviceInfo]`
Return the last enumerated device list (no network call). Returns an empty list if `ListDevices` has not been called yet.

`DeviceInfo` fields:

| Key | Type | Description |
|---|---|---|
| `mxid` | `str` | Unique device identifier (also used as the menu name). |
| `name` | `str` | Human-readable device name. |
| `state` | `str` | Connection state, e.g. `UNBOOTED`, `BOOTLOADER`, `FLASH_BOOTED`. |
| `protocol` | `str` | Transport protocol, e.g. `USB`, `TCP`. |

---

### Frame Data

Frames are written by the background pump thread and read-locked for safe access from cook callbacks.

> Streams are decoded on demand: the worker skips decoding a stream that has no Streams-block/registry binding and has never been pulled via `GetFrame` (its queue is still drained, so the device never backs up; `rgb` is always decoded — it drives the `Fps` readout). The first `GetFrame(name)` on such a stream registers interest and may return `None` until the next frame arrives.

#### `GetFrame(name: str) -> FrameData | None`
Return the latest frame for the given stream name, or `None` if no frame has arrived yet.

`FrameData` is one of:
- `numpy.ndarray` — image, mask, or point-cloud data (see format notes below)
- `list[DetectionEntry]` — NN object-detection results (`dai.ImgDetections`)
- `list[PoseEntry]` — NN pose / skeleton results (`dai.ImgDetections` with keypoints)
- `list[ExtendedDetectionEntry]` — depthai-nodes `ImgDetectionsExtended` (rotated box + keypoints)
- `list[ClassificationEntry]` — depthai-nodes `Classifications`
- `list[list[float]]` — keypoints `[[x, y, score], …]` or lines `[[x1, y1, x2, y2, conf], …]`
- `list[list[list[float]]]` — clusters `[[[x, y], …], …]`
- `ImuData` — accelerometer + gyroscope packet
- any value returned by a custom Handler `parse` (see [Script OP Integration](#script-op-integration))

#### `GetFrameNames() -> list[str]`
Return the stream names for which at least one frame has been received.

#### Convenience getters

```python
oak.GetRgb()      # → ndarray (H, W, 3) uint8 RGB, or None
oak.GetDepth()    # → ndarray (H, W) uint16, or None
oak.GetIrLeft()   # → ndarray (H, W) uint8, or None
oak.GetIrRight()  # → ndarray (H, W) uint8, or None
oak.GetImu()      # → ImuData, or None
oak.GetPose()     # → list[PoseEntry], or None
```

**Frame formats by stream:**

| Stream | dtype | Shape | Notes |
|---|---|---|---|
| `rgb` | `uint8` | `(H, W, 3)` | RGB channel order, vertically flipped to TD convention |
| `depth` | `uint16` | `(H, W)` | Millimetres. Shown raw in a TOP it looks nearly black — see `depth_handler.py` |
| `amplitude` | `uint16` | `(H, W)` | ToF amplitude (signal strength) |
| `color` | `uint8` | `(H, W, 3)` | RGB, aligned to ToF depth perspective |
| `ir_left` / `ir_right` | `uint8` | `(H, W)` | Rectified infrared frames |
| `pointcloud` | `float32` | `(H, W, 3)` | Organized XYZ in mm (R=X, G=Y, B=Z) — drops into a 32-bit float TOP |
| `segmentation` | `uint16` | `(H, W)` | depthai-nodes `SegmentationMask` class indices (background → 0), flipped to TD convention — drops into a TOP |
| `imu` | — | — | `ImuData` dict, not an ndarray |
| `detections` | — | — | `list[DetectionEntry]`, not an ndarray |
| `pose` | — | — | `list[PoseEntry]`, not an ndarray |

> The `depth` stream is `uint16` millimetres for the stereo and ToF pipelines, but `float32` **relative** inverse depth (`[0,1]`, larger = nearer) for `mono_depth_pipeline.py`, since that comes from a neural `Map2D` rather than a metric sensor.

> Stream names above are conventions, not requirements — `GetFrame(name)` keys off whatever names your pipeline's `build()` returns. The dtype/shape is determined by the decoded message type, not the name. Every colour image is RGB and every image is bottom-up (TouchDesigner's convention), whatever the stream is called.

**`DetectionEntry` fields:**

| Key | Type |
|---|---|
| `label` | `int` |
| `confidence` | `float` (0–1) |
| `xmin` / `ymin` / `xmax` / `ymax` | `float` (0–1, normalised) |
| `x` / `y` / `z` | `float`, millimetres in camera space — only for spatial detections (`dai.SpatialImgDetections`) |

**`PoseEntry` fields** (one per detected person):

| Key | Type | Description |
|---|---|---|
| `confidence` | `float` (0–1) | Detection confidence |
| `box` | `list[float]` | `[xmin, ymin, xmax, ymax]`, normalised 0–1 |
| `keypoints` | `list[list[float]]` | Joint positions `[[x, y], …]`, normalised 0–1 (COCO-17 order for the bundled pose model) |
| `edges` | `list[list[int]]` | Skeleton bone connections `[[i, j], …]` indexing into `keypoints` |

**`ExtendedDetectionEntry` fields** (depthai-nodes `ImgDetectionsExtended`, one per detection):

| Key | Type | Description |
|---|---|---|
| `label` | `int` | Class index |
| `confidence` | `float` (0–1) | Detection confidence |
| `box` | `list[float]` | Axis-aligned `[xmin, ymin, xmax, ymax]` outer bounds of the rotated box, normalised 0–1 |
| `angle` | `float` | Rotated-box rotation in degrees (`0` for upright boxes) |
| `keypoints` | `list[list[float]]` | `[[x, y, score], …]`, normalised 0–1; empty when the model emits none |

> `ImgDetectionsExtended` may also carry a message-level segmentation mask. It isn't part of the default payload — read it in a custom `parse` if you need it.

**`ClassificationEntry` fields** (depthai-nodes `Classifications`, one per class):

| Key | Type | Description |
|---|---|---|
| `label` | `str` | Class name |
| `score` | `float` (0–1) | Class probability |

**Bare lists** — `Keypoints` → `[[x, y, score], …]`; `Lines` → `[[x1, y1, x2, y2, confidence], …]`; `Clusters` → `[[[x, y], …], …]` (all normalised 0–1).

**`ImuData` fields:**

| Key | Type | Description |
|---|---|---|
| `accel` | `list[float]` | Latest reading, `[x, y, z]` m/s² |
| `gyro` | `list[float]` | Latest reading, `[x, y, z]` rad/s |
| `samples` | `list[list[float]]` | Every packet of the message, oldest first: `[ax, ay, az, gx, gy, gz, t]` (`t` = device timestamp in seconds). The IMU reports faster than TouchDesigner cooks — use this when you need all of it. |

---

### Camera Intrinsics

Intrinsics are read from the device calibration after `pipeline.start()` and cached for the lifetime of the connection. They are cleared when the pipeline stops.

#### `GetIntrinsics(socket: str = 'CAM_A') -> dict[str, float] | None`

Return the intrinsic matrix coefficients for the given camera socket at its native sensor resolution. Returns `None` if not yet connected or if the socket has no calibration.

```python
k = oak.GetIntrinsics('CAM_A')
# → {'fx': 461.3, 'fy': 461.3, 'cx': 320.0, 'cy': 240.0, 'width': 640.0, 'height': 480.0}

k = oak.GetIntrinsics('CAM_B')
# → {'fx': 855.1, 'fy': 856.2, 'cx': 638.4, 'cy': 403.7, 'width': 1280.0, 'height': 800.0}
```

`width`/`height` are the native sensor resolution the coefficients refer to. When reprojecting a stream delivered at a different resolution, scale `fx`/`cx` by `frameW / width` and `fy`/`cy` by `frameH / height`.

Socket names match `dai.CameraBoardSocket` member names: `'CAM_A'`, `'CAM_B'`, `'CAM_C'`.

**Point cloud reconstruction from ToF depth:**

```python
import numpy as np

k     = oak.GetIntrinsics('CAM_A')
depth = oak.GetDepth()          # (H, W) uint16, mm

if k and depth is not None:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32) / 1000.0       # metres
    x = (u - k['cx']) * z / k['fx']
    y = (v - k['cy']) * z / k['fy']
    points = np.stack([x, y, z], axis=-1)       # (H, W, 3)
```

---

### Runtime Controls

Pipeline nodes expose named input queues that accept depthai message objects. These queues are non-blocking (`maxSize=1`): if a new message arrives before the device has consumed the previous one, the previous message is silently dropped. This is intentional — for real-time parameter control you always want the latest value, not a stale queue.

`InputQueue.send()` is thread-safe; `SendControl` may be called from TD's main thread, an Execute DAT, or a panel callback without coordination.

#### `SendControl(name: str, message: object) -> bool`
Send a depthai control or config message to the named input queue. Returns `True` on success, `False` if the queue name is not registered for the active pipeline (e.g. the pipeline is not running, or that control was not declared by the pipeline script). A send that raises returns `False` and reports `control[name]: …` in `Lastwarning`.

```python
import depthai as dai

# Manual exposure
ctrl = dai.CameraControl()
ctrl.setManualExposure(10000, 400)          # exposure_us, iso
oak.SendControl('cam_ctrl', ctrl)

# Auto exposure
ctrl = dai.CameraControl()
ctrl.setAutoExposureEnable()
oak.SendControl('cam_ctrl', ctrl)

# Auto white balance
ctrl = dai.CameraControl()
ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
oak.SendControl('cam_ctrl', ctrl)
```

#### `SendFrame(name: str, source: TOP | np.ndarray, imgType=None) -> bool`
Send a **host image** into a named input queue, so the device runs a pipeline on TD-supplied
frames instead of an OAK camera. `source` is any TOP (read via `.numpyArray()`) or an
`(H, W, 3|4)` numpy array; it's coerced to a top-down uint8 BGR `dai.ImgFrame` (float TOPs are
treated as normalized 0–1). Returns `False` if the queue isn't registered. Main-thread only —
call it once per cook, e.g. from an Execute/CHOP-Execute callback. See
`src/TDOAK/examples/detection_input_pipeline.py`, which runs YOLO object detection on whatever TOP you
feed it (an on-device `ImageManip` letterboxes/repacks to the model input, so the source can be any size).

```python
# in an Execute DAT's onFrameStart, or a CHOP Execute callback:
op('TDOAK').SendFrame('image_in', op('moviefilein1'))
```

The matching pipeline declares the queue with `node.someInput.createInputQueue(...)` and returns
it in the controls dict — `SendFrame` and `SendControl` target the same queue registry.

#### `GetControlNames() -> list[str]`
Return the names of all registered input queues for the currently running pipeline. Returns an empty list when no pipeline is active.

```python
print(oak.GetControlNames())
# e.g. → ['cam_ctrl', 'stereo_cfg']
```

**Built-in control queues by pipeline:**

| Pipeline | Control name | Message type | Notes |
|---|---|---|---|
| Built-in RGB | `cam_ctrl` | `dai.CameraControl` | Exposure, focus, WB, etc. |
| Built-in RGB + depth | `cam_ctrl` | `dai.CameraControl` | |
| Built-in RGB + depth | `stereo_cfg` | `dai.StereoDepthConfig` | LR-check, subpixel, etc. |
| Built-in ToF | `tof_cfg` | `dai.ToFConfig` | ToF base configuration |
| `src/TDOAK/examples/pipeline.py` | `cam_ctrl`, `stereo_cfg` | same as above | |
| `src/TDOAK/examples/tof_showcase_pipeline.py` | `tof_cfg` | `ToFConfig` | |
| `src/TDOAK/examples/detection_input_pipeline.py` | `image_in` | `ImgFrame` (via `SendFrame`) | host-frame input, no camera |

---

### Script OP Integration

Stream data reaches a Script Operator in two stages, split across threads:

1. **decode** — turn the (model-specific) depthai message into plain, thread-safe data. Runs on the **background pump thread**, so it never blocks cooking/rendering.
2. **fill** — write that data into a Script TOP/CHOP/SOP/DAT. Runs on the **main thread**, where TD object access is legal.

Built-in decoders cover images (`rgb`, `depth`, `amplitude`, `color`, `ir_left/right`), `pointcloud`, `imu`, `dai.ImgDetections` (→ `detections`/`pose`), and the depthai-nodes message family — `Map2D` (monocular depth / heatmaps → float32 `(H, W)`), `ImgDetectionsExtended`, `Keypoints`, `SegmentationMask`, `Classifications`, `Lines`, and `Clusters` — so common streams need no code at all. Decoding keys off the **message type**, not the stream name, so these work whatever you name the stream. For any other model output, attach a per-stream **Handler DAT** (its `parse` overrides the default decode).

> These are zero-config defaults, not a fixed contract. A model-specific format (different keypoint order, custom tensor layout, etc.) is still handled the normal way: a Handler `parse` returns whatever structure you like and a `fill` shapes it into the Script OP.

#### Per-stream Handler DAT

Point a `Streams` block's `Handler` parameter at a DAT. It may define either or both of:

```python
# `dai`, `np`, and `oak` (the OAK component) are already in scope. Use `oak` to reach
# the promoted API from a fill (e.g. oak.GetIntrinsics(...)) — fill runs inside the
# extension, where a bare op('...') would resolve inside the component itself.

def parse(msg):
    # BACKGROUND THREAD — no op()/me/TD access. depthai message -> plain data
    # (numpy array, or list/dict of primitives). Return None to skip this frame.
    return msg.getTensor('output')

def fill(scriptOp, data):
    # MAIN THREAD — full TD access. plain data -> the bound operator.
    scriptOp.clear()
    ...
```

- `parse` **overrides** the built-in decoder for that stream; omit it to keep the default decode.
- `fill` **overrides** the default push; omit it to use the default (numpy → Script TOP).
- Handlers are loaded on **Start**, and **hot-reload into a running pipeline** when you add/edit a Streams block, its `Scriptop`, or its `Handler` DAT — automatically while `Auto Reload` is on, or on demand via the `Reload Handlers` pulse / `ReloadHandlers()`. No device restart is needed; only adding a brand-new **physical stream** (a `build()` change) requires a `Rebuild`. A failing handler reports to `Lasterror` and is skipped — it never stalls the pump.

> Binding an existing ndarray stream to a plain **Script TOP** (no Handler) updates live on its own — the sequence is read every cook. Anything needing a custom `parse`/`fill` (Script CHOP/SOP/DAT, or a non-default decode) is what the hot-reload applies.

#### Push vs pull, by operator family

Script POPs, TOPs, CHOPs, and DATs support *Modify Outside of Cook*, so the extension writes into them directly each cook. Script SOPs build geometry inside their cook, so they have to **pull** instead:

| Operator | How it's filled | Cook callback needed |
|---|---|---|
| Script **TOP** | pushed — default `copyNumpyArray`, or your `fill` | No |
| Script **CHOP** | pushed — your `fill` (e.g. `copyNumpyArray((chans, samples) float32)`) | No |
| **DAT** | pushed — your `fill` (e.g. `clear()` + `appendRows([...])`) | No |
| Script **POP** | pushed — your `fill` (staging API works outside the cook) | No |
| Script **SOP** | pulled — your `fill` runs *inside* the cook | Yes — `FillScriptOp(me)` |

For a plain image stream the ready-made **TDOAKStream** component (see [Component Parameters](#the-tdoakstream-component)) is the shortest path: it is a Script TOP that pulls with `FillScriptOp(scriptOp, stream=…)`.

> **Pushing into a Script TOP of your own:** a new Script TOP's default `onCook` writes a small noise image, which overwrites the pushed frame whenever the TOP cooks. Empty that `onCook` (leave only `return`), or use the pull form below instead. `Modify Outside of Cook` is switched on for you.

For a Script SOP, add a one-line cook callback:

```python
def onCook(scriptOp):
    op('TDOAK').FillScriptOp(scriptOp)
```

Any Script OP can also **pull without any binding at all** — pass the stream (and optionally a Handler DAT) directly; see the `FillScriptOp` reference below:

```python
def onCook(scriptOp):
    op('TDOAK').FillScriptOp(scriptOp, stream='rgb')
```

#### Binding streams to operators

Two ways to bind, both honouring the Handler/push/pull rules above:

**Sequential block (UI)** — add a `Streams` block, set `Stream`, point `Scriptop` at the operator, and (optionally) `Handler` at a DAT.

**Code registry** — bind from a script instead of the UI:

```python
oak.RegisterScriptop('depth', op('script_depth'))   # multiple OPs per stream OK
oak.UnregisterScriptop('depth', op('script_depth')) # remove one
oak.UnregisterScriptop('rgb')                        # remove all for the stream
```

#### Examples

**Default Script TOP** (no Handler — automatic): just bind `rgb` to a Script TOP; the extension pushes it. No cook code required.

**Pose skeleton into a Script SOP** — a `fill`-only Handler DAT on the `pose` block (decode stays the built-in `PoseEntry`):

```python
def fill(scriptOp, people):          # people: list[PoseEntry]
    scriptOp.clear()
    if not people:
        return
    for person in people:
        pts = [scriptOp.appendPoint() for _ in person['keypoints']]
        for p, (x, y) in zip(pts, person['keypoints']):
            p.x, p.y, p.z = x, 1.0 - y, 0.0     # y-flip: rgb is flipud'd for display
        for i, j in person['edges']:
            prim = scriptOp.appendPoly(2, closed=False, addPoints=False)
            prim[0].point, prim[1].point = pts[i], pts[j]
```
…with `op('TDOAK').FillScriptOp(me)` in the SOP's `onCook`.

#### `RegisterScriptop(streamName: str, scriptop: OP) -> None`
Bind a Script Operator to receive frames for `streamName`. Multiple OPs can be registered for the same stream.

#### `UnregisterScriptop(streamName: str, scriptop: OP | None = None) -> None`
Remove a registered binding. Pass `scriptop=None` to remove all bindings for the stream.

#### `RegisterComponent(scriptop: OP, streamName: str | None = None, handlerDat: OP | None = None) -> None`
One-call wiring for a **self-contained stream component**: binds the scriptop to its stream *and* ingests its Handler DAT. `streamName`/`handlerDat` default to `Stream`/`Handler` custom parameters on the scriptop's parent COMP, so a wrapper component wires itself with `RegisterComponent(op('script1'))`. Call on init and whenever the component's Stream/Handler parameter or the handler text changes — not per cook. Re-binds cleanly across a Stream change (the op is cleared from any prior stream first).

#### `UnregisterComponent(scriptop: OP) -> None`
Tear-down counterpart (call e.g. on component delete): unbinds the op from every stream and drops its `fill`. A stream's `parse` is left in place, since it's shared per physical queue — `Rebuild`/`ReloadHandlers` re-derives it.

#### `RegisterHandler(scriptop: OP, streamName: str, handlerDat: OP | None) -> None` / `UnregisterHandler(scriptop: OP, streamName: str | None = None) -> None`
Lower-level halves of `RegisterComponent`: ingest (or drop) a Handler DAT for one operator — its `fill` is keyed by the op's path, its `parse` installed per stream (last registration wins; parse is shared across all consumers of that stream). Usually you want `RegisterComponent` instead.

#### `FillScriptOp(scriptop: OP, stream: str | None = None, handler: OP | str | None = None) -> bool`
Pull entry point: call from a Script Operator's cook callback to fill it *inside* its cook. Three ways to use it, from most to least wiring:

```python
def onCook(scriptOp):
    op('TDOAK').FillScriptOp(scriptOp)                    # 1. registered binding
    op('TDOAK').FillScriptOp(scriptOp, stream='rgb')      # 2. explicit stream
    op('TDOAK').FillScriptOp(scriptOp, stream='depth',    # 3. explicit stream + handler
                             handler=op('text_pointcloudHandler'))
```

1. **No arguments** — the operator's registered binding is used (a **Streams** block or `RegisterScriptop`; the `fill` is keyed by the operator's path). This is the classic Script-SOP form, since SOPs can't be written outside their cook.
2. **`stream='name'`** — pull that stream explicitly with **zero registration**: no Streams block, no registry entry. The first call marks the stream as wanted, so the worker starts decoding it even if nothing else is bound to it (data then arrives on the next frames). Works for any Script OP family, not just SOPs — useful for ad-hoc consumers that shouldn't appear in the component's UI.
3. **`handler=`** — a Handler DAT (or its path) to apply on top of an explicit stream: its `fill(scriptOp, data)` runs against the stream's latest frame, and if it defines a `parse(msg)` that parse is installed for the stream's worker-side decode. The DAT is re-exec'd only when its text changes, so calling this every cook is cheap.

When no `fill` is available (neither registered nor passed), built-in fallbacks apply: an ndarray frame into a Script **TOP** (`copyNumpyArray`), a float point array into a Script **POP**. Returns `True` if the operator was filled; `False` means the stream is unknown/unbound, no frame has arrived yet (normal during the first cooks after Start), or no fill/fallback matched. A `fill` that raises reports to `Lasterror` and returns `False` — it never stalls the cook.

#### `GetFrameForScriptop(scriptop: OP) -> FrameData | None`
Look up the raw frame for the given Script OP (registry + sequential block). A lower-level pull alternative to `FillScriptOp` when you want the data rather than auto-fill.

```python
def cook(scriptOp):
    data = op('TDOAK').GetFrameForScriptop(me)
```

---

### Status

#### `GetStatus() -> StatusInfo`
Return a snapshot of the current pipeline state.

`StatusInfo` fields:

| Key | Type | Description |
|---|---|---|
| `connected` | `bool` | Whether frames are arriving. |
| `fps` | `float` | Measured frame rate (updated once per second, driven by the `rgb` stream). |
| `latency_ms` | `float` | Frame latency in milliseconds (device timestamp vs host receipt). |
| `error` | `str` | Why the pipeline is not running, or empty string. |
| `warning` | `str` | Latest problem the pipeline survived, or empty string. |
| `streams` | `list[str]` | Active stream names. |
| `status` | `str` | Lifecycle text, same as the `Status` parameter. |
| `stream_age` | `dict[str, float]` | Seconds since each stream's last message (updated about once a second). |

#### `GetLastTraceback() -> str`
Full Python traceback of the last failed run — the detail behind the one-line `Lasterror`.

---

## Pipeline Scripts

Drop a Python DAT into the **Pipeline DAT** parameter to replace the built-in auto-detect pipeline with a custom one. Enable **Auto Reload** to have the pipeline rebuild automatically whenever the DAT changes.

### Contract

The DAT must define a `build` function:

```python
def build(pipeline: dai.Pipeline) -> tuple[
    dict[str, dai.DataOutputQueue],   # output streams  (required)
    list[str],                         # stream names    (required)
    dict[str, dai.InputQueue],         # control inputs  (optional)
]:
```

The third element is optional — returning a 2-tuple is also accepted (no control queues will be registered).

- **Output queues:** keys become the stream names available via `GetFrame(name)`.
- **Stream names:** the ordered list used to populate the `Stream` menus on sequential blocks.
- **Control queues:** keys become the names accepted by `SendControl(name, msg)`. Create these with `node.someInput.createInputQueue(maxSize=1, blocking=False)`.

In scope for the script: `dai`, and `PROJECT_FOLDER` (the folder of the .toe, as a string). The script runs on the worker thread — TouchDesigner objects (`op`, `project`, `me`) are **not** available in it.

Optional extras a script may define:

```python
FIRST_FRAME_TIMEOUT_S = 30   # override the Tune page's First Frame Timeout for this pipeline
STALL_TIMEOUT_S       = 0    # ... and the Stall Timeout (0 = off: a pipeline that only answers
                             #     frames you send it is silent by design)

def started(pipeline, device):
    # Called on the worker right after pipeline.start(), for device calls that need a
    # running pipeline - e.g. the IR emitters of an OAK-D Pro:
    device.setIrLaserDotProjectorIntensity(0.5)
    device.setIrFloodLightIntensity(0.0)
```

Don't request an explicit frame type in `requestOutput()` unless you know the camera supports it (`cam.requestOutput((1280, 720))`, no `type=`): an unsupported type leaves the pipeline running without frames, which the First Frame Timeout then reports.

### Minimal example

```python
import depthai as dai

def build(pipeline):
    cam = pipeline.create(dai.node.Camera).build()
    out = cam.requestOutput((1280, 720))

    queues   = {'rgb': out.createOutputQueue()}
    controls = {'cam_ctrl': cam.inputControl.createInputQueue(maxSize=1, blocking=False)}

    return queues, ['rgb'], controls
```

### Model Pre-fetch

Resolving a neural model inline (the Model Zoo's online freshness check + download, or loading an archive) is largely Python-level work that **holds the GIL** — done on the camera pump it still froze TouchDesigner for several seconds, because the worker and TD share one interpreter. So declared models are fetched in a **separate process** (its own GIL) before the device pipeline is built, and progress streams back into the read-only **Models** sequence while the UI stays responsive.

#### Declaring models

A pipeline script declares the models it uses with a module-level `MODELS` list (a bare `MODEL = '...'` is also accepted). This is read on **Start** by statically parsing the script (`ast`) — it is never executed, so the scan has no side effects. The declaration must be a literal or reference other module-level literal assignments (`MODELS = [MODEL_A, MODEL_B]` works); a computed value is skipped and `build()` then resolves it inline.

```python
MODEL  = 'luxonis/yolov8-nano-pose-estimation:coco-512x384'
MODELS = [MODEL]            # one entry per model build() resolves
```

Each entry is classified automatically:

- **HubAI zoo slug** (anything that isn't a local file) → downloaded via `getModelFromZoo` in the pre-fetch subprocess, into the shared cache (`.depthai_cached_models/`, set with `DEPTHAI_ZOO_CACHE_PATH`). Progress (depthai's JSON download bytes) is mirrored into the block's `Progress` 0→1.
- **Local file** (a path that resolves, absolute or relative to `project.folder`) → an NNArchive/blob already on disk. No download; the block shows `1.0` (ready) immediately.

After a successful pre-fetch of a zoo slug, the in-process `build()` runs with `DEPTHAI_ZOO_INTERNET_CHECK=0` so it loads the cached archive instead of repeating the online check. An all-local pipeline leaves the check on, so any incidental zoo resolve still works.

The subprocess uses the interpreter of the **virtual environment depthai was loaded from** (wherever it is), falling back to `<project folder>/.venv`. If neither exists, or the subprocess fails, `Lastwarning` says why and the model is resolved in-process by `build()` instead — it still works, but the download can freeze the UI. On macOS a stand-alone Python can't load depthai from an environment created by TouchDesigner (code-signing), so there the in-process path is the normal one.

> Declaring `MODELS` is optional. With none declared, `build()` resolves models itself inline — simplest, but that's the path that briefly hangs the UI.

#### Custom & converted models

Pass a local model straight to the parser node — `ParsingNeuralNetwork.build()` accepts a `dai.NNArchive` and skips the zoo entirely:

```python
MODELS = ['models/mymodel.tar.xz']     # resolves to a file → classified local, no download

def build(pipeline):
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork
    cam     = pipeline.create(dai.node.Camera).build()
    archive = dai.NNArchive(f'{PROJECT_FOLDER}/models/mymodel.tar.xz')
    nn      = pipeline.create(ParsingNeuralNetwork).build(cam, archive)
    return ({'rgb': nn.passthrough.createOutputQueue(maxSize=1, blocking=False),
             'out': nn.out.createOutputQueue(maxSize=1, blocking=False)},
            ['rgb', 'out'])
```

A bare blob without parser metadata won't work with `ParsingNeuralNetwork` (it needs the NNArchive config to pick parsers) — use a plain `dai.node.NeuralNetwork` for those.

**Conversion** of a raw model (ONNX/OpenVINO/TFLite → NNArchive for RVC2/RVC4, or a legacy MyriadX blob via `blobconverter` for RVC2 only) is a **one-time offline step** done with Luxonis tooling ([Hub](https://models.luxonis.com/) / `modelconverter`). It can take minutes — never run it inside `build()` on the Start path. Commit the resulting NNArchive and reference it as a local entry in `MODELS`.

### Bundled pipeline scripts

All paths relative to `src/TDOAK/examples/` (resolve at runtime with `mod.TDOAK.examplesPath()`).

| File | Device | Streams | Controls |
|---|---|---|---|
| `pipeline.py` | OAK-D (RGB + stereo depth, synced) | `rgb`, `depth` | `cam_ctrl`, `stereo_cfg` |
| `tof_showcase_pipeline.py` | OAK-D SR PoE (ToF depth + amplitude; `rgb` when a color socket exists; organized point cloud opt-in via `STREAM_POINTCLOUD`) | `rgb`, `depth`, `amplitude` (+ `pointcloud`) | `tof_cfg` |
| `yolo_pipeline.py` | Any RGB OAK (YOLOv6-nano object detection) | `rgb`, `detections` | — |
| `detection_input_pipeline.py` | Any OAK (YOLO detection on **host** frames via `SendFrame` — no camera) | `rgb`, `detections` | `image_in` |
| `pose_pipeline.py` | Any RGB OAK (YOLOv8-nano pose / skeleton — needs `depthai-nodes`) | `rgb`, `pose` | — |
| `mono_depth_pipeline.py` | Any RGB OAK (MiDaS v2.1 monocular depth — needs `depthai-nodes`) | `rgb`, `depth` | — |
| `face_pipeline.py` | Any RGB OAK (YuNet face detection + 5 landmarks — needs `depthai-nodes`) | `rgb`, `faces` | — |
| `hand_pipeline.py` | Any RGB OAK (MediaPipe two-stage hand tracking — needs `depthai-nodes`; pair with `hand_handler.py`) | `rgb`, `hands` | — |
| `classification_pipeline.py` | Any RGB OAK (EfficientNet-Lite0 classification — needs `depthai-nodes`) | `rgb`, `classes` | — |
| `segmentation_pipeline.py` | Any RGB OAK (DeepLabV3+ person segmentation — needs `depthai-nodes`) | `rgb`, `segmentation` | — |

### Bundled Handler DATs

| File | Stream | What it does |
|---|---|---|
| `pointcloud_handler.py` | `depth` | fill-only — reprojects the depth map through the device calibration (`GetIntrinsics`) into a metric point cloud in a Script **POP**. Pairs with `pipeline.py`. |
| `face_handler.py` | `faces` | fill-only — draws the detected face landmark points into a Script **POP**. |
| `hand_handler.py` | `hands` | `parse` + `fill` — decodes the GatheredData message, back-maps each hand's crop-local keypoints onto the full frame, and draws 21-point skeletons into a Script **POP** (pushed, no cook callback). Required by `hand_pipeline.py`. |
| `depth_handler.py` | `depth` | fill-only — turns the 16-bit millimetre depth image into a 32-bit float image for a Script **TOP**: metres, or 0–1 across a display range (near = white). Also for disparity streams. |
| `classification_handler.py` | `classes` | fill-only — writes the top results into a Table **DAT** (`label \| score` rows, sorted). |
| `segmentation_handler.py` | `segmentation` | fill-only — shapes the class mask into silhouette points in a Script **POP**. |
| `pose_handler.py`, `skeleton_handler.py` | — | blank `parse`/`fill` templates to start a custom Handler from. |

---

## Threading Model

The camera pump runs in a **background thread** managed by TD's `ThreadManager`. Frame data is transferred to the main thread via a `threading.Lock`-protected dictionary.

- **Background thread** (`cameraPump`): opens the device, **pre-fetches declared models on a subprocess** (streaming download progress into shared state — see [Model Pre-fetch](#model-pre-fetch)), builds and starts the pipeline, polls output queues in a tight loop, **decodes** each message (built-in decoder or a Handler's `parse`), and writes the plain result to the shared dict. On exit (any path) it closes the device in its `finally`, keeping that blocking call off the main thread.
- **Main thread** (`onRefresh`): called by the Thread Manager each cook, reads shared state, updates TD parameters, mirrors per-model download progress into the **Models** sequence, and **fills** Script OPs (push for TOP/CHOP/DAT, pull for SOP via `FillScriptOp`).
- **Model pre-fetch subprocess**: spawned from the background thread with its own interpreter, so the GIL-holding zoo download/check runs without blocking either TD thread. It reports progress as JSON on stdout, which the background thread parses and the main thread reflects.
- **`SendControl`**: calls `InputQueue.send()` directly from the main thread. depthai input queues are designed for cross-thread use; no intermediate queue is needed.

The background thread **must not access any TD objects** (`op()`, `me`, `run()`, parameters, etc.) — this is why a Handler's `parse` (worker thread) must stay pure-Python/numpy while `fill` (main thread) does the operator I/O. Handler DAT text is read on the main thread (it's TD access) and the resulting `parse`/`fill` callables are stored on the extension; the worker reads `self._parsers` fresh per message rather than capturing it, so `ReloadHandlers()` can swap the whole dict on the main thread (an atomic reference reassignment) and the running worker picks it up with no restart and no lock. All TD access happens exclusively in the Thread Manager hooks.

---

## Development

The extension code lives in the package, not in the `.tox`:

| Where | What |
|---|---|
| `src/TDOAK/ext/OAKManagerExt.py`, `CameraControlExt.py` | The extension classes. Each component's extension DAT is a few-line shim calling `TDOAK.ext.create('<Class>', me)`. |
| `src/TDOAK/core/` | Pure-Python logic without TouchDesigner or depthai imports: lifecycle state, frame watchdog, retry policy, error channel, decoders, pipeline-script scan, interpreter lookup, version checks. |
| `tests/` | `pytest` unit tests for `core` and the `ext` loader, plus static checks of the example scripts. |

Run the tests from the repo root with any Python that has `pytest` and `numpy` (no TouchDesigner, no depthai, no camera needed):

```
python -m pytest
```

To pick up an edit to `core/` or `ext/` in a running TouchDesigner, pulse **Re-Init Extensions** on the component: changed modules are reloaded, and a running pipeline is restarted. Editing the files alone changes nothing until then.

---

## Releasing

The package version lives in one place: `__version__` in `src/TDOAK/__init__.py` (`pyproject.toml` reads it dynamically). The depthai range a release is tested with is the pin in `pyproject.toml` **and** `TESTED_DEPTHAI` in `src/TDOAK/core/compat.py` (a test keeps them in step).

1. Run `python -m pytest`.
2. Run the **hardware checklist** below.
3. If a `.tox` changed (parameters, the parexec, a shim — *not* extension code, which is in the package), save it with neutral values: `Device` on `-- Select Device --`, empty `Device Address`/`Pipeline DAT`, one empty `Streams` block, Tune page at defaults, and stamp the About page first so **.tox Save Build** names the build that saved it:
     ```python
     op('TDOAK').par.Toxsavebuild = app.build
     ```
   A `.tox` doesn't record its save build anywhere TouchDesigner can read back, so this field is a stamp, not a live value. **Version** and **Depthai Version** fill themselves at runtime from the installed packages.
   Careful when adding parameters to the **Device** page: a parameter appended after the `Models` sequence corrupts the sequence. Remove the sequence, add and sort the parameters, then re-create it (`appendSequence('Models')`, parameters `Model` and `Progress`, `seq.blockSize = 2`).
4. Bump `__version__` (e.g. `"0.2.1"`), commit, push.
5. Tag and push the tag:
   ```
   git tag v0.2.1
   git push origin v0.2.1
   ```

The `Release` GitHub workflow triggers on `v*` tags, verifies the tag matches `__version__` (mismatch fails the run), runs the tests, builds sdist + wheel, and publishes them to a GitHub Release with generated notes. CI (`ci.yml`) runs the tests on Linux, macOS and Windows and smoke-tests the built wheel on every push.

### Hardware checklist

Run on at least one USB camera and one PoE camera before a release (about 20 minutes).

1. Blank project, Quick Start followed literally: the image appears in TDOAKStream.
2. Start, Stop, Start ten times: no crash; Status reads correctly each time.
3. Rebuild five times while streaming.
4. Pull the USB cable (or the Ethernet link) mid-stream: an error within the Stall Timeout. With `Reconnect` = Forever: recovery after replugging.
5. Start with no camera attached: *No OAK device found*.
6. Pulse Re-Init Extensions while streaming, and edit the Pipeline DAT while streaming (Auto Reload on): a clean restart each time.
7. Stereo depth with `Auto Calibration` Off: runs for 5 minutes. On a ToF camera: `tof_showcase_pipeline.py`, depth and amplitude.
8. One model pipeline (e.g. `yolo_pipeline.py`) with an empty model cache: it starts, detections arrive.
9. Break and fix a Handler DAT, break and fix the Pipeline DAT: Last Warning / Last Error report it, and the fixed version runs without pressing Start.
10. Quit TouchDesigner while streaming: a clean exit.
11. Thirty-minute soak: Fps steady, Status *Streaming*.
12. On a USB2 cable: Device Info shows the link speed and Last Warning flags it.
