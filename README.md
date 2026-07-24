# TDOAK

TouchDesigner extension for OAK cameras via [depthai](https://docs.luxonis.com/software-v3/) v3.

> Contributing or digging into the internals? Read **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — component map, threading model, data flow, lifecycle state machine, and the crash-safety invariants behind the device teardown path.

Supports USB OAK-D (RGB + stereo depth + IMU) and OAK-D SR PoE (ToF depth + aligned color) with automatic camera-type detection, plus neural pipelines (object detection, human-pose/skeleton tracking). Pipeline scripts are hot-reloadable DATs. Runtime camera controls (exposure, stereo config, ToF config) can be sent from the main thread at any time while the pipeline is running.

Stream data is decoded off the main thread and pushed into Script Operators. Common streams (rgb, depth, amplitude, detections, pose, imu…) are handled out of the box; any other model's output can be decoded and shaped with a per-stream **Handler DAT** — see [Script OP Integration](#script-op-integration).

Neural models declared by a pipeline are **pre-fetched on a subprocess** before the device pipeline is built, so a Model Zoo download never freezes the UI — progress is reported live in the **Models** sequence. Custom local models (NNArchive/blob on disk) are detected and loaded directly without a download. See [Model Pre-fetch](#model-pre-fetch).

---

## Requirements

- TouchDesigner 2025.x (Python 3.11 embedded)
- depthai 3.6.1 installed in TD's Python environment
- depthai-nodes (only for the bundled pose pipeline — provides `ParsingNeuralNetwork`)
- numpy (already bundled with TD)
- A project `.venv` (`/.venv/Scripts/python.exe`) with `depthai` installed — used by the [model pre-fetch](#model-pre-fetch) subprocess (falls back to TD's Python if absent)

---

## Quick Start

1. Drop the OAK COMP into your project.
2. Click **List Devices** to populate the Device menu.
3. Select a device (or leave as `-- Select Device --` to use whichever camera connects first).
4. Click **Start**.

The component's `Connected`, `Fps`, `Activestreams`, and `Lasterror` parameters update every cook. Wire Script TOPs, CHOPs, SOPs, or DATs to the **Streams** sequential block to receive frame data.

---

## Component Parameters

| Parameter | Type | Description |
|---|---|---|
| `Device` | Menu | Target device MXID or IP. `-- Select Device --` uses the first available device. Populated by **List Devices**. |
| `Log Level` | Menu | depthai verbosity (`trace`…`critical`), applied to the device on Start. Default `warn` silences the periodic `[info]` chatter (temperatures, cpu usage). |
| `Start` | Pulse | Start the camera pipeline. Resets the retry counter. |
| `Stop` | Pulse | Stop the pipeline cleanly. |
| `Rebuild` | Pulse | Stop and restart (useful after changing parameters). |
| `List Devices` | Pulse | Enumerate all visible OAK devices and update the Device menu. |
| `Reload Handlers` | Pulse | Hot-reload the Streams blocks' Handler DATs (`parse`/`fill`) into the **running** pipeline — applies handler/binding edits without a restart. See [Script OP Integration](#script-op-integration). |
| `Pipeline DAT` | DAT ref | Optional custom pipeline script. See [Pipeline Scripts](#pipeline-scripts). Leave empty for the built-in auto-detect pipeline. |
| `Auto Reload` | Toggle | When enabled, automatically rebuilds when the Pipeline DAT changes. |
| `Enable Depth` | Toggle | Enable stereo depth output on RGB cameras (ignored for ToF devices). |
| `Enable IMU` | Toggle | Enable IMU output (accelerometer + gyroscope) on RGB cameras. |
| `Fps` | Float (read-only) | Measured frame rate, updated once per second. |
| `Connected` | Toggle (read-only) | `1` while the pipeline is running and connected. |
| `Active Streams` | String (read-only) | Comma-separated list of active stream names. |
| `Last Error` | String (read-only) | Most recent error message, or empty if no error. |
| `Streams` (seq) | Sequential block | Bind stream names to Script Operators. Each block has a `Stream` menu, a `Scriptop` OP reference, and an optional `Handler` DAT reference (custom `parse`/`fill` — see [Script OP Integration](#script-op-integration)). |
| `Models` (seq) | Sequential block (read-only) | Status readout, one block per model the pipeline declares. `Model` is the slug or local path; `Progress` (0–1) tracks the pre-fetch. Populated automatically on Start — see [Model Pre-fetch](#model-pre-fetch). |

---

## `OAKManagerExt` API Reference

Access the extension from any TD script:

```python
oak = op('OAK').ext.OAKManagerExt
```

All methods listed here are safe to call from the **main thread** unless otherwise noted.

---

### Lifecycle

#### `Start() -> None`
Start the camera pipeline. Resets the retry counter to zero (so errors before this call are forgotten). If the pipeline fails, it retries up to 3 times with a 10-second delay between attempts. After 3 failures, `Lasterror` shows the final error and the pipeline stops retrying — call `Start()` again to reset.

#### `Stop() -> None`
Signal the background pump to exit cleanly. The pipeline and device are closed **on the background thread** (the pump's `finally`), not in a main-thread hook — `device.close()`/`pipeline.stop()` are blocking depthai calls, so keeping them off the main thread avoids freezing the UI for several frames on each stop or retry.

#### `Rebuild() -> None`
Stop the pipeline and restart it after 5 frames. Equivalent to `Stop()` followed by a delayed `Start()`. Triggered automatically when `Auto Reload` is on and the Pipeline DAT changes.

#### `ReloadHandlers() -> None`
Re-load the Streams blocks' Handler DATs and hot-swap their `parse`/`fill` into the **running** pipeline — no device or pipeline restart. Call after adding or editing a Streams block, its `Scriptop`, or its `Handler` DAT. The worker reads the new `parse` on its next message; the main thread picks up the new `fill` on the next cook. Adding a brand-new **physical stream** still needs a `Rebuild()` (that's a `build()` change). Safe to call while stopped — it just primes state for the next Start. Triggered automatically when `Auto Reload` is on and a Streams block/binding/Handler changes, or manually via the `Reload Handlers` pulse.

---

### Device Discovery

#### `ListDevices(callback=None) -> None`
Enumerate all visible OAK devices in a background thread and update the `Device` parameter menu. Optionally accepts a callback that receives the device list when enumeration completes.

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
oak.GetRgb()      # → ndarray (H, W, 3) uint8 BGR, or None
oak.GetDepth()    # → ndarray (H, W) uint16, or None
oak.GetIrLeft()   # → ndarray (H, W) uint8, or None
oak.GetIrRight()  # → ndarray (H, W) uint8, or None
oak.GetImu()      # → ImuData, or None
oak.GetPose()     # → list[PoseEntry], or None
```

**Frame formats by stream:**

| Stream | dtype | Shape | Notes |
|---|---|---|---|
| `rgb` | `uint8` | `(H, W, 3)` | BGR channel order, vertically flipped to TD convention |
| `depth` | `uint16` | `(H, W)` | Millimetres |
| `amplitude` | `uint16` | `(H, W)` | ToF amplitude (signal strength) |
| `color` | `uint8` | `(H, W, 3)` | BGR, aligned to ToF depth perspective |
| `ir_left` / `ir_right` | `uint8` | `(H, W)` | Rectified infrared frames |
| `pointcloud` | `float32` | `(H, W, 3)` | Organized XYZ in mm (R=X, G=Y, B=Z) — drops into a 32-bit float TOP |
| `segmentation` | `uint16` | `(H, W)` | depthai-nodes `SegmentationMask` class indices (background → 0), flipped to TD convention — drops into a TOP |
| `imu` | — | — | `ImuData` dict, not an ndarray |
| `detections` | — | — | `list[DetectionEntry]`, not an ndarray |
| `pose` | — | — | `list[PoseEntry]`, not an ndarray |

> The `depth` stream is `uint16` millimetres for the stereo and ToF pipelines, but `float32` **relative** inverse depth (`[0,1]`, larger = nearer) for `mono_depth_pipeline.py`, since that comes from a neural `Map2D` rather than a metric sensor.

> Stream names above are conventions, not requirements — `GetFrame(name)` keys off whatever names your pipeline's `build()` returns. The dtype/shape is determined by the decoded message type, not the name.

**`DetectionEntry` fields:**

| Key | Type |
|---|---|
| `label` | `int` |
| `confidence` | `float` (0–1) |
| `xmin` / `ymin` / `xmax` / `ymax` | `float` (0–1, normalised) |

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
| `accel` | `list[float]` | `[x, y, z]` m/s² |
| `gyro` | `list[float]` | `[x, y, z]` rad/s |

---

### Camera Intrinsics

Intrinsics are read from the device calibration after `pipeline.start()` and cached for the lifetime of the connection. They are cleared when the pipeline stops.

#### `GetIntrinsics(socket: str = 'CAM_A') -> dict[str, float] | None`

Return the intrinsic matrix coefficients for the given camera socket at its native sensor resolution. Returns `None` if not yet connected or if the socket has no calibration.

```python
k = oak.GetIntrinsics('CAM_A')
# → {'fx': 461.3, 'fy': 461.3, 'cx': 320.0, 'cy': 240.0}

k = oak.GetIntrinsics('CAM_B')
# → {'fx': 855.1, 'fy': 856.2, 'cx': 638.4, 'cy': 403.7}
```

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
Send a depthai control or config message to the named input queue. Returns `True` on success, `False` if the queue name is not registered for the active pipeline (e.g. the pipeline is not running, or that control was not declared by the pipeline script).

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
`src/TDOAK/examples/mono_depth_input_pipeline.py`, which runs MiDaS depth on whatever TOP you feed it (an
on-device `ImageManip` resizes/repacks to the model input, so the source can be any size).

```python
# in an Execute DAT's onFrameStart, or a CHOP Execute callback:
op('OAK').ext.OAKManagerExt.SendFrame('image_in', op('moviefilein1'))
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
| `src/TDOAK/examples/tof_pipeline.py` | `cam_ctrl`, `tof_cfg` | `CameraControl`, `ToFConfig` | `cam_ctrl` targets CAM_B color camera |

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

Script TOPs, CHOPs, and DATs support *Modify Outside of Cook*, so the extension writes into them directly each cook. Script SOPs build geometry inside their cook, so they **pull** instead:

| Operator | How it's filled | Cook callback needed |
|---|---|---|
| Script **TOP** | pushed — default `copyNumpyArray`, or your `fill` | No |
| Script **CHOP** | pushed — your `fill` (e.g. `copyNumpyArray((chans, samples) float32)`) | No |
| **DAT** | pushed — your `fill` (e.g. `clear()` + `appendRows([...])`) | No |
| Script **POP** | pushed — your `fill` (staging API works outside the cook) | No |
| Script **SOP** | pulled — your `fill` runs *inside* the cook | Yes — `FillScriptOp(me)` |

For a Script SOP, add a one-line cook callback:

```python
def onCook(scriptOp):
    op('OAK').ext.OAKManagerExt.FillScriptOp(scriptOp)
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
…with `op('OAK').ext.OAKManagerExt.FillScriptOp(me)` in the SOP's `onCook`.

#### `RegisterScriptop(streamName: str, scriptop: OP) -> None`
Bind a Script Operator to receive frames for `streamName`. Multiple OPs can be registered for the same stream.

#### `UnregisterScriptop(streamName: str, scriptop: OP | None = None) -> None`
Remove a registered binding. Pass `scriptop=None` to remove all bindings for the stream.

#### `FillScriptOp(scriptop: OP) -> bool`
Pull entry point for Script SOPs (which can't be written outside their cook): call it from the SOP's `onCook` to run the bound stream's `fill(scriptOp, data)` inside the cook. TOPs/CHOPs/DATs are auto-pushed and don't need this. Falls back to `copyNumpyArray` for a Script TOP when no custom `fill` is set. Returns `True` if it filled the operator.

#### `GetFrameForScriptop(scriptop: OP) -> FrameData | None`
Look up the raw frame for the given Script OP (registry + sequential block). A lower-level pull alternative to `FillScriptOp` when you want the data rather than auto-fill.

```python
def cook(scriptOp):
    data = op('OAK').ext.OAKManagerExt.GetFrameForScriptop(me)
```

---

### Status

#### `GetStatus() -> StatusInfo`
Return a snapshot of the current pipeline state.

`StatusInfo` fields:

| Key | Type | Description |
|---|---|---|
| `connected` | `bool` | Whether the pipeline is running. |
| `fps` | `float` | Measured frame rate (updated once per second, driven by the `rgb` stream). |
| `latency_ms` | `float` | Frame latency in milliseconds (device timestamp vs host receipt). |
| `error` | `str` | Last error message, or empty string. |
| `streams` | `list[str]` | Active stream names. |

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

### Minimal example

```python
import depthai as dai

def build(pipeline):
    cam = pipeline.create(dai.node.Camera).build()
    out = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)

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

The subprocess uses the project's **`.venv` interpreter** (`/.venv/Scripts/python.exe`, falling back to TD's Python); that environment must have `depthai` installed.

> Declaring `MODELS` is optional. With none declared, `build()` resolves models itself inline — simplest, but that's the path that briefly hangs the UI.

#### Custom & converted models

Pass a local model straight to the parser node — `ParsingNeuralNetwork.build()` accepts a `dai.NNArchive` and skips the zoo entirely:

```python
MODELS = ['models/mymodel.tar.xz']     # resolves to a file → classified local, no download

def build(pipeline):
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork
    cam     = pipeline.create(dai.node.Camera).build()
    archive = dai.NNArchive(f'{project.folder}/models/mymodel.tar.xz')
    nn      = pipeline.create(ParsingNeuralNetwork).build(cam, archive)
    return ({'rgb': nn.passthrough.createOutputQueue(maxSize=1, blocking=False),
             'out': nn.out.createOutputQueue(maxSize=1, blocking=False)},
            ['rgb', 'out'])
```

A bare blob without parser metadata won't work with `ParsingNeuralNetwork` (it needs the NNArchive config to pick parsers) — use a plain `dai.node.NeuralNetwork` for those.

**Conversion** of a raw model (ONNX/OpenVINO/TFLite → NNArchive for RVC2/RVC4, or a legacy MyriadX blob via `blobconverter` for RVC2 only) is a **one-time offline step** done with Luxonis tooling ([Hub](https://models.luxonis.com/) / `modelconverter`). It can take minutes — never run it inside `build()` on the Start path. Commit the resulting NNArchive and reference it as a local entry in `MODELS`.

### Bundled pipeline scripts

| File | Device | Streams | Controls |
|---|---|---|---|
| `src/TDOAK/examples/pipeline.py` | OAK-D (RGB + stereo depth, synced) | `rgb`, `depth` | `cam_ctrl`, `stereo_cfg` |
| `src/TDOAK/examples/tof_pipeline.py` | OAK-D SR PoE (ToF + aligned color) | `depth`, `color`, `amplitude` | `cam_ctrl`, `tof_cfg` |
| `src/TDOAK/examples/tof_showcase_pipeline.py` | OAK-D SR PoE (ToF depth + amplitude + organized point cloud) | `depth`, `amplitude`, `pointcloud` | `tof_cfg` |
| `src/TDOAK/examples/detection_pipeline.py` | Any RGB OAK (YOLOv6-nano object detection) | `rgb`, `detections` | — |
| `src/TDOAK/examples/pose_pipeline.py` | Any RGB OAK (YOLOv8-nano pose / skeleton — needs `depthai-nodes`) | `rgb`, `pose` | — |
| `src/TDOAK/examples/mono_depth_pipeline.py` | Any RGB OAK (MiDaS v2.1 monocular depth — needs `depthai-nodes`) | `rgb`, `depth` | — |
| `src/TDOAK/examples/mono_depth_input_pipeline.py` | Any OAK (MiDaS depth on **host** frames via `SendFrame` — no camera) | `depth` | `image_in` |

> A standalone harness, `test_pipeline.py`, runs any of these pipeline scripts outside TouchDesigner against a real device (frame shapes, latency, keypoint counts) for quick verification: `python test_pipeline.py src/TDOAK/examples/pose_pipeline.py --device <ip_or_mxid>`.

---

## Threading Model

The camera pump runs in a **background thread** managed by TD's `ThreadManager`. Frame data is transferred to the main thread via a `threading.Lock`-protected dictionary.

- **Background thread** (`cameraPump`): opens the device, **pre-fetches declared models on a subprocess** (streaming download progress into shared state — see [Model Pre-fetch](#model-pre-fetch)), builds and starts the pipeline, polls output queues in a tight loop, **decodes** each message (built-in decoder or a Handler's `parse`), and writes the plain result to the shared dict. On exit (any path) it closes the device in its `finally`, keeping that blocking call off the main thread.
- **Main thread** (`onRefresh`): called by the Thread Manager each cook, reads shared state, updates TD parameters, mirrors per-model download progress into the **Models** sequence, and **fills** Script OPs (push for TOP/CHOP/DAT, pull for SOP via `FillScriptOp`).
- **Model pre-fetch subprocess**: spawned from the background thread with its own interpreter, so the GIL-holding zoo download/check runs without blocking either TD thread. It reports progress as JSON on stdout, which the background thread parses and the main thread reflects.
- **`SendControl`**: calls `InputQueue.send()` directly from the main thread. depthai input queues are designed for cross-thread use; no intermediate queue is needed.

The background thread **must not access any TD objects** (`op()`, `me`, `run()`, parameters, etc.) — this is why a Handler's `parse` (worker thread) must stay pure-Python/numpy while `fill` (main thread) does the operator I/O. Handler DAT text is read on the main thread (it's TD access) and the resulting `parse`/`fill` callables are stored on the extension; the worker reads `self._parsers` fresh per message rather than capturing it, so `ReloadHandlers()` can swap the whole dict on the main thread (an atomic reference reassignment) and the running worker picks it up with no restart and no lock. All TD access happens exclusively in the Thread Manager hooks.
