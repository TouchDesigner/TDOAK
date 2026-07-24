# Bug report: regression 3.6.1 → 3.7.1/3.8.0 — streaming a mono camera crashes OAK-D-Lite device firmware within ~2 s

*File against: `luxonis/depthai-core`. Separate issue from the host-side teardown AV report — this one is a device-side firmware crash with a deterministic single-node repro and a clean version bisection.*

---

**Title suggestion:** Regression 3.6.1 → 3.7.1: streaming CAM_B mono (OV7251) on OAK-D-Lite crashes device firmware ~1–2 s after `pipeline.start()` (all XLink streams incl. `__watchdog` drop at once)

## Summary

On depthai **3.7.1 and 3.8.0**, streaming a mono camera on an **OAK-D-Lite** (USB3) kills
the device firmware roughly 1–2 seconds after `pipeline.start()`: every XLink stream —
including the internal `__telemetry`, `__timesync`, `__log` and `__watchdog` channels —
drops with `X_LINK_ERROR` simultaneously, and the host-side monitor occasionally receives
garbage as the device dies (`Monitor thread exception caught: [json.exception.type_error.316]
invalid UTF-8 byte at index 27: 0x80`).

On **3.6.1**, the identical script on the identical device/host/port/cable streams
indefinitely at full rate. Verified back-to-back, repeatedly, minutes apart.

The color camera is unaffected: an RGB-only pipeline on 3.8.0 streams clean at 30 fps.
The trigger is the mono path — a single `Camera` node on CAM_B is sufficient.

## Environment

- **depthai:** crashes on **3.7.1** (2/2 runs) and **3.8.0** (bindings build 2026-07-10,
  core commit `1aba5e372326c6160b77445b0007325f52907e0b`; ~10 crashed runs, one outlier
  clean run); stable on **3.6.1** (3/3 runs, plus long-running production use).
  No 3.7.0 exists on PyPI, so 3.6.1 → 3.7.1 is the tightest public bisection.
- **Device:** OAK-D-Lite over USB3 (SuperSpeed confirmed via `getUsbSpeed()`), RVC2,
  deviceId `184430101165C80F00`; 3.8.0 runs boot the wheel's bundled
  `depthai-device-fwp-c3f4bc9cef3937fa987135129757d950d1fbbd19`
- **OS / host:** Windows 11 Pro build 10.0.26200 x64, plain CPython 3.11 venv
- Hardware ruled out: same unit streams stereo fine on another machine with an older
  depthai, and on this machine under 3.6.1 immediately before/after failing 3.7.1/3.8.0
  runs; cold power-cycles between tests don't change the outcome.

## Minimal repro

```python
import time
import depthai as dai

device   = dai.Device()             # OAK-D-Lite, USB3
pipeline = dai.Pipeline(device)
cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
q = cam.requestFullResolutionOutput().createOutputQueue(maxSize=1, blocking=False)
pipeline.start()
n, t0 = 0, time.monotonic()
while time.monotonic() - t0 < 10 and pipeline.isRunning():
    if q.tryGet() is not None:
        n += 1
print('frames:', n)
```

Results (same host, device, port, cable; `DEPTHAI_AUTOCALIBRATION=OFF` everywhere to
exclude the auto-injected node as a variable):

| depthai | CAM_B alone | CAM_C alone | both monos | RGB only | full stereo pipeline |
|---|---|---|---|---|---|
| 3.6.1 | **OK** ~30 fps (3/3) | — | — | — | **OK** ~30 fps |
| 3.7.1 | **crash** (2/2) | — | — | — | — |
| 3.8.0 | **crash** (3/3+) | OK ~30 fps | crash | OK ~30 fps | crash (one outlier clean run in ~8) |

The one clean 3.8.0 stereo run suggests a race rather than a hard config error — the
overwhelming majority of runs die 1–2 s in.

## Log (3.8.0, single CAM_B node, condensed)

```
[Camera(0)] [debug] Sensor type MONO selected automatically for camera on socket CAM_B. Officially supported sensor types: [MONO]
[Camera(0)] [debug] Best camera config selected: 640x480 @ 30fps
<pipeline start, ~1-2 s of normal operation>
[depthai] [error] Communication exception - possible device error/misconfiguration. Original message 'Couldn't read data from stream: '__x_0_0' (X_LINK_ERROR)'
[host] [info] Monitor thread exception caught: [json.exception.type_error.316] invalid UTF-8 byte at index 27: 0x80
[host] [debug] Watchdog thread exception caught: Couldn't write data to stream: '__watchdog' (X_LINK_ERROR)
```

The watchdog stream dying together with everything else indicates the device itself went
down, not a single stream/queue problem. `DEPTHAI_CRASHDUMP` retrieval did not produce a
device crash dump ("Device likely crashed but did not reboot in time to get the crash
dump" on the runs where retrieval was attempted).

## Side observation: AutoCalibration engages outside its documented support

While isolating this we noticed that on 3.6.1 the default-on AutoCalibration node logs
`AutoCalibration: Not initialized - currently supports only sensors with 1280x800
resolution.` and deactivates on the OAK-D-Lite's 640x480 monos, whereas 3.8.0 shows no
such message and proceeds (`AutoCalibration started to work!`), although the current
release notes still state AutoCalibration "initializes only for stereo inputs at
1280x800". This is not the cause of the crash above (it reproduces with
`DEPTHAI_AUTOCALIBRATION=OFF` and a single mono node) but looks like a lost guard worth
checking independently. Also minor: the release notes document `DEPTHAI_AUTOCALIBRATION=OFF`,
and `0` is rejected with a warning — but `0` still appears to disable the injection,
which is confusing.

## Expected behavior

Mono camera streaming on OAK-D-Lite works on 3.7.1+ as it does on 3.6.1, or at minimum
fails with a diagnosable error instead of a silent firmware death that takes the watchdog
down with it.

## Workaround

Pin `depthai==3.6.1`.
