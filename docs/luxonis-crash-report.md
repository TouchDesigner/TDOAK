# Bug report: null-pointer AV in depthai-core during device teardown (Windows, embedded Python host)

*File against: `luxonis/depthai-core` (or `depthai-python`). Search open issues for "access violation close" / "crash on device close" first and link this as a comment if one matches.*

---

**Title suggestion:** Access violation (null deref) inside depthaicore during device teardown, right after "XLinkInHost node stopped - exception: MessageQueue was closed" (Windows, depthai 3.6.1 AND 3.8.0, TCP/PoE, ToF pipeline)

## Summary

Stopping a running pipeline/device intermittently crashes the host process with an
`EXCEPTION_ACCESS_VIOLATION` inside `depthaicore`. The faulting instruction reads
address `0x64` — a null object pointer dereferenced at member offset `0x64`. The crash
happens on the host thread that drives the depthai API (the thread calling into
pipeline/device teardown), moments after the normal
`XLinkInHost node stopped - exception: MessageQueue was closed` teardown messages and
while the `DEPTHAI_CRASHDUMP` device-crash-dump check runs (log shows the
`DEPTHAI_CRASHDUMP` / `DEPTHAI_CRASHDUMP_TIMEOUT` env-var probes immediately before the
fault).

## Environment

- **depthai:** reproduced on **3.6.1** (bindings build 2026-05-01, core commit
  `8c0d8eca65dda9e272f93606263d0cf0569c85f2`, libusb enabled) and again on **3.8.0**
  after upgrading — same exception, same call structure (see offset table below)
- **OS:** Windows 11 Pro, build 10.0.26200, x64
- **Host process:** TouchDesigner 2025.33070 (embedded Python 3.11); depthai runs on a
  background thread of this process, plain polling API (`queue.tryGet()`), no callbacks
- **Device:** ToF camera <!-- TODO: fill exact model, e.g. OAK-D SR PoE --> connected
  over **TCP** (PoE — confirmed by winsock frames inside the XLink threads in the dump)
- **Pipeline:** ToF depth + amplitude streams; host output queues created with
  `maxSize=1, blocking=False`

## What happens

On stopping the pipeline (host calls `pipeline.stop()` → `pipeline.wait()` →
`device.close()`, all from the same worker thread that owns the device):

```
[info] XLinkInHost node stopped - exception: MessageQueue was closed
[trace] Received message from device (__x_1_amplitude) - parsing time: 109385µs, data size: 614400, ...
[info] XLinkInHost node stopped - exception: MessageQueue was closed
[info] Environment variable DEPTHAI_CRASHDUMP is not set. Using default value: ''
[info] Environment variable DEPTHAI_CRASHDUMP_TIMEOUT is not set. Using default value: '1'
[info] Environment variable DEPTHAI_CRASHDUMP is not set. Using default value: ''
[info] Environment variable DEPTHAI_CRASHDUMP is not set. Using default value: ''
<host process crashes>
```

Note a 614 KB amplitude message was still mid-parse when the queues closed
(trace-level logging was enabled at the time, which stretches the teardown window).

## Crash details (from the host minidump)

- **Exception:** `EXCEPTION_ACCESS_VIOLATION` (0xC0000005), **read** at address
  **`0x0000000000000064`** → null pointer + member offset 0x64
- **Faulting instruction:** `depthaicore-51763win.dll + 0xaef258`
- **Faulting thread:** the host worker thread that entered depthai via a pybind call

Call chain on the faulting thread (module+offset; you have the symbols for the
depthai wheel DLL `depthaicore-51763win.dll`, version 3.6.1 win-amd64):

```
python311.dll (interpreter frames)
depthai.cp311-win_amd64.pyd + 0xbcb8e      <- pybind trampoline into the API call
depthaicore-51763win.dll + 0x220357
depthaicore-51763win.dll + 0x1af779
depthaicore-51763win.dll + 0x1af98d
depthaicore-51763win.dll + 0x1bfaec
depthaicore-51763win.dll + 0x175bba
depthaicore-51763win.dll + 0x1b0807
depthaicore-51763win.dll + 0x1ad9be
depthaicore-51763win.dll + 0x18cfbb
depthaicore-51763win.dll + 0x151820 / 0x14730c / 0x401e1f
depthaicore-51763win.dll + 0x150ced0 / 0x15849e / 0x158a42
depthaicore-51763win.dll + 0x3f3852
depthaicore-51763win.dll + 0x146b29 / 0x1496ca / 0x150cc78 / 0x150cd60
depthaicore-51763win.dll + 0x3f3bfe
depthaicore-51763win.dll + 0xaeec95
depthaicore-51763win.dll + 0xafa64c
depthaicore-51763win.dll + 0xaf9b49
depthaicore-51763win.dll + 0xaef258      <- FAULT: read [null + 0x64]
```

(Reconstructed by stack scan without PDBs; the attached `.dmp` files allow exact symbol
resolution on your side. The `0xad0000–0xaf0000` region of the DLL also appears in the
XLink connection threads, consistent with the fault living in XLink/connection or the
crash-dump-retrieval code that reconnects to the device after close.)

### Same crash on 3.8.0

After upgrading to depthai 3.8.0 the crash reproduced identically:
`EXCEPTION_ACCESS_VIOLATION`, read at `0x64` (null + 0x64), in the matching code at the
new build's offsets. Frame correspondence between the two dumps:

| 3.6.1 `depthaicore-51763win.dll` | 3.8.0 `depthaicore-26883win.dll` |
|---|---|
| `+0xaef258` (fault) | `+0xb2d1a8` (fault) |
| `+0xaeec95` | `+0xb2cbe5` |
| `+0xafa64c` | `+0xb3859c` |
| `+0xaf9b49` | `+0xb37a99` |
| `+0x3f3bfe` | `+0x41bcfe` |
| `+0x150cd60` / `+0x150cc78` | `+0x154fd60` / `+0x154fc78` |

## Second failure mode in the same window: unhandled exception → abort (3.8.0)

With `DEPTHAI_CRASHDUMP=0` set (so the post-close crash-dump check is skipped), stopping
the same ToF pipeline still occasionally kills the host — this time not via the AV above
but via `std::terminate`/`abort`. Host log at the moment of Stop:

```
[host] [warning] Closed connection
[host] [warning] Attempting to reconnect. Timeout is 10000ms
[depthai] [error] Communication exception - possible device error/misconfiguration. Original message 'Couldn't read data from stream: '__x_1_amplitude' (X_LINK_ERROR)'
[depthai] [error] Communication exception - possible device error/misconfiguration. Original message 'Couldn't read data from stream: '__x_3_dynamicOutputs_0' (X_LINK_ERROR)'
[depthai] [error] Communication exception - possible device error/misconfiguration. Original message 'Couldn't read data from stream: '__x_1__depth' (X_LINK_ERROR)'
[ThreadedNode] [error] Node threw exception, stopping the node. Exception message: Device already closed or disconnected: io error
<abort>
```

The XLink **auto-reconnect** (10000 ms default = `DEPTHAI_RECONNECT_TIMEOUT`) engaged on
the connection that the host itself was closing. A minidump of the aborted process shows:

- a `depthaicore`-owned thread whose entire stack is thread-start →
  `ucrtbase` abort/terminate frames → `RtlRaiseException` — i.e. a C++ exception
  (`Device already closed or disconnected: io error`) escaped a depthai internal thread
  with no handler and hit `std::terminate`;
- the host worker thread suspended mid-`RaiseException` inside a pybind depthai call
  (`depthaicore-26883win.dll+0x236c91`);
- an XLink TCP thread still blocked in `recv` (`mswsock` →
  `depthaicore-26883win.dll+0xb1704d`), consistent with the in-flight reconnect.

So both failure modes live in the same teardown window: anything that re-engages the
connection after the host initiates close (crash-dump check in the first report,
auto-reconnect here) races the destruction of the connection objects. Setting
`DEPTHAI_RECONNECT_TIMEOUT=0` is our current workaround for this second mode.

## Suspicion

The timeline (queues closed → `DEPTHAI_CRASHDUMP` probes → AV) suggests the post-close
crash-dump check (or connection monitor) uses a connection/device object that has
already been destroyed during the same teardown, and dereferences it. Possibly racing
with a large in-flight message still being parsed by XLinkInHost when the queues close.

## Frequency / repro

Update 2026-07-24: reproduced **deterministically outside the host app** — a ~50-line
plain-Python script (venv Python 3.11, depthai 3.8.0, USB OAK-D-Lite this time, not PoE)
that builds Camera+StereoDepth+Sync, streams 15 s, then calls
`pipeline.stop(); pipeline.wait(); device.close()` died with `0xC0000005` on teardown on
3 out of 3 runs, after the `XLinkInHost node stopped - exception: MessageQueue was closed`
messages and the `DEPTHAI_CRASHDUMP` probes. `DEPTHAI_RECONNECT_TIMEOUT=0` did not
prevent it (the crash-dump device check still reconnects). So this is not TouchDesigner-,
ToF-, or PoE-specific.

In the original host app the crash was intermittent — most stops are clean; occasionally
(order of 1 in N stop/rebuild cycles with active ToF streams) it crashes as above. No callbacks are used; all depthai calls
happen on one worker thread except `MessageQueue.send()` on input queues from another
thread (documented thread-safe).

## Attachments

- `TouchDesignerCrash.2025.33070_1.dmp` — host minidump of the depthai **3.6.1** crash,
  with full exception record and thread stacks (written by the host app's crash handler).
- `TouchDesignerCrash.2025.33070_2.dmp` — host minidump of the identical crash on
  depthai **3.8.0**.

## Side note: `DEPTHAI_INSTALL_SIGNAL_HANDLER` default

Worth flagging separately: with the default (handler installed), `import depthai`
replaces the host process's `SetUnhandledExceptionFilter`. In an embedded-Python host
app (TouchDesigner here), that silently disables the host's own crash reporting — the
app just vanishes with no dump, which made this bug invisible for a long time. Consider
defaulting the backward-cpp handler OFF when depthai is loaded as a library into a
process that already has an exception filter installed, or at least documenting this
prominently. We now set `DEPTHAI_INSTALL_SIGNAL_HANDLER=0` before import to get usable
host crash dumps (which is how this report's dump was captured).
