from __future__ import annotations

import threading

# Process-lifetime worker state per OAK component, keyed by component path:
#   'alive' — Event: set while a camera pump is running (cleared in its finally)
#   'stop'  — Event: the CURRENT run's stop flag (replaced with a fresh Event per run)
# It lives HERE — a real installed module, not the extension DAT — so it survives both
# extension re-inits and DAT edits/reloads. A freshly re-initialized OAKManagerExt must
# still see (and be able to stop) a previous instance's still-running worker: starting a
# new pump while the old one is mid device.close() races depthai's native teardown and
# can hard-crash the process.
_workerState: dict[str, dict[str, threading.Event]] = {}


def workerState(compPath: str) -> dict[str, threading.Event]:
    """Return the shared worker-state dict for an OAK component, creating it on first
    use. The same dict is handed to every extension instance created for that component."""
    return _workerState.setdefault(
        compPath, {'alive': threading.Event(), 'stop': threading.Event()})
