"""TouchDesigner OAK camera integration"""

from __future__ import annotations

import threading

__version__ = "0.1.0"

# Process-lifetime registry of camera-pump liveness Events, keyed by component path.
# It lives HERE — a real installed module, not the extension DAT — so it survives both
# extension re-inits and DAT edits/reloads. A freshly re-initialized OAKManagerExt must
# still see a previous instance's still-running worker: starting a new pump while the old
# one is mid device.close() races depthai's native teardown and can hard-crash the process.
_workerAlive: dict[str, threading.Event] = {}


def workerAliveEvent(compPath: str) -> threading.Event:
    """Return the worker-liveness Event for an OAK component, creating it on first use.
    The same Event is handed to every extension instance created for that component."""
    return _workerAlive.setdefault(compPath, threading.Event())
