from __future__ import annotations

import atexit
import threading
import time

# Process-lifetime worker state per OAK component, keyed by component path:
#   'alive'  — Event: set while a camera pump is running (cleared in its finally)
#   'stop'   — Event: the CURRENT run's stop flag (replaced with a fresh Event per run)
#   'thread' — the Thread running the current pump (None until it is enqueued). The Event
#              alone can lie: if a worker dies without reaching its finally, 'alive' would
#              stay set forever and every Start would be a silent no-op. isWorkerAlive()
#              cross-checks the Event against the thread and clears a stale flag.
#   'owner'  — the extension instance that currently owns the component. A re-init creates
#              a new instance; delayed callbacks and ThreadManager hooks of the OLD instance
#              check this and stand down.
#   'resume' — a time.monotonic() deadline: until then, a restart is owed because a re-init
#              interrupted a running pipeline (0.0 = none). It lives here, not on an
#              instance, because the instance that stopped the run (in onDestroyTD, or a
#              first of several inits for one reload) is not the one that restarts it. A
#              deadline rather than a flag, so a component that was destroyed for good
#              (deleted, project closed) can't leave a stale 'please restart' behind.
# It lives HERE — a real installed module, not the extension DAT — so it survives both
# extension re-inits and DAT edits/reloads. A freshly re-initialized OAKManagerExt must
# still see (and be able to stop) a previous instance's still-running worker: starting a
# new pump while the old one is mid device.close() races depthai's native teardown and
# can hard-crash the process.
#
# The registry survives importlib.reload(state) too (development convenience): a reload
# must never orphan a running worker's flags.
_workerState: dict[str, dict[str, object]] = globals().get('_workerState', {})


def workerState(compPath: str) -> dict[str, object]:
    """Return the shared worker-state dict for an OAK component, creating it on first
    use. The same dict is handed to every extension instance created for that component."""
    state = _workerState.setdefault(compPath, {})
    state.setdefault('alive', threading.Event())
    state.setdefault('stop', threading.Event())
    state.setdefault('thread', None)
    state.setdefault('owner', None)
    if not isinstance(state.get('resume'), float):
        state['resume'] = 0.0
    return state


def isWorkerAlive(state: dict[str, object]) -> bool:
    """True while a pump is running (or booked to run). Clears a STALE 'alive' flag — set,
    but the recorded thread has exited — and returns False, so a worker that died without
    clearing the flag can never block Start forever."""
    alive: threading.Event = state['alive']  # type: ignore[assignment]
    if not alive.is_set():
        return False
    thread = state.get('thread')
    if thread is not None and not thread.is_alive():  # type: ignore[attr-defined]
        alive.clear()
        state['thread'] = None
        return False
    return True


def claim(state: dict[str, object], owner: object) -> None:
    """Make `owner` (an extension instance) the component's current owner."""
    state['owner'] = owner


def isOwner(state: dict[str, object], owner: object) -> bool:
    return state.get('owner') is owner


def requestResume(state: dict[str, object], within: float = 10.0) -> None:
    """Note that the pipeline should be started again by whichever extension instance
    initializes within the next `within` seconds."""
    state['resume'] = time.monotonic() + within


def resumeRequested(state: dict[str, object]) -> bool:
    return float(state.get('resume') or 0.0) > time.monotonic()  # type: ignore[arg-type]


def cancelResume(state: dict[str, object]) -> None:
    state['resume'] = 0.0


def stopAll() -> None:
    """Set every component's current stop flag. Registered with atexit so a pump still
    running when the interpreter shuts down exits its loop instead of holding the device."""
    for state in list(_workerState.values()):
        stop = state.get('stop')
        if stop is not None:
            stop.set()  # type: ignore[attr-defined]


if not globals().get('_atexitRegistered'):
    # Interpreter shutdown joins every non-daemon thread BEFORE ordinary atexit callbacks
    # run, so a plain atexit hook would come too late to release a pump that is still
    # polling. threading._register_atexit (CPython 3.9+) runs ahead of that join.
    _early = getattr(threading, '_register_atexit', None)
    try:
        if _early is not None:
            _early(stopAll)
    except RuntimeError:
        pass  # interpreter already shutting down
    atexit.register(stopAll)
    _atexitRegistered = True
