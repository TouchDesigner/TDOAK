"""TouchDesigner extension classes of the TDOAK components.

The extension code lives HERE, in regular package modules, not in DATs inside the .tox.
Each component's extension DAT is a few-line shim that calls create(). Two reasons:

- A DAT module is cleared and re-executed in place when its text changes. A camera worker
  thread running code from that module loses its globals mid-flight (NameError inside the
  device teardown) — the cause of the 'Start does nothing until TouchDesigner restarts'
  bug. A package module is never recompiled behind a running thread's back.
- The .tox no longer depends on a file next to it, so it works dragged into any project,
  and its code always is the installed package's version.

Development: edit a module, then pulse Re-Init Extensions on the component — create()
reloads every TDOAK.core / TDOAK.ext module whose file changed since it was loaded.
"""
from __future__ import annotations

import importlib
import os
import sys

_PREFIXES = ('TDOAK.core.', 'TDOAK.ext.')


def _mtime(module: object) -> float | None:
    path = getattr(module, '__file__', None)
    try:
        return os.path.getmtime(path) if path else None
    except OSError:
        return None


def reloadChanged() -> list[str]:
    """Reload the TDOAK.core / TDOAK.ext modules when any of their files changed since
    it was loaded. Core modules reload before ext modules (ext imports names from core);
    importlib.reload re-executes into the SAME module dict without clearing it, so code
    still running from the old version keeps valid globals. Returns the reloaded names."""
    modules = [(name, mod) for name, mod in sorted(sys.modules.items())
               if name.startswith(_PREFIXES) and mod is not None]
    changed = False
    for _name, mod in modules:
        seen = getattr(mod, '_tdoakMtime', None)
        now = _mtime(mod)
        if seen is None:
            mod._tdoakMtime = now  # type: ignore[attr-defined]
        elif now is not None and now != seen:
            changed = True
    if not changed:
        return []
    reloaded = []
    for name, mod in modules:  # sorted: 'TDOAK.core.*' before 'TDOAK.ext.*'
        importlib.reload(mod)
        mod._tdoakMtime = _mtime(mod)  # type: ignore[attr-defined]
        reloaded.append(name)
    return reloaded


def create(className: str, ownerComp: object) -> object:
    """Build the extension instance for a component: the class `className` from the module
    TDOAK.ext.<className>. Called by the component's extension shim DAT."""
    reloadChanged()
    module = importlib.import_module(f'TDOAK.ext.{className}')
    if getattr(module, '_tdoakMtime', None) is None:
        module._tdoakMtime = _mtime(module)  # type: ignore[attr-defined]
    return getattr(module, className)(ownerComp)
