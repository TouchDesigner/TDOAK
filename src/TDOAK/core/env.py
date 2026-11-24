"""Finding a Python interpreter for the model pre-fetch subprocess (pure Python).

The pre-fetch must run in a SEPARATE process (its own GIL) that can import depthai. The
right interpreter is the one of the virtual environment depthai was loaded from; the
project's .venv is the fallback. TouchDesigner's own sys.executable is never used: inside
TouchDesigner it can be the application binary itself, and starting that with `-c ...`
would launch a second TouchDesigner instead of a Python script.
"""
from __future__ import annotations

import os
from pathlib import Path


def venvPython(root: str | os.PathLike, osName: str = os.name) -> Path:
    """The interpreter path inside a virtual environment root, per platform."""
    root = Path(root)
    return root / 'Scripts' / 'python.exe' if osName == 'nt' else root / 'bin' / 'python'


def venvRoot(moduleFile: str | os.PathLike | None) -> Path | None:
    """The virtual environment a module file lives in: the nearest parent directory that
    holds a pyvenv.cfg. None when the module isn't inside a virtual environment."""
    if not moduleFile:
        return None
    for parent in Path(moduleFile).resolve().parents:
        if (parent / 'pyvenv.cfg').is_file():
            return parent
    return None


def findPython(depthaiFile: str | os.PathLike | None, projectFolder: str | os.PathLike | None,
               osName: str = os.name) -> str | None:
    """Interpreter for the pre-fetch subprocess, or None when there is none to use (the
    caller then skips the pre-fetch and lets build() resolve models in-process).

    1. the virtual environment depthai was imported from — wherever it is;
    2. <projectFolder>/.venv."""
    candidates = []
    root = venvRoot(depthaiFile)
    if root is not None:
        candidates.append(venvPython(root, osName))
    if projectFolder:
        candidates.append(venvPython(Path(projectFolder) / '.venv', osName))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None
