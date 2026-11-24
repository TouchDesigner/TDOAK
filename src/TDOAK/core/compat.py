"""Which depthai versions this TDOAK release was tested against (pure Python).

depthai 3.x has shipped regressions between minor versions (device firmware crashes,
frame-type support, auto-calibration behaviour), so the package pins the range that went
through the hardware checklist and the About page says when the loaded version is outside
it. Keep TESTED_DEPTHAI in step with the depthai pin in pyproject.toml.
"""
from __future__ import annotations

import re

TESTED_DEPTHAI = ((3, 12, 0), (3, 13, 0))   # [min, max) — the pyproject pin: >=3.12,<3.13
TESTED_LABEL   = '3.12.x'

_VERSION = re.compile(r'(\d+)\.(\d+)(?:\.(\d+))?')


def parseVersion(text: str | None) -> tuple[int, int, int] | None:
    """'3.12.2' / '3.12.2.dev0+abc' -> (3, 12, 2); None when there is no version in it."""
    m = _VERSION.match((text or '').strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def isTested(version: str | None) -> bool:
    parsed = parseVersion(version)
    return parsed is not None and TESTED_DEPTHAI[0] <= parsed < TESTED_DEPTHAI[1]


def depthaiStatus(version: str | None) -> str:
    """Text for the About page's Depthai Version readout."""
    if not version:
        return ''
    if parseVersion(version) is None:
        return f'{version} (unrecognised version)'
    if isTested(version):
        return f'{version} (tested)'
    return f'{version} - outside the tested range {TESTED_LABEL}; expect differences'
