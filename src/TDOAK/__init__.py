from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = [
    '__version__',
    '__td_requires_touchdesigner__',
    '__td_package_type__',
    '__td_collection__',
    'assetsPath',
    'examplesPath',
    'toxPath',
    'toxPaths',
]

__version__ = "0.1.1"
__td_requires_touchdesigner__ = "2025.33070"
__td_package_type__ = ""
__td_collection__ = {
    "TDOAK": Path(str(resources.files(__package__) / "TDOAK.tox")),
    "TDOAKCamCtl": Path(str(resources.files(__package__) / "TDOAKCamCtl.tox")),
}


def toxPath(name: str | None = None) -> Path:
    """Return the path to the packaged root .tox file."""
    key = "TDOAK" if name is None else str(name)
    try:
        return __td_collection__[key]
    except KeyError as exc:
        raise KeyError(f"Unknown packaged .tox: {key}") from exc


def toxPaths() -> dict[str, Path]:
    """Return all packaged root .tox files keyed by name."""
    return dict(__td_collection__)


def assetsPath() -> Path:
    """Return the path to the packaged assets directory."""
    return Path(str(resources.files(__package__) / "assets"))


def examplesPath() -> Path:
    """Return the path to the packaged examples directory (pipeline/handler DAT scripts)."""
    return Path(str(resources.files(__package__) / "examples"))
