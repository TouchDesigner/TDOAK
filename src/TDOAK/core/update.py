"""Check GitHub for a newer TDOAK release (About page > Check for Update).

Check-only: this reports whether a newer release exists and how to get it. It never
installs anything. TDOAK ships as a Python package whose .tox is loaded from inside the
package, so an update is a package upgrade (pip), not a component swap — and depthai is
a native module that can't be replaced in a running TouchDesigner. Installing is left
to the user (or a later, separate step).

Pure Python, no TouchDesigner imports: the network call runs on a ThreadManager worker
thread, and the helpers are unit-testable outside TD.

A private repo answers 404 without credentials, so a token is read from the
TDOAK_GITHUB_TOKEN environment variable — never from a parameter, which would be saved
into the user's .toe.
"""
from __future__ import annotations

import json
import os
import re
import ssl
from pathlib import Path
from typing import TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO        = 'TouchDesigner/TDOAK'
RELEASE_API = f'https://api.github.com/repos/{REPO}/releases/latest'
INSTALL_URL = f'git+https://github.com/{REPO}'
TOKEN_ENV   = 'TDOAK_GITHUB_TOKEN'
TIMEOUT_S   = 15

_VERSION = re.compile(r'v?(\d+)\.(\d+)\.(\d+)')


class Release(TypedDict):
    version: str        # tag without the leading 'v'
    url:     str        # release page
    wheel:   str        # wheel asset name, '' when the release has none


# ── pure helpers ──────────────────────────────────────────────────────────

def versionKey(v: str | None) -> tuple[int, int, int] | None:
    """'0.1.3' or 'v0.1.3' -> (0, 1, 3); None for anything else."""
    m = _VERSION.fullmatch((v or '').strip())
    return tuple(int(x) for x in m.groups()) if m else None  # type: ignore[return-value]


def isNewer(latest: str, current: str) -> bool:
    lk, ck = versionKey(latest), versionKey(current)
    return lk is not None and ck is not None and lk > ck


def isDevCheckout(packageFile: str) -> bool:
    """True when TDOAK is imported from a source tree (e.g. src/ on the path) rather
    than an installed copy in site-packages — pip upgrades don't apply there."""
    parts = {p.lower() for p in Path(packageFile).resolve().parts}
    return not ({'site-packages', 'dist-packages'} & parts)


def parseRelease(payload: dict) -> Release:
    tag = str(payload.get('tag_name') or '').strip()
    wheels = [a.get('name', '') for a in payload.get('assets') or []
              if str(a.get('name', '')).endswith('.whl')]
    return Release(version=tag.lstrip('v'), url=str(payload.get('html_url') or ''),
                   wheel=wheels[0] if wheels else '')


def describe(current: str, release: Release, devCheckout: bool) -> str:
    """One-line Update Status text for a successful check."""
    if versionKey(release['version']) is None:
        return f"Latest release has an unexpected tag: {release['version']!r}"
    if not isNewer(release['version'], current):
        return f"Up to date: {current} (latest release {release['version']})"
    how = ('development checkout: git pull' if devCheckout
           else f'pip install --upgrade {INSTALL_URL}@v{release["version"]}, then restart TouchDesigner')
    return f"Update available: {release['version']} (you have {current}) — {how}"


def reason(exc: BaseException) -> str:
    """Readable failure text for Update Status."""
    if isinstance(exc, HTTPError):
        if exc.code == 404:
            return f'no published release found (private repo? set {TOKEN_ENV})'
        if exc.code == 403:
            return 'GitHub refused the request (rate limit? try again later)'
        return f'GitHub answered {exc.code}'
    if isinstance(exc, URLError):
        return str(exc.reason)
    return f'{type(exc).__name__}: {exc}'


# ── network ───────────────────────────────────────────────────────────────

_SSL_CONTEXT: ssl.SSLContext | None | bool = False  # False = not resolved yet


def sslContext() -> ssl.SSLContext | None:
    """A context with real trust anchors, or None to keep stdlib behaviour.

    TouchDesigner's bundled OpenSSL can report CA paths from its build machine, so the
    default context may load no CAs and every https fetch fails CERTIFICATE_VERIFY_FAILED.
    Fall back to certifi (shipped with TD) only when the default store came up empty."""
    global _SSL_CONTEXT
    if _SSL_CONTEXT is not False:
        return _SSL_CONTEXT  # type: ignore[return-value]
    _SSL_CONTEXT = None
    try:
        ctx = ssl.create_default_context()
        if not ctx.cert_store_stats().get('x509_ca'):
            import certifi
            ctx.load_verify_locations(cafile=certifi.where())
            _SSL_CONTEXT = ctx
    except Exception:
        _SSL_CONTEXT = None  # let urlopen fail with its own, more specific error
    return _SSL_CONTEXT  # type: ignore[return-value]


def fetchLatest(timeout: float = TIMEOUT_S) -> Release:
    """WORKER thread. The latest published GitHub release. Raises on network/HTTP errors."""
    req = Request(RELEASE_API, headers={'Accept': 'application/vnd.github+json',
                                        'User-Agent': 'TDOAK-update-check'})
    token = os.environ.get(TOKEN_ENV, '').strip()
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    ctx = sslContext()
    with (urlopen(req, timeout=timeout, context=ctx) if ctx
          else urlopen(req, timeout=timeout)) as r:
        return parseRelease(json.loads(r.read()))
