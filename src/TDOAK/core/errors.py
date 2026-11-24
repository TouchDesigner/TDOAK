"""Error reporting helpers (pure Python).

Two channels, shown in two parameters:
  error   — why the pipeline is not running (a failed start, a lost device, a stall).
  warning — something is wrong but the pipeline keeps running (a handler raised, a stream
            went quiet, a USB2 link, a model download that failed).
Both carry the time they happened, so a message that is still on screen hours later is
recognisably old instead of looking current.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Callable, Iterable


class PipelineError(RuntimeError):
    """A run-ending condition TDOAK itself detected (stall, lost device, ...). Its text is
    complete as written — no exception type or source location is added when shown."""


def formatTaskError(exc: object) -> str:
    """Turn a ThreadManager ExceptHook payload into a one-line message.
    The hook receives sys.exc_info() — a (type, value, traceback) tuple — or None when a
    task is cancelled. str() of the raw tuple is unreadable, so render it as
    'Type: message (file:line in func)', pointing at the innermost frame that raised.
    A PipelineError is rendered as its bare message."""
    if not exc:
        return ''
    if isinstance(exc, BaseException):
        exc = (type(exc), exc, exc.__traceback__)
    if not isinstance(exc, tuple):
        return str(exc)
    excType, excValue, tb = (tuple(exc) + (None, None, None))[:3]
    if excType is None and excValue is None:
        return ''
    if isinstance(excValue, PipelineError):
        return str(excValue)
    name = getattr(excType, '__name__', None) or type(excValue).__name__
    detail = str(excValue) if excValue is not None else ''
    msg = f'{name}: {detail}' if detail else name
    if tb is not None:
        while tb.tb_next is not None:
            tb = tb.tb_next
        code = tb.tb_frame.f_code
        where = os.path.basename(code.co_filename) or code.co_filename
        msg += f' ({where}:{tb.tb_lineno} in {code.co_name})'
    return msg


# (substrings matched case-insensitively, advice). First match wins.
_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (('messagequeue was closed', 'x_link_error', "couldn't read data from stream",
      'pipeline stopped by itself', 'device already closed', 'communication exception'),
     'the device dropped the connection: firmware crash, power dip or a bad cable. Check '
     'the cable and port (USB3, short, direct or powered hub); with stereo depth keep '
     'Auto Calibration Off'),
    (('input/output error', 'usb i/o', 'libusb', 'usb error', 'failed to boot',
      'x_link_device_not_found', 'failed to find device', 'x_link_timeout',
      'cannot find any device', 'no available devices', 'device not found'),
     'the device did not answer: replug it, check the cable/port (USB) or network and '
     'power (PoE), and make sure no other program has it open'),
    (('x_link_already_open', 'already in use', 'already booted', 'insufficient permissions'),
     'the device is in use or not accessible: close other programs using it (OAK Viewer, '
     'another TouchDesigner) and replug'),
    (('out of memory', 'not enough memory'),
     'the device ran out of memory: lower resolutions/fps or remove nodes'),
    (('no frames within', 'no frames for'),
     'run with the environment variable TOUCH_TEXT_CONSOLE=1 to see what depthai reports'),
    (('different team ids',),
     'macOS refuses to load depthai in a stand-alone Python started from this environment, '
     'so the model pre-fetch cannot run - the model is fetched in-process instead (the UI '
     'may hitch during the first download)'),
    (('no module named', 'modulenotfounderror'),
     'a Python package is missing from the environment TouchDesigner uses - see Quick Start'),
)


def hint(message: str) -> str:
    """One line of advice for a known failure text, or ''."""
    low = message.lower()
    for needles, advice in _HINTS:
        if any(n in low for n in needles):
            return advice
    return ''


def withHint(message: str) -> str:
    advice = hint(message)
    return f'{message} - {advice}' if advice else message


class ErrorLog:
    """Thread-safe-by-caller store for the two channels (the extension holds its lock
    around every call). Records when each message was set and which messages have not
    been printed yet, so a per-frame error floods neither the parameter nor the textport."""

    KINDS = ('error', 'warning')

    def __init__(self, clock: Callable[[], float] = time.time,
                 maxRemembered: int = 200) -> None:
        self._clock = clock
        self._text = {kind: '' for kind in self.KINDS}
        self._stamp = {kind: 0.0 for kind in self.KINDS}
        self._printed: deque[str] = deque(maxlen=maxRemembered)
        self._pending: list[tuple[str, str]] = []

    def set(self, kind: str, message: str) -> None:
        """Record a message ('' clears the channel). The timestamp only moves when the
        text changes, so a repeating per-frame error keeps its first-seen time."""
        if message == self._text[kind]:
            return
        self._text[kind] = message
        self._stamp[kind] = self._clock() if message else 0.0
        if message and message not in self._printed:
            self._printed.append(message)
            self._pending.append((kind, message))

    def clear(self, kind: str, prefixes: Iterable[str] = ()) -> None:
        """Clear a channel — or, with prefixes, only when its message starts with one."""
        prefixes = tuple(prefixes)
        if not prefixes or self._text[kind].startswith(prefixes):
            self.set(kind, '')

    def text(self, kind: str) -> str:
        return self._text[kind]

    def display(self, kind: str) -> str:
        """Parameter text: '[HH:MM:SS] message', or '' when the channel is clear."""
        message = self._text[kind]
        if not message:
            return ''
        return f"[{time.strftime('%H:%M:%S', time.localtime(self._stamp[kind]))}] {message}"

    def takePending(self) -> list[tuple[str, str]]:
        """Messages recorded since the last call that were never printed before."""
        pending, self._pending = self._pending, []
        return pending


class Tail:
    """Keeps the last few non-empty lines of a text stream (a child process's output), so
    a failure can say WHY instead of just 'exit 1'."""

    def __init__(self, maxLines: int = 4, maxChars: int = 300) -> None:
        self._lines: deque[str] = deque(maxlen=maxLines)
        self._maxChars = maxChars

    def add(self, line: str) -> None:
        line = line.strip()
        if line:
            self._lines.append(line)

    def full(self) -> str:
        return ' / '.join(self._lines)

    def text(self) -> str:
        joined = ' / '.join(self._lines)
        return joined if len(joined) <= self._maxChars else '...' + joined[-self._maxChars:]


def summarizeCrashDump(dump: object) -> str:
    """One line from a depthai CrashDump: error source, processor and crashed thread of
    each report ('' when the dump holds nothing). RVC2 dumps carry structured reports
    (dump.crashReports is a collection that in turn holds the list); other platforms
    carry an opaque payload, which is only acknowledged. Every field is optional —
    the layout differs between depthai versions and platforms."""
    reports = getattr(dump, 'crashReports', None)
    reports = getattr(reports, 'crashReports', reports)  # unwrap CrashReportCollection
    try:
        reports = list(reports or [])
    except TypeError:
        reports = []
    parts = []
    for report in reports:
        fields = []
        source = getattr(report, 'errorSource', None)
        if source is not None:
            fields.append(str(source).split('.')[-1])
        processor = getattr(report, 'processor', None)
        if processor is not None:
            fields.append(str(processor).split('.')[-1])
        threadId = getattr(report, 'crashedThreadId', None)
        for thread in getattr(report, 'threadCallstack', None) or []:
            if getattr(thread, 'threadId', object()) == threadId:
                name = getattr(thread, 'threadName', '')
                if name:
                    fields.append(f'thread {name}')
                break
        info = getattr(report, 'errorSourceInfo', None)
        func = getattr(getattr(info, 'assertContext', None), 'functionName', '')
        trap = getattr(getattr(info, 'trapContext', None), 'trapName', '')
        if func:
            fields.append(f'assert in {func}')
        elif trap:
            fields.append(f'trap {trap}')
        if fields:
            parts.append(', '.join(fields))
    prefix = 'device firmware crashed before this connection'
    if parts:
        return f'{prefix}: ' + '; '.join(parts)
    if reports or getattr(dump, 'data', None):
        return f'{prefix} (crash dump present, no readable report)'
    return ''
