import sys
from types import SimpleNamespace as NS

from conftest import FakeClock

from TDOAK.core.errors import (ErrorLog, PipelineError, Tail, formatTaskError, hint,
                               summarizeCrashDump, withHint)


def excInfo(exc):
    try:
        raise exc
    except Exception:
        return sys.exc_info()


def test_format_none_and_cancelled():
    assert formatTaskError(None) == ''
    assert formatTaskError((None, None, None)) == ''


def test_format_exc_info_points_at_raising_frame():
    msg = formatTaskError(excInfo(ValueError('bad value')))
    assert msg.startswith('ValueError: bad value (test_errors.py:')
    assert msg.endswith(' in excInfo)')


def test_format_exception_instance():
    assert formatTaskError(KeyError('k')).startswith("KeyError: 'k'")


def test_format_pipeline_error_is_bare_message():
    assert formatTaskError(excInfo(PipelineError('no frames for 5 s'))) == 'no frames for 5 s'


def test_format_plain_string():
    assert formatTaskError('boom') == 'boom'


def test_hints():
    assert 'dropped the connection' in hint('QueueException: MessageQueue was closed')
    assert 'did not answer' in hint('RuntimeError: Failed to find device after booting, error message: X_LINK_DEVICE_NOT_FOUND')
    assert 'in use' in hint('X_LINK_ALREADY_OPEN')
    assert 'TOUCH_TEXT_CONSOLE' in hint('no frames within 15 s of start')
    assert 'pre-fetch cannot run' in hint('not valid for use in process: mapping process and mapped file (non-platform) have different Team IDs')
    assert hint('SyntaxError: invalid syntax') == ''


def test_with_hint():
    assert withHint('SyntaxError: x') == 'SyntaxError: x'
    assert withHint('MessageQueue was closed').startswith('MessageQueue was closed - the device')


def test_error_log_channels_and_timestamp():
    clock = FakeClock(1000.0)
    log = ErrorLog(clock=clock)
    log.set('error', 'device lost')
    log.set('warning', 'fill[rgb]: boom')
    assert log.text('error') == 'device lost' and log.text('warning') == 'fill[rgb]: boom'
    shown = log.display('warning')
    assert shown.startswith('[') and shown.endswith('] fill[rgb]: boom')
    clock.advance(60)
    log.set('warning', 'fill[rgb]: boom')          # repeat: first-seen time is kept
    assert log.display('warning') == shown


def test_error_log_pending_dedupes():
    log = ErrorLog()
    for _ in range(100):
        log.set('warning', 'parse[x]: bad')
        log.set('warning', 'parse[y]: bad')
    assert log.takePending() == [('warning', 'parse[x]: bad'), ('warning', 'parse[y]: bad')]
    assert log.takePending() == []
    log.set('warning', 'parse[x]: bad')
    assert log.takePending() == []                 # already printed once


def test_error_log_clear():
    log = ErrorLog()
    log.set('warning', 'fill[rgb]: boom')
    log.clear('warning', ('stream ',))
    assert log.text('warning') == 'fill[rgb]: boom'
    log.clear('warning', ('fill[', 'parse['))
    assert log.text('warning') == '' and log.display('warning') == ''
    log.set('error', 'x')
    log.clear('error')
    assert log.text('error') == ''


def test_tail():
    t = Tail(maxLines=2)
    for line in ('a', '', 'b\n', '  c  '):
        t.add(line)
    assert t.text() == 'b / c'
    long = Tail(maxLines=1, maxChars=10)
    long.add('x' * 50)
    assert long.text() == '...' + 'x' * 10 and long.full() == 'x' * 50
    assert Tail().text() == ''


def test_crash_dump_summary():
    report = NS(errorSource='ErrorSource.RTEMS_FATAL_SOURCE_ASSERT', processor='ProcessorType.LEON_CSS',
                crashedThreadId=7,
                threadCallstack=[NS(threadId=3, threadName='main'), NS(threadId=7, threadName='PlgXLinkIn')],
                errorSourceInfo=NS(assertContext=NS(functionName='int RefCnt::DecRef()')))
    text = summarizeCrashDump(NS(crashReports=[report]))
    assert text == ('device firmware crashed before this connection: RTEMS_FATAL_SOURCE_ASSERT, '
                    'LEON_CSS, thread PlgXLinkIn, assert in int RefCnt::DecRef()')


def test_crash_dump_rvc2_collection_wrapper():
    report = NS(errorSource='RTEMS_FATAL_SOURCE_EXCEPTION', processor='LEON_OS', crashedThreadId=1,
                threadCallstack=[], errorSourceInfo=NS(assertContext=NS(functionName=''),
                                                       trapContext=NS(trapName='data_access')))
    dump = NS(crashReports=NS(crashReports=[report], deviceId='x'))
    assert summarizeCrashDump(dump).endswith('RTEMS_FATAL_SOURCE_EXCEPTION, LEON_OS, trap data_access')


def test_crash_dump_summary_tolerates_missing_fields():
    assert summarizeCrashDump(NS(crashReports=[])) == ''
    assert summarizeCrashDump(NS()) == ''
    assert summarizeCrashDump(NS(crashReports=NS(crashReports=[]))) == ''
    assert 'no readable report' in summarizeCrashDump(NS(crashReports=[NS()]))
    assert 'no readable report' in summarizeCrashDump(NS(data=b'abc'))      # RVC4: opaque payload
