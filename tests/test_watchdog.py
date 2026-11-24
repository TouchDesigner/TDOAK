from conftest import FakeClock

from TDOAK.core.watchdog import FrameWatchdog


def make(first=15, stall=5, streams=('rgb', 'depth')):
    clock = FakeClock()
    return FrameWatchdog(first, stall, streams, clock=clock), clock


def test_quiet_before_first_frame_timeout():
    wd, clock = make()
    clock.advance(14.9)
    assert wd.check() is None and not wd.anyFrame


def test_first_frame_timeout():
    wd, clock = make()
    clock.advance(15)
    assert 'no frames within 15 s' in wd.check()


def test_first_frame_late_but_in_time():
    wd, clock = make()
    clock.advance(14)
    wd.mark('rgb')
    clock.advance(4)
    assert wd.check() is None and wd.anyFrame


def test_stall():
    wd, clock = make()
    wd.mark('rgb')
    clock.advance(4.9)
    assert wd.check() is None
    clock.advance(0.2)
    assert 'no frames for 5 s' in wd.check()


def test_any_stream_keeps_it_alive():
    wd, clock = make()
    for _ in range(10):
        clock.advance(3)
        wd.mark('depth')
        assert wd.check() is None


def test_disabled_timeouts_never_fire():
    wd, clock = make(first=0, stall=0)
    clock.advance(3600)
    assert wd.check() is None
    wd.mark('rgb')
    clock.advance(3600)
    assert wd.check() is None
    assert wd.silentStreams() == []


def test_stall_disabled_first_frame_enabled():
    wd, clock = make(first=10, stall=0)
    clock.advance(10)
    assert wd.check() is not None


def test_one_dead_stream_reported_once():
    wd, clock = make()
    wd.mark('rgb')
    wd.mark('depth')
    for _ in range(3):
        clock.advance(2)
        wd.mark('rgb')
    assert wd.silentStreams() == ['depth']
    assert wd.silentStreams() == []
    assert wd.check() is None  # rgb still flows: a warning, not a failure
    wd.mark('depth')           # resumes ...
    for _ in range(3):
        clock.advance(2)
        wd.mark('rgb')
    assert wd.silentStreams() == ['depth']  # ... and is reported again


def test_stream_never_seen_gets_the_first_frame_allowance():
    wd, clock = make(first=15, stall=5)
    for _ in range(7):          # 14 s: depth is still warming up, rgb flows
        clock.advance(2)
        wd.mark('rgb')
        assert wd.silentStreams() == []
    clock.advance(2)            # 16 s: past the first-frame allowance
    wd.mark('rgb')
    assert wd.silentStreams() == ['depth']


def test_stream_never_seen_with_first_frame_check_off_uses_stall():
    wd, clock = make(first=0, stall=5)
    for _ in range(3):
        clock.advance(2)
        wd.mark('rgb')
    assert wd.silentStreams() == ['depth']


def test_no_silent_report_before_any_frame():
    wd, clock = make()
    clock.advance(10)
    assert wd.silentStreams() == []


def test_ages():
    wd, clock = make()
    clock.advance(1)
    wd.mark('rgb')
    clock.advance(2)
    ages = wd.ages()
    assert ages['rgb'] == 2 and ages['depth'] == 3
