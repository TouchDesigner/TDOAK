from TDOAK.core.retry import RetryPolicy


def test_off_never_retries():
    p = RetryPolicy('off')
    assert p.onFailure() is None


def test_limited_gives_up_on_third_failure():
    p = RetryPolicy('limited')
    assert p.onFailure() == 5
    assert p.describe() == 'attempt 2 of 3'
    assert p.onFailure() == 10
    assert p.onFailure() is None


def test_forever_backs_off_to_cap():
    p = RetryPolicy('forever')
    assert [p.onFailure() for _ in range(7)] == [5, 10, 20, 40, 60, 60, 60]
    assert p.describe() == 'attempt 8'


def test_healthy_run_resets_counter():
    p = RetryPolicy('limited')
    p.onFailure()
    p.onFailure()
    assert p.onFailure(ranFor=31) == 5      # streamed long enough: counts as failure 1
    assert p.failures == 1


def test_short_runs_do_not_reset():
    p = RetryPolicy('limited')
    p.onFailure(ranFor=29)
    p.onFailure(ranFor=29)
    assert p.onFailure(ranFor=29) is None


def test_reset():
    p = RetryPolicy('limited')
    p.onFailure()
    p.onFailure()
    p.reset()
    assert p.onFailure() == 5


def test_unknown_mode_falls_back_to_limited():
    assert RetryPolicy('bogus').mode == 'limited'
