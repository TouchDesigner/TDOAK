import threading

from TDOAK.core import state


def freshState(name):
    state._workerState.pop(name, None)
    return state.workerState(name)


def deadThread():
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    return t


def test_same_dict_per_component():
    s = freshState('/a')
    assert state.workerState('/a') is s
    assert state.workerState('/b') is not s
    assert set(s) >= {'alive', 'stop', 'thread', 'owner', 'resume'}
    assert s['resume'] == 0.0


def test_upgrades_a_dict_from_an_older_version():
    state._workerState['/old'] = {'alive': threading.Event(), 'stop': threading.Event()}
    s = state.workerState('/old')
    assert s['thread'] is None and s['owner'] is None


def test_not_alive_when_flag_unset():
    assert state.isWorkerAlive(freshState('/a')) is False


def test_alive_without_recorded_thread_is_trusted():
    s = freshState('/a')
    s['alive'].set()
    assert state.isWorkerAlive(s) is True


def test_alive_with_running_thread():
    s = freshState('/a')
    gate = threading.Event()
    t = threading.Thread(target=gate.wait)
    t.start()
    try:
        s['alive'].set()
        s['thread'] = t
        assert state.isWorkerAlive(s) is True
    finally:
        gate.set()
        t.join()


def test_stale_flag_is_cleared():
    s = freshState('/a')
    s['alive'].set()
    s['thread'] = deadThread()
    assert state.isWorkerAlive(s) is False
    assert not s['alive'].is_set()
    assert s['thread'] is None


def test_owner():
    s = freshState('/a')
    a, b = object(), object()
    assert not state.isOwner(s, a)
    state.claim(s, a)
    assert state.isOwner(s, a) and not state.isOwner(s, b)
    state.claim(s, b)
    assert state.isOwner(s, b) and not state.isOwner(s, a)


def test_stop_all_sets_every_stop_flag():
    a, b = freshState('/a'), freshState('/b')
    state.stopAll()
    assert a['stop'].is_set() and b['stop'].is_set()


def test_reload_keeps_registry():
    import importlib
    s = freshState('/keep')
    importlib.reload(state)
    assert state.workerState('/keep') is s


def test_resume_request_expires(monkeypatch):
    s = freshState('/r')
    assert not state.resumeRequested(s)
    now = [50.0]
    monkeypatch.setattr(state.time, 'monotonic', lambda: now[0])
    state.requestResume(s, within=10)
    assert state.resumeRequested(s)
    now[0] = 59.9
    assert state.resumeRequested(s)
    now[0] = 60.1
    assert not state.resumeRequested(s)


def test_resume_cancel():
    s = freshState('/r')
    state.requestResume(s)
    state.cancelResume(s)
    assert not state.resumeRequested(s)


def test_old_boolean_resume_is_upgraded():
    state._workerState['/b'] = {'resume': True}
    assert state.workerState('/b')['resume'] == 0.0
