from TDOAK.core import env


def makeVenv(root, osName):
    root.mkdir(parents=True)
    (root / 'pyvenv.cfg').write_text('home = x\n')
    py = env.venvPython(root, osName)
    py.parent.mkdir(parents=True)
    py.write_text('')
    return py


def test_prefers_the_venv_depthai_was_loaded_from(tmp_path):
    py = makeVenv(tmp_path / 'envs' / 'oak', 'posix')
    makeVenv(tmp_path / 'project' / '.venv', 'posix')
    sitePackages = tmp_path / 'envs' / 'oak' / 'lib' / 'python3.11' / 'site-packages'
    sitePackages.mkdir(parents=True)
    so = sitePackages / 'depthai.cpython-311-darwin.so'
    so.write_text('')
    assert env.findPython(so, tmp_path / 'project', 'posix') == str(py.resolve())


def test_windows_layout(tmp_path):
    py = makeVenv(tmp_path / 'venv', 'nt')
    pyd = tmp_path / 'venv' / 'Lib' / 'site-packages' / 'depthai.cp311-win_amd64.pyd'
    pyd.parent.mkdir(parents=True)
    pyd.write_text('')
    found = env.findPython(pyd, None, 'nt')
    assert found == str(py.resolve()) and found.endswith('python.exe')


def test_falls_back_to_project_venv(tmp_path):
    py = makeVenv(tmp_path / 'project' / '.venv', 'posix')
    outside = tmp_path / 'elsewhere' / 'depthai.so'       # not inside any venv
    outside.parent.mkdir()
    outside.write_text('')
    assert env.findPython(outside, tmp_path / 'project', 'posix') == str(py)


def test_none_when_nothing_usable(tmp_path):
    assert env.findPython(None, None) is None
    assert env.findPython(tmp_path / 'x' / 'depthai.so', tmp_path / 'project', 'posix') is None


def test_venv_without_interpreter_is_skipped(tmp_path):
    root = tmp_path / 'broken'
    root.mkdir()
    (root / 'pyvenv.cfg').write_text('')
    so = root / 'lib' / 'depthai.so'
    so.parent.mkdir()
    so.write_text('')
    assert env.findPython(so, None, 'posix') is None


def test_venv_root(tmp_path):
    assert env.venvRoot(None) is None
    assert env.venvRoot(tmp_path / 'nothing' / 'here.py') is None
