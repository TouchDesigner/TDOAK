from TDOAK.core import compat


def test_parse_version():
    assert compat.parseVersion('3.12.2') == (3, 12, 2)
    assert compat.parseVersion('3.12') == (3, 12, 0)
    assert compat.parseVersion('3.13.0.dev0+abc') == (3, 13, 0)
    assert compat.parseVersion('unknown') is None
    assert compat.parseVersion(None) is None


def test_is_tested():
    assert compat.isTested('3.12.0') and compat.isTested('3.12.9')
    assert not compat.isTested('3.11.4') and not compat.isTested('3.13.0')
    assert not compat.isTested('') and not compat.isTested('garbage')


def test_status_text():
    assert compat.depthaiStatus('3.12.2') == '3.12.2 (tested)'
    assert 'outside the tested range 3.12.x' in compat.depthaiStatus('3.8.0')
    assert compat.depthaiStatus('') == ''
    assert 'unrecognised' in compat.depthaiStatus('unknown')


def test_tested_range_matches_the_pyproject_pin():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text(encoding='utf-8')
    lo, hi = compat.TESTED_DEPTHAI
    assert f'"depthai>={lo[0]}.{lo[1]},<{hi[0]}.{hi[1]}"' in text
