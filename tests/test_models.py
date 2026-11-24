import pytest

from TDOAK.core.models import declaredModels, scanConstants, timeoutOverride


def test_models_list_with_name_references():
    src = "A = 'luxonis/a:1'\nB: str = 'luxonis/b:2'\nMODELS = [A, B]\n"
    assert declaredModels(scanConstants(src)) == ['luxonis/a:1', 'luxonis/b:2']


def test_bare_model():
    assert declaredModels(scanConstants("MODEL = 'luxonis/x'\n")) == ['luxonis/x']


def test_models_wins_over_model():
    src = "MODEL = 'a'\nMODELS = ['b', 'c']\n"
    assert declaredModels(scanConstants(src)) == ['b', 'c']


def test_computed_values_are_skipped():
    src = "import os\nMODELS = [os.environ['X']]\nN = 3\n"
    consts = scanConstants(src)
    assert declaredModels(consts) == []
    assert consts['N'] == 3


def test_nothing_is_executed():
    scanConstants("raise SystemExit(1)\nMODELS = ['a']\n")


def test_empty_and_none():
    assert scanConstants(None) == {} and scanConstants('') == {}
    assert declaredModels({}) == []


def test_syntax_error_propagates():
    with pytest.raises(SyntaxError):
        scanConstants('def build(:\n')


def test_timeout_override():
    consts = scanConstants('STALL_TIMEOUT_S = 0\nFIRST_FRAME_TIMEOUT_S = 2.5\nBAD = "x"\nFLAG = True\n')
    assert timeoutOverride(consts, 'STALL_TIMEOUT_S', 5) == 0
    assert timeoutOverride(consts, 'FIRST_FRAME_TIMEOUT_S', 15) == 2.5
    assert timeoutOverride(consts, 'MISSING', 5) == 5
    assert timeoutOverride(consts, 'BAD', 5) == 5
    assert timeoutOverride(consts, 'FLAG', 5) == 5
    assert timeoutOverride({'STALL_TIMEOUT_S': -3}, 'STALL_TIMEOUT_S', 5) == 0
