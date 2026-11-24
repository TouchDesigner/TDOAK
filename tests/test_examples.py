"""Static checks of the bundled pipeline / handler scripts (nothing is executed)."""
import ast
from pathlib import Path

import pytest

from TDOAK.core.models import declaredModels, scanConstants

EXAMPLES = Path(__file__).resolve().parents[1] / 'src' / 'TDOAK' / 'examples'
SCRIPTS = sorted(EXAMPLES.glob('*.py'))


def source(path):
    return path.read_text(encoding='utf-8-sig')


def functions(path):
    return {n.name for n in ast.parse(source(path)).body if isinstance(n, ast.FunctionDef)}


def test_examples_are_found():
    assert len(SCRIPTS) >= 20


@pytest.mark.parametrize('path', SCRIPTS, ids=lambda p: p.name)
def test_script_parses(path):
    ast.parse(source(path))


@pytest.mark.parametrize('path', [p for p in SCRIPTS if p.name.endswith('pipeline.py')], ids=lambda p: p.name)
def test_pipeline_defines_build_and_scans(path):
    assert 'build' in functions(path)
    constants = scanConstants(source(path))
    for model in declaredModels(constants):
        assert isinstance(model, str) and model


@pytest.mark.parametrize('path', [p for p in SCRIPTS if p.name.endswith('_handler.py')], ids=lambda p: p.name)
def test_handler_defines_parse_or_fill(path):
    assert functions(path) & {'parse', 'fill'}


def test_host_frame_pipeline_switches_the_watchdog_off():
    constants = scanConstants(source(EXAMPLES / 'detection_input_pipeline.py'))
    assert constants['STALL_TIMEOUT_S'] == 0 and constants['FIRST_FRAME_TIMEOUT_S'] == 0


def test_no_pipeline_requests_an_explicit_frame_type():
    # An explicit ImgFrame type in requestOutput() is not supported by every camera
    # (it left an OAK4 at 'Connected, 0 fps'). Bundled pipelines take the camera default.
    for path in SCRIPTS:
        for node in ast.walk(ast.parse(source(path))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'requestOutput'):
                assert len(node.args) <= 1 and not any(k.arg == 'type' for k in node.keywords), path.name
