import importlib
import os
import sys
import types

import TDOAK.ext as loader


def test_reload_only_when_a_file_changed(tmp_path, monkeypatch):
    import TDOAK.core
    monkeypatch.setattr(TDOAK.core, '__path__', list(TDOAK.core.__path__) + [str(tmp_path)])
    monkeypatch.setattr(loader, '__path__', list(loader.__path__) + [str(tmp_path)])
    corePath = tmp_path / '_t_core.py'
    corePath.write_text('VALUE = 1\n')
    (tmp_path / '_t_ext.py').write_text('from TDOAK.core._t_core import VALUE\n')
    importlib.invalidate_caches()
    core = importlib.import_module('TDOAK.core._t_core')
    ext = importlib.import_module('TDOAK.ext._t_ext')
    try:
        assert loader.reloadChanged() == []          # first sight: baselines recorded
        assert loader.reloadChanged() == []          # nothing changed
        corePath.write_text('VALUE = 2\n')
        st = os.stat(corePath)
        os.utime(corePath, (st.st_atime, st.st_mtime + 5))
        importlib.invalidate_caches()
        reloaded = loader.reloadChanged()
        assert 'TDOAK.core._t_core' in reloaded and 'TDOAK.ext._t_ext' in reloaded
        assert reloaded.index('TDOAK.core._t_core') < reloaded.index('TDOAK.ext._t_ext')
        assert core.VALUE == 2 and ext.VALUE == 2    # ext re-imported the new core value
        assert loader.reloadChanged() == []
    finally:
        sys.modules.pop('TDOAK.core._t_core', None)
        sys.modules.pop('TDOAK.ext._t_ext', None)


def test_create_instantiates_class(tmp_path, monkeypatch):
    mod = types.ModuleType('TDOAK.ext.FakeExt')

    class FakeExt:
        def __init__(self, ownerComp):
            self.ownerComp = ownerComp

    mod.FakeExt = FakeExt
    monkeypatch.setitem(sys.modules, 'TDOAK.ext.FakeExt', mod)
    inst = loader.create('FakeExt', 'comp')
    assert isinstance(inst, FakeExt) and inst.ownerComp == 'comp'
