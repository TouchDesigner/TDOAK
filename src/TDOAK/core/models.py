"""Static scan of a pipeline script (pure Python, no exec).

A pipeline DAT declares things the host needs BEFORE it runs build() — the models to
pre-fetch (MODELS / MODEL) and optional watchdog overrides (FIRST_FRAME_TIMEOUT_S /
STALL_TIMEOUT_S). They are read by parsing the script with ast: nothing is executed, so
the scan has no side effects and can't run module-level code a second time.
"""
from __future__ import annotations

import ast


def resolveLiteral(node: ast.expr, env: dict[str, object]) -> object:
    """Literal value of an ast expression, resolving Name references from previously
    assigned module-level values. Raises ValueError for anything non-literal."""
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(node.id)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [resolveLiteral(e, env) for e in node.elts]
    return ast.literal_eval(node)


def scanConstants(script: str | None) -> dict[str, object]:
    """Module-level literal assignments of a script as {name: value}. Name references
    between them resolve (MODELS = [MODEL_A, MODEL_B]); computed values are skipped.
    Raises SyntaxError when the script doesn't parse."""
    env: dict[str, object] = {}
    if not script:
        return env
    for node in ast.parse(script).body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value = node.value
        else:
            continue
        if not targets or value is None:
            continue
        try:
            resolved = resolveLiteral(value, env)
        except (ValueError, TypeError, SyntaxError):
            continue  # computed value — not statically resolvable
        for name in targets:
            env[name] = resolved
    return env


def declaredModels(constants: dict[str, object]) -> list[str]:
    """The models a script declares via MODELS = [...] (a bare MODEL = '...' also counts)."""
    decl = constants.get('MODELS', constants.get('MODEL'))
    if isinstance(decl, str):
        decl = [decl]
    if not isinstance(decl, (list, tuple)):
        return []
    return [str(m) for m in decl if m]


def timeoutOverride(constants: dict[str, object], name: str, default: float) -> float:
    """A watchdog timeout from the script (e.g. STALL_TIMEOUT_S = 0 for a host-frame
    pipeline that legitimately sends nothing until fed), else the component default."""
    value = constants.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(default)
    return max(float(value), 0.0)
