# parExec DAT — attached to OAK Base COMP
# Routes parameter value changes and pulses to extension methods.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from td import Par


def onValueChange(par: Par, prev: object) -> None:
    fn = getattr(parent(), 'OnValueChange' + par.name, None)
    if fn:
        fn(par, prev)


def onPulse(par: Par) -> None:
    fn = getattr(parent(), 'OnPulse' + par.name, None)
    if fn:
        fn(par)
