# Parameter Execute DAT of the OAK component (this is the text of parexec1 inside
# TDOAK.tox - kept here for reference).
#
# Routes every custom pulse to the extension method OnPulse<Parname>, so a new pulse
# parameter only needs a matching method in OAKManagerExt - this DAT never changes.
# DAT settings: OP = the component, Parameters = *, Custom on, Built-In off, On Pulse on.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from td import Par


def onPulse(par: Par) -> None:
    fn = getattr(parent.TDOAK, 'OnPulse' + par.name, None)
    if fn is not None:
        fn(par)
