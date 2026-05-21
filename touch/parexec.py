# parExec DAT — attached to OAK Base COMP
# Routes parameter value changes and pulses to extension methods.

def onValueChange(par, prev):
    if hasattr(parent(), 'OnValueChange' + par.name):
        getattr(parent(), 'OnValueChange' + par.name)(par, prev)


def onPulse(par):
    if hasattr(parent(), 'OnPulse' + par.name):
        getattr(parent(), 'OnPulse' + par.name)(par)
