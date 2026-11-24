from __future__ import annotations

# CameraControlExt — extension of the OAKCameraControl component (TDOAKCamCtl.tox).
# Builds dai.CameraControl messages from the component's custom parameters and sends
# them into a running OAK pipeline's control input queue (OAKManagerExt.SendControl).
# Runs entirely on TD's main thread: constructing a CameraControl is pure message
# assembly and input-queue send() is non-blocking (maxSize=1) and thread-safe.
#
# Like OAKManagerExt this is a package module; the component's extension DAT is a shim
# calling TDOAK.ext.create('CameraControlExt', me).

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from td import COMP, Par

# Parameter name -> group. A change to any parameter re-sends its whole group so the
# device always sees a consistent state (e.g. manual exposure needs time AND iso).
PAR_GROUPS = {
    'Autoexposure':     'exposure',
    'Exposuretime':     'exposure',
    'Iso':              'exposure',
    'Aecompensation':   'exposure',
    'Aelock':           'exposure',
    'Antibanding':      'exposure',
    'Focusmode':        'focus',
    'Focusposition':    'focus',
    'Autowhitebalance': 'whitebalance',
    'Awblock':          'whitebalance',
    'Whitebalance':     'whitebalance',
    'Brightness':       'image',
    'Contrast':         'image',
    'Saturation':       'image',
    'Sharpness':        'image',
    'Lumadenoise':      'image',
    'Chromadenoise':    'image',
}


class CameraControlExt:

    def __init__(self, ownerComp: COMP):
        self.ownerComp = ownerComp

    # ── parexec entry points ──────────────────────────────────────────────

    def OnParValue(self, par: Par) -> None:
        """Called for every custom parameter value change."""
        if par.name in ('Oakcomp', 'Queue', 'Autosend', 'Status'):
            return
        if not self.ownerComp.par.Autosend.eval():
            return
        group = PAR_GROUPS.get(par.name)
        if group is None:
            return
        ctrl = self.makeControl()
        getattr(self, 'apply' + group.capitalize())(ctrl)
        self.push(ctrl)

    def OnPulse(self, par: Par) -> None:
        if par.name == 'Sendall':
            self.SendAll()
        elif par.name == 'Aftrigger':
            ctrl = self.makeControl()
            ctrl.setAutoFocusTrigger()
            self.push(ctrl)

    # ── public API ────────────────────────────────────────────────────────

    def SendAll(self) -> bool:
        """Push the complete parameter state to the device in one control message.
        Useful right after the pipeline (re)starts, since queue state does not persist."""
        ctrl = self.makeControl()
        self.applyExposure(ctrl)
        self.applyFocus(ctrl)
        self.applyWhitebalance(ctrl)
        self.applyImage(ctrl)
        return self.push(ctrl)

    # ── control assembly ──────────────────────────────────────────────────

    def makeControl(self):
        import depthai as dai
        return dai.CameraControl()

    def applyExposure(self, ctrl) -> None:
        import depthai as dai
        p = self.ownerComp.par
        banding = {'off':       dai.CameraControl.AntiBandingMode.OFF,
                   'mains50hz': dai.CameraControl.AntiBandingMode.MAINS_50_HZ,
                   'mains60hz': dai.CameraControl.AntiBandingMode.MAINS_60_HZ,
                   'auto':      dai.CameraControl.AntiBandingMode.AUTO}
        ctrl.setAntiBandingMode(banding[p.Antibanding.eval()])
        if p.Autoexposure.eval():
            ctrl.setAutoExposureEnable()
            ctrl.setAutoExposureCompensation(int(p.Aecompensation.eval()))
            ctrl.setAutoExposureLock(bool(p.Aelock.eval()))
        else:
            ctrl.setManualExposure(int(p.Exposuretime.eval()), int(p.Iso.eval()))

    def applyFocus(self, ctrl) -> None:
        import depthai as dai
        p = self.ownerComp.par
        mode = p.Focusmode.eval()
        if mode == 'manual':
            ctrl.setManualFocus(int(p.Focusposition.eval()))
        elif mode == 'auto':
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
        else:
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)

    def applyWhitebalance(self, ctrl) -> None:
        import depthai as dai
        p = self.ownerComp.par
        if p.Autowhitebalance.eval():
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            ctrl.setAutoWhiteBalanceLock(bool(p.Awblock.eval()))
        else:
            ctrl.setManualWhiteBalance(int(p.Whitebalance.eval()))

    def applyImage(self, ctrl) -> None:
        p = self.ownerComp.par
        ctrl.setBrightness(int(p.Brightness.eval()))
        ctrl.setContrast(int(p.Contrast.eval()))
        ctrl.setSaturation(int(p.Saturation.eval()))
        ctrl.setSharpness(int(p.Sharpness.eval()))
        ctrl.setLumaDenoise(int(p.Lumadenoise.eval()))
        ctrl.setChromaDenoise(int(p.Chromadenoise.eval()))

    # ── delivery ──────────────────────────────────────────────────────────

    def push(self, ctrl) -> bool:
        p    = self.ownerComp.par
        comp = p.Oakcomp.eval()
        if comp is None:
            self.setStatus('no OAK component assigned')
            return False
        queue = p.Queue.eval() or 'cam_ctrl'
        try:
            ok = comp.SendControl(queue, ctrl)
        except Exception as e:
            self.setStatus(f'send failed: {e}')
            return False
        if ok:
            self.setStatus(f'sent to {queue!r}')
        else:
            # SendControl returns False when the queue name is not registered — either
            # the pipeline is not running or it does not expose this control queue.
            names = []
            try:
                names = comp.GetControlNames()
            except Exception:
                pass
            hint = f' (available: {", ".join(names)})' if names else ' (pipeline not running?)'
            self.setStatus(f'queue {queue!r} unavailable{hint}')
        return ok

    def setStatus(self, message: str) -> None:
        self.ownerComp.par.Status.val = message
