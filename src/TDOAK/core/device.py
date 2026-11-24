"""Device readout helpers (pure Python)."""
from __future__ import annotations

# depthai UsbSpeed names that carry a full-rate OAK stream.
_FAST_USB = ('SUPER', 'SUPER_PLUS')


def enumName(value: object) -> str:
    """'UsbSpeed.SUPER' / 'XLinkProtocol.X_LINK_USB_VSC' -> 'SUPER' / 'USB_VSC'."""
    text = str(value).split('.')[-1]
    return text.split('X_LINK_')[-1]


def describeDevice(name: str, platform: str, protocol: str,
                   usbSpeed: str | None = None) -> tuple[str, str | None]:
    """(Deviceinfo text, warning or None) for a connected device.

    A USB device linked below SuperSpeed gets a warning: USB2 carries roughly 40 MB/s,
    less than a single 1080p stream, so frames drop or the device browns out in ways that
    look like a firmware crash from the host."""
    isUsb = 'USB' in protocol.upper()
    link = f'USB {usbSpeed}' if isUsb and usbSpeed else (
        'PoE' if 'TCP' in protocol.upper() else protocol)
    text = ' | '.join(part for part in (name, platform, link) if part)
    warning = None
    if isUsb and usbSpeed and usbSpeed.upper() not in _FAST_USB:
        warning = (f'USB link is {usbSpeed} (USB2 speed) - use a USB3 cable and port; '
                   'high-resolution or multi-stream pipelines will drop frames or stall')
    return text, warning
