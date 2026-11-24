from TDOAK.core.device import describeDevice, enumName


def test_enum_name():
    assert enumName('UsbSpeed.SUPER') == 'SUPER'
    assert enumName('XLinkProtocol.X_LINK_USB_VSC') == 'USB_VSC'
    assert enumName('X_LINK_TCP_IP') == 'TCP_IP'


def test_usb3_has_no_warning():
    text, warning = describeDevice('OAK-D-LITE', 'RVC2', 'USB_VSC', 'SUPER')
    assert text == 'OAK-D-LITE | RVC2 | USB SUPER'
    assert warning is None


def test_usb2_warns():
    text, warning = describeDevice('OAK-D-LITE', 'RVC2', 'USB_VSC', 'HIGH')
    assert 'USB HIGH' in text
    assert 'USB3 cable' in warning


def test_poe():
    text, warning = describeDevice('OAK-D-SR-POE', 'RVC2', 'TCP_IP', None)
    assert text == 'OAK-D-SR-POE | RVC2 | PoE'
    assert warning is None


def test_unknown_usb_speed_does_not_warn():
    assert describeDevice('X', 'RVC2', 'USB_VSC', None)[1] is None
