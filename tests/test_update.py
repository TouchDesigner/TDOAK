from urllib.error import HTTPError, URLError

from TDOAK.core import update


def test_version_key():
    assert update.versionKey('0.1.3') == (0, 1, 3)
    assert update.versionKey('v1.20.3') == (1, 20, 3)
    assert update.versionKey('1.2') is None
    assert update.versionKey(None) is None


def test_is_newer():
    assert update.isNewer('0.1.10', '0.1.9')
    assert not update.isNewer('0.1.3', '0.1.3')
    assert not update.isNewer('garbage', '0.1.3')


def test_parse_release():
    r = update.parseRelease({'tag_name': 'v0.2.0', 'html_url': 'u',
                             'assets': [{'name': 'a.tar.gz'}, {'name': 'TDOAK-0.2.0-py3-none-any.whl'}]})
    assert r == {'version': '0.2.0', 'url': 'u', 'wheel': 'TDOAK-0.2.0-py3-none-any.whl'}
    assert update.parseRelease({})['version'] == ''


def test_describe():
    rel = {'version': '0.2.0', 'url': '', 'wheel': ''}
    assert update.describe('0.2.0', rel, False).startswith('Up to date')
    assert 'pip install --upgrade' in update.describe('0.1.3', rel, False)
    assert 'git pull' in update.describe('0.1.3', rel, True)
    assert 'unexpected tag' in update.describe('0.1.3', {'version': 'nightly', 'url': '', 'wheel': ''}, False)


def test_reason():
    assert 'private repo' in update.reason(HTTPError('u', 404, 'nf', None, None))
    assert 'rate limit' in update.reason(HTTPError('u', 403, 'no', None, None))
    assert update.reason(URLError('timed out')) == 'timed out'
    assert update.reason(ValueError('x')) == 'ValueError: x'


def test_dev_checkout(tmp_path):
    assert update.isDevCheckout(str(tmp_path / 'src' / 'TDOAK' / '__init__.py'))
    assert not update.isDevCheckout(str(tmp_path / 'lib' / 'site-packages' / 'TDOAK' / '__init__.py'))
