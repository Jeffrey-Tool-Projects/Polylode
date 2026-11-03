import json

import pytest

from polylode.wuthering_waves import keys


def test_candidates_extract_hex_wide_and_contiguous_mov():
    first = bytes(range(32))
    second = bytes(range(32, 64))
    mov = b''.join(b'\xc7\x40' + bytes([i * 4]) + second[i * 4:i * 4 + 4] for i in range(8))
    assert list(keys._candidates(first.hex().encode() + b'!' + mov)) == [first, second]
    assert first in list(keys._candidates(first.hex().encode('utf-16le')))
    assert not list(keys._candidates(b'\x01' * 200))


def test_plaintext_mount_is_not_evidence_and_preserves_output(tmp_path, monkeypatch):
    binary = tmp_path / 'Client/Binaries/Win64/Client.exe'
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b'11' * 32)
    output = tmp_path / 'out.json'
    output.write_text('existing')
    monkeypatch.setattr(keys, '_index', lambda *args: ({'plain'}, {'encrypted': '1234'}))
    with pytest.raises(keys.DiscoveryError, match='沒有找到'):
        keys.discover_keys(tmp_path, tmp_path / 'reader', tmp_path, output, lambda _: None)
    assert output.read_text() == 'existing'


def test_dynamic_keys_must_cover_all_archives(tmp_path, monkeypatch):
    binary = tmp_path / 'Client/Binaries/Win64/Client.exe'
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b'11' * 32 + b'!' + b'22' * 32)
    output = tmp_path / 'out.json'
    def index(reader, dotnet, install, folder, data):
        if folder.name == 'baseline':
            return {'plain'}, {'one': '1111', 'two': '2222'}
        if folder.name == 'trial-0':
            return {'plain', 'one'}, {'two': '2222'}
        if folder.name == 'trial-1':
            return {'plain', 'two'}, {'one': '1111'}
        assert {x['guid'] for x in data['dynamicKeys']} == {'1111', '2222'}
        folder.mkdir(parents=True)
        (folder / 'keys.json').write_text(json.dumps(data))
        return {'plain', 'one', 'two'}, {}
    def run(reader, dotnet, install, keyfile, folder, mode, argument=None):
        from polylode.wuthering_waves import scope
        target = folder / ('package.json' if mode == 'json' else scope.FILES[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('content')
    monkeypatch.setattr(keys, '_index', index)
    monkeypatch.setattr(keys, '_run', run)
    keys.discover_keys(tmp_path, tmp_path / 'reader', tmp_path, output, lambda _: None)
    assert len(json.loads(output.read_text())['dynamicKeys']) == 2


def test_key_discovery_result_persists_only_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from polylode.extractor import app as A, settings as S
    monkeypatch.setattr(S, 'SETTINGS', tmp_path / 'settings.json')
    S.save(S.Settings())
    qt = QApplication.instance() or QApplication([])
    window = A.Window()
    page = next(window.pages.widget(i) for i in range(window.pages.count())
                if getattr(getattr(window.pages.widget(i), 'source', None), 'id', '') == 'wuwa')
    valid = tmp_path / 'keys.json'
    valid.write_text(json.dumps({'mainKey': '0x' + '11' * 32, 'dynamicKeys': []}))
    class Process:
        def errorString(self):
            return 'failed'
    for code, expected in ((1, ''), (0, str(valid))):
        page.process = Process()
        page.lines = A.Lines()
        page.keys_written = valid
        page.keys_settings_file = tmp_path / 'temp.json'
        page.keys_settings_file.write_text('{}')
        page._keys_done(code, None)
        assert S.load().tools.get('wuwa_keys', '') == expected
        assert not page.keys_settings_file.exists()
        assert page.keys_button.isEnabled()
    window.close()
    del qt
