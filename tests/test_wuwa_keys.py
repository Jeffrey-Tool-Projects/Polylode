import json

import pytest

from polylode.wuthering_waves import keys


def test_candidates_extract_hex_wide_and_contiguous_mov():
    first = bytes(range(32))
    second = bytes(range(32, 64))
    mov = b''.join(b'\xc7\x40' + bytes([i * 4]) + second[i * 4:i * 4 + 4] for i in range(8))
    assert set(keys._candidates(first.hex().encode() + b'!' + mov)) == {first, second}
    assert first in list(keys._candidates(first.hex().encode('utf-16le')))
    assert not list(keys._candidates(b'\x01' * 200))


@pytest.mark.parametrize('reverse', [False, True])
def test_candidates_decode_stack_stores_and_zero_displacement(reverse):
    key = bytes(range(1, 33))
    offsets = list(range(0, 32, 4))
    if reverse:
        offsets.reverse()
    stack = b''.join(b'\xc7\x44\x24' + bytes([i]) + key[i:i + 4] for i in offsets)
    direct = b'\xc7\x00' + key[:4] + b''.join(
        b'\xc7\x40' + bytes([i]) + key[i:i + 4] for i in range(4, 32, 4))
    assert key in list(keys._candidates(stack))
    assert key in list(keys._candidates(direct))


def test_stores_to_different_registers_are_not_joined():
    key = bytes(range(1, 33))
    code = b''.join((b'\x41' if i >= 16 else b'') + b'\xc7\x40' + bytes([i]) + key[i:i + 4]
                    for i in range(0, 32, 4))
    assert key not in list(keys._candidates(code))


def test_immediate_byte_is_not_mistaken_for_rex_prefix():
    key = bytes(range(64, 96))
    code = b''.join(b'\xc7\x40' + bytes([i]) + key[i:i + 4] for i in range(0, 32, 4))
    assert key in list(keys._candidates(code))


def test_opcode_bytes_in_key_do_not_interrupt_stores():
    key = bytes([1, 2, 3, 4, 5, 6, 0xc7, 0x40, *range(9, 33)])
    code = b''.join(b'\xc7\x40' + bytes([i]) + key[i:i + 4] for i in range(0, 32, 4))
    assert key in list(keys._candidates(code))


def test_stopping_unstarted_worker_never_inspects_system_process(monkeypatch):
    from types import SimpleNamespace
    import psutil
    from polylode.extractor.app import SourcePage
    calls = []
    process = SimpleNamespace(processId=lambda: 0, kill=lambda: calls.append('kill'),
                              waitForFinished=lambda ms: calls.append('wait'))
    page = SimpleNamespace(process=process, keys_settings_file='snapshot.json')
    monkeypatch.setattr(psutil, 'Process', lambda pid: pytest.fail('PID zero must not be inspected'))
    SourcePage.stop_key_discovery(page)
    assert calls == ['kill', 'wait']


def test_missing_offline_libraries_does_not_start_reader(tmp_path, monkeypatch):
    from polylode.core import paths
    monkeypatch.setattr(paths, 'derived_root', lambda: tmp_path)
    monkeypatch.setattr(keys.subprocess, 'run', lambda *a, **kw: pytest.fail('reader would download libraries'))
    with pytest.raises(keys.DiscoveryError, match='未下載'):
        keys._run(tmp_path / 'reader', tmp_path, tmp_path, tmp_path / 'keys.json', tmp_path / 'out', 'json', 'model')


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
        def readAllStandardOutput(self):
            return b''
        def deleteLater(self):
            pass
    for code, expected in ((1, ''), (0, str(valid))):
        page.process = Process()
        page.lines = A.Lines()
        page.keys_written = valid
        page.keys_settings_file = tmp_path / 'temp.json'
        page.keys_settings_file.write_text('{}')
        page._keys_done(code, None)
        assert S.load().tools.get('wuwa_keys', '') == expected
        assert page.keys_settings_file is None
        assert not (tmp_path / 'temp.json').exists()
        assert page.keys_button.isEnabled()
    window.close()
    del qt


@pytest.mark.parametrize('success', [True, False])
def test_discovery_button_runs_background_process_and_preserves_settings(tmp_path, monkeypatch, success):
    import sys
    import time
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from polylode.extractor import app as A, settings as S
    monkeypatch.setattr(S, 'SETTINGS', tmp_path / 'settings.json')
    S.save(S.Settings(tools={'wuwa_keys': 'previous.json'}))
    qt = QApplication.instance() or QApplication([])
    window = A.Window()
    page = next(window.pages.widget(i) for i in range(window.pages.count())
                if getattr(getattr(window.pages.widget(i), 'source', None), 'id', '') == 'wuwa')
    valid = tmp_path / 'wuwa_aes_key.json'
    valid.write_text(json.dumps({'mainKey': '0x' + '11' * 32, 'dynamicKeys': []}))
    worker = tmp_path / 'worker.py'
    worker.write_text('import sys\nprint("OUTPUT " + sys.argv[1], flush=True)\nsys.exit(int(sys.argv[2]))\n')
    commands = []
    def command(snapshot):
        commands.append(snapshot)
        assert S.load(snapshot).tools['wuwa_keys'] == 'previous.json'
        return [sys.executable, str(worker), str(valid), '0' if success else '1']
    monkeypatch.setattr(A.jobs, 'keys_command', command)
    monkeypatch.setattr(page, 'ask_install', lambda: True)
    monkeypatch.setattr(page, 'ask_tools', lambda skip_keys: skip_keys)
    page.keys_button.click()
    page.discover_keys()
    assert len(commands) == 1
    assert not page.scan_button.isEnabled() and not page.export_button.isEnabled()
    deadline = time.monotonic() + 10
    while page.process is not None and time.monotonic() < deadline:
        qt.processEvents()
        time.sleep(.01)
    assert page.process is None
    assert S.load().tools['wuwa_keys'] == (str(valid) if success else 'previous.json')
    assert page.keys_button.isEnabled()
    assert not commands[0].exists()
    window.close()
