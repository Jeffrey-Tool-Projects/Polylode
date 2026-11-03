from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

from . import extract

PUBLIC_URL = 'https://yarik0chka.github.io/wuwa-keys/keys.json'
MAX_DOWNLOAD = 256 * 1024
_LIMIT = 256
_HEX = re.compile(rb'(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])')
_WIDE = re.compile(rb'(?<![0-9A-Fa-f]\x00)(?:[0-9A-Fa-f]\x00){64}(?![0-9A-Fa-f]\x00)')
_MOV = re.compile(rb'\xc7[\x00-\x04\x06\x07\x40-\x47\x80-\x87]')


class DiscoveryError(RuntimeError):
    pass


def _candidates(data: bytes):
    seen = set()
    group = {}
    previous_end, previous_base = -1, None
    for match in _MOV.finditer(data):
        if match.start() < previous_end:
            continue
        pos = match.end()
        modrm = data[match.start() + 1]
        mode, base = modrm >> 6, modrm & 7
        rex = data[match.start() - 1] if match.start() and match.start() - 1 >= previous_end else 0
        if 0x48 <= rex <= 0x4f:
            continue
        if base == 4:
            if pos >= len(data) or data[pos] & 0x38 != 0x20:
                continue
            base = data[pos] & 7
            pos += 1
            if mode == 0 and base == 5:
                continue
        if 0x40 <= rex <= 0x47:
            base += (rex & 1) * 8
        size = {0: 0, 1: 1, 2: 4}[mode]
        if pos + size + 4 > len(data):
            continue
        displacement = int.from_bytes(data[pos:pos + size], 'little', signed=True)
        pos += size
        if match.start() - previous_end > 8 or base != previous_base:
            group = {}
        group[displacement] = data[pos:pos + 4]
        previous_end, previous_base = pos + 4, base
        if len(group) == 8:
            offsets = sorted(group)
            if offsets == list(range(offsets[0], offsets[0] + 32, 4)):
                key = b''.join(group[offset] for offset in offsets)
                if key not in seen:
                    seen.add(key)
                    yield key
            group = {}
    for pattern in (_HEX, _WIDE):
        for match in pattern.finditer(data):
            text = match.group().replace(b'\x00', b'')
            key = bytes.fromhex(text.decode('ascii'))
            if key not in seen:
                seen.add(key)
                yield key


def _run(reader, dotnet, install, keys, output, mode, argument=None):
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DOTNET_ROOT=str(dotnet), DOTNET_CLI_TELEMETRY_OPTOUT='1')
    from ..core.paths import derived_root
    cwd = derived_root() / 'assets_cache' / 'wuthering_waves' / 'reader_runtime'
    cwd.mkdir(parents=True, exist_ok=True)
    if mode != 'index' and (not (cwd / 'zlib-ng2.dll').is_file() or not any(
            (cwd / name).is_file() for name in ('oodle-data-shared.dll', 'oo2core_9_win64.dll'))):
        raise DiscoveryError('離線驗證缺少本機 Oodle 或 zlib-ng 解壓縮元件，未下載任何檔案')
    try:
        done = subprocess.run(extract.command(reader, dotnet, install, keys, output, mode, argument),
                              cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=120, check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.TimeoutExpired) as failure:
        raise DiscoveryError('無法啟動或等待遊戲讀取器') from failure
    if done.returncode:
        raise DiscoveryError('遊戲讀取器無法驗證金鑰')


def _index(reader, dotnet, install, folder, data):
    folder.mkdir(parents=True, exist_ok=True)
    keys = folder / 'keys.json'
    keys.write_text(json.dumps(data), encoding='utf-8')
    _run(reader, dotnet, install, keys, folder, 'index')
    try:
        report = json.loads((folder / 'mount.json').read_text(encoding='utf-8'))
        return {x['Path'].casefold() for x in report['mounted']}, {
            x['Path'].casefold(): x['guid'] for x in report['unmounted']}
    except (OSError, KeyError, ValueError, TypeError) as failure:
        raise DiscoveryError('遊戲讀取器的掛載報告無效') from failure


def discover_keys(install: Path, reader: Path, dotnet: Path, output: Path, log=print) -> Path:
    binaries = install / 'Client' / 'Binaries' / 'Win64'
    files = sorted(p for p in binaries.glob('Client*') if p.is_file() and p.suffix.lower() in ('.exe', '.dll'))
    if not files:
        raise DiscoveryError('找不到本機遊戲程式，無法取得金鑰')
    candidates = []
    for path in files:
        log(f'分析本機遊戲程式：{path.name}')
        for key in _candidates(path.read_bytes()):
            if key not in candidates and any(key):
                candidates.append(key)
                if len(candidates) >= _LIMIT:
                    break
        if len(candidates) >= _LIMIT:
            break
    log(f'找到 {len(candidates)} 個本機候選金鑰，正在驗證')
    if not candidates:
        raise DiscoveryError('本機程式中沒有找到可驗證所有加密封存檔的金鑰，此版本可能尚不支援；可使用「下載公開 AES 金鑰…」')
    zero = '0x' + '00' * 32
    with tempfile.TemporaryDirectory(prefix='polylode-wuwa-keys-') as temp:
        root = Path(temp)
        baseline, pending = _index(reader, dotnet, install, root / 'baseline', {'mainKey': zero, 'dynamicKeys': []})
        if not pending:
            raise DiscoveryError('沒有可供驗證的加密封存檔')
        remaining = set(pending)
        main = zero
        dynamic = {}
        deadline = time.monotonic() + 300
        for number, candidate in enumerate(candidates):
            if time.monotonic() > deadline:
                raise DiscoveryError('本機金鑰驗證超時，未變更金鑰檔')
            log(f'驗證本機候選金鑰 {number + 1} / {len(candidates)}')
            key = '0x' + candidate.hex().upper()
            trial = {'mainKey': key, 'dynamicKeys': [{'guid': guid, 'key': key} for guid in set(pending.values())]}
            try:
                mounted, _ = _index(reader, dotnet, install, root / f'trial-{number}', trial)
            except DiscoveryError:
                continue
            for path in (mounted - baseline) & remaining:
                guid = pending[path]
                if re.sub('[^0-9a-f]', '', guid.lower()).strip('0'):
                    dynamic[guid] = key
                else:
                    main = key
                remaining.remove(path)
            if not remaining:
                break
        if remaining:
            raise DiscoveryError('本機程式中沒有找到可驗證所有加密封存檔的金鑰，此版本可能尚不支援；可使用「下載公開 AES 金鑰…」')
        result = {'mainKey': main, 'dynamicKeys': [{'guid': guid, 'key': key} for guid, key in dynamic.items()]}
        _verify_and_publish(reader, dotnet, install, output, root, baseline, pending, result)
    return output


def _verify_and_publish(reader, dotnet, install, output, root, baseline, pending, result):
    final = root / 'final'
    mounted, _ = _index(reader, dotnet, install, final, result)
    if not set(pending).issubset(mounted - baseline):
        raise DiscoveryError('金鑰未能解鎖目前遊戲的所有加密封存檔，可能已過期；請稍後再試或選擇已有的金鑰檔')
    from . import assembly, scope
    _run(reader, dotnet, install, final / 'keys.json', final / 'model', 'json', assembly.MODEL_TABLE)
    if not (final / 'model' / 'package.json').is_file() or not (final / 'model' / 'package.json').stat().st_size:
        raise DiscoveryError('模型設定讀取驗證失敗')
    wanted = final / 'wanted.txt'
    wanted.write_text(scope.FILES[0] + '\n', encoding='utf-8')
    _run(reader, dotnet, install, final / 'keys.json', final / 'role', 'raw-list', str(wanted))
    if not (final / 'role' / scope.FILES[0]).is_file() or not (final / 'role' / scope.FILES[0]).stat().st_size:
        raise DiscoveryError('角色設定讀取驗證失敗')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.wuwa-keys-', dir=output.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
        os.replace(name, output)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _parse_public(data):
    try:
        obj = json.loads(data)
        def hex_value(value, length):
            if not isinstance(value, str):
                raise ValueError()
            value = value.removeprefix('0x').removeprefix('0X')
            if not re.fullmatch(r'[0-9a-fA-F]{%d}' % length, value):
                raise ValueError()
            return value.upper()
        main = '0x' + hex_value(obj['mainKey'], 64)
        if not isinstance(obj['dynamicKeys'], list):
            raise ValueError()
        dynamic = {}
        for entry in obj['dynamicKeys']:
            guid = entry['guid']
            if isinstance(guid, str) and re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', guid):
                guid = guid.replace('-', '')
            guid = hex_value(guid, 32)
            key = '0x' + hex_value(entry['key'], 64)
            if guid in dynamic and dynamic[guid] != key:
                raise ValueError()
            dynamic[guid] = key
        return {'mainKey': main, 'dynamicKeys': [{'guid': guid, 'key': key} for guid, key in dynamic.items()]}
    except (ValueError, TypeError, KeyError, UnicodeDecodeError):
        raise DiscoveryError('公開金鑰資料格式無效，請稍後再試或選擇已有的金鑰檔') from None


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        from urllib.parse import urlsplit
        source, target = urlsplit(PUBLIC_URL), urlsplit(newurl)
        if (target.scheme, target.netloc) != (source.scheme, source.netloc):
            raise DiscoveryError('公開金鑰來源重新導向至其他網站，已停止下載')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_keys(install: Path, reader: Path, dotnet: Path, output: Path, log=print) -> Path:
    log('正在下載公開金鑰並驗證本機遊戲版本')
    try:
        opener = urllib.request.build_opener(_SameOriginRedirect())
        deadline = time.monotonic() + 30
        with opener.open(PUBLIC_URL, timeout=30) as response:
            data = bytearray()
            while len(data) <= MAX_DOWNLOAD:
                if time.monotonic() > deadline:
                    raise TimeoutError()
                chunk = response.read1(min(65536, MAX_DOWNLOAD + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
        if len(data) > MAX_DOWNLOAD:
            raise DiscoveryError('公開金鑰資料超過大小限制，請稍後再試')
    except (OSError, urllib.error.URLError, TimeoutError) as failure:
        raise DiscoveryError('無法下載公開金鑰，請檢查網路後再試') from failure
    result = _parse_public(data)
    with tempfile.TemporaryDirectory(prefix='polylode-wuwa-keys-') as temp:
        root = Path(temp)
        baseline, pending = _index(reader, dotnet, install, root / 'baseline',
                                   {'mainKey': '0x' + '00' * 32, 'dynamicKeys': []})
        if not pending:
            raise DiscoveryError('沒有可供驗證的加密封存檔')
        _verify_and_publish(reader, dotnet, install, output, root, baseline, pending, result)
    return output
