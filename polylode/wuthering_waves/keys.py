"""Discover and verify keys using only files in the local game installation."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from . import extract

_LIMIT = 256
_HEX = re.compile(rb'(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])')
_WIDE = re.compile(rb'(?<![0-9A-Fa-f]\x00)(?:[0-9A-Fa-f]\x00){64}(?![0-9A-Fa-f]\x00)')
# MOV dword ptr [base+disp8], imm32 or [base+disp32], imm32.
_MOV = re.compile(rb'\xc7[\x40-\x47]([\x00-\xff])(.{4})|\xc7[\x80-\x87](.{4})(.{4})', re.DOTALL)


class DiscoveryError(RuntimeError):
    pass


def _candidates(data: bytes):
    seen = set()
    for pattern in (_HEX, _WIDE):
        for match in pattern.finditer(data):
            text = match.group().replace(b'\x00', b'')
            key = bytes.fromhex(text.decode('ascii'))
            if key not in seen:
                seen.add(key)
                yield key
    group = []
    previous_end = -1
    for match in _MOV.finditer(data):
        displacement = int.from_bytes(match.group(1) or match.group(3), 'little', signed=True)
        immediate = match.group(2) or match.group(4)
        opcode = match.group()[1] & 7
        if not group or match.start() - previous_end > 8 or opcode != group[-1][0] or displacement != group[-1][1] + 4:
            group = []
        group.append((opcode, displacement, immediate))
        previous_end = match.end()
        if len(group) == 8:
            key = b''.join(item[2] for item in group)
            if key not in seen:
                seen.add(key)
                yield key
            group = []


def _run(reader, dotnet, install, keys, output, mode, argument=None):
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DOTNET_ROOT=str(dotnet), DOTNET_CLI_TELEMETRY_OPTOUT='1')
    from ..core.paths import derived_root
    cwd = derived_root() / 'assets_cache' / 'wuthering_waves' / 'reader_runtime'
    cwd.mkdir(parents=True, exist_ok=True)
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
        for key in _candidates(path.read_bytes()):
            if key not in candidates and any(key):
                candidates.append(key)
                if len(candidates) >= _LIMIT:
                    break
        if len(candidates) >= _LIMIT:
            break
    log(f'找到 {len(candidates)} 個本機候選金鑰，正在驗證')
    zero = '0x' + '00' * 32
    with tempfile.TemporaryDirectory(prefix='polylode-wuwa-keys-') as temp:
        root = Path(temp)
        baseline, pending = _index(reader, dotnet, install, root / 'baseline', {'mainKey': zero, 'dynamicKeys': []})
        if not pending:
            raise DiscoveryError('沒有可供驗證的加密封存檔')
        remaining = set(pending)
        main = zero
        dynamic = {}
        for number, candidate in enumerate(candidates):
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
            raise DiscoveryError('本機程式中沒有找到可驗證所有加密封存檔的金鑰，此版本可能尚不支援')
        result = {'mainKey': main, 'dynamicKeys': [{'guid': guid, 'key': key} for guid, key in dynamic.items()]}
        final = root / 'final'
        mounted, _ = _index(reader, dotnet, install, final, result)
        if not set(pending).issubset(mounted - baseline):
            raise DiscoveryError('完整金鑰組合未能通過掛載驗證')
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
    return output
