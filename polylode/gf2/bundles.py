"""Read GF2's local bundles without changing the installed files.

The format encrypts the first 0x8000 bytes of each UnityFS member with a
repeating 16-byte XOR mask. Members can be concatenated; their own big-endian
size field, rather than the installation file size, delimits each member.
"""
from __future__ import annotations

import gc
import json
import re
import struct
from pathlib import Path

import numpy as np
import UnityPy
from UnityPy.exceptions import UnityVersionFallbackError

from ..core import progress

HEADER = b'UnityFS\0\0\0\0\x075.x.'


def members(path: Path):
    """Yield (offset, decoded member); reject unknown or truncated inputs."""
    with path.open('rb') as stream:
        length = path.stat().st_size
        while stream.tell() < length:
            offset = stream.tell()
            header = stream.read(64)
            if len(header) < 38:
                raise ValueError(f'{path.name}@{offset}: truncated bundle header')
            if header.startswith(b'GFF\0'):
                raise ValueError(f'{path.name}@{offset}: GFF is not a Unity bundle')
            mask = np.zeros(16, dtype=np.uint8) if header.startswith(b'UnityFS\0') else (
                np.frombuffer(header[:16], dtype=np.uint8) ^ np.frombuffer(HEADER, dtype=np.uint8))
            clear = bytes(v ^ int(mask[i % 16]) for i, v in enumerate(header))
            if not clear.startswith(b'UnityFS\0'):
                raise ValueError('unsupported bundle signature')
            # UnityFS strings terminate before the big-endian bundle size.
            cursor = 12
            for _ in range(2):
                cursor = clear.index(b'\0', cursor) + 1
            size = struct.unpack_from('>Q', clear, cursor)[0]
            if size < cursor + 20 or offset + size > length:
                raise ValueError(f'{path.name}@{offset}: invalid member size {size}')
            stream.seek(offset)
            data = bytearray(stream.read(size))
            encrypted = np.frombuffer(data, dtype=np.uint8, count=min(size, 0x8000))
            encrypted ^= np.resize(mask, encrypted.shape)
            yield offset, bytes(data)


def bundle_paths(install: Path):
    data = install / 'GF2_Exilium_Data'
    roots = [data / 'StreamingAssets' / 'aa' / 'StandaloneWindows64',
             data / 'LocalCache' / 'Data' / 'AssetBundles_Windows']
    return sorted({p for root in roots if root.is_dir() for p in root.rglob('*.bundle')})


def configure_unity(install: Path) -> dict[str, str]:
    """Use this installation's authored bundle version for stripped headers."""
    from UnityPy.helpers.UnityVersion import UnityVersion
    from UnityPy.streams import EndianBinaryReader

    for path in bundle_paths(install):
        with path.open('rb') as stream:
            raw = stream.read(128)
        if len(raw) < 38 or raw.startswith(b'GFF\0'):
            continue
        mask = bytes(16) if raw.startswith(b'UnityFS\0') else bytes(
            a ^ b for a, b in zip(raw[:16], HEADER))
        clear = bytes(value ^ mask[i % 16] for i, value in enumerate(raw))
        reader = EndianBinaryReader(clear)
        if reader.read_string_to_null() != 'UnityFS':
            continue
        reader.read_u_int()
        reader.read_string_to_null()
        version = reader.read_string_to_null()
        if not re.fullmatch(r'\d{4}\.\d+\.\d+[abfp]\d+\w*', version):
            continue
        UnityVersion.from_str(version)
        UnityPy.config.FALLBACK_UNITY_VERSION = version
        return {'version': version, 'source': path.relative_to(install).as_posix()}
    raise ValueError(f'{install}: no authored Unity engine version in local bundle headers')


def directories(path: Path):
    """Read only UnityFS directory metadata, without decompressing asset blocks."""
    from UnityPy.helpers.CompressionHelper import DECOMPRESSION_MAP
    from UnityPy.streams import EndianBinaryReader

    with path.open('rb') as stream:
        length = path.stat().st_size
        offset = 0
        while offset < length:
            stream.seek(offset)
            encrypted = stream.read(128)
            if len(encrypted) < 38 or encrypted.startswith(b'GFF\0'):
                raise ValueError(f'{path.name}@{offset}: unsupported or truncated header')
            mask = bytes(16) if encrypted.startswith(b'UnityFS\0') else bytes(
                a ^ b for a, b in zip(encrypted[:16], HEADER))
            clear = bytes(v ^ mask[i % 16] for i, v in enumerate(encrypted))
            reader = EndianBinaryReader(clear)
            if reader.read_string_to_null() != 'UnityFS':
                raise ValueError('unsupported bundle signature')
            version = reader.read_u_int()
            reader.read_string_to_null()
            engine = reader.read_string_to_null()
            size = reader.read_long()
            compressed = reader.read_u_int()
            uncompressed = reader.read_u_int()
            flags = reader.read_u_int()
            if size <= reader.Position or offset + size > length:
                raise ValueError('invalid bundle member size')
            if version >= 7 or engine.startswith('2019.4.') and int(engine.split('.')[2].split('f')[0]) >= 15:
                reader.align_stream(16)
            start = size - compressed if flags & 128 else reader.Position
            if start < reader.Position or start + compressed > size:
                raise ValueError('invalid directory range')
            stream.seek(offset + start)
            raw = stream.read(compressed)
            raw = bytes(v ^ mask[(start + i) % 16] if start + i < 0x8000 else v
                        for i, v in enumerate(raw))
            data = DECOMPRESSION_MAP[flags & 63](raw, uncompressed)
            if len(data) != uncompressed:
                raise ValueError('directory decompression size mismatch')
            info = EndianBinaryReader(data)
            info.read_bytes(16)
            blocks = info.read_int()
            if blocks < 0 or blocks * 10 > len(data):
                raise ValueError('invalid block count')
            info.Position += blocks * 10
            count = info.read_int()
            names = []
            for _ in range(count):
                info.read_long()
                info.read_long()
                info.read_u_int()
                names.append(info.read_string_to_null())
            yield offset, names
            offset += size


def scan_directories(install: Path, output: Path):
    """Build dependency locations only; this does not enumerate other models."""
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = bundle_paths(install)
    with output.open('w', encoding='utf8') as log:
        for number, path in enumerate(paths, 1):
            progress.report(number, len(paths), '讀 bundle 目錄')
            row = {'path': path.relative_to(install).as_posix(), 'members': []}
            try:
                row['members'] = [{'offset': offset, 'cabs': [{'name': n} for n in names]}
                                  for offset, names in directories(path)]
            except (ValueError, KeyError, OSError, EOFError, NotImplementedError) as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            log.write(json.dumps(row) + '\n')
    return output


def scan(install: Path, output: Path, limit: int = 0):
    """A resumable CAB/dependency/container index, one bundle member at a time."""
    output.parent.mkdir(parents=True, exist_ok=True)
    version = configure_unity(install)
    print(f'UNITY {version["version"]} from {version["source"]}', flush=True)
    previous = {}
    if output.exists():
        for line in output.read_text(encoding='utf8').splitlines():
            row = json.loads(line)
            previous[row['path']] = row
    paths = bundle_paths(install)
    if limit:
        paths = paths[:limit]
    with output.open('a', encoding='utf8') as log:
        for number, path in enumerate(paths, 1):
            progress.report(number, len(paths), '建立 bundle 索引')
            relative = path.relative_to(install).as_posix()
            stamp = [path.stat().st_size, path.stat().st_mtime_ns]
            if previous.get(relative, {}).get('stamp') == stamp and not previous[relative].get('error'):
                continue
            row = {'path': relative, 'stamp': stamp, 'members': []}
            try:
                for offset, data in members(path):
                    env = UnityPy.Environment(path=str(output.parent))
                    env.load_file(data)
                    member = {'offset': offset, 'cabs': [], 'containers': []}
                    for cab, file in env.cabs.items():
                        if not hasattr(file, 'objects'):
                            continue
                        counts = {}
                        for obj in file.objects.values():
                            counts[obj.type.name] = counts.get(obj.type.name, 0) + 1
                        entry = {'name': cab, 'externals': [e.path for e in file.externals],
                                 'types': counts}
                        member['cabs'].append(entry)
                    containers = [entry for file in env.assets for entry in file.container.items()]
                    for container, ptr in containers:
                        if not ptr or ptr.m_FileID:
                            continue
                        obj = ptr.deref()
                        entry = {'container': container, 'cab': obj.assets_file.name,
                                 'id': obj.path_id, 'type': obj.type.name}
                        if obj.type.name in ('GameObject', 'Mesh', 'Animator', 'Avatar'):
                            entry['name'] = obj.read().m_Name if obj.type.name != 'Animator' else ''
                        member['containers'].append(entry)
                    row['members'].append(member)
                    del env, data
                del member
            except (ValueError, TypeError, KeyError, IndexError, OSError, EOFError,
                    NotImplementedError, RuntimeError, AttributeError, UnityVersionFallbackError) as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            if number % 100 == 0 or row.get('error'):
                print(f'{number}/{len(paths)} {path.name} {row.get("error", "indexed")}', flush=True)
            gc.collect()
    return output
