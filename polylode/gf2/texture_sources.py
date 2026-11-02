"""Read SHA-bound original GF2 texture payloads and complete authored mip chains."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

FORMATS = {'DXT1': 8, 'DXT5': 16, 'BC7': 16, 'RGBAHalf': 8, 'RGBA32': 4}


@dataclass(frozen=True)
class Mip:
    """One original mip slice, in Unity's native bottom-left storage order."""
    width: int
    height: int
    offset: int
    size: int


def mip_layout(format_name: str, width: int, height: int, count: int) -> tuple[Mip, ...]:
    """Validate dimensions/count and partition native blocks without regenerating mips."""
    if format_name not in FORMATS:
        raise ValueError(f'unsupported original GF2 texture format: {format_name}')
    block_width = 1 if format_name in ('RGBAHalf', 'RGBA32') else 4
    return _mip_slices(width, height, count, FORMATS[format_name], block_width)


def _mip_slices(width: int, height: int, count: int, block_bytes: int, block_width: int) -> tuple[Mip, ...]:
    """Partition known codec blocks; this does not authorize a GPU format mapping."""
    if any(type(value) is not int or value <= 0 for value in (width, height, count)):
        raise ValueError('invalid original GF2 texture dimensions/count')
    if count > max(width, height).bit_length():
        raise ValueError('GF2 texture declares more than a complete mip chain')
    result, offset = [], 0
    for level in range(count):
        w, h = max(1, width >> level), max(1, height >> level)
        size = ((w + block_width - 1) // block_width) * ((h + block_width - 1) // block_width) * block_bytes
        result.append(Mip(w, h, offset, size))
        offset += size
    return tuple(result)


def identity_key(cab: str, name: str) -> str:
    """Stable identity of a source Texture2D instance, independent of mesh/material names."""
    return hashlib.sha256(json.dumps([cab, name], separators=(',', ':')).encode()).hexdigest()


def _child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('GF2 texture source leaves its own store')
    return path


def _sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def descriptor(root: Path, cab: str, name: str) -> tuple[dict, str]:
    """Read the exact own-source descriptor; missing or changed evidence fails."""
    index = json.loads((root / 'texture_sources/index.json').read_text('utf8'))
    if index.get('version') != 1:
        raise ValueError('unsupported GF2 texture source index')
    reference = index['assets'][identity_key(cab, name)]
    path = _child(root, reference['file'])
    if _sha(path) != reference['sha256']:
        raise ValueError('GF2 texture source descriptor changed')
    record = json.loads(path.read_text('utf8'))
    if record['identity'] != [cab, name]:
        raise ValueError('GF2 texture source descriptor belongs to another texture')
    return record, reference['sha256']


def texture_key(root: Path, entry: dict) -> tuple:
    """Require the material's own color space/wrap/payload declaration before binding."""
    record, digest = descriptor(root, entry['cab'], entry['name'])
    if type(entry['srgb']) is not bool:
        raise ValueError('invalid GF2 source color space')
    if record['srgb'] != entry['srgb'] or record['wrap'] != entry['wrap'] or record['source'] != entry['source']:
        raise ValueError('GF2 material texture declaration differs from its original source')
    return ('gf2source', entry['cab'], entry['name'], digest)


def read_payload(root: Path, key: tuple) -> tuple[dict, bytes, tuple[Mip, ...]]:
    """Validate an original complete payload before any GL allocation; never read PNG previews."""
    if len(key) != 4 or key[0] != 'gf2source':
        raise ValueError('invalid GF2 original texture key')
    record, digest = descriptor(root, key[1], key[2])
    if digest != key[3]:
        raise ValueError('GF2 bound texture descriptor changed')
    source = record['source']
    layout = mip_layout(source['format'], source['width'], source['height'], source['mip_count'])
    payload = _child(root, source['payload']).read_bytes()
    if len(payload) != record['payload_bytes'] or len(payload) != layout[-1].offset + layout[-1].size:
        raise ValueError('GF2 original texture payload/mip length mismatch')
    if hashlib.sha256(payload).hexdigest() != record['payload_sha256']:
        raise ValueError('GF2 original texture payload changed')
    return record, payload, layout


def upload_plan(record: dict, layout: tuple[Mip, ...]) -> dict:
    """Map verified Windows Texture2D formats/settings to native GL storage."""
    if (record['platform'], record['dimension'], record['image_count']) != (19, 2, 1):
        raise ValueError('unsupported GF2 texture platform/dimension')
    srgb, name = record['srgb'], record['source']['format']
    if type(srgb) is not bool or (name == 'RGBAHalf' and srgb):
        raise ValueError('unsupported GF2 original color space')
    formats = {'DXT1': (0x83F1, 0x8C4D), 'DXT5': (0x83F3, 0x8C4F),
               'BC7': (0x8E8C, 0x8E8D), 'RGBAHalf': (0x881A, 0x881A), 'RGBA32': (0x8058, 0x8C43)}
    if name not in formats:
        raise ValueError('unsupported GF2 texture GPU format')
    sampler = record['sampler']
    mode, anisotropy, bias = sampler['filter'], sampler['anisotropy'], sampler['mip_bias']
    if type(mode) is not int or mode not in (0, 1, 2):
        raise ValueError('unsupported GF2 texture filter')
    if type(anisotropy) is not int or not 0 <= anisotropy <= 16 or not math.isfinite(bias):
        raise ValueError('invalid GF2 texture sampler')
    wraps = {0: 0x2901, 1: 0x812F, 2: 0x8370}
    if len(record['wrap']) != 2 or any(type(v) is not int or v not in wraps for v in record['wrap']):
        raise ValueError('unsupported GF2 texture wrap')
    return dict(internal=formats[name][int(srgb)], compressed=name not in ('RGBAHalf', 'RGBA32'),
                min_filter=((0x2700, 0x2701, 0x2703)[mode] if len(layout) > 1 else (0x2600 if mode == 0 else 0x2601)),
                mag_filter=0x2600 if mode == 0 else 0x2601,
                wrap=tuple(wraps[v] for v in record['wrap']), anisotropy=max(1, anisotropy), bias=bias)


def resolve_texture(cache, key: tuple, *, root: Path | None = None) -> int:
    """Upload original levels from the explicit root (or current store root).

    Preserve native precision/mips without PNG conversion, flips or regeneration.
    """
    from OpenGL import GL
    from . import store
    from ..gl.pixel_transfer import cpu_unpack

    resident = cache.get(key)
    if resident is not None:
        return resident
    record, payload, layout = read_payload(store.root() if root is None else root, key)
    plan = upload_plan(record, layout)
    # Require the driver's native codec rather than silently decompressing/re-encoding.
    name = record['source']['format']
    version = (int(GL.glGetIntegerv(GL.GL_MAJOR_VERSION)), int(GL.glGetIntegerv(GL.GL_MINOR_VERSION)))
    if name in ('DXT1', 'DXT5') or (name == 'BC7' and version < (4, 2)):
        required = b'GL_EXT_texture_compression_s3tc' if name != 'BC7' else b'GL_ARB_texture_compression_bptc'
        supported = any(GL.glGetStringi(GL.GL_EXTENSIONS, n) == required
                        for n in range(int(GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS))))
        if not supported:
            raise ValueError('GPU does not support this original GF2 compressed format')
    if plan['anisotropy'] > 1:
        maximum = float(GL.glGetFloatv(0x84FF))
        if plan['anisotropy'] > maximum:
            raise ValueError('GPU cannot preserve the original GF2 anisotropy')

    def fill():
        with cpu_unpack():
            for level, mip in enumerate(layout):
                data = payload[mip.offset:mip.offset + mip.size]
                if plan['compressed']:
                    GL.glCompressedTexImage2D(GL.GL_TEXTURE_2D, level, plan['internal'], mip.width, mip.height, 0, data)
                else:
                    scalar = GL.GL_HALF_FLOAT if name == 'RGBAHalf' else GL.GL_UNSIGNED_BYTE
                    GL.glTexImage2D(GL.GL_TEXTURE_2D, level, plan['internal'], mip.width, mip.height, 0, GL.GL_RGBA, scalar, data)
        for parameter, value in ((GL.GL_TEXTURE_BASE_LEVEL, 0), (GL.GL_TEXTURE_MAX_LEVEL, len(layout) - 1),
                                 (GL.GL_TEXTURE_MIN_FILTER, plan['min_filter']), (GL.GL_TEXTURE_MAG_FILTER, plan['mag_filter']),
                                 (GL.GL_TEXTURE_WRAP_S, plan['wrap'][0]), (GL.GL_TEXTURE_WRAP_T, plan['wrap'][1])):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, parameter, value)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_LOD_BIAS, plan['bias'])
        if plan['anisotropy'] > 1:
            GL.glTexParameterf(GL.GL_TEXTURE_2D, 0x84FE, plan['anisotropy'])

    return cache.upload_native(key, len(payload), fill)
