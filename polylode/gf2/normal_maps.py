"""Verified GF2 base-normal component for the procedural lighting preview (§7.772).

This binds neither the game's lighting nor its detail, wet, LOD or billboard
branches. Unsupported source settings fail before changing the target material.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..mesh.mesh import Material, Mesh
from ..mesh.tangent import validate

if TYPE_CHECKING:
    from ..gl.textures import TextureCache

SHADER_SHA256 = {
    'gf_shader/pbr/character/uber':
        '7ea5f9b2bb45419e0d47e436e8ab1b3652600b4a7412003747b20adb99ea6d5d',
    'gf_shader/pbr/character/ubertrans':
        'd5b0eaa72ce9aff60ae9a23c6e84a476e7f7954ebca7aaedb95fff5c4a474227',
}
NORMAL_NEUTRAL_GLOBALS = frozenset({'_ALPHATEST_ON', '_USE_STOCKING'})
SHADER_GLOBALS = frozenset({
    '_ALPHATEST_ON', '_USE_STOCKING', '_MAIN_LIGHT_SHADOWS', '_MAIN_LIGHT_SHADOWS_CASCADE',
    '_ADDITIONAL_LIGHT_SHADOWS', '_MRT', '_MOTION_VECTOR', '_CHARACTER_EFFECT', '_WET_EFFECT',
    '_GF_CHAR_LOD1', '_GF_CHAR_LOD2', '_USE_BLEND_TEX', '_EXPONENTIAL_HEIGHT_FOG',
})


def _texture_binding(entry: dict) -> tuple[tuple, tuple[float, ...]]:
    original = entry.get('source', {})
    if original.get('preview_quantized') and not {'format', 'payload', 'mip_count', 'width', 'height'} <= original.keys():
        raise ValueError('GF2 float normal-map source requires an exact float upload')
    if type(entry['srgb']) is not bool:
        raise ValueError('invalid GF2 source texture color space')
    transform = np.asarray([*entry['scale'], *entry['offset']], dtype=float)
    if transform.shape != (4,) or not np.isfinite(transform).all():
        raise ValueError('invalid GF2 source normal-map transform')
    if len(entry['wrap']) != 2 or any(value not in (0, 1, 2) for value in entry['wrap']):
        raise ValueError('unsupported GF2 source texture wrap')
    key = ('gf2map', entry['cab'], entry['name'], entry['srgb'], *entry['wrap'])
    if original:
        from . import store
        from .texture_sources import texture_key
        key = texture_key(store.root(), entry)
    return key, tuple(map(float, transform))


def bind_base_normal(target: Material, source: dict, mesh: Mesh, shader_sha256: str,
                     global_keywords: tuple[str, ...] = ()) -> tuple:
    """Bind a proven source normal map/frame; return its exact texture-cache key.

    The caller supplies the SHA of the original serialized shader, not a shader
    name inferred from mesh labels. The target remains a lighting preview.
    """
    family = source['_shader']
    if family not in SHADER_SHA256 or shader_sha256 != SHADER_SHA256[family]:
        raise ValueError('unverified GF2 source shader')
    active_globals = set(global_keywords) | (set(source['keywords']) & SHADER_GLOBALS)
    if active_globals - NORMAL_NEUTRAL_GLOBALS:
        raise ValueError('GF2 global-effect normal branches are not transcribed')
    floats = source['floats']
    controls = [float(floats.get(name, 0)) for name in ('_UseBillboard', '_DetailNormalIntensity')]
    if not np.isfinite(controls).all():
        raise ValueError('nonfinite GF2 source normal control')
    if float(floats.get('_UseBillboard', 0)) > 0.5:
        raise ValueError('GF2 billboard frame is not transcribed')
    keywords = set(source['keywords'])
    if keywords & {'_USE_VOLUMETRIC', '_ANISOTROPIC_SPECULAR'}:
        raise ValueError('GF2 alternate normal-lighting branch is not transcribed')
    if '_DETAIL_MAP' in keywords:
        if family.endswith('/ubertrans') or float(floats.get('_DetailNormalIntensity', 0)) != 0:
            raise ValueError('GF2 active detail-normal branch is not transcribed')
    if target.toon_family != 0:
        raise ValueError('GF2 normal component requires procedural preview lighting')
    validate(mesh.tangents, mesh.vertex_count)
    if len(mesh.tangents) != mesh.vertex_count:
        raise ValueError('GF2 normal component requires authored tangents')
    if mesh.normals.shape != (mesh.vertex_count, 3) or not np.isfinite(mesh.normals).all():
        raise ValueError('GF2 normal component requires complete finite authored normals')
    key, transform = _texture_binding(source['maps']['_BumpMap'])
    target.normal_map = key
    target.normal_transform = transform
    target.normal_encoding = 1
    return key


def resolve_texture(cache: TextureCache, key: tuple) -> int:
    """Load a GF2 source PNG with its own color space and both wrap axes; fail loudly."""
    from PIL import Image
    from OpenGL import GL
    from . import store

    if len(key) != 6 or key[0] != 'gf2map' or type(key[3]) is not bool:
        raise ValueError('invalid GF2 texture-cache key')
    wraps = {0: GL.GL_REPEAT, 1: GL.GL_CLAMP_TO_EDGE, 2: GL.GL_MIRRORED_REPEAT}
    if key[4] not in wraps or key[5] not in wraps:
        raise ValueError('unsupported GF2 source texture wrap')
    resident = cache.get(key)
    if resident is not None:
        return resident
    with Image.open(store.texture_path(key[1], key[2])) as image:
        pixels = np.array(image.convert('RGBA'), dtype=np.uint8)
    return cache.upload(key, pixels, srgb=key[3], wrap=(wraps[key[4]], wraps[key[5]]))
