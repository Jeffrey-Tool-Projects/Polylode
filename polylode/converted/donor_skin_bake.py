"""Build clean donor-skin inputs through the existing body create/draw pipeline.

Only the explicitly selected local donor card supplies its skin settings.
No actor geometry is refitted, no original costume is changed, and this
conversion does not certify the target character's material appearance.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..chara.cardimport import read_origin_card
from ..mesh.mesh import Material
from ..origin import bake
from ..origin import colorspace as CS
from ..origin.figure import body_head as B
from ..origin.figure import create_tex as CT
from ..origin.figure import toon
from ..origin.figure.attach import apply_shader, shader_normal
from ..origin.figure.report import DressReport
from . import store


@dataclass(frozen=True)
class SkinSource:
    """Complete material parameters, explicit source-space maps and reproducible donor provenance."""
    material: dict
    maps: dict[str, CS.Map]
    provenance: dict


def array_digest(array: np.ndarray) -> str:
    """Hash exact contiguous array bytes; callers separately retain shape and dtype."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _map_record(value: CS.Map) -> dict:
    if not isinstance(value, CS.Map) or type(value.srgb) is not bool:
        raise store.WarehouseError('donor source map has no explicit color-space flag')
    array = value.px
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'srgb': value.srgb,
            'pixels_sha256': array_digest(array)}


def _props_record(props: dict) -> dict:
    return {key: {'map': _map_record(value)} if isinstance(value, CS.Map) else value
            for key, value in props.items()}


def _inputs(card_path: Path):
    card = read_origin_card(card_path)
    if int(card.parameter.get('sex', -1)) != 1:
        raise store.WarehouseError('the recorded female donor requires an explicit female source card')
    source = bake.read_item('base', 'body')
    if source is None:
        raise store.WarehouseError('recorded donor base/body archive is unavailable')
    pieces = [piece for piece in source.pieces if piece.mesh.name == B.BODY_MESH]
    if len(pieces) != 1:
        raise store.WarehouseError('recorded donor body mesh is missing or ambiguous')
    props, create = B.create_props(B.BODY_MATERIAL[0]), B.create_props('cf_m_body_create')
    if props.get('_shader') != 'Shader Forge/main_skin' or not create:
        raise store.WarehouseError('recorded donor create/draw material is unavailable or unsupported')
    for name in ('skin', 'skin_mask'):
        if not isinstance(source.maps.get(name), CS.Map):
            raise store.WarehouseError(f'donor source is missing its own {name} map')
    for name in ('_NormalMap', '_DetailMask'):
        if not isinstance(props.get(name), CS.Map):
            raise store.WarehouseError(f'donor material is missing its own {name} map')
    return card, source, pieces[0], props, create


def build(card_path: Path) -> SkinSource:
    """Bake this selected card's skin with all card paint/sunburn/overlay/crest layers disabled.

    Uses the exact archived base/body mesh's UV0..3 and COLOR for the existing
    decal-to-UV0 create/draw pipeline. Full source dimensions are retained.
    Skin normal decoding uses shader_normal's actual alpha/green rule, never
    the older appearance-based tangent_normal heuristic.
    """
    card_path = Path(card_path)
    source_path = bake.bake_dir() / 'base/body.npz'
    source_sha, card_sha = store.digest(source_path), store.digest(card_path)
    producers = {'body_head': B, 'create_tex': CT, 'colorspace': CS, 'toon': toon}
    producer_shas = {name: store.digest(Path(module.__file__)) for name, module in producers.items()}
    producer_shas['donor_skin_bake'] = store.digest(Path(__file__))
    card, source, piece, props, create = _inputs(card_path)
    report, selected = DressReport(), {}
    for category, field in (('nip', 'nipId'), ('underhair', 'underhairId'), ('body_detail', 'detailId')):
        ident = card.body().get(field)
        image = B._tex(category, ident, None, report=report)
        selected[category] = {'id': ident, 'map': _map_record(image) if image is not None else None}
    if report.missing or report.errors:
        raise store.WarehouseError(f'donor skin has unresolved source textures: {report.missing}')
    albedo, shade = B.body_texture(card, source.maps['skin'], source.maps['skin_mask'],
                                    mesh=piece.mesh, sex=0, marks=False, report=report)
    normal = shader_normal(props['_NormalMap'].px)
    detail = CT.detail_map(props['_DetailMask'].px, props)
    ramp = toon.ramp_map()
    if report.missing or report.errors or any(value is None for value in (albedo, shade, normal, detail, ramp)):
        raise store.WarehouseError(f'donor skin create/draw inputs are incomplete: {report.missing}')
    material = apply_shader(Material(name='Rebuilt body skin'), {**props, **(piece.shader or {})})
    if material.toon_family != toon.FAMILY_SKIN:
        raise store.WarehouseError('donor body does not use its supported source skin shader')
    material.base_color = (1., 1., 1., 1.)
    material.normal_transform = tuple(props['_NormalMap_ST'])
    power, nail = B.gloss_powers(card, head=False)
    if power is not None:
        material.origin_specular = power
    if nail is not None:
        material.spec_nail = nail
    maps = {'texture': CS.Map(albedo, True), 'shade_mask': CS.Map(shade, False),
            'normal_map': CS.Map(normal, props['_NormalMap'].srgb),
            'detail_mask': CS.Map(detail, props['_DetailMask'].srgb), 'ramp': ramp}
    values = asdict(material)
    for name in maps:
        values[name] = {'texture': name}
    provenance = {'source_body_archive_sha256': source_sha, 'card_sha256': card_sha, 'raw_sex': 1,
                  'source_mesh': B.BODY_MESH, 'vertices': piece.mesh.vertex_count,
                  'uv0_float32_sha256': array_digest(piece.mesh.uvs.astype('<f4')),
                  'body_shader': _props_record({**props, **(piece.shader or {})}),
                  'create_shader': _props_record(create),
                  'source_maps': {name: _map_record(source.maps[name]) for name in ('skin', 'skin_mask')},
                  'selected_layers': selected, 'card_marks': False, 'card_skin_overlays': False,
                  'card_material_overrides': False, 'source_report_missing': [],
                  'producer_sources': producer_shas}
    if (store.digest(source_path) != source_sha or store.digest(card_path) != card_sha
            or store.digest(Path(__file__)) != producer_shas['donor_skin_bake']
            or any(store.digest(Path(module.__file__)) != producer_shas[name] for name, module in producers.items())):
        raise store.WarehouseError('donor card, source body or producer changed during material creation')
    return SkinSource(values, maps, provenance)
