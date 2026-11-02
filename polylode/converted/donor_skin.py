"""Self-contained clean donor skin, complete material fields and strict texture uploads.

The archive retains the donor's own skin pipeline, not another game's material
template. Preparing/binding these inputs never marks appearance accepted.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..core.paths import write_json_atomic
from ..mesh.mesh import Material
from ..origin.figure import toon
from . import store

if TYPE_CHECKING:
    from ..gl.textures import TextureCache
    from .donor_skin_bake import SkinSource

VERSION = 1
MAPS = frozenset(('texture', 'normal_map', 'detail_mask', 'shade_mask', 'ramp'))
TEXTURES = MAPS | frozenset(('view_ramp', 'gloss', 'pattern', 'highlight', 'glass_ramp'))


def _json(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return float(value)
    if isinstance(value, (tuple, list)):
        return [_json(part) for part in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json(part) for key, part in value.items()}
    raise store.WarehouseError('donor material has an unsupported or nonfinite parameter')


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _digest(value) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(part not in '0123456789abcdef' for part in value):
        raise store.WarehouseError('donor material requires a complete source digest')


def _material(values: dict) -> dict:
    template = Material()
    if set(values) != {field.name for field in fields(Material)}:
        raise store.WarehouseError('donor material needs every original runtime material parameter')
    values = _json(values)
    for name, value in values.items():
        default = getattr(template, name)
        if name in MAPS:
            valid = value == {'texture': name}
        elif name in TEXTURES or default is None:
            valid = value is None
        elif isinstance(default, bool):
            valid = type(value) is bool
        elif isinstance(default, int):
            valid = type(value) is int
        elif isinstance(default, float):
            valid = type(value) in (int, float)
        elif isinstance(default, str):
            valid = isinstance(value, str) and bool(value)
        else:
            valid = (isinstance(value, list) and len(value) == len(default)
                     and all(type(part) in (int, float) for part in value))
        if not valid:
            raise store.WarehouseError(f'unsupported or incomplete donor material parameter: {name}')
    if values['toon_family'] != toon.FAMILY_SKIN or values['normal_encoding'] != 0 or values['extra_uniforms']:
        raise store.WarehouseError('donor material must retain its own supported skin pipeline')
    return values


def _pixels(array: np.ndarray, shape: list[int] | None = None) -> np.ndarray:
    if (not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim != 3
            or min(array.shape[:2]) <= 0 or array.shape[2] not in (3, 4)
            or shape is not None and list(array.shape) != shape):
        raise store.WarehouseError('donor map must retain its complete uint8 RGB/RGBA source pixels')
    return array


def _provenance(value: dict) -> dict:
    value = _json(value)
    for name in ('source_body_archive_sha256', 'card_sha256', 'uv0_float32_sha256'):
        _digest(value[name])
    if (type(value['raw_sex']) is not int or value['raw_sex'] != 1
            or value['source_mesh'] != 'o_body_a' or type(value['vertices']) is not int
            or value['vertices'] <= 0 or any(value[name] is not False for name in
                                           ('card_marks', 'card_skin_overlays', 'card_material_overrides'))
            or value['source_report_missing'] != []):
        raise store.WarehouseError('donor material must retain an explicit clean female source recipe')
    return value


def _source_document(source: SkinSource) -> dict:
    material, provenance = _material(source.material), _provenance(source.provenance)
    if source.maps.keys() != MAPS:
        raise store.WarehouseError('donor skin is missing one of its own five material maps')
    textures = {}
    for name, image in source.maps.items():
        array = _pixels(image.px)
        if type(image.srgb) is not bool:
            raise store.WarehouseError('donor map has no explicit color-space flag')
        textures[name] = {'shape': list(array.shape), 'srgb': image.srgb, 'wrap': 'clamp',
                          'pixels_sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    return {'version': VERSION, 'pipeline': 'origin_main_skin', 'material': material,
            'provenance': provenance, 'textures': textures}


def source_identity(source: SkinSource) -> str:
    """Content identity of all actual source parameters, pixels, color flags and provenance."""
    return _sha(_source_document(source))


def write_source(folder: Path, source: SkinSource) -> dict:
    """Write an immutable source material/maps archive, reusing a matching complete existing one."""
    body = _source_document(source)
    identity = _sha(body)
    folder = Path(folder)
    if (folder / 'material.json').exists():
        existing = read_source(folder)
        if existing['identity'] != identity:
            raise store.WarehouseError('donor source material changed; use a new immutable archive')
        return existing
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'maps.npz', **{name: image.px for name, image in source.maps.items()})
    document = {**body, 'identity': identity, 'maps_sha256': store.digest(folder / 'maps.npz')}
    write_json_atomic(folder / 'material.json', document, indent=1)
    return read_source(folder)


def _document(folder: Path, expected: str | None = None) -> dict:
    raw = store.child(folder, 'material.json').read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise store.WarehouseError('donor material descriptor changed')
    document = json.loads(raw)
    if type(document['version']) is not int or document['version'] != VERSION or document['pipeline'] != 'origin_main_skin':
        raise store.WarehouseError('unsupported donor source material')
    _material(document['material'])
    _provenance(document['provenance'])
    body = {key: value for key, value in document.items() if key not in ('identity', 'maps_sha256')}
    if _sha(body) != document['identity'] or store.digest(store.child(folder, 'maps.npz')) != document['maps_sha256']:
        raise store.WarehouseError('donor source material or maps changed')
    if document['textures'].keys() != MAPS:
        raise store.WarehouseError('donor source descriptor has missing or unrelated maps')
    for description in document['textures'].values():
        if type(description['srgb']) is not bool or description['wrap'] != 'clamp':
            raise store.WarehouseError('donor map has no verified color-space/wrap state')
    return document


def read_source(folder: Path) -> dict:
    """Verify an entire self-contained source material, including all map pixels and flags."""
    folder = Path(folder)
    document = _document(folder)
    with np.load(folder / 'maps.npz', allow_pickle=False) as archive:
        if set(archive.files) != MAPS or document['textures'].keys() != MAPS:
            raise store.WarehouseError('donor source archive has missing or unrelated maps')
        for name, description in document['textures'].items():
            array = _pixels(archive[name], description['shape'])
            if (type(description['srgb']) is not bool or description['wrap'] != 'clamp'
                    or hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != description['pixels_sha256']):
                raise store.WarehouseError('donor source map pixels or texture state changed')
    return document


def _member_name(name: str) -> str:
    return 'donor_skin/' + name


def attach(folder: Path, source: Path) -> None:
    """Bind a verified source to this exact candidate body/UVs; preserve all geometry and acceptance gates."""
    folder, source = Path(folder), Path(source)
    metadata, document = store.read(folder), read_source(source)
    if metadata['ready_for_display'] is not False or (folder / 'card.png').exists():
        raise store.WarehouseError('donor material replacement requires a separate unaccepted candidate')
    provenance = document['provenance']
    if (metadata['donor']['source'] != 'base/body:o_body_a'
            or metadata['provenance']['source_body_archive_sha256'] != provenance['source_body_archive_sha256']):
        raise store.WarehouseError('donor material belongs to another source body')
    with np.load(store.child(folder, 'body.npz'), allow_pickle=False) as body:
        uv = np.ascontiguousarray(body['donor_uv'], dtype='<f4')
        if (uv.shape != (provenance['vertices'], 2) or metadata['donor']['vertices'] != len(uv)
                or hashlib.sha256(uv.tobytes()).hexdigest() != provenance['uv0_float32_sha256']):
            raise store.WarehouseError('donor material UVs differ from this fitted body')
    destination = folder / 'donor_skin'
    destination.mkdir(exist_ok=True)
    for name in ('material.json', 'maps.npz'):
        target, original = destination / name, source / name
        if target.exists():
            if store.digest(target) != store.digest(original):
                raise store.WarehouseError('candidate already has another immutable donor material')
        else:
            os.link(original, target)
        metadata['files'][_member_name(name)] = store.digest(target)
    binding = {'version': VERSION, 'source': metadata['source'], 'ident': metadata['ident'],
               'body_sha256': metadata['files']['body.npz'], 'material_identity': document['identity'],
               'material_sha256': metadata['files'][_member_name('material.json')]}
    write_json_atomic(folder / 'donor_material.json', binding, indent=1)
    metadata['files']['donor_material.json'] = store.digest(folder / 'donor_material.json')
    metadata['donor'].update(material_prepared=True, source_skin_texture_used=True, tattoo_texture_imported=False)
    metadata['pending'] = ['donor skin shader integration and appearance acceptance' if value ==
                           'clean donor skin material binding' else value for value in metadata['pending']]
    write_json_atomic(folder / 'warehouse.json', metadata, indent=1)
    read(folder, metadata)


@dataclass(frozen=True)
class StoredSkin:
    """A verified body-specific material source, resolvable without donor cards or game installations."""
    folder: Path
    document: dict

    def material(self) -> Material:
        """Create real source-skin runtime parameters and strict, persistent archive texture keys."""
        values = {}
        template = Material()
        for name, value in self.document['material'].items():
            if name in MAPS:
                value = ('convertedskin', str(self.folder.resolve()), store.digest(self.folder / 'material.json'), name)
            elif isinstance(getattr(template, name), tuple):
                value = tuple(value)
            values[name] = value
        return Material(**values)


def read(folder: Path, metadata: dict | None = None) -> StoredSkin:
    """Require this body's certified binding and all five maps; never substitute flat skin or another actor."""
    folder = Path(folder)
    metadata = store.read(folder) if metadata is None else metadata
    for name in ('donor_material.json', _member_name('material.json'), _member_name('maps.npz')):
        if store.digest(store.child(folder, name)) != metadata['files'].get(name):
            raise store.WarehouseError('uncertified donor material member')
    binding = json.loads((folder / 'donor_material.json').read_text(encoding='utf-8'))
    document = read_source(folder / 'donor_skin')
    if (type(binding['version']) is not int or binding['version'] != VERSION
            or binding['source'] != metadata['source'] or binding['ident'] != metadata['ident']
            or binding['body_sha256'] != metadata['files']['body.npz']
            or binding['material_sha256'] != metadata['files'][_member_name('material.json')]
            or binding['material_identity'] != document['identity']
            or document['provenance']['source_body_archive_sha256'] !=
            metadata['provenance']['source_body_archive_sha256']):
        raise store.WarehouseError('donor material binding belongs to another body or character')
    return StoredSkin(folder / 'donor_skin', document)


def resolve_texture(cache: TextureCache, key: tuple) -> int:
    """Strict source-skin upload; exceptions propagate instead of entering the generic white fallback."""
    from OpenGL import GL
    if len(key) != 4 or key[0] != 'convertedskin' or key[3] not in MAPS:
        raise store.WarehouseError('invalid explicit donor texture binding')
    folder, expected, name = Path(key[1]), key[2], key[3]
    _digest(expected)
    cache_key = (key[0], expected, name)
    existing = cache.get(cache_key)
    if existing is not None:
        return existing
    document = _document(folder, expected)
    description = document['textures'][name]
    with np.load(folder / 'maps.npz', allow_pickle=False) as archive:
        array = _pixels(archive[name], description['shape'])
    if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != description['pixels_sha256']:
        raise store.WarehouseError('donor texture pixels changed')
    if array.shape[2] == 3:
        rgba = np.empty((*array.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3], rgba[:, :, 3] = array, 255
        array = rgba
    return cache.upload(cache_key, array, mipmap=True, wrap=GL.GL_CLAMP_TO_EDGE, srgb=description['srgb'])


def prepare_all(source_archive: Path, base: Path | None = None, *,
                source: str | None = None, ident: str | None = None) -> dict:
    """Dynamically prepare eligible candidates, checkpoint errors and refresh only their source indexes."""
    from . import build, discovery
    if ident and source is None:
        raise store.WarehouseError('a donor material target identifier requires its explicit source')
    base = Path(base) if base is not None else store.root()
    document = read_source(source_archive)
    found = discovery.scan(base, source=source)
    if found.problems:
        raise store.WarehouseError(f'invalid character warehouses: {found.problems}')
    # an own-body package (``blender_source``) has no lent body to give a skin to
    rows = [row for row in found.items if row['source'] in build.DELIVERY_SOURCES
            and (not ident or row['ident'] == ident)]
    if not rows:
        raise store.WarehouseError('no matching source-qualified donor material candidates')
    results = []
    report = {'source_archive': str(Path(source_archive).resolve()), 'material_identity': document['identity'],
              'results': results, 'images_created': 0, 'runtime_accepted': False}
    for index, row in enumerate(rows, 1):
        print(f'[{index}/{len(rows)}] donor skin {row["source"]}/{row["ident"]}', flush=True)
        outcome = {'source': row['source'], 'ident': row['ident'], 'material_prepared': False}
        try:
            folder = row['path'].parent
            before = store.read(folder)
            attach(folder, source_archive)
            after = store.read(folder)
            if any(after['files'].get(name) != digest for name, digest in before['files'].items()):
                raise store.WarehouseError('preexisting actor geometry or source files changed')
            outcome.update(material_prepared=True, existing_files_retained=True,
                           warehouse_sha256=store.digest(row['path']), ready_for_display=after['ready_for_display'])
        except (ValueError, OSError, KeyError) as error:
            outcome['error'] = str(error)
            print(str(error), flush=True)
        results.append(outcome)
        write_json_atomic(base / '_conversion/donor_material_preparation.json', report, indent=1)
    for game in sorted({row['source'] for row in rows}):
        current = discovery.scan(base, source=game)
        if current.problems:
            raise store.WarehouseError(f'invalid prepared warehouses: {current.problems}')
        build.index(base, game, [row['path'].parent for row in current.items])
    return report
