"""Read source-bound supplementary streams without rewriting immutable character archives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..mesh.tangent import validate

if TYPE_CHECKING:
    from ..origin.bake import BakedItem


def _sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('source tangent supplement leaves its character store')
    return path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf8'))


def _descriptor(root: Path, ident: str) -> tuple[Path, dict] | None:
    index_path = root / 'source_tangents/index.json'
    if not index_path.is_file():
        return None
    index = _read(index_path)
    if index.get('version') != 1:
        raise ValueError('unsupported source tangent supplement schema')
    entry = index['items'].get(ident)
    if entry is None:
        return None
    path = _child(root, entry['file'])
    if _sha(path) != entry['sha256']:
        raise ValueError('source tangent descriptor changed')
    descriptor = _read(path)
    if descriptor['ident'] != ident or _sha(_child(root, descriptor['source_archive'])) != descriptor['source_archive_sha256']:
        raise ValueError('source tangent supplement belongs to a different character archive')
    return path, descriptor


def attach(root: Path, ident: str, item: BakedItem | None) -> BakedItem | None:
    """Attach only exact own-source vertex streams; reject stale evidence before mutation."""
    found = _descriptor(root, ident)
    if found is None or item is None:
        return item
    _, descriptor = found
    if len(item.pieces) != descriptor['piece_count']:
        raise ValueError('source tangent supplement piece count changed')
    pending = []
    for binding in descriptor['bindings']:
        asset = binding['tangent_asset']
        values = None
        if asset is not None:
            path = _child(root, asset['file'])
            if _sha(path) != asset['sha256']:
                raise ValueError('original authored tangent asset changed')
            with np.load(path, allow_pickle=False) as archive:
                raw = archive['tangent_unity']
                if raw.dtype != np.float32:
                    raise ValueError('original authored tangent dtype changed')
                validate(raw, binding['vertices'])
                if len(raw) != binding['vertices']:
                    raise ValueError('source-present authored tangent asset is empty')
                values = raw.copy()
                values[:, (0, 3)] *= -1
        for index in binding['piece_indices']:
            mesh = item.pieces[index].mesh
            if mesh.name != binding['source_node'] or mesh.vertex_count != binding['vertices']:
                raise ValueError('source tangent node or original vertex order changed')
            for name, expected in (('positions', binding['positions_sha256']), ('normals', binding['normals_sha256'])):
                actual = hashlib.sha256(np.ascontiguousarray(getattr(mesh, name), dtype=np.float32).tobytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f'original source tangent {name} correspondence changed')
            if values is None:
                if len(mesh.tangents):
                    raise ValueError('source-absent tangent supplement disagrees with archive')
                continue
            if len(mesh.tangents) and not np.array_equal(mesh.tangents, values):
                raise ValueError('archive tangents disagree with immutable original source')
            pending.append((mesh, values))
    for mesh, values in pending:
        mesh.tangents = values.copy()
    return item


def shape_frames(root: Path, ident: str, source_node: str) -> list[dict] | None:
    """Return original channel/frame weights and mirrored local P/N/T deltas, without interpolation."""
    found = _descriptor(root, ident)
    if found is None:
        return None
    _, descriptor = found
    binding = next((row for row in descriptor['bindings'] if row['source_node'] == source_node), None)
    if binding is None:
        raise ValueError('source node is absent from its own tangent supplement')
    asset = binding['shape_asset']
    if asset is None:
        return []
    path = _child(root, asset['file'])
    if _sha(path) != asset['sha256']:
        raise ValueError('original source expression frame asset changed')
    results = []
    with np.load(path, allow_pickle=False) as archive:
        for frame in binding['shape_frames']:
            result = dict(frame)
            result['indices'] = archive[f"f{frame['frame']}_indices"].copy()
            for name in ('position', 'normal', 'tangent'):
                vectors = archive[f"f{frame['frame']}_{name}_unity"].copy()
                vectors[:, 0] *= -1
                result[name] = vectors
            results.append(result)
    return results


def signature_paths(root: Path, ident: str) -> list[Path]:
    """Include this character's exact supplementary source files in export cache identity."""
    found = _descriptor(root, ident)
    if found is None:
        return []
    path, descriptor = found
    paths = {path}
    for binding in descriptor['bindings']:
        for key in ('tangent_asset', 'shape_asset'):
            asset = binding[key]
            if asset is not None:
                paths.add(_child(root, asset['file']))
    return sorted(paths)
