"""Bind native/body acceptance to the exact local geometry, rig, maps and masks."""
from __future__ import annotations

import hashlib
import json
from zipfile import BadZipFile

import numpy as np

from . import store
from .actor import ActorError
from .body_archive import file_digest


def _path(root, relative: str):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f'certification input missing or outside store: {relative}')
    return path


def _item_maps(metadata: dict, index: dict) -> tuple[set, set]:
    paths, missing = set(metadata.get('extras', {}).values()), set()
    for piece in metadata.get('pieces', []):
        paths.update(piece.get('maps', {}).values())
        for descriptor in (piece.get('shader') or {}).get('maps', {}).values():
            from ..origin.bake import named_key
            key = named_key(store.SOURCE, descriptor['cab'], descriptor['name'])
            entry = index.get(key)
            if entry and entry.get('file') and not entry.get('absent'):
                paths.add(entry['file'])
            else:
                missing.add(key)
    return paths, missing


def capture(character: str, manifest: dict) -> dict:
    """Hash declared stored inputs without decoding textures or modifying assets."""
    root = store.home(character).resolve()
    index = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))['items']
    paths, missing = set(), set()
    for kind in ('chara', 'body'):
        key = f'{kind}/{character}'
        entry = index.get(key) or {}
        if not entry.get('file'):
            raise FileNotFoundError(f'certification input missing: {key}')
        paths.add(entry['file'])
        with np.load(_path(root, entry['file']), allow_pickle=False) as archive:
            metadata = json.loads(bytes(archive['meta']).decode('utf-8'))
        maps, absent = _item_maps(metadata, index)
        paths.update(maps)
        missing.update(absent)
    table = index.get(f'table/{store.SOURCE}/chara/{character}') or {}
    if not table.get('file'):
        raise FileNotFoundError('certification input missing: original character rig table')
    paths.add(table['file'])
    paths.update(manifest.get('facial_evidence_files') or [])
    composite = {key: value for key, value in manifest.items() if key != 'source_certification'}
    encoded = json.dumps(composite, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return {'version': 1, 'files': {path: file_digest(_path(root, path)) for path in sorted(paths)},
            'missing_maps': sorted(missing), 'composite_sha256': hashlib.sha256(encoded).hexdigest()}


def require_current(character: str, manifest: dict) -> None:
    """Reject missing/stale certificates before native and added meshes are loaded."""
    expected = manifest.get('source_certification')
    if not isinstance(expected, dict) or expected.get('version') != 1:
        raise ActorError(f'body/{character}: source certification is missing; rebuild this body')
    try:
        actual = capture(character, manifest)
    except (OSError, ValueError, KeyError, BadZipFile, EOFError) as exc:
        raise ActorError(f'body/{character}: source certification cannot be checked: {exc}') from exc
    if actual != expected:
        raise ActorError(f'body/{character}: source certification is stale; rebuild this body')
