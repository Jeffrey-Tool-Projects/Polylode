"""Collect validated interchange files separately from conversion work files."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ..core.paths import write_json_atomic
from . import store, gender, playable


def root():
    return store.root().parent / 'GF2_Converted'


def publish(row):
    if not row.get('ok') or row.get('skipped'):
        raise ValueError('only validated exports may enter the converted folder')
    if row.get('export_scope') != 'main_playable_humans' or not row.get('blender_selection', {}).get('eligible'):
        raise ValueError('only main playable human characters may enter the converted folder')
    if not gender.selection_verified(row['blender_selection']):
        raise ValueError('only confirmed female playable characters may enter the converted folder')
    ident = row['ident']
    if Path(ident).name != ident or '/' in ident or '\\' in ident:
        raise ValueError('invalid export identity')
    if not playable.families(ident, row['blender_selection']):
        raise ValueError('export identity does not belong to its confirmed female characters')
    destination = root() / ident
    source = Path(row['fbx']).parent
    # Legacy full-prefab Blender files predate the playable-only contract.
    # Keep them in the work store; publish only files with character selection
    # AND a recorded successful native read-back of that exact selection.
    selection = row.get('blender_selection', {})
    measurements = row.get('measurements', {})
    blend_allowed = (selection.get('eligible') is True and
                     selection.get('scope') == 'playable_only' and
                     set(measurements.get('blend_parts', [])) == set(selection.get('parts', [])) and
                     bool(selection.get('parts')) and row.get('blend') is not None)
    result = dict(row, fbx=str(destination / f'{ident}.fbx'),
                  blend=str(destination / f'{ident}.blend') if blend_allowed else None)
    if source.resolve() == destination.resolve():
        return result
    document_path = source / f'{ident}.json'
    document = json.loads(document_path.read_text(encoding='utf8'))
    files = {f'{ident}.fbx', f'{ident}.json', document['arrays'],
             f'{ident}.write.json', f'{ident}.check.json'}
    files.update(measurements.get('previews', []))
    for record in document['materials'].values():
        files.update(m['file'] for m in record['maps'].values())
    if blend_allowed:
        files.add(f'{ident}.blend')
    destination.mkdir(parents=True, exist_ok=True)
    for relative in sorted(files):
        old = (source / relative).resolve()
        new = (destination / relative).resolve()
        if not old.is_relative_to(source.resolve()) or not new.is_relative_to(destination.resolve()):
            raise ValueError('export file escapes its model folder')
        if not old.is_file():
            raise FileNotFoundError(old)
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, new)
    write_json_atomic(destination / 'validation.json', result, indent=1)
    return result


def manifest(rows):
    output = root()
    output.mkdir(parents=True, exist_ok=True)
    successful = [r for r in rows if r.get('ok') and not r.get('skipped')
                  and r.get('export_scope') == 'main_playable_humans'
                  and gender.selection_verified(r.get('blender_selection', {}))
                  and playable.families(r['ident'], r['blender_selection'])]
    write_json_atomic(output / 'manifest.json', {
        'scope': 'confirmed female main playable human characters only, for both FBX and Blender',
        'gender_scope': gender.POLICY,
        'models': [{'ident': r['ident'], 'fbx': str(Path(r['fbx']).relative_to(output)),
                    'blend': str(Path(r['blend']).relative_to(output)) if r.get('blend') else None,
                    'validation': f"{r['ident']}/validation.json"} for r in successful],
        'models_count': len(successful), 'blender_count': sum(bool(r.get('blend')) for r in successful)
    }, indent=1)


def quarantine(ident):
    """Preserve obsolete/non-character bundles outside the final folder."""
    if Path(ident).name != ident or '/' in ident or '\\' in ident:
        raise ValueError('invalid export identity')
    source_root = root().resolve()
    source = (source_root / ident).resolve()
    target_root = (store.root() / 'excluded_exports').resolve()
    target = (target_root / ident).resolve()
    if not source.is_relative_to(source_root) or not target.is_relative_to(target_root):
        raise ValueError('quarantine path escapes its intended folder')
    if not source.is_dir():
        return None
    target_root.mkdir(parents=True, exist_ok=True)
    count = 1
    while target.exists():
        count += 1
        target = (target_root / f'{ident}.{count}').resolve()
    if not target.is_relative_to(target_root):
        raise ValueError('quarantine target escapes its intended folder')
    os.rename(source, target)
    return str(target)
