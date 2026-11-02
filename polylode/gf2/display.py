"""Apply exact per-outfit display profiles while retaining native source data."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

from ..core.paths import write_json_atomic
from . import fbx_display, presentation, store


def document_profile(document: dict, profile: dict) -> dict:
    """Copy display flags into a sidecar without changing native declarations."""
    mask = presentation.validate_profile(document['name'], document['parts'], profile)
    visible = [dict(p, hidden=mask[p['name']]) for p in document['parts']]
    if presentation.display_lods(visible)['hidden'] != mask:
        raise ValueError('GF2 display profile enables overlapping levels of one LOD family')
    result = copy.deepcopy(document)
    for part in result['parts']:
        part.setdefault('source_hidden', part['hidden'])
        part['hidden'] = mask[part['name']]
    result['display_profile'] = copy.deepcopy(profile)
    return result


def digest(path: Path) -> str:
    """Hash an artifact without loading its entire content into memory."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stored_profile(ident: str) -> dict | None:
    """Load this exact outfit's persisted display decision, checking its archive."""
    path = store.root() / 'display_profiles' / (ident + '.json')
    if not path.is_file():
        return None
    profile = json.loads(path.read_text(encoding='utf8'))
    if profile.get('ident') != ident or profile.get('archive_sha256') != digest(store.root() / 'chara' / (ident + '.npz')):
        raise ValueError('GF2 stored display profile does not match the current native archive')
    return profile


def _prepare(record: dict, converted: Path) -> dict:
    from . import gender, playable
    ident = record['ident']
    if Path(ident).name != ident or '/' in ident or '\\' in ident:
        raise ValueError('invalid GF2 display identity')
    folder = (converted / ident).resolve()
    if not folder.is_relative_to(converted.resolve()):
        raise ValueError('GF2 display folder escapes converted root')
    validation = json.loads((folder / 'validation.json').read_text(encoding='utf8'))
    if (validation.get('ok') is not True or validation.get('export_scope') != 'main_playable_humans'
            or not gender.selection_verified(validation.get('blender_selection', {}))
            or not playable.families(ident, validation.get('blender_selection', {'characters': []}))):
        raise ValueError('GF2 display requires a validated female playable human export')
    document = json.loads((folder / (ident + '.json')).read_text(encoding='utf8'))
    profile = {'ident': ident, 'body_change_allowed': False, 'source': record['plan'],
               'archive_sha256': digest(store.root() / 'chara' / (ident + '.npz'))}
    wanted = document_profile(document, profile)
    if not (folder / (ident + '.fbx')).is_file() or not (folder / (ident + '.blend')).is_file():
        raise FileNotFoundError('GF2 display requires both validated formal formats')
    return {'ident': ident, 'folder': str(folder), 'document': wanted, 'profile': profile}


def _publish_one(row: dict, report: dict) -> dict:
    ident, folder = row['ident'], Path(row['folder'])
    target = folder / (ident + '.blend')
    candidate = folder / (ident + '.display.blend')
    fbx = folder / (ident + '.fbx')
    before = {'blend': digest(target), 'fbx': digest(fbx)}
    updated, proof = fbx_display.fbx_visibility(fbx.read_bytes(), row['profile']['source']['hidden'])
    temporary = folder / (ident + '.display.fbx')
    temporary.write_bytes(updated)
    if not report['unchanged_content'] or not report['visibility_verified'] or not candidate.is_file():
        raise ValueError('Blender display read-back did not pass')
    candidate.replace(target)
    temporary.replace(fbx)
    write_json_atomic(folder / (ident + '.json'), row['document'], indent=1)
    write_json_atomic(store.root() / 'display_profiles' / (ident + '.json'), row['profile'], indent=1)
    result = dict(report, fbx=proof, before_sha256=before, profile=row['profile'],
                  after_sha256={'blend': digest(target), 'fbx': digest(fbx)})
    validation = folder / 'validation.json'
    validated = json.loads(validation.read_text(encoding='utf8'))
    validated['display_validation'] = result
    write_json_atomic(validation, validated, indent=1)
    return result


def _reusable(row: dict, output: Path) -> dict | None:
    path = output / (row['ident'] + '.json')
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding='utf8'))
    folder, ident = Path(row['folder']), row['ident']
    expected = {'blend': digest(folder / (ident + '.blend')), 'fbx': digest(folder / (ident + '.fbx'))}
    if (report.get('profile') == row['profile'] and report.get('after_sha256') == expected
            and report.get('unchanged_content') is True and report.get('visibility_verified') is True):
        document = json.loads((folder / (ident + '.json')).read_text(encoding='utf8'))
        if document == row['document'] and stored_profile(ident) == row['profile']:
            return report
    return None


def apply_profiles(records: list[dict], converted: Path, blender: Path) -> list[dict]:
    """Validate all decisions first, then save and reopen each formal display."""
    if len({r['ident'] for r in records}) != len(records):
        raise ValueError('duplicate GF2 display profile identity')
    prepared = [_prepare(r, converted) for r in records]
    output = store.root() / 'display_validation'
    output.mkdir(parents=True, exist_ok=True)
    results, pending = [], []
    for row in prepared:
        cached = _reusable(row, output)
        if cached is None:
            pending.append(row)
        else:
            results.append(cached)
    worker = Path(__file__).with_name('display_worker.py')
    for index in range(0, len(pending), 8):
        batch = pending[index:index + 8]
        job, report_path = output / 'job.json', output / 'worker.json'
        report_path.unlink(missing_ok=True)
        write_json_atomic(job, batch, indent=1)
        print(f'GF2 DISPLAY {index + 1}..{index + len(batch)}/{len(pending)}; reused {len(prepared)-len(pending)}', flush=True)
        command = [str(blender), '--background', '--factory-startup', '--threads', '6',
                   '--python-exit-code', '1', '--python', str(worker), '--', str(job), str(report_path)]
        with (output / 'blender.log').open('w', encoding='utf8') as stream:
            done = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if done.returncode:
            raise RuntimeError(f'GF2 display worker exit {done.returncode}; see {output / "blender.log"}')
        reports = json.loads(report_path.read_text(encoding='utf8'))
        if [r['ident'] for r in reports] != [r['ident'] for r in batch]:
            raise ValueError('GF2 display worker result identities changed')
        for row, report in zip(batch, reports):
            results.append(_publish_one(row, report))
            write_json_atomic(output / (row['ident'] + '.json'), results[-1], indent=1)
        write_json_atomic(output / 'results.json', results, indent=1)
    return results
