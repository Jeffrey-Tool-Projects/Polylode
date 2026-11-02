"""Export GF2 bake items using the same rig scene and FBX checks as Endfield."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..core.paths import write_json_atomic
from ..endfield import fbxfile
from ..endfield import rigscene as RS
from ..origin import bake
from . import display, frames, playable, precision, publish, store


def worker(blender, mode, document, fbx):
    script = Path(__file__).with_name('blender_worker.py')
    report = fbx.with_suffix(f'.{mode}.json')
    done = subprocess.run([str(blender), '--background', '--factory-startup', '--threads', '6',
                           '--python-exit-code', '1', '--python', str(script), '--', mode,
                           str(document), str(fbx), str(report)], capture_output=True,
                          text=True, encoding='utf8', errors='replace', check=False)
    if done.returncode or not report.is_file():
        raise RuntimeError(f'Blender {mode} exit {done.returncode}: {(done.stdout + done.stderr)[-1600:]}')
    return json.loads(report.read_text(encoding='utf8'))


def check_report(scene, report):
    problems = []
    if report['shape_delta_mm'] > .01:
        problems.append('blend shape vertex delta changed')
    if any(value != 0 for keys in report['shape_state'].values() for name, value, group in keys[1:]):
        problems.append('blend shapes default to a non-neutral expression')
    expected_bones = report.get('expected_instance_bones', len(scene.bones))
    expected_armatures = report.get('expected_frame_armatures', 1)
    if report['armatures'] != expected_armatures or report['bones'] != expected_bones or report['bones_missing'] or report['parents_wrong']:
        problems.append('armature count, bone count, names or parents changed')
    for key, limit in [('bone_offset_mm', .01), ('bone_turn_deg', .01),
                       ('bind_offset_mm', .01), ('bind_turn_deg', .01),
                       ('native_bone_offset_mm', .01), ('native_bone_turn_deg', .01),
                       ('blend_bone_offset_mm', .01), ('blend_bone_turn_deg', .01)]:
        if key not in report:
            continue
        if report[key] > limit:
            problems.append(f'{key}={report[key]} exceeds {limit}')
    if report.get('blend_visibility_changed'):
        problems.append('saved Blender source visibility changed')
    if report.get('blend_rest_mm', 0) > .01:
        problems.append('saved Blender character vertex positions changed')
    if report.get('blend_posed_mm', 0) > .05:
        problems.append('saved Blender posed vertex positions changed')
    if report['bones_in_file'] != expected_bones or not report['binds']:
        problems.append('missing file bones or skin clusters')
    if len(report['parts']) != len(scene.parts):
        problems.append('mesh count changed')
    for part, row in zip(scene.parts, report['parts']):
        if not row['found']:
            problems.append(f'{part.name}: mesh missing')
            continue
        expected = len(part.triangles) - report['degenerate'].get(part.name, 0) - report['repeated'].get(part.name, 0)
        if row['vertices'] != len(part.positions) or row['triangles'] != expected:
            problems.append(f'{part.name}: vertex or triangle count changed')
        if not row['armature'] or row['weighted'] != row['weighted_expected']:
            problems.append(f'{part.name}: weights missing')
        if row['rest_mm'] > .01 or row['posed_mm'] > .05:
            problems.append(f'{part.name}: rest/posed vertex error exceeds Endfield tolerances')
        if row['weighted'] and row['moved_mm'] <= .05:
            problems.append(f'{part.name}: pose did not move weighted vertices')
        if row['shape_keys'] != len(part.shapes):
            problems.append(f'{part.name}: blend shapes missing')
    return problems


#: GF2's eye shaders, whose pass render states were read from the install (§7.1455)
EYE = 'gf_shader/pbr/character/eye'
EYE_ADD = 'gf_shader/pbr/character/eyeblend_add'
EYE_MULTIPLY = 'gf_shader/pbr/character/eyeblend_multiply'


def coverage_fields(record: dict) -> None:
    """How *record* is covered, in the writers' terms (``tint``, ``alpha_from``, ``blend``, ``cutoff``):
    ``presentation.surface``'s verified rule — the base map's alpha times ``_BaseColor``'s, cut at ``_Cutoff``
    when ``_ALPHATEST_ON``, blended for ``ubertrans`` (§7.1455).  The colour is left as it was (only the
    coverage is the rule's); a shader the rule does not cover is left opaque and says why in ``coverage``.
    """
    from . import presentation
    if record.get('shader') == EYE:
        record['coverage'] = 'opaque: Blend One Zero, ZWrite on (the pass render state)'
        return
    if record.get('shader') in (EYE_ADD, EYE_MULTIPLY):
        # read from the shaders (§7.1455): the forward pass's fragment is the base map times _MainColor, with
        # _SpecularIntensity as its strength — lerp(white, it, strength) multiplied onto what is behind
        # (Blend DstColor Zero), or it times strength (and the main light's 0.2 + 0.8 * N.L) added onto it
        # (Blend One One); ZWrite off.  The colour layer's own alpha takes no part in either blend.
        colour = record.get('colors', {}).get('_MainColor', [1.0, 1.0, 1.0, 1.0])
        record['tint'] = [float(v) for v in colour[:3]] + [1.0]
        record['strength'] = float(record.get('floats', {}).get('_SpecularIntensity', 1.0))
        record['blend'] = 'add' if record['shader'] == EYE_ADD else 'multiply'
        return
    if record.get('shader') not in (presentation.UBER, presentation.TRANS):
        record['coverage'] = f"not verified for {record.get('shader')}: written opaque"
        return
    try:
        profile = presentation.surface(record)
    except (KeyError, ValueError) as refusal:
        record['coverage'] = f'not verified: {refusal}'
        return
    record['tint'] = [1.0, 1.0, 1.0, float(profile['base_color'][3])]
    record['alpha_from'] = 'a'
    if profile['transparent']:
        # glTF has no discard under a blend: the cutoff of a transparent surface is kept in extras only
        record['blend'] = 'alpha'
    elif profile['cutoff'] is not None:
        record['cutoff'] = float(profile['cutoff'])

def scene_document(ident, declared=None):
    """The playable selection of *ident* as an interchange document in ``exports/<ident>``.

    Returns ``(scene, selection, layout, folder, document)``; the maps are copied beside it.
    """
    item = store.read_item(ident)
    extra = store.read_extra(ident)
    if declared is None:
        conversion = json.loads((store.root() / 'conversion.json').read_text(encoding='utf8'))
        declared = playable.roster(Path(conversion['install']))
    scene = RS.scene_of(item.pieces, item.nodes, ident)
    source_textures = {key: f'{RS.TEXTURE_DIR}/{file}' for file, key in scene.textures.items()}
    selection = playable.eligibility(extra, declared, scene)
    scene = playable.character_scene(scene, selection)
    # The shared scene builder understands Endfield lighting. GF2's preview
    # base binding is taken instead from Unity's source shader declaration.
    by_key = source_textures
    for record in scene.materials.values():
        match = [p.shader for p in item.pieces if RS._material(p.shader, by_key) == record]
        if match:
            record['base'] = match[0].get('_base_property')
            coverage_fields(record)
    folder = store.root() / 'exports' / ident
    folder.mkdir(parents=True, exist_ok=True)
    for file, (cab, name) in scene.textures.items():
        destination = folder / RS.TEXTURE_DIR / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(store.texture_path(cab, name), destination)
    document = RS.write(scene, folder, ident, extra, f'gf2/chara/{ident}')
    document_data = json.loads(document.read_text(encoding='utf8'))
    document_data['save_blend'] = selection['eligible']
    document_data['blender_selection'] = selection
    profile = display.stored_profile(ident)
    if profile is not None:
        document_data = display.document_profile(document_data, profile)
    layout = frames.plan(scene.bones, scene.parts)
    if any(any(v != 0 for v in frame['origin']) for frame in layout['frames']):
        document_data['local_frames'] = layout
    write_json_atomic(document, document_data, indent=1)
    return scene, selection, layout, folder, document


def export_one(blender, ident, declared=None):
    scene, selection, layout, folder, document = scene_document(ident, declared)
    document_data = json.loads(document.read_text(encoding='utf8'))
    fbx = folder / f'{ident}.fbx'
    counts = worker(blender, 'write', document, fbx)
    if 'local_frames' in document_data:
        bones = frames.expanded_bones(scene, layout)
        file_to_scene = np.eye(4)
        refined = precision.write_rest(fbx, bones, counts['source_mesh_worlds'],
                                      file_to_scene=file_to_scene,
                                      armature_worlds=counts['source_armature_worlds'])
    else:
        bones = [(b.name, b.parent, b.rest) for b in scene.bones]
        file_to_scene = fbxfile.FILE_TO_SCENE
        refined = precision.write_rest(fbx, bones, counts['source_mesh_worlds'])
    report = worker(blender, 'check', document, fbx)
    measured = precision.bone_errors(fbx, bones, file_to_scene)
    report.update(counts)
    report.update(bones_in_file=measured['bones'], bone_offset_mm=measured['offset_mm'],
                  bone_turn_deg=measured['turn_deg'], binds=measured['binds'],
                  bind_offset_mm=measured['bind_offset_mm'], bind_turn_deg=measured['bind_turn_deg'],
                  source_rest_write=refined)
    problems = check_report(scene, report)
    row = {'ident': ident, 'fbx': str(fbx),
           'export_scope': 'main_playable_humans',
           'gender_scope': selection['gender_scope'],
           'blend': str(fbx.with_suffix('.blend')) if selection['eligible'] else None,
           'blender_selection': selection,
           'source_signature': source_signature(ident),
           'vertices': sum(len(p.positions) for p in scene.parts), 'meshes': len(scene.parts),
           'bones': len(scene.bones), 'textures': len(scene.textures),
           'ok': not problems, 'problems': problems,
           'preview_no_base_texture': [n for n, r in scene.materials.items() if r['base'] is None],
           'shader_parity': False, 'measurements': report}
    write_json_atomic(folder / 'validation.json', row, indent=1)
    return publish.publish(row) if row['ok'] else row


def source_signature(ident):
    """Do not reuse an export after its archive, metadata or exporter changes."""
    from ..mesh import tangent
    from ..mesh import mesh
    from . import source_tangents
    from . import gender, female_roster
    from . import fbx_streams, fbx_tangents
    digest = hashlib.sha256()
    paths = [store.root() / 'chara' / f'{ident}.npz',
             store.root() / 'table' / store.SOURCE / 'chara' / f'{ident}.json',
             Path(__file__), Path(__file__).with_name('blender_worker.py'),
             Path(__file__).with_name('frame_worker.py'),
             Path(__file__).with_name('rig_runtime.py'),
             Path(playable.__file__), Path(precision.__file__), Path(fbxfile.__file__),
             Path(frames.__file__),
             Path(display.__file__),
             Path(RS.__file__), Path(__file__).resolve().parents[2] / 'tools/ef_fbx_blender.py']
    paths.extend([Path(tangent.__file__), Path(mesh.__file__)])
    paths.extend([Path(gender.__file__), Path(female_roster.__file__)])
    paths.extend([Path(fbx_streams.__file__), Path(fbx_tangents.__file__)])
    paths.extend([Path(source_tangents.__file__), *source_tangents.signature_paths(store.root(), ident)])
    profile = store.root() / 'display_profiles' / (ident + '.json')
    if profile.is_file():
        paths.append(profile)
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validated_items() -> set[str]:
    """Require successful conversion and actual archive read-back before export."""
    report = json.loads((store.root() / 'conversion.json').read_text(encoding='utf8'))
    return {row['ident'] for row in report['results']
            if row.get('status') == 'converted' and row.get('archive_readback') is True}


def export_all(blender, only='', limit=0, retry_failed=False):
    validated = validated_items()
    with store.opened():
        keys = [k.split('/', 1)[1] for k, value in bake.manifest()['items'].items()
                if k.startswith('chara/') and 'kind' not in value and only.lower() in k.lower()]
    path = store.root() / 'export_validation.json'
    previous = json.loads(path.read_text(encoding='utf8')) if path.exists() else {'results': []}
    rows = {r['ident']: r for r in previous['results']}
    conversion = json.loads((store.root() / 'conversion.json').read_text(encoding='utf8'))
    declared = playable.roster(Path(conversion['install']))
    eligible = set()
    selections = {}
    scope_path = store.root() / 'export_scope.json'
    for ident in sorted((set(keys) | set(rows)) & validated):
        selection = playable.stored_eligibility(ident, declared)
        selections[ident] = selection
        row = rows.get(ident, {})
        if not selection['eligible']:
            moved = publish.quarantine(ident)
            rows[ident] = dict(row, ident=ident, ok=False, skipped=True,
                               scope_excluded=True, export_scope='main_playable_humans',
                               blender_selection=selection, preserved_folder=moved or row.get('preserved_folder'),
                               reason='not a main playable human character')
        else:
            eligible.add(ident)
            previous_selection = row.get('blender_selection', {})
            selection_changed = (set(previous_selection.get('parts', [])) != set(selection['parts']) or
                                 set(previous_selection.get('bones', [])) != set(selection['bones']) or
                                 not playable.gender.selection_verified(previous_selection))
            if row.get('ok') and (row.get('export_scope') != 'main_playable_humans' or selection_changed):
                moved = publish.quarantine(ident)
                rows[ident] = dict(row, ok=False, skipped=False, needs_human_export=True,
                                   preserved_folder=moved, blender_selection=selection)
    write_json_atomic(scope_path, {'scope': 'main_playable_humans', 'gender_scope': 'female_only',
                                   'eligible': sorted(eligible),
                                   'excluded': sorted(set(selections)-eligible), 'selections': selections}, indent=1)
    keys = [k for k in keys if k in eligible]
    for ident, row in list(rows.items()):
        if ident in validated and row.get('ok') and Path(row['fbx']).parent.resolve() != (publish.root() / ident).resolve():
            rows[ident] = publish.publish(row)
    if rows:
        write_json_atomic(path, {'results': list(rows.values())}, indent=1)
        publish.manifest(list(rows.values()))
    if retry_failed:
        keys = [k for k in keys if k in rows and not rows[k].get('ok')]
    excluded = (set(keys) | set(rows)) - validated
    for ident in sorted(excluded):
        if rows.get(ident, {}).get('scope_excluded'):
            continue
        rows[ident] = dict(rows.get(ident, {}), ident=ident, ok=False, skipped=True,
                           error='conversion archive read-back has not passed; export excluded')
        if ident in keys:
            print('SKIP_UNVALIDATED', ident, flush=True)
    if excluded:
        write_json_atomic(path, {'results': list(rows.values())}, indent=1)
        publish.manifest(list(rows.values()))
    keys = [ident for ident in keys if ident in validated]
    print(f'EXPORT_QUEUE {len(keys)} validated items; {len(excluded)} excluded', flush=True)
    if not keys:
        return 0
    write_json_atomic(store.root() / 'playable_roster.json', declared, indent=1)
    failed = attempted = 0
    for ident in sorted(keys):
        cached = rows.get(ident, {})
        if (cached.get('ok') and cached.get('source_signature') == source_signature(ident)
                and Path(cached['fbx']).exists()
                and (cached.get('blend') is None or Path(cached['blend']).exists())):
            continue
        print('EXPORT', ident, flush=True)
        try:
            row = export_one(blender, ident, declared)
        except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError, AssertionError) as exc:
            row = {'ident': ident, 'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        rows[ident] = row
        failed += int(not row['ok'])
        attempted += 1
        print('OK' if row['ok'] else 'FAILED', ident, row.get('error') or row.get('problems'), flush=True)
        write_json_atomic(path, {'results': list(rows.values())}, indent=1)
        publish.manifest(list(rows.values()))
        if limit and attempted >= limit:
            break
    return int(failed > 0)
