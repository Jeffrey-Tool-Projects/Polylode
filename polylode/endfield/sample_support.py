"""Pure source checks shared by Blender sample consumers and tests."""
from pathlib import Path

from .body_archive import file_digest


def source_fingerprint(source: Path) -> dict:
    """Identify both authored FBX and its material/bone declaration sidecar."""
    return {'fbx_sha256': file_digest(source), 'metadata_sha256': file_digest(source.with_suffix('.json'))}


def artifact_fingerprint(blend: Path, previews: list[str]) -> dict:
    """Identify the saved Blender file and every generated inspection image."""
    root = blend.parent.resolve()
    paths = {name: (root / name).resolve() for name in previews}
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise ValueError('preview file is outside its output directory')
    return {'blend_sha256': file_digest(blend), 'previews': {
        name: file_digest(path) for name, path in paths.items()}}


def artifacts_current(row: dict, source: Path, blend: Path) -> tuple[bool, str]:
    """Historical or failed files cannot certify the latest attempted conversion."""
    if row.get('status') not in ('built', 'needs_review'):
        return False, 'latest build has no successful validation'
    if not row.get('source_fingerprint') or not row.get('artifact_fingerprint'):
        return False, 'source/output fingerprints are missing'
    try:
        if row['source_fingerprint'] != source_fingerprint(source):
            return False, 'source FBX or metadata changed'
        if row['artifact_fingerprint'] != artifact_fingerprint(blend, row.get('previews', [])):
            return False, 'output Blender file or previews changed'
    except (OSError, ValueError, TypeError) as exc:
        return False, str(exc)
    return True, ''


def preview_render_settings(engine: str, body_available: bool) -> dict:
    """Choose an explicit inspection renderer, preserving calibrated materials."""
    if engine in ('cycles-cpu', 'cycles-gpu'):
        return {'engine': 'CYCLES', 'device': 'CPU' if engine == 'cycles-cpu' else 'GPU'}
    if engine == 'auto':
        return {'engine': 'BLENDER_EEVEE' if body_available else 'BLENDER_WORKBENCH'}
    raise ValueError(f'unknown preview engine: {engine}')


def select_cycles_gpu(preferences) -> dict:
    """Enable a discovered GPU backend; never silently substitute CPU rendering."""
    supported = {entry[0] for entry in preferences.get_device_types(None)}
    for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
        if backend not in supported:
            continue
        devices = preferences.get_devices_for_type(backend)
        names = [device.name for device in devices if device.type == backend]
        if not names:
            continue
        preferences.compute_device_type = backend
        for device in preferences.devices:
            device.use = device.type == backend
        return {'backend': backend, 'devices': names}
    raise RuntimeError('no available Cycles GPU; choose cycles-cpu explicitly')


def bone_aliases(rows: list[dict], available: set[str]) -> dict[str, str]:
    """Validate the authored canonical-node to exported-name table exactly."""
    aliases, owners = {}, {}
    for row in rows:
        canonical, exported = row.get('node'), row.get('name')
        if not isinstance(canonical, str) or not isinstance(exported, str) or not canonical or not exported:
            raise ValueError('bone metadata requires canonical node and exported name')
        if exported not in available:
            raise ValueError(f'exported metadata bone is absent from FBX: {exported}')
        if canonical in aliases and aliases[canonical] != exported:
            raise ValueError(f'ambiguous canonical bone alias: {canonical}')
        if exported in owners and owners[exported] != canonical:
            raise ValueError(f'exported bone has multiple canonical nodes: {exported}')
        aliases[canonical], owners[exported] = exported, canonical
    return aliases


def binding_names(bones: list[str], aliases: dict[str, str], available: set[str],
                  used: set[int]) -> list[str]:
    """Translate only declared aliases, retaining the original bone-index table."""
    names = [aliases.get(name, name) for name in bones]
    for index in used:
        if index < 0 or index >= len(names):
            raise ValueError(f'body binding index outside source bone table: {index}')
        if names[index] not in available:
            raise ValueError(f'body binding missing FBX bone: {bones[index]} -> {names[index]}')
    if len({names[i] for i in used}) != len(used):
        raise ValueError('two used source bones resolve to the same exported bone')
    return names


def preview_omission(material: dict, directory: Path) -> str:
    """Preview only the supported authored albedo surfaces; never invent gray."""
    shader = material.get('shader')
    if shader not in {'HGRP/CharacterNPR', 'HGRP/CharacterNPR_Skin',
                      'HGRP/CharacterNPR_Hair', 'HGRP/CharacterNPR_Eye'}:
        return f'unsupported preview shader: {shader or "undeclared material"}'
    base = material.get('base')
    descriptor = material.get('maps', {}).get(base) if base else None
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get('file'), str):
        return f'missing authored base map for preview shader: {shader}'
    if not (directory / descriptor['file']).is_file():
        return f'preview base map file is missing: {descriptor["file"]}'
    return ''


def preview_alpha_mode(material: dict) -> tuple[str, float]:
    """Use the authored shader render state, never FBX texture-alpha guesses."""
    keywords = set(material.get('keywords', []))
    floats = material.get('floats', {})
    if '_ALPHATEST_ON' in keywords:
        return 'clip', float(floats.get('_AlphaClipThreshold', 0.0))
    blend = (int(floats.get('_SrcBlend', 1)), int(floats.get('_DstBlend', 0)))
    if ('_ALPHABLEND_ON' in keywords or blend != (1, 0)
            or floats.get('_SurfaceType') == 1):
        return 'blend', 0.0
    return 'opaque', 0.0


def body_render_accepted(job: dict) -> bool:
    """Rejected fits remain editable sources; old prepared jobs retain compatibility."""
    return bool(job.get('body_arrays') and job.get('body_render_accepted',
                                                 job.get('body_status') == 'prepared'))


def given_coverage(job: dict, faces: int) -> list | None:
    """The whole-outfit mask the build measured for this job, or None for a job without one.

    A job the build prepared carries ``covered_by_rule`` (the body faces its
    outfit covers, by the rule ``coverage_rule`` names) and the worker cuts
    the dressed body by it instead of auditing rays itself.  A list that does
    not fit the body it is for is an error, not a list to trim.
    """
    given = job.get("covered_by_rule")
    if given is None:
        return None
    if not job.get("coverage_rule"):
        raise ValueError("the job's coverage list names no rule")
    if len(set(given)) != len(given) or any(type(f) is not int or not 0 <= f < faces for f in given):
        raise ValueError("the job's coverage list does not fit the added body's faces")
    return sorted(given)


def outfit_visibility_expression(domain: str, expression: str) -> str:
    """Restore garment-covered body faces when the outfit is hidden in this domain."""
    if domain not in ('hide_viewport', 'hide_render'):
        raise ValueError(f'unknown outfit visibility domain: {domain}')
    if expression not in ('outfit', '1-outfit', '1'):
        raise ValueError(f'unknown outfit visibility expression: {expression}')
    if expression == '1':
        return '1'
    controls = ('viewport_hidden', 'eye_hidden', 'layer_excluded') if domain == 'hide_viewport' else (
        'render_hidden', 'layer_excluded')
    effective = '(outfit' + ''.join(f'*(1-{control})' for control in controls) + ')'
    return effective if expression == 'outfit' else '1-' + effective
