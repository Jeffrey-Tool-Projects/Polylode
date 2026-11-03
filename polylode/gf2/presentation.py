"""GF2 display rules measured from local serialized shaders (section 7.662)."""
from __future__ import annotations

import re

import numpy as np

UBER = 'gf_shader/pbr/character/uber'
TRANS = 'gf_shader/pbr/character/ubertrans'
# These properties have explicit serialized m_DefTexture.m_DefaultName=white
# in both local shader assets; this is Unity's authored unbound-texture value.
DECLARED_WHITE = frozenset(('_BaseMap', '_DetailAlbedo', '_DetailMask'))


def validate_profile(ident: str, parts: list[dict], profile: dict) -> dict[str, bool]:
    """Require a complete visibility decision for this exact character variant.

    Selection belongs to the individual source/review profile. This function
    checks its identity and coverage; it never chooses parts for another model.
    """
    if profile.get('ident') != ident:
        raise ValueError('GF2 display profile belongs to a different character variant')
    if profile.get('body_change_allowed') is not False:
        raise ValueError('appearance profiles cannot authorize body deformation')
    source = profile['source']
    if source.get('unresolved'):
        raise ValueError('GF2 individual source visibility is unresolved')
    names = [p['name'] for p in parts]
    hidden = source['hidden']
    if len(set(names)) != len(names) or set(hidden) != set(names):
        raise ValueError('GF2 profile must cover exactly the source mesh instances')
    if any(type(value) is not bool for value in hidden.values()):
        raise ValueError('GF2 visibility must be an explicit boolean for each mesh')
    if all(hidden.values()):
        raise ValueError('GF2 profile has no visible source geometry')
    return dict(hidden)


def display_lods(parts: list[dict]) -> dict:
    """Select declared standard LOD0 for inspection, retaining alternate meshes.

    The explicit project display policy prefers the standard renderer over its
    `_lodm0` counterpart. Suffixes and duplicated source instances stay separate.
    This is not a transcription of the original game's distance switching.
    """
    groups = {}
    hidden = {part['name']: bool(part['hidden']) for part in parts}
    for part in parts:
        match = re.fullmatch(r'(.*)_lod(m?)(\d+)(.*)', part['name'], re.IGNORECASE)
        if match and not part['hidden']:
            family = match[1] + match[4]
            rank = (int(match[3]), bool(match[2]))
            groups.setdefault(family, []).append((rank, part['name']))
    selected = {}
    for family, choices in groups.items():
        best = min(rank for rank, name in choices)
        selected[family] = [name for rank, name in choices if rank == best]
        for rank, name in choices:
            hidden[name] = rank != best
    return {'hidden': hidden, 'selected': selected, 'policy': 'standard_highest_visible_lod'}


def combat_variants(parts: list[dict]) -> dict[str, bool]:
    """Each part hidden when it is a mode variant (``…_lod0_Dorm``) of a renderer that has a combat one
    (``…_lod0_Fight``): Nikketa's SSR0102 carries a combat and a dormitory pair of shoes on one prefab.

    The ``slg`` prefabs are the combat models, and the game's combat code picks ``(\\w*_lod[0-1](?:_Fight)?)``
    (the literal beside its ``root_merger``), and its swappable-parts code names ``Fight`` and no ``Dorm``.  Those
    literals are what is read — the code itself is encrypted.  A renderer with no suffix is not a variant.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for part in parts:
        match = re.fullmatch(r'(.*_lodm?\d+)_([A-Za-z]+)', part['name'])
        if match:
            groups.setdefault(match[1], []).append((match[2], part['name']))
    hidden = {part['name']: False for part in parts}
    for members in groups.values():
        if any(mode == 'Fight' for mode, _name in members):
            for mode, name in members:
                hidden[name] = mode != 'Fight'
    return hidden

def surface(record: dict) -> dict:
    """Require source parameters for the verified base/detail coverage rules."""
    shader = record['shader']
    if shader not in (UBER, TRANS):
        raise ValueError(f'GF2 surface coverage has not been verified: {shader}')
    keywords, floats = set(record['keywords']), record['floats']
    profile = {'base': '_BaseMap', 'base_color': record['colors']['_BaseColor'],
               'cutoff': floats['_Cutoff'] if '_ALPHATEST_ON' in keywords else None,
               'transparent': shader == TRANS, 'detail': None}
    if shader == TRANS:
        pair = (floats['_BlendMode'], floats['_PreMulAlpha'])
        if pair not in ((5, 0), (1, 1)):
            raise ValueError(f'unverified GF2 blend/premultiply combination: {pair}')
    if '_DETAIL_MAP' in keywords:
        profile['detail'] = {'alpha_mode': floats['_DetailAlphaMode'],
                             'alpha_intensity': floats['_DetailAlphaIntensity'],
                             'color_intensity': floats['_DetailAlbedoIntensity']}
    return profile


def coverage(record: dict, base: np.ndarray, mask: np.ndarray | None = None,
             detail: np.ndarray | None = None) -> np.ndarray:
    """Evaluate source opacity and strict less-than discard, without geometry edits."""
    profile = surface(record)
    alpha = np.asarray(base)[..., 3] * profile['base_color'][3]
    settings = profile['detail']
    if settings and profile['transparent']:
        if mask is None or detail is None:
            raise ValueError('detail alpha requires its actual mask and detail samples')
        first = np.clip(alpha * mask[..., 0], 0, 1)
        second = np.clip(mask[..., 1] * detail[..., 3] * settings['alpha_intensity'], 0, 1)
        alpha = first * second if settings['alpha_mode'] > 0 else np.maximum(first, second)
    visible = np.ones_like(alpha) if profile['cutoff'] is None else (alpha >= profile['cutoff'])
    return alpha * visible if profile['transparent'] else np.asarray(visible, dtype=float)
