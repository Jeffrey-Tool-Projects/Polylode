"""GF2 display rules measured from local serialized shaders (section 7.662)."""
from __future__ import annotations

import re

import numpy as np

from . import parts

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


def switched_on(part: dict, worn: list[str], cut_scene: bool) -> dict | None:
    """Why the game switches on a part its prefab ships switched off, or None: ``{"by": …, "node": …}``.

    Only a node is switched on, never a disabled renderer (a ``Renderer.enabled`` false is how these prefabs keep
    their lower levels off); *node* is the part's nearest switched-off node (``source_off``):

    * ``worn_choice`` — the node is one of the replaceable options the game dresses the outfit in (``parts``).
      GF2 3.0's ``_model`` prefabs ship every option's node off (Nikketa SSR0102: 84 nodes, her body, legs and
      clothes among them) and the replaceable-parts code switches the worn one on; in 2.x they shipped on.
    * ``cut_scene_character`` — the prefab has a director (a cut scene: ``_Class``, ``_Upgrade``, ``_Gacha``).  The
      game fills in the character the scene is of (its character track plays ``c_…_Exclusive``, ``scene_facts``),
      and the prefab ships her switched off whichever way: her instance node (``VectorSSR01(Clone)``, Loreley's
      nested ``c_LoreleySSR01_Class (1)``), each part's own node with no track that turns it on (Biyoca, Voymastina
      SSR01 Class: their character clips bind no ``m_IsActive``), a node a control track activates (Lene's
      ``LeneSSR01_Mirage``).  The parts are hers already (``playable.character_scene``); the lower levels that
      come on with them are left to the LOD rule.
    """
    off = part.get('source_off') or {}
    node = off.get('inactive_node')
    if not node or off.get('renderer_disabled'):
        return None
    if worn and parts.of(node, worn):
        return {'by': 'worn_choice', 'node': node}
    if cut_scene:
        return {'by': 'cut_scene_character', 'node': node}
    return None


#: a garment's shapes (Helen SSR0101's ``cloth2``, SSR0102's tops in every option): one of each is drawn
SHAPE = re.compile(r'_(Bend|Flat|General)(?:\.\d+)?$', re.IGNORECASE)


def shape_variants(parts: list[dict]) -> dict[str, bool]:
    """Each part hidden when it is the ``_Bend`` or ``_Flat`` shape of a garment the model also has as ``_General``.

    The three are one garment over the same span, about the same vertices, differing in how far it stands off the
    chest (Helen SSR0101 ``cloth2_lod0``: 5 521, 5 525, 5 589 vertices reaching 0.237, 0.149, 0.197 m forward for
    Bend, Flat, General); the game picks one (no literal names them in its code, the code itself is encrypted).
    SSR0101's prefab ships all three on, so all three were drawn through each other; ``General`` is shown."""
    general = any((m := SHAPE.search(part['name'])) and m[1].lower() == 'general' for part in parts)
    return {part['name']: general and bool((m := SHAPE.search(part['name'])) and m[1].lower() != 'general')
            for part in parts}


#: mode suffixes of renderers only the dormitory wears
DORMITORY_MODES = frozenset({'dorm'})
#: a dormitory model's name (``c_PhaetusaDorm_slg_skin``, ``c_VepleyDorm_skin``): the one look that wears them
DORMITORY_MODEL = re.compile(r'Dorm(?:_|$)')


def combat_variants(parts: list[dict], model: str = '') -> dict[str, bool]:
    """Each part hidden when it is a dormitory version (``…_lod0_Dorm``) and *model* is not the dormitory model.

    The ``slg`` prefabs are the combat models, and the game's combat code picks ``(\\w*_lod[0-1](?:_Fight)?)``
    (the literal beside its ``root_merger``), and its swappable-parts code names ``Fight`` and no ``Dorm``.  Those
    literals are what is read — the code itself is encrypted.  An outfit carries both versions, and they need
    not share a renderer: Nikketa's SSR0102 has a combat and a dormitory pair of shoes on one, Springfield's
    SSR0101 its combat heels on ``cloth2_lod0_Fight`` and its dormitory sandals on ``body_lod0_Dorm``; Zhaohui's,
    Vepley's and Cheeta's spell the combat suffix ``_fight``, Nagant's SR0101 ``_Flight``.  Shown together they
    were two outfits worn through each other.  So what is read is the dormitory suffix, without case, and the
    model's own name: the 63 dormitory models are named ``…Dorm_…``, and the one among them with ``_Dorm`` parts
    (Phaetusa's) wears nothing else.  A renderer with no suffix, or another one (``_General``, ``_Effect``), is
    worn by both.
    """
    if DORMITORY_MODEL.search(model):
        return {part['name']: False for part in parts}
    hidden = {}
    for part in parts:
        match = re.fullmatch(r'.*_lodm?\d+_([A-Za-z]+)', part['name'])
        hidden[part['name']] = bool(match) and match[1].lower() in DORMITORY_MODES
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
