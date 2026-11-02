"""Attach real Koikatsu assets to a procedurally built character.

Everything here is optional and additive: if no install is configured, or an
item the card references is not present in that install, the procedural piece
stays. Items resolve through two paths — the base game's ChaListData tables, and
sideloader ``.zipmod`` lists via the card's resolver data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import swallowed
from ..mesh.mesh import Material
from . import install as _install
from . import lists as _lists
from . import sideloader as _sl
from .rigadopt import adopt as _adopt_rig
from .rigadopt import rest_apose as _rest_apose
from .skinmap import BoneMapper as _BoneMapper


# Illusion accessory parent keys -> our bones
@dataclass
class Attached:
    name: str
    category: str
    source: str
    tris: int


@dataclass
class ApplyReport:
    attached: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    replaced: list = field(default_factory=list)

    def describe(self) -> str:
        lines = []
        if self.attached:
            lines.append(f"已套用 {len(self.attached)} 件原作素材：")
            for a in self.attached:
                lines.append(f"  · {a.category:12s} {a.name}  ({a.tris:,} 面, {a.source})")
        if self.missing:
            lines.append("")
            # 「維持程序生成」是 §7.36 之前的行為，那條路已經整個刪掉了
            # （CLAUDE.md 一之三）：現在少一件就是畫面上少一件。
            lines.append(f"以下 {len(self.missing)} 項在此安裝中找不到，沒有畫出來：")
            for m in self.missing[:14]:
                lines.append(f"  · {m}")
            if len(self.missing) > 14:
                lines.append(f"  · …另外 {len(self.missing) - 14} 項")
        return "\n".join(lines) or "沒有可套用的原作素材。"


# --------------------------------------------------------------------------
def hair_material(rgb=(0.86, 0.84, 0.82), tip=None) -> Material:
    base = tuple(float(c) for c in rgb[:3])
    return Material(
        name="origin_hair",
        base_color=(*base, 1.0),
        shadow_color=(*(np.clip(np.array(base) * 0.68 + 0.06, 0, 1)), 1.0),
        specular_color=(1.0, 0.98, 0.95, 1.0),
        specular_strength=0.32,
        specular_power=70.0,
        rim_strength=0.28,
        outline_color=(*(np.array(base) * 0.30), 1.0),
        outline_width=0.95,
        shade_steps=2.0,
        shade_feather=0.06,
        two_sided=True,
    )


DEFAULT_OUTFIT = 0          # the character's primary coordinate


def current_outfit(card) -> int:
    """Which coordinate to dress the character in.

    A card stores seven, and coordinate 0 is the character's main outfit -- the
    one the game puts them in and the one a card is recognised by.
    ``Status.coordinateType`` is *not* it: that records whichever tab the maker
    was last left on, so honouring it dresses the character in whatever the
    author happened to be editing (a yukata, on the test card).
    """
    return DEFAULT_OUTFIT


def coordinate_count(card) -> int:
    """How many coordinates this card actually stores (EmotionCreators: 1)."""
    sections = getattr(card, "sections", None)
    coord = sections.get("Coordinate") if isinstance(sections, dict) else None
    return len(coord) if isinstance(coord, list) else 1


def maker_outfit(card) -> int:
    """The coordinate the card's **own thumbnail** was shot in.

    Not what to dress a character in — that is :func:`current_outfit`, and it is
    0 on purpose.  This is the one number that makes the card's two baked-in
    PNGs comparable with our render: ``CustomCapture`` shoots them in the maker
    on whatever tab was selected, which ``Status.coordinateType`` records.
    Locally **530 of 1,241 female cards (42.7%)** and 64 of 145 male ones were
    saved on a tab other than 0 — including the project's own test card
    (``帰国子女``, tab 5) — so a comparison that dresses coordinate 0 and calls
    the thumbnail ground truth is looking at two different outfits on half the
    library, and every accessory placement read off it inherits that.

    Returns ``-1`` when the recorded tab is not a coordinate this card has: six
    cards here write 7, 8, 9 or 11 into a seven-entry list, and dressing an
    empty coordinate silently produces a naked character that reads as a
    loading bug (§一之一).  The caller decides what to do about it; it must not
    be papered over with a 0 that looks like an answer.
    """
    sections = getattr(card, "sections", None)
    status = sections.get("Status") if isinstance(sections, dict) else None
    got = None
    if isinstance(status, dict):
        for key in ("coordinateType", "CoordinateType"):
            v = status.get(key)
            if isinstance(v, (int, float)):
                got = max(0, int(v))
                break
    if got is None:
        return DEFAULT_OUTFIT
    return got if got < coordinate_count(card) else -1


def outfit_makeup(card, index: int = 0) -> tuple[bool, dict]:
    """``(enableMakeup, makeup)`` of a coordinate.

    ``ChaFileCoordinate.SaveBytes`` writes ``len|clothes  len|accessory
    bool|enableMakeup  len|makeup`` — that lone boolean between the second and
    third block is why :func:`outfit_blocks` always stops after two: its loop
    reads the flag byte plus three bytes of the next length as a length and
    gives up.  Lip, eyeshadow, cheek and face paint all live in that third
    block, and the character's own ``face.baseMakeup`` is only used when the
    coordinate does not override it (``ChaControl.ChangeSettingLip`` et al.).
    """
    from ..chara.cardimport import _Reader, _unpack
    coord = card.sections.get("Coordinate")
    if isinstance(coord, list):
        if index >= len(coord):
            return False, {}
        blob = coord[index]
    else:
        blob = (getattr(card, "raw", {}) or {}).get("Coordinate")
    if not isinstance(blob, (bytes, bytearray)):
        return False, {}
    data = bytes(blob)
    r = _Reader(data, 0)
    try:
        for _ in range(2):                           # clothes, accessory
            r.bytes(r.i32())
        enable = bool(data[r.i])
        r.i += 1
        makeup = _unpack(r.bytes(r.i32()))
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.apply.outfit_makeup: for _ in range(2):", exc)
        return False, {}
    return enable, makeup if isinstance(makeup, dict) else {}


# There used to be an unreferenced ``_accessory_transform`` here that decoded
# ``addMove`` into one matrix meant to be multiplied onto a finished accessory.
# That is not where the game puts it — ``UpdateAccessoryMoveFromInfo`` writes
# the two groups onto the prefab's own ``N_move`` / ``N_move2`` nodes, in the
# middle of the hierarchy — and this copy also read only the first of the two.
# The one decoder is :func:`figure.attach.card_move_locals`; a second one with
# nobody calling it is just somewhere for the next reader to copy the old rule
# out of.


def bone_origin(skel, name: str) -> np.ndarray:
    """Bind-pose world position of a bone.

    ``Skeleton.world_position`` sums local offsets and ignores rotation, which
    is exact only while every bind rotation is identity -- true for our
    procedural rig, false once we adopt Illusion's.  Reading it off the bind
    matrix stays correct either way; using the old helper hoisted every hair
    accessory a good 18cm above the head.
    """
    if not skel.has(name):
        return np.zeros(3)
    return skel.bones[skel.id(name)].bind_world[:3, 3].copy()


# Illusion's accessory CategoryNo -> our base-list category name.  The order is
# the maker's own tab order starting at 120 (= none), confirmed against the
# parent nodes the cards use: 121 sits on ``a_n_hair_pin``, 122 on
# ``a_n_headtop``, 128 on ``a_n_wrist``, 129 on ``a_n_hand``.
# --------------------------------------------------------------------------
MAX_IMPORT_TRIS = 900_000       # ceiling for one full card (hair+outfit+acc)


def prepare_for_assets(built, unit: float = 1.0, *, enabled: bool = True,
                       report=None, apose: bool = False, full: bool = True,
                       scale: float | None = None,
                       head_nodes: dict | None = None):
    """Put the skeleton into Illusion's rest pose and clear what won't fit.

    ``full=True`` (default) grafts the **complete** Origin armature from
    ``oo_base`` (ChaControl style — no bone collapse).  Weights then map by
    exact name like ``AssignedAnotherWeights``.

    ``apose`` defaults to **False** (T-pose bind, same as the game).

    ``scale`` overrides the graft size.  Without it the armature is fitted to
    our procedural head height, which silently imposes our head-to-body ratio
    on Illusion's body; a caller holding the card can pass the height its own
    slider asks for instead.

    ``head_nodes`` is the hierarchy of the head prefab the card wears; the
    face-local subtree comes from there rather than from ``oo_base`` (§7.102).

    Returns ``(mapper, rig_scale, adopted)``.
    """
    from .rig import rig as _rig, graft_full_skeleton
    from ..mesh.skeleton import MAX_BONES
    skel = built.skeleton
    ref = _rig()
    origin_head_pos = ref.get("cf_j_head")
    rig_scale = 1.0
    adopted = enabled and origin_head_pos is not None and origin_head_pos[1] > 0.1
    exact = False
    if adopted:
        my_y = float(bone_origin(skel, "cf_j_head")[1])
        rig_scale = my_y / float(origin_head_pos[1]) if origin_head_pos[1] > 0.1 else 1.0
        if scale is not None and float(scale) > 1e-3:
            rig_scale = float(scale)
        if full:
            n = graft_full_skeleton(skel, ref, rig_scale, max_bones=MAX_BONES,
                                    head_nodes=head_nodes)
            exact = n > 0
            if report is not None and n:
                # not a "missing" — record as replaced procedural rig
                report.replaced.append(f"rig: full armature ×{n}")
        if not exact:
            _adopt_rig(skel, ref, rig_scale, apose=False)
        if apose:
            _rest_apose(skel)
        for part in list(built.model.parts):
            name = str(getattr(part, "name", ""))
            if name.startswith("cloth_"):
                built.model.remove(name)
                if report is not None:
                    report.replaced.append(name)
    mapper = _BoneMapper(skel, ref, rig_scale if adopted else unit, exact=exact)
    return mapper, rig_scale, adopted

def apply_card_assets(built, card, *, hair_color=None, replace_hair: bool = True,
                      use_base_body: bool = True, outfit: int | None = None,
                      max_items: int = 64,
                      max_tris: int = MAX_IMPORT_TRIS) -> ApplyReport:
    """Attach the hair, outfit and accessories a card references.

    Delegates to the rewritten ``figure.dress_character`` pipeline so every
    caller (tools, maker, Free H) uses the same full load path.
    """
    from .figure import dress_character
    dressed = dress_character(built, card, outfit=outfit, hair_color=hair_color)
    report = ApplyReport(
        attached=[Attached(a.name, a.category, a.source, a.tris)
                  for a in dressed.attached],
        missing=list(dressed.missing),
        replaced=list(dressed.replaced),
    )
    return report


# ``apply_card_assets_legacy`` and its ``_apply_clothes`` helper lived here.
# They were the pre-``figure`` path, kept "for A/B comparison", and nothing
# called either of them.  They had also stopped being a comparison: the head
# code still read ``cf_Ohitomi_L`` as the iris (it is the sclera) and knew
# nothing of the eye gradient, the makeup layers or the list-driven face
# maps, so running it would not have shown what the loader does now — only
# what it did before §7.17.  That record belongs in PROGRESS.md; a second,
# wrong implementation sitting in the tree is a trap.


# ``apply_base_body`` lived here: the first attempt at swapping in
# ``o_body_a``.  ``figure.body_head.apply_body`` replaced it and nothing
# called it any more.  It was not an alternative either — it hard-coded the
# female prefab (§7.25 made that per-sex), took the albedo raw instead of
# through ``create_body``, and knew nothing of the colour mask, the toon
# parameters, the normal map or the garment alpha mask.


# Body-conforming overlays that exist only to depict sexual fluids.  They come
# through the ordinary clothing slots, so they are filtered by name here rather
# than by anything structural.  **This is the one list** (CLAUDE.md 五 points
# here): the loader and the attach step each carried their own copy of it until
# §7.360, while the predicate beside this tuple had no caller at all.
_EXPLICIT_OVERLAY = ("sirutex", "siru_", "_siru", "汁", "cum_", "_cum")


def is_explicit_overlay(label: str) -> bool:
    """Whether a renderer / mesh name is one of the fluid overlays (五)."""
    low = (label or "").lower()
    return any(m.lower() in low for m in _EXPLICIT_OVERLAY)


# Full face stack from the base head prefab.  Earlier builds only took a
# subset and dropped ``Ohitomi_*02`` (the second iris layer) plus noseline —
# that is why eyes looked hollow / incomplete.
# Still left out: ``*_atari`` (collision) and ``gag_eye_*`` (joke).  The tears
# (``namida_*``) load hidden and ``body_head.set_tears`` shows one by
# ``tearsLv`` (§7.414).
# Transparent overlays.  Drawn opaque they become white slabs over sockets.
# A second ``_settle_mesh`` lived here — a whole-piece rotation toward gravity,
# with a floor test so a long sleeve would not swing through the ground.  It had
# no callers left, and the one in ``figure/attach.py`` that did have one was the
# reason three front-hair strands lay flat across a face (§7.66 三 6): the game
# never rotates an attached piece at all.  Sway is ``DynamicBone``, solved to
# steady state by ``prefab.sway_rest`` and applied per bone by
# ``BoneMapper.rest_pose`` / ``_settle_chain`` — which is where a piece whose
# sway components we cannot read is approximated, and the only place it should be.
# Deleted rather than kept dead so it does not come back.


def available() -> bool:
    return _install.enabled()


def describe() -> str:
    inst = _install.configured()
    if not inst.valid:
        return inst.describe()
    base = _lists.index(inst)
    mods = _sl.index()
    return (f"{inst.describe()}\n"
            f"原版素材 {sum(base.categories().values())} 件 / "
            f"模組 {mods.stats()['mods']} 個 {mods.stats()['items']} 件")
