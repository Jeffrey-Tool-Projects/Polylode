"""Play the converted H clips on an Endfield character with a completed body.

An H clip is absolute local transforms on Koikatsu's bones, and everything
that reads it — the size blend, the insertion aim, the phase lists — asks a
skeleton with those names.  A completed Endfield character has its own
skeleton, with the Koikatsu body fitted onto it (``endfield/body.py``), so it
plays a clip in two steps:

1. the clip is sampled on a **stand-in**: the reference rig the fitted body
   came from, with the bones its source body added;
2. the stand-in's world pose is carried onto the character
   (:class:`~polylode.endfield.retarget.Retarget`, the same world-turn rule
   the project's own poses use), and the source joints the fit kept
   (``body.GRAFT``, ``Origin_body_*``: the helper joints the lent skin hangs on, the man's shaft)
   move against the bone they hang under the way their sources move against
   that bone's source — which is how a clip's knee and hip scales reach the
   skin.

**The pelvis follows ``cf_j_waist02``, not ``cf_j_hips``.**  In Koikatsu's rig
``cf_j_hips`` is the root of the spine *and* of the waist chain; the legs and
the genitals hang off ``cf_j_waist02``, and a clip turns the two differently
(on ``khs_f_00`` WLoop the hips are 111° from bind and ``cf_j_waist02``
125–136°, and the stroke is ``cf_j_waist02``'s height).  The fit anchors the
pelvis the same way (``fit_body``: ``cf_s_waist02`` onto the Hips node).

**Where a Koikatsu point is on the character** (:meth:`StandIn.point`) is
the fit's own rule: a point rides the nearest carried bone above it, displaced
by what the fit displaced that bone's skin by.  The insertion aim reads the
woman's entrance null and the man's shaft base this way, so the aim is done
between the two bodies that are drawn, not between their stand-ins.

Completed adult bodies and certified native opaque reconstructions with a
local adult age declaration get a stand-in. Anatomical body scope stays
separate (``hscope`` and ``body_scope``).
"""
from __future__ import annotations

import numpy as np

from ..anim.clip import Pose
from ..core import math3d as m3
from ..mesh.skeleton import Skeleton
from . import body_scope, hscope, store
from .actor import ActorError
from .body import GRAFT
from .retarget import HIPS, SOURCE_BONES, Retarget

#: Unity human bone → the stand-in's bone an H clip moves it by.
H_BONES = {**SOURCE_BONES, HIPS: "cf_j_waist02"}
#: Human bones whose turn an H clip puts on the reference rig's joint below the one they stand at:
#: the clips bend the ankle on ``cf_j_foot`` (27° past ``cf_j_leg03`` either way in 後背位 and 壁バック),
#: and carried from ``cf_j_leg03`` a foot kept its rest bend — toes 9 cm into the floor kneeling (§7.1072)
FOOT_TURNS = {"LeftFoot": "cf_j_foot_L", "RightFoot": "cf_j_foot_R"}
#: The bone the fit displaces the pelvis' skin by (``body.fit_body``).
PELVIS_ANCHOR = "cf_s_waist02"
#: The fitted body's piece in the stored source body.
SOURCE_BODY = "o_body_a"
#: The stand-in's bone at the middle of its lips (:meth:`StandIn.point`).
SOURCE_MOUTH = "cf_J_MouthMove"
#: The rig families that are women; the others in scope are men.
FEMALE_FAMILIES = hscope.FEMALE_FAMILIES
#: The men an H scene offers: the one man converted (``body_scope.MEN``; CLAUDE.md, §7.919).
MEN = body_scope.MEN


def stand_in_skeleton(implant: list | None) -> Skeleton:
    """The reference rig as a skeleton, with a source body's own bones (*implant*) added."""
    from ..origin.rig import graft_full_skeleton, rig
    skeleton = Skeleton("h_stand_in")
    if not graft_full_skeleton(skeleton, rig()):
        raise ActorError("the reference rig is not converted; convert the base body first")
    for row in implant or ():
        name, parent = str(row.get("name") or ""), str(row.get("parent") or "")
        if not name or skeleton.has(name) or not skeleton.has(parent):
            continue
        pos, rot, scale = m3.decompose(np.asarray(row["local"], dtype=np.float64).reshape(4, 4))
        skeleton.add(name, parent, pos, rot, scale)
    skeleton.finalize()
    return skeleton


def family(character: str) -> str:
    """The supported adult H-role rig family; ``ValueError`` outside it."""
    return hscope.family(store.read_extra("chara", character) or {}, character)


#: the source body a completed body was fitted from (``body.bake_body``) → whether it is a woman's
SOURCE_BODY_FEMALE = {"base/body": True, "base/body_m": False}


def body_female(character: str) -> bool | None:
    """Whether *character*'s completed body is a woman's; ``None`` when it has none.

    Read off the source body its card chose when it was built, not off the
    rig's label: the body that is drawn is that one.
    """
    return SOURCE_BODY_FEMALE.get((store.read_extra("body", character) or {}).get("source_body"))


def completed(character: str, have: set | None = None) -> bool:
    """Whether *character* has a completed body that loads: masks done and its source body stored."""
    have = set(store.items()) if have is None else have
    return (("body", character) in have and ("body_source", character) in have
            and (store.read_extra("body", character) or {}).get("composite") is not None)


def declared_minor(character: str) -> bool:
    """Whether the owner's age table declares *character* under 18."""
    age = hscope.declared_age(store.read_extra("chara", character) or {})
    return age is not None and age < 18


def native_reconstruction(character: str) -> bool:
    """Whether a declared adult has a current certified native opaque reconstruction.

    This source carries poses on its own rig without an anatomical body or
    fitted source joints. Unknown ages and stale certifications are excluded.
    """
    from . import reconstruction
    extra = store.read_extra("chara", character) or {}
    age = hscope.declared_age(extra)
    if age is None or age < 18:
        return False
    try:
        kind = family(character)
    except ValueError:
        return False
    if kind not in FEMALE_FAMILIES:
        return False
    if not store.has(reconstruction.CATEGORY, character):
        return False
    rebuilt = store.read_extra(reconstruction.CATEGORY, character) or {}
    manifest = rebuilt.get("reconstruction") or {}
    if manifest.get("source_kind") != "opaque_npc" or manifest.get("opaque_source_verified") is not True:
        return False
    try:
        reconstruction.require_current(character, rebuilt)
    except ActorError:
        return False
    return True


def h_scope(character: str, *, native: bool = False) -> bool:
    """Whether the body an H scene plays for *character* is a woman's; ``ActorError`` when it may not play it.

    A character the age table declares under 18 is refused, and so is a man
    who is not one of :data:`MEN`.  The body is the completed body's source
    (:func:`body_female`), or for a *native* opaque reconstruction its rig family.
    Asked of every character a clip is played on: through its stand-in, or on
    the source game's rig fitted to it (``ActorController.h_stand_in``).
    """
    if declared_minor(character):
        raise ActorError(f"{character}: the age table declares this character under 18")
    female = (family(character) in FEMALE_FAMILIES) if native else bool(body_female(character))
    if not female and character not in MEN:
        raise ActorError(f"{character}: not one of the men an H scene plays")
    return female


def eligible() -> list[tuple[str, bool]]:
    """``(character, whether a woman)`` of every converted character an H scene may load.

    A completed body (:func:`completed`) — a woman's, or of the men those in
    :data:`MEN` — whose character the age table does not declare a minor.
    The rig's label is not asked: building the body already passed the
    scope check, and which body it is comes from its source (:func:`body_female`).
    A character without one is offered by its certified native opaque
    reconstruction, if it has one, or — by the same rule as a completed
    body — by a certified complete-body reconstruction that stands on the
    source rig's joints its body fit kept (what :func:`category` plays; Si,
    owner 2026-10-15, §7.1088).  Anything else is left out, not refused
    later: a list only offers what can be played.
    """
    from . import reconstruction
    have = set(store.items())
    out = set()
    for kind, ident in sorted(have):
        if kind != "body" or not completed(ident, have) or declared_minor(ident):
            continue
        female = body_female(ident)
        if female or (female is False and ident in MEN):
            out.add((ident, female))
    for kind, ident in sorted(have):
        if kind != reconstruction.CATEGORY or completed(ident, have):
            continue
        if native_reconstruction(ident):
            out.add((ident, True))
        elif not declared_minor(ident) and category(ident) == reconstruction.CATEGORY:
            female = body_female(ident)
            if female or (female is False and ident in MEN):
                out.add((ident, female))
    return sorted(out)



def category(character: str) -> str:
    """Which stored item of *character* an H scene loads: its completed body, else its certified reconstruction.

    The completed body is the card's body (its skin, nipples included) fitted
    to the character and joined to her own skin; it is played whenever it is
    there.  Without one, a certified reconstruction whose source joints fit
    its bones, or a declared adult native opaque reconstruction on its own
    rig, is played; short of that, ``"body"`` (which then fails to load).
    """
    from . import reconstruction
    if completed(character):
        return "body"
    if native_reconstruction(character):
        return reconstruction.CATEGORY
    stored = store.has(reconstruction.CATEGORY, character)
    extra = store.read_extra(reconstruction.CATEGORY, character) if stored else None
    if extra is None:
        return "body"
    try:
        reconstruction.require_current(character, extra)
    except ActorError:
        return "body"
    item = store.read_item(reconstruction.CATEGORY, character)
    from .actor import build_skeleton
    fits = item is not None and item.nodes and reconstruction.source_joints(
        character, build_skeleton(item.nodes, character))
    return reconstruction.CATEGORY if fits else "body"


def label(character: str) -> str:
    """A character's name as a list shows it: the model's own name, capitalised.

    A converted source's item names its character (``display_name``, from its
    verified folder name) and its model: both are shown.
    """
    named = (store.read_extra("chara", character) or {}).get("display_name")
    if named:
        return f"{named} · {character}"
    parts = character.split("_")
    name = parts[2] if len(parts) > 2 and parts[0] == "chr" else character
    return name[:1].upper() + name[1:]


def shape_owned() -> dict:
    """``{bone: (position components, rotation, scale)}`` the source game's shape pass writes.

    ``ShapeBodyInfoFemale.Update`` sets these channels from the body sliders
    every frame, after the Animator: a clip's curve on one of them never
    reaches the screen there.  The clips carry such curves all the same (a
    ``cf_s_leg01`` position half a metre off), so a pose is read without them
    — the stand-in has no sliders, and its rest is what the pass would write.
    """
    global _SHAPE_OWNED
    if _SHAPE_OWNED is None:
        from ..origin.figure.shape_update import BODY_UPDATE
        owned: dict = {}
        for bone, setter, *_rest in BODY_UPDATE:
            axes, rotation, scale = owned.get(bone, (frozenset(), False, False))
            if setter == "Scale":
                scale = True
            elif setter.startswith("Rotation"):
                rotation = True
            elif setter == "Position":
                axes = frozenset((0, 1, 2))
            elif setter.startswith("Position"):
                axes = axes | {"XYZ".index(setter[-1])}
            owned[bone] = (axes, rotation, scale)
        _SHAPE_OWNED = owned
    return _SHAPE_OWNED


_SHAPE_OWNED: dict | None = None


def without_shape_channels(pose: Pose) -> Pose:
    """*pose* without the channels :func:`shape_owned` lists (a new dict where one is dropped)."""
    owned = shape_owned()
    out = pose
    for bone in pose.keys() & owned.keys():
        axes, rotation, scale = owned[bone]
        entry = dict(pose[bone])
        if scale:
            entry.pop("scale", None)
        if rotation:
            entry.pop("rot", None)
        if axes and entry.get("pos") is not None:
            moved = np.array(entry["pos"], dtype=np.float64)
            moved[list(axes)] = 0.0
            entry["pos"] = moved
        if out is pose:
            out = dict(pose)
        out[bone] = entry
    return out


def _additive_many(bones: list, local: np.ndarray) -> Pose:
    """:func:`_additive` for many bones at once: ``{bone name: entry}`` for the ``(K, 4, 4)`` *local*."""
    position = local[:, :3, 3]
    rot, scale = m3.quats_from_matrices(local)
    # after the bone's own rest: conj(rest) * rot
    rest = np.array([bone.local_rot for bone in bones], dtype=np.float64)
    ax, ay, az, aw = -rest[:, 0], -rest[:, 1], -rest[:, 2], rest[:, 3]
    bx, by, bz, bw = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    turn = np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], axis=1)
    turn /= np.maximum(np.linalg.norm(turn, axis=1, keepdims=True), m3.EPS)
    rest_pos = np.array([bone.local_pos for bone in bones], dtype=np.float64)
    rest_scale = np.array([getattr(bone, "local_scale", (1.0, 1.0, 1.0)) for bone in bones], dtype=np.float64)
    moved = position - rest_pos
    scaled = scale / np.maximum(rest_scale, 1e-8)
    entries = {bone.name: {"pos": moved[k], "rot": turn[k], "scale": scaled[k]} for k, bone in enumerate(bones)}
    # Flat grafts can contain shear inherited from their source's scaled hierarchy.
    # Keep that authored transform rather than projecting it onto rotation and scale.
    from ..mesh.skeleton import _compose_many
    reconstructed = _compose_many(position, rot, scale)
    bound = 64 * np.finfo(float).eps * np.maximum(1., np.abs(local).max(axis=(1, 2)))
    sheared = np.abs(reconstructed - local).max(axis=(1, 2)) > bound
    for k in np.flatnonzero(sheared):
        entries[bones[k].name]["matrix"] = local[k].copy()
    return entries


def _additive(bone, local: np.ndarray) -> dict:
    """The absolute local matrix *local* as the entry ``Skeleton.evaluate`` takes for *bone*."""
    return _additive_many([bone], np.asarray(local, dtype=np.float64)[None])[bone.name]


#: The share of the lent skin nearest a helper bone's vertices that one kept joint has to hold
#: for the bone to follow it (``StandIn.bind_worn``).
FOLLOW_SHARE = 0.5
#: The human bones whose own bones below them take the clip's turns too (:func:`finger_pairs`).
EXTREMITIES = ("LeftHand", "RightHand")
#: The limbs whose ends a clip places (:meth:`StandIn.reach`): root, middle joint, end.
LIMBS = (("LeftUpperArm", "LeftLowerArm", "LeftHand"), ("RightUpperArm", "RightLowerArm", "RightHand"),
         ("LeftUpperLeg", "LeftLowerLeg", "LeftFoot"), ("RightUpperLeg", "RightLowerLeg", "RightFoot"))


def height_slider(scale: float) -> float | None:
    """The body-height slider at which the source game scales a body by *scale*; None when it cannot be read.

    The clips are authored three times over, for a small, a middle and a
    large woman, and both partners play the mix her height slider picks: that
    is how the game keeps two bodies of different sizes in place against each
    other.  A converted character has no slider, but it has a size — its
    pelvis stands at *scale* times its stand-in's — and the slider's own
    curve (``shape_apply.height_scale``) says which slider that is.  Not
    clamped to the slider's range: the blend that reads it clamps.
    """
    from types import SimpleNamespace

    from ..origin.figure import shape_apply

    def at(value: float) -> float | None:
        values = [0.5] * (shape_apply.HEIGHT_SLIDER + 1)
        values[shape_apply.HEIGHT_SLIDER] = value
        # a woman's card: the slider is hers (a man's is overwritten by the game, ``height_override``)
        return shape_apply.height_scale(SimpleNamespace(body=lambda: {"shapeValueBody": values}, sex=1,
                                                        parameter={"sex": 1}))
    low, high = at(0.0), at(1.0)
    if low is None or high is None or abs(high - low) < 1e-9:
        return None
    lo, hi = 0.0, 1.0
    if not min(low, high) <= scale <= max(low, high):
        return lo + (scale - low) / (high - low)              # past the slider's ends: along its two ends' line
    for _ in range(40):                                        # the curve rises with the slider
        middle = (lo + hi) / 2.0
        if (at(middle) < scale) == (low < high):
            lo = middle
        else:
            hi = middle
    return (lo + hi) / 2.0


def _ring_normal(points, faces: np.ndarray) -> np.ndarray:
    """The area-weighted normal of *faces* (vertex rows into *points*, an array or a dict)."""
    a, b, c = (np.array([points[int(v)] for v in faces[:, k]], dtype=np.float64) for k in range(3))
    normal = np.cross(b - a, c - a).sum(axis=0)
    length = float(np.linalg.norm(normal))
    return normal / length if length > 1e-12 else np.array([0.0, 1.0, 0.0])


def finger_pairs(target: Skeleton, humanoid: dict, source: Skeleton) -> dict:
    """``{character's bone: source joint}`` for the character's own bones below its hands.

    The Avatar names no fingers, so a clip's finger curls had nowhere to go.
    The fit did map them: each kept source joint hangs under the character's
    bone the fit put it on (``body.helper_nodes``).  Of the kept joints under
    one such bone, the source joint standing nearest the bone is the one that
    bone follows.  Only below the hands: bones elsewhere that the Avatar does
    not name belong to something else (cloth, hair, a tail).
    """
    below = set()
    for human in EXTREMITIES:
        node = humanoid.get(human)
        if node and target.has(node):
            below |= set(target.descendants(node)) - {target.index[node]}
    named = {target.index[node] for node in humanoid.values() if target.has(node)}
    bind = target.evaluate({})
    nearest: dict[int, tuple[float, str]] = {}
    for index, bone in enumerate(target.bones):
        name = bone.name[len(GRAFT):]
        parent = bone.parent
        if (not bone.name.startswith(GRAFT) or not source.has(name) or parent not in below
                or parent in named or target.bones[parent].name.startswith(GRAFT)):
            continue
        apart = float(np.linalg.norm(bind[index][:3, 3] - bind[parent][:3, 3]))
        if parent not in nearest or apart < nearest[parent][0]:
            nearest[parent] = (apart, name)
    return {target.bones[parent].name: name for parent, (_apart, name) in sorted(nearest.items())}


def worn_followers(bones: list, carried: set, driven: set, lent, worn: list, owners=None) -> list[tuple[int, int]]:
    """``[(helper bone, kept joint)]``: the rule of :meth:`StandIn.bind_worn`, on the character's own skeleton.

    *bones* are the character's skeleton's bones; *carried* the bones the
    Avatar names (by index), *driven* the kept joints a clip moves.  *lent*
    is the lent body's whole mesh, *worn* the meshes of everything else.
    *owners* (``bone index of a mesh → bone index of *bones*``) reads meshes
    whose bone indices are another skeleton's (the source game's rig the
    character is on, ``converted_rig.Rebound``); None when they are these.
    """
    from scipy.spatial import cKDTree

    def heaviest(mesh) -> np.ndarray:
        found = mesh.bone_idx[np.arange(mesh.vertex_count), np.argmax(mesh.bone_wt, axis=1)].astype(np.int64)
        return found if owners is None else np.asarray(owners, dtype=np.int64)[found]
    skin_joint = heaviest(lent)
    tree = cKDTree(lent.positions.astype(np.float64))
    votes: dict[int, np.ndarray] = {}
    for mesh in worn:
        if not mesh.vertex_count or not len(mesh.indices):
            continue
        used = np.unique(mesh.indices.astype(np.int64))
        own = heaviest(mesh)[used]
        _distance, near = tree.query(mesh.positions.astype(np.float64)[used])
        for bone in np.unique(own):
            b = bones[int(bone)]
            if (int(bone) in carried or b.name.startswith(GRAFT) or b.parent not in carried
                    or any(not bones[child].name.startswith(GRAFT) for child in b.children)):
                continue                       # a bone with a chain of its own under it is cloth or hair
            tally = votes.setdefault(int(bone), np.zeros(len(bones), dtype=np.int64))
            np.add.at(tally, skin_joint[near[own == bone]], 1)
    found = []
    for bone, tally in sorted(votes.items()):
        joint = int(np.argmax(tally))
        if (tally[joint] < FOLLOW_SHARE * tally.sum() or joint not in driven
                or bones[joint].parent != bones[bone].parent):
            continue
        found.append((bone, joint))
    return found


def _face_frames(posed: np.ndarray, faces: np.ndarray, outward: np.ndarray) -> tuple:
    """``(middles, unit normals, a tree of the middles)`` of the triangles *faces* of the vertices *posed*.

    Each normal is turned *outward* (±1).  The tree is made once per pose and
    asked once per hand and step: a dozen joints against every face in reach
    directly took 18 ms a frame, the tree a few.
    """
    from scipy.spatial import cKDTree
    corners = posed[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal *= outward[:, None] / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    middle = corners.mean(axis=1)
    return middle, normal, cKDTree(middle)


def _outward(mesh) -> np.ndarray:
    """Per face of *mesh*, +1 where its winding faces the way its corners' normals do, else -1.

    Winding is not one way over the meshes a woman is drawn with (EndminF's
    own skin is wound inward against her lent body, §7.1103); the normals
    are, since she is shaded by them.
    """
    rest, normals = mesh.skin_rest()
    faces = mesh.indices.reshape(-1, 3).astype(np.int64)
    corners = np.asarray(rest, dtype=np.float64)[faces]
    winding = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    if len(normals) == len(rest):
        shaded = np.asarray(normals, dtype=np.float64)[faces].sum(axis=1)
        sign = np.where(np.einsum("ij,ij->i", winding, shaded) < 0.0, -1.0, 1.0)
    else:
        sign = np.ones(len(faces))
    return sign


class SkinPair:
    """A body a clip holds things against, the body drawn in its place, and the ways from one to the other.

    The first is the source body as the clip poses it (:meth:`_posed_source`,
    ``_source_body`` its piece); the second is her drawn skin, the lent body
    :attr:`lent` and her own surfaces :attr:`native` on :attr:`target`, each
    skinned as :attr:`native_modes` says.  The lent body's vertices are the
    source body's, in order.  :class:`StandIn` is one; a character on the
    source game's rig is another (``endfield.rig_hold``).
    """

    def on_skin(self, at: np.ndarray, worlds: tuple[np.ndarray, np.ndarray], lent) -> np.ndarray | None:
        """Where the point *at* beside the stand-in's skin (stand-in frame) is beside the character's.

        The lent body's vertices are the source body's, in order (``barefoot``):
        the source vertex nearest *at*, on the stand-in as *worlds* poses it,
        is the same vertex of the lent body *lent* on the character.  *at*
        keeps its offset from it, turned as the skin's normal there is turned
        from the stand-in's to hers: kept as it was, a hand held over a hip
        whose skin faces another way went into it (§7.1073).  None without a
        source body.
        """
        if self._source_body is None or lent is None:
            return None
        source, target = worlds
        posed = self._posed_source(source)
        count = min(len(posed), lent.vertex_count)
        vertex = int(np.argmin(np.linalg.norm(posed[:count] - at, axis=1)))
        faces = self._ring(vertex)
        faces = faces[np.all(faces < count, axis=1)]
        ring = np.unique(np.append(faces, vertex))
        drawn = {int(v): self._lent_at(lent, target, int(v)) for v in ring}
        turn = (m3.quat_from_to(_ring_normal(posed, faces), _ring_normal(drawn, faces)) if len(faces)
                else m3.quat_identity())
        return drawn[vertex] + m3.quat_rotate(turn, at - posed[vertex])

    #: How far from a skin a point is looked at by :meth:`clearance` (m): beyond it, it does not touch.
    CLEARANCE_REACH = 0.04
    #: How far off her skin a hand the clip has on the stand-in's is still brought in (m): her skin is
    #: where her proportions put it, and hands placed by the stand-in stood 61-127 mm off it (§7.1073).
    PULL_REACH = 0.2

    def clearance(self, points: np.ndarray, clipped: np.ndarray, worlds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """How far to move *points* (her model frame) for none to be deeper in her drawn skin than in the stand-in's.

        *clipped* are the same points where the clip has them, against the stand-in.

        His hands are put on her by the stand-in's skin (:meth:`on_skin`), but
        her thighs and hips are not the stand-in's shape, and a hand placed by
        one vertex went 12-18 mm into her (3 mm on a card woman: the clip's own
        grip, §7.1103).  Each point's signed distance to the stand-in's skin, as
        the clip has them, is what it should keep from hers; the point that
        falls shortest says how far, and her skin's normal there which way.
        When nothing is short, a hand the clip holds on the stand-in's skin but
        that is off hers is brought in by the least excess.  A zero vector when
        nothing needs it or nothing is near.
        """
        if self._source_body is None:
            return np.zeros(3)
        theirs = self._signed(self._source_faces(worlds[0]), clipped)
        found = self._drawn_signed(points, worlds)
        if found is None:
            return np.zeros(3)
        gap, depth, normal = found
        their_gap, their_depth, _normal = theirs
        near = (gap < self.CLEARANCE_REACH) & (their_gap < self.CLEARANCE_REACH)
        short = np.where(near, their_depth - depth, 0.0)
        worst = int(np.argmax(short))
        if short[worst] > 1e-4:
            return normal[worst] * short[worst]
        # the other way: on the stand-in in the clip, in the air beside her (her upper body is her own
        # skin and no vertex of the lent body answers for it: Zhuangfy's hands 70 mm off her, §7.1103).
        # Brought in by the least any of them is too far, so none goes deeper than the clip has it
        held = (their_gap < self.CLEARANCE_REACH) & (gap < self.PULL_REACH)
        if not held.any():
            return np.zeros(3)
        excess = np.where(held, depth - their_depth, np.inf)
        least = int(np.argmin(excess))
        return -normal[least] * excess[least] if 1e-4 < excess[least] < np.inf else np.zeros(3)

    def _drawn_signed(self, points: np.ndarray, worlds) -> tuple | None:
        """:meth:`_signed` against her drawn skin (the lent body and her own), per point the nearest surface's."""
        found = [self._signed(faces, points) for faces in self._drawn_faces(worlds[1])]
        if not found:
            return None
        which = np.argmin(np.stack([gap for gap, _depth, _normal in found]), axis=0)
        rows = np.arange(len(points))
        return tuple(np.stack([entry[k] for entry in found])[which, rows] for k in range(3))

    def _frame_cached(self, key: str, matrices: np.ndarray, make):
        """*make()*, kept while *matrices* is the same array: one H frame asks for it per hand, per step."""
        cache = getattr(self, "_frame_cache", None)
        if cache is None:
            cache = self._frame_cache = {}
        held = cache.get(key)
        if held is not None and held[0] is matrices:
            return held[1]
        made = make()
        cache[key] = (matrices, made)
        return made

    def _outward(self, mesh) -> np.ndarray:
        """:func:`_outward` of *mesh*, made once: its faces and normals do not change while it is drawn."""
        cache = getattr(self, "_outward_cache", None)
        if cache is None:
            cache = self._outward_cache = {}
        held = cache.get(id(mesh))
        if held is None or held[0] is not mesh:
            held = cache[id(mesh)] = (mesh, _outward(mesh))
        return held[1]

    def _source_faces(self, source: np.ndarray) -> tuple:
        """:func:`_face_frames` of the source body's faces on the stand-in as *source* poses it."""
        return self._frame_cached("source faces", source, lambda: _face_frames(
            self._posed_source(source), self._source_body.mesh.indices.reshape(-1, 3).astype(np.int64),
            self._outward(self._source_body.mesh)))

    def _drawn_faces(self, target: np.ndarray) -> list:
        """:func:`_face_frames` of each of her drawn surfaces as *target* poses her, each skinned its way."""
        from ..mesh import skinning

        def make():
            skin = self.target.skinning_matrices(world=target)
            modes = getattr(self, "native_modes", None) or {}
            rests = self.target.rest_frames()
            out = []
            for mesh in ([self.lent] if self.lent is not None else []) + list(self.native or ()):
                mode, share = modes.get(id(mesh), (skinning.LINEAR, None))
                posed = skinning.skinned(mesh.skin_rest()[0].astype(np.float64), mesh.bone_idx, mesh.bone_wt, skin,
                                         mode, rests, None if share is None or not len(share) else share)
                out.append(_face_frames(posed, mesh.indices.reshape(-1, 3).astype(np.int64), self._outward(mesh)))
            return out
        return self._frame_cached("drawn faces", target, make)

    @staticmethod
    def _signed(faces: tuple, points: np.ndarray) -> tuple:
        """``(distance to the nearest face's middle, signed distance along its normal, the normal)`` per point."""
        middle, normal, tree = faces
        gap, nearest = tree.query(points)
        return gap, np.einsum("ij,ij->i", points - middle[nearest], normal[nearest]), normal[nearest]

    def _ring(self, vertex: int) -> np.ndarray:
        """The source body's triangles round *vertex* (``(F, 3)`` vertex rows)."""
        if self._rings is None:
            tri = self._source_body.mesh.indices.reshape(-1, 3).astype(np.int64)
            order = np.argsort(tri.ravel(), kind="stable")
            self._rings = (tri, order, np.searchsorted(tri.ravel()[order], np.arange(tri.max() + 2)))
        tri, order, starts = self._rings
        if vertex + 1 >= len(starts):
            return np.zeros((0, 3), dtype=np.int64)
        return tri[np.unique(order[starts[vertex]:starts[vertex + 1]] // 3)]

    def _lent_at(self, lent, target: np.ndarray, vertex: int) -> np.ndarray:
        """The lent body's *vertex* where the character's world matrices *target* put it (linear blend).

        As it is drawn: from where its bones bind it (``Mesh.skin_rest``), by the binds the skin was
        given (``body.unblend``).  Its rest position by the fitted binds blends twice, up to 2 cm off
        at the shoulders (§7.958).
        """
        bones = lent.bone_idx[vertex].astype(np.int64)
        weight = lent.bone_wt[vertex].astype(np.float64)
        place = np.append(lent.skin_rest()[0][vertex].astype(np.float64), 1.0)
        return sum(w * (target[b] @ self.target.bones[b].inv_bind @ place)[:3]
                   for w, b in zip(weight, bones) if w > 0)

class StandIn(SkinPair):
    """The skeleton an H clip is sampled on for one character, and the way back.

    *target* is the character's skeleton, *humanoid* its human-bone table.
    *native*: the character is played as its native opaque reconstruction
    (the reference rig alone, no source body); otherwise its completed body,
    whose source says whether it is a woman's (:func:`body_female`).
    """

    def __init__(self, character: str, target: Skeleton, humanoid: dict, *, native: bool = False):
        #: whether the body played is a woman's
        self.female = h_scope(character, native=native)
        source = None if native else store.read_item("body_source", character)
        if source is None and not native:
            raise ActorError(f"{character}: the source body its completed body was fitted from is not stored")
        body = next((p for p in source.pieces if p.mesh.name == SOURCE_BODY), None) if source else None
        if not native and (body is None or PELVIS_ANCHOR not in body.bones):
            raise ActorError(f"{character}: the stored source body has no {SOURCE_BODY} with {PELVIS_ANCHOR}")
        self.target = target
        self.skeleton = stand_in_skeleton(getattr(source, "implant", None))
        #: the character's finger bones ← the source joints fitted onto them (:func:`finger_pairs`)
        self.fingers = finger_pairs(target, humanoid, self.skeleton)
        hand_of = {}
        for human in EXTREMITIES:
            if humanoid.get(human) and target.has(humanoid[human]):
                below = {target.bones[i].name for i in target.descendants(humanoid[human])}
                hand_of.update({node: human for node in self.fingers if node in below})
        self.retarget = Retarget(target, {**humanoid, **{node: node for node in self.fingers}},
                                 source=self.skeleton, names={**H_BONES, **self.fingers}, alike=hand_of,
                                 turns=FOOT_TURNS)
        self._source_bind = self.skeleton.evaluate({})
        self._truth = None
        #: the joint correction for the stand-in (False: not converted), and the scaffold's way onto it
        self._corrector = None
        self._from_scaffold = None
        self._target_bind = target.evaluate({})
        self._target_unbind = np.linalg.inv(self._target_bind)
        if native:
            if not self.skeleton.has(PELVIS_ANCHOR):
                raise ActorError(f"{character}: the reference rig has no {PELVIS_ANCHOR}")
            anchor = self._source_bind[self.skeleton.index[PELVIS_ANCHOR]][:3, 3]
        else:
            anchor = np.asarray(body.bone_world[body.bones.index(PELVIS_ANCHOR)], dtype=np.float64)[:3, 3]
        #: carried source bone index → (target bone index, what the fit moved its skin by)
        self._carried: dict[int, tuple[int, np.ndarray]] = {}
        for human, (src, tgt) in self.retarget.pairs.items():
            base = anchor if human == HIPS else self._source_bind[src][:3, 3]
            self._carried[src] = (tgt, self._target_bind[tgt][:3, 3] - base)
        #: whether the kept joints are moved by the clip (off: they follow their bones rigidly,
        #: which is what the skin did before the joints were kept — for measuring one against the other)
        self.drive_kept = True
        #: whether the hands and feet are brought to where the clip has them (:meth:`reach`)
        self.reach_limbs = True
        #: the stand-in bone the playing row is about, when that is not the pelvis (:meth:`hold_anchor`)
        self.anchor: str | None = None
        #: ``(barefoot.Barefoot, the sides whose foot is bare)`` while a completed body's feet are bare,
        #: or None (:meth:`_end_goal`)
        self.feet = None
        #: the lent body (``CompositeVisibility.source``) a hand on her own body is put on (:meth:`_end_goal`)
        self.lent = None
        #: her own skin drawn with the lent body (the native skin parts' meshes, :meth:`keep_above`)
        self.native = None
        #: ``{id(mesh): (skinning mode, dual share)}`` of :attr:`native` (linear when absent)
        self.native_modes: dict = {}
        #: ``{id(mesh): (mesh, its vertex rows above the feet, of how many)}`` (:meth:`_above_feet`)
        self._body_rows: dict = {}
        #: the other body's joints in this character's frame for the frame being carried, or None (:meth:`_end_goal`)
        self.other: np.ndarray | None = None
        #: (the stand-in's head, its mouth, the character's two lip joints) when all are there
        self._mouth = None
        from .morph import LIPS
        if "Head" in self.retarget.pairs and self.skeleton.has(SOURCE_MOUTH) and all(target.has(n) for n in LIPS):
            self._mouth = (self.retarget.pairs["Head"][0], self.skeleton.index[SOURCE_MOUTH],
                           tuple(target.index[n] for n in LIPS))
        hips_source, hips_target = self.retarget.pairs[HIPS]
        #: what the carried pelvis stands off from the stand-in's, scaled to the character's size
        self._size = float(self.retarget._hip_ratio)
        self._stand_off = self._target_bind[hips_target][:3, 3] - self._size * self._source_bind[hips_source][:3, 3]
        #: how high the stand-in's ankles stand at rest: a hand or foot no higher is on the floor (:meth:`_goal`)
        feet = [self.retarget.pairs[side][0] for side in ("LeftFoot", "RightFoot") if side in self.retarget.pairs]
        self._ankle = float(np.mean([self._source_bind[foot][1, 3] for foot in feet])) if feet else None
        #: the character's own helper bones that go with a kept joint (:meth:`bind_worn`):
        #: (bone, joint, ``inverse(joint bind) · bone bind``)
        self._followers: list[tuple[int, int, np.ndarray]] = []
        #: ``_hung`` as arrays, made on first use
        self._hung_table = None
        #: the last pose carried and what it came to:
        #: (pose, source worlds, carried pose, target worlds or None, whether the kept joints were driven)
        self._last = None
        #: ``target bone → 4×4``: body edits of the kept joints, in each joint's own frame (``endfield/edits.py``)
        self.joint_offsets: dict = {}
        self._source_body = body
        #: the source body's triangles by vertex (:meth:`_ring`), made on first use
        self._rings = None
        self._pelvis = self.retarget.pairs[HIPS]
        source_of = {tgt: src for src, tgt in self.retarget.pairs.values()}
        #: fitted joints whose parent is a fitted joint too: (target index, source index), parents first
        self._nested: list[tuple[int, int]] = []
        #: fitted joints under one of the character's carried bones: (target index, source index,
        #: that bone's source index, ``inverse(bone bind) · fit · source bone bind``)
        self._hung: list[tuple[int, int, int, np.ndarray]] = []
        #: fitted joints under one of the character's finger bones: (target index, source index, the
        #: hand's source index, ``inverse(hand bind) · fit · source hand bind``, the character's bones
        #: from under the hand down to the joint's parent).  The finger bone turns about its own
        #: root, which is not where the source's finger joint stands; the lent skin goes by the
        #: source's chain, measured from the hand.
        self._fingered: list[tuple[int, int, int, np.ndarray, list[int]]] = []
        carried_fingers = {target.index[node] for node in self.fingers}
        for index, bone in enumerate(target.bones):
            name = bone.name[len(GRAFT):]
            if not bone.name.startswith(GRAFT) or not self.skeleton.has(name):
                continue
            src = self.skeleton.index[name]
            if bone.parent >= 0 and target.bones[bone.parent].name.startswith(GRAFT):
                self._nested.append((index, src))
            elif bone.parent in carried_fingers:
                chain, hand = [], bone.parent
                while hand in carried_fingers or hand not in source_of:
                    chain.append(hand)
                    hand = target.bones[hand].parent
                fit = self._target_bind[index] @ np.linalg.inv(self._source_bind[src])
                main = source_of[hand]
                self._fingered.append((index, src, main,
                                       self._target_unbind[hand] @ fit @ self._source_bind[main], chain[::-1]))
            elif bone.parent in source_of:
                # where the fit put the joint, over where its source stands: the fit's map for it
                fit = self._target_bind[index] @ np.linalg.inv(self._source_bind[src])
                main = source_of[bone.parent]
                self._hung.append((index, src, main,
                                   self._target_unbind[bone.parent] @ fit @ self._source_bind[main]))

    def bind_worn(self, lent, worn: list) -> dict:
        """Have the character's own helper bones go with the lent skin they lie over; ``{bone: joint}``.

        What is worn is weighted to the character's own rig, which has bones
        the Avatar does not name: limb twists, joint correctives, a breast
        joint.  Its game drives them; nothing here does, so under a clip the
        lent skin moves on its kept joints and the cloth over it stays put —
        and the skin comes through.  Such a bone *B* follows a kept joint *J*
        rigidly when both hang under the same carried bone and *J* is what
        most of the lent skin nearest *B*'s vertices is weighted to
        (:data:`FOLLOW_SHARE`): the cloth then moves as the skin under it does.
        Only a bone with no bones of the character's under it: one that heads
        a chain is a cape, a ribbon or a lock of hair, and following the skin
        at its root while its neighbours do not tears the cloth between them
        (10 edges of EndminM's coat stretched fourfold in the first version).
        *lent* is the lent body's whole mesh, *worn* the meshes of everything else.
        """
        bones = self.target.bones
        carried = {tgt for _src, tgt in self.retarget.pairs.values()}
        driven = {tgt for tgt, *_rest in self._hung}
        pairs = worn_followers(bones, carried, driven, lent, worn)
        self._followers = [(bone, joint, self._target_unbind[joint] @ self._target_bind[bone]) for bone, joint in pairs]
        return {bones[bone].name: bones[joint].name for bone, joint in pairs}

    # -- the clip's pose on the character ---------------------------------------
    def carry(self, pose: Pose, source: np.ndarray | None = None) -> Pose:
        """*pose* (stand-in bone names) as a pose for the character's skeleton."""
        last = self._last
        state = (self.drive_kept, self.reach_limbs, self.anchor, None if self.feet is None else frozenset(self.feet[1]),
                 None if self.other is None else self.other.tobytes())
        if source is None and last is not None and last[0] is pose and last[4] == state:
            return dict(last[2])                       # the same frame's pose again (the aim, then the draw)
        if source is None:
            source = self.skeleton.evaluate(without_shape_channels(pose))
        out = self.retarget.carry(source)
        world = self.hold_anchor(out, source)
        if self.reach_limbs:
            self.reach(out, source, world)
            if self.stand(out, source):
                self.reach(out, source)
        if self.drive_kept:
            out.update(self.kept(source, out))
        if self.reach_limbs:
            # measured with the kept joints posed: before them a kneeling body's kneecap stood 67 mm
            # in the floor and lifted her 62 mm.  The whole body moves, hands and feet with it: placed
            # again, the feet pinned a kneeling body's shins and the knees bent back up (§7.1089)
            self.keep_above(out, source)
        self._last = (pose, source, out, None, (self.drive_kept, self.reach_limbs, self.anchor, None if self.feet is None else frozenset(self.feet[1]),
                 None if self.other is None else self.other.tobytes()))
        return dict(out)

    def hold_anchor(self, out: Pose, source: np.ndarray) -> np.ndarray | None:
        """Move the whole body so the point the row is about stands where the clip has it.

        A carried body is placed by its pelvis.  That is the right place for
        a row that joins the two bodies there, and for nothing else: a trunk
        that is 1.02 to 1.16 of the stand-in's puts the head 6 to 9 cm from
        where the clip has it, and in a row whose aim is at her mouth the
        other body is then that far from it.  :attr:`anchor` is the stand-in
        bone the row's aim points at (a null on the head, on the chest);
        where that point is on the character (:meth:`point`) is put where the
        clip has it, for a body of the character's size, by moving the pelvis
        — and so everything — by the difference.  The hands and feet are
        brought back to their own places after (:meth:`reach`).  Returns the
        character's world matrices for *out* as it is left, or None when the
        row is about the pelvis or about nothing.
        """
        index = self.skeleton.index.get(self.anchor, -1) if self.anchor else -1
        above = index
        while above >= 0 and above not in self._carried:
            above = self.skeleton.bones[above].parent
        hips_source, hips_target = self._pelvis
        if above < 0 or above == hips_source:
            return None
        world = self.target.evaluate(out)
        at = self.point(self.anchor, (source, world))
        move = (self._size * source[index][:3, 3] + self._stand_off) - at
        entry = out[self.target.bones[hips_target].name]
        entry["pos"] = np.asarray(entry["pos"], dtype=np.float64) + self.retarget._into_hips_parent @ move
        world[:, :3, 3] += move            # every bone the limbs are solved on hangs from the pelvis
        return world

    def _goal(self, where: np.ndarray) -> np.ndarray:
        """Where a hand or foot the clip has at *where* (the stand-in's frame) goes for this character.

        On the floor, it is the stand-in's place scaled about the frame's
        origin as the pelvis is: the floor stays where it is and a body of the
        character's size reaches it.  Off the floor it is held by what does
        not scale — the other body, a wall, a fence — and stays where the clip
        has it: scaled, a smaller character's hands and her legs round the
        other body were pulled into it (EndminM's trunk crossed Typhoea's skin
        564 times in 駅弁, 70 for a card woman; 179 held, §7.1069).  Between the
        two, over one more ankle's height, the one gives way to the other, so
        a foot that lifts does not jump.  "On the floor" is at most the
        stand-in's ankle standing at rest.

        A hand held by her own body is put on it instead (:meth:`_end_goal`).
        """
        scaled = self._size * where + self._stand_off
        if self._ankle is None:
            return scaled
        off = float(np.clip(where[1] / self._ankle - 1.0, 0.0, 1.0))
        return scaled * (1.0 - off) + where * off

    def _heel_rise(self, human: str, where: np.ndarray) -> np.ndarray:
        """How much higher a foot's ankle goes on the floor than the stand-in's, scaled: her rig's own ankle rise.

        The character's rig stands its ankle where its shoe's heel has it:
        Si's 156 mm over her sole at rest, the stand-in's 79 mm scaled.  Put
        where the stand-in's goes, her sole went 50-70 mm into the floor
        (§7.1094).  On the floor the goal rises by the difference at rest,
        giving way off the floor as :meth:`_goal` does; a bare lent foot
        laid on the source body's foot has its own rule (``Barefoot.ankle_for``).
        """
        if self._ankle is None:
            return np.zeros(3)
        ankle = float(self._target_bind[self.retarget.pairs[human][1]][1, 3])
        rise = ankle - self._size * self._ankle
        on_floor = 1.0 - float(np.clip(where[1] / self._ankle - 1.0, 0.0, 1.0))
        return np.array([0.0, rise * on_floor, 0.0])

    def _end_goal(self, human: str, source: np.ndarray, world: np.ndarray | None = None) -> np.ndarray:
        """Where the end bone *human* (a hand or a foot) goes: :meth:`_goal`; for a bare foot where its
        ankle has to be for the foot to lie on the source body's foot there (``Barefoot.ankle_for``); for
        a hand on her own body, the place on her drawn body it holds (:meth:`on_skin`, *world* her pose)."""
        where = source[self.retarget.pairs[human][0]][:3, 3]
        goal = self._goal(where)
        if human.endswith("Hand"):
            if world is not None and self.lent is not None and self._holds_herself(where, source):
                held = self.on_skin(where, (source, world), self.lent)
                return goal if held is None else held
            return goal
        if not human.endswith("Foot"):
            return goal
        side = human[:-len("Foot")]
        if self.feet is None or self._source_body is None or side not in self.feet[1]:
            return goal + self._heel_rise(human, where)
        barefoot, _bare = self.feet
        from ..mesh import skinning
        rest, index, weights, unbind = self._source_skin()
        onto = skinning.skinned(rest, index, weights, source @ unbind) + (goal - where)
        found = barefoot.ankle_for(side, onto)
        return goal if found is None else found

    def _holds_herself(self, hand: np.ndarray, source: np.ndarray) -> bool:
        """Whether the clip has her *hand* on her own body: nearer her skin than any of his joints.

        A hand that holds her own knee (開脚正常位) goes to that knee as
        her body is drawn (:meth:`_end_goal`); one on him stays where the
        clip has it (:meth:`_goal`).  Only hands: telling a foot's holder by
        the joints near it also took a foot behind his back for hers, beside
        her own hips, and 駅弁 crossed him more again (§7.1069).  Her side is
        her skin (the source body on the stand-in), his his joints.  Scaled
        about the origin instead of put on her body, a hand lying beside her
        head pulled her upper arm 3 cm under the floor (§7.1083).
        """
        if self.other is None or not len(self.other) or self._source_body is None:
            return False
        from ..mesh import skinning
        rest, index, weights, unbind = self._source_skin()
        skin = skinning.skinned(rest, index, weights, source @ unbind)
        return float(np.linalg.norm(skin - hand, axis=1).min()) < float(np.linalg.norm(self.other - hand, axis=1).min())

    def keep_above(self, out: Pose, source: np.ndarray) -> float:
        """Move the body up or down to rest as low as the source body does in the clip; how far (signed).

        Her drawn body is the lent body and her own skin (:attr:`native`), or
        a reconstruction's own surfaces alone (Si: her back 20-60 mm in the
        floor lying, §7.1094).  Lying, her back is where her proportions put it under a pelvis placed
        by her size: Wulfa's chest was 35 mm under the floor in 正常位, a card
        woman 7 (§7.1083).  The source body on the stand-in, scaled as the
        floor is (:meth:`_goal`), is as low as the clip lets a body go: a body
        lower is lifted to it.  A body higher is let down to it only when the
        clip rests the source body on the floor (its lowest point no higher
        than the stand-in's ankle at rest, as :meth:`_goal` tells a hand or
        foot on the floor): kneeling, her knees stood 19-28 mm off the floor
        (a card woman −9 to 2, §7.1089).  Off the floor, nothing holds it.  How low
        the body is drawn counts her own skin too (:attr:`native`): measured on the
        lent body alone, it was let down until her own knees were 10-16 mm in the floor.
        """
        if (self.lent is None and not self.native) or self._source_body is None or HIPS not in self.retarget.pairs:
            return 0.0
        from ..mesh import skinning
        rest, index, weights, unbind = self._source_skin()
        lent = self.lent
        body = self._source_above_feet(index, weights)
        if not len(body):
            return 0.0
        posed = skinning.skinned(rest[body], index[body], weights[body], source @ unbind)
        lowest = float((self._size * posed[:, 1]).min() + self._stand_off[1])
        world = self.target.evaluate(out)
        # as it is drawn: from where its bones bind it, by the binds the skin was given (``body.unblend``)
        skin = self.target.skinning_matrices(world=world)
        # each surface skinned the way it is drawn: Si's body is dual quaternion, 5 mm lower at the
        # shoulder lying than a linear blend says
        modes = getattr(self, "native_modes", None) or {}
        rests = self.target.rest_frames()

        def low(mesh, rows):
            mode, share = modes.get(id(mesh), (skinning.LINEAR, None))
            share = None if share is None or not len(share) else share[rows]
            return float(skinning.skinned(mesh.skin_rest()[0][rows].astype(np.float64), mesh.bone_idx[rows],
                                          mesh.bone_wt[rows], skin, mode, rests, share)[:, 1].min())
        drawn = min(low(mesh, rows)
                    for mesh, rows in ([(lent, self._above_feet(lent, len(rest)))] if lent is not None else [])
                    + [(own, self._above_feet(own)) for own in (self.native or ())]
                    if len(rows))
        lift = lowest - drawn
        resting = self._ankle is not None and lowest - self._stand_off[1] <= self._size * self._ankle
        if lift < 0.0 and resting:
            # no lower than takes her ankles below where they go: kneeling on all fours with the feet
            # on the floor and the knees still high, it put the feet 28-32 mm in it
            room = self._feet_room(out, source)
            lift = max(lift, -max(room, 0.0))
        if abs(lift) <= 1e-4 or (lift < 0.0 and not resting):
            return 0.0
        entry = out[self.target.bones[self.retarget.pairs[HIPS][1]].name]
        entry["pos"] = (np.asarray(entry.get("pos", np.zeros(3)), dtype=np.float64)
                        + self.retarget._into_hips_parent @ np.array([0.0, lift, 0.0]))
        return lift

    def _source_above_feet(self, index: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """The source body's vertex rows not hanging mostly from the stand-in's feet (as :meth:`_above_feet`)."""
        if getattr(self, "_source_rows", None) is None:
            feet = set()
            for side in ("LeftFoot", "RightFoot"):
                if side in self.retarget.pairs:
                    feet |= set(self.skeleton.descendants(self.skeleton.bones[self.retarget.pairs[side][0]].name))
            heaviest = index[np.arange(len(index)), np.argmax(weights, axis=1)]
            self._source_rows = np.flatnonzero(~np.isin(heaviest, list(feet)))
        return self._source_rows

    def _feet_room(self, out: Pose, source: np.ndarray) -> float:
        """How far the ankles stand above where they go (:meth:`_end_goal`): how far the body can come down.

        The ankle's goal is where it stands for the foot to lie on the source
        body's foot; a foot is turned onto it only after (``Barefoot.hold_as``),
        so the foot's own vertices are no measure here.
        """
        world = self.target.evaluate(out)
        room = float("inf")
        for side in ("LeftFoot", "RightFoot"):
            if side in self.retarget.pairs:
                ankle = world[self.retarget.pairs[side][1]][1, 3]
                room = min(room, float(ankle - self._end_goal(side, source)[1]))
        return room

    def _above_feet(self, lent, count: int | None = None) -> np.ndarray:
        """A mesh's vertex rows (of the first *count*: the source body's) not hanging from the feet.

        The feet are put on the source body's later (``barefoot.Barefoot.hold_as``); until then a foot of
        a heeled rig points its toes down, and they are no measure of how low the body is.
        """
        if count is None:
            count = lent.vertex_count
        kept = self._body_rows.get(id(lent))
        if kept is None or kept[0] is not lent or kept[2] != count:
            feet = set()
            for side in ("LeftFoot", "RightFoot"):
                if side in self.retarget.pairs:
                    feet |= set(self.target.descendants(self.target.bones[self.retarget.pairs[side][1]].name))
            heaviest = lent.bone_idx[np.arange(lent.vertex_count), np.argmax(lent.bone_wt, axis=1)]
            rows = np.flatnonzero(~np.isin(heaviest, list(feet)))
            kept = self._body_rows[id(lent)] = (lent, rows[rows < count], count)
        return kept[1]

    def stand(self, out: Pose, source: np.ndarray) -> float:
        """Lower the body until a foot the clip has on the floor reaches it (*out* is changed); how far.

        The pelvis is placed by the character's size (the ratio of pelvis
        heights) but her legs are not that size: EndminF's are 0.884 of the
        stand-in's against 0.943, and the foot she stands on in 立位 hung 9 cm
        over the floor, the leg straightened towards it (§7.1076).  A foot on
        the floor (as :meth:`_goal` reads it) that :meth:`reach` left short
        takes the body down by the shortfall; a body none of whose feet is on
        the floor is held where it is.
        """
        pairs = self.retarget.pairs
        if self._ankle is None or HIPS not in pairs:
            return 0.0
        world = self.target.evaluate(out)
        short = 0.0
        for side in ("LeftFoot", "RightFoot"):
            if side not in pairs:
                continue
            source_end, end = pairs[side]
            where = source[source_end][:3, 3]
            if where[1] > self._ankle:
                continue
            short = max(short, float(world[end][1, 3] - self._end_goal(side, source)[1]))
        if short <= 1e-4:
            return 0.0
        entry = out[self.target.bones[pairs[HIPS][1]].name]
        entry["pos"] = (np.asarray(entry.get("pos", np.zeros(3)), dtype=np.float64)
                        + self.retarget._into_hips_parent @ np.array([0.0, -short, 0.0]))
        return short

    def reach(self, out: Pose, source: np.ndarray, world: np.ndarray | None = None) -> float:
        """Bring the character's hands and feet to where the clip has them (*out* is changed in place).

        The pose is carried bone by bone as turns, and the pelvis as a place.
        A character's limbs are not its stand-in's in proportion (EndminF's
        upper arm is 0.81 of it, her thigh 0.85, her trunk 1.02), so with
        every turn right her hands and feet end centimetres from where the
        clip puts them — off the floor, off the other body.  Where the clip
        puts them is :meth:`_goal`.  Each limb is bent at its middle joint to the length that reaches
        that place and swung onto it from its root; one too short to reach
        is straightened towards it.  The hand or foot keeps the turn the
        clip gave it.  Returns the farthest any end still is from its place.
        *world* is the character's world matrices for *out* when the caller
        has them.
        """
        pairs = self.retarget.pairs
        bones = self.target.bones
        rest = self.retarget._rest_local
        farthest = 0.0
        # the legs first: a hand that holds her own knee goes where the leg put it
        legs_placed = False
        for limb in LIMBS[2:] + LIMBS[:2]:
            if not all(human in pairs for human in limb):
                continue
            (_su, up), (_sm, mid), (source_end, end) = (pairs[human] for human in limb)
            if bones[mid].parent != up or bones[end].parent != mid:
                continue                              # a joint between them that the pose does not name
            if world is None or (limb is LIMBS[0] and legs_placed):
                world = self.target.evaluate(out)         # after the legs: what a hand on her own knee holds
            legs_placed = limb[2].endswith("Foot")
            a, b, c = world[up][:3, 3], world[mid][:3, 3], world[end][:3, 3]
            goal = self._end_goal(limb[2], source, world if limb[2].endswith("Hand") else None)
            first, second = float(np.linalg.norm(b - a)), float(np.linalg.norm(c - b))
            axis = np.cross(b - a, c - b)
            if first < 1e-6 or second < 1e-6 or np.linalg.norm(axis) < 1e-9 or np.linalg.norm(goal - a) < 1e-6:
                continue                              # a limb dead straight has no way to bend that is its own
            axis /= np.linalg.norm(axis)
            wanted = float(np.clip(np.linalg.norm(goal - a), abs(first - second) + 1e-4, first + second - 1e-4))

            def inner(length: float, first: float = first, second: float = second) -> float:
                return float(np.arccos(np.clip((first * first + second * second - length * length)
                                               / (2.0 * first * second), -1.0, 1.0)))
            # the angle inside the joint grows as the limb straightens: turn the lower bone back by the change
            bend = m3.quat_from_axis_angle(axis, inner(float(np.linalg.norm(c - a))) - inner(wanted))
            moved = b + m3.quat_rotate(bend, c - b)
            swing = m3.quat_from_to((moved - a) / np.linalg.norm(moved - a), (goal - a) / np.linalg.norm(goal - a))
            turns = m3.quats_from_matrices(world[[bones[up].parent, up, mid, end]])[0]
            above, upper, lower, tip = turns
            upper = m3.quat_normalize(m3.quat_mul(swing, upper))
            lower = m3.quat_normalize(m3.quat_mul(swing, m3.quat_mul(bend, lower)))
            for index, parent, turn in ((up, above, upper), (mid, upper, lower), (end, lower, tip)):
                local = m3.quat_mul(m3.quat_conj(parent), turn)
                entry = dict(out.get(bones[index].name) or {})
                entry["rot"] = m3.quat_normalize(m3.quat_mul(m3.quat_conj(rest[index]), local))
                out[bones[index].name] = entry
            farthest = max(farthest, abs(float(np.linalg.norm(goal - a)) - wanted))
        return farthest

    def kept(self, source: np.ndarray, carried: Pose | None = None) -> Pose:
        """The kept source joints' entries for the stand-in's world matrices *source*.

        *carried* is the pose the character's own bones have for it (its
        finger bones' turns are read from it; at rest when left out).
        """
        bones, parents = self.target.bones, self.skeleton.bones
        undone: dict[int, np.ndarray] = {}
        local: dict[int, np.ndarray] = {}
        targets: list[int] = []
        matrices: list[np.ndarray] = []
        for tgt, src, main, before, chain in self._fingered:
            if main not in undone:
                undone[main] = np.linalg.inv(source[main])
            under = np.eye(4)
            for index in chain:
                if index not in local:
                    bone = bones[index]
                    turn = ((carried or {}).get(bone.name) or {}).get("rot")
                    rot = bone.local_rot if turn is None else m3.quat_mul(bone.local_rot, turn)
                    local[index] = m3.compose(bone.local_pos, rot, getattr(bone, "local_scale", (1.0, 1.0, 1.0)))
                under = under @ local[index]
            targets.append(tgt)
            matrices.append(np.linalg.inv(under) @ before @ undone[main] @ source[src])
        if self._hung:
            # the joint as its source moves it against the bone it hangs under, on the character's
            # bone: every one of them in one go (a body has a hundred, every frame)
            if self._hung_table is None:
                self._hung_table = (np.array([row[0] for row in self._hung]), np.array([row[1] for row in self._hung]),
                                    np.array([row[2] for row in self._hung]), np.stack([row[3] for row in self._hung]))
            hung, src, main, before = self._hung_table
            moved = before @ np.linalg.inv(source[main]) @ source[src]
            targets.extend(hung.tolist())
            matrices.extend(moved)
            at = {int(tgt): k for k, tgt in enumerate(hung)}
            for bone, joint, offset in self._followers:
                targets.append(bone)
                matrices.append(moved[at[joint]] @ offset)
        for tgt, src in self._nested:
            targets.append(tgt)
            matrices.append(np.linalg.inv(source[parents[src].parent]) @ source[src])
        if not targets:
            return {}
        stacked = np.stack(matrices)
        for k, tgt in enumerate(targets):
            offset = self.joint_offsets.get(tgt)
            if offset is not None:
                # a body edit moves the joint in its own frame, the way it moved its rest
                stacked[k] = stacked[k] @ offset
        return _additive_many([bones[tgt] for tgt in targets], stacked)

    # -- a pose that is not a clip ------------------------------------------------
    def corrected(self, world: np.ndarray) -> Pose | None:
        """One of the project's own poses on the stand-in, its helper joints set by the joint correction.

        *world* is the world matrices of the scaffold the pose library is
        written on, under the pose.  They are carried onto the stand-in (the same world-turn rule), and the source
        game's joint correction — what its studio runs for a pose that is not
        a clip — sets the helper joints from where the main joints then are
        (``origin.joint_correct``).  None when that was never converted.
        """
        if self._corrector is None:
            from ..origin import joint_correct
            self._corrector = joint_correct.corrector(self.skeleton) or False
            if self._corrector:
                self._from_scaffold = Retarget(self.skeleton, {human: name for human, name in SOURCE_BONES.items()
                                                               if self.skeleton.has(name)})
        if not self._corrector:
            return None
        return self._corrector.apply(self._from_scaffold.carry(world))

    def mouth_push(self, pose: Pose, carried: Pose) -> Pose:
        """Her two lip joints moved as far as the stand-in's mouth was pushed (``bp_dan``'s ``cf_J_MouthMove``).

        The stand-in's mouth stands for the middle of her lips (:meth:`point`),
        scaled to her size; *pose* is the stand-in's, *carried* hers as it will
        be evaluated (her face's own entries in it).  Each lip keeps its turn
        and moves by the mouth's displacement, written in the channel its entry
        already uses.  Empty when nothing pushed the mouth or she has no lips.
        """
        entry = pose.get(SOURCE_MOUTH) or {}
        if self._mouth is None or entry.get("pos") is None:
            return {}
        _head, mouth, lips = self._mouth
        source, _target = self.worlds(pose)
        unpushed = {**pose, SOURCE_MOUTH: {k: v for k, v in entry.items() if k != "pos"}}
        rest = self.skeleton.evaluate(unpushed, only=[mouth])[mouth]
        moved = self._size * (source[mouth][:3, 3] - rest[:3, 3])
        if not np.any(moved):
            return {}
        parents = [self.target.bones[lip].parent for lip in lips]
        world = self.target.evaluate(carried, only=list(lips))
        out: Pose = {}
        for lip, parent in zip(lips, parents):
            bone = self.target.bones[lip]
            local = np.linalg.inv(world[parent]) @ world[lip] if parent >= 0 else world[lip].copy()
            local[:3, 3] = (np.linalg.inv(world[parent]) @ np.append(world[lip][:3, 3] + moved, 1.0))[:3]                 if parent >= 0 else world[lip][:3, 3] + moved
            had = dict(carried.get(bone.name) or {})
            if "matrix" in had:
                had["matrix"] = local
            else:
                rest_at = bone.authored_local[:3, 3] if bone.authored_local is not None else bone.local_pos
                had["pos"] = local[:3, 3] - np.asarray(rest_at, dtype=np.float64)
            out[bone.name] = had
        return out

    def worlds(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        """The stand-in's and the character's world matrices for *pose*."""
        self.carry(pose)
        _pose, source, carried, target, kept = self._last
        if target is None:
            target = self.target.evaluate(carried)
            self._last = (pose, source, carried, target, kept)
        return source, target

    def source_edges(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        """``((E, 2) vertex pairs, (E,) length posed / length at rest)`` of the source body under *pose*.

        The body the lent skin came from, on its own rig, each bone moved as
        the pose moves it from rest: what the clip really does to it, vertex
        for vertex the lent skin's (``tools/ef_body_deform.py`` measures the
        lent skin against it).  At rest it is 1.0 everywhere.
        """
        from ..mesh import skinning
        from ..mesh.skin import split_archive_skin
        if self._source_body is None:
            raise ActorError("native opaque reconstruction has no lent source body to measure")
        if self._truth is None:
            piece = self._source_body
            count = len(piece.binds)                      # a piece may list a bone past its binds
            weights, index = split_archive_skin(np.asarray(piece.skin, dtype=np.float64), bone_count=count)
            mine = np.stack(piece.bone_world[:count]) @ np.asarray(piece.binds, dtype=np.float64)
            local = np.c_[piece.mesh.positions.astype(np.float64), np.ones(piece.mesh.vertex_count)]
            rest = np.einsum("vk,vkij,vj->vi", weights, mine[index], local)[:, :3]
            bones = np.array([self.skeleton.index[name] for name in piece.bones[:count]], dtype=np.int64)
            tri = piece.mesh.indices.reshape(-1, 3).astype(np.int64)
            pairs = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
            # each bone by how far the pose takes it from the stand-in's own rest.  Not from the
            # mesh's bind: 21 breast bones rest elsewhere on the skeleton than the mesh binds them
            # (their place is the shape pass's, which a stand-in does not run), and skinned by the
            # mesh's bind the body is off its own rest shape before any pose — p99 0.36 of an edge
            # on EndminF's source, all of it at the breasts (§7.944)
            unbind = np.tile(np.eye(4), (len(self.skeleton.bones), 1, 1))
            unbind[bones] = np.linalg.inv(self._source_bind[bones])
            self._truth = (rest, bones[index], weights, unbind, pairs,
                           np.linalg.norm(rest[pairs[:, 0]] - rest[pairs[:, 1]], axis=1))
        rest, index, weights, unbind, pairs, at_rest = self._truth
        world = self.skeleton.evaluate(without_shape_channels(pose))
        posed = skinning.skinned(rest, index, weights, world @ unbind)
        length = np.linalg.norm(posed[pairs[:, 0]] - posed[pairs[:, 1]], axis=1)
        return pairs, length / np.maximum(at_rest, 1e-12)

    def _source_skin(self) -> tuple:
        """The source body's ``(rest positions, bone rows, weights, unbind)`` on the stand-in (cached)."""
        if self._truth is None:
            self.source_edges({})
        return self._truth[:4]

    def source_points(self, pose: Pose) -> np.ndarray | None:
        """The source body's vertices on the stand-in as *pose* puts it (stand-in frame); None without one."""
        if self._source_body is None:
            return None
        from ..mesh import skinning
        rest, index, weights, unbind = self._source_skin()
        return skinning.skinned(rest, index, weights, self.worlds(pose)[0] @ unbind)

    def _posed_source(self, source: np.ndarray) -> np.ndarray:
        """The source body's vertices on the stand-in as *source* poses it (once per pose)."""
        from ..mesh import skinning

        def make():
            rest, index, weights, unbind = self._source_skin()
            return skinning.skinned(rest, index, weights, source @ unbind)
        return self._frame_cached("source", source, make)

    def _carried_below(self, index: int) -> int | None:
        """The carried stand-in bone fewest levels under *index*, or None with none."""
        best, depth = None, None
        for carried in self._carried:
            steps, at = 0, carried
            while at >= 0 and at != index:
                at, steps = self.skeleton.bones[at].parent, steps + 1
            if at == index and (depth is None or steps < depth):
                best, depth = carried, steps
        return best

    def point(self, bone: str, worlds: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
        """Where the stand-in's *bone* is on the character, in the character's model space.

        A fitted joint is read off the character.  Any other bone rides the
        nearest carried bone above it, displaced by what the fit displaced
        that bone's skin by — or, with none above it, the nearest one below
        it (:meth:`_carried_below`); ``None`` when there is neither.
        """
        source, target = worlds
        if self.target.has(GRAFT + bone):
            return target[self.target.index[GRAFT + bone]][:3, 3].copy()
        index = self.skeleton.index.get(bone, -1)
        if index < 0:
            return None
        above = index
        while above >= 0 and above not in self._carried:
            above = self.skeleton.bones[above].parent
        if above < 0:
            # nothing carried above it (a bone between the root and the pelvis, and what hangs off it):
            # it is placed against the nearest carried bone below it, by the same rest relation
            above = self._carried_below(index)
            if above is None:
                return None
        tgt, moved = self._carried[above]
        if self._mouth is not None and above == self._mouth[0]:
            # a point on the head is where it is against the mouth: the two heads are not the same
            # head, and the joint each turns on is not at the same place in it
            _head, mouth, lips = self._mouth
            middle = (target[lips[0]][:3, 3] + target[lips[1]][:3, 3]) / 2.0
            return middle + self._size * (source[index][:3, 3] - source[mouth][:3, 3])
        at_bind = (self._source_bind[above] @ np.linalg.inv(source[above]) @ source[index])[:3, 3] + moved
        return (target[tgt] @ self._target_unbind[tgt] @ np.append(at_bind, 1.0))[:3]
