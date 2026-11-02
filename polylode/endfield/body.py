"""Locally baked, complete body fitted to an Endfield character's rest rig.

The source mesh and skin sheet come from the user's converted Koikatsu assets.
The fitted item lives in the user's ``Endfield/body`` store, never in a build.
The character's face and hair remain their original Endfield pieces.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from ..chara.cardimport import read_origin_card
from ..mesh.mesh import Material, Part
from ..mesh.skin import split_archive_skin
from ..origin import bake
from ..origin import colorspace as CS
from ..origin import textures as TEX
from ..origin.figure.body_head import GENITAL_MESHES, body_texture
from ..origin.prefab import Piece
from . import store
from .actor import ActorError, _rest_mesh, _texture_key, build_skeleton, load_actor
from .body_scope import FEMALE_HUMAN_TYPES, require_body_scope
from .part_roles import (
    declared_role,
    declared_tails,
    face_reference,
    facial_evidence,
    has_sdf_face,
)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest rotation taking one limb axis to the other."""
    a, b = _unit(a), _unit(b)
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3)
    if dot < -1.0 + 1e-9:
        axis = _unit(np.cross(a, [0, 0, 1] if abs(a[2]) < .9 else [0, 1, 0]))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -cross[2], cross[1]],
                  [cross[2], 0, -cross[0]],
                  [-cross[1], cross[0], 0]], dtype=np.float64)
    return np.eye(3) + k + k @ k * ((1.0 - dot) / np.dot(cross, cross))


#: what names the lent anatomy's joints (the fit merges them into the hips)
ANATOMY = ("kokan", "ana", "Vagina")


def _bone_target(name: str, humanoid: dict, available: set[str]) -> str:
    side = "L" if name.endswith("_L") else "R" if name.endswith("_R") else ""
    stem = name[:-2] if side else name
    if side:
        role = "Left" if side == "L" else "Right"
        prefix = f"Bip001_{side}_"
        for token, target in (("thumb", "Finger0"), ("index", "Finger1"),
                              ("middle", "Finger2"), ("ring", "Finger3"),
                              ("little", "Finger4")):
            if token in stem:
                digit = stem[-2:]
                suffix = "" if digit == "01" else "1" if digit == "02" else "2"
                finger = prefix + target + suffix
                return finger if finger in available else humanoid[role + "Hand"]
        if "toes" in stem:
            return humanoid[role + "Toes"]
        if "foot" in stem:
            return humanoid[role + "Foot"]
        if any(t in stem for t in ("leg0", "knee")):
            return humanoid[role + "LowerLeg"]
        if any(t in stem for t in ("thigh", "s_leg")):
            return humanoid[role + "UpperLeg"]
        if any(t in stem for t in ("forearm", "elbo")):
            return humanoid[role + "LowerArm"]
        if any(t in stem for t in ("wrist", "hand")):
            return humanoid[role + "Hand"]
        if "arm" in stem:
            return humanoid[role + "UpperArm"]
        if "shoulder" in stem:
            return humanoid[role + "Shoulder"]
        if "siri" in stem:
            return humanoid["Hips"]
        if "bust" in stem or "bnip" in stem:
            return humanoid["UpperChest"]
    if "head" in name:
        return humanoid["Head"]
    if "neck" in name:
        return humanoid["Neck"]
    if "spine03" in name:
        return humanoid["UpperChest"]
    if "spine02" in name:
        return humanoid["UpperChest"]
    if "bust" in name or "bnip" in name:
        return humanoid["UpperChest"]
    if "spine01" in name or "waist01" in name:
        return humanoid["Chest"]
    if "waist02" in name or any(t in name for t in ANATOMY):
        return humanoid["Hips"]
    raise ValueError(f"no Endfield bone mapping for {name}")


#: The lent body's finger tokens, in the order the reference rig's finger joints name them.
FINGER_TOKENS = ("thumb", "index", "middle", "ring", "little")


class DeclaredMapping(dict):
    """A complete bone mapping that knows which of its targets are the character's finger joints.

    Endfield's own fit tells a finger joint by its rig's name ("Finger"); a
    declared table's joints are named as their source named them.
    """

    def __init__(self, mapping: dict, fingers):
        super().__init__(mapping)
        self.fingers = frozenset(fingers)


def declared_bone_mapping(piece: Piece, target, humanoid: dict, fingers: dict | None) -> dict | None:
    """The fit's complete bone mapping when the character's table pairs its fingers; ``None`` otherwise.

    *fingers* is ``{reference finger joint (cf_j_<finger>0<k>_<side>): the character's node}`` — the
    pairs a converted source measured (``converted.blender_source.pair_fingers``).  A lent finger
    bone goes with the character's joint its reference joint is paired with; every other bone with
    :func:`_bone_target`'s.  Endfield's tables name no fingers: their bones keep :func:`_bone_target`.
    """
    if not fingers:
        return None
    available = set(target.index)
    mapping = {}
    for name in piece.bones:
        side = name[-1] if name.endswith(("_L", "_R")) else ""
        token = next((t for t in FINGER_TOKENS if t in name), None)
        joint = f"cf_j_{token}{name[:-2][-2:]}_{side}" if token and side else None
        if joint is not None and joint not in fingers:
            raise ValueError(f"the character's finger table has no pair for {joint} ({name})")
        mapping[name] = fingers[joint] if joint is not None else _bone_target(name, humanoid, available)
    return DeclaredMapping(mapping, fingers.values())


def _segment_anchor(point, first, last, destination_first, destination_last, *, stretch_axis=False):
    """Fit an authored pivot, optionally stretching the entire segment axis.

    Axis stretching retains transverse dimensions and makes the transform
    independent of which helper along this segment carries a vertex's weight.
    """
    axis = last - first
    square = float(axis @ axis)
    if square <= 1e-20:
        raise ValueError('body fitting requires a nonzero source limb segment')
    fraction = float((point - first) @ axis) / square
    radial = point - first - fraction * axis
    rotation = _align(axis, destination_last - destination_first)
    anchor = destination_first + fraction * (destination_last - destination_first) + rotation @ radial
    if stretch_axis:
        ratio = float(np.linalg.norm(destination_last - destination_first)) / np.sqrt(square)
        if ratio <= 0:
            raise ValueError('body fitting requires a nonzero destination limb segment')
        rotation = rotation @ (np.eye(3) + (ratio - 1) * np.outer(axis, axis) / square)
    return rotation, anchor


def _span(a: np.ndarray, b: np.ndarray, what: str) -> float:
    length = float(np.linalg.norm(b - a))
    if length <= 1e-9:
        raise ValueError(f"body fitting requires a nonzero {what}")
    return length


#: How a limb segment is sized across its axis: ``"girth"``, by the character's own skin round it
#: (:meth:`FitScales.girth`); ``"length"``, uniformly by the segment's own length ratio, as every
#: body was fitted before PROGRESS #173.  A source whose skin cannot be told from its clothes yet
#: (the GF2 and Wuthering Waves converted builds, PROGRESS #177) asks for ``"length"`` by name.
#: ``"bones"``: as the source game itself carries a body of other lengths (KKABMX's length form): each
#: limb joint stands where the segment puts it and keeps its own skin at its own length, only sized
#: across by the girth — the segment's length is in how far apart its joints stand, not in the skin
#: stretched along it.  Stretched along a petite forearm (0.68 of the source's) the skin at the elbow
#: sat where no joint bends it, and folded flat in 2_2 (area 0.004–0.04 of its rest; the source game's
#: own body at her lengths keeps 0.12–0.16, §7.1299).
CROSS_SECTIONS = ("girth", "length", "bones")
#: The rule Endfield's completed bodies are fitted by (:func:`bake_body`); a body keeps the one it was
#: fitted by in its table (``fit_cross_section``), absent on the bodies fitted by ``"girth"`` before.
#: ``"bones"`` costs a short forearm some of its mid-forearm skin in a bent arm (Wulfa 2_2: area 0.50 to
#: 0.39 of its rest) — less than the source game's own body at her lengths loses there (0.30), and the
#: price of an elbow that no longer folds flat (0.012 to 0.166).
ENDFIELD_CROSS_SECTION = "bones"
#: Which of the lent hand's thumb joints have their skin carried by the hand's own fit instead of
#: translated onto the character's thumb joint (:func:`_limb_transform`).  Her thumb root stands 6–9 mm
#: nearer the arm's axis than the hand's fit carries the source's, and the lent thenar followed it in:
#: the hand came out 5–12% small against the forearm (Avywen +20/-20 mm round 1.27 against the source's
#: 1.45).  Carried, the skin keeps the source's proportion and is still bound to her thumb joint, which
#: it bends about — so it stretches more round the thumb's root (2_2 wrist p99 1.50 to 1.85), a trade the
#: owner took.  Recorded per body (``fit_thumb_carry``); a body with none recorded is ``"none"``.
THUMB_CARRIES = {"none": (), "thumb01": ("thumb01",), "thumb01+02": ("thumb01", "thumb02")}

#: The source point paired to UpperArm. The weighted arm01 frame is distal to the
#: animation pivot; shoulder02 coincides with arm00 in the source rig (§7.1348).
#: Unrecorded fits keep ``skin``; changing it requires a complete body rebuild.
ARM_ROOTS = {"skin": "cf_s_arm01", "joint": "cf_s_shoulder02"}
#: How the torso joints round an arm root carry the residual the field moves the shoulder by:
#: ``joints``, each its own share at its own place (the fit as it was); ``carried``, re-split by
#: :func:`carry_arm_residual` — Texas's bow, four squashed shoulder edges to two, but Lappland's
#: standing pose and wave gained some (§7.1445): a measured choice per character, recorded.
ARM_RESIDUALS = ("joints", "carried")


#: How far a rig's pelvis joint may stand from the lent body's, as a share of the neck's height, and its
#: trunk still be measured by the pelvis-to-neck span (``FitScales.torso_measure``).  The lent body's
#: ``cf_s_waist02`` stands at 0.692 of its ``cf_s_neck``'s height; the Endfield rigs' Hips at 0.660–0.699
#: (0.032 off at most, EndminM); Lappland's at 0.768 (0.076 off: her rig puts Hips at the waist, and the
#: span made her trunk 0.722 of the lent body's against a stature of 0.951 — a 382 mm waist where her own
#: skin is 557).  Half way between the two, 0.018 clear of the Endfield rigs and 0.026 of hers.
PELVIS_OFF = 0.05


class FitScales:
    """How big each part of the character is next to the lent body's: dimensionless, from the two skeletons.

    The lent body is the card's adult body; the character may be another
    size (a petite rig's torso is 0.83 of the source's, its forearm 0.68).
    Each ratio is a span of the character's own joints over the same span
    of the source body's: ``"torso"`` (pelvis to neck, or the neck's height
    over the floor where her rig puts its pelvis joint elsewhere,
    :meth:`torso_measure`), and per side
    (``"<part> L"``) the upper arm, forearm, hand (wrist to the middle
    finger's last joint, by the fit's own finger mapping), thigh and shin —
    each segment's length — and the palm (index to little finger's root, by
    the same mapping): the hand's breadth across its knuckles.
    A ratio is measured when a fitted bone first asks for it; a joint it
    needs that either skeleton lacks raises ``ValueError``.

    A segment's length is not its thickness (§7.1283): how thick each
    limb is, :meth:`girth`, is measured on the character's own skin
    (``limb_girth``) and handed in as *girth* ``{"arm": ratio | None, "leg":
    …}``.
    """

    #: the source joints and the character's joints (human-bone keys) of each span; ``None``: the
    #: finger joint the source's maps to (``"palm"``: both ends are fingers)
    SPANS: ClassVar[dict[str, tuple[str, str, str | None, str | None]]] = {
        "torso": ("cf_s_waist02", "cf_s_neck", "Hips", "Neck"),
        "upper arm": ("cf_s_arm01", "cf_s_forearm01", "UpperArm", "LowerArm"),
        "forearm": ("cf_s_forearm01", "cf_s_hand", "LowerArm", "Hand"),
        "hand": ("cf_s_hand", "cf_j_middle03", "Hand", None),
        "palm": ("cf_j_index01", "cf_j_little01", None, None),
        "thigh": ("cf_s_thigh01", "cf_s_leg01", "UpperLeg", "LowerLeg"),
        "shin": ("cf_s_leg01", "cf_j_foot", "LowerLeg", "Foot"),
    }

    def __init__(self, source: dict[str, np.ndarray], target, humanoid: dict,
                 *, bone_mapping: dict | None = None, girth: dict | None = None,
                 cross_section: str = "girth", thumb_carry: str = "none", arm_root: str = "skin"):
        if cross_section not in CROSS_SECTIONS:
            raise ValueError(f"unknown limb cross-section rule {cross_section!r}")
        if thumb_carry not in THUMB_CARRIES:
            raise ValueError(f"unknown thumb carry {thumb_carry!r}")
        if arm_root not in ARM_ROOTS:
            raise ValueError(f"unknown arm root {arm_root!r}")
        self.source, self.target, self.humanoid = source, target, humanoid
        self.bone_mapping = bone_mapping
        self.measured_girth = dict(girth or {})
        #: ``"girth"``: limbs sized across by :meth:`girth`; ``"length"``: each segment uniformly by its
        #: own length, the rule before PROGRESS #173 (:data:`CROSS_SECTIONS`)
        self.cross_section = cross_section
        #: the thumb joints whose skin the hand's own fit carries (:data:`THUMB_CARRIES`)
        self.thumb_carry = thumb_carry
        self.arm_root = arm_root
        self._known: dict[str, float] = {}

    def __getitem__(self, key: str) -> float:
        if key not in self._known:
            self._known[key] = self._measure(key)
        return self._known[key]

    def girth(self, limb: str) -> float:
        """How thick *limb* (``"arm"``/``"leg"``) is next to the lent body's: its cross-section's scale.

        The character's own, measured (``limb_girth``); a limb none of her
        skin goes round takes the torso's — the scale the trunk is carried
        at, so the lent body keeps its own proportion of limb to trunk.  Not
        the other limb's: EndminM's forearm skin is 1.19 of the lent body's,
        and his legs at that would stand out of his trousers (their
        tightest ring is 1.09).
        """
        value = self.measured_girth.get(limb)
        if value is None:
            return self["torso"]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"body fitting was given a {limb} girth of {value}")
        return float(value)

    def _point(self, name: str, human: str | None, role: str) -> np.ndarray:
        if human is not None:
            return self.target.world_position(self.humanoid[role + human])
        node = (self.bone_mapping[name] if self.bone_mapping is not None
                else _bone_target(name, self.humanoid, set(self.target.index)))
        return self.target.world_position(node)

    def torso_measure(self) -> str:
        """What the trunk is sized by: ``"span"``, the pelvis-to-neck span, or ``"neck height"``.

        The span, unless the character's pelvis joint stands more than
        :data:`PELVIS_OFF` of the neck's height away from where the lent body's
        does: the span then measures where her rig put the joint, not her
        trunk, and the neck's height over the floor (both rigs stand on it)
        says how big she is.
        """
        hips = self.target.world_position(self.humanoid["Hips"])[1]
        neck = self.target.world_position(self.humanoid["Neck"])[1]
        source = self.source["cf_s_waist02"][1] / self.source["cf_s_neck"][1]
        return "neck height" if abs(hips / neck - source) > PELVIS_OFF else "span"

    def _measure(self, key: str) -> float:
        if key == "torso" and self.torso_measure() == "neck height":
            return float(self.target.world_position(self.humanoid["Neck"])[1] / self.source["cf_s_neck"][1])
        part, side = (key[:-2], key[-1]) if key.endswith((" L", " R")) else (key, "")
        a, b, c, d = self.SPANS[part]
        if part == "upper arm":
            a = ARM_ROOTS[self.arm_root]
        role = {"L": "Left", "R": "Right", "": ""}[side]
        if side:
            a, b = f"{a}_{side}", f"{b}_{side}"
        try:
            first, last = self._point(a, c, role), self._point(b, d, role)
            source = _span(self.source[a], self.source[b], f"source {part}")
        except KeyError as missing:
            raise ValueError(f"body fitting needs {missing} to size the {key}") from None
        return _span(first, last, f"character {part}") / source


def _segment_scale(first: np.ndarray, last: np.ndarray, along: float, across: float) -> np.ndarray:
    """A limb segment's scale (3×3, source frame): *along* its axis (its length's ratio), *across* it (its girth's)."""
    axis = _unit(last - first)
    return across * np.eye(3) + (along - across) * np.outer(axis, axis)


def _hand_scale(source: dict, side: str, scales: FitScales) -> np.ndarray:
    """The hand's scale (3×3, source frame): its length along it, its knuckles' breadth across them, the arm's girth through the palm.

    The fingers stand on the character's own finger joints, so the palm is
    as broad as her knuckles are apart; how thick it is goes with the arm it
    is on (``FitScales.girth``).
    """
    try:
        axis = _unit(source[f"cf_j_middle03_{side}"] - source[f"cf_s_hand_{side}"])
        breadth = source[f"cf_j_index01_{side}"] - source[f"cf_j_little01_{side}"]
    except KeyError as missing:
        raise ValueError(f"body fitting needs {missing} to size the hand") from None
    breadth = _unit(breadth - (breadth @ axis) * axis)
    through = np.cross(axis, breadth)
    return (scales[f"hand {side}"] * np.outer(axis, axis) + scales[f"palm {side}"] * np.outer(breadth, breadth)
            + scales.girth("arm") * np.outer(through, through))


#: The joint helpers of the elbows and knees: what the clips turn and move to keep a bent joint's volume
#: (cf_s_elboback 48°, cf_s_kneeB 60° in 2_2, the same on a converted rig as on a card).
JOINT_HELPERS = ("elbo", "forearm01", "arm03", "leg01", "kneeB", "kneeF", "thigh03")


def _along(scales: FitScales, segment: str, name: str = "") -> float:
    """How a limb influence's skin is sized along its segment: the length's ratio, or under ``"bones"``
    1 for an elbow or knee helper (:data:`JOINT_HELPERS`), which keeps the source's skin round the joint."""
    if scales.cross_section == "bones" and any(part in name for part in JOINT_HELPERS):
        return 1.0
    return scales[segment]


def _limb_transform(name: str, source: dict[str, np.ndarray], target,
                    humanoid: dict, scales: dict, *, bone_mapping: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Source anchor and destination direction for an arm or leg influence, at the character's size.

    A segment is stretched along its axis to the character's segment length
    and sized across it by the limb's girth (``FitScales.girth``), the way a
    foot has always been stretched along it; a hand by :func:`_hand_scale`.
    """
    side = "L" if name.endswith("_L") else "R" if name.endswith("_R") else ""
    if not side:
        return np.eye(3), np.zeros(3)
    key = lambda stem: f"{stem}_{side}"
    role = "Left" if side == "L" else "Right"
    dst = lambda stem: target.world_position(humanoid[role + stem])
    if any(t in name for t in ("arm", "elbo", "wrist", "hand", "finger", "thumb",
                                "index", "middle", "ring", "little", "shoulder")):
        try:
            s0, s1, s2 = (source[key(n)] for n in (ARM_ROOTS[scales.arm_root], "cf_s_forearm01", "cf_s_hand"))
        except KeyError as missing:
            raise ValueError(f"body fitting needs {missing} for the source arm") from None
        d0, d1, d2 = (dst(n) for n in ("UpperArm", "LowerArm", "Hand"))
        upper = source[name][0] * (1 if side == "L" else -1) < abs(s1[0])
        if upper:
            rotation, anchor = _segment_anchor(source[name], s0, s1, d0, d1)
        else:
            rotation, anchor = _segment_anchor(source[name], s1, s2, d1, d2)
        # Finger pivots use the destination rig's authored positions.
        mapped = (bone_mapping[name] if bone_mapping is not None
                  else _bone_target(name, humanoid, set(target.index)))
        if any(name.startswith(f"cf_j_{stem}_") for stem in THUMB_CARRIES[scales.thumb_carry]):
            # carried as the hand's own skin is, not onto her thumb joint (:data:`THUMB_CARRIES`); what
            # it is bound to is ``_fit``'s mapping, her thumb joint, unchanged
            mapped = humanoid[role + "Hand"]
        # a finger joint: Endfield names its own "Finger"; a declared table names its pairs (``DeclaredMapping``)
        finger = "Finger" in mapped or mapped in getattr(bone_mapping, "fingers", ())
        if finger and mapped in target.index:
            anchor = target.world_position(mapped)
        if scales.cross_section == "length":
            part = ("hand" if mapped == humanoid[role + "Hand"] or finger
                    else "upper arm" if upper else "forearm")
            rotation = rotation * scales[f"{part} {side}"]
        elif mapped == humanoid[role + "Hand"] or finger:
            rotation = rotation @ _hand_scale(source, side, scales)
        elif upper:
            rotation = rotation @ _segment_scale(s0, s1, _along(scales, f"upper arm {side}", name), scales.girth("arm"))
        else:
            rotation = rotation @ _segment_scale(s1, s2, _along(scales, f"forearm {side}", name), scales.girth("arm"))
        return rotation, anchor - rotation @ source[name]
    if any(t in name for t in ("thigh", "leg", "knee", "foot", "toes")):
        s0, s1, s2 = (source[key(n)] for n in ("cf_s_thigh01", "cf_s_leg01", "cf_j_foot"))
        d0, d1, d2 = (dst(n) for n in ("UpperLeg", "LowerLeg", "Foot"))
        if (any(t in name for t in ("foot", "toes"))
                and key("cf_j_toes") in source and role + "Toes" in humanoid):
            rotation, anchor = _segment_anchor(
                source[name], s2, source[key("cf_j_toes")], d2, dst("Toes"), stretch_axis=True
            )
        elif source[name][1] > s1[1]:
            rotation, anchor = _segment_anchor(source[name], s0, s1, d0, d1)
            rotation = (rotation * scales[f"thigh {side}"] if scales.cross_section == "length"
                        else rotation @ _segment_scale(s0, s1, _along(scales, f"thigh {side}", name),
                                                       scales.girth("leg")))
        else:
            rotation, anchor = _segment_anchor(source[name], s1, s2, d1, d2)
            rotation = (rotation * scales[f"shin {side}"] if scales.cross_section == "length"
                        else rotation @ _segment_scale(s1, s2, _along(scales, f"shin {side}", name),
                                                       scales.girth("leg")))
        return rotation, anchor - rotation @ source[name]
    return np.eye(3), np.zeros(3)


#: What the fit prefixes a source joint with when it keeps it on the character (``store.GRAFT``:
#: the store reads what an earlier version named otherwise)
GRAFT = store.GRAFT

#: The spine joints of the lent body and the character's human bones they stand on, pelvis first.
SPINE = (("cf_s_waist02", "Hips"), ("cf_s_spine01", "Chest"), ("cf_s_spine02", "UpperChest"),
         ("cf_s_neck", "Neck"))


class TorsoField:
    """Where the fit puts a torso point: the torso at its size, bent through the character's spine joints.

    The trunk is carried uniformly by the torso's length about the pelvis
    (``FitScales``); what is left between that and each of the character's
    spine joints (:data:`SPINE`) is a residual, interpolated by the point's
    source height between the joints and held past the ends.  One field for
    every torso influence, so neighbouring influences agree: pinning each to
    its own joint left steps the skin blended across — the chest met the
    shoulders 7 cm apart, a ridge across the chest and a ring under the neck
    (§7.1064).

    The arms hang on the character's own shoulder joints, which stand about
    2 cm in and down from where the torso's size puts the lent arm roots
    (every character measured, §7.1064): the trunk round each arm root goes
    that way with it, fully at the root, fading towards the middle of the
    chest and down to the upper chest joint (from the chest joint down, the
    shoulder skin Avywen keeps of her own lost its cut, §7.1064).  Without it
    the trunk met each arm 2 cm out of line and the armpit was pinched in.
    """

    def __init__(self, source: dict[str, np.ndarray], target, humanoid: dict, scale: float,
                 *, arm_root: str = "skin"):
        if arm_root not in ARM_ROOTS:
            raise ValueError(f"unknown arm root {arm_root!r}")
        self.scale = scale
        self.pelvis = source["cf_s_waist02"]
        self.there = target.world_position(humanoid["Hips"])
        rows = sorted(((float(source[name][1]),
                       target.world_position(humanoid[bone]) - self.base(source[name][None])[0])
                      for name, bone in SPINE if name in source and humanoid.get(bone) in target.index),
                      key=lambda row: row[0])
        if "cf_s_spine03" in source and rows:
            # the breasts and the top of the chest keep the upper chest's place up to the lent
            # body's top spine joint, and only above it lean to the neck: leaning from the upper
            # chest joint, Avywen's breasts went 7-12 mm back under her own chest skin (§7.1064)
            height = float(source["cf_s_spine03"][1])
            below = [residual for at, residual in rows if at <= height]
            if below:
                rows = sorted(rows + [(height, below[-1])], key=lambda row: row[0])
        self.heights = np.asarray([height for height, _ in rows])
        self.residuals = np.asarray([residual for _, residual in rows])
        #: ``(source arm root, its residual beyond the spine's)`` per side
        self.arms = []
        chest = source.get("cf_s_spine02")
        self.chest_height = None if chest is None else float(chest[1])
        for side, role in (("L", "Left"), ("R", "Right")):
            root = source.get(f"{ARM_ROOTS[arm_root]}_{side}")
            if arm_root == "joint" and root is None and humanoid.get(role + "UpperArm") in target.index:
                raise ValueError(f"body fitting needs {ARM_ROOTS[arm_root]}_{side} for the torso arm root")
            if root is None or self.chest_height is None or humanoid.get(role + "UpperArm") not in target.index:
                continue
            spine = self.base(root[None])[0] + self._spine(root[1])[0]
            self.arms.append((root, target.world_position(humanoid[role + "UpperArm"]) - spine))

    def base(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(points, dtype=np.float64) - self.pelvis) + self.there

    def _spine(self, heights) -> np.ndarray:
        heights = np.atleast_1d(np.asarray(heights, dtype=np.float64))
        return np.stack([np.interp(heights, self.heights, self.residuals[:, axis]) for axis in range(3)], axis=1)

    def residual(self, points) -> np.ndarray:
        """What the field adds to :meth:`base` at source *points* (N×3)."""
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        out = self._spine(points[:, 1])
        for share, (_root, delta) in zip(self.arm_shares(points).T, self.arms):
            out += share[:, None] * delta
        return out

    def arm_shares(self, points) -> np.ndarray:
        """How much of each arm root's residual the field gives source *points*: N × (one per arm)."""
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        out = np.zeros((len(points), len(self.arms)))
        for side, (root, _delta) in enumerate(self.arms):
            across = np.clip(points[:, 0] / root[0], 0, 1)
            down = np.clip((points[:, 1] - self.chest_height) / max(root[1] - self.chest_height, 1e-6), 0, 1)
            out[:, side] = (across * across * (3 - 2 * across)) * (down * down * (3 - 2 * down))
        return out


#: The most influences a lent vertex keeps once :func:`carry_arm_residual` has given one more joint weight:
#: what the runtime skins a vertex with (a converted body's own skin carries 8).
CARRY_WIDTH = 8


def carry_arm_residual(weights: np.ndarray, indices: np.ndarray, torso: np.ndarray, joint_shares: np.ndarray,
                       vertex_shares: np.ndarray) -> np.ndarray:
    """Dense weights (V × bones) in which the torso joints short of a point's arm residual pass it on.

    The field moves a torso point by *vertex_shares* of an arm root's residual
    (:meth:`TorsoField.arm_shares`), but a torso joint is fitted rigidly with
    the share at its own place (*joint_shares*, bones × arms): the upper
    chest joint none, the shoulder joint all.  A point weighted to both is
    held by each joint 40 mm from where the other holds it (Texas's arm root
    stands that far below her trunk's size), and bending the trunk pulled the
    two apart: the shoulder's front squashed (§7.1444).  So a torso joint
    with less of the residual than the point gives the joint with the most,
    on the same side, ``(s_v − s_k) / (s_top − s_k)`` of its weight: that
    part of it then carries the point's share.  Joints with more keep theirs
    (making those exact too is not yet built, §7.1445); limb joints are not
    torso joints.
    """
    weights = np.asarray(weights, dtype=np.float64)
    count = len(torso)
    dense = np.zeros((len(weights), count))
    rows = np.repeat(np.arange(len(weights)), weights.shape[1])
    np.add.at(dense, (rows, np.asarray(indices).reshape(-1)), weights.reshape(-1))
    torso = np.asarray(torso, dtype=bool)
    for side in range(joint_shares.shape[1]):
        share = np.where(torso, joint_shares[:, side], -np.inf)
        receiver = int(np.argmax(share))
        top = joint_shares[receiver, side]
        need = vertex_shares[:, side][:, None]
        own = joint_shares[:, side][None, :]
        give = np.where(torso[None, :] & (own < need) & (top > own),
                        np.clip((need - own) / np.maximum(top - own, 1e-12), 0, 1), 0.0)
        give[:, receiver] = 0.0
        moved = dense * give
        dense -= moved
        dense[:, receiver] += moved.sum(axis=1)
    return dense


def _fit(piece: Piece, target, humanoid: dict, *, bone_mapping: dict | None = None,
         girth: dict | None = None, cross_section: str = "girth", thumb_carry: str = "none",
         arm_root: str = "skin") -> tuple:
    """:func:`fit_transforms` and, per bone, whether it is a torso influence, with the torso's field."""
    source = {name: np.asarray(matrix, dtype=np.float64)[:3, 3]
              for name, matrix in zip(piece.bones, piece.bone_world)}
    if bone_mapping is not None:
        missing = set(piece.bones) - set(bone_mapping)
        if missing:
            raise ValueError(f"body fit has no explicit bone mapping for {sorted(missing)}")
    mapped = [(bone_mapping[name] if bone_mapping is not None
               else _bone_target(name, humanoid, set(target.index))) for name in piece.bones]
    missing = sorted(set(mapped) - set(target.index))
    if missing:
        raise ValueError(f"Endfield skeleton is missing {missing}")
    # Move the source mesh into Origin rest-world coordinates first. The female
    # base stores world positions, while the male base stores positions in a
    # renderer-local frame offset from its bones.
    # Every influence is carried at the character's size, not the source's
    # (``FitScales``): a limb influence scaled by its own segment, a torso
    # influence uniformly by the torso's length about its pivot.  Not by the
    # spacing of the shoulder and hip joints — that is not a body's breadth
    # (sized by it, a body came out narrower than the character's own
    # shoulder skin, §7.1060).
    scales = FitScales(source, target, humanoid, bone_mapping=bone_mapping, girth=girth,
                       cross_section=cross_section, thumb_carry=thumb_carry, arm_root=arm_root)
    # a joint that weighs no vertex moves nothing: it is only there to be measured
    weights, indices = split_archive_skin(np.asarray(piece.skin, dtype=np.float64), bone_count=len(piece.binds))
    used = set(np.unique(indices[weights > 0]).tolist())
    field = None
    transforms, offsets, torso = [], [], []
    for k, (name, mapped_name) in enumerate(zip(piece.bones, mapped)):
        # the top of the shoulder goes with the trunk: pinned to the clavicle joint, which a
        # character's rig puts beside the neck, it stood 9 cm in from its arm (§7.1064)
        limb = mapped_name != humanoid.get("Hips") and name.endswith(("_L", "_R")) and not any(
            t in name for t in ("bust", "bnip", "siri", "shoulder02"))
        torso.append(k in used and not limb)
        if k not in used:
            rotation, offset = np.eye(3), np.zeros(3)
        elif limb:
            if bone_mapping is None:
                rotation, offset = _limb_transform(name, source, target, humanoid, scales)
            else:
                rotation, offset = _limb_transform(name, source, target, humanoid, scales,
                                                   bone_mapping=bone_mapping)
        else:
            if field is None:
                field = TorsoField(source, target, humanoid, scales["torso"], arm_root=arm_root)
            # the joint's own point takes the field's residual there; each point of
            # its skin takes the residual at its own height (``fit_body``)
            rotation = np.eye(3) * field.scale
            offset = field.base(np.zeros((1, 3)))[0] + field.residual(source[name])[0]
        transforms.append(rotation)
        offsets.append(offset)
    return mapped, np.asarray(transforms), np.asarray(offsets), np.asarray(torso, dtype=bool), field


def _carried(piece: Piece, weights, indices, torso, field, points) -> tuple:
    """*piece*'s source skin with :func:`carry_arm_residual` applied: ``(weights, indices)``."""
    if field is None or not field.arms:
        return weights, indices
    joints = np.stack([np.asarray(piece.bone_world[k], dtype=np.float64)[:3, 3] for k in range(len(piece.binds))])
    dense = carry_arm_residual(weights, indices, torso[:len(piece.binds)], field.arm_shares(joints),
                               field.arm_shares(points))
    # the joint given weight is one more influence: room for it, as many as the runtime skins
    width = int(min(max(weights.shape[1], np.count_nonzero(dense > 0, axis=1).max()), CARRY_WIDTH))
    order = np.argsort(-dense, axis=1, kind="stable")[:, :width]
    kept = np.take_along_axis(dense, order, axis=1)
    weights = kept / kept.sum(axis=1, keepdims=True)
    return weights, np.where(weights > 0, order, 0)


def fitted_skin(piece: Piece, target, humanoid: dict, arm_residual: str = "joints", **fit) -> tuple:
    """The weights :func:`fit_body` gives *piece*'s skin (source bones): ``(weights, indices)``.

    *fit*: :func:`fit_transforms`'s keywords, as the body was fitted with them; *arm_residual* likewise.
    """
    weights, indices = split_archive_skin(np.asarray(piece.skin, dtype=np.float64), bone_count=len(piece.binds))
    if arm_residual != "carried":
        return weights, indices
    _mapped, _transforms, _offsets, torso, field = _fit(piece, target, humanoid, **fit)
    rest = np.stack(piece.bone_world[:len(piece.binds)]) @ piece.binds
    blend = np.einsum("vk,vkij->vij", weights, rest[indices])
    points = np.einsum("vij,vj->vi", blend, np.c_[piece.mesh.positions, np.ones(len(weights))])[:, :3]
    return _carried(piece, weights, indices, torso, field, points)


def fit_transforms(piece: Piece, target, humanoid: dict, *, bone_mapping: dict | None = None,
                   girth: dict | None = None, cross_section: str = "girth", thumb_carry: str = "none",
                   arm_root: str = "skin") -> tuple:
    """``(mapped, transforms, offsets)`` per bone of *piece*: where the fit sends its skin.

    ``mapped[k]`` is the character's bone the source bone ``piece.bones[k]``
    goes with; a point *p* that bone carries lands at ``transforms[k] @ p +
    offsets[k]`` — for a torso influence, the point at the joint's own height
    (``fit_body`` bends the rest of its skin with :class:`TorsoField`).
    *girth*, *cross_section*, *thumb_carry*, *arm_root*: :class:`FitScales`'s, as the body was fitted with them.
    """
    mapped, transforms, offsets, _torso, _field = _fit(piece, target, humanoid, bone_mapping=bone_mapping,
                                                       girth=girth, cross_section=cross_section,
                                                       thumb_carry=thumb_carry, arm_root=arm_root)
    return mapped, transforms, offsets


def fit_body(piece: Piece, target, humanoid: dict, face_mesh=None, collar_shift=None,
             *, bone_mapping: dict | None = None, girth: dict | None = None,
             cross_section: str = "girth", thumb_carry: str = "none", arm_root: str = "skin",
             arm_residual: str = "joints") -> Piece:
    """Fit adult source skin without changing source meshes, weights or rigs.

    All mapped pelvis influences share its rest displacement. Foot influences
    share one axial stretch, retaining the authored transverse dimensions.
    *collar_shift* (a horizontal offset) moves the neck so the lent collar
    stands under the head opening it is fitted to, ramped from the upper
    chest to the head (``recompose.collar_offset``).
    *bone_mapping*, when supplied, must name every source influence explicitly;
    it lets other source adapters reuse this fit without changing global mappers.
    *girth* (``{"arm": ratio | None, "leg": …}``, :func:`measured_girth`):
    how thick the character's limbs are next to the lent body's
    (:meth:`FitScales.girth`).  *cross_section*: :data:`CROSS_SECTIONS`; *thumb_carry*: :data:`THUMB_CARRIES`.
    *arm_root*: :data:`ARM_ROOTS`, shared by the torso field, limb spans and helper positions.
    *arm_residual*: :data:`ARM_RESIDUALS`; ``carried`` re-splits the torso weights (:func:`carry_arm_residual`).
    """
    if piece.skin is None or piece.bone_world is None:
        raise ValueError("Origin body bake has no skin or rest bone transforms")
    mapped, transforms, offsets, torso, field = _fit(piece, target, humanoid, bone_mapping=bone_mapping,
                                                     girth=girth, cross_section=cross_section,
                                                     thumb_carry=thumb_carry, arm_root=arm_root)
    weights, indices = split_archive_skin(np.asarray(piece.skin, dtype=np.float64), bone_count=len(piece.binds))
    if len(piece.binds) <= int(indices.max()):
        raise ValueError("Origin body bake is missing a bind matrix for a skin bone")
    rest = np.stack(piece.bone_world[:len(piece.binds)]) @ piece.binds
    bind_blend = np.einsum("vk,vkij->vij", weights, rest[indices])
    src = np.einsum("vij,vj->vi", bind_blend,
                    np.c_[piece.mesh.positions, np.ones(len(weights))])[:, :3]
    transformed = np.einsum("vkij,vj->vki", transforms[indices], src) + offsets[indices]
    fitted = np.einsum("vk,vki->vi", weights, transformed)
    if field is not None:
        # each torso influence carried the residual at its joint's height; its share of
        # a point takes the residual at the point's own height instead
        share = (weights * torso[indices]).sum(axis=1)
        at_joints = np.einsum("vk,vki->vi", weights * torso[indices], offsets[indices]) \
            - share[:, None] * field.base(np.zeros((1, 3)))[0]
        fitted += share[:, None] * field.residual(src) - at_joints
    if arm_residual == "carried":
        weights, indices = _carried(piece, weights, indices, torso, field, src)
    if face_mesh is not None:
        # Place the Origin collar inside the lower Endfield face. The face sits
        # forward of the prefab's neck bone, so the bones alone leave a gap
        # when viewed from the side.
        face_pos = np.asarray(face_mesh.positions, dtype=np.float64)
        face_low = float(face_pos[:, 1].min())
        band_height = .06 * float(np.ptp(face_pos[:, 1]))
        face_band = face_pos[face_pos[:, 1] < face_low + band_height]
        body_top = float(fitted[:, 1].max())
        body_band = fitted[fitted[:, 1] > body_top - band_height]
        if len(face_band) and len(body_band):
            depth = float(np.mean(face_band[:, 2]) - np.mean(body_band[:, 2]))
            neck_y = float(target.world_position(humanoid["Neck"])[1])
            head_y = float(target.world_position(humanoid["Head"])[1])
            neck_span = max(head_y - neck_y, 1e-6)
            # The face's lowest vertices are its chin, not a neck opening.
            # Keep the Origin neck inside the head and ramp its forward offset
            # along the target's neck-to-head span.
            blend = np.clip((fitted[:, 1] - neck_y) / neck_span, 0, 1)
            blend = blend * blend * (3 - 2 * blend)
            fitted[:, 2] += depth * blend
    if collar_shift is not None:
        # over the whole upper chest and neck: a petite neck is 8 cm, and a collar moved
        # 6.5 cm over that span folds on itself (33 crossings; from the upper chest, none, §7.1060)
        low = float(target.world_position(humanoid["UpperChest"])[1])
        high = float(target.world_position(humanoid["Head"])[1])
        ramp = np.clip((fitted[:, 1] - low) / max(high - low, 1e-6), 0, 1)
        ramp = ramp * ramp * (3 - 2 * ramp)
        shift = np.asarray(collar_shift, dtype=np.float64) * [1.0, 0.0, 1.0]
        fitted += ramp[:, None] * shift
    mesh = piece.mesh.copy()
    mesh.name = "endfield_complete_body"
    mesh.positions = fitted.astype(np.float32)
    mesh.compute_normals()
    bones = sorted(set(mapped))
    remap = np.asarray([bones.index(name) for name in mapped], dtype=np.float32)
    skin = np.concatenate((weights.astype(np.float32), remap[indices]), axis=1)
    worlds = [target.bones[target.index[name]].bind_world.astype(np.float32) for name in bones]
    binds = np.linalg.inv(np.stack(worlds)).astype(np.float32)
    return Piece(mesh, bones, skin, binds, bone_world=worlds)


def graft_male_nodes(nodes, humanoid, source_body, anatomy, reference):
    """Append authored source joints while preserving every native node.

    The stock Origin reference contains the male joints even when its uncensor
    implant list is empty. Placement is rigid — the joints turn when an H row
    aims them — and puts the anatomy's root where ``fit_body``'s sized pelvis
    puts that point; child rest transforms remain their authored local
    matrices.
    """
    skeleton = build_skeleton(nodes, "native_body_graft")
    source_world = dict(zip(source_body.bones, source_body.bone_world))
    if "cf_s_waist02" not in source_world:
        raise ValueError("male body lacks authored pelvis anchor")
    pelvis = np.asarray(source_world["cf_s_waist02"])[:3, 3]
    positions = {name: np.asarray(world, dtype=np.float64)[:3, 3] for name, world in source_world.items()}
    size = FitScales(positions, skeleton, humanoid)["torso"]
    required = set()
    for piece in anatomy:
        weights, indices = split_archive_skin(piece.skin, bone_count=len(piece.bones))
        active = indices[weights > 0]
        for index in np.unique(active):
            name = piece.bones[index]
            if not name.startswith("cm_J_dan"):
                raise ValueError(f"unsupported authored anatomy influence {name}")
            visited = set()
            while name.startswith("cm_J_dan"):
                if name in visited or name not in reference.matrices:
                    raise ValueError(f"missing or cyclic authored anatomy joint {name}")
                visited.add(name); required.add(name)
                name = reference.parents.get(name, "")
        for name, world in zip(piece.bones, piece.bone_world):
            if name in required and np.max(np.abs(np.asarray(world) - reference.matrices[name])) > 1e-4:
                raise ValueError(f"authored anatomy reference transform mismatch for {name}")
    roots = sorted(name for name in required if reference.parents.get(name, "") not in required)
    if not roots:
        raise ValueError("authored male anatomy has no root joint")
    root = np.mean([np.asarray(reference.matrices[name])[:3, 3] for name in roots], axis=0)
    placement = np.eye(4)
    placement[:3, 3] = skeleton.world_position(humanoid["Hips"]) + size * (root - pelvis) - root
    augmented = dict(nodes)
    mapping = {name: GRAFT + name for name in sorted(required)}
    pending = set(required)
    while pending:
        progress = False
        for name in sorted(pending):
            source_parent = reference.parents.get(name, "")
            if source_parent in pending:
                continue
            grafted = mapping[name]
            if grafted in nodes:
                raise ValueError(f"authored body joint collides with native node {grafted}")
            if source_parent in mapping:
                parent = mapping[source_parent]
                local = np.linalg.inv(reference.matrices[source_parent]) @ reference.matrices[name]
            else:
                parent = humanoid["Hips"]
                local = (np.linalg.inv(skeleton.bones[skeleton.index[parent]].bind_world)
                         @ placement @ reference.matrices[name])
            augmented[grafted] = (parent, np.asarray(local).tolist())
            pending.remove(name); progress = True
        if not progress:
            raise ValueError("authored male body hierarchy could not be resolved")
    return augmented, mapping, placement


def fit_anatomy_piece(piece, target, mapping, placement):
    """Carry the source surface and exact skin weights to appended joints."""
    weights, indices = split_archive_skin(np.asarray(piece.skin, dtype=np.float64), bone_count=len(piece.binds))
    rest = np.stack(piece.bone_world[:len(piece.binds)]) @ piece.binds
    blend = np.einsum("vk,vkij->vij", weights, rest[indices])
    positions = np.einsum("vij,vj->vi", blend,
                         np.c_[piece.mesh.positions, np.ones(len(weights))])
    mesh = piece.mesh.copy()
    mesh.name = f"endfield_{piece.mesh.name}"
    mesh.positions = (positions @ placement.T)[:, :3].astype(np.float32)
    normals = np.einsum("vij,vj->vi", blend[:, :3, :3], piece.mesh.normals)
    normals = normals @ placement[:3, :3].T
    mesh.normals = (normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    active = sorted({piece.bones[index] for index in np.unique(indices[weights > 0])})
    names = [mapping[name] for name in active]
    remap = np.zeros(len(piece.bones), dtype=np.float32)
    for i, name in enumerate(piece.bones):
        if name in active:
            remap[i] = active.index(name)
    worlds = [target.bones[target.index[name]].bind_world.astype(np.float32) for name in names]
    skin = np.c_[weights, remap[indices]].astype(np.float32)
    return Piece(mesh, names, skin, np.linalg.inv(np.stack(worlds)).astype(np.float32),
                 bone_world=worlds, shader=piece.shader)


def helper_nodes(piece: Piece, target, humanoid: dict, *, bone_mapping: dict | None = None,
                 girth: dict | None = None, cross_section: str = "girth", thumb_carry: str = "none",
                 arm_root: str = "skin") -> dict:
    """``{node: (parent, local 4×4)}``: the source joints *piece*'s skin hangs on, kept on *target*.

    The lent skin's weights are for a rig whose knees, elbows, hips and chest
    have helper joints, and the clips written for that rig move and scale
    those joints to keep a bent limb's volume.  Merged into the character's
    bones they have nothing to move.  Each joint the skin uses is kept as a
    node of its own (:data:`GRAFT` + its name) under the character's bone it
    goes with, standing where the fit puts it, so that left alone it follows
    that bone exactly as the merged weights did — and a clip can move it
    (``hrole.StandIn``).  A joint the skeleton already has is not made again.
    """
    mapped, transforms, offsets = fit_transforms(piece, target, humanoid, bone_mapping=bone_mapping, girth=girth,
                                                 cross_section=cross_section, thumb_carry=thumb_carry, arm_root=arm_root)
    weights, indices = split_archive_skin(np.asarray(piece.skin, dtype=np.float64), bone_count=len(piece.binds))
    out = {}
    for k in np.unique(indices[weights > 0]):
        name = GRAFT + piece.bones[k]
        if target.has(name):
            continue
        fit = np.eye(4)
        fit[:3, :3], fit[:3, 3] = transforms[k], offsets[k]
        world = fit @ np.asarray(piece.bone_world[k], dtype=np.float64)
        parent = mapped[k]
        local = np.linalg.inv(target.bones[target.index[parent]].bind_world) @ world
        out[name] = (parent, local.tolist())
    return out


#: How far the stored merged weights may be from the source's summed by bone and still be that skin.
SAME_SKIN = 1e-3


def keep_helpers(character: str, *, write: bool = False) -> dict:
    """Bind ``body/<character>``'s lent skin to the source's own joints (:func:`helper_nodes`).

    Positions, seams, tints and masks are untouched: only what each vertex is
    weighted to changes, from the character's bones to the kept joints that
    hang under them.  Refused when the stored weights are not the source's
    weights summed by bone — then this is not the skin it thinks it is.
    Every kept joint also gets its twin (:func:`twin_nodes`), which the
    loader skins the lent skin near a seam with (:func:`rest_twins`).
    """
    from .certification import capture
    from .samples import _body_piece
    extra = store.read_extra("chara", character) or {}
    body_extra = store.read_extra("body", character) or {}
    native = store.read_item("chara", character)
    item = store.read_item("body", character)
    source = store.read_item("body_source", character)
    if native is None or item is None or source is None:
        raise ActorError(f"body/{character}: the body, its character or its source is not in the store")
    origin_game = next((piece for piece in source.pieces if piece.mesh.name == "o_body_a"), None)
    if origin_game is None:
        raise ActorError(f"body/{character}: its source has no o_body_a mesh")
    humanoid = extra.get("humanoid") or {}
    skeleton = build_skeleton(native.nodes, character)
    append_body_nodes(skeleton, item.nodes)
    # the girth the body was fitted with (``bake_body``): the joints stand where that fit put them
    girth = body_extra.get("limb_girth")
    across = body_extra.get("fit_cross_section", "girth")
    thumbs = body_extra.get("fit_thumb_carry", "none")
    arm_root = body_extra.get("fit_arm_root", "skin")
    mapping = declared_bone_mapping(origin_game, skeleton, humanoid, extra.get("fingers"))
    added = helper_nodes(origin_game, skeleton, humanoid, girth=girth, cross_section=across, thumb_carry=thumbs,
                         bone_mapping=mapping, arm_root=arm_root)
    twins = twin_nodes({**item.nodes, **added})
    report = {"character": character, "written": False, "joints": len(added), "twins": len(twins)}
    if not added and not twins:
        report["reason"] = "its skin already hangs on the source's joints, and they have their twins"
        return report
    nodes = {**item.nodes, **added, **twins}
    if not added:
        # only the twins are new (``rest_twins``): nothing is weighted to them in the store
        if not write:
            return report
        with store.opened(character, write=True):
            bake.write_item("body", character, item.pieces, source=store.SOURCE, nodes=nodes, label=character,
                            extra_maps=item.maps)
        manifest = body_extra.get("composite")
        if manifest:
            manifest["source_certification"] = capture(character, manifest)
        with store.opened(character, write=True):
            bake.write_named_table(store.SOURCE, "body", character, body_extra)
        report["written"] = True
        return report
    kept = build_skeleton(native.nodes, character)
    append_body_nodes(kept, nodes)
    # the weights the fit gave the skin, not the source's as authored (``carry_arm_residual``)
    weights, indices = fitted_skin(origin_game, skeleton, humanoid, body_extra.get("fit_arm_residual", "joints"),
                                   bone_mapping=mapping, girth=girth, cross_section=across, thumb_carry=thumbs,
                                   arm_root=arm_root)
    mapped, _transforms, _offsets = fit_transforms(origin_game, skeleton, humanoid, bone_mapping=mapping, girth=girth,
                                                   cross_section=across,
                                                   thumb_carry=thumbs, arm_root=arm_root)
    # a bone no vertex is weighted to was not kept; nothing reads its row
    to_kept = np.array([kept.index.get(GRAFT + name, 0) for name in origin_game.bones], dtype=np.int64)
    to_merged = np.array([kept.index[name] for name in mapped], dtype=np.int64)
    count = origin_game.mesh.vertex_count
    pieces, worst = [], 0.0
    for piece in item.pieces:
        mesh = _rest_mesh(piece, kept)
        if mesh.vertex_count < count:
            raise ActorError(f"body/{character}: a stored piece has fewer vertices than its source body")
        was = np.zeros((count, len(kept.bones)))
        np.add.at(was, (np.arange(count)[:, None], mesh.bone_idx[:count].astype(np.int64)), mesh.bone_wt[:count])
        now = np.zeros_like(was)
        np.add.at(now, (np.arange(count)[:, None], to_merged[indices]), weights)
        worst = max(worst, float(np.abs(was - now).max()))
        width = max(mesh.bone_idx.shape[1], indices.shape[1])
        index = np.zeros((mesh.vertex_count, width), dtype=np.int32)
        weight = np.zeros((mesh.vertex_count, width), dtype=np.float32)
        index[:, :mesh.bone_idx.shape[1]], weight[:, :mesh.bone_wt.shape[1]] = mesh.bone_idx, mesh.bone_wt
        index[:count], weight[:count] = 0, 0.0
        index[:count, :indices.shape[1]] = to_kept[indices]
        weight[:count, :indices.shape[1]] = weights
        mesh.bone_idx, mesh.bone_wt = index, weight
        pieces.append(_body_piece(mesh, kept))
    report["stored_weights_differ"] = worst
    if worst > SAME_SKIN:
        raise ActorError(f"body/{character}: its stored weights are not its source's summed by bone "
                         f"(off by {worst:.2e})")
    if not write:
        return report
    with store.opened(character, write=True):
        bake.write_item("body", character, pieces, source=store.SOURCE, nodes=nodes, label=character,
                        extra_maps=item.maps)
    body_extra["helper_joints"] = sorted(added)
    manifest = body_extra.get("composite")
    if manifest:
        manifest["source_certification"] = capture(character, manifest)
    with store.opened(character, write=True):
        bake.write_named_table(store.SOURCE, "body", character, body_extra)
    report["written"] = True
    return report


def append_body_nodes(skeleton, nodes):
    """Append body joints without changing native runtime bone indices."""
    pending = {name for name in nodes if not skeleton.has(name)}
    while pending:
        progress = False
        for name in sorted(pending):
            parent, local = nodes[name]
            if parent and not skeleton.has(parent):
                continue
            skeleton.add_matrix(name, parent or None, local)
            pending.remove(name); progress = True
        if not progress:
            raise ActorError("body graft has unresolved authored parent nodes")
    skeleton.finalize()
    return skeleton


def _merge_fitted_skin(pieces, skeleton, source_pieces):
    """Keep the primary archive piece as one complete fitted skin source."""
    from ..mesh.mesh import Mesh
    meshes = [_rest_mesh(piece, skeleton) for piece in pieces]
    ranges = []; vertex_offset = face_offset = 0
    merged = Mesh("endfield_complete_body")
    present = {name for mesh in meshes for name, _ in Mesh.OPTIONAL_STREAMS
               if len(getattr(mesh, name)) == mesh.vertex_count}
    for piece, mesh, source_piece in zip(pieces, meshes, source_pieces, strict=True):
        for name, width in Mesh.OPTIONAL_STREAMS:
            if name in present and len(getattr(mesh, name)) != mesh.vertex_count:
                setattr(mesh, name, np.zeros((mesh.vertex_count, width), dtype=np.float32))
        ranges.append({"mesh": piece.mesh.name, "vertex_start": vertex_offset,
                       "vertices": mesh.vertex_count, "face_start": face_offset,
                       "triangles": mesh.tri_count, "source_mesh": source_piece.mesh.name,
                       "hidden_with_outfit": source_piece.mesh.name in GENITAL_MESHES[1]})
        merged.append(mesh)
        vertex_offset += mesh.vertex_count; face_offset += mesh.tri_count
    names = [bone.name for bone in skeleton.bones]
    worlds = np.stack([bone.bind_world for bone in skeleton.bones]).astype(np.float32)
    return (Piece(merged, names, np.c_[merged.bone_wt, merged.bone_idx].astype(np.float32),
                  np.linalg.inv(worlds).astype(np.float32), bone_world=list(worlds)), ranges)


def card_skin(card, source, mesh, raw_sex: int, keep_marks: bool = False):
    """The card's body skin for a body lent to an Endfield character: ``(albedo, shade)``.

    Without the card's marks unless *keep_marks* — its paints, sunburn, skin
    overlays and crest belong to the card's own character (``body_texture``).
    """
    return body_texture(card, source.maps["skin"], source.maps["skin_mask"], mesh=mesh,
                        sex=0 if raw_sex == 1 else 1, marks=keep_marks)


def source_skin_normal(raw_sex: int) -> CS.Map | None:
    """Decode the source body's own linear tangent normal, using the card's sex convention."""
    from ..origin.figure.attach import tangent_normal
    from ..origin.figure.body_head import BODY_MATERIAL, create_props

    if raw_sex not in (0, 1):
        raise ValueError("source body normal requires a declared card sex")
    source = create_props(BODY_MATERIAL[0 if raw_sex == 1 else 1]).get("_NormalMap")
    if source is None:
        return None
    if source.srgb:
        raise ActorError("source body normal must have a linear color-space flag")
    decoded = tangent_normal(CS.pixels(source))
    if decoded is None:
        raise ActorError("source body normal cannot be decoded")
    return CS.Map(decoded, False)


def source_skin_normal_strength(raw_sex: int) -> float:
    """The strength the source body's material gives its normal map (its ``_BumpScale``).

    Kept with the body when it is baked: the runtime draws the lent normal at
    it when the character's own skin program has no strength of its own
    (:func:`skin_material`), and reads no install to find it.
    """
    from ..origin.figure.body_head import BODY_MATERIAL, create_props

    if raw_sex not in (0, 1):
        raise ValueError("source body normal requires a declared card sex")
    props = create_props(BODY_MATERIAL[0 if raw_sex == 1 else 1])
    if "_BumpScale" not in props:
        raise ActorError("source body material declares a normal map without its _BumpScale")
    return float(props["_BumpScale"])


def rebake_skin(character: str, *, keep_marks: bool = False, write: bool = False) -> dict:
    """Bake ``body/<character>``'s skin again from its card, without the card's marks.

    The fitted body, its bones and its seams are kept as they are; only the
    albedo map is replaced, and the source certification (which binds the
    archive) is renewed.  Returns what changed: the share of texels that
    differ, and how far the texels the seam tints were calibrated on moved —
    which has to be nothing, or the tints no longer match the native skin and
    the body has to be built again instead.
    """
    from .certification import capture
    extra = store.read_extra("body", character) or {}
    from ..core.paths import stored_card
    named = extra.get("source_card") or ""
    card_path = stored_card(named)
    if card_path is None:
        raise ActorError(f"body/{character}: its source card {named} is not here")
    item = store.read_item("body", character)
    source = store.read_item("body_source", character)
    if item is None or source is None or item.maps.get("albedo") is None:
        raise ActorError(f"body/{character}: the fitted body or its source is not in the store")
    card = read_origin_card(card_path)
    raw_sex = int(card.parameter.get("sex", -1))
    origin_game = next((piece for piece in source.pieces if piece.mesh.name == "o_body_a"), None)
    if origin_game is None:
        raise ActorError(f"body/{character}: its source has no o_body_a mesh")
    albedo, _shade = card_skin(card, source, origin_game.mesh, raw_sex, keep_marks)
    if albedo is None:
        raise ActorError(f"body/{character}: cannot bake the card's skin")
    old = item.maps["albedo"]
    new = CS.Map(albedo, True)
    report = {"character": character, "card": card_path.name, "size": list(np.shape(albedo)[:2]),
              "written": False}
    if np.shape(old.px) == np.shape(albedo):
        changed = np.any(np.asarray(old.px) != np.asarray(albedo), axis=-1)
        report["texels_changed"] = float(changed.mean())
    manifest = extra.get("composite") or {}
    tints = _retinted(character, item, manifest, old, new) if manifest.get("render_tints") else None
    if tints is not None:
        report.update(seam_samples=tints[1], tints_reproduced=tints[2])
        if tints[2] > TINTS_REPRODUCED:
            raise ActorError(f"body/{character}: its stored seam tints cannot be made again from "
                             f"its stored data (off by {tints[2]:.2e}); build the body again")
    if not write:
        return report
    with store.opened(character, write=True):
        bake.write_item("body", character, item.pieces, source=store.SOURCE, label=character,
                        extra_maps={**item.maps, "albedo": new}, nodes=item.nodes)
    extra["skin_marks"] = "kept" if keep_marks else "cleared"
    if item.maps.get("normal") is not None:
        extra["source_normal_strength"] = source_skin_normal_strength(raw_sex)
    if tints is not None:
        manifest["render_tints"] = tints[0].tolist()
    if manifest:
        manifest["source_certification"] = capture(character, manifest)
    with store.opened(character, write=True):
        bake.write_named_table(store.SOURCE, "body", character, extra)
    report["written"] = True
    return report


def _retinted(character: str, item, manifest: dict, old, new) -> tuple:
    """``(tints for the new skin, seam samples, how well the old tints were made again)``.

    The tints make the lent skin's albedo equal the native skin's along the
    seams and carry the median of that correction over the rest of the body
    (``skin_color.seam_tints``).  What they were aimed at is not stored, but
    it is the old skin's colour times the old tint at each seam vertex; with
    that, the same calibration is run on the new skin.  Run on the old skin
    it has to give the stored tints back, which is the third value.
    """
    from .skin_color import baseline_rows, seam_tints, uv_colors
    native = store.read_item("chara", character)
    skeleton = build_skeleton(native.nodes, character)
    append_body_nodes(skeleton, item.nodes)
    mesh = _rest_mesh(item.pieces[0], skeleton)
    stored = np.asarray(manifest["render_tints"], dtype=np.float64)
    seams = manifest["seams"]
    rows = list(seams["seams"]) + list(seams.get("head_seams", []))
    index = np.array([v for row in rows for v in row["render_native_vertices"]], dtype=np.int64)
    if not len(index):
        return stored.astype(np.float32), 0, 0.0
    band = max(row["bridge_band_m"] for row in rows) * 8
    before = uv_colors(old, mesh.uvs)
    targets = before[index] * stored[index, :3]
    rows_ = baseline_rows(seams)
    again = seam_tints(mesh.positions, before, index, targets, band, rows_)
    fresh = seam_tints(mesh.positions, uv_colors(new, mesh.uvs), index, targets, band, rows_)
    return fresh, len(index), float(np.abs(again - stored).max())


#: how far the stored seam tints may be from the same calibration run again (float32 and JSON)
TINTS_REPRODUCED = 1e-3


def measured_girth(piece: Piece, pieces: list, skeleton, humanoid: dict, unmeasured: tuple = ()) -> dict:
    """How thick the character's limbs are next to the lent body *piece*'s, on her own skin (``limb_girth``).

    *pieces*: the character's own; the ones drawn with her skin program are
    measured.  ``{"arm": ratio | None, "leg": …, "rings": […]}``, kept in the
    body's table and handed to :func:`fit_body` as its *girth*.  *unmeasured*
    (:func:`unmeasured_rule`): limbs whose own surface is declared a garment
    standing off the skin; they are not measured (None: the trunk's ratio).
    """
    from . import limb_girth
    weights, indices = split_archive_skin(np.asarray(piece.skin, dtype=np.float64), bone_count=len(piece.binds))
    rest = np.stack(piece.bone_world[:len(piece.binds)]) @ np.asarray(piece.binds, dtype=np.float64)
    blend = np.einsum("vk,vkij->vij", weights, rest[indices])
    positions = np.einsum("vij,vj->vi", blend, np.c_[piece.mesh.positions, np.ones(len(weights))])[:, :3]
    joints = {name: np.asarray(matrix, dtype=np.float64)[:3, 3] for name, matrix in zip(piece.bones, piece.bone_world)}
    available = set(skeleton.index)
    mapped = np.array([skeleton.index[_bone_target(name, humanoid, available)] for name in piece.bones[:len(piece.binds)]])
    member = {}
    for limb, root in limb_girth.ROOT.items():
        for side, role in (("L", "Left"), ("R", "Right")):
            under = np.zeros(len(skeleton.bones), dtype=bool)
            under[skeleton.descendants(humanoid[role + root])] = True
            member[(limb, side)] = np.sum(weights * under[mapped][indices], axis=1) >= limb_girth.MEMBER
    own = []
    for own_piece in pieces:
        if ((own_piece.shader or {}).get("_shader") != limb_girth.SKIN_PROGRAM
                and declared_role(own_piece) != "native_skin"):
            continue
        mesh = _rest_mesh(own_piece, skeleton)
        own.append((mesh.positions.astype(np.float64), mesh.indices.reshape(-1, 3).astype(np.int64),
                    mesh.bone_idx.astype(np.int64), mesh.bone_wt.astype(np.float64)))
    girth = limb_girth.measure(positions, piece.mesh.indices.reshape(-1, 3).astype(np.int64), member, joints,
                               own, skeleton, humanoid)
    for limb in unmeasured:
        girth[limb] = None
    girth["rings"] = [row for row in girth["rings"] if row[0] not in unmeasured]
    return girth


def fit_rule(character: str, chosen: str | None = None) -> str:
    """The cross-section rule *character*'s body is fitted by: *chosen*, else the one it was fitted by.

    A body already stored keeps its recorded ``fit_cross_section``, and one fitted before the record
    existed was fitted by ``"girth"`` — so a rebuild does not move a body to a newer default unless
    asked.  Only a body not stored yet takes :data:`ENDFIELD_CROSS_SECTION`.
    """
    if chosen is not None:
        if chosen not in CROSS_SECTIONS:
            raise ValueError(f"cross_section must be one of {CROSS_SECTIONS}, not {chosen!r}")
        return chosen
    table = store.read_extra("body", character)
    if table is None:
        return ENDFIELD_CROSS_SECTION
    return str(table.get("fit_cross_section", "girth"))


def thumb_rule(character: str, chosen: str | None = None) -> str:
    """The thumb carry *character*'s body is fitted with: *chosen*, else its recorded ``fit_thumb_carry``, else ``"none"``."""
    if chosen is not None:
        if chosen not in THUMB_CARRIES:
            raise ValueError(f"thumb_carry must be one of {tuple(THUMB_CARRIES)}, not {chosen!r}")
        return chosen
    return str((store.read_extra("body", character) or {}).get("fit_thumb_carry", "none"))


def arm_root_rule(character: str, chosen: str | None = None) -> str:
    """The recorded source arm pivot; an explicit choice requires rebuilding the fitted body."""
    value = chosen if chosen is not None else (store.read_extra("body", character) or {}).get("fit_arm_root", "skin")
    if value not in ARM_ROOTS:
        raise ValueError(f"unknown arm root {value!r}")
    return value


def arm_residual_rule(character: str, chosen: str | None = None) -> str:
    """The recorded :data:`ARM_RESIDUALS` choice (``fit_arm_residual``); unrecorded bodies keep ``joints``."""
    value = chosen if chosen is not None else (store.read_extra("body", character) or {}).get("fit_arm_residual", "joints")
    if value not in ARM_RESIDUALS:
        raise ValueError(f"unknown arm residual rule {value!r}")
    return value


def unmeasured_rule(character: str, chosen=None) -> tuple:
    """The limbs *character*'s girth is not measured on: *chosen*, else its recorded ``fit_unmeasured_limbs``, else none.

    ``limb_girth`` measures a limb on her own skin, the garments painted into
    it included, because on this game's characters those lie on the skin.  A
    converted model drawn as one piece (skin and clothes in one mesh, declared
    ``native_skin``) can have a garment standing off it: Texas's puffed sleeves
    measured her arm 1.62 times the lent body's (rings 1.39–2.55), the lent
    arm was fitted that thick, the skin in front of the armpit stood across
    ``cf_s_arm01``'s pivot instead of inside it, and lowering the arm crushed
    it to 0.19 (§7.1441).  Which limbs those are is the converter's to say,
    as it says ``lent_regions``; a declared limb keeps the lent body's
    proportion to the trunk, as one her skin does not go round.
    """
    from . import limb_girth
    value = chosen if chosen is not None else (store.read_extra("body", character) or {}).get(
        "fit_unmeasured_limbs", ())
    value = tuple(sorted(set(value)))
    unknown = set(value) - set(limb_girth.LIMBS)
    if unknown:
        raise ValueError(f"unmeasured limbs must be among {tuple(limb_girth.LIMBS)}, not {sorted(unknown)}")
    return value


def bake_body(character: str, card_path: Path, *, keep_marks: bool = False, collar_shift=None,
              cross_section: str | None = None, thumb_carry: str | None = None,
              arm_root: str | None = None, unmeasured_limbs=None, arm_residual: str | None = None) -> Path:
    """Fit an adult human body; the explicit card selects source sex and skin.

    The skin is the card's without its marks (paints, sunburn, overlays,
    crest) unless *keep_marks*.  *collar_shift*: :func:`fit_body`'s, kept in
    the body's table, as is the girth of her limbs it is fitted with
    (:func:`measured_girth`; :func:`keep_helpers` reads it back) and the
    cross-section rule (:func:`fit_rule`: *cross_section*, else the body's own) and the thumb carry
    (:func:`thumb_rule`: *thumb_carry*, else the body's own). *arm_root* likewise
    keeps the recorded rule (:func:`arm_root_rule`), defaulting to the legacy skin pivot, and
    *unmeasured_limbs* the limbs left unmeasured (:func:`unmeasured_rule`), and *arm_residual*
    the torso joints' carry (:func:`arm_residual_rule`).
    """
    rule = fit_rule(character, cross_section)
    residual = arm_residual_rule(character, arm_residual)
    thumbs = thumb_rule(character, thumb_carry)
    arm = arm_root_rule(character, arm_root)
    unmeasured = unmeasured_rule(character, unmeasured_limbs)
    extra = store.read_extra("chara", character) or {}
    family = require_body_scope(extra, character)
    card = read_origin_card(card_path)
    raw_sex = int(card.parameter.get("sex", -1))
    expected = 1 if family in FEMALE_HUMAN_TYPES else 0
    if raw_sex != expected:
        raise ValueError(f"{card_path}: card body sex does not match declared {family} rig family")
    source_ident = "body" if raw_sex == 1 else "body_m"
    source = bake.read_item("base", source_ident)
    actor = store.read_item("chara", character)
    if source is None or actor is None:
        raise FileNotFoundError("convert the Origin body and Endfield character first")
    origin_game = next((p for p in source.pieces if p.mesh.name == "o_body_a"), None)
    if origin_game is None:
        raise ValueError("Origin body bake has no complete o_body_a mesh")
    skeleton = build_skeleton(actor.nodes, character)
    humanoid = extra.get("humanoid") or {}
    if not humanoid:
        raise ValueError(f"{character} has no humanoid mapping")
    evidence = None if has_sdf_face(actor.pieces) else facial_evidence(extra)
    face_mesh = face_reference(actor.pieces, skeleton, extra, evidence)
    nodes = actor.nodes
    fitted_pieces = []
    source_pieces = [origin_game]
    graft = []
    if raw_sex == 0:
        anatomy = [p for name in GENITAL_MESHES[1] for p in source.pieces if p.mesh.name == name]
        missing = set(GENITAL_MESHES[1]) - {p.mesh.name for p in anatomy}
        if missing:
            raise ValueError(f"male source body lacks authored skin pieces {sorted(missing)}")
        source_pieces.extend(anatomy)
        from ..origin.rig import rig as source_rig
        nodes, mapping, placement = graft_male_nodes(actor.nodes, humanoid, origin_game, anatomy, source_rig())
        skeleton = build_skeleton(nodes, character)
        girth = measured_girth(origin_game, actor.pieces, skeleton, humanoid, unmeasured)
        fitted_pieces.append(fit_body(origin_game, skeleton, humanoid, face_mesh, collar_shift, girth=girth,
                                             cross_section=rule, thumb_carry=thumbs, arm_root=arm,
                                             arm_residual=residual))
        fitted_pieces.extend(fit_anatomy_piece(p, skeleton, mapping, placement) for p in anatomy)
        graft = [{"name": name, "parent": nodes[name][0], "local": nodes[name][1],
                  "bind_world": skeleton.bones[skeleton.index[name]].bind_world.tolist()}
                 for name in nodes if name not in actor.nodes]
    else:
        girth = measured_girth(origin_game, actor.pieces, skeleton, humanoid, unmeasured)
        mapping = declared_bone_mapping(origin_game, skeleton, humanoid, extra.get("fingers"))
        fitted_pieces.append(fit_body(origin_game, skeleton, humanoid, face_mesh, collar_shift, girth=girth,
                                             bone_mapping=mapping, cross_section=rule, thumb_carry=thumbs, arm_root=arm,
                                             arm_residual=residual))
    fitted, source_ranges = _merge_fitted_skin(fitted_pieces, skeleton, source_pieces)
    albedo, _shade = card_skin(card, source, origin_game.mesh, raw_sex, keep_marks)
    if albedo is None:
        raise ValueError(f"cannot bake Origin skin from {card_path}")
    maps = {"albedo": CS.Map(albedo, True)}
    normal = source_skin_normal(raw_sex)
    if normal is not None:
        maps["normal"] = normal
    with store.opened(character, write=True):
        bake.write_item("body_source", character, source.pieces,
                        source=store.SOURCE, label=character,
                        extra_maps=source.maps, nodes=source.nodes,
                        implant=source.implant)
        path = bake.write_item("body", character, [fitted], source="endfield",
                               label=character, extra_maps=maps,
                               nodes=nodes)
        bake.write_named_table(store.SOURCE, "body", character, {
            "source_card": str(card_path.resolve()), "source_body": f"base/{source_ident}",
            "model": extra.get("model"), "rig_family": family,
            "source_vertices": origin_game.mesh.vertex_count,
            "complete_source_vertices": fitted.mesh.vertex_count,
            "source_archive": f"body_source/{character}",
            "source_piece_ranges": source_ranges, "body_graft": graft,
            "skin_marks": "kept" if keep_marks else "cleared",
            "collar_shift": None if collar_shift is None else [float(v) for v in collar_shift],
            "limb_girth": girth,
            "fit_cross_section": rule,
            "fit_thumb_carry": thumbs,
            "fit_arm_root": arm,
            "fit_unmeasured_limbs": list(unmeasured),
            "fit_arm_residual": residual,
            "source_normal_strength": None if normal is None else source_skin_normal_strength(raw_sex),
            # what the trunk was sized by (``FitScales.torso_measure``)
            "fit_torso_measure": FitScales({name: np.asarray(matrix, dtype=np.float64)[:3, 3] for name, matrix
                                            in zip(origin_game.bones, origin_game.bone_world)},
                                           skeleton, humanoid).torso_measure(),
            "male_skin_material":("Origin apply_body/sync_genital_skin card body albedo and source UVs"
                                   if raw_sex == 0 else None),
            "composite_status": "requires native skin and garment masks"})
    return path


def skin_material(native_shader: dict, albedo: CS.Map, *, source_normal: CS.Map | None = None,
                  source_normal_strength: float | None = None) -> Material:
    """The added body's material: the character's own skin program over the lent skin.

    *native_shader* is the stored record of the character's native body skin.
    The body is drawn by that very program — its ramp, its shadow table, its
    colours and constants — so it answers the light as the hands and the face
    do. The base map and optional decoded linear ``source_normal`` use the
    lent mesh's UVs. Native UV maps remain off. Source RGB normals receive an
    opaque alpha channel because the skin program reads X as R*A. Vertex
    tints match the base to native skin (``shading.VERTEX_TINT``).

    The normal's strength is the program's own ``_BumpScale``; a Principled
    program with none (a converted skin that had no normal map) draws it at
    *source_normal_strength*, what the lent body's material gives it
    (:func:`source_skin_normal_strength`, kept with the body).  Lappland's
    skin has no map of its own, and the Principled program refused the
    lent one for want of a strength (§7.1441).
    """
    from . import shading
    record = {**native_shader,
              "_rendName": "endfield_complete_body", "_material": "endfield_complete_skin",
              "maps": {prop: entry for prop, entry in native_shader["maps"].items()
                       if prop in shading.LOOKUP_MAPS},
              "keywords": [word for word in native_shader.get("keywords", [])
                           if word in shading.LOOKUP_KEYWORDS],
              shading.VERTEX_TINT: True}
    if source_normal is not None:
        pixels = np.asarray(source_normal.px)
        if (source_normal.srgb or pixels.dtype != np.uint8 or pixels.ndim != 3
                or pixels.shape[-1] not in (3, 4)):
            raise ActorError("source skin normal must be decoded linear RGB/RGBA bytes")
        pixels = np.concatenate([pixels[..., :3], np.full((*pixels.shape[:2], 1), 255, np.uint8)], axis=2)
        record["maps"]["_BumpMap"] = source_normal
        record["keywords"].append("_NORMALMAP")
        from ..gl import shaders_endfield as SE
        if (SE.FAMILIES.get(record.get("_shader")) == SE.FAMILY_PRINCIPLED
                and "_BumpScale" not in record.get("floats", {})):
            if source_normal_strength is None:
                raise ActorError("the lent body's normal map has no recorded strength and the character's "
                                 "skin program gives none; bake the body's skin again (rebake_skin)")
            record["floats"] = {**record.get("floats", {}), "_BumpScale": float(source_normal_strength)}

    def texture(entry, repeat):
        if source_normal is not None and entry is source_normal:
            return TEX.store(pixels, srgb=False)
        return _texture_key(entry, repeat)

    material, note = shading.material(record, texture)
    if material is None:
        raise ActorError(f"the character's skin material cannot be drawn: {note}")
    material.texture = TEX.store(albedo.px, srgb=albedo.srgb)
    return material


def native_skin_shader(pieces: list) -> dict | None:
    """The stored record of a character's native body skin, or ``None`` without one.

    Of *pieces* (the character's own), the skin program's material with the
    most triangles — one that is not the face's when there is one (the face
    has the SDF light map), since some characters draw their body skin with
    the face's material and have no other.
    """
    from ..gl import shaders_endfield as SE
    from . import shading
    best = {False: (0, None), True: (0, None)}
    for piece in pieces:
        shader = piece.shader or {}
        if SE.FAMILIES.get(shader.get("_shader")) != SE.FAMILY_SKIN:
            continue
        face = "_SDFLIGHTMAP" in shading.effective_keywords(shader)
        if piece.mesh.tri_count > best[face][0]:
            best[face] = (piece.mesh.tri_count, shader)
    found = best[False][1] or best[True][1]
    if found is None:
        # a converted source without this game's skin program: the native skin its conversion declared
        declared = [piece for piece in pieces if declared_role(piece) == "native_skin"]
        if declared:
            found = max(declared, key=lambda piece: piece.mesh.tri_count).shader
    return found


#: The human bones whose joints below the shoulder and the hip a limb's skin hangs on.
LIMB_BONES = tuple(side + part for side in ("Left", "Right")
                   for part in ("UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot", "Toes"))


def limb_share(mesh, skeleton, humanoid: dict, seams: list) -> np.ndarray:
    """``(vertices, 1)``: each vertex's share of dual-quaternion blending on the lent body.

    The lent skin's weights were made for a rig with helper joints at the
    elbows and knees; on the character's bones alone, the weighted mean of the
    bones' matrices pulls a bent joint's skin into the joint (a knee at 120°
    leaves faces at a tenth of their area).  Blending the bones' motions as
    rigid ones keeps that skin out — but at the shoulder and across the chest
    it bulges instead, so the share is how much of a vertex's weight is on
    limb bones (:data:`LIMB_BONES` and what hangs under them): 1 down a limb,
    falling to 0 across the shoulder and the hip into the torso.

    It also falls to 0 at each seam, over the band the seam fit moved
    (``refit_radius_m``): the native skin beside it is blended linearly, and
    the two have to move as one where they meet.
    """
    limb = np.zeros(len(skeleton.bones), dtype=bool)
    roots = {skeleton.index[humanoid[name]] for name in LIMB_BONES if name in humanoid}
    for index, bone in enumerate(skeleton.bones):            # parents come first
        limb[index] = index in roots or (bone.parent >= 0 and limb[bone.parent])
    share = np.sum(mesh.bone_wt * limb[mesh.bone_idx.astype(np.int64)], axis=1)
    position = mesh.positions.astype(np.float64)
    for seam in seams:
        ring = position[np.asarray(seam["render_native_vertices"], dtype=np.int64)]
        radius = float(seam.get("refit_radius_m") or 0.0)
        if not len(ring) or radius <= 0.0:
            continue
        near = np.full(len(position), np.inf)
        for start in range(0, len(position), 4096):
            block = position[start:start + 4096, None, :] - ring[None]
            near[start:start + 4096] = np.sqrt(np.min(np.sum(block * block, axis=2), axis=1))
        ramp = np.clip(near / radius, 0.0, 1.0)
        share *= ramp * ramp * (3.0 - 2.0 * ramp)
    return np.clip(share, 0.0, 1.0).astype(np.float32)[:, None]


def nearest_on_surface(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, reach: float,
                       candidates: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each point, the nearest place on a triangle mesh: ``(distance, barycentric (N, 3), face)``.

    Looked for among the *candidates* triangles whose centres are nearest;
    a point farther than *reach* from every one of those centres' triangles
    gets ``inf``.  The closest point on a triangle is the standard region
    test (Ericson, *Real-Time Collision Detection* 5.1.5).
    """
    from scipy.spatial import cKDTree
    tri = vertices[faces]
    count = len(points)
    best = np.full(count, np.inf)
    best_bary = np.zeros((count, 3))
    best_face = np.zeros(count, dtype=np.int64)
    if not len(faces) or not count:
        return best, best_bary, best_face
    spread = float(np.linalg.norm(tri - tri.mean(axis=1, keepdims=True), axis=2).max())
    _far, near = cKDTree(tri.mean(axis=1)).query(points, k=min(candidates, len(faces)),
                                                 distance_upper_bound=reach + spread)
    near = near.reshape(count, -1)
    for column in range(near.shape[1]):
        rows = np.flatnonzero(near[:, column] < len(faces))
        if not len(rows):
            continue
        f = near[rows, column]
        a, b, c = tri[f, 0], tri[f, 1], tri[f, 2]
        p = points[rows]
        ab, ac, ap = b - a, c - a, p - a
        d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
        bp = p - b
        d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
        cp = p - c
        d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        total = np.where(np.abs(va + vb + vc) > 1e-30, va + vb + vc, 1.0)
        v, w = vb / total, vc / total                              # the face's interior
        bary = np.stack([1.0 - v - w, v, w], axis=1)
        # the regions outside it, each overriding the one before where it applies
        edge_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        t = np.where(edge_bc, d4 - d3, 0.0) / np.where(edge_bc, (d4 - d3) + (d5 - d6), 1.0)
        bary[edge_bc] = np.stack([np.zeros(len(t)), 1.0 - t, t], axis=1)[edge_bc]
        edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = np.where(edge_ac, d2, 0.0) / np.where(edge_ac, d2 - d6, 1.0)
        bary[edge_ac] = np.stack([1.0 - t, np.zeros(len(t)), t], axis=1)[edge_ac]
        edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = np.where(edge_ab, d1, 0.0) / np.where(edge_ab, d1 - d3, 1.0)
        bary[edge_ab] = np.stack([1.0 - t, t, np.zeros(len(t))], axis=1)[edge_ab]
        bary[(d6 >= 0) & (d5 <= d6)] = (0.0, 0.0, 1.0)
        bary[(d3 >= 0) & (d4 <= d3)] = (0.0, 1.0, 0.0)
        bary[(d1 <= 0) & (d2 <= 0)] = (1.0, 0.0, 0.0)
        closest = bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:] * c
        distance = np.linalg.norm(p - closest, axis=1)
        better = distance < best[rows]
        best[rows[better]] = distance[better]
        best_bary[rows[better]] = bary[better]
        best_face[rows[better]] = f[better]
    best[best > reach] = np.inf
    return best, best_bary, best_face


def _mixed(index: np.ndarray, weight: np.ndarray, share: np.ndarray,
           their_index: np.ndarray, their_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(index, weight, rows changed)``: each row's weights with *share* of them replaced by theirs.

    The same bone on both sides is one influence; the heaviest that fit the
    row's width are kept and made to sum to one.
    """
    index, weight = index.copy(), weight.copy()
    width = index.shape[1]
    rows = np.flatnonzero(share > 0.0)
    if not len(rows):
        return index, weight, rows
    merged_index = np.concatenate([index[rows], their_index[rows]], axis=1)
    merged_weight = np.concatenate([weight[rows] * (1.0 - share[rows, None]),
                                    their_weight[rows] * share[rows, None]], axis=1)
    order = np.argsort(merged_index, axis=1, kind="stable")
    merged_index = np.take_along_axis(merged_index, order, axis=1)
    merged_weight = np.take_along_axis(merged_weight, order, axis=1)
    same = merged_index[:, 1:] == merged_index[:, :-1]
    for column in range(merged_index.shape[1] - 1):
        merged_weight[:, column + 1] += np.where(same[:, column], merged_weight[:, column], 0.0)
        merged_weight[:, column] = np.where(same[:, column], 0.0, merged_weight[:, column])
    keep = np.argsort(-merged_weight, axis=1, kind="stable")[:, :width]
    index[rows] = np.take_along_axis(merged_index, keep, axis=1)
    kept = np.take_along_axis(merged_weight, keep, axis=1)
    weight[rows] = kept / np.maximum(kept.sum(axis=1, keepdims=True), 1e-12)
    return index, weight, rows


def _seam_ramp(distance: np.ndarray, band: float) -> np.ndarray:
    t = np.clip(1.0 - distance / band, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def by_the_head(mesh, head_side: set) -> np.ndarray:
    """``(F,)`` bool: the faces of a native skin mesh that hang on the head's side.

    A face with a corner whose heaviest bone is the neck or under it.  Not
    the whole island such a face is on: Avywen's own skin is one island from
    her neck to her elbows, and counted as neck all of it her elbows were
    bound the neck's way (95 faces of the lent forearm through the native
    upper arm when the elbow bent, §7.964).
    """
    faces = mesh.indices.reshape(-1, 3).astype(np.int64)
    if not len(faces):
        return np.zeros(0, dtype=bool)
    heaviest = mesh.bone_idx[np.arange(mesh.vertex_count), np.argmax(mesh.bone_wt, axis=1)].astype(np.int64)
    return np.isin(heaviest, list(head_side))[faces].any(axis=1)


#: The joints of the limbs (elbows, wrists, knees, ankles): the segments either side of each.  A
#: wrist is one as much as an elbow: once the lent skin that follows the native skin there is skinned
#: at the character's rest (:func:`rest_twins`), EndminF's own hands carry her wrists (originact
#: 2,2 / 1,1 edge stretch p99 within 45 mm of the wrist, left: 3.3 / 3.9 → 1.3 / 1.3; the source card
#: 1.5 / 1.7, §7.1289).
HINGES = tuple((side + upper, side + lower) for side in ("Left", "Right")
               for upper, lower in (("UpperArm", "LowerArm"), ("LowerArm", "Hand"),
                                    ("UpperLeg", "LowerLeg"), ("LowerLeg", "Foot")))
#: A seam's ring lies across a hinge when at least this share of its weight is on each side of it.
#: Measured over the five bodies with limb seams (§7.1283): a ring on one segment has 0.00 on any
#: other (EndminM's forearms, Wulfa's ten palm rings, Zhuangfy's upper arms), Avywen's elbows 0.12
#: on the upper arm, EndminF's wrists 0.44 on the hand — any value between classifies alike.
HINGE_SHARE = 0.05
#: How far past a hinge rim the lent skin goes with the native skin, in seam bands.  Measured on
#: Avywen's elbows (originact 2,0 / 2,2 / 1,1, edge stretch p99 within 55 mm of the elbow, left):
#: handed the other way 8.7 / 5.8 / 2.8; one band 2.2 / 1.8 / 1.3; two 2.0 / 1.7 / 1.2; three the
#: same as two (the source card 1.6 / 1.6 / 1.3).
HINGE_REACH = 2.0


def hinge_rims(mesh, seams: list, skeleton, humanoid: dict) -> list:
    """The rings (rest positions, ``(N, 3)``) of the limb seams whose native ring lies across a hinge.

    *mesh* is the lent body with the composer's seam rings in it (the rings
    carry the native skin's weights, ``seams._native_ring``).  A ring's
    weight is put on the human bone each of its bones hangs under; it lies
    across a hinge when the segments either side of it (:data:`HINGES`)
    each hold :data:`HINGE_SHARE` of it — the character's own skin there is
    the joint, and carries its bend.
    """
    owner = np.full(len(skeleton.bones), "", dtype=object)
    human = sorted((name for name, bone in humanoid.items() if bone in skeleton.index),
                   key=lambda name: len(skeleton.descendants(humanoid[name])), reverse=True)
    for name in human:                            # the deepest human bone a bone hangs under wins
        owner[skeleton.descendants(humanoid[name])] = name
    out = []
    for seam in seams:
        ring = np.asarray(seam.get("render_native_vertices") or [], dtype=np.int64)
        if not len(ring):
            continue
        share: dict = {}
        for index, weight in zip(mesh.bone_idx[ring].ravel(), mesh.bone_wt[ring].ravel()):
            share[owner[index]] = share.get(owner[index], 0.0) + float(weight) / len(ring)
        if any(min(share.get(upper, 0.0), share.get(lower, 0.0)) >= HINGE_SHARE for upper, lower in HINGES):
            out.append(mesh.positions[ring].astype(np.float64))
    return out


def holds_its_weights(mesh, head_side: set, rims: list = (), reach: float = 0.0) -> np.ndarray:
    """``(F,)`` bool: the faces of a native skin mesh that keep their own weights at a seam.

    Those by the head (:func:`by_the_head`), and those within *reach* of a
    ring that lies across a hinge (:func:`hinge_rims`): the native skin there
    is the joint, and the lent skin it meets goes with it
    (:func:`match_seam_weights`).  Handed the other way, the elbow Avywen's
    own skin covers was bent by the lent body's helper joints on one side of
    the seam and by her own rig on the other: posed in 正常位 the bridges
    between them stretched to 9.5 times their length (§7.1281, §7.1283).
    """
    held = by_the_head(mesh, head_side)
    if not len(held) or not len(rims) or reach <= 0.0:
        return held
    from scipy.spatial import cKDTree
    near, _ = cKDTree(np.concatenate(rims)).query(mesh.positions.astype(np.float64))
    return held | (near[mesh.indices.reshape(-1, 3).astype(np.int64)] <= reach).any(axis=1)


def lend_limb_weights(lent, drawn: np.ndarray, native: list, band: float, head_side: set,
                      rims: list = ()) -> int:
    """Bind the ends of the character's own limb skin like the lent skin they meet; vertices changed.

    The neck's own skin is left as it is and the lent skin goes with it
    (:func:`match_seam_weights`): it runs on into a face.  A limb's own skin
    is the other way round.  It ends just past a joint, under what was a
    sleeve, and is bound rigid to the limb there (EndminF's forearm skin
    starts 5 cm below the elbow: ten vertices round, 0.6 forearm twist and
    0.4 forearm, no upper arm at all) — while the lent skin above it has the
    joint's own weights and helper joints.  Made to follow the native skin
    near the seam, the lent skin could only bend in what was left above the
    band, and the native rim stood out as a ledge whenever the joint bent.
    So within *band* of the lent skin that is drawn (*drawn*: its faces), a
    native vertex of an island that does not reach the head takes the lent
    skin's weights there, wholly at the seam and fading out down the limb —
    except near *rims* (:func:`hinge_rims`), where the native skin is the
    joint and keeps its own (:func:`holds_its_weights`).
    """
    faces = lent.indices.reshape(-1, 3).astype(np.int64)[drawn] if drawn.dtype == bool else drawn
    if not len(faces) or band <= 0.0:
        return 0
    skin = lent.positions.astype(np.float64)
    wide = lent.bone_idx.shape[1]
    changed = 0
    for mesh in native:
        if not mesh.vertex_count or not len(mesh.indices):
            continue
        # a vertex of a face by the head (or across a hinge) keeps its weights, whatever else it is a corner of
        every = mesh.indices.reshape(-1, 3).astype(np.int64)
        at_head = holds_its_weights(mesh, head_side, rims, band)
        rows = np.setdiff1d(np.unique(every[~at_head]), np.unique(every[at_head]))
        if not len(rows):
            continue
        distance, bary, face = nearest_on_surface(mesh.positions.astype(np.float64)[rows], skin, faces, band)
        near = np.flatnonzero(np.isfinite(distance))
        if not len(near):
            continue
        width = max(mesh.bone_idx.shape[1], wide)
        index = np.zeros((mesh.vertex_count, width), dtype=np.int64)
        weight = np.zeros((mesh.vertex_count, width))
        index[:, :mesh.bone_idx.shape[1]] = mesh.bone_idx
        weight[:, :mesh.bone_wt.shape[1]] = mesh.bone_wt
        share = np.zeros(mesh.vertex_count)
        share[rows[near]] = _seam_ramp(distance[near], band)
        corner = faces[face[near]]
        their_index = np.zeros((mesh.vertex_count, 3 * wide), dtype=np.int64)
        their_weight = np.zeros((mesh.vertex_count, 3 * wide))
        their_index[rows[near]] = lent.bone_idx.astype(np.int64)[corner].reshape(len(near), -1)
        their_weight[rows[near]] = (lent.bone_wt.astype(np.float64)[corner] * bary[near][:, :, None]).reshape(len(near), -1)
        index, weight, moved = _mixed(index, weight, share, their_index, their_weight)
        mesh.bone_idx = index.astype(mesh.bone_idx.dtype)
        mesh.bone_wt = weight.astype(mesh.bone_wt.dtype)
        changed += len(moved)
    return changed


#: How far a seam ring's lent vertex lies from the surface it is joined to, at most: the composer
#: welds the ring onto it (rest gap at most 0.04 mm over the seven bodies, PROGRESS #173).
JOINED = 1e-4


def joined_strips(meshes: list, rings: np.ndarray) -> list:
    """The faces of *meshes* (the head's surfaces) that a seam ring is joined to, one mesh each.

    *rings*: the lent body's seam-ring positions.  A face counts when one of
    its corners lies on a ring (:data:`JOINED`).  Where the lent skin is
    sewn onto the head it has to bend as that border does: Wulfa's throat is
    the bottom edge of her face mesh, on Head 0.58 / Neck 0.30 / jaw 0.12,
    and the lent neck bound to the head alone beside it opened 10.5 mm in
    originact 2,2 (:func:`match_seam_weights` takes these as skin it lies on).
    """
    from scipy.spatial import cKDTree
    if not len(rings):
        return []
    tree = cKDTree(np.asarray(rings, dtype=np.float64))
    strips = []
    for mesh in meshes:
        faces = mesh.indices.reshape(-1, 3).astype(np.int64)
        if not len(faces):
            continue
        near = np.isfinite(tree.query(mesh.positions.astype(np.float64), distance_upper_bound=JOINED)[0])
        on = near[faces].any(axis=1)
        if on.any():
            strip = mesh.copy()
            strip.indices = faces[on].ravel().astype(mesh.indices.dtype)
            strips.append(strip)
    return strips


def match_seam_weights(mesh, native: list, head: list, head_bone: int, band: float,
                       head_side: set | None = None) -> int:
    """Bind the lent skin like what it lies against, near it; returns how many vertices changed.

    The lent skin keeps its source's weights, and where it lies under the
    character's own skin the two are weighted differently (a native neck at
    Neck 0.91 / Head 0.09 over a lent one at 0.95 / 0.05), so they part when
    the head turns: 6 mm against the neck skin and 12 mm against the face at
    40° (§7.925).  Within *band* of a native skin vertex (*native*: meshes) a
    lent vertex takes that vertex's weights, wholly at the surface and fading
    out across the band; within *band* of the head's own surfaces (*head*) it
    goes with *head_bone* alone — the face hangs on facial joints, and what is
    tucked inside the head has to turn with the head, not with a cheek.
    *head_side* (bone indices: the neck and everything under it) limits that
    to lent vertices whose own heaviest bone is one of them — hair that hangs
    down the chest is a head surface too, and without the limit the chest
    under it was hung on the head (a sheet of skin pulled after the head
    whenever it turned from the chest, §7.944).
    Native skin is applied last: where both are near, the skin it touches wins.
    The heaviest influences that fit the mesh are kept.
    """
    from scipy.spatial import cKDTree
    position = mesh.positions.astype(np.float64)
    count, width = mesh.bone_idx.shape
    index = mesh.bone_idx.astype(np.int64)
    weight = mesh.bone_wt.astype(np.float64)
    changed = np.zeros(count, dtype=bool)

    def blend(share, their_index, their_weight):
        nonlocal index, weight
        index, weight, rows = _mixed(index, weight, share, their_index, their_weight)
        changed[rows] = True

    def ramp(distance):
        return _seam_ramp(distance, band)

    def drawn(m):
        # a part split by role keeps every vertex of its piece; only the ones its own faces use are
        # its surface (a leg given to the outfit is not skin the lent thigh lies against)
        return np.unique(m.indices.astype(np.int64))

    points = [m.positions.astype(np.float64)[drawn(m)] for m in head if m.vertex_count and len(m.indices)]
    if points and band > 0.0:
        distance, _near = cKDTree(np.concatenate(points)).query(position)
        alone_index = np.zeros((count, width), dtype=np.int64)
        alone_index[:, 0] = head_bone
        alone_weight = np.zeros((count, width))
        alone_weight[:, 0] = 1.0
        share = ramp(distance)
        if head_side is not None:
            heaviest = index[np.arange(count), np.argmax(weight, axis=1)]
            share = np.where(np.isin(heaviest, list(head_side)), share, 0.0)
        blend(share, alone_index, alone_weight)
    meshes = [m for m in native if m.vertex_count and len(m.indices)]
    if meshes and band > 0.0:
        theirs = np.concatenate([m.positions.astype(np.float64)[drawn(m)] for m in meshes])
        wide = max(m.bone_idx.shape[1] for m in meshes)
        their_index = np.concatenate([np.pad(m.bone_idx.astype(np.int64)[drawn(m)],
                                             ((0, 0), (0, wide - m.bone_idx.shape[1]))) for m in meshes])
        their_weight = np.concatenate([np.pad(m.bone_wt.astype(np.float64)[drawn(m)],
                                              ((0, 0), (0, wide - m.bone_wt.shape[1]))) for m in meshes])
        # against the skin's surface, not its vertices: a native limb is a dozen vertices round,
        # and a lent vertex lying on a face of it is centimetres from every corner — it got a
        # share of the native weights for that distance, and a ledge stood at the rim of the
        # native skin when the limb bent (§7.958)
        local = np.cumsum([0] + [len(drawn(m)) for m in meshes])
        corners = []
        for k, m in enumerate(meshes):
            rows = drawn(m)
            where = np.full(m.vertex_count, -1, dtype=np.int64)
            where[rows] = np.arange(len(rows)) + local[k]
            corners.append(where[m.indices.reshape(-1, 3).astype(np.int64)])
        corners = np.concatenate(corners)
        distance, bary, face = nearest_on_surface(position, theirs, corners, band)
        near = np.flatnonzero(np.isfinite(distance))
        share = np.zeros(count)
        share[near] = ramp(distance[near])
        at = corners[face[near]]                                  # (M, 3) native vertices
        spread_index = np.zeros((count, 3 * wide), dtype=np.int64)
        spread_weight = np.zeros((count, 3 * wide))
        spread_index[near] = their_index[at].reshape(len(near), -1)
        spread_weight[near] = (their_weight[at] * bary[near][:, :, None]).reshape(len(near), -1)
        blend(share, spread_index, spread_weight)
    mesh.bone_idx = index.astype(mesh.bone_idx.dtype)
    mesh.bone_wt = weight.astype(mesh.bone_wt.dtype)
    return int(changed.sum())


#: Off only to measure the stored skinning against (a ruler's switch).
UNBLEND = True


#: What a kept joint's twin is named after it (:func:`rest_twins`): still a :data:`GRAFT` node, so
#: whatever folds the kept joints back onto the character's bones folds the twin too, and the
#: stand-in, which drives a kept joint by its source name, never finds one.
REST = "@rest"


def twin_nodes(nodes: dict) -> dict:
    """``{twin: (kept joint, identity)}`` for every kept joint in *nodes* without its twin (:func:`rest_twins`)."""
    eye = np.eye(4).tolist()
    return {name + REST: (name, eye) for name in nodes
            if name.startswith(GRAFT) and not name.endswith(REST) and name + REST not in nodes}


def rest_twins(skeleton, meshes: list, piece, zones: list, band: float) -> int:
    """Skin the lent skin near the seams at the character's rest, through the kept joints' twins; vertices changed.

    The lent skin is blended once from the source's rest (:func:`unblend`):
    each kept joint is bound where the source binds it, and a vertex sits
    where those binds put it.  The character's own bones are bound at her
    rest.  A vertex weighted to both (the lent skin following the native skin
    at a seam, :func:`match_seam_weights`; the native skin taking the lent
    skin's weights, :func:`lend_limb_weights`) blends matrices taken from two
    rests, and when its bones move apart it leaves both shapes — the 2–7 mm
    bridges at EndminF's wrists stretched 4.4 times, and her wrist p99 was
    3.3 rolled 95° (originact 2,2).  Each kept joint has a twin
    (:func:`twin_nodes`: a child at the identity, so it moves exactly as the
    joint does) that keeps the character's rest as its bind; near a seam a
    vertex's kept-joint weight goes to the twins.  *zones*: ``[(ring points,
    reach)]`` — wholly within *reach* of a ring, fading to none over *band*
    past it.  Seams at the neck are left out: there it made EndminF's neck
    worse (2,2 p99 3.85 → 4.22).
    """
    from scipy.spatial import cKDTree

    from ..mesh.skin import MAX_INFLUENCES
    twin = np.arange(len(skeleton.bones))
    for name in piece.bones[:len(piece.binds)]:
        joint, double = GRAFT + name, GRAFT + name + REST
        if skeleton.has(joint) and skeleton.has(double):
            twin[skeleton.index[joint]] = skeleton.index[double]
    has_twin = twin != np.arange(len(skeleton.bones))
    if not has_twin.any() or not len(zones) or band <= 0.0:
        return 0
    trees = [(cKDTree(points), reach) for points, reach in zones]
    changed = 0
    for mesh in meshes:
        if not mesh.vertex_count:
            continue
        position = mesh.positions.astype(np.float64)
        share = np.zeros(mesh.vertex_count)
        for tree, reach in trees:
            distance, _ = tree.query(position)
            share = np.maximum(share, _seam_ramp(np.maximum(distance - reach, 0.0), band))
        index = mesh.bone_idx.astype(np.int64)
        weight = mesh.bone_wt.astype(np.float64)
        on = (weight > 0) & has_twin[index]
        rows = np.flatnonzero((share > 0) & on.any(axis=1))
        if not len(rows):
            continue
        part = share[rows, None]
        both_index = np.concatenate([index[rows], np.where(on[rows], twin[index[rows]], 0)], axis=1)
        both_weight = np.concatenate([np.where(on[rows], weight[rows] * (1.0 - part), weight[rows]),
                                      np.where(on[rows], weight[rows] * part, 0.0)], axis=1)
        order = np.argsort(-both_weight, axis=1, kind="stable")[:, :MAX_INFLUENCES]
        kept_index = np.take_along_axis(both_index, order, axis=1)
        kept_weight = np.take_along_axis(both_weight, order, axis=1)
        kept_weight /= np.maximum(kept_weight.sum(axis=1, keepdims=True), 1e-12)
        width = kept_index.shape[1]
        out_index = np.zeros((mesh.vertex_count, width), dtype=np.int64)
        out_weight = np.zeros((mesh.vertex_count, width))
        out_index[:, :index.shape[1]], out_weight[:, :weight.shape[1]] = index, weight
        out_index[rows], out_weight[rows] = kept_index, kept_weight
        mesh.bone_idx = out_index.astype(mesh.bone_idx.dtype)
        mesh.bone_wt = out_weight.astype(mesh.bone_wt.dtype)
        changed += len(rows)
    return changed


def unblend(skeleton, mesh, piece) -> int:
    """Have the lent skin blended once, from the source's rest; how many of its joints bind elsewhere now.

    The lent body is the source body carried onto this rig joint by joint:
    each source joint has a rigid fit (``fit_transforms``), and the stored
    rest shape is the source's skin with those fits blended by its weights —
    one linear blend already, that puts a T-posed body's arms where this
    rig's hang.  Posing that shape blends a second time, and two blends are
    not one: skinned as stored, the shoulders of the four bodies are up to
    2 cm from where one blend from the source's rest puts them (kneeling,
    arms raised; §7.958).

    So the skin is given the one blend.  Each kept joint is bound where the
    *source* binds it (*piece*'s own bind) instead of where the fit stands
    it, and every vertex is stored where those binds, blended by its weights,
    put it exactly at its rest position (``Mesh.skin_positions``).  At rest
    nothing moves; posed, a vertex goes by each of its joints from the
    source's own shape.  The character's own bones keep their binds, so a
    vertex weighted to them alone is where it was.

    Only for linear skinning: a rigid blend of binds that are not the rest is
    another shape at rest.
    """
    count = len(skeleton.bones)
    fit = np.tile(np.eye(4), (count, 1, 1))
    bound: dict[int, np.ndarray] = {}
    for k, name in enumerate(piece.bones[:len(piece.binds)]):
        if skeleton.has(GRAFT + name):
            joint = skeleton.index[GRAFT + name]
            source_bind = np.asarray(piece.bone_world[k], dtype=np.float64)
            fit[joint] = skeleton.bones[joint].bind_world @ np.linalg.inv(source_bind)
            bound[joint] = np.linalg.inv(source_bind)
    if not bound:
        return 0
    index = mesh.bone_idx.astype(np.int64)
    blended = np.einsum("vk,vkij->vij", mesh.bone_wt.astype(np.float64), fit[index])
    back = np.linalg.inv(blended)
    rest = np.c_[mesh.positions.astype(np.float64), np.ones(mesh.vertex_count)]
    mesh.skin_positions = np.einsum("vij,vj->vi", back, rest)[:, :3].astype(np.float32)

    def carried(normals) -> np.ndarray:
        # a normal goes by the inverse transpose of what its surface goes by
        normal = np.einsum("vji,vj->vi", blended[:, :3, :3], normals.astype(np.float64))
        return (normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    mesh.skin_normals = carried(mesh.normals)
    if len(mesh.outline_normals) == mesh.vertex_count:
        mesh.skin_outline_normals = carried(mesh.outline_normals)      # the outline's averaged normal goes the same way
    for joint, inverse in bound.items():
        skeleton.bind_skin(joint, inverse)
    return len(bound)


#: The stretches of a body a chain of hair can lie against: (the bone the capsule rides, from, to),
#: by human bone.  The shoulder line rides the chest and runs from one upper arm's root to the other's.
BODY_SEGMENTS = (
    ("UpperChest", "LeftUpperArm", "RightUpperArm"),
    ("Spine", "Spine", "Neck"),
    ("Neck", "Neck", "Head"),
    ("LeftUpperArm", "LeftUpperArm", "LeftLowerArm"), ("LeftLowerArm", "LeftLowerArm", "LeftHand"),
    ("RightUpperArm", "RightUpperArm", "RightLowerArm"), ("RightLowerArm", "RightLowerArm", "RightHand"),
    ("LeftUpperLeg", "LeftUpperLeg", "LeftLowerLeg"), ("LeftLowerLeg", "LeftLowerLeg", "LeftFoot"),
    ("RightUpperLeg", "RightUpperLeg", "RightLowerLeg"), ("RightLowerLeg", "RightLowerLeg", "RightFoot"),
)


#: The stretches another body's cloth meets: its own (:data:`BODY_SEGMENTS`) and the pelvis, which a tail
#: hanging behind a woman meets on the man behind her (Wulfa, 壁バック, §7.1074)
OTHER_BODY_SEGMENTS = BODY_SEGMENTS + (("Hips", "Hips", "Spine"),)


def body_colliders(skeleton, humanoid: dict, mesh, drawn: np.ndarray, stretches=BODY_SEGMENTS) -> list:
    """Capsules for a lent body's own shape, as bone cloth colliders (``cloth.Collider``).

    One per stretch of *stretches* the Avatar names.  Its radius is
    measured, not chosen: the median distance from the stretch's axis of the
    lent skin's drawn vertices (*drawn*: the mesh's faces that are ever
    drawn) that lie alongside it — nearer that axis than any other stretch's
    and between its two ends.  A stretch with no such vertices gets no capsule.
    """
    from .cloth import CAPSULE, Collider
    bind = np.stack([bone.bind_world for bone in skeleton.bones])
    position = mesh.positions.astype(np.float64)
    shown = np.zeros(mesh.vertex_count, dtype=bool)
    shown[mesh.indices.reshape(-1, 3).astype(np.int64)[drawn]] = True
    points = position[shown]
    segments = []
    for rider, start, end in stretches:
        names = [humanoid.get(rider), humanoid.get(start), humanoid.get(end)]
        if all(name and skeleton.has(name) for name in names):
            a, b = (bind[skeleton.index[name]][:3, 3] for name in names[1:])
            if np.linalg.norm(b - a) > 1e-6:
                segments.append((skeleton.index[names[0]], a, b))
    if not segments or not len(points):
        return []
    along, away = [], []
    for _rider, a, b in segments:
        axis = b - a
        t = (points - a) @ axis / float(axis @ axis)
        along.append(t)
        away.append(np.linalg.norm(points - (a + np.clip(t, 0.0, 1.0)[:, None] * axis), axis=1))
    nearest = np.argmin(np.stack(away), axis=0)
    out = []
    for k, (rider, a, b) in enumerate(segments):
        beside = (nearest == k) & (along[k] >= 0.0) & (along[k] <= 1.0)
        if not beside.any():
            continue
        radius = float(np.median(away[k][beside]))
        into = np.linalg.inv(bind[rider])
        out.append(Collider(CAPSULE, rider, (into @ np.append(a, 1.0))[:3], (into @ np.append(b, 1.0))[:3],
                            radius, radius, source_id="lent body"))
    return out


#: How deep under the lent skin a head part's vertex may lie and still be lifted onto it, and how
#: far above the skin it is put.  Hair hangs on the body it was made for; the lent body is fuller in
#: places (Avywen's locks lie up to 7 mm inside her chest, §7.954).  Deeper than this it is not
#: resting on the skin but passing through something.
LIFT_REACH = 0.02
LIFT_CLEAR = 0.001


def resting_parts(parts, sewn=frozenset()) -> list:
    """The head's *parts* that rest on the skin, lifted out of it (:func:`lift_onto_body`, ``hair_lift``): all but its own body skin.

    Hair and the like lie on the body and are lifted out of the lent skin.
    The character's own body skin that the composer gave to the head (her
    collar, ``given_skin_faces``) is skin sewn to the lent skin at the head
    seam, not something lying on it: lifted, Typhoea's collar stood 1.0 mm
    proud of the lent chest all along its open edge (18 vertices, 0.92–1.00
    mm), and the pose-by-pose lift (``hair_lift``, 4 mm) held it 4.0 mm off
    in every pose: background showed through at the V and its corners with
    the outline off (20 px), and the outline filled that gap as a line round
    the neck (#161).  Body skin is the skin program without the face's light
    map (``skin_match.body_skin``).  *sewn* names the parts the head seam
    is sewn to (:func:`sewn_head_parts`): a converted source's declared face
    carries her neck down to the seam, and lifted there it stood 4.0 mm off the
    lent neck all round in every pose — a ring 25 mm wider than the neck below
    it (Lappland, 201.6 against 177.0 mm).
    """
    from .skin_match import body_skin
    skin = {id(part) for part in body_skin(parts)}
    return [part for part in parts if id(part) not in skin and str(part.name).split("|", 1)[0] not in sewn]


def sewn_head_parts(character: str) -> frozenset:
    """``rendName#slot`` of every part of *character* the composite's head seams are sewn to."""
    seams = (((store.read_extra("body", character) or {}).get("composite") or {}).get("seams") or {})
    # the head surface's own piece in the character's item (``piece_index`` counts head surfaces)
    rows = [int(row.get("source_piece_index", -1)) for row in seams.get("head_seams") or []]
    rows = [index for index in rows if index >= 0]
    pieces = store.read_item("chara", character).pieces
    from .part_roles import DECLARED_ROLE, HEAD_SKIN_CONTINUATION
    rows.extend(index for index, piece in enumerate(pieces)
                if piece.shader.get(DECLARED_ROLE) == "head"
                and piece.shader.get(HEAD_SKIN_CONTINUATION) is True)
    return frozenset(f"{pieces[index].shader['_rendName']}#{pieces[index].mat_slot}" for index in rows)


def lift_onto_body(body, parts: list, head_side: set) -> int:
    """Move the vertices of *parts* that lie just under the lent skin out onto it; how many moved.

    *parts* are the character's head surfaces (hair, mostly); only the skin
    away from the head counts (*head_side*: the neck's bones and below it in
    the tree) — inside the head the lent neck is meant to be under them.  A
    vertex within :data:`LIFT_REACH` under the nearest such skin vertex, by
    that vertex's normal, is put :data:`LIFT_CLEAR` above it.
    """
    from scipy.spatial import cKDTree
    heaviest = body.bone_idx[np.arange(body.vertex_count), np.argmax(body.bone_wt, axis=1)].astype(np.int64)
    away = np.flatnonzero(~np.isin(heaviest, list(head_side)))
    if not len(away):
        return 0
    skin = body.positions.astype(np.float64)[away]
    normal = body.normals.astype(np.float64)[away]
    tree = cKDTree(skin)
    moved = 0
    for mesh in parts:
        if not mesh.vertex_count or not len(mesh.indices):
            continue
        used = np.unique(mesh.indices.astype(np.int64))
        at = mesh.positions.astype(np.float64)[used]
        distance, near = tree.query(at, distance_upper_bound=LIFT_REACH)
        found = np.isfinite(distance)
        height = np.einsum("ij,ij->i", at[found] - skin[near[found]], normal[near[found]])
        under = (height < LIFT_CLEAR) & (height > -LIFT_REACH)
        if not under.any():
            continue
        rows = used[found][under]
        lifted = at[found][under] + normal[near[found]][under] * (LIFT_CLEAR - height[under])[:, None]
        positions = mesh.positions.copy()
        positions[rows] = lifted.astype(positions.dtype)
        mesh.positions = positions
        moved += len(rows)
    return moved


def dual_share(character: str, mesh) -> np.ndarray:
    """:func:`limb_share` for a stored completed body's render mesh."""
    native = store.read_item("chara", character)
    body_item = store.read_item("body", character)
    extra = store.read_extra("chara", character) or {}
    manifest = (store.read_extra("body", character) or {}).get("composite") or {}
    skeleton = build_skeleton(native.nodes, character)
    append_body_nodes(skeleton, body_item.nodes)
    seams = manifest.get("seams") or {}
    return limb_share(mesh, skeleton, extra.get("humanoid") or {},
                      list(seams.get("seams", [])) + list(seams.get("head_seams", [])))


def load_body_part(character: str, skeleton, native_shader: dict | None = None) -> Part:
    """Load the fitted body for an Endfield actor, with the selected Origin skin.

    With the character's *native_shader* it is drawn by that skin program
    (:func:`skin_material`); without one — a character with no native body
    skin to take after — by the plain material over the lent skin.
    """
    require_body_scope(store.read_extra("chara", character) or {}, character)
    item = store.read_item("body", character)
    if item is None or not item.pieces or item.maps.get("albedo") is None:
        raise ActorError(f"body/{character} is not baked")
    mesh = _rest_mesh(item.pieces[0], skeleton)
    from .skin_color import restore_tints
    restore_tints(mesh, (store.read_extra('body', character) or {}).get('composite'))
    if native_shader is not None:
        strength = (store.read_extra("body", character) or {}).get("source_normal_strength")
        material = skin_material(native_shader, item.maps["albedo"], source_normal=item.maps.get("normal"),
                                 **({} if strength is None else {"source_normal_strength": strength}))
    else:
        material = Material(name="endfield_complete_skin")
        material.texture = TEX.store(item.maps["albedo"].px,
                                     srgb=item.maps["albedo"].srgb)
        if item.maps.get("normal") is not None:
            material.normal_map = TEX.store(item.maps["normal"].px, srgb=False)
    return Part("endfield_complete_body", mesh, material=material,
                group="endfield_body")


def load_complete_actor(character: str):
    """Load the original actor and added body under validated composite rules.

    This is a separate source choice. Missing bodies raise an error; a clothed
    prefab is never silently substituted for a requested complete body.
    """
    require_body_scope(store.read_extra("chara", character) or {}, character)
    body_extra = store.read_extra("body", character) or {}
    manifest = body_extra.get("composite")
    if manifest is None:
        raise ActorError(f"body/{character}: native skin and outfit composite metadata is missing")
    if manifest.get('coverage_verified') is not True or manifest.get('seams', {}).get('unresolved'):
        raise ActorError(f"body/{character}: garment coverage or native seams require validation")
    from .seams import COMPOSITION_REVISION
    seams = manifest.get('seams') or {}
    if seams.get('accepted') is not True or seams.get('revision') != COMPOSITION_REVISION:
        raise ActorError(f"body/{character}: current seam certification is missing; rebuild this body")
    from .certification import require_current
    require_current(character, manifest)
    actor = load_actor("chara", character)
    body_item = store.read_item("body", character)
    if body_item is None:
        raise ActorError(f"body/{character}: fitted source is missing")
    append_body_nodes(actor.skeleton, body_item.nodes)
    native = store.read_item("chara", character)
    body = load_body_part(character, actor.skeleton,
                          native_skin_shader(native.pieces if native is not None else []))
    # down the limbs the lent skin is blended as rigid motions (``limb_share``)
    from ..mesh import skinning
    extra = store.read_extra("chara", character) or {}
    body.mesh.skin_dual = limb_share(body.mesh, actor.skeleton, extra.get("humanoid") or {},
                                     list(seams.get("seams", [])) + list(seams.get("head_seams", [])))
    body.skinning = skinning.MIXED
    #: whether the lent skin hangs on its source's own joints (``keep_helpers``): a clip that
    #: moves them is then skinned the way their game skins, linearly (``ActorController``)
    actor.kept_joints = bool(body_extra.get("helper_joints"))
    if getattr(actor.model, "skeleton", None) is None:
        actor.model.skeleton = actor.skeleton
    from .composite import CompositeVisibility, split_original_parts
    outfit = split_original_parts(actor, manifest)
    rows = list(seams.get("seams", [])) + list(seams.get("head_seams", []))
    head_node = (extra.get("humanoid") or {}).get("Head")
    if rows and head_node in actor.skeleton.index:
        # near the character's own skin the lent skin is weighted like it; the band is the
        # one the seam tints fade over (``_retinted``)
        own = {group: [part.mesh for part in actor.model.parts if part.group == group]
               for group in ("endfield_native_skin", "endfield_head")}
        neck = (extra.get("humanoid") or {}).get("Neck")
        head_side = set(actor.skeleton.descendants(neck if neck in actor.skeleton.index else head_node))
        band = max(row["bridge_band_m"] for row in rows) * 8
        shown = np.ones(body.mesh.tri_count, dtype=bool)
        shown[manifest["native_excluded_faces"]] = False
        limbs = actor.kept_joints and UNBLEND             # a limb's own skin needs the kept joints' binds
        necks = own["endfield_native_skin"]
        rims = hinge_rims(body.mesh, list(seams.get("seams", [])), actor.skeleton, extra.get("humanoid") or {})
        hinges = []
        if limbs:
            necks = []
            for mesh in own["endfield_native_skin"]:
                faces = mesh.indices.reshape(-1, 3)
                at_head = by_the_head(mesh, head_side)
                neck_only = mesh.copy()
                neck_only.indices = faces[at_head].ravel().copy()
                necks.append(neck_only)
                hinge = mesh.copy()
                hinge.indices = faces[holds_its_weights(mesh, head_side, rims, band) & ~at_head].ravel().copy()
                hinges.append(hinge)
        if any(len(mesh.indices) for mesh in hinges):
            # the lent skin past a hinge the native skin covers goes with it over the reach of the joint
            match_seam_weights(body.mesh, hinges, [], actor.skeleton.index[head_node], band * HINGE_REACH, head_side)
        rings = body.mesh.positions[np.concatenate([np.asarray(row["render_native_vertices"], dtype=np.int64)
                                                    for row in rows])]
        # the head's own border where the lent skin is sewn onto it is skin it lies on
        necks = list(necks) + joined_strips(own["endfield_head"], rings)
        match_seam_weights(body.mesh, necks, own["endfield_head"],
                           actor.skeleton.index[head_node], band, head_side)
        if limbs:
            # after the lent skin has taken the neck's weights: where the two rules meet (the foot of
            # a neck) the native skin copies what the lent skin ended with, and the seam stays shut
            lend_limb_weights(body.mesh, shown, own["endfield_native_skin"], band, head_side, rims)
        lift_onto_body(body.mesh, [part.mesh for part in resting_parts(
            [part for part in actor.model.parts if part.group == "endfield_head"],
            sewn_head_parts(character))], head_side)
    source = store.read_item("body_source", character) if actor.kept_joints and UNBLEND else None
    lent = next((piece for piece in (source.pieces if source is not None else [])
                 if piece.mesh.name == "o_body_a"), None)
    if lent is not None and rows and head_node in actor.skeleton.index:
        # near a limb seam the lent skin is skinned at the character's rest, like the native skin it meets
        hinge = None
        if rims:
            from scipy.spatial import cKDTree
            hinge = cKDTree(np.concatenate(rims))
        heaviest = body.mesh.bone_idx[np.arange(body.mesh.vertex_count), np.argmax(body.mesh.bone_wt, axis=1)]
        zones = []
        for row in seams.get("seams", []):
            ring = np.asarray(row["render_native_vertices"], dtype=np.int64)
            if not len(ring) or np.isin(heaviest[ring], list(head_side)).mean() >= 0.5:
                continue
            points = body.mesh.positions[ring].astype(np.float64)
            across = hinge is not None and bool(np.all(hinge.query(points)[0] <= 1e-6))
            zones.append((points, band * (HINGE_REACH if across else 1.0)))
        rest_twins(actor.skeleton, [body.mesh] + [part.mesh for part in actor.model.parts
                                                  if part.group == "endfield_native_skin"], lent, zones, band)
    # the lent skin, her own and the skin given to her head were one mesh in the game (``body_01``): one
    # averaged normal for the outline across the welds (PROGRESS #161), made before the skin is unblended
    from ..mesh import outline_normals
    from .skin_match import body_skin
    # (the lent skin by the faces it ever draws: those under her own skin are not that mesh's, ``composite``)
    welded = [body, *(part for part in actor.model.parts if part.group == "endfield_native_skin"),
              *body_skin([part for part in actor.model.parts if part.group == "endfield_head"])]
    welded = [part for part in welded if getattr(part.material, "outline_average", False)]
    drawn = {id(body.mesh): np.delete(body.mesh.indices.reshape(-1, 3), manifest["native_excluded_faces"], axis=0)}
    outline_normals.fill([part.mesh for part in welded], outline_normals.SEAM_WELD,
                         faces=[drawn.get(id(part.mesh)) for part in welded])
    if lent is not None and unblend(actor.skeleton, body.mesh, lent):
        from ..mesh import skinning
        body.skinning = skinning.LINEAR
        for part in actor.model.parts:
            if part.group == "endfield_native_skin":
                unblend(actor.skeleton, part.mesh, lent)       # its limb ends hang on kept joints now
    actor.model.add(body)
    # the tail is the character's own, as hair is: not undressed, not listed with the clothes
    from .part_roles import tail_parts
    tails = tail_parts(outfit, actor.skeleton) | declared_tails(outfit, extra.get("roles") or {})
    outfit = [part for index, part in enumerate(outfit) if index not in tails]
    actor.composite_visibility = CompositeVisibility(body, outfit, manifest)
    lo, hi = actor.model.bounds()
    actor.height = float(hi[1] - lo[1])
    return actor
