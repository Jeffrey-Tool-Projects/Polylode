"""Origin ``Lookat_dan`` — aim male shaft bones at a female null (not root slide).

Transcribed from ``Assembly-CSharp`` (``Lookat_dan`` / ``LookAtDankonAdd`` /
``HSceneProc.ChangeAnimator``) and ``abdata/h/list`` TextAssets (§7.117).

Illusion places male and female on the **same** world root.  Insertion
fine-tuning is this component:

1. ``ChangeAnimator`` loads ``h/list/{fileLookAt}`` (e.g. ``dan_khs_00``), the
   action row's ``paramMale.fileLookAt`` — rows keyed by state name (``WLoop``,
   ``Insert``, …).  The import side bakes the action → list map and the lists'
   lines into ``h_anim/lookat.json`` (``hlist.bake_lookat``); this module reads
   only that file (§7.501).
2. Each LateUpdate (``SetLookAt``): keep the row while the first female's base
   state is still its motion, else take the row named exactly by that state,
   and keep the previous row when none is.
3. Aim ``cm_J_dan101_00`` (base) at the female null (often ``k_f_kokan_00``):
   the row's ``lookAtCalcKind`` 0 through the component's own ``LookAt``
   (up vector from ``cm_J_dan100_00``), anything else ``Transform.LookAt``
   (world up) — :func:`aim_up_vector`, §7.510.
4. If ``bTopStick``: set ``cm_J_dan109_00`` world position to that null.

We apply the same math into our additive pose dict after sampling Origin clips.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..core import math3d as m3
from ..core import swallowed

# Defaults from ``LookAtDankonAdd.get_ArgsDefault`` (§7.117).
DEFAULT_BASE = "cm_J_dan101_00"
DEFAULT_TOP = "cm_J_dan109_00"
DEFAULT_REF = "cm_J_dan100_00"

#: ``Lookat_dan.lookat`` as level18's two components (``lookDan``,
#: ``lookDan1``) serialise it, both alike (§7.510): ``targetAxisType`` 2 (Z),
#: ``upAxisType`` 1 (Y), ``sourceAxisType`` 1 (Y), ``limitAxisType`` 6 (none,
#: so ``rotOrder`` / ``limitMin`` / ``limitMax`` are never read).  With them
#: ``LookAt.Update`` is: ``f = normalize(target - base)``, ``r =
#: normalize(cross(up, f))``, ``u = cross(f, r)``, base rotation = the basis
#: ``(r, u, f)`` — the base's +Z at the null — where ``up`` is ``trfUpAxis.up``,
#: the transform ``SetMale`` found by ``nameRefBone``.
LOOKAT_UP_BONE = DEFAULT_REF

#: Null names a skeleton lacked, each printed once (:func:`resolve_null_bone`).
_MISSING_NULLS: set[str] = set()


@dataclass
class Woman:
    """One entry of ``Lookat_dan.females``: what a row's ``numFemale`` indexes.

    *shape* is her ``ChaControl.GetShapeBodyValue`` (``None`` for a
    character with no card, whose ``lstShape`` entries then add nothing).
    """
    pose: dict
    xf: np.ndarray
    skel: object
    shape: object = None          # callable(index) -> float | None


@dataclass
class ShapeOffset:
    shape: int
    min_pos: np.ndarray
    max_pos: np.ndarray


@dataclass
class MotionLookAt:
    """One row of a ``dan_*`` list (``MotionLookAtList``)."""
    motion: str
    num_female: int = 0
    look_at_null: str = ""
    top_stick: bool = False
    look_at_calc_kind: int = 0
    shapes: list[ShapeOffset] = field(default_factory=list)


@dataclass
class DanLookList:
    """Parsed ``dan_*`` TextAsset: motion name → settings."""
    name: str
    by_motion: dict[str, MotionLookAt] = field(default_factory=dict)
    #: the list's first line's motion: ``LoadList`` ends with ``SetInfo(lstLookAt[0])`` (IL_0254–0271)
    first_motion: str = ""

    @property
    def first(self) -> MotionLookAt | None:
        """``lstLookAt[0]``: the row a freshly loaded list aims by until a state names another."""
        return self.by_motion.get(self.first_motion) if self.first_motion else None

    def row_for(self, state: str, current: MotionLookAt | None) -> MotionLookAt | None:
        """``Lookat_dan.SetLookAt``: the row for base state *state*, else *current*.

        Origin asks the first female's layer-0 state ``IsName`` of the playing
        row's motion, then of each row's; a state no row names leaves the
        previous row in place.  *state* is the base state name
        (``HRuntime.base_state``), compared exactly: the leaf a size tree
        plays (``L_WLoop1``) is not a state, and neither is a substring.
        """
        if current is not None and state == current.motion:
            return current
        return self.by_motion.get(state, current) if state else current


def _parse_dan_text(name: str, text: str) -> DanLookList:
    out = DanLookList(name=name)
    for line in text.replace("\r\n", "\n").split("\n"):
        if not line.strip():
            continue
        cols = line.split("\t")
        motion = (cols[0] if cols else "").strip()
        if not motion:
            continue
        num_f = 0
        try:
            num_f = int((cols[1] if len(cols) > 1 else "0") or "0")
        except ValueError:
            num_f = 0
        null = (cols[2] if len(cols) > 2 else "").strip()
        top_stick = (cols[3] if len(cols) > 3 else "") == "1"
        kind = 0
        try:
            kind = int((cols[4] if len(cols) > 4 else "0") or "0")
        except ValueError:
            kind = 0
        shapes: list[ShapeOffset] = []
        i = 5
        while i + 6 < len(cols):
            raw_id = (cols[i] or "").strip()
            if not raw_id:
                break
            try:
                sid = int(raw_id)
            except ValueError:
                break
            try:
                mn = np.array([float(cols[i + 1]), float(cols[i + 2]),
                               float(cols[i + 3])], dtype=np.float64)
                mx = np.array([float(cols[i + 4]), float(cols[i + 5]),
                               float(cols[i + 6])], dtype=np.float64)
            except (TypeError, ValueError, IndexError):
                break
            shapes.append(ShapeOffset(sid, mn, mx))
            i += 7
        if not out.first_motion:
            out.first_motion = motion
        out.by_motion[motion] = MotionLookAt(
            motion=motion, num_female=num_f, look_at_null=null,
            top_stick=top_stick, look_at_calc_kind=kind, shapes=shapes)
    return out


@lru_cache(maxsize=1)
def _baked() -> dict:
    """``h_anim/lookat.json`` (``hlist.bake_lookat``), or ``{}`` before it is baked."""
    import json

    from .hanim import LOOKAT_FILE
    from .hlist import LOOKAT_VERSION
    if not LOOKAT_FILE.is_file():
        return {}
    try:
        d = json.loads(LOOKAT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        swallowed.note("origin.lookat_dan._baked", exc)
        return {}
    return d if int(d.get("version") or 0) == LOOKAT_VERSION else {}


def baked() -> bool:
    """Whether the Lookat_dan lists have been converted."""
    return bool(_baked())


def list_for_action(mode: int, action_id: int, *, second_male: bool = False) -> str:
    """The action row's ``dan_*`` list, ``""`` for none (or not baked).

    ``paramMale.fileLookAt`` for the man (``lookDan``); with *second_male* the
    MMF rows' ``paramFemale1.fileDynamicBoneRef`` (``lookDan1``, §7.505).
    """
    key = "actions_male1" if second_male else "actions"
    return str((_baked().get(key) or {}).get(f"{mode}/{action_id}") or "")


@lru_cache(maxsize=64)
def load_dan_list(name: str) -> DanLookList | None:
    """The ``dan_*`` list *name* from ``lookat.json``, or None when it has no lines."""
    lines = (_baked().get("lists") or {}).get((name or "").strip())
    if not lines:
        return None
    return _parse_dan_text(name, "\n".join(lines))


@lru_cache(maxsize=1)
def null_names() -> frozenset:
    """Every null a baked ``dan_*`` row aims at (the rows' ``lookAtNull``).

    The H bake keeps these bones' curves (``hanim._kept_bone``): Origin's clips
    bind all 176 ``k_f_*`` nulls and animate the ones the lists name —
    ``k_f_kokan_00`` moves up to 301 mm inside one clip of female
    ``02_00_00`` — and ``SetInfo``'s target is that animated transform
    (§7.511).  Empty before ``lookat.json`` is baked.
    """
    out: set = set()
    for name in (_baked().get("lists") or {}):
        lst = load_dan_list(name)
        for row in (lst.by_motion.values() if lst is not None else ()):
            if row.look_at_null:
                out.add(row.look_at_null)
    return frozenset(out)


def resolve_null_bone(skel, null_name: str) -> str | None:
    """The skeleton's bone named *null_name* (a list row's ``strLookAtNull``), or None.

    ``SetInfo`` looks the name up under the woman's body bone (``FindLoop``);
    when it is not there ``transLookAtNull`` is null and nothing is aimed.  All
    five names the install's lists use are on the reference armature every
    character is grafted from (§7.503), so there is no stand-in bone: a miss
    aims nothing and is printed once per name.
    """
    if not null_name or skel is None:
        return None
    if skel.has(null_name):
        return null_name
    if null_name not in _MISSING_NULLS:
        _MISSING_NULLS.add(null_name)
        print(f"[lookat_dan] 骨架沒有 {null_name}：不瞄準", file=sys.stderr)
    return None


def _bone_world(skel, pose: dict, model_xf: np.ndarray, bone: str):
    if skel is None or not skel.has(bone):
        return None
    # one bone and its ancestors, not the whole rig (§7.416)
    world = skel.evaluate(pose, only=[skel.id(bone)])
    local = world[skel.id(bone)]
    M = np.asarray(model_xf, dtype=np.float64) @ local
    return M


def _quat_abs_to_additive(skel, bone: str, abs_local_q: np.ndarray) -> np.ndarray:
    """Unity absolute local rot → our additive ``pose[bone]['rot']``."""
    b = skel.bones[skel.id(bone)]
    return m3.quat_normalize(
        m3.quat_mul(m3.quat_conj(b.local_rot),
                    np.asarray(abs_local_q, dtype=np.float64)))


def aim_up_vector(skel, pose: dict, model_xf: np.ndarray, kind: int) -> np.ndarray:
    """The up vector ``Lookat_dan.LookAtProc`` aims with, for a row's *kind*.

    ``lookAtCalcKind`` 0 runs the component's ``LookAt.Update``, whose
    ``GetUpVector`` returns ``trfUpAxis.up`` for ``upAxisType`` 1 — the ref
    bone's own +Y, its ``Transform.rotation`` times ``Vector3.up`` (the X
    mirror carries Unity's ``up`` to that axis) — and ``Vector3.up`` when
    ``SetMale`` found no such bone.  Any other kind is
    ``Transform.LookAt(target)``, whose up is ``Vector3.up``.  The two bases
    share the aimed axis and differ in the roll about it.

    The rotation leaves the bone's scale out (``bp_dan.unity_rotation``): its
    world matrix's column 1 is not that axis when a parent scales unevenly,
    and is nothing when a clip scales the bone to nothing (§7.1223).
    """
    world_up = np.array([0.0, 1.0, 0.0])
    if int(kind) != 0:
        return world_up
    if skel is None or not skel.has(LOOKAT_UP_BONE):
        return world_up
    from .bp_dan import unity_rotation
    return unity_rotation(skel, pose, model_xf, LOOKAT_UP_BONE)[:, 1].copy()


def _apply_look_at_base(a_skel, a_pose: dict, a_xf: np.ndarray,
                        base_name: str, target_w: np.ndarray,
                        up_w=(0.0, 1.0, 0.0)) -> None:
    """Rotate *base* so its local +Z (shaft) points toward *target_w*, rolled by *up_w*.

    ``LookAtQuat`` refuses a basis whose trace is -1 (a half turn) and keeps
    the previous rotation; the look rotation here is always a proper basis,
    so that branch is not reproduced.
    """
    if not a_skel.has(base_name):
        return
    bi = a_skel.id(base_name)
    world = a_skel.evaluate(a_pose, only=[bi])     # reads bi and its parent
    base_w = np.asarray(a_xf, dtype=np.float64) @ world[bi]
    base_pos = base_w[:3, 3]
    fwd = target_w - base_pos
    if float(np.linalg.norm(fwd)) < 1e-8:
        return
    # World rotation that aims local +Z along fwd, +Y toward up_w.
    want_w = m3.quat_look_rotation(fwd, up_w)
    # parent world rotation
    parent = a_skel.bones[bi].parent
    if parent < 0:
        parent_w_q = m3.quat_identity()
    else:
        parent_M = np.asarray(a_xf, dtype=np.float64) @ world[parent]
        parent_w_q = m3.quat_from_mat4(parent_M)
    # abs local = conj(parent_world) * want_world
    abs_local = m3.quat_normalize(m3.quat_mul(m3.quat_conj(parent_w_q), want_w))
    add = _quat_abs_to_additive(a_skel, base_name, abs_local)
    cur = dict(a_pose.get(base_name) or {})
    cur["rot"] = add
    a_pose[base_name] = cur


def _set_world_position(skel, pose: dict, xf: np.ndarray,
                        name: str, target_w: np.ndarray) -> None:
    """``Transform.position = target_w`` on bone *name* in *pose*: its local translation, rotation kept.

    ``bTopStick`` does it to the shaft's tip (``LookAtProc``), ``LateUpdate`` to
    the null after its ``lstShape`` sum (:func:`apply_lookat_dan`).
    """
    if not skel.has(name):
        return
    ti = skel.id(name)
    parent = skel.bones[ti].parent
    if parent < 0:
        return
    world = skel.evaluate(pose, only=[parent])
    parent_M = np.asarray(xf, dtype=np.float64) @ world[parent]
    inv = np.linalg.inv(parent_M)
    local_abs = (inv @ np.array([target_w[0], target_w[1], target_w[2], 1.0],
                                dtype=np.float64))[:3]
    bind_local = np.asarray(skel.bones[ti].local_pos, dtype=np.float64)
    off = local_abs - bind_local
    cur = dict(pose.get(name) or {})
    cur["pos"] = off
    pose[name] = cur


def shape_offset(row: MotionLookAt, null_M: np.ndarray, shape) -> np.ndarray:
    """``Lookat_dan.LateUpdate``'s ``lstShape`` sum, in world space (§7.512).

    For each entry ``v = GetShapeBodyValue(shape)``; below 0.5 the offset is
    ``Lerp(minPos, 0, InverseLerp(0, 0.5, v))``, else ``Lerp(0, maxPos,
    InverseLerp(0.5, 1, v))`` (both clamp to the ends), and
    ``transLookAtNull.TransformDirection`` turns it to world space — the
    null's rotation only, so the list's X is mirrored onto ours.  The sum is
    written to the null's ``position`` (:func:`apply_lookat_dan` does it); the
    next frame's Animator puts the null back (every clip binds it, §7.511),
    so it does not accumulate.
    """
    total = np.zeros(3)
    if not row.shapes or shape is None:
        return total
    rot = np.asarray(null_M, dtype=np.float64)[:3, :3]
    rot = rot / np.maximum(np.linalg.norm(rot, axis=0), 1e-12)
    for sh in row.shapes:
        v = shape(sh.shape)
        if v is None:
            continue
        if v < 0.5:
            off = sh.min_pos * (1.0 - min(max(v / 0.5, 0.0), 1.0))
        else:
            off = sh.max_pos * min(max((v - 0.5) / 0.5, 0.0), 1.0)
        total += rot @ (np.asarray(off, dtype=np.float64) * (-1.0, 1.0, 1.0))
    return total


def apply_lookat_dan(
        a_pose: dict, a_xf: np.ndarray, a_skel,
        females,
        row: MotionLookAt | None,
        *,
        base_name: str = DEFAULT_BASE,
        tip_name: str = DEFAULT_TOP,
        place=None,
) -> dict:
    """Mutate *a_pose* (male) with Lookat_dan for list *row*.  Returns a_pose.

    *place*, when given, is ``(woman index, null bone, target) → target``: it
    moves the point aimed at.  A role whose pose is carried onto another
    skeleton is not drawn where *a_skel* / the woman's skeleton say, and the
    aim has to close between the bodies that are drawn (``h_runtime``).

    *females* is ``Lookat_dan.females`` (:class:`Woman` each, the main woman
    first): ``SetInfo`` finds the null on ``females[numFemale]`` and
    ``LateUpdate`` reads that woman's sliders — the FFM lists aim 66 rows at
    the second woman (§7.512).  A row naming a woman who is not there aims
    at nothing.
    """
    if row is None or a_skel is None or not row.look_at_null:
        return a_pose
    women = list(females or ())
    if not 0 <= row.num_female < len(women) or women[row.num_female].skel is None:
        return a_pose
    her = women[row.num_female]
    null_bone = resolve_null_bone(her.skel, row.look_at_null)
    if null_bone is None:
        return a_pose
    target_M = _bone_world(her.skel, her.pose, her.xf, null_bone)
    if target_M is None:
        return a_pose
    shift = shape_offset(row, target_M, her.shape)
    target = target_M[:3, 3] + shift
    if np.any(shift):
        # ``transLookAtNull.position = …`` (``LateUpdate`` IL_0136–013d): the null itself moves, not
        # only the point aimed at, so what reads it after this (BetterPenetration, the IK, the
        # drawn pose) finds it where the game has it; its rotation stays the clip's
        _set_world_position(her.skel, her.pose, her.xf, null_bone, target)
    if place is not None:
        target = place(row.num_female, null_bone, target)
    # ``SetMale`` finds the base and the tip by name; ``LookAtProc`` returns
    # without aiming when the base is missing and skips the stick when the
    # tip is — no other bone stands in (the same rule as the null, §7.503).
    if not a_skel.has(base_name):
        return a_pose
    up = aim_up_vector(a_skel, a_pose, a_xf, row.look_at_calc_kind)
    # 1) aim shaft base at target
    _apply_look_at_base(a_skel, a_pose, a_xf, base_name, target, up)
    # 2) stick tip when list says so
    if row.top_stick and a_skel.has(tip_name):
        _set_world_position(a_skel, a_pose, a_xf, tip_name, target)
    return a_pose


def aim_dan_at(a_pose: dict, a_xf: np.ndarray, a_skel,
               target_w, *, stick_at=None, aim: bool = True,
               base_name: str = DEFAULT_BASE,
               tip_name: str = DEFAULT_TOP) -> bool:
    """Aim the shaft at *target_w*; optionally stick the tip at *stick_at*.

    The same two operations :func:`apply_lookat_dan` performs, with the target
    handed in instead of read from a ``dan_*`` row.  That list is Origin's own data
    and only exists for Origin's clips; the authored dual layouts need the *rule*,
    not the data, and the rule is "insertion is aimed with the shaft bones,
    the roots are not slid" (§7.117).

    Returns False when this skeleton carries no shaft, so a caller can report
    rather than quietly do nothing.
    """
    if a_skel is None:
        return False
    base = next((b for b in (base_name, DEFAULT_BASE, DEFAULT_REF)
                 if a_skel.has(b)), None)
    if base is None:
        return False
    target_w = np.asarray(target_w, dtype=np.float64)
    if aim:
        _apply_look_at_base(a_skel, a_pose, a_xf, base, target_w)
    tip = tip_name if a_skel.has(tip_name) else (
        DEFAULT_TOP if a_skel.has(DEFAULT_TOP) else None)
    if tip is not None and stick_at is not None:
        _set_world_position(a_skel, a_pose, a_xf, tip,
                            np.asarray(stick_at, dtype=np.float64))
    return True


def clear_lookat_cache() -> None:
    load_dan_list.cache_clear()
    null_names.cache_clear()
    _baked.cache_clear()
