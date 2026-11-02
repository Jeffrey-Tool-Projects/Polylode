"""Illusion's *other* sway component: ``DynamicBone_Ver02``, transcribed.

:mod:`dynbone` is Will Hong's ``DynamicBone`` — skirts, sleeves, ribbons, hair.
The breasts and buttocks do **not** use it.  They hang off Illusion's own
``DynamicBone_Ver02``, a different script with a different serialised layout and
a different solver, and there are exactly four of them, all on the armature
prefab ``p_cf_body_bone`` in ``chara/oo_base.unity3d``::

    Comment  左胸 / 右胸 / 左尻 / 右尻   →  ChaControl.dictDynamicBoneBust[0..3]

``dynbone.parse_component`` only knows the other layout, so
``PrefabLoader.dynamic_bones()`` finds **zero** chains in ``oo_base`` — which is
why the bust was never settled at all (§7.42).

**Reading the parameters.**  ``oo_base`` ships type trees (the clothing bundles
do not), so :func:`from_tree` is the primary path and :func:`parse_component`
the byte-order fallback, deserialised from ``Assembly-CSharp``'s field list::

    Comment(string) Root(PPtr) UpdateRate(f) ReflectSpeed(f)
    HeavyLoopMaxCount(int) Gravity(v3) Force(v3)
    Colliders(List<PPtr>) Bones(List<PPtr>) Patterns(List<BonePtn>)
    PtnNo(int) DragAndDrop(string)

with ``BonePtn`` = ``Name(string) Gravity(v3) EndOffset(v3)
EndOffsetDamping/Elasticity/Stiffness/Inert(f) Params(List<BoneParameter>)``.
The field list also shows a public ``ParticlePtns`` after that, and it is
**not** on disk: ``ParticlePtn`` carries no ``[Serializable]``, so Unity skips
the field entirely.  Reading a count for it eats the next pattern's name length
and the object over-runs by four bytes per pattern — which is exactly what the
"a parse counts only if it consumes the object's bytes exactly" rule is for.

**The rig.**  The solver never touches a Skeleton.  It talks to any object
offering these six calls, keyed by whatever the caller uses for a bone (we pass
names):

===========================  ===============================================
``has(key)``                 is there such a bone
``matrix(key)``              ``Transform.localToWorldMatrix`` — 4×4 with scale
``rotation(key)``            ``Transform.rotation`` — world quaternion, scale
                             ignored, i.e. the product of the local rotations
``set_rotation(key, q)``     ``Transform.rotation = q``
``set_position(key, p)``     ``Transform.position = p``
``local(key)``               ``(localPosition, localRotation, localScale)``
``set_local(key, p, q, s)``  writes the same three back
===========================  ===============================================

**The solver** is ``UpdateParticles1`` / ``UpdateParticles2`` /
``ApplyParticlesToTransforms``; see each function.  Note what does *not* appear
in it: no ``m_LocalGravity`` projection, no freeze axis, no distribution curves.
Ver02 stores a full parameter block per bone instead, and blends between three
named *patterns* (``通常`` / ``ブラ`` / ``まっすぐ``) rather than sampling a
curve along the chain.

**Why a rest solve is the same thing the game does.**  ``LateUpdate`` calls
``InitTransforms()`` *before* ``UpdateDynamicBones``, which puts every bone back
on its reference transform — so the transforms the solver reads are the animated
rest pose on every frame, while the particles carry their positions over.
Iterating with the transforms held still is therefore not an approximation of
the per-frame loop, it *is* the per-frame loop with the animation standing
still.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from ..core import math3d as m3
from .dynbone import _R, collide

_ZERO = np.zeros(3)


def _v3(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=np.float64)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))


# ---------------------------------------------------------------------------
# serialised data
# ---------------------------------------------------------------------------
@dataclass
class BoneParam:
    """One ``BoneParameter`` — the per-bone block of a pattern.

    The defaults are ``BoneParameter..ctor``'s, not zero: ``NextBoneLength``
    starts at **1**, which is what the synthetic end-of-chain parameter gets.
    """
    name: str = ""
    ref: tuple = (0, 0)                       # RefTransform PPtr
    ref_name: str = ""
    is_rotation_calc: bool = False
    damping: float = 0.0
    elasticity: float = 0.0
    stiffness: float = 0.0
    inert: float = 0.0
    next_bone_length: float = 1.0
    collision_radius: float = 0.0
    is_move_limit: bool = False
    move_limit_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    move_limit_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    keep_length_limit_min: float = 0.0
    keep_length_limit_max: float = 0.0
    is_crush: bool = False
    crush_move_area_min: float = 0.0
    crush_move_area_max: float = 0.0
    crush_add_xy_min: float = 0.0
    crush_add_xy_max: float = 0.0


@dataclass
class Pattern:
    """One ``BonePtn`` — a named set of parameters the chain can switch to."""
    name: str = ""
    gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    end_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    end_damping: float = 0.0
    end_elasticity: float = 0.0
    end_stiffness: float = 0.0
    end_inert: float = 0.0
    params: list = field(default_factory=list)


@dataclass
class Ver02Spec:
    """One ``DynamicBone_Ver02`` component."""
    comment: str = ""
    root: tuple = (0, 0)
    root_name: str = ""
    update_rate: float = 60.0
    reflect_speed: float = 1.0
    heavy_loop_max_count: int = 3
    gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    colliders: list = field(default_factory=list)
    bones: list = field(default_factory=list)          # PPtrs
    bone_names: list = field(default_factory=list)
    patterns: list = field(default_factory=list)
    ptn_no: int = 0
    enabled: bool = True

    def mirror_x(self) -> "Ver02Spec":
        """The same component in this project's mirrored-X space.

        Everything else that reads a Unity transform here mirrors X (see
        :mod:`prefab`), so the vectors that live in those frames have to be
        mirrored too, or the ``ブラ`` pattern's sideways gravity would push both
        breasts the same way.  A move limit is an interval, so mirroring its
        axis swaps which end is the minimum.
        """
        def mv(v):
            v = np.asarray(v, dtype=np.float64)
            return np.array([-v[0], v[1], v[2]])

        out = Ver02Spec(
            comment=self.comment, root=self.root, root_name=self.root_name,
            update_rate=self.update_rate, reflect_speed=self.reflect_speed,
            heavy_loop_max_count=self.heavy_loop_max_count,
            gravity=mv(self.gravity), force=mv(self.force),
            colliders=list(self.colliders), bones=list(self.bones),
            bone_names=list(self.bone_names), ptn_no=self.ptn_no,
            enabled=self.enabled)
        for ptn in self.patterns:
            p = Pattern(name=ptn.name, gravity=mv(ptn.gravity),
                        end_offset=mv(ptn.end_offset),
                        end_damping=ptn.end_damping,
                        end_elasticity=ptn.end_elasticity,
                        end_stiffness=ptn.end_stiffness,
                        end_inert=ptn.end_inert)
            for prm in ptn.params:
                lo, hi = mv(prm.move_limit_min), mv(prm.move_limit_max)
                q = BoneParam(**{**prm.__dict__})
                q.move_limit_min = np.array([hi[0], lo[1], lo[2]])
                q.move_limit_max = np.array([lo[0], hi[1], hi[2]])
                p.params.append(q)
            out.patterns.append(p)
        return out


def _tree_v3(d) -> np.ndarray:
    d = d or {}
    return _v3(d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0))


def from_tree(tree: dict, name_of=None) -> Ver02Spec | None:
    """Build a spec from a type tree — the path ``oo_base`` supports.

    Prefer this over :func:`parse_component`: it cannot be thrown off by a
    field being added in a future build.  ``name_of(path_id) -> str`` resolves
    the transform references, which is what the solver keys bones by.
    """
    def nm(ref):
        pid = (ref or {}).get("m_PathID") or 0
        return (name_of(pid) if (name_of is not None and pid) else "")

    try:
        spec = Ver02Spec(
            comment=str(tree.get("Comment") or ""),
            root=(0, (tree.get("Root") or {}).get("m_PathID") or 0),
            root_name=nm(tree.get("Root")),
            update_rate=float(tree.get("UpdateRate", 60.0)),
            reflect_speed=float(tree.get("ReflectSpeed", 1.0)),
            heavy_loop_max_count=int(tree.get("HeavyLoopMaxCount", 3)),
            gravity=_tree_v3(tree.get("Gravity")),
            force=_tree_v3(tree.get("Force")),
            ptn_no=int(tree.get("PtnNo", 0)),
            enabled=bool(tree.get("m_Enabled", 1)))
        spec.colliders = [(0, (c or {}).get("m_PathID") or 0)
                          for c in (tree.get("Colliders") or [])]
        for b in (tree.get("Bones") or []):
            spec.bones.append((0, (b or {}).get("m_PathID") or 0))
            spec.bone_names.append(nm(b))
        for pt in (tree.get("Patterns") or []):
            ptn = Pattern(
                name=str(pt.get("Name") or ""),
                gravity=_tree_v3(pt.get("Gravity")),
                end_offset=_tree_v3(pt.get("EndOffset")),
                end_damping=float(pt.get("EndOffsetDamping", 0.0)),
                end_elasticity=float(pt.get("EndOffsetElasticity", 0.0)),
                end_stiffness=float(pt.get("EndOffsetStiffness", 0.0)),
                end_inert=float(pt.get("EndOffsetInert", 0.0)))
            for pr in (pt.get("Params") or []):
                ptn.params.append(BoneParam(
                    name=str(pr.get("Name") or ""),
                    ref=(0, (pr.get("RefTransform") or {}).get("m_PathID") or 0),
                    ref_name=nm(pr.get("RefTransform")),
                    is_rotation_calc=bool(pr.get("IsRotationCalc", 0)),
                    damping=float(pr.get("Damping", 0.0)),
                    elasticity=float(pr.get("Elasticity", 0.0)),
                    stiffness=float(pr.get("Stiffness", 0.0)),
                    inert=float(pr.get("Inert", 0.0)),
                    next_bone_length=float(pr.get("NextBoneLength", 1.0)),
                    collision_radius=float(pr.get("CollisionRadius", 0.0)),
                    is_move_limit=bool(pr.get("IsMoveLimit", 0)),
                    move_limit_min=_tree_v3(pr.get("MoveLimitMin")),
                    move_limit_max=_tree_v3(pr.get("MoveLimitMax")),
                    keep_length_limit_min=float(pr.get("KeepLengthLimitMin", 0.0)),
                    keep_length_limit_max=float(pr.get("KeepLengthLimitMax", 0.0)),
                    is_crush=bool(pr.get("IsCrush", 0)),
                    crush_move_area_min=float(pr.get("CrushMoveAreaMin", 0.0)),
                    crush_move_area_max=float(pr.get("CrushMoveAreaMax", 0.0)),
                    crush_add_xy_min=float(pr.get("CrushAddXYMin", 0.0)),
                    crush_add_xy_max=float(pr.get("CrushAddXYMax", 0.0))))
            spec.patterns.append(ptn)
    except (KeyError, TypeError, ValueError):
        return None
    return spec


def _read_param(r: _R) -> BoneParam:
    p = BoneParam()
    p.name = r.string()
    p.ref = r.pptr()
    p.is_rotation_calc = r.boolean()
    p.damping, p.elasticity = r.f(), r.f()
    p.stiffness, p.inert = r.f(), r.f()
    p.next_bone_length, p.collision_radius = r.f(), r.f()
    p.is_move_limit = r.boolean()
    p.move_limit_min, p.move_limit_max = r.v3(), r.v3()
    p.keep_length_limit_min, p.keep_length_limit_max = r.f(), r.f()
    p.is_crush = r.boolean()
    p.crush_move_area_min, p.crush_move_area_max = r.f(), r.f()
    p.crush_add_xy_min, p.crush_add_xy_max = r.f(), r.f()
    return p


def parse_component(raw: bytes, name_of=None) -> Ver02Spec | None:
    """Deserialise one ``DynamicBone_Ver02``; ``None`` if the bytes do not fit."""
    r = _R(raw)
    try:
        r.pptr()                              # m_GameObject
        enabled = r.boolean()                 # m_Enabled
        r.pptr()                              # m_Script
        r.string()                            # m_Name
        spec = Ver02Spec(comment=r.string(), root=r.pptr(),
                         update_rate=r.f(), reflect_speed=r.f(),
                         heavy_loop_max_count=r.i(),
                         gravity=r.v3(), force=r.v3(), enabled=enabled)
        spec.colliders = r.pptr_list()
        spec.bones = r.pptr_list()
        for _ in range(r.i()):
            ptn = Pattern(name=r.string(), gravity=r.v3(), end_offset=r.v3(),
                          end_damping=r.f(), end_elasticity=r.f(),
                          end_stiffness=r.f(), end_inert=r.f())
            ptn.params = [_read_param(r) for _ in range(r.i())]
            # ``BonePtn.ParticlePtns`` is a public field and the field list
            # shows it, but there is **no list here**: ``ParticlePtn`` is not
            # marked serialisable, so Unity writes nothing for it.  Reading a
            # count anyway swallows the next pattern's name length -- and the
            # object then over-runs by exactly the four bytes per pattern that
            # the exact-consumption check exists to catch.
            spec.patterns.append(ptn)
        spec.ptn_no = r.i()
        r.string()                            # DragAndDrop
    except (struct.error, IndexError):
        return None
    if r.p != len(raw):
        return None
    if name_of is not None:
        spec.root_name = name_of(spec.root[1]) if spec.root[1] else ""
        spec.bone_names = [name_of(b[1]) if b[1] else "" for b in spec.bones]
        for ptn in spec.patterns:
            for prm in ptn.params:
                prm.ref_name = name_of(prm.ref[1]) if prm.ref[1] else ""
    return spec


# ---------------------------------------------------------------------------
# runtime state
# ---------------------------------------------------------------------------
@dataclass
class ParticlePtn:
    """``AppendParticlePtn`` — a pattern's per-particle runtime block."""
    is_rotation_calc: bool = False
    damping: float = 0.0
    elasticity: float = 0.0
    stiffness: float = 0.0
    inert: float = 0.0
    scale_next_bone_length: float = 1.0
    radius: float = 0.0
    is_move_limit: bool = False
    move_limit_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    move_limit_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    keep_length_limit_min: float = 0.0
    keep_length_limit_max: float = 0.0
    is_crush: bool = False
    crush_move_area_min: float = 0.0
    crush_move_area_max: float = 0.0
    crush_add_xy_min: float = 0.0
    crush_add_xy_max: float = 0.0
    end_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    init_local_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    init_local_rot: np.ndarray = field(default_factory=m3.quat_identity)
    init_local_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    ref_key: object = None
    local_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Particle:
    """One solver particle.  ``key`` is ``None`` for the end-of-chain node."""
    key: object = None
    parent: int = -1
    is_rotation_calc: bool = False
    damping: float = 0.0
    elasticity: float = 0.0
    stiffness: float = 0.0
    inert: float = 0.0
    scale_next_bone_length: float = 1.0
    radius: float = 0.0
    is_move_limit: bool = False
    move_limit_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    move_limit_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    keep_length_limit_min: float = 0.0
    keep_length_limit_max: float = 0.0
    is_crush: bool = False
    crush_move_area_min: float = 0.0
    crush_move_area_max: float = 0.0
    crush_add_xy_min: float = 0.0
    crush_add_xy_max: float = 0.0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev: np.ndarray = field(default_factory=lambda: np.zeros(3))
    end_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    local_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    init_local_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    init_local_rot: np.ndarray = field(default_factory=m3.quat_identity)
    init_local_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    ref_key: object = None
    # the rest pose the transforms sit in for the whole solve, cached because
    # ``InitTransforms`` puts them back there every frame anyway
    rest_world: np.ndarray | None = None
    rest_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Chain:
    """A component's particles plus the frame-level state the solver reads."""
    spec: Ver02Spec
    particles: list = field(default_factory=list)
    ptn_no: int = 0
    object_scale: float = 1.0
    weight: float = 1.0
    gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # ``Root.position - ObjectPrevPosition`` — how far the character moved
    # since the last frame.  Zero for a rest solve, and ``UpdateDynamicBones``
    # zeroes it again after every inner iteration anyway.
    object_move: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # the component's key where its owner keeps it (``dictDynamicBoneBust``'s 0‥3), -1 when none
    index: int = -1


# ---------------------------------------------------------------------------
# setup — Awake
# ---------------------------------------------------------------------------
def _append_particle_ptn(prm: BoneParam, end_offset, rig) -> ParticlePtn:
    """``AppendParticlePtn``."""
    q = ParticlePtn(
        is_rotation_calc=prm.is_rotation_calc,
        damping=_clamp01(prm.damping), elasticity=_clamp01(prm.elasticity),
        stiffness=_clamp01(prm.stiffness), inert=_clamp01(prm.inert),
        scale_next_bone_length=max(prm.next_bone_length, 0.0),
        radius=max(prm.collision_radius, 0.0),
        is_move_limit=prm.is_move_limit,
        move_limit_min=np.asarray(prm.move_limit_min, dtype=np.float64).copy(),
        move_limit_max=np.asarray(prm.move_limit_max, dtype=np.float64).copy(),
        keep_length_limit_min=prm.keep_length_limit_min,
        keep_length_limit_max=prm.keep_length_limit_max,
        is_crush=prm.is_crush,
        crush_move_area_min=prm.crush_move_area_min,
        crush_move_area_max=prm.crush_move_area_max,
        crush_add_xy_min=prm.crush_add_xy_min,
        crush_add_xy_max=prm.crush_add_xy_max)
    ref = prm.ref_name or None
    if ref is not None and rig.has(ref):
        pos, rot, scl = rig.local(ref)
        q.init_local_pos, q.init_local_rot, q.init_local_scale = pos, rot, scl
        q.ref_key = ref
    else:
        q.end_offset = np.asarray(end_offset, dtype=np.float64).copy()
    return q


def _init_node_particle(spec: Ver02Spec, rig) -> list:
    """``InitNodeParticle`` — one ``ParticlePtn`` table per pattern.

    The extra entry at the end is the chain's free tip: a particle with no
    transform, whose parameters come from the pattern's ``EndOffset*`` fields
    laid over a **default** ``BoneParameter`` (so ``NextBoneLength`` is 1, not
    0 — that default is a constructor initialiser, not a zero field).
    """
    out = []
    for ptn in spec.patterns:
        table = []
        if len(ptn.params) != len(spec.bones):
            out.append(table)
            continue
        for prm in ptn.params:
            table.append(_append_particle_ptn(prm, _ZERO, rig))
        end = BoneParam(damping=ptn.end_damping, elasticity=ptn.end_elasticity,
                        stiffness=ptn.end_stiffness, inert=ptn.end_inert)
        table.append(_append_particle_ptn(end, ptn.end_offset, rig))
        out.append(table)
    return out


def _append_particles(particles: list, key, prm: BoneParam, end_offset,
                      parent: int, rig) -> Particle:
    """``AppendParticles``."""
    p = Particle(
        key=key, parent=parent,
        is_rotation_calc=prm.is_rotation_calc,
        damping=_clamp01(prm.damping), elasticity=_clamp01(prm.elasticity),
        stiffness=_clamp01(prm.stiffness), inert=_clamp01(prm.inert),
        scale_next_bone_length=max(prm.next_bone_length, 0.0),
        radius=max(prm.collision_radius, 0.0),
        is_move_limit=prm.is_move_limit,
        move_limit_min=np.asarray(prm.move_limit_min, dtype=np.float64).copy(),
        move_limit_max=np.asarray(prm.move_limit_max, dtype=np.float64).copy(),
        keep_length_limit_min=prm.keep_length_limit_min,
        keep_length_limit_max=prm.keep_length_limit_max,
        is_crush=prm.is_crush,
        crush_move_area_min=prm.crush_move_area_min,
        crush_move_area_max=prm.crush_move_area_max,
        crush_add_xy_min=prm.crush_add_xy_min,
        crush_add_xy_max=prm.crush_add_xy_max)
    if key is not None:
        world = rig.matrix(key)
        p.pos = world[:3, 3].copy()
        p.prev = p.pos.copy()
        pos, rot, _scl = rig.local(key)
        p.init_local_pos, p.init_local_rot = pos, rot
        p.ref_key = key
        if parent >= 0:
            # CalcLocalPosition: the particle's position in its parent's frame
            pw = rig.matrix(particles[parent].key)
            p.local_pos = m3.transform_point(m3.inverse(pw), p.pos)
    else:
        pk = particles[parent].key
        p.end_offset = np.asarray(end_offset, dtype=np.float64).copy()
        p.pos = m3.transform_point(rig.matrix(pk), p.end_offset)
        p.prev = p.pos.copy()
    particles.append(p)
    return p


def _set_ptn(chain: Chain, tables: list, ptn_no: int) -> None:
    """``setPtn`` — copy a pattern's blocks onto the live particles."""
    spec = chain.spec
    if not (0 <= ptn_no < len(spec.patterns)):
        return
    table = tables[ptn_no]
    if len(table) != len(chain.particles):
        return
    chain.ptn_no = ptn_no
    chain.gravity = np.asarray(spec.patterns[ptn_no].gravity,
                               dtype=np.float64).copy()
    for p, q in zip(chain.particles, table):
        p.is_rotation_calc = q.is_rotation_calc
        p.damping = _clamp01(q.damping)
        p.elasticity = _clamp01(q.elasticity)
        p.stiffness = _clamp01(q.stiffness)
        p.inert = _clamp01(q.inert)
        p.scale_next_bone_length = max(q.scale_next_bone_length, 0.0)
        p.radius = max(q.radius, 0.0)
        p.is_move_limit = q.is_move_limit
        p.move_limit_min = q.move_limit_min.copy()
        p.move_limit_max = q.move_limit_max.copy()
        p.keep_length_limit_min = q.keep_length_limit_min
        p.keep_length_limit_max = q.keep_length_limit_max
        p.is_crush = q.is_crush
        p.crush_move_area_min = q.crush_move_area_min
        p.crush_move_area_max = q.crush_move_area_max
        p.crush_add_xy_min = q.crush_add_xy_min
        p.crush_add_xy_max = q.crush_add_xy_max
        p.init_local_pos = q.init_local_pos.copy()
        p.init_local_rot = q.init_local_rot.copy()
        p.init_local_scale = q.init_local_scale.copy()
        p.ref_key = q.ref_key
        p.local_pos = q.local_pos.copy()
        p.end_offset = q.end_offset.copy()


def _init_local_position(spec: Ver02Spec, chain: Chain, tables: list,
                         rig) -> None:
    """``InitLocalPosition`` — each pattern's rest offsets, in the ref frames.

    The rest target a particle is pulled toward is **not** measured between the
    bones: it is measured between the pattern's ``RefTransform``s, which are the
    ``cf_j_bustNN_L_01`` siblings the solver never moves.  The transforms are
    posed onto those references while the measurement is taken and put back
    afterwards, exactly as the game does — with our armature they already agree,
    but a rig where they did not would otherwise be measured mid-solve.
    """
    saved = [(p.key, rig.local(p.key)) for p in chain.particles
             if p.key is not None and rig.has(p.key)]
    for pi, ptn in enumerate(spec.patterns):
        table = tables[pi]
        if len(table) != len(chain.particles):
            continue
        for j, prm in enumerate(ptn.params):
            ref = prm.ref_name or None
            if ref is None or not rig.has(ref):
                continue
            pos, rot, scl = rig.local(ref)
            table[j].init_local_pos = pos
            table[j].init_local_rot = rot
            table[j].init_local_scale = scl
            table[j].ref_key = ref
        for p, q in zip(chain.particles, table):
            if p.key is not None and rig.has(p.key):
                rig.set_local(p.key, q.init_local_pos, q.init_local_rot,
                              q.init_local_scale)
        for j in range(1, len(ptn.params)):
            a, b = ptn.params[j].ref_name, ptn.params[j - 1].ref_name
            if not a or not b or not rig.has(a) or not rig.has(b):
                continue
            table[j].local_pos = m3.transform_point(
                m3.inverse(rig.matrix(b)), rig.matrix(a)[:3, 3])
    for key, (pos, rot, scl) in saved:
        rig.set_local(key, pos, rot, scl)


def _init_transforms(chain: Chain, rig, keep=frozenset()) -> None:
    """``InitTransforms`` — put every bone back on its reference pose (but the ones in *keep*)."""
    for p in chain.particles:
        if p.key is None or not rig.has(p.key) or p.key in keep:
            continue
        if p.ref_key is not None and rig.has(p.ref_key):
            rig.set_local(p.key, *rig.local(p.ref_key))
        else:
            rig.set_local(p.key, p.init_local_pos, p.init_local_rot,
                          p.init_local_scale)


def _is_ref_transform(spec: Ver02Spec, rig) -> bool:
    """``IsRefTransform`` — does every pattern parameter name a live transform?

    It gates the ``setPtn`` in ``Awake``: without the references there are no
    per-pattern blocks worth selecting, and the particles keep what
    ``AppendParticles`` gave them (pattern 0's own parameters, and rest offsets
    measured between the bones themselves rather than between the references).
    """
    for ptn in spec.patterns:
        for prm in ptn.params:
            if not prm.ref_name or not rig.has(prm.ref_name):
                return False
    return True


def setup(spec: Ver02Spec, rig, ptn_no: int | None = None) -> Chain | None:
    """``Awake`` — build the particles and select a pattern.

    ``None`` when the component cannot be set up on this rig at all: a root and
    every bone have to exist, and the bone count has to match the pattern's
    parameter count (``SetupParticles`` bails out otherwise).
    """
    if not spec.patterns or not spec.bones:
        return None
    root = spec.root_name or (spec.bone_names[0] if spec.bone_names else "")
    if not root or not rig.has(root):
        return None
    if len(spec.bones) != len(spec.patterns[0].params):
        return None
    if any(not n or not rig.has(n) for n in spec.bone_names):
        return None

    chain = Chain(spec=spec)
    tables = _init_node_particle(spec, rig)

    # SetupParticles takes ``Root.lossyScale.x``; UpdateDynamicBones recomputes
    # it every frame as ``Abs(Root.lossyScale.x)``, which is the one that ends
    # up scaling the force, so take the magnitude here.
    chain.object_scale = float(np.linalg.norm(rig.matrix(root)[:3, 0]))
    parent = -1
    for i, name in enumerate(spec.bone_names):
        _append_particles(chain.particles, name, spec.patterns[0].params[i],
                          _ZERO, parent, rig)
        parent += 1
    _append_particles(chain.particles, None, BoneParam(),
                      spec.patterns[0].end_offset, parent, rig)

    _init_local_position(spec, chain, tables, rig)
    if _is_ref_transform(spec, rig):
        _set_ptn(chain, tables, spec.ptn_no if ptn_no is None else int(ptn_no))
    _init_transforms(chain, rig)

    # The transforms are back on the rest pose now and stay there for the whole
    # solve (see the module docstring), so cache what UpdateParticles2 reads.
    for p in chain.particles:
        if p.key is not None:
            p.rest_world = rig.matrix(p.key)
            p.rest_pos = p.rest_world[:3, 3].copy()
    chain.force = np.asarray(spec.force, dtype=np.float64).copy()
    return chain


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
def update1(chain: Chain) -> None:
    """``UpdateParticles1`` — integrate.

    ::

        force = (Gravity + Force) * ObjectScale
        per particle with a parent:
            v     = (pos - prev) * ReflectSpeed
            rmove = ObjectMove * inert
            prev  = pos + rmove
            pos  += v * (1 - damping) + force + rmove
        the root instead: prev = pos; pos = Transform.position

    ``ObjectMove`` is how far the character travelled since the last frame; for
    a rest solve it is zero and gravity is the only thing left pulling.
    """
    force = (chain.gravity + chain.force) * chain.object_scale
    reflect = chain.spec.reflect_speed
    for p in chain.particles:
        if p.parent < 0:
            p.prev = p.pos.copy()
            p.pos = p.rest_pos.copy()
            continue
        v = (p.pos - p.prev) * reflect
        rmove = chain.object_move * p.inert
        p.prev = p.pos + rmove
        p.pos = p.pos + v * (1.0 - p.damping) + force + rmove


def update2(chain: Chain, colliders=None) -> None:
    """``UpdateParticles2`` — elasticity, stiffness, colliders, length, limits.

    The rest target is rebuilt from the parent's **rest** matrix with its
    translation column replaced by the parent's solved position, and the target
    offset is the pattern's ``LocalPosition`` (or ``EndOffset`` for the tip).
    The segment length it is held to is measured on the rest pose, not on the
    solved one, which is why it does not drift.
    """
    for i in range(1, len(chain.particles)):
        p = chain.particles[i]
        pa = chain.particles[p.parent]
        if p.key is not None:
            d = float(np.linalg.norm(pa.rest_pos - p.rest_pos))
        else:
            d = float(np.linalg.norm(p.end_offset)) * chain.object_scale
        m = pa.rest_world.copy()
        m[:3, 3] = pa.pos
        rest = m3.transform_point(
            m, p.local_pos if p.key is not None else p.end_offset)

        stiff = 1.0 + (p.stiffness - 1.0) * chain.weight   # Mathf.Lerp(1, k, w)
        if stiff > 0.0 or p.elasticity > 0.0:
            v = rest - p.pos
            p.pos = p.pos + v * p.elasticity
            if stiff > 0.0:
                v = rest - p.pos
                ln = float(np.linalg.norm(v))
                mx = d * (1.0 - stiff) * 2.0
                if ln > mx:
                    p.pos = p.pos + v * ((ln - mx) / ln)
        if colliders:
            r = p.radius * chain.object_scale
            for col in colliders:
                collide(p.pos, r, col)
        v = pa.pos - p.pos
        ln = float(np.linalg.norm(v))
        if ln > 0.0:
            f = (ln - d) / ln
            if not (p.keep_length_limit_min < f):
                p.pos = p.pos + v * (f - p.keep_length_limit_min)
            elif not (f < p.keep_length_limit_max):
                p.pos = p.pos + v * (f - p.keep_length_limit_max)
        if p.key is not None and p.is_move_limit:
            m = p.rest_world.copy()
            m[:3, 3] = rest
            local = m3.transform_point(m3.inverse(m), p.pos)
            local = np.clip(local, p.move_limit_min, p.move_limit_max)
            p.pos = m3.transform_point(m, local)


def solve(chain: Chain, *, iterations: int = 600, tol: float = 1e-7,
          colliders=None) -> int:
    """Run to a steady state and report how many iterations it took.

    The game reaches this by simulating from the first frame with the pose
    standing still; we want it as a *rest* shape, so the same integrator runs
    until the particles stop moving.  It converges because damping and the
    length constraint take energy out on every step.
    """
    n = 0
    for n in range(1, max(1, int(iterations)) + 1):
        before = [p.pos.copy() for p in chain.particles]
        update1(chain)
        update2(chain, colliders=colliders)
        moved = max((float(np.abs(p.pos - b).max())
                     for p, b in zip(chain.particles, before)), default=0.0)
        if moved < tol:
            break
    return n


def apply_to_transforms(chain: Chain, rig) -> int:
    """``ApplyParticlesToTransforms`` — write the solve onto the rig.

    Two separate things happen, and only for some particles:

    * the **parent** is re-aimed, but only when *its* ``IsRotationCalc`` is
      set — on the bust that is ``cf_j_bustNN_L`` alone, so the first joint
      never turns;
    * the particle's own transform is moved to its solved world position.

    The tip particle has no transform and only contributes the rotation of the
    bone before it.  ``ScaleNextBoneLength`` shortens the direction the parent
    aims along, which is how a chain can point past its own child.
    """
    moved = 0
    for i in range(1, len(chain.particles)):
        p = chain.particles[i]
        pa = chain.particles[p.parent]
        if pa.is_rotation_calc and pa.key is not None and rig.has(pa.key):
            v3 = p.local_pos if p.key is not None else p.end_offset
            v4 = m3.quat_rotate(rig.rotation(pa.key), v3)
            v5 = p.pos - pa.pos
            if float(np.linalg.norm(v3)) != 0.0:
                v5 = (p.pos + (-v4) * (1.0 - pa.scale_next_bone_length)) - pa.pos
            q = m3.quat_from_to(v4, v5)
            rig.set_rotation(pa.key, m3.quat_mul(q, rig.rotation(pa.key)))
            moved += 1
        if p.key is None or not rig.has(p.key):
            continue
        if p.is_crush:
            local = m3.transform_point(m3.inverse(rig.matrix(p.key)), p.pos)
            if local[2] > 0.0:
                t = _clamp01(_inverse_lerp(0.0, p.crush_move_area_max, local[2]))
                add = p.crush_add_xy_max * t
            else:
                t = _clamp01(_inverse_lerp(p.crush_move_area_min, 0.0, local[2]))
                add = p.crush_add_xy_min * (1.0 - t)
            pos, rot, _scl = rig.local(p.key)
            rig.set_local(p.key, pos, rot,
                          p.init_local_scale + np.array([add, add, 0.0]))
        rig.set_position(p.key, p.pos)
        moved += 1
    return moved


def _inverse_lerp(a: float, b: float, v: float) -> float:
    """``Mathf.InverseLerp`` — 0 when the interval is empty."""
    return 0.0 if b == a else (v - a) / (b - a)
