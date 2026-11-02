"""Illusion's sway bones: the real ``DynamicBone``, transcribed.

Skirts, sleeves, hair ties and ribbons hang off bone chains that exist only in
the item's own prefab, driven at run time by ``DynamicBone`` components.  This
module reads those components and runs their solver, instead of the one-shot
gravity rotation that used to stand in for it (which is why skirts flared into
a bell and one sleeve stayed horizontal).

**Reading the parameters.**  UnityPy returns nothing for these objects — the
script has no type tree — so the MonoBehaviour is deserialised by field order,
taken from ``Assembly-CSharp``'s field list, which *is* Unity's serialisation
order for public instance fields::

    m_Root(PPtr) m_UpdateRate(f) m_Damping(f) m_DampingDistrib(curve)
    m_Elasticity(f) m_ElasticityDistrib(curve) m_Stiffness(f)
    m_StiffnessDistrib(curve) m_Inert(f) m_InertDistrib(curve)
    m_Radius(f) m_RadiusDistrib(curve) m_EndLength(f) m_EndOffset(v3)
    m_Gravity(v3) m_Force(v3) m_Colliders(list) m_Exclusions(list)
    m_FreezeAxis(int) m_DistantDisable(bool) m_ReferenceObject(PPtr)
    m_DistanceToObject(f) m_notRolls(list)

The check that the layout is right is not that the values look plausible: the
parse has to consume each object's bytes **exactly**.  It does — 96/96 in
``co_bot_00`` (skirts) and 136/136 in ``bo_hair_b_00``, zero mismatches — and
the values that fall out are coherent (skirt: damping 0.15, elasticity 0.10,
stiffness 0.60, inert 0.80, radius 0.03; hair: damping 0.20, stiffness 0.10).

**The solver** is transcribed from ``UpdateParticles1`` / ``UpdateParticles2``
/ ``ApplyParticlesToTransforms``; see each function.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# AnimationCurve
# ---------------------------------------------------------------------------
@dataclass
class Curve:
    """Unity ``AnimationCurve`` — cubic Hermite between keyframes."""
    keys: list = field(default_factory=list)   # (time, value, inSlope, outSlope)

    def __bool__(self) -> bool:
        return bool(self.keys)

    def evaluate(self, t: float) -> float:
        k = self.keys
        if not k:
            return 1.0
        if t <= k[0][0]:
            return k[0][1]
        if t >= k[-1][0]:
            return k[-1][1]
        for i in range(len(k) - 1):
            t0, v0, _i0, o0 = k[i]
            t1, v1, i1, _o1 = k[i + 1]
            if not (t0 <= t <= t1):
                continue
            dt = t1 - t0
            if dt <= 1e-9:
                return v0
            # Unity writes ``Infinity`` for a *stepped* tangent: the segment
            # holds the left key's value until the next one.  Feeding that
            # through the Hermite basis gives 0 × ∞ = NaN, and a NaN particle
            # radius then poisons the position the collider pushes it to —
            # ten baked items came out with NaN bone tracks before this.
            if not (math.isfinite(o0) and math.isfinite(i1)):
                return v0
            u = (t - t0) / dt
            u2, u3 = u * u, u * u * u
            # the standard Hermite basis Unity uses for its tangents
            return ((2 * u3 - 3 * u2 + 1) * v0
                    + (u3 - 2 * u2 + u) * dt * o0
                    + (-2 * u3 + 3 * u2) * v1
                    + (u3 - u2) * dt * i1)
        return k[-1][1]


# ---------------------------------------------------------------------------
# MonoBehaviour deserialisation
# ---------------------------------------------------------------------------
class _R:
    __slots__ = ("b", "p")

    def __init__(self, b: bytes):
        self.b = b
        self.p = 0

    def i(self) -> int:
        v = struct.unpack_from("<i", self.b, self.p)[0]
        self.p += 4
        return v

    def f(self) -> float:
        v = struct.unpack_from("<f", self.b, self.p)[0]
        self.p += 4
        return v

    def boolean(self) -> bool:
        v = self.b[self.p]
        self.p = (self.p + 1 + 3) & ~3
        return bool(v)

    def pptr(self):
        fid = self.i()
        pid = struct.unpack_from("<q", self.b, self.p)[0]
        self.p += 8
        return (fid, pid)

    def v3(self):
        return np.array((self.f(), self.f(), self.f()), dtype=np.float64)

    def string(self) -> str:
        n = self.i()
        s = self.b[self.p:self.p + n].decode("utf-8", "replace")
        self.p = (self.p + n + 3) & ~3
        return s

    def curve(self) -> Curve:
        n = self.i()
        keys = [(self.f(), self.f(), self.f(), self.f()) for _ in range(n)]
        self.i(), self.i(), self.i()          # pre / post infinity, rotation order
        return Curve(keys)

    def pptr_list(self) -> list:
        return [self.pptr() for _ in range(self.i())]


@dataclass
class BoneSpec:
    """One ``DynamicBone`` component's parameters."""
    root: tuple = (0, 0)
    update_rate: float = 60.0
    damping: float = 0.1
    damping_curve: Curve = field(default_factory=Curve)
    elasticity: float = 0.1
    elasticity_curve: Curve = field(default_factory=Curve)
    stiffness: float = 0.1
    stiffness_curve: Curve = field(default_factory=Curve)
    inert: float = 0.0
    inert_curve: Curve = field(default_factory=Curve)
    radius: float = 0.0
    radius_curve: Curve = field(default_factory=Curve)
    end_length: float = 0.0
    end_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    freeze_axis: int = 0
    exclusions: list = field(default_factory=list)
    # Illusion's addition: children ``AppendParticles`` skips as particles,
    # hanging their own children on the skipped one's parent (PPtrs)
    not_rolls: list = field(default_factory=list)
    colliders: list = field(default_factory=list)   # PPtrs, resolved by the loader
    enabled: bool = True                            # the serialised m_Enabled
    # the component's own GameObject: ``transform`` in ``SetupParticles``
    # (``m_ObjectScale``) and ``AppendParticles`` (an end particle's offset)
    game_object: tuple = (0, 0)


def parse_component(raw: bytes) -> BoneSpec | None:
    """Deserialise one ``DynamicBone``; ``None`` if the bytes do not fit.

    Two layouts are accepted (exact byte consume):

    1. **Stock Origin clothes/hair** — no ``m_UpdateMode``, ends with ``m_notRolls``.
    2. **BetterPenetration body packs** (§7.123) — ``int m_UpdateMode`` right
       after ``m_UpdateRate``, and **no** ``m_notRolls`` at the end.  Measured
       on ``oo_base_bpprofundis``: 13/13 of the 320-byte vagina components
       fit layout 2 with ``rem=0``; layout 1 overshoots.
    """
    for layout in ("stock", "bp_update_mode"):
        spec = _parse_component_layout(raw, layout)
        if spec is not None:
            return spec
    return None


def _parse_component_layout(raw: bytes, layout: str) -> BoneSpec | None:
    r = _R(raw)
    try:
        game_object = r.pptr()
        enabled = r.boolean()             # m_Enabled
        r.pptr()                          # m_Script
        r.string()                        # m_Name
        root = r.pptr()
        update_rate = r.f()
        if layout == "bp_update_mode":
            r.i()                         # m_UpdateMode (Normal=0, …)
        spec = BoneSpec(
            root=root,
            update_rate=update_rate,
            damping=r.f(), damping_curve=r.curve(),
            elasticity=r.f(), elasticity_curve=r.curve(),
            stiffness=r.f(), stiffness_curve=r.curve(),
            inert=r.f(), inert_curve=r.curve(),
            radius=r.f(), radius_curve=r.curve(),
            end_length=r.f(),
            end_offset=r.v3(), gravity=r.v3(), force=r.v3(),
        )
        spec.enabled = bool(enabled)
        spec.game_object = game_object
        spec.colliders = r.pptr_list()    # resolved in prefab._colliders_for
        spec.exclusions = r.pptr_list()
        spec.freeze_axis = r.i()
        r.boolean()                       # m_DistantDisable
        r.pptr()                          # m_ReferenceObject
        r.f()                             # m_DistanceToObject
        if layout == "stock":
            spec.not_rolls = r.pptr_list()
    except (struct.error, IndexError):
        return None
    # every byte has to be accounted for, or the field order is wrong
    return spec if r.p == len(raw) else None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class Chain:
    """A particle chain built from one bone hierarchy — BetterPenetration's.

    The garments' chains go through :class:`Tree`, which builds the particle
    list the way ``AppendParticles`` does; this one is the straight chain
    ``better_penetration`` assembles itself.

    ``parent[i]`` indexes this array; index 0 is the pinned root.  ``local``
    holds each particle's rest position in its parent's frame, which is what
    ``UpdateParticles2`` reconstructs the target from.
    """
    pos: np.ndarray                 # (n,3) current world positions
    prev: np.ndarray                # (n,3) previous positions (verlet)
    parent: np.ndarray              # (n,) int
    local: np.ndarray               # (n,3) rest offset from parent, parent frame
    rest_len: np.ndarray            # (n,) |local|
    damping: np.ndarray
    elasticity: np.ndarray
    stiffness: np.ndarray
    inert: np.ndarray
    radius: np.ndarray
    # parent orientation, updated as the chain moves; column-major 3x3 per bone
    basis: np.ndarray               # (n,3,3)


def step(chain: Chain, spec: BoneSpec, *, object_move=None,
         object_scale: float = 1.0, weight: float = 1.0,
         colliders: list | None = None) -> None:
    """One solver iteration — ``UpdateParticles1`` then ``UpdateParticles2``.

    ``UpdateParticles1``::

        force = m_Gravity
        fdir  = force.normalized
        proj  = fdir * max(Dot(rootDir(m_LocalGravity), fdir), 0)
        force = (m_Gravity - proj + m_Force) * m_ObjectScale
        per particle with a parent:
            v     = pos - prev
            rmove = m_ObjectMove * inert
            prev  = pos + rmove
            pos  += v * (1 - damping) + force + rmove

    ``UpdateParticles2`` then pulls each particle back toward the position its
    parent's frame says it should have (elasticity), limits how far it may
    stray (stiffness), and finally restores the segment length exactly.
    """
    om = np.zeros(3) if object_move is None else np.asarray(object_move, float)
    g = np.asarray(spec.gravity, dtype=np.float64)
    gn = float(np.linalg.norm(g))
    # m_LocalGravity is the root-frame rest gravity; with the root upright it
    # projects onto itself, so the term cancels and only m_Force remains.
    force = (g + np.asarray(spec.force, dtype=np.float64)) * object_scale

    par = chain.parent
    for i in range(len(chain.pos)):
        if par[i] < 0:
            chain.prev[i] = chain.pos[i]
            continue
        v = chain.pos[i] - chain.prev[i]
        rmove = om * chain.inert[i]
        chain.prev[i] = chain.pos[i] + rmove
        chain.pos[i] += v * (1.0 - chain.damping[i]) + force + rmove

    for i in range(len(chain.pos)):
        p = int(par[i])
        if p < 0:
            continue
        rest = chain.pos[p] + chain.basis[p] @ chain.local[i]
        stiff = 1.0 + (chain.stiffness[i] - 1.0) * weight   # Lerp(1, k, weight)
        if stiff > 0.0 or chain.elasticity[i] != 0.0:
            d = rest - chain.pos[i]
            chain.pos[i] = chain.pos[i] + d * chain.elasticity[i]
            if stiff > 0.0:
                d = rest - chain.pos[i]
                ln = float(np.linalg.norm(d))
                maxlen = chain.rest_len[i] * (1.0 - stiff) * 2.0
                if ln > maxlen > 0.0:
                    chain.pos[i] = chain.pos[i] + d * ((ln - maxlen) / ln)
        if colliders:
            # ``UpdateParticles2`` runs the colliders here, between the
            # stiffness clamp and the freeze axis, with the particle's own
            # distributed radius scaled by the object.
            pr = float(chain.radius[i]) * object_scale
            for col in colliders:
                collide(chain.pos[i], pr, col)
        if spec.freeze_axis:
            # project onto the plane through the parent whose normal is the
            # parent's right / up / forward axis (FreezeAxis 1 / 2 / 3)
            nrm = chain.basis[p][:, spec.freeze_axis - 1]
            chain.pos[i] -= nrm * float(np.dot(nrm, chain.pos[i] - chain.pos[p]))
        dd = chain.pos[p] - chain.pos[i]
        ll = float(np.linalg.norm(dd))
        if ll > 1e-9:
            chain.pos[i] += dd * ((ll - chain.rest_len[i]) / ll)
    # Note the ordering: Origin re-orients the transforms in a *separate* pass
    # (``ApplyParticlesToTransforms``), after the whole constraint loop, and
    # the frames it rotates are re-posed from the animation before the next
    # step.  Rotating inside the loop makes the elasticity/stiffness target
    # follow the particle it is supposed to restrain, so stiffness stops doing
    # anything at all — a chain with stiffness 0.95 fell as far as one with
    # 0.05.  ``apply()`` is that second pass; callers run it when they want
    # the orientations, not per iteration.
    del gn


def apply(chain: Chain) -> np.ndarray:
    """``ApplyParticlesToTransforms`` — the solved orientation per bone.

    Returns the basis array so a caller can skin from it.  Run once the chain
    has been stepped, never between constraint iterations.
    """
    for i in range(len(chain.pos)):
        p = int(chain.parent[i])
        if p >= 0:
            _reorient(chain, p, i)
    return chain.basis


def _reorient(chain: Chain, p: int, i: int) -> None:
    """``ApplyParticlesToTransforms``: rotate the parent to follow the child.

    ``Quaternion.FromToRotation(parent.TransformDirection(localPos), v)`` —
    the parent's frame is turned by the rotation that takes the rest direction
    onto the direction the solved particles now describe.
    """
    rest = chain.basis[p] @ chain.local[i]
    cur = chain.pos[i] - chain.pos[p]
    a = float(np.linalg.norm(rest))
    b = float(np.linalg.norm(cur))
    if a < 1e-9 or b < 1e-9:
        return
    u, v = rest / a, cur / b
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return
    axis = np.cross(u, v)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:                              # opposite: any perpendicular axis
        axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(u, np.array([0.0, 1.0, 0.0]))
        s = float(np.linalg.norm(axis))
    axis = axis / s
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    ang = float(np.arccos(c))
    rot = np.eye(3) + np.sin(ang) * k + (1.0 - np.cos(ang)) * (k @ k)
    chain.basis[p] = rot @ chain.basis[p]


# ---------------------------------------------------------------------------
# DynamicBoneCollider
# ---------------------------------------------------------------------------
# Serialised field order from ``Assembly-CSharp``, same rule as the bone
# itself: ``m_Center(v3) m_Radius(f) m_Height(f) m_Direction(int)
# m_Bound(int)``, after the MonoBehaviour header.  A parse is accepted only if
# it consumes the object exactly.
DIR_X, DIR_Y, DIR_Z = 0, 1, 2
BOUND_OUTSIDE, BOUND_INSIDE = 0, 1


@dataclass
class Collider:
    """One ``DynamicBoneCollider``, already placed in world space.

    ``Collide`` is transcribed below; the placement (``TransformPoint`` of the
    centre, and ``m_Radius * abs(lossyScale.z)`` — the **z** component whatever
    the direction axis is) is done by the caller, which has the prefab's
    transforms.
    """
    centre: np.ndarray
    radius: float = 0.0
    height: float = 0.0
    direction: int = DIR_Y
    bound: int = BOUND_OUTSIDE
    # world-space capsule ends, filled in by :func:`place_collider`
    a: np.ndarray | None = None
    b: np.ndarray | None = None
    world_radius: float = 0.0


def parse_collider(raw: bytes) -> Collider | None:
    """Deserialise one ``DynamicBoneCollider``; ``None`` if the bytes do not fit."""
    r = _R(raw)
    try:
        r.pptr()                          # m_GameObject
        r.boolean()                       # m_Enabled
        r.pptr()                          # m_Script
        r.string()                        # m_Name
        col = Collider(centre=r.v3(), radius=r.f(), height=r.f(),
                       direction=r.i(), bound=r.i())
    except (struct.error, IndexError):
        return None
    return col if r.p == len(raw) else None


def mirrored(col: Collider) -> Collider:
    """*col* read in Unity's space, with ``m_Center`` carried into this project's (X mirrored).

    The transforms are mirrored by conjugation (``S·M·S``), so a local point ``c``
    the game places with ``TransformPoint`` is ``S·c`` on our transform.
    """
    centre = np.asarray(col.centre, dtype=np.float64) * np.array([-1.0, 1.0, 1.0])
    return Collider(centre=centre, radius=col.radius, height=col.height,
                    direction=col.direction, bound=col.bound)


def collider_from_tree(tree: dict) -> Collider | None:
    """Same fields, when the bundle happens to carry a type tree.

    ``oo_base`` does and the clothes bundles do not, which is why the byte-order
    parse above exists at all.  Prefer this where it works: it cannot be thrown
    off by a field being added in a future build.
    """
    try:
        centre = tree["m_Center"]
        return Collider(
            centre=np.array([float(centre["x"]), float(centre["y"]),
                             float(centre["z"])]),
            radius=float(tree["m_Radius"]), height=float(tree["m_Height"]),
            direction=int(tree["m_Direction"]), bound=int(tree["m_Bound"]))
    except (KeyError, TypeError, ValueError):
        return None


def place_collider(col: Collider, world: np.ndarray) -> Collider:
    """Resolve a collider's world-space capsule from its transform.

    ``Collide``::

        radius = m_Radius * abs(lossyScale.z)
        h      = (m_Height - m_Radius) * 0.5
        h <= 0                    -> sphere at TransformPoint(m_Center)
        otherwise                 -> capsule between TransformPoint(centre -/+ h
                                     on the m_Direction axis)
    """
    m = np.asarray(world, dtype=np.float64)
    scale_z = float(np.linalg.norm(m[:3, 2])) or 1.0
    out = Collider(centre=col.centre, radius=col.radius, height=col.height,
                   direction=col.direction, bound=col.bound)
    out.world_radius = col.radius * abs(scale_z)
    h = (col.height - col.radius) * 0.5

    def point(local):
        return m[:3, :3] @ np.asarray(local, dtype=np.float64) + m[:3, 3]

    if h <= 0.0:
        out.a = out.b = point(col.centre)
        return out
    c0 = np.array(col.centre, dtype=np.float64)
    c1 = np.array(col.centre, dtype=np.float64)
    axis = min(max(int(col.direction), 0), 2)
    c0[axis] -= h
    c1[axis] += h
    out.a, out.b = point(c0), point(c1)
    return out


def collide_rows(pos: np.ndarray, particle_radius, col: Collider) -> None:
    """:func:`collide` over the rows of *pos* at once, in place, same branches.

    *particle_radius* is a scalar or one radius per row.
    """
    if col.a is None:
        return
    r = col.world_radius + np.asarray(particle_radius, dtype=np.float64)
    r2 = r * r
    inside = col.bound == BOUND_INSIDE
    if col.b is None or np.array_equal(col.a, col.b):
        near = np.broadcast_to(col.a, pos.shape)
        dv = pos - col.a
    else:
        seg = col.b - col.a
        d = pos - col.a
        t = d @ seg
        seg2 = float(seg @ seg)
        near = np.where((t <= 0.0)[:, None], col.a,
                        np.where((t >= seg2)[:, None], col.b,
                                 col.a + seg * (t / seg2)[:, None]))
        dv = np.where((t <= 0.0)[:, None], d, pos - near)
    d2 = np.einsum("ij,ij->i", dv, dv)
    hit = (d2 > r2) if inside else ((d2 > 0.0) & (d2 < r2))
    if hit.any():
        rr = np.broadcast_to(r, d2.shape)
        pos[hit] = near[hit] + dv[hit] * (rr[hit] / np.sqrt(d2[hit]))[:, None]


def collide(pos: np.ndarray, particle_radius: float, col: Collider) -> bool:
    """``DynamicBoneCollider.Collide`` — push *pos* out of (or into) the shape.

    Modifies ``pos`` in place, returns whether it moved.  ``OutsideSphere`` and
    ``OutsideCapsule`` transcribed; the ``Inside`` pair keeps a particle within
    the shape: the same ``radius + particleRadius`` (Origin's build adds, IL
    ``ldarg.3; ldarg.1; add``) and the same projection onto it, taken only when
    the particle is *outside* (``ble.un`` skips while ``len² <= r²``).
    """
    if col.a is None:
        return False
    r = col.world_radius + particle_radius
    r2 = r * r
    inside = col.bound == BOUND_INSIDE

    if col.b is None or np.array_equal(col.a, col.b):
        d = pos - col.a
        d2 = float(d @ d)
        if d2 <= 0.0:
            return False
        if (d2 <= r2) if inside else (d2 >= r2):
            return False
        pos[:] = col.a + d * (r / np.sqrt(d2))
        return True

    # capsule: which Voronoi region of the segment the particle is in
    seg = col.b - col.a
    d = pos - col.a
    t = float(d @ seg)
    if t <= 0.0:
        near, dv = col.a, d
    else:
        seg2 = float(seg @ seg)
        if t >= seg2:
            dv = pos - col.b
            near = col.b
        else:
            near = col.a + seg * (t / seg2)
            dv = pos - near
    d2 = float(dv @ dv)
    if d2 <= 0.0:
        return False
    if (d2 <= r2) if inside else (d2 >= r2):
        return False
    pos[:] = near + dv * (r / np.sqrt(d2))
    return True


# ---------------------------------------------------------------------------
# The particle tree and its solver, as ``DynamicBone`` builds and runs it
# ---------------------------------------------------------------------------
#: ``m_UpdateRate`` of every shipped garment component.
UPDATE_RATE = 60.0
#: ``UpdateDynamicBones``: ``if (++loop >= 3) { m_Time = 0; break; }``.
MAX_STEPS = 3
#: Steps a fresh tree runs from the authored rest before it is drawn.
SETTLE_STEPS = 240
#: Largest particle step (metres) at which a tree counts as at rest: 0.6 mm/s at
#: 60 Hz, and with the damping every shipped garment has (≥ 0.1 a step) what is
#: left to travel is a tenth of a millimetre.  The game keeps stepping a tree
#: at rest; a step there moves nothing.
REST_STEP = 1e-5


def end_offset(spec: BoneSpec, leaf_world, leaf_parent_world, component_world) -> np.ndarray:
    """``AppendParticles``' ``m_EndOffset`` for the end particle under a leaf.

    In the leaf's local space::

        m_EndLength > 0:  leaf.InverseTransformPoint(2·leaf.position − parent.position)
                          × m_EndLength            ((m_EndLength, 0, 0) without a parent)
        otherwise:        leaf.InverseTransformPoint(transform.TransformDirection(m_EndOffset)
                                                     + leaf.position)

    where ``transform`` is the component's own (rotation only, as
    ``TransformDirection`` is).
    """
    leaf = np.asarray(leaf_world, dtype=np.float64)
    inv = np.linalg.inv(leaf)
    if spec.end_length > 0.0:
        if leaf_parent_world is None:
            return np.array([float(spec.end_length), 0.0, 0.0])
        x = leaf[:3, 3] * 2.0 - np.asarray(leaf_parent_world, dtype=np.float64)[:3, 3]
        return (inv[:3, :3] @ x + inv[:3, 3]) * float(spec.end_length)
    rot = _ortho(np.asarray(component_world, dtype=np.float64)[None, :3, :3])[0]
    x = rot @ np.asarray(spec.end_offset, dtype=np.float64) + leaf[:3, 3]
    return inv[:3, :3] @ x + inv[:3, 3]


def has_end(spec: BoneSpec) -> bool:
    """Whether ``AppendParticles`` hangs an end particle under a particle it added no child particle to."""
    return spec.end_length > 0.0 or bool(np.any(np.asarray(spec.end_offset) != 0.0))


def particle_tree(root, children, world, parent_of, spec: BoneSpec, *,
                  excluded=frozenset(), not_rolls=frozenset(), component_world=None):
    """``AppendParticles`` from *root*: ``(nodes, parents, end_offsets, bone_length, problem)``.

    *children(node)* is a transform's children in order (all of them),
    *world(node)* its world matrix, *parent_of(node)* its parent or ``None``.
    A node of ``None`` is an end particle.

    The install replaces ``DynamicBone.AppendParticles``: ``IllusionFixes/
    KK_Fix_DynamicBones.dll``'s ``Hooks.DivertAppendParticles`` is a
    ``[HarmonyPrefix]`` returning false that calls its own
    ``Hooks.AppendParticles`` instead, transcribed here::

        add the particle; boneLength += |parent.position − position|
        b = the node; appended = false
        while b has children:
            for each child of b: skip it if excluded; else (m_notRolls being a list)
                hit = child in m_notRolls; last = its index  — both overwritten by
                every child that is not excluded; if not hit: recurse, appended = true
            if not hit: stop                 (only the *last* such child counts)
            b = b's child at last            (go down the notRoll, again and again)
        if not appended and (m_EndLength > 0 or m_EndOffset ≠ 0): add an end particle

    So an end particle hangs under every particle no child particle was added to
    — a strand whose last bones are all notRolls gets one under its last
    particle, where the stock method (``childCount == 0``) gave none.

    A particle under a skipped notRoll is the one layout where a particle's
    parent particle is not its transform's parent, and the targets and the
    write-back would then need the transform's own local position in the
    other frame.  None of the 466 components in the shipped garments has one
    (§7.540), so it is not transcribed: *problem* names it and the caller
    leaves the chain alone.
    """
    nodes: list = []
    parents: list = []
    ends: list = []
    lengths: list = []
    problem = []
    end = has_end(spec)

    def append(node, parent: int, length: float) -> None:
        off = np.zeros(3)
        if node is None:
            leaf = nodes[parent]
            up = parent_of(leaf)
            off = end_offset(spec, world(leaf), None if up is None else world(up), component_world)
            m = np.asarray(world(leaf), dtype=np.float64)
            pos = m[:3, :3] @ off + m[:3, 3]
        else:
            pos = np.asarray(world(node), dtype=np.float64)[:3, 3]
        if parent >= 0:
            length += float(np.linalg.norm(np.asarray(world(nodes[parent]))[:3, 3] - pos))
        index = len(nodes)
        nodes.append(node)
        parents.append(parent)
        ends.append(off)
        lengths.append(length if parent >= 0 else 0.0)
        if node is None:
            return
        appended = False
        b, under_roll = node, False
        while True:
            kids = list(children(b))
            if not kids:
                break
            hit, last = False, 0
            for i, k in enumerate(kids):
                if k in excluded:
                    continue
                hit, last = k in not_rolls, i
                if hit:
                    continue
                if under_roll:
                    problem.append(f"m_notRolls 底下還有粒子（{k}）")
                append(k, index, length)
                appended = True
            if not hit:
                break
            b, under_roll = kids[last], True
        if not appended and end:
            append(None, index, length)

    append(root, -1, 0.0)
    return (nodes, parents, np.array(ends, dtype=np.float64).reshape(-1, 3),
            np.array(lengths, dtype=np.float64), "；".join(problem))


def distribute(spec: BoneSpec, bone_length) -> dict:
    """Per-particle parameters — ``SetupParticles``'s curve distribution.

    Each particle takes the component's base value, multiplied by the matching
    ``*Distrib`` curve at ``m_BoneLength / m_BoneTotalLength`` when the tree
    has length and the curve has keys; ``m_BoneLength`` is the path from the
    root (``AppendParticles``) and the total the longest of them.  Then
    ``Clamp01`` on all but the radius, which only cannot go below 0.
    """
    bl = np.asarray(bone_length, dtype=np.float64)
    total = float(bl.max()) if len(bl) else 0.0
    a = bl / total if total > 0.0 else np.zeros(len(bl))
    out = {}
    for name in ("damping", "elasticity", "stiffness", "inert", "radius"):
        v = np.full(len(bl), float(getattr(spec, name)), dtype=np.float64)
        curve = getattr(spec, name + "_curve")
        if total > 0.0 and curve:
            v = v * np.array([curve.evaluate(float(x)) for x in a])
        out[name] = np.maximum(v, 0.0) if name == "radius" else np.clip(v, 0.0, 1.0)
    return out


def _ortho(m: np.ndarray) -> np.ndarray:
    """The rotation part of each matrix (columns normalised)."""
    n = np.linalg.norm(m, axis=-2, keepdims=True)
    return m / np.where(n > 1e-12, n, 1.0)


def rotations(T: np.ndarray) -> np.ndarray:
    """The rotation part of each 4×4 in *T*, the frames :class:`Tree` turns."""
    return _ortho(np.asarray(T, dtype=np.float64)[:, :3, :3])


def _from_to(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """``Quaternion.FromToRotation(v0, v1)`` as 3×3 matrices, one per row."""
    out = np.broadcast_to(np.eye(3), (len(v0), 3, 3)).copy()
    a = np.linalg.norm(v0, axis=1)
    b = np.linalg.norm(v1, axis=1)
    ok = (a > 1e-12) & (b > 1e-12)
    if not ok.any():
        return out
    u = v0[ok] / a[ok, None]
    v = v1[ok] / b[ok, None]
    c = np.clip(np.einsum("ij,ij->i", u, v), -1.0, 1.0)
    axis = np.cross(u, v)
    s = np.linalg.norm(axis, axis=1)
    flip = (s < 1e-12) & (c < 0.0)
    if flip.any():
        # opposite vectors: half a turn about any axis square to them
        alt = np.cross(u[flip], np.array([1.0, 0.0, 0.0]))
        weak = np.linalg.norm(alt, axis=1) < 1e-9
        alt[weak] = np.cross(u[flip][weak], np.array([0.0, 1.0, 0.0]))
        axis[flip] = alt
    n = np.linalg.norm(axis, axis=1)
    axis = axis / np.where(n > 1e-12, n, 1.0)[:, None]
    k = np.zeros((len(axis), 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -axis[:, 2], axis[:, 1]
    k[:, 1, 0], k[:, 1, 2] = axis[:, 2], -axis[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -axis[:, 1], axis[:, 0]
    sin = np.where(flip, 0.0, s)
    rot = np.eye(3) + sin[:, None, None] * k + (1.0 - c)[:, None, None] * (k @ k)
    rot[(s < 1e-12) & (c > 0.0)] = np.eye(3)
    out[ok] = rot
    return out


class PlacedColliders:
    """One frame's placed colliders, in order, with what the cull needs."""

    def __init__(self, caps: list):
        self.caps = [c for c in caps if c.a is not None]
        self.n = len(self.caps)
        if not self.n:
            return
        self.a = np.array([c.a for c in self.caps])
        b = np.array([c.a if c.b is None else c.b for c in self.caps])
        self.seg = b - self.a
        self.seg2 = np.einsum("ij,ij->i", self.seg, self.seg)
        self.radius = np.array([c.world_radius for c in self.caps])
        self.inside = np.array([c.bound == BOUND_INSIDE for c in self.caps])

    def collide(self, rows: np.ndarray, particle_radius: np.ndarray) -> None:
        """``DynamicBoneCollider.Collide`` for every collider in turn, in place.

        A collider is run only where some row could touch it: a row's distance
        to a capsule's axis moves no more than the row does, so one that starts
        ``slack`` beyond its reach stays out until the pushes before it have
        moved it that far.  ``Inside`` colliders are always run.
        """
        d = rows[:, None, :] - self.a[None]
        t = np.einsum("nkj,kj->nk", d, self.seg) / np.where(self.seg2 > 0.0, self.seg2, 1.0)
        near = self.a[None] + self.seg[None] * np.clip(t, 0.0, 1.0)[:, :, None]
        dist = np.linalg.norm(rows[:, None, :] - near, axis=2)
        slack = dist - (self.radius[None, :] + particle_radius[:, None])
        moved = np.zeros(len(rows))
        k = 0
        while k < self.n:
            hit = np.flatnonzero((slack[:, k:] < moved[:, None]).any(axis=0) | self.inside[k:])
            if not hit.size:
                return
            k += int(hit[0])
            before = rows.copy()
            collide_rows(rows, particle_radius, self.caps[k])
            moved += np.linalg.norm(rows - before, axis=1)
            k += 1


class Tree:
    """The particles of one or more components, and their state.

    Built from chain specs — per chain ``parent``, the distributed per-particle
    ``damping`` / ``elasticity`` / ``stiffness`` / ``inert`` / ``radius``,
    ``end`` / ``end_offset``, ``force``, ``gravity``, ``freeze_axis`` and
    ``collide`` (``prefab.sway_chain_specs``) — and solved from the particles'
    transform matrices: ``T[i]`` is particle *i*'s transform, an end
    particle's being its leaf's (its position is ``TransformPoint(m_EndOffset)``).
    A particle's parent particle is its transform's parent (:func:`particle_tree`
    refuses the one layout where it is not), so a rest offset is the
    difference of two positions.  The same core runs the load-time settle (the
    prefab's rest, held) and the per-frame solve (the posed skeleton).

    *single* is ``childCount <= 1`` for each particle's transform — whether
    ``ApplyParticlesToTransforms`` turns it; *setup* the transforms when the
    component set up (``m_LocalGravity`` comes from the roots there).
    """

    def __init__(self, chains: list, single, setup):
        parent, chain_of, end, offs = [], [], [], []
        rows: dict = {k: [] for k in ("damping", "elasticity", "stiffness", "inert", "radius")}
        force, gravity, freeze, collide = [], [], [], []
        self.roots: list = []
        for c, spec in enumerate(chains):
            base = len(parent)
            par = [int(x) for x in spec["parent"]]
            n = len(par)
            parent.extend(base + x if x >= 0 else -1 for x in par)
            chain_of.extend([c] * n)
            flags = spec.get("end")
            end.extend([False] * n if flags is None else [bool(x) for x in flags])
            off = spec.get("end_offset")
            offs.extend(np.zeros((n, 3)) if off is None
                        else np.asarray(off, dtype=np.float64).reshape(n, 3))
            for k in rows:
                rows[k].extend(float(x) for x in np.asarray(spec[k]).ravel())
            self.roots.append(base)
            force.append(np.asarray(spec.get("force", (0, 0, 0)), dtype=np.float64))
            gravity.append(np.asarray(spec.get("gravity", (0, 0, 0)), dtype=np.float64))
            freeze.append(int(spec.get("freeze_axis", 0) or 0))
            collide.append(bool(spec.get("collide")))
        self.parent = np.array(parent, dtype=np.int64)
        self.chain = np.array(chain_of, dtype=np.int64)
        self.end = np.array(end, dtype=bool)
        self.end_offset = np.array(offs, dtype=np.float64).reshape(-1, 3)
        for k, v in rows.items():
            setattr(self, k, np.array(v, dtype=np.float64))
        self.force = np.array(force, dtype=np.float64).reshape(-1, 3)
        self.gravity = np.array(gravity, dtype=np.float64).reshape(-1, 3)
        self.freeze = np.array(freeze, dtype=np.int64)
        collide = np.array(collide, dtype=bool)
        depth = np.zeros(len(parent), dtype=np.int64)
        for i, p in enumerate(self.parent):
            depth[i] = 0 if p < 0 else depth[p] + 1
        self.levels = [np.flatnonzero(depth == d) for d in range(1, int(depth.max(initial=0)) + 1)]
        #: per chain, which of the collider sets handed to :meth:`advance` it is solved against
        #: (its ``m_Colliders``); −1 for none.  A spec's ``caps``, else set 0 when it ``collide``s.
        self.cap_set = np.array([int(spec["caps"]) if spec.get("caps") is not None
                                 else (0 if c else -1) for spec, c in zip(chains, collide)], dtype=np.int64)
        self.single = np.asarray(single, dtype=bool)
        # ``m_LocalGravity = m_Root.InverseTransformDirection(m_Gravity)``
        setup = np.asarray(setup, dtype=np.float64)
        self.local_gravity = np.einsum("cji,cj->ci", _ortho(setup[self.roots, :3, :3]),
                                       self.gravity) if self.roots else np.zeros((0, 3))
        self.pos = self.prev = None
        self._time = 0.0

    def __len__(self) -> int:
        return len(self.parent)

    # ------------------------------------------------------------------ frames
    def positions(self, T: np.ndarray) -> np.ndarray:
        """Where each particle's transform is: an end particle at its offset."""
        P = T[:, :3, 3].copy()
        if self.end.any():
            e = T[self.end]
            P[self.end] = np.einsum("nij,nj->ni", e[:, :3, :3], self.end_offset[self.end]) + e[:, :3, 3]
        return P

    def reset(self) -> None:
        """Start over from the authored rest on the next solve."""
        self.pos = self.prev = None
        self._time = 0.0

    def place(self, T: np.ndarray) -> None:
        """``ResetParticlesPosition``: every particle where its transform puts it, at rest.

        ``m_Time`` is left as it is (the method does not touch it), and no settle follows:
        the next step starts from the transforms.
        """
        P = self.positions(T)
        self.pos, self.prev = P.copy(), P.copy()

    def moving(self) -> float:
        """The largest particle step of the last solve (inf before the first)."""
        if self.pos is None:
            return float("inf")
        return float(np.abs(self.pos - self.prev).max()) if len(self.pos) else 0.0

    def settle(self, T: np.ndarray, caps: PlacedColliders, scale: float = 1.0,
               steps: int = SETTLE_STEPS) -> None:
        """From the authored rest, *steps* steps against frames held still."""
        P, R = self.positions(T), rotations(T)
        self.pos, self.prev = P.copy(), P.copy()
        for _ in range(int(steps)):
            self._step(P, R, caps, np.zeros(3), scale)
        self._time = 0.0

    def _per_chain(self, move, scale) -> tuple:
        """``m_ObjectMove`` ``(chains, 3)`` and ``m_ObjectScale`` ``(chains,)``: one for all, or one per chain."""
        c = len(self.roots)
        move = np.broadcast_to(np.asarray(move, dtype=np.float64).reshape(-1, 3), (c, 3)).copy()
        scale = np.broadcast_to(np.asarray(scale, dtype=np.float64).reshape(-1), (c,)).copy()
        return move, scale

    def advance(self, T: np.ndarray, caps, move, scale, dt: float) -> None:
        """One frame of ``UpdateDynamicBones``: as many 60 Hz steps as *dt* holds.

        *move* / *scale* are ``m_ObjectMove`` / ``m_ObjectScale``: one for every
        chain, or one per chain (each component measures its own object).
        *caps* is one :class:`PlacedColliders` for every colliding chain, or a
        list of them indexed by the chains' ``caps`` set (:attr:`cap_set`).

        A fresh tree settles first, so a single still frame is not the first
        frame of a fall.  With no step due nothing moves: the install's
        ``IllusionFixes/KK_Fix_DynamicBones.dll`` makes ``SkipUpdateParticles`` a no-op
        (``Hooks.SkipUpdateParticles``, a ``[HarmonyPrefix]`` that is ``ldc.i4.0; ret``),
        so the particles stay where the last step left them — no ``m_ObjectMove``, no
        stiffness or length pass — and only ``ApplyParticlesToTransforms`` runs.
        """
        move, scale = self._per_chain(move, scale)
        if self.pos is None:
            self.settle(T, caps, scale)
            return
        P, R = self.positions(T), rotations(T)
        self._time += max(0.0, float(dt))
        step = 1.0 / UPDATE_RATE
        n = 0
        while self._time >= step:
            self._time -= step
            n += 1
            if n >= MAX_STEPS:
                self._time = 0.0
                break
        for k in range(n):
            self._step(P, R, caps, move if k == 0 else np.zeros_like(move), scale)

    def _step(self, P, R, caps, move, scale) -> None:
        """``UpdateParticles1`` then ``UpdateParticles2``, every chain at once.

        *move* ``(chains, 3)`` and *scale* ``(chains,)`` are each component's own
        ``m_ObjectMove`` and ``m_ObjectScale``.
        """
        move, scale = self._per_chain(move, scale)
        root = self.parent < 0
        kid = ~root
        # UpdateParticles1 — the force, with m_LocalGravity's projection
        g = self.gravity
        gn = np.linalg.norm(g, axis=1, keepdims=True)
        fdir = np.where(gn > 1e-12, g / np.where(gn > 1e-12, gn, 1.0), 0.0)
        rf = np.einsum("cij,cj->ci", R[self.roots], self.local_gravity) \
            if len(self.roots) else np.zeros((0, 3))
        proj = fdir * np.maximum(np.einsum("ci,ci->c", rf, fdir), 0.0)[:, None]
        force = (g - proj + self.force) * scale[:, None]
        v = self.pos - self.prev
        rmove = move[self.chain] * self.inert[:, None]
        prev = self.pos + rmove
        pos = self.pos + v * (1.0 - self.damping)[:, None] + force[self.chain] + rmove
        self.prev = np.where(kid[:, None], prev, self.pos)
        pos[root] = P[root]
        # UpdateParticles2 — level by level, parents before children
        for li, idx in enumerate(self.levels):
            par = self.parent[idx]
            off = P[idx] - P[par]
            rest_len = np.linalg.norm(off, axis=1)
            rest = pos[par] + off
            el = self.elasticity[idx]
            st = self.stiffness[idx]
            active = (st > 0.0) | (el > 0.0)
            q = pos[idx]
            q = np.where(active[:, None], q + (rest - q) * el[:, None], q)
            d = rest - q
            ln = np.linalg.norm(d, axis=1)
            maxlen = rest_len * (1.0 - st) * 2.0
            pull = active & (st > 0.0) & (ln > maxlen) & (ln > 1e-12)
            if pull.any():
                q[pull] = q[pull] + d[pull] * ((ln[pull] - maxlen[pull]) / ln[pull])[:, None]
            sets = self.cap_set[self.chain[idx]]
            for k, placed in enumerate(caps if isinstance(caps, (list, tuple)) else [caps]):
                hit = sets == k
                if not placed.n or not hit.any():
                    continue
                rows = q if hit.all() else q[hit]
                placed.collide(rows, self.radius[idx][hit] * scale[self.chain[idx][hit]])
                if not hit.all():
                    q[hit] = rows
            fa = self.freeze[self.chain[idx]]
            for axis in (1, 2, 3):
                m = fa == axis
                if m.any():
                    nrm = R[par[m]][:, :, axis - 1]
                    dot = np.einsum("ij,ij->i", nrm, q[m] - pos[par[m]])
                    q[m] = q[m] - nrm * dot[:, None]
            dd = pos[par] - q
            ll = np.linalg.norm(dd, axis=1)
            ok = ll > 1e-12
            q[ok] = q[ok] + dd[ok] * ((ll[ok] - rest_len[ok]) / ll[ok])[:, None]
            pos[idx] = q
        self.pos = pos

    def turn(self, T: np.ndarray, local: np.ndarray | None = None) -> tuple:
        """``ApplyParticlesToTransforms``: ``(world rotations, position gaps)``.

        *local*: each particle's ``m_LocalPosition`` — its transform's local position, which the
        game stores when the particles are set up.  Without it that is recovered from the parent's
        world matrix, which a bone scaled to 0 (a card hiding a part through ABMX) makes singular;
        ``TransformDirection`` ignores the scale, so in the game such a parent turns all the same.

        Parents first, a transform with at most one child turns by
        ``FromToRotation(TransformDirection(child.localPosition), child − self)``
        (an end particle's ``m_EndOffset`` in place of the local position), and
        a child rides on its parent's turn.  The game then sets every child's
        position to its particle; *gaps* holds the ones that are not already
        there (under a parent that did not turn), as world vectors.
        """
        P, R = self.positions(T), rotations(T)
        inv = None if local is not None else np.linalg.inv(T)
        solved = R.copy()
        gaps: dict = {}
        for idx in self.levels:
            par = self.parent[idx]
            if local is not None:
                lp = np.asarray(local, dtype=np.float64)[idx].copy()
            else:
                lp = np.einsum("nij,nj->ni", inv[par][:, :3, :3], P[idx]) + inv[par][:, :3, 3]
            lp[self.end[idx]] = self.end_offset[idx][self.end[idx]]
            turn = self.single[par]
            if turn.any():
                i, p = idx[turn], par[turn]
                v0 = np.einsum("nij,nj->ni", solved[p], lp[turn])
                solved[p] = _from_to(v0, self.pos[i] - self.pos[p]) @ solved[p]
            carry = solved[par] @ np.swapaxes(R[par], 1, 2)
            solved[idx] = carry @ R[idx]
            # where the turns alone put each child (a root keeps its place)
            at = np.where((self.parent[par] < 0)[:, None], P[par], self.pos[par]) \
                + np.einsum("nij,nj->ni", carry, P[idx] - P[par])
            gap = self.pos[idx] - at
            far = ~self.end[idx] & (np.einsum("ij,ij->i", gap, gap) > 1e-14)
            for i, g in zip(idx[far], gap[far]):
                gaps[int(i)] = g
        return solved, gaps
