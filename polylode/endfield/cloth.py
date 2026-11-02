"""Bone cloth for a converted Endfield item: a chain solver of this project's own.

The game simulates hair, skirts, coats and ribbons with a modified copy of a
commercial cloth package; its ``BeyondBoneCloth`` components came through the
converter whole (``extra["components"]``), but the solver is native code with
no source to read.  So **nothing here is a transcription**.  The parameters are
used in the sense the package's public documentation gives them — the
documentation says what each one is for and gives no formula — and the
arithmetic is ours.  It makes cloth that hangs, lags and settles; it is not a
claim about how far or how fast the game's does (``Animation_Progress.md`` G8
lists what would have to be measured in the game to say so).

What is taken from the data, per component:

* **the chains** — every bone below a root bone (``rootBones``) is a particle,
  joined to its parent bone and to nothing else (``connectionMode`` 0, the
  only mode the game uses);
* **which particles are held** — the component's painted points
  (``selectionData``: 1 fixed, 2 moving, 0 not simulated) matched to bones by
  position; a bone with no painted point within the component's
  ``maxConnectionDistance`` is fixed if it is a root and moving otherwise;
* **depth** — a particle's distance along its chain from the fixed point it
  hangs from, over the longest such distance in the component; a parameter
  with a curve is ``value × curve(depth)``;
* gravity and its falloff, damping, the particle radius, world and local
  inertia with their speed limits and the depth inertia, the tether's
  compression, distance stiffness, angle restoration (stiffness, velocity
  attenuation, gravity falloff), the angle limit, the animation-pose ratio, the
  two rotation controls, teleport, and the sphere, capsule and plane colliders
  with friction.

Movement limits use spheres around the mixed reference pose: max distance
keeps a particle inside its sphere; backstop keeps it outside a sphere behind
the reference normal, with its nearest surface at ``backstopDistance``.
The semantics follow https://magicasoft.jp/en/mc2_magicacloth_movementlimit/
and https://magicasoft.jp/en/mc2_magicacloth_normalsettings/; the projection
and velocity handling are this project's arithmetic.

World inertia is smoothed with an exponential average of velocity; centrifugal
augmentation uses omega x (omega x radius), and reset stabilization ramps
external acceleration over its stored time. These are our interpretations of
https://magicasoft.jp/en/mc2_magicacloth_inertia/ and the public serialize-data
documentation, not the game's formulas.

Explicit scene wind uses world acceleration with the stored influence, depth,
frequency, turbulence, sine/noise blend and baseline synchronization settings.
The deterministic waveform and force arithmetic are ours (:mod:`.wind`);
Studio strength is acceleration, not the native wind zone's speed. Reset
restarts the waveform while retaining the requested scene field. Nonzero
movingWind is explicitly reported as unsupported; all local values are zero.

Not used: the spring block (the documentation gives it to another cloth type),
self collision, and the culling and level-of-detail settings.

**The reference pose** is where the chain would be with no cloth at all: each
particle under its parent as the bind pose and this frame's animation have it,
mixed by the animation-pose ratio, hung from the animated fixed points.  An
edge is restored toward its direction *in that pose*, not toward where its
parent's frame has been turned to: with the second reading a particle pushed a
few centimetres by a collider near the root carried a long chain's tip 37 cm
off while the character stood still.

One step (:data:`STEP`, at most :data:`MAX_STEPS` a frame), in the world:

1. the character moved — a particle follows that movement by ``1 − inertia``
   (the world part: the component's own node; the local part: the fixed point
   its chain hangs from, relative to that node);
2. damping, gravity, the speed limit, and the particle moves;
3. root to tip along each chain: the edge from the parent turns back toward
   its reference direction (angle restoration), is held within the angle
   limit, is given its length back (distance), and the particle is kept from
   closing on its fixed point (tether);
4. colliders push the particle's sphere out;
5. what restoration moved a particle counts toward its velocity less the
   velocity attenuation; what a collider moved it does not count at all, and
   the collider stops the part of its velocity that was heading in.

The bones are then written from the particles: a moving bone turns from its
reference rotation by how its own edge was deflected, or by how its children's
were, or between the two (``rotationalInterpolation``); a fixed one by its
children's alone, scaled by ``rootRotation``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..mesh.skeleton import Skeleton
from .wind import normalize_wind, response_acceleration

CLOTH = "BeyondBoneCloth"
SPHERE = "BeyondBoneSphereCollider"
CAPSULE = "BeyondBoneCapsuleCollider"
PLANE = "BeyondBonePlaneCollider"
MAGICA_CAPSULE = "MagicaCapsuleCollider"
MAGICA_SPHERE = "MagicaSphereCollider"
MAGICA_PLANE = "MagicaPlaneCollider"
COLLIDER_KINDS = (SPHERE, CAPSULE, PLANE, MAGICA_CAPSULE, MAGICA_SPHERE, MAGICA_PLANE)
#: the simulation's own step, and how many of them one frame may take
STEP = 1.0 / 90.0
MAX_STEPS = 3
MAX_COLLISION_PASSES = 128
#: ``selectionData`` attribute values
INVALID, FIXED, MOVE = 0, 1, 2
#: what a particle is to the solver: held at the animation, simulated, or not
#: simulated but hanging below something that is
HELD, FREE, CARRIED = 0, 1, 2
_MIRROR = np.array([-1.0, 1.0, 1.0])
_EPS = 1e-9
_FORWARD = np.array([0.0, 0.0, 1.0])


def _vec(d: dict) -> np.ndarray:
    """A serialized Unity vector in this project's handedness."""
    return np.array([float(d["x"]), float(d["y"]), float(d["z"])]) * _MIRROR


def _hermite(keys: list, t: float) -> float:
    """A Unity ``AnimationCurve`` at *t*: constant outside its keys."""
    if not keys:
        return 1.0
    if t <= keys[0]["time"]:
        return float(keys[0]["value"])
    for a, b in zip(keys, keys[1:]):
        if t <= b["time"]:
            span = b["time"] - a["time"]
            u = (t - a["time"]) / span
            m0, m1 = a["outSlope"] * span, b["inSlope"] * span
            u2, u3 = u * u, u * u * u
            return float((2 * u3 - 3 * u2 + 1) * a["value"] + (u3 - 2 * u2 + u) * m0
                         + (-2 * u3 + 3 * u2) * b["value"] + (u3 - u2) * m1)
    return float(keys[-1]["value"])


def curve_value(spec: dict, depth: float) -> float:
    """``value``, times the curve at *depth* when the curve is in use."""
    value = float(spec["value"])
    if spec.get("useCurve"):
        value *= _hermite((spec.get("curve") or {}).get("m_Curve") or [], depth)
    return value


def _size(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("...i,...i->...", v, v))


def _unit(v: np.ndarray) -> np.ndarray:
    size = _size(v)[..., None]
    return v / np.where(size > _EPS, size, 1.0)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def _axis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The unit axis that turns unit *a* toward unit *b*; any normal of *a*
    where the two are parallel."""
    axis = _cross(a, b)
    size = _size(axis)
    flat = size <= _EPS
    axis = axis / np.where(flat, 1.0, size)[:, None]
    if flat.any():
        other = np.where(np.abs(a[flat, :1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        axis[flat] = _unit(_cross(a[flat], other))
    return axis


def turn_toward(a: np.ndarray, b: np.ndarray, share) -> np.ndarray:
    """Unit vectors *a* turned by *share* of their angle to *b* (unit)."""
    angle = np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0))
    axis = _axis(a, b)
    phi = (angle * share)[:, None]
    return a * np.cos(phi) + _cross(axis, a) * np.sin(phi)


def turn_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The 3×3 rotations that take unit *a* onto unit *b* the short way."""
    axis = _axis(a, b)
    phi = np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0))
    k = np.zeros((len(a), 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -axis[:, 2], axis[:, 1]
    k[:, 1, 0], k[:, 1, 2] = axis[:, 2], -axis[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s, c = np.sin(phi)[:, None, None], np.cos(phi)[:, None, None]
    return np.eye(3)[None] + s * k + (1.0 - c) * (k @ k)


def _rotation(matrix: np.ndarray) -> np.ndarray:
    """The rotations of 4×4 (or 3×3) matrices: their columns at unit length."""
    r = matrix[..., :3, :3]
    return r / np.sqrt(np.einsum("...ij,...ij->...j", r, r))[..., None, :]


def _quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, 3, 3) as quaternions (n, 4), ``w`` last."""
    out = np.empty((len(r), 4))
    out[:, 3] = np.sqrt(np.maximum(0.0, 1.0 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2])) / 2.0
    out[:, 0] = np.sqrt(np.maximum(0.0, 1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2])) / 2.0
    out[:, 1] = np.sqrt(np.maximum(0.0, 1.0 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2])) / 2.0
    out[:, 2] = np.sqrt(np.maximum(0.0, 1.0 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2])) / 2.0
    out[:, 0] = np.copysign(out[:, 0], r[:, 2, 1] - r[:, 1, 2])
    out[:, 1] = np.copysign(out[:, 1], r[:, 0, 2] - r[:, 2, 0])
    out[:, 2] = np.copysign(out[:, 2], r[:, 1, 0] - r[:, 0, 1])
    return out


def _matrix(q: np.ndarray) -> np.ndarray:
    q = q / _size(q)[:, None]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty((len(q), 3, 3))
    out[:, 0, 0], out[:, 0, 1], out[:, 0, 2] = 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)
    out[:, 1, 0], out[:, 1, 1], out[:, 1, 2] = 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)
    out[:, 2, 0], out[:, 2, 1], out[:, 2, 2] = 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
    return out


def blend_rotations(a: np.ndarray, b: np.ndarray, share: np.ndarray) -> np.ndarray:
    """Rotations *a* toward *b* by *share* (0‥1 each), the short way round."""
    qa, qb = _quat(a), _quat(b)
    qb = np.where(np.einsum("ij,ij->i", qa, qb)[:, None] < 0.0, -qb, qb)
    return _matrix(qa + (qb - qa) * share[:, None])


@dataclass
class Collider:
    """One sphere, capsule or plane, in its bone's own space."""
    kind: str
    bone: int
    start: np.ndarray          # sphere centre / capsule end / a point of the plane
    end: np.ndarray            # the capsule's other end; the plane's normal
    radius0: float = 0.0
    radius1: float = 0.0
    source_id: str | None = None


@dataclass
class _ContactShape:
    """One posed collider and its free particles, reused for solver passes."""
    particles: np.ndarray
    which: int
    first: np.ndarray
    axis: np.ndarray
    radius0: float
    radius_delta: float
    plane: bool
    friction: float


def _contact(point, particle_radius, shape: _ContactShape):
    """Signed separation and outward direction at the unchanged points."""
    if shape.plane:
        normal = np.broadcast_to(shape.axis, point.shape)
        return (point - shape.first) @ shape.axis - particle_radius, normal
    span = float(shape.axis @ shape.axis)
    along = (np.zeros(len(point)) if span < _EPS else
             np.clip((point - shape.first) @ shape.axis / span, 0.0, 1.0))
    apart = point - (shape.first + along[:, None] * shape.axis)
    gap = _size(apart)
    depth = gap - shape.radius0 - shape.radius_delta * along - particle_radius
    normal = np.where((gap > _EPS)[:, None],
                      apart / np.where(gap > _EPS, gap, 1.0)[:, None], [0.0, 1.0, 0.0])
    return depth, normal


def _sphere_intersection(point, first, radius0, second, radius1, *, samples=1):
    """Nearest point on two sphere surfaces' circle, or no intersection.

    Exterior sphere constraints are nonconvex. Their radial projections can
    form a two-point cycle even when the exteriors share this circle. The
    exact circle supplies a deterministic escape without increasing budgets.
    """
    axis = second - first
    distance = float(_size(axis))
    if distance <= _EPS or distance > radius0 + radius1 or distance < abs(radius0 - radius1):
        return None
    axis /= distance
    along = (radius0 * radius0 - radius1 * radius1 + distance * distance) / (2 * distance)
    centre = first + axis * along
    reach = math.sqrt(max(0.0, radius0 * radius0 - along * along))
    tangent = point - centre
    tangent -= axis * float(tangent @ axis)
    if float(_size(tangent)) <= _EPS:
        basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
        tangent = basis - axis * float(basis @ axis)
    tangent = _unit(tangent)
    if samples == 1:
        return centre + tangent * reach
    # Sample a bounded set of arcs as well as the nearest pair point: that
    # point can lie inside another collider even though other arcs are clear.
    angles = np.arange(samples) * (2 * math.pi / samples)
    perpendicular = _cross(axis, tangent)
    return centre + reach * (np.cos(angles)[:, None] * tangent +
                             np.sin(angles)[:, None] * perpendicular)


def collider(kind: str, record: dict, skeleton: Skeleton) -> Collider | None:
    """The stored collider *record*; None when it is off or its bone is gone.

    A capsule's length is the whole capsule, caps included; it is centred on
    its ``center`` (``alignedOnCenter``) or starts there.  A plane keeps what
    it collides with on the side its bone's +Y points to.
    """
    if not record.get("enabled", True) or not skeleton.has(record["node"]):
        return None
    bone = skeleton.index[record["node"]]
    centre = _vec(record["center"])
    if kind == MAGICA_CAPSULE and "axis" in record:
        axis = np.zeros(3)
        axis[int(record["axis"])] = 1.0
        axis *= _MIRROR
        half = max(0.0, float(record["length"])) / 2.0
        return Collider(kind, bone, centre - axis * half, centre + axis * half,
                        float(record["startRadius"]), float(record["endRadius"]))
    if kind == MAGICA_SPHERE and "radius" in record:
        radius = float(record["radius"])
        return Collider(kind, bone, centre, centre, radius, radius)
    size = record.get("size") or {"x": 0.0, "y": 0.0, "z": 0.0}
    if kind in (SPHERE, MAGICA_SPHERE):
        return Collider(kind, bone, centre, centre, float(size["x"]), float(size["x"]))
    if kind in (PLANE, MAGICA_PLANE):
        return Collider(PLANE, bone, centre, np.array([0.0, 1.0, 0.0]))
    axis = np.zeros(3)
    axis[int(record.get("direction", 0))] = -1.0 if record.get("reverseDirection") else 1.0
    axis = axis * _MIRROR
    first = float(size["x"])
    second = float(size["y"]) if record.get("radiusSeparation") else first
    reach = max(float(size["z"]) - first - second, 0.0)
    start = centre - axis * (reach / 2.0) if record.get("alignedOnCenter", 1) else centre + axis * first
    return Collider(kind, bone, start, start + axis * reach, first, second)


@dataclass
class Report:
    """What :class:`Cloth` made of an item's components."""
    components: int = 0
    particles: int = 0
    moving: int = 0
    #: bones whose held / moving / not-simulated state came from a painted point
    painted: int = 0
    #: Serialized null slots: no collider assigned, not missing converted data.
    empty_collider_slots: int = 0
    #: Moving bones claimed by two components; fixed roots may be shared.
    overlaps: list = field(default_factory=list)
    #: ``component node: reason`` for what was left out
    skipped: list = field(default_factory=list)


def _limit(spec: dict) -> float:
    return float(spec["value"]) if spec.get("use") else math.inf


def _within(speed, limit: float):
    """The share of a movement at *speed* that counts as inertia under *limit*."""
    speed = np.asarray(speed, dtype=np.float64)
    return np.where(speed <= limit, 1.0, limit / np.where(speed > _EPS, speed, 1.0))


#: per-particle parameters, in the order :meth:`Cloth._parameters` returns them
_COLUMNS = ("damping", "radius", "restore", "limit", "limit_stiffness", "distance", "attenuation",
            "restore_falloff", "tether", "speed", "local_inertia", "interpolation", "ratio",
            "max_distance", "backstop_radius", "backstop_distance", "motion_stiffness")


class Cloth:
    """Every bone-cloth component of one item, stepped together.

    *components* is the item's ``extra["components"]``.  The solver keeps its
    particles in the world, so :meth:`step` takes the frame's bone matrices
    (model space) and the item's placement, and returns the matrices with the
    cloth bones moved (every bone below a root bone is one).  :attr:`report`
    says what was built and what was left out.
    """

    def __init__(self, skeleton: Skeleton, components: dict):
        self.skeleton = skeleton
        self.report = Report()
        self.wind = np.zeros(3)  # explicit scene world acceleration, m/s²
        bind = np.stack([b.bind_world for b in skeleton.bones])
        self._bind = bind
        self._parents = np.array([b.parent for b in skeleton.bones], dtype=np.int64)
        #: another body's capsules ``(start, end, radius)`` in world space for this frame (:meth:`set_others`)
        self._others: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.colliders, by_id = [], {}
        for kind in COLLIDER_KINDS:
            for record in components.get(kind, ()):
                made = collider(kind, record, skeleton)
                if made is not None:
                    made.source_id = str(record['id'])
                    by_id[str(record["id"])] = len(self.colliders)
                    self.colliders.append(made)

        bone, parent, state, comp, depth = [], [], [], [], []
        columns = {name: [] for name in _COLUMNS}
        self._components = []
        taken: set[int] = set()
        for record in components.get(CLOTH, ()):
            node = record.get("node")
            data = record.get("serializeData") or {}
            mismatch = [f"{key}={data.get(key)!r}" for key, wanted in
                        (("clothType", 1), ("connectionMode", 0)) if data.get(key) != wanted]
            if mismatch:
                self.report.skipped.append(f"{node}: unsupported " + ", ".join(mismatch))
                continue
            motion = data.get("motionConstraint") or {}
            if motion.get("useBackstop") and (data.get("normalAxis", 1) != 1 or
                    (data.get("normalAlignmentSetting") or {}).get("alignmentMode", 0) != 0):
                self.report.skipped.append(f"{node}: unsupported backstop normalAxis or normal alignment")
                continue
            if not record.get("enabled", True) or not skeleton.has(node):
                self.report.skipped.append(f"{node}: disabled or its node is not a bone")
                continue
            made = self._chains(record, taken)
            if made is None:
                continue
            bones, above, states, depths = made
            first = len(bone)
            bone += bones
            parent += [-1 if p < 0 else first + p for p in above]
            state += states
            depth += depths
            comp += [len(self._components)] * len(bones)
            data = record["serializeData"]
            for d in depths:
                for name, value in zip(_COLUMNS, self._parameters(data, d)):
                    columns[name].append(value)
            self._components.append(self._component(
                record, np.arange(first, first + len(bones)), by_id))
            self.report.components += 1

        self.bone = np.array(bone, dtype=np.int64)
        self.parent = np.array(parent, dtype=np.int64)
        self.state = np.array(state, dtype=np.int64)
        self.comp = np.array(comp, dtype=np.int64)
        self.depth = np.array(depth, dtype=np.float64)
        # Each component needs its own held and carried input particles.
        # Only one component writes a moving bone. A carried copy must never
        # overwrite another component's moving result.
        owner = {}
        priority = {CARRIED: 0, HELD: 1, FREE: 2}
        for particle, (b, s) in enumerate(zip(bone, state)):
            previous = owner.get(b)
            if previous is None or priority[s] > priority[state[previous]]:
                owner[b] = particle
        self._writers = np.array(list(owner.values()), dtype=np.int64)
        for name, values in columns.items():
            setattr(self, name, np.array(values, dtype=np.float64))
        count = len(self.bone)
        self.report.particles = count
        self.report.moving = int((self.state == FREE).sum())
        # the fixed point each particle hangs from, and the levels of the tree
        self.origin = np.arange(count)
        hops = np.zeros(count, dtype=np.int64)
        for i in range(count):
            p = self.parent[i]
            if p >= 0:
                hops[i] = hops[p] + 1
                if self.state[i] != HELD:
                    self.origin[i] = self.origin[p]
        attached = []
        by_bone = {}
        for i, b in enumerate(bone):
            by_bone.setdefault(b, []).append(i)
        for i, b in enumerate(bone):
            if self.state[i] != HELD:
                continue
            sources = [other for other in by_bone[b] if other != i and self.comp[other] != self.comp[i]
                       and (self.state[other] == FREE or
                            (self.state[other] == CARRIED and self.parent[other] >= 0))]
            if sources:
                source = max(sources, key=lambda other: self.state[other] == FREE)
                attached.append((i, source))
        self._attached = np.array(attached, dtype=np.int64).reshape(-1, 2)
        # Nested components must run after their bone's real parent, even
        # when that parent belongs to another component's particle tree.
        bone_hops = np.zeros(len(skeleton.bones), dtype=np.int64)
        for b, p in enumerate(self._parents):
            if p >= 0:
                bone_hops[b] = bone_hops[p] + 1
        hops = bone_hops[self.bone]
        below = self.state != HELD
        self._levels = [level for level in (
            np.nonzero(below & (hops == h))[0] for h in range(1, int(hops.max()) + 1 if count else 1))
            if len(level)]
        self._child = np.nonzero(self.parent >= 0)[0]
        local = self._local(bind)
        self._rest_offset = local[:, :3, 3].copy()
        self._rest_turn = _rotation(local) if count else np.zeros((0, 3, 3))
        # how far each particle rests from its fixed point: the tether's measure
        self._span = _size(bind[self.bone][:, :3, 3] - bind[self.bone[self.origin]][:, :3, 3]) \
            if count else np.zeros(0)
        self._down = (np.stack([c["down"] for c in self._components])[self.comp]
                      if count else np.zeros((0, 3)))
        self._weight = np.array([c["weight"] for c in self._components])[self.comp] if count else np.zeros(0)
        has_child = np.zeros(count, dtype=bool)
        has_child[self.parent[self._child]] = True
        # how far toward its children's deflection each bone is written
        self._bend = np.where(self.state == FREE, self.interpolation, 0.0)
        held = self.state == HELD
        self._bend[held] = np.array([c["root_rotation"] for c in self._components])[self.comp[held]] \
            if held.any() else 0.0
        self._bend[~has_child] = 0.0
        self.position = np.zeros((count, 3))
        self.velocity = np.zeros((count, 3))
        self._centre: list = []
        self._origin_frame = np.zeros((count, 4, 4))
        self._debt = 0.0
        self._live = False
        self._anchor_frames = []
        self._world_velocity = np.zeros((len(self._components), 3))
        self._angular_velocity = np.zeros((count, 3))
        self._age = np.zeros(len(self._components))

    # -- setup -------------------------------------------------------------
    def set_others(self, capsules) -> None:
        """Another body the chains meet this frame: ``(start, end, radius)`` capsules in world space ([] for none).

        A chain that collides at all (the game lists colliders for it) is
        tested against them too; its own game data names only its wearer's.
        """
        self._others = [(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), float(r))
                        for a, b, r in capsules]

    def add_colliders(self, colliders: list, under: set) -> int:
        """Give the chains that hang from *under* (bone indices) further colliders; how many chains took them.

        The game lists, per chain, the colliders it is tested against, and it
        lists what that chain can reach on the character's own body in the
        game's own poses: EndminF's hair has her thighs, pelvis and spine and
        nothing for her shoulders or arms.  A body lent from elsewhere and
        poses from elsewhere put more in a chain's way
        (``endfield/body.body_colliders``).  Only chains every particle of
        which is under one of *under* take them: hair and what is worn on the
        head, not a coat that is meant to lie on the shoulders.
        """
        first = len(self.colliders)
        self.colliders.extend(colliders)
        added = list(range(first, len(self.colliders)))
        taken = 0
        for comp in self._components:
            mine = self.bone[comp["particles"]]
            if len(mine) and all(int(bone) in under for bone in mine):
                comp["colliders"] = list(comp["colliders"]) + added
                taken += 1
        return taken

    def _local(self, world: np.ndarray) -> np.ndarray:
        """Each particle's bone in its parent bone's frame, for matrices *world*."""
        if not len(self.bone):
            return np.zeros((0, 4, 4))
        above = self._parents[self.bone]
        frames = np.where((above >= 0)[:, None, None], world[np.maximum(above, 0)], np.eye(4))
        return np.linalg.inv(frames) @ world[self.bone]

    @staticmethod
    def _parameters(data: dict, depth: float) -> tuple:
        """One particle's parameters at *depth*, in :data:`_COLUMNS` order."""
        inertia = data["inertiaConstraint"]
        restore = data["angleRestorationConstraint"]
        limit = data["angleLimitConstraint"]
        speed = inertia["particleSpeedLimit"]
        motion = data.get("motionConstraint") or {}
        return (
            curve_value(data["damping"], depth),
            curve_value(data["radius"], depth),
            curve_value(restore["stiffness"], depth) if restore.get("useAngleRestoration") else 0.0,
            math.radians(curve_value(limit["limitAngle"], depth)) if limit.get("useAngleLimit") else math.inf,
            float(limit["stiffness"]),
            curve_value(data["distanceConstraint"]["stiffness"], depth),
            float(restore["velocityAttenuation"]),
            float(restore["gravityFalloff"]),
            float(data["tetherConstraint"]["distanceCompression"]),
            float(speed["value"]) if speed.get("use") else math.inf,
            # the depth inertia takes local inertia away near the fixed point
            float(inertia["localInertia"]) * (1.0 - float(inertia["depthInertia"]) * (1.0 - depth)),
            float(data["rotationalInterpolation"]),
            float(data["animationPoseRatio"]),
            max(0.0, curve_value(motion["maxDistance"], depth)) if motion.get("useMaxDistance") else math.inf,
            max(0.0, float(motion["backstopRadius"])) if motion.get("useBackstop") else 0.0,
            curve_value(motion["backstopDistance"], depth) if motion.get("useBackstop") else 0.0,
            float(np.clip(motion.get("stiffness", 1.0), 0.0, 1.0)))

    def _component(self, record: dict, particles: np.ndarray, by_id: dict) -> dict:
        data = record["serializeData"]
        inertia = data["inertiaConstraint"]
        collision = data["colliderCollisionConstraint"]
        slots = collision.get("colliderList") or ()
        self.report.empty_collider_slots += sum(c is None for c in slots)
        named = [c for c in slots if c is not None]
        listed = [by_id[str(c.get("component"))] for c in named
                  if isinstance(c, dict) and str(c.get("component")) in by_id]
        if len(listed) != len(named):
            self.report.skipped.append(
                f"{record['node']}: {len(named) - len(listed)} colliders are not converted")
        anchor_name = (inertia.get("anchor") or {}).get("node")
        if anchor_name and not self.skeleton.has(anchor_name):
            self.report.skipped.append(f"{record['node']}: anchor {anchor_name} is not in the skeleton")
        wind = dict(data.get("wind") or {})
        if wind.get("movingWind", 0) != 0:
            self.report.skipped.append(f"{record['node']}: nonzero movingWind is unsupported")
        return {
            "id": str(record["id"]),
            "wind": wind,
            "node": self.skeleton.index[record["node"]],
            "particles": particles,
            "gravity": _vec(data["gravityDirection"]) * float(data["gravity"]),
            # which way is down, whether or not anything falls
            "down": _unit(_vec(data["gravityDirection"])),
            "gravity_falloff": float(data["gravityFalloff"]),
            "world_inertia": float(inertia["worldInertia"]),
            "smoothing": float(np.clip(inertia.get("movementInertiaSmoothing", 0.0), 0.0, 1.0)),
            "centrifugal": float(inertia.get("centrifualAcceleration", 0.0)),
            "stabilization": max(0.0, float(data.get("stablizationTimeAfterReset", 0.0))),
            # Beyond's game-specific simulation weight is interpreted as an
            # output blend, alongside the public package's blendWeight.
            "weight": float(np.clip(data.get("blendWeight", 1.0) * data.get("clothSimulateWeight", 1.0), 0.0, 1.0)),
            "anchor": self.skeleton.index[anchor_name] if self.skeleton.has(anchor_name) else None,
            "anchor_inertia": float(np.clip(inertia.get("anchorInertia", 0.0), 0.0, 1.0)),
            "move_limit": _limit(inertia["movementSpeedLimit"]),
            "turn_limit": math.radians(_limit(inertia["rotationSpeedLimit"])),
            "local_move_limit": _limit(inertia["localMovementSpeedLimit"]),
            "local_turn_limit": math.radians(_limit(inertia["localRotationSpeedLimit"])),
            "teleport": int(inertia["teleportMode"]),
            "teleport_distance": float(inertia["teleportDistance"]),
            "teleport_turn": math.radians(float(inertia["teleportRotation"])),
            "root_rotation": float(data["rootRotation"]),
            "colliders": listed if int(collision.get("mode", 0)) else [],
            # whether the chain collides at all: what it meets of another body is not in its list
            "collides": bool(int(collision.get("mode", 0))),
            "friction": min(1.0, float(collision.get("friction", 0.0))),
        }

    def _chains(self, record: dict, taken: set):
        """``(bones, parent within the list, state, depth)`` for one component."""
        skeleton, bind = self.skeleton, self._bind
        data = record["serializeData"]
        node = record["node"]
        ignored: set[int] = set()
        for entry in data.get("ignoreFromRootBones") or ():
            name = (entry or {}).get("node")
            if name and skeleton.has(name):
                ignored.update(skeleton.descendants(name))
        bones: list[int] = []
        for entry in data.get("rootBones") or ():
            name = (entry or {}).get("node")
            if not name or not skeleton.has(name):
                self.report.skipped.append(f"{node}: root bone {name} is not in the skeleton")
                continue
            bones += [b for b in sorted(skeleton.descendants(name))
                      if b not in bones]
        if not bones:
            self.report.skipped.append(f"{node}: no chain")
            return None
        bones.sort()                         # parents first: a parent has the lower index
        where = {b: i for i, b in enumerate(bones)}
        above = [where.get(int(self._parents[b]), -1) for b in bones]
        painted = self._painted(record.get("selectionData") or {}, bind[bones][:, :3, 3])
        states = []
        for i, b in enumerate(bones):
            attribute = painted[i]
            if attribute is None:
                attribute = FIXED if above[i] < 0 else MOVE
            else:
                self.report.painted += 1
            if b in ignored:
                attribute = INVALID
            if above[i] < 0 or attribute == FIXED:
                states.append(HELD)          # the top of a chain has nothing to hang from
            else:
                states.append(FREE if attribute == MOVE else CARRIED)
        wanted = [i for i, value in enumerate(states) if value == FREE]
        overlap = [i for i in wanted if bones[i] in taken]
        if overlap:
            self.report.overlaps.append({"id": str(record["id"]), "node": node,
                                         "bones": [skeleton.bones[bones[i]].name for i in overlap]})
            if len(overlap) == len(wanted):
                self.report.skipped.append(f"{node}: no chain (all moving bones overlap an earlier component)")
                return None
            for i in overlap:
                states[i] = CARRIED
        taken.update(b for b, value in zip(bones, states) if value == FREE)
        # depth: the way along the chain from the held particle above
        along = [0.0] * len(bones)
        for i, b in enumerate(bones):
            if states[i] != HELD:
                p = above[i]
                along[i] = along[p] + float(_size(bind[b][:3, 3] - bind[bones[p]][:3, 3]))
        longest = max(along) or 1.0
        return bones, above, states, [a / longest for a in along]

    @staticmethod
    def _painted(selection: dict, positions: np.ndarray) -> list:
        """The painted attribute of each bone, None where no point is near."""
        points = selection.get("positions") or []
        values = [int(a.get("Value", MOVE)) for a in selection.get("attributes") or []]
        if not points or len(values) != len(points):
            return [None] * len(positions)
        painted = np.array([_vec(p) for p in points])
        reach = float(selection.get("maxConnectionDistance") or 0.0)
        out = []
        for position in positions:
            gap = _size(painted - position)
            nearest = int(gap.argmin())
            out.append(values[nearest] if gap[nearest] <= reach else None)
        return out

    # -- per frame ---------------------------------------------------------
    def reset(self) -> None:
        """Forget the motion: the next step starts from the pose it is given."""
        self._live = False

    def _reference(self, world: np.ndarray, placed: np.ndarray):
        """The chain with no cloth, this frame: ``(turn, aim, length)``.

        *turn* is each particle's rotation in the world, *aim* the unit
        direction from its parent to it and *length* that distance.  Under its
        parent a particle rests as the bind pose and the animated pose have
        it, mixed by the animation-pose ratio; a particle that is not
        simulated takes the animated one.
        """
        local = self._local(world)
        ratio = np.where(self.state == CARRIED, 1.0, self.ratio)
        offset = self._rest_offset + (local[:, :3, 3] - self._rest_offset) * ratio[:, None]
        animated = _rotation(local)
        own = np.where((ratio >= 1.0)[:, None, None], animated, self._rest_turn)
        mixed = (ratio > 0.0) & (ratio < 1.0)
        if mixed.any():
            own[mixed] = blend_rotations(self._rest_turn[mixed], animated[mixed], ratio[mixed])
        turn = _rotation(placed[self.bone])
        for level in self._levels:
            turn[level] = turn[self.parent[level]] @ own[level]
        scale = _size(placed[np.maximum(self._parents[self.bone], 0)][:, :3, 0])
        rest = np.zeros((len(self.bone), 3))
        child = self._child
        rest[child] = np.einsum("nij,nj->ni", turn[self.parent[child]], offset[child] * scale[child, None])
        length = _size(rest)
        return turn, rest / np.where(length > _EPS, length, 1.0)[:, None], length

    def set_wind(self, direction, strength: float) -> None:
        """Set explicit scene world acceleration; strength is m/s², not game wind speed."""
        self.wind = normalize_wind(direction, strength)

    def step(self, dt: float, world: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Advance by *dt* seconds; returns *world* with the cloth bones moved."""
        if not len(self.bone):
            return world
        transform = np.asarray(transform, dtype=np.float64)
        placed = transform @ world
        target = placed[self.bone]
        turn, aim, length = self._reference(world, placed)
        reference = target[:, :3, 3].copy()
        for level in self._levels:
            reference[level] = reference[self.parent[level]] + aim[level] * length[level, None]
        self._motion_reference = reference
        # All current prefabs use normalAxis=1 (+Y), with no alignment.
        self._motion_normal = turn[:, :, 1]
        self._motion_scale = _size(placed[self.bone][:, :3, 0])
        centres = [placed[c["node"]] for c in self._components]
        origin_frame = target[self.origin]
        if not self._live:
            self.position = target[:, :3, 3].copy()
            self.velocity[:] = 0.0
            self._debt = 0.0
            self._live = True
            self._world_velocity[:] = 0.0
            self._angular_velocity[:] = 0.0
            self._age[:] = 0.0
        else:
            self._follow(target, centres, origin_frame, max(dt, STEP), placed)
        self._debt = min(self._debt + max(dt, 0.0), STEP * MAX_STEPS)
        steps = int(self._debt / STEP + 1e-9)
        self._debt -= steps * STEP
        held = self.state == HELD
        self.position[held] = target[held][:, :3, 3]
        for _ in range(steps):
            self._one_step(aim, length, centres, placed)
        if not steps:
            self._carry(aim, length)
        self._centre = centres
        self._origin_frame = origin_frame.copy()
        self._anchor_frames = [placed[c["anchor"]].copy() if c["anchor"] is not None else np.eye(4)
                               for c in self._components]
        return self._write(world, placed, transform, turn, aim)

    def _follow(self, target, centres, origin_frame, dt: float, placed) -> None:
        """Carry the particles with the character by what inertia leaves over.

        A movement in one frame past the component's teleport distance or
        angle is not inertia at all: mode 1 starts that component again from
        the pose, mode 2 carries its particles along whole.
        """
        free = self.state == FREE
        self._angular_velocity[:] = 0.0
        for index, comp in enumerate(self._components):
            everyone = comp["particles"]
            mine = everyone[free[everyone]]
            before, now = self._centre[index], centres[index]
            moved = now[:3, 3] - before[:3, 3]
            spin = _rotation(now) @ _rotation(before).T
            angle = math.acos(float(np.clip((np.trace(spin) - 1.0) / 2.0, -1.0, 1.0)))
            distance = float(_size(moved))
            jumped = bool(comp["teleport"]) and (distance > comp["teleport_distance"]
                                                 or angle > comp["teleport_turn"])
            if jumped and comp["teleport"] == 1:
                self.position[everyone] = target[everyone][:, :3, 3]
                self.velocity[everyone] = 0.0
                self._world_velocity[index] = 0.0
                self._age[index] = 0.0
                continue
            if not len(mine):
                continue
            origin_before = self._origin_frame[mine]
            if comp["anchor"] is not None:
                anchor_delta = placed[comp["anchor"]] @ np.linalg.inv(self._anchor_frames[index])
                anchor_rotation = _rotation(anchor_delta)
                anchor_share = 1.0 - comp["anchor_inertia"]
                anchor_moved = (np.einsum("ij,nj->ni", anchor_delta[:3, :3], self.position[mine])
                                + anchor_delta[:3, 3] - self.position[mine])
                self.position[mine] += anchor_moved * anchor_share
                self.velocity[mine] = self.velocity[mine] @ (
                    np.eye(3) + anchor_share * (anchor_rotation - np.eye(3))).T
                before = anchor_delta @ before
                origin_before = anchor_delta @ origin_before
                moved = now[:3, 3] - before[:3, 3]
                spin = _rotation(now) @ _rotation(before).T
                angle = math.acos(float(np.clip((np.trace(spin) - 1.0) / 2.0, -1.0, 1.0)))
                distance = float(_size(moved))
            keep = 0.0 if jumped else comp["world_inertia"]
            keep_move = keep * float(_within(distance / dt, comp["move_limit"]))
            keep_turn = keep * float(_within(angle / dt, comp["turn_limit"]))
            arm = self.position[mine] - before[:3, 3]
            retention = comp["smoothing"] ** (dt / STEP)
            missed = keep_move * moved / dt
            self._world_velocity[index] = (retention * self._world_velocity[index] + (1.0 - retention) * missed)
            if jumped:
                self._world_velocity[index] = 0.0
            self.position[mine] += moved - self._world_velocity[index] * dt + (1.0 - keep_turn) * (arm @ spin.T - arm)
            if angle > _EPS and not jumped:
                q = _quat(spin[None])[0]
                axis = _unit(q[:3]) * (1 if q[3] >= 0 else -1)
                self._angular_velocity[mine] = axis * (angle / dt) * keep_turn
            self.velocity[mine] = self.velocity[mine] @ (
                np.eye(3) + (1.0 - keep_turn) * (spin - np.eye(3))).T
            # the local part: the fixed point's own movement inside the character
            was = (now @ np.linalg.inv(before)) @ origin_before
            is_now = origin_frame[mine]
            shift = is_now[:, :3, 3] - was[:, :3, 3]
            twist = _rotation(is_now) @ np.transpose(_rotation(was), (0, 2, 1))
            turned = np.arccos(np.clip((np.trace(twist, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
            keep = np.zeros(len(mine)) if jumped else self.local_inertia[mine]
            keep_shift = keep * _within(_size(shift) / dt, comp["local_move_limit"])
            keep_twist = keep * _within(turned / dt, comp["local_turn_limit"])
            arm = self.position[mine] - was[:, :3, 3]
            swung = np.einsum("nij,nj->ni", twist, arm) - arm
            self.position[mine] += ((1.0 - keep_shift)[:, None] * shift
                                    + (1.0 - keep_twist)[:, None] * swung)
            spinning = turned > _EPS
            if spinning.any() and not jumped:
                q = _quat(twist[spinning])
                axis = _unit(q[:, :3]) * np.where(q[:, 3:] >= 0, 1.0, -1.0)
                self._angular_velocity[mine[spinning]] += axis * (
                    turned[spinning] / dt * keep_twist[spinning])[:, None]

    def _one_step(self, aim, length, centres, placed) -> None:
        free = self.state == FREE
        start = self.position.copy()
        velocity = self.velocity * (1.0 - self.damping)[:, None]
        wind_active = np.any(self.wind)
        for index, comp in enumerate(self._components):
            self._age[index] += STEP
            ramp = min(1.0, self._age[index] / comp["stabilization"]) if comp["stabilization"] else 1.0
            pull = comp["gravity"]
            size = float(_size(pull))
            if size > _EPS:
                # the falloff: no gravity while the character stands as it was
                # bound, more of it the further it tilts
                down = pull / size
                tilt = _rotation(centres[index]) @ _rotation(self._bind[comp["node"]]).T
                pull = pull * (1.0 - comp["gravity_falloff"] * max(0.0, float(np.dot(tilt @ down, down))))
            mine = comp["particles"]
            velocity[mine] += pull * STEP * ramp
            if wind_active:
                acceleration = response_acceleration(self.wind, comp["wind"], self._age[index],
                                                     self.depth[mine], self.bone[self.origin[mine]])
                velocity[mine] += acceleration * STEP * ramp
            if comp["centrifugal"]:
                omega = self._angular_velocity[mine]
                arm = self.position[mine] - self.position[self.origin[mine]]
                acceleration = -_cross(omega, _cross(omega, arm)) * comp["centrifugal"]
                velocity[mine] += acceleration * STEP * ramp
        speed = _size(velocity)
        over = speed > self.speed
        velocity[over] *= (self.speed[over] / speed[over])[:, None]
        self.position[free] = start[free] + velocity[free] * STEP
        restored = self._chain_pass(aim, length)
        moved = (self.position - start - restored * self.attenuation[:, None]) / STEP
        slowed, normal, motion_normal = self._project_constraints(placed)
        moved -= np.minimum(0.0, np.einsum("ij,ij->i", moved, motion_normal))[:, None] * motion_normal
        # a collider gives no speed: being pushed out of one is not a movement,
        # what was heading into it is stopped, and what slides along it is
        # slowed by the friction
        into = np.minimum(0.0, np.einsum("ij,ij->i", moved, normal))[:, None] * normal
        moved -= into
        moved -= (moved - np.einsum("ij,ij->i", moved, normal)[:, None] * normal) * slowed[:, None]
        self.velocity[free] = moved[free]

    def _project_constraints(self, placed: np.ndarray):
        """Alternate hard motion and collider projections within a fixed budget.

        Fractional motion stiffness remains one relaxation per time step;
        applying it on every solver pass would silently turn it into a hard
        limit. Only stiffness one is projected again. Conflicting geometry
        can retain residuals after the bounded passes, recorded by the audit.
        """
        motion_normals = self._motion_limit()
        shapes = self._collision_shapes(placed)
        hard = ((self.state == FREE) & (self.motion_stiffness >= 1.0)
                & ((self.backstop_radius > 0) | np.isfinite(self.max_distance)))
        if not hard.any():
            slowed, normals = self._collide(placed, shapes=shapes)
            return slowed, normals, motion_normals
        slowed = np.zeros(len(self.bone))
        normals = np.zeros_like(self.position)
        for iteration in range(MAX_COLLISION_PASSES):
            before = self.position.copy()
            motion = self._motion_limit(hard_only=True)
            motion_change = np.max(_size(self.position - before), initial=0.0)
            hit = _size(motion) > _EPS
            motion_normals[hit] = motion[hit]
            friction, normal = self._collision_pass(placed, shapes=shapes)
            collision_change = np.max(self._last_collision_change, initial=0.0)
            hit = _size(normal) > _EPS
            slowed[hit] = friction[hit]
            normals[hit] = normal[hit]
            if iteration % 8 == 7:
                escaped = self._escape_backstop_contacts(shapes, hard)
                hit = _size(escaped) > _EPS
                motion_normals[hit] = escaped[hit]
                for shape in shapes:
                    mine = shape.particles[hit[shape.particles]]
                    if not len(mine):
                        continue
                    depth, contact_normal = _contact(self.position[mine], self.radius[mine], shape)
                    touching = mine[depth <= 1e-8]
                    normals[touching] = contact_normal[depth <= 1e-8]
                    slowed[touching] = shape.friction
                escaped_contacts = self._escape_collider_contacts(shapes)
                self._refresh_contact_normals(shapes, escaped_contacts, normals, slowed)
            # Check each projection separately: opposite corrections may
            # cancel over a whole pass while both constraints remain broken.
            if max(motion_change, collision_change) <= 1e-8:
                break
            active = (_size(motion) > _EPS) | (_size(normal) > _EPS)
            self._retain_active_shapes(shapes, active)
        return slowed, normals, motion_normals

    def _escape_backstop_contacts(self, shapes, hard):
        """Try exact sphere intersections for stalled hard backstop contacts.

        Capsules contribute their closest axis sphere as a candidate only.
        Acceptance checks every actual collider, the full backstop sphere and
        max distance, so taper/axis changes cannot claim false clearance.
        Soft motion limits never participate in this extra projection.
        """
        radius = self.backstop_radius * self._motion_scale
        centres = self._motion_reference - self._motion_normal * (
            radius + self.backstop_distance * self._motion_scale)[:, None]
        overlap = radius - _size(self.position - centres)
        escaped = np.zeros_like(self.position)
        for particle in np.flatnonzero(hard & (radius > 0) & (overlap > 1e-8)):
            touching = [shape for shape in shapes if particle in shape.particles]
            point = self.position[particle]
            accepted = []
            for shape in touching:
                if shape.plane:
                    continue
                span = float(shape.axis @ shape.axis)
                along = (0.0 if span < _EPS else
                         float(np.clip((point - shape.first) @ shape.axis / span, 0, 1)))
                centre = shape.first + along * shape.axis
                other_radius = shape.radius0 + along * shape.radius_delta + self.radius[particle]
                candidates = _sphere_intersection(point, centres[particle], radius[particle], centre, other_radius,
                                                  samples=64)
                if candidates is None:
                    continue
                clear = _size(candidates - self._motion_reference[particle]) <= (
                    self.max_distance[particle] * self._motion_scale[particle] + 1e-8)
                for other in touching:
                    depth, _ = _contact(candidates, self.radius[particle], other)
                    clear &= depth >= -1e-8
                if clear.any():
                    choices = candidates[clear]
                    accepted.append(choices[int(np.argmin(_size(choices - point)))])
            if accepted:
                self.position[particle] = min(accepted, key=lambda candidate: float(_size(candidate - point)))
                escaped[particle] = _unit(self.position[particle] - centres[particle])
        return escaped

    def _escape_collider_contacts(self, shapes):
        """Bounded pair-circle search for feasible collider-only radial cycles.

        Use at most the four closest non-plane contacts per particle, then
        validate all actual shapes and every hard motion limit. This is our
        geometric projection, not the unavailable game's collision formula.
        """
        overlap = np.zeros(len(self.bone))
        for shape in shapes:
            mine = shape.particles
            if len(mine):
                depth, _ = _contact(self.position[mine], self.radius[mine], shape)
                overlap[mine] = np.maximum(overlap[mine], -depth)
        escaped = np.zeros(len(self.bone), dtype=bool)
        for particle in np.flatnonzero(overlap > 1e-8):
            point = self.position[particle]
            touching = [shape for shape in shapes if particle in shape.particles]
            near = []
            for shape in touching:
                depth, _ = _contact(point[None], self.radius[particle], shape)
                if not shape.plane and depth[0] <= 1e-7:
                    span = float(shape.axis @ shape.axis)
                    along = 0 if span < _EPS else float(np.clip((point - shape.first) @ shape.axis / span, 0, 1))
                    near.append((float(depth[0]), shape.first + along * shape.axis,
                                 shape.radius0 + along * shape.radius_delta + self.radius[particle]))
            near.sort(key=lambda entry: entry[0])
            accepted = []
            for first, second in combinations(near[:4], 2):
                candidates = _sphere_intersection(point, first[1], first[2], second[1], second[2], samples=64)
                if candidates is None:
                    continue
                clear = np.ones(len(candidates), dtype=bool)
                for shape in touching:
                    depth, _ = _contact(candidates, self.radius[particle], shape)
                    clear &= depth >= -1e-8
                if self.motion_stiffness[particle] >= 1:
                    scale = self._motion_scale[particle]
                    clear &= _size(candidates - self._motion_reference[particle]) <= self.max_distance[particle] * scale + 1e-8
                    radius = self.backstop_radius[particle] * scale
                    centre = self._motion_reference[particle] - self._motion_normal[particle] * (
                        radius + self.backstop_distance[particle] * scale)
                    clear &= _size(candidates - centre) >= radius - 1e-8
                if clear.any():
                    choices = candidates[clear]
                    accepted.append(choices[int(np.argmin(_size(choices - point)))])
            if accepted:
                self.position[particle] = min(accepted, key=lambda candidate: float(_size(candidate - point)))
                escaped[particle] = True
        return escaped

    def _refresh_contact_normals(self, shapes, escaped, normals, slowed):
        normals[escaped] = 0
        slowed[escaped] = 0
        for shape in shapes:
            mine = shape.particles[escaped[shape.particles]]
            if len(mine):
                depth, normal = _contact(self.position[mine], self.radius[mine], shape)
                hit = mine[depth <= 1e-8]
                normals[hit] = normal[depth <= 1e-8]
                slowed[hit] = shape.friction

    @staticmethod
    def _retain_active_shapes(shapes, active):
        """Independent settled particles cannot reenter constraints this pass."""
        for shape in shapes:
            shape.particles = shape.particles[active[shape.particles]]

    def _motion_limit(self, *, hard_only: bool = False) -> np.ndarray:
        """Project free particles inside max distance and outside backstop.

        Backstop distance measures the gap from the reference point to the
        sphere's nearest surface, so the centre is radius + distance behind
        the normal. A zero gap leaves the reference pose on the sphere.
        These collision corrections add no speed.
        """
        normals = np.zeros_like(self.position)
        free = self.state == FREE
        if hard_only:
            free &= self.motion_stiffness >= 1.0
        scale = self._motion_scale
        apart = self.position - self._motion_reference
        gap = _size(apart)
        limit = self.max_distance * scale
        hit = free & (gap > limit)
        if hit.any():
            outward = apart[hit] / gap[hit, None]
            self.position[hit] -= outward * ((gap[hit] - limit[hit]) * self.motion_stiffness[hit])[:, None]
            normals[hit] = -outward
        radius = self.backstop_radius * scale
        centre = self._motion_reference - self._motion_normal * (
            radius + self.backstop_distance * scale)[:, None]
        apart = self.position - centre
        gap = _size(apart)
        hit = free & (radius > 0.0) & (gap < radius)
        if hit.any():
            outward = np.where((gap[hit] > _EPS)[:, None],
                               apart[hit] / np.maximum(gap[hit, None], _EPS), self._motion_normal[hit])
            self.position[hit] += outward * ((radius[hit] - gap[hit]) * self.motion_stiffness[hit])[:, None]
            normals[hit] = outward
        normals[self.motion_stiffness <= 0.0] = 0.0
        return normals

    def _carry(self, aim, length) -> None:
        """Put the particles that are not simulated under their parents."""
        for level in self._levels:
            rigid = level[(self.state[level] == CARRIED) | (length[level] <= _EPS)]
            self.position[rigid] = self.position[self.parent[rigid]] + aim[rigid] * length[rigid, None]
            self._sync_attached()

    def _sync_attached(self) -> None:
        """Shared fixed input follows the component carrying its real parent."""
        if len(self._attached):
            self.position[self._attached[:, 0]] = self.position[self._attached[:, 1]]

    def _chain_pass(self, aim, length) -> np.ndarray:
        """Angle restoration, angle limit, distance and tether, root to tip.

        Returns what the restoration alone moved each particle by — the turn
        of its own edge.  The edge runs from the parent *as this pass has
        already put it*: a parent that was pulled drags its child after it,
        the way a rope does, and that drag is a movement like any other.
        """
        restored = np.zeros_like(self.position)
        for level in self._levels:
            root = self.position[self.parent[level]]
            want, far = aim[level], length[level]
            # not simulated, or sitting on its parent: it goes where the pose puts it
            rigid = (self.state[level] == CARRIED) | (far <= _EPS)
            edge = self.position[level] - root
            reach = _size(edge)
            way = np.where((reach > _EPS)[:, None], edge / np.where(reach > _EPS, reach, 1.0)[:, None], want)
            against = np.maximum(0.0, -np.einsum("ij,ij->i", want, self._down[level]))
            share = self.restore[level] * (1.0 - self.restore_falloff[level] * against)
            turned = turn_toward(way, want, np.clip(share, 0.0, 1.0))
            angle = np.arccos(np.clip(np.einsum("ij,ij->i", turned, want), -1.0, 1.0))
            excess = np.maximum(0.0, angle - self.limit[level])
            back = np.where(angle > _EPS, excess * self.limit_stiffness[level]
                            / np.where(angle > _EPS, angle, 1.0), 0.0)
            held = turn_toward(turned, want, back)
            new = root + held * (reach + (far - reach) * self.distance[level])[:, None]
            # the tether: no nearer its fixed point than the compression allows
            anchor = self.position[self.origin[level]]
            apart = new - anchor
            gap = _size(apart)
            least = (1.0 - self.tether[level]) * self._span[level]
            short = (gap < least) & (gap > _EPS)
            new[short] = anchor[short] + apart[short] * (least[short] / gap[short])[:, None]
            new[rigid] = root[rigid] + want[rigid] * far[rigid, None]
            restored[level] = np.where(rigid[:, None], 0.0, (turned - way) * reach[:, None])
            self.position[level] = new
            self._sync_attached()
        return restored

    def _collide(self, placed: np.ndarray, *, shapes=None):
        """Push the moving particles out of their component's colliders.

        Returns ``(friction, normal)`` per particle: the friction of the
        collider it touched (0 where it touched none) and the way it was pushed.
        """
        # A later collider can push back into an earlier one. Repeat only
        # while positions change; the fixed budget bounds conflicting data.
        slowed = np.zeros(len(self.bone))
        normals = np.zeros((len(self.bone), 3))
        if shapes is None:
            shapes = self._collision_shapes(placed)
        for iteration in range(MAX_COLLISION_PASSES):
            friction, normal = self._collision_pass(placed, shapes=shapes)
            hit = _size(normal) > _EPS
            slowed[hit] = friction[hit]
            normals[hit] = normal[hit]
            if iteration % 8 == 7:
                escaped = self._escape_collider_contacts(shapes)
                self._refresh_contact_normals(shapes, escaped, normals, slowed)
                hit |= escaped
            if np.max(self._last_collision_change, initial=0.0) <= 1e-8:
                break
            self._retain_active_shapes(shapes, hit)
        return slowed, normals

    def _collision_shapes(self, placed: np.ndarray):
        """Pose collider geometry once; it cannot change during projections."""
        free = self.state == FREE
        shapes = []
        for comp in self._components:
            mine = comp["particles"][free[comp["particles"]]]
            if not len(mine) or not (comp["colliders"] or (self._others and comp.get("collides"))):
                continue
            for which in comp["colliders"]:
                shape = self.colliders[which]
                matrix = placed[shape.bone]
                first = matrix[:3, :3] @ shape.start + matrix[:3, 3]
                plane = shape.kind == PLANE
                axis = (_unit(matrix[:3, :3] @ shape.end) if plane else
                        matrix[:3, :3] @ shape.end + matrix[:3, 3] - first)
                scale = float(_size(matrix[:3, 0]))
                shapes.append(_ContactShape(mine, which, first, axis,
                                            shape.radius0 * scale,
                                            (shape.radius1 - shape.radius0) * scale,
                                            plane, comp['friction']))
            for start, end, radius in self._others:
                shapes.append(_ContactShape(mine, -1, start, end - start, radius, 0.0, False, comp['friction']))
        return shapes

    def collision_residual(self, placed: np.ndarray, *, details=False):
        """Largest direct penetration per particle; never mutates solver state."""
        residual = np.zeros(len(self.bone))
        which = np.full(len(self.bone), -1, dtype=np.int64)
        for shape in self._collision_shapes(placed):
            mine = shape.particles
            depth, _ = _contact(self.position[mine], self.radius[mine], shape)
            greater = -depth > residual[mine]
            which[mine[greater]] = shape.which
            residual[mine] = np.maximum(residual[mine], np.maximum(0.0, -depth))
        return (residual, which) if details else residual

    def _collision_pass(self, placed: np.ndarray, *, shapes=None):
        slowed = np.zeros(len(self.bone))
        normals = np.zeros((len(self.bone), 3))
        self._last_collision_change = np.zeros(len(self.bone))
        if shapes is None:
            shapes = self._collision_shapes(placed)
        for shape in shapes:
            mine = shape.particles
            if not len(mine):
                continue
            depth, normal = _contact(self.position[mine], self.radius[mine], shape)
            inside = depth < -1e-8
            hit = mine[inside]
            if len(hit):
                self.position[hit] -= depth[inside][:, None] * normal[inside]
                self._last_collision_change[hit] = np.maximum(self._last_collision_change[hit], -depth[inside])
                slowed[hit] = shape.friction
                normals[hit] = normal[inside]
        return slowed, normals

    def _write(self, world, placed, transform, turn, aim) -> np.ndarray:
        """The frame's matrices with the cloth bones set from the particles."""
        count = len(self.bone)
        child = self._child
        above = self.parent[child]
        edge = np.zeros((count, 3))
        edge[child] = _unit(self.position[child] - self.position[above])
        # how a bone's own edge was deflected, and how its children's were
        moved = (self.state != HELD) & (_size(aim) > _EPS) & (_size(edge) > _EPS)
        own = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
        own[moved] = turn_between(aim[moved], edge[moved])
        want = np.zeros((count, 3))
        have = np.zeros((count, 3))
        np.add.at(want, above, aim[child])
        np.add.at(have, above, edge[child])
        lone = (_size(want) <= _EPS) | (_size(have) <= _EPS)
        below = turn_between(_unit(np.where(lone[:, None], _FORWARD, want)),
                             _unit(np.where(lone[:, None], _FORWARD, have)))
        bent = blend_rotations(own, below, self._bend) @ turn
        animated_turn = _rotation(placed[self.bone])
        bent = blend_rotations(animated_turn, bent, self._weight)
        out = placed.copy()
        sizes = np.sqrt(np.einsum("nij,nij->nj", placed[self.bone][:, :3, :3], placed[self.bone][:, :3, :3]))
        writers = self._writers
        out[self.bone[writers], :3, :3] = (bent * sizes[:, None, :])[writers]
        animated_position = placed[self.bone, :3, 3]
        blended_position = animated_position + (self.position - animated_position) * self._weight[:, None]
        out[self.bone[writers], :3, 3] = blended_position[writers]
        result = world.copy()
        result[self.bone] = np.linalg.inv(transform) @ out[self.bone]
        return result
