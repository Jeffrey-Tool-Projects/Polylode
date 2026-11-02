"""Full-body biped IK: the game's FinalIK ``FullBodyBipedIK`` solver, transcribed from its IL.

Source: ``RootMotion.FinalIK`` as compiled into the game's
``Assembly-CSharp-firstpass.dll`` -- ``IKSolver``, ``IKSolverFullBody``,
``IKSolverFullBodyBiped``, ``FBIKChain`` (+ ``ChildConstraint``), ``IKEffector``,
``IKConstraintBend``, ``IKMapping.BoneMap``, ``IKMappingSpine``,
``IKMappingLimb``, ``IKMappingBone``, ``V3Tools`` and ``QuaTools``.  Each
function names the IL method it follows; branch order, operand order and the
places a value is written back to a serialized field are kept as the IL has
them.  The H scenes' ``MotionIK`` runs this solver on the character after its
animation.

Everything here is in Unity's frame (left-handed; quaternions are x, y, z, w)
and runs in float64 where the game runs float32.  The caller converts at the
boundary.

What a frame looks like::

    settings = settings_from_serialized(component["solver"], names.get)
    ik = FullBodyBipedIK(settings, parents, pose_at_initiate)   # IKSolver.Initiate
    pose = ik.update(animated_pose, targets, root_rotation)     # IKSolver.Update

``pose`` maps bone name -> (world position (3,), world rotation xyzw (4,)) for
every bone in ``parents``; ``targets`` maps the effectors' target and bend-goal
transform names to the same pair.  The solver keeps what FinalIK keeps between
frames (see :class:`FullBodyBipedIK`), and -- like the component -- writes back
into the settings objects it was given where the IL writes a serialized field.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

Vec = np.ndarray
Quat = np.ndarray
Pose = Mapping[str, tuple]

# --------------------------------------------------------------------------
# Unity engine math.  These live in UnityEngine's native code, not in the
# DLL; they follow Unity 5.6's documented semantics (the game's engine).
# --------------------------------------------------------------------------

_V_EPS = 1e-5                         # Vector3.kEpsilon
_V_EQ_SQR = 9.99999944e-11            # Vector3 ==: sqrMagnitude of the difference below this
_Q_EQ_DOT = 0.999999                  # Quaternion ==: dot product above this
_MATHF_EPS = 1.401298464324817e-45    # Mathf.Epsilon (float.Epsilon)
_RAD2DEG = 57.29578                   # Mathf.Rad2Deg
_ONE_OVER_SQRT2 = 0.7071067811865476  # k1OverSqrt2 in OrthoNormalVectorFast
_FROM_TO_PARALLEL = 1e-6              # Matrix3x3f::SetFromToRotation's EPSILON
_INF = math.inf
_ZERO = np.zeros(3)
_UP = np.array([0.0, 1.0, 0.0])
_Z_AXIS = np.array([0.0, 0.0, 1.0])
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _f(v) -> list:
    """A small vector's or quaternion's components as Python floats.

    The helpers below do their arithmetic on these: numpy's per-call cost on
    3- and 4-element arrays was most of a solve.  The formulas, their operand
    order and every threshold are the ones the IL has.
    """
    try:
        return v.tolist()
    except AttributeError:
        return [float(c) for c in v]


def _t3(v) -> tuple:
    """A position as three floats; anything else is an error, as ``_vec``'s reshape makes it."""
    if isinstance(v, np.ndarray) and v.shape == (3,):
        return tuple(v.tolist())
    if isinstance(v, tuple) and len(v) == 3 and all(type(c) is float for c in v):
        return v
    return tuple(_vec(v).tolist())


def _t4(q) -> tuple:
    """A rotation as four floats."""
    if isinstance(q, np.ndarray) and q.shape == (4,):
        return tuple(q.tolist())
    if isinstance(q, tuple) and len(q) == 4 and all(type(c) is float for c in q):
        return q
    return tuple(_quat(q).tolist())


def _vec(v) -> Vec:
    return np.array(v, dtype=np.float64).reshape(3)


def _quat(q) -> Quat:
    return np.array(q, dtype=np.float64).reshape(4)


def _clamp(x: float, lo: float, hi: float) -> float:
    """Mathf.Clamp: below ``lo`` wins first, so ``lo > hi`` returns ``lo`` only when ``x < lo``."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _div(a: float, b: float) -> float:
    """IEEE float division, as C# does it: a zero divisor gives inf/NaN, not an exception."""
    if b != 0.0:
        return a / b
    if a == 0.0 or math.isnan(a):
        return math.nan
    return math.copysign(_INF, a) * math.copysign(1.0, b)


def _vdiv(v: Vec, d: float) -> Vec:
    """Vector3 / float with C#'s IEEE semantics."""
    return np.array([_div(c, d) for c in _f(v)])


def _dot(a, b) -> float:
    """Vector3.Dot / Quaternion.Dot."""
    return sum(x * y for x, y in zip(_f(a), _f(b)))


def _cross(a, b) -> Vec:
    """Vector3.Cross."""
    ax, ay, az = _f(a)
    bx, by, bz = _f(b)
    return np.array([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])


def _mag(v: Vec) -> float:
    x, y, z = _f(v)
    return math.sqrt(x * x + y * y + z * z)


def _normalized(v: Vec) -> Vec:
    """Vector3.normalized: zero below kEpsilon."""
    x, y, z = _f(v)
    m = math.sqrt(x * x + y * y + z * z)
    return np.array([x / m, y / m, z / m]) if m > _V_EPS else _ZERO.copy()


def _v_eq(a: Vec, b: Vec) -> bool:
    """Vector3 ==, approximate."""
    ax, ay, az = _f(a)
    bx, by, bz = _f(b)
    dx, dy, dz = ax - bx, ay - by, az - bz
    return dx * dx + dy * dy + dz * dz < _V_EQ_SQR


def _lerp(a: Vec, b: Vec, t: float) -> Vec:
    """Vector3.Lerp, ``t`` clamped to [0, 1]."""
    t = _clamp(t, 0.0, 1.0)
    ax, ay, az = _f(a)
    bx, by, bz = _f(b)
    return np.array([ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t])


def _project(v: Vec, on_normal: Vec) -> Vec:
    """Vector3.Project."""
    nx, ny, nz = _f(on_normal)
    sqr = nx * nx + ny * ny + nz * nz
    if sqr < _MATHF_EPS:
        return _ZERO.copy()
    vx, vy, vz = _f(v)
    d = vx * nx + vy * ny + vz * nz
    return np.array([nx * d / sqr, ny * d / sqr, nz * d / sqr])


def _angle(a: Vec, b: Vec) -> float:
    """Vector3.Angle in degrees (Unity 5.6: acos of the clamped dot of the normalized pair)."""
    ax, ay, az = _f(_normalized(a))
    bx, by, bz = _f(_normalized(b))
    return math.acos(_clamp(ax * bx + ay * by + az * bz, -1.0, 1.0)) * _RAD2DEG


def _ortho_normal_vector_fast(n: Vec) -> Vec:
    """OrthoNormalVectorFast: a unit vector perpendicular to unit ``n``."""
    x, y, z = _f(n)
    if abs(z) > _ONE_OVER_SQRT2:
        k = 1.0 / math.sqrt(y * y + z * z)
        return np.array([0.0, -z * k, y * k])
    k = 1.0 / math.sqrt(x * x + y * y)
    return np.array([-y * k, x * k, 0.0])


def _ortho_normalize(normal: Vec, tangent: Vec) -> tuple:
    """Vector3.OrthoNormalize(ref normal, ref tangent) -> (normal, tangent)."""
    nx, ny, nz = _f(normal)
    m = math.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = (nx / m, ny / m, nz / m) if m > _V_EPS else (1.0, 0.0, 0.0)
    tx, ty, tz = _f(tangent)
    d = nx * tx + ny * ty + nz * tz
    tx, ty, tz = tx - d * nx, ty - d * ny, tz - d * nz
    m = math.sqrt(tx * tx + ty * ty + tz * tz)
    n = np.array([nx, ny, nz])
    t = _ortho_normal_vector_fast(n) if m < _V_EPS else np.array([tx / m, ty / m, tz / m])
    return n, t


def _axis_angle_rotate(axis: Vec, angle: float, v: Vec) -> Vec:
    """Matrix3x3f::SetAxisAngle(axis, angle) applied to ``v`` (Rodrigues)."""
    c, s = math.cos(angle), math.sin(angle)
    ax, ay, az = _f(axis)
    vx, vy, vz = _f(v)
    cx, cy, cz = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx
    k = (ax * vx + ay * vy + az * vz) * (1.0 - c)
    return np.array([vx * c + cx * s + ax * k, vy * c + cy * s + ay * k, vz * c + cz * s + az * k])


def _slerp(a: Vec, b: Vec, t: float) -> Vec:
    """Vector3.Slerp: direction by angle, magnitude by lerp; ``t`` clamped to [0, 1]."""
    t = _clamp(t, 0.0, 1.0)
    am, bm = _mag(a), _mag(b)
    if am < _V_EPS or bm < _V_EPS:
        return _lerp(a, b, t)
    mag = am + (bm - am) * t
    dot = _dot(a, b) / (am * bm)
    if dot > 1.0 - _V_EPS:
        return _lerp(a, b, t)
    an = np.array([c / am for c in _f(a)])
    if dot < -1.0 + _V_EPS:
        return _axis_angle_rotate(_ortho_normal_vector_fast(an), math.pi * t, an) * mag
    axis = _normalized(_cross(a, b))
    return _axis_angle_rotate(axis, math.acos(dot) * t, an) * mag


def _mul4(a, b) -> tuple:
    """Quaternion * Quaternion on float sequences."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by + ay * bw + az * bx - ax * bz,
            aw * bz + az * bw + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz)


def _rot3(q, p) -> tuple:
    """Quaternion * Vector3 on float sequences, Unity's expanded matrix form (a zero quaternion leaves ``p`` alone)."""
    x, y, z, w = q
    px, py, pz = p
    x2, y2, z2 = x * 2.0, y * 2.0, z * 2.0
    xx, yy, zz = x * x2, y * y2, z * z2
    xy, xz, yz = x * y2, x * z2, y * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    return ((1.0 - (yy + zz)) * px + (xy - wz) * py + (xz + wy) * pz,
            (xy + wz) * px + (1.0 - (xx + zz)) * py + (yz - wx) * pz,
            (xz - wy) * px + (yz + wx) * py + (1.0 - (xx + yy)) * pz)


def _q_mul(a: Quat, b: Quat) -> Quat:
    """Quaternion * Quaternion."""
    return np.array(_mul4(_f(a), _f(b)))


def _q_rot(q: Quat, p: Vec) -> Vec:
    """Quaternion * Vector3 (:func:`_rot3`)."""
    return np.array(_rot3(_f(q), _f(p)))


def _q_inv(q: Quat) -> Quat:
    """Quaternion.Inverse: the conjugate."""
    x, y, z, w = _f(q)
    return np.array([-x, -y, -z, w])


def _q_eq(a: Quat, b: Quat) -> bool:
    """Quaternion ==: dot above 0.999999 (sign-sensitive, as Unity's is)."""
    return _dot(a, b) > _Q_EQ_DOT


def _q_normalize(q: Quat) -> Quat:
    x, y, z, w = _f(q)
    m = math.sqrt(x * x + y * y + z * z + w * w)
    return np.array([x / m, y / m, z / m, w / m])


def _q_lerp(a: Quat, b: Quat, t: float) -> Quat:
    """Quaternion.Lerp: componentwise toward the nearer sign of ``b``, then normalized."""
    t = _clamp(t, 0.0, 1.0)
    a, b = _f(a), _f(b)
    if sum(x * y for x, y in zip(a, b)) < 0.0:
        return _q_normalize([x + t * (-y - x) for x, y in zip(a, b)])
    return _q_normalize([x + t * (y - x) for x, y in zip(a, b)])


def _q_from_matrix(m) -> Quat:
    """MatrixToQuaternion (Shoemake), then normalized; *m* is 3×3 as rows."""
    trace = m[0][0] + m[1][1] + m[2][2]
    q = [0.0, 0.0, 0.0, 0.0]
    if trace > 0.0:
        root = math.sqrt(trace + 1.0)
        q[3] = 0.5 * root
        root = 0.5 / root
        q[0] = (m[2][1] - m[1][2]) * root
        q[1] = (m[0][2] - m[2][0]) * root
        q[2] = (m[1][0] - m[0][1]) * root
        return _q_normalize(q)
    i = 0
    if m[1][1] > m[0][0]:
        i = 1
    if m[2][2] > m[i][i]:
        i = 2
    j, k = (i + 1) % 3, (i + 2) % 3
    root = math.sqrt(m[i][i] - m[j][j] - m[k][k] + 1.0)
    q[i] = 0.5 * root
    root = 0.5 / root
    q[3] = (m[k][j] - m[j][k]) * root
    q[j] = (m[j][i] + m[i][j]) * root
    q[k] = (m[k][i] + m[i][k]) * root
    return _q_normalize(q)


def _from_to_matrix(f: Vec, t: Vec) -> list:
    """Matrix3x3f::SetFromToRotation (Moller-Hughes) for unit ``f`` and ``t``; 3×3 as rows."""
    f, t = _f(f), _f(t)
    e = f[0] * t[0] + f[1] * t[1] + f[2] * t[2]
    if abs(e) > 1.0 - _FROM_TO_PARALLEL:
        x = [abs(c) for c in f]
        axis = [0.0, 0.0, 0.0]
        if x[0] < x[1]:
            axis[0 if x[0] < x[2] else 2] = 1.0
        else:
            axis[1 if x[1] < x[2] else 2] = 1.0
        u = [a - c for a, c in zip(axis, f)]
        v = [a - c for a, c in zip(axis, t)]
        c1 = 2.0 / (u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        c2 = 2.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        c3 = c1 * c2 * (u[0] * v[0] + u[1] * v[1] + u[2] * v[2])
        return [[(1.0 if r == c else 0.0) - c1 * u[r] * u[c] - c2 * v[r] * v[c] + c3 * v[r] * u[c]
                 for c in range(3)] for r in range(3)]
    v0, v1, v2 = f[1] * t[2] - f[2] * t[1], f[2] * t[0] - f[0] * t[2], f[0] * t[1] - f[1] * t[0]
    h = 1.0 / (1.0 + e)
    return [[e + h * v0 * v0, h * v0 * v1 - v2, h * v0 * v2 + v1],
            [h * v0 * v1 + v2, e + h * v1 * v1, h * v1 * v2 - v0],
            [h * v0 * v2 - v1, h * v1 * v2 + v0, e + h * v2 * v2]]


def _q_from_to(a: Vec, b: Vec) -> Quat:
    """Quaternion.FromToRotation: identity when either vector is shorter than kEpsilon."""
    am, bm = _mag(a), _mag(b)
    if am < _V_EPS or bm < _V_EPS:
        return IDENTITY.copy()
    return _q_from_matrix(_from_to_matrix([c / am for c in _f(a)], [c / bm for c in _f(b)]))


def _q_look(forward: Vec, up: Vec) -> Quat:
    """Quaternion.LookRotation; falls back to from-to(+Z, forward) when ``up`` is parallel."""
    mag = _mag(forward)
    if mag < _V_EPS:
        return IDENTITY.copy()
    zx, zy, zz = (c / mag for c in _f(forward))
    ux, uy, uz = _f(up)
    xx, xy, xz = uy * zz - uz * zy, uz * zx - ux * zz, ux * zy - uy * zx
    xm = math.sqrt(xx * xx + xy * xy + xz * xz)
    if xm < _V_EPS:
        return _q_from_matrix(_from_to_matrix(_Z_AXIS, [zx, zy, zz]))
    xx, xy, xz = xx / xm, xy / xm, xz / xm
    yx, yy, yz = zy * xz - zz * xy, zz * xx - zx * xz, zx * xy - zy * xx
    return _q_from_matrix([[xx, yx, zx], [xy, yy, zy], [xz, yz, zz]])


# --------------------------------------------------------------------------
# V3Tools (RootMotion), from the IL.
# --------------------------------------------------------------------------

def _extract_vertical(v: Vec, vertical_axis: Vec, weight: float) -> Vec:
    """V3Tools.ExtractVertical."""
    if weight == 0.0:
        return _ZERO.copy()
    return _project(v, vertical_axis) * weight


def _extract_horizontal(v: Vec, normal: Vec, weight: float) -> Vec:
    """V3Tools.ExtractHorizontal."""
    if weight == 0.0:
        return _ZERO.copy()
    _, tangent = _ortho_normalize(normal, v.copy())
    return _project(v, tangent) * weight


def _clamp_direction(direction: Vec, normal_direction: Vec, clamp_weight: float,
                     clamp_smoothing: int) -> tuple:
    """V3Tools.ClampDirection -> (direction, changed)."""
    if not clamp_weight > 0.0:
        return direction, False
    if not clamp_weight < 1.0:
        return normal_direction, True
    dot = 1.0 - _angle(normal_direction, direction) / 180.0
    if dot > clamp_weight:
        return direction, False
    target_clamp = (_clamp(1.0 - _div(clamp_weight - dot, 1.0 - dot), 0.0, 1.0)
                    if clamp_weight > 0.0 else 1.0)
    clamp_mlp = _clamp(_div(dot, clamp_weight), 0.0, 1.0) if clamp_weight > 0.0 else 1.0
    for _ in range(clamp_smoothing):
        clamp_mlp = math.sin(clamp_mlp * 3.14159 * 0.5)
    return _slerp(normal_direction, direction, clamp_mlp * target_clamp), True


# --------------------------------------------------------------------------
# Transforms: the bones' world pose with Unity's parent/child propagation.
# --------------------------------------------------------------------------

class _Transforms:
    """World poses of a bone set; setting one moves its descendants as Unity's Transform does.

    A bone's local position is kept in its parent's *rotation* frame with the
    parent's scale already applied (``inv(parent_rot) * (pos - parent_pos)``),
    so the hierarchy needs no scales: that is exact while the scales above the
    solved bones are uniform and do not change during a solve.  A bone whose
    parent is not in the set (``None``) is placed in world space directly.
    Poses are held as float tuples; the accessors hand out arrays.
    """

    def __init__(self, parents: Mapping[str, str | None]):
        self.parent = dict(parents)
        for name, par in self.parent.items():
            if par is not None and par not in self.parent:
                raise ValueError(f"bone '{name}' names parent '{par}', which is not in the bone set")
        self.order = self._topological()
        self.children: dict = {n: [] for n in self.order}
        for n in self.order:
            if self.parent[n] is not None:
                self.children[self.parent[n]].append(n)
        self.pos: dict = {}
        self.rot: dict = {}
        self.lpos: dict = {}
        self.lrot: dict = {}

    def _topological(self) -> list:
        order, state = [], {}
        for start in self.parent:
            path = []
            n = start
            while n is not None and n not in state:
                if n in path:
                    raise ValueError(f"bone hierarchy has a cycle through '{n}'")
                path.append(n)
                n = self.parent[n]
            for m in reversed(path):
                state[m] = True
                order.append(m)
        return order

    def load(self, pose: Pose) -> None:
        """Take a world pose for every bone (the animated pose of this frame)."""
        missing = [n for n in self.order if n not in pose]
        if missing:
            raise KeyError(f"pose lacks bones {missing}")
        for n in self.order:
            p, q = pose[n]
            self.pos[n], self.rot[n] = _t3(p), _t4(q)
        for n in self.order:
            self._store_local(n)

    def _store_local(self, n: str) -> None:
        par = self.parent[n]
        if par is None:
            self.lpos[n], self.lrot[n] = self.pos[n], self.rot[n]
            return
        x, y, z, w = self.rot[par]
        inv = (-x, -y, -z, w)
        p, pp = self.pos[n], self.pos[par]
        self.lpos[n] = _rot3(inv, (p[0] - pp[0], p[1] - pp[1], p[2] - pp[2]))
        self.lrot[n] = _mul4(inv, self.rot[n])

    def _propagate(self, n: str) -> None:
        p, q = self.pos[n], self.rot[n]
        for c in self.children[n]:
            d = _rot3(q, self.lpos[c])
            self.pos[c] = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
            self.rot[c] = _mul4(q, self.lrot[c])
            self._propagate(c)

    def _world_from_local(self, n: str) -> None:
        par = self.parent[n]
        if par is None:
            self.pos[n], self.rot[n] = self.lpos[n], self.lrot[n]
        else:
            p, q = self.pos[par], self.rot[par]
            d = _rot3(q, self.lpos[n])
            self.pos[n] = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
            self.rot[n] = _mul4(q, self.lrot[n])
        self._propagate(n)

    def position(self, n: str) -> Vec:
        return np.array(self.pos[n])

    def rotation(self, n: str) -> Quat:
        return np.array(self.rot[n])

    def set_position(self, n: str, p: Vec) -> None:
        """Transform.position = p."""
        self.pos[n] = _t3(p)
        self._store_local(n)
        self._propagate(n)

    def set_rotation(self, n: str, q: Quat) -> None:
        """Transform.rotation = q; Unity stores the local rotation normalized."""
        par = self.parent[n]
        q = _t4(q)
        if par is None:
            local = q
        else:
            x, y, z, w = self.rot[par]
            local = _mul4((-x, -y, -z, w), q)
        m = math.sqrt(local[0] * local[0] + local[1] * local[1] + local[2] * local[2] + local[3] * local[3])
        self.lrot[n] = tuple(c / m for c in local) if m > _V_EPS else (0.0, 0.0, 0.0, 1.0)
        self._world_from_local(n)

    def local_position(self, n: str) -> Vec:
        return np.array(self.lpos[n])

    def local_rotation(self, n: str) -> Quat:
        return np.array(self.lrot[n])

    def set_local_position(self, n: str, p: Vec) -> None:
        self.lpos[n] = _t3(p)
        self._world_from_local(n)

    def set_local_rotation(self, n: str, q: Quat) -> None:
        self.lrot[n] = _t4(q)
        self._world_from_local(n)

    def snapshot(self) -> dict:
        return {n: (np.array(self.pos[n]), np.array(self.rot[n])) for n in self.order}


# --------------------------------------------------------------------------
# Serialized settings (the component's fields, bone references as names).
# --------------------------------------------------------------------------

@dataclass
class NodeSettings:
    """IKSolver.Node's serialized fields the solve reads before overwriting them."""
    bone: str
    offset: Vec = field(default_factory=lambda: _ZERO.copy())


@dataclass
class ChildConstraintSettings:
    """FBIKChain.ChildConstraint: bones whose chains' first nodes keep their distance."""
    bone1: str
    bone2: str
    push_elasticity: float = 0.0
    pull_elasticity: float = 0.0


@dataclass
class BendSettings:
    """IKConstraintBend: ``bend_goal`` names the goal transform (or None)."""
    bend_goal: str | None = None
    direction: Vec = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    rotation_offset: Quat = field(default_factory=lambda: np.zeros(4))
    weight: float = 0.0


@dataclass
class ChainSettings:
    """FBIKChain."""
    nodes: list
    pin: float = 0.0
    pull: float = 1.0
    push: float = 0.0
    push_parent: float = 0.0
    reach: float = 0.1
    reach_smoothing: int = 1
    push_smoothing: int = 1
    children: list = field(default_factory=list)
    child_constraints: list = field(default_factory=list)
    bend: BendSettings = field(default_factory=BendSettings)


@dataclass
class EffectorSettings:
    """IKEffector.  ``target`` names the target transform (or None)."""
    bone: str
    target: str | None = None
    position_weight: float = 0.0
    rotation_weight: float = 0.0
    position: Vec = field(default_factory=lambda: _ZERO.copy())
    rotation: Quat = field(default_factory=lambda: IDENTITY.copy())
    position_offset: Vec = field(default_factory=lambda: _ZERO.copy())
    effect_child_nodes: bool = True
    maintain_relative_position_weight: float = 0.0
    child_bones: list = field(default_factory=list)
    plane_bone1: str | None = None
    plane_bone2: str | None = None
    plane_bone3: str | None = None
    plane_rotation_offset: Quat = field(default_factory=lambda: IDENTITY.copy())


@dataclass
class SpineMappingSettings:
    """IKMappingSpine."""
    spine_bones: list
    left_upper_arm: str
    right_upper_arm: str
    left_thigh: str
    right_thigh: str
    iterations: int = 3
    twist_weight: float = 1.0


@dataclass
class LimbMappingSettings:
    """IKMappingLimb."""
    bone1: str
    bone2: str
    bone3: str
    parent_bone: str | None = None
    maintain_rotation_weight: float = 0.0
    weight: float = 1.0


@dataclass
class BoneMappingSettings:
    """IKMappingBone."""
    bone: str
    maintain_rotation_weight: float = 1.0


@dataclass
class FBBIKSettings:
    """IKSolverFullBodyBiped's serialized fields."""
    chains: list
    effectors: list
    spine: SpineMappingSettings
    bone_mappings: list
    limb_mappings: list
    root_node: str | None
    ik_position_weight: float = 1.0
    iterations: int = 4
    spine_stiffness: float = 0.5
    pull_body_vertical: float = 0.5
    pull_body_horizontal: float = 0.0


def _v3(d: Mapping) -> Vec:
    return np.array([d["x"], d["y"], d["z"]], dtype=np.float64)


def _q4(d: Mapping) -> Quat:
    return np.array([d["x"], d["y"], d["z"], d["w"]], dtype=np.float64)


def settings_from_serialized(solver: Mapping, bone_name: Callable[[int], str | None]) -> FBBIKSettings:
    """Read a serialized ``IKSolverFullBodyBiped`` (the component's ``solver`` field).

    ``bone_name`` resolves an object id (``m_PathID``) to the transform's name;
    id 0 is a null reference.  An id it cannot resolve is an error.
    """
    def ref(pptr: Mapping) -> str | None:
        pid = int(pptr["m_PathID"])
        if pid == 0:
            return None
        name = bone_name(pid)
        if name is None:
            raise KeyError(f"object id {pid} does not resolve to a transform")
        return name

    def chain(c: Mapping) -> ChainSettings:
        b = c["bendConstraint"]
        return ChainSettings(
            nodes=[NodeSettings(ref(n["transform"]), _v3(n["offset"])) for n in c["nodes"]],
            pin=c["pin"], pull=c["pull"], push=c["push"], push_parent=c["pushParent"],
            reach=c["reach"], reach_smoothing=int(c["reachSmoothing"]),
            push_smoothing=int(c["pushSmoothing"]), children=[int(i) for i in c["children"]],
            child_constraints=[ChildConstraintSettings(ref(k["bone1"]), ref(k["bone2"]),
                                                       k["pushElasticity"], k["pullElasticity"])
                               for k in c["childConstraints"]],
            bend=BendSettings(ref(b["bendGoal"]), _v3(b["direction"]), _q4(b["rotationOffset"]),
                              b["weight"]))

    def effector(e: Mapping) -> EffectorSettings:
        return EffectorSettings(
            bone=ref(e["bone"]), target=ref(e["target"]), position_weight=e["positionWeight"],
            rotation_weight=e["rotationWeight"], position=_v3(e["position"]),
            rotation=_q4(e["rotation"]), position_offset=_v3(e["positionOffset"]),
            effect_child_nodes=bool(e["effectChildNodes"]),
            maintain_relative_position_weight=e["maintainRelativePositionWeight"],
            child_bones=[ref(b) for b in e["childBones"]], plane_bone1=ref(e["planeBone1"]),
            plane_bone2=ref(e["planeBone2"]), plane_bone3=ref(e["planeBone3"]),
            plane_rotation_offset=_q4(e["planeRotationOffset"]))

    sm = solver["spineMapping"]
    return FBBIKSettings(
        chains=[chain(c) for c in solver["chain"]],
        effectors=[effector(e) for e in solver["effectors"]],
        spine=SpineMappingSettings([ref(b) for b in sm["spineBones"]], ref(sm["leftUpperArmBone"]),
                                   ref(sm["rightUpperArmBone"]), ref(sm["leftThighBone"]),
                                   ref(sm["rightThighBone"]), int(sm["iterations"]),
                                   sm["twistWeight"]),
        bone_mappings=[BoneMappingSettings(ref(b["bone"]), b["maintainRotationWeight"])
                       for b in solver["boneMappings"]],
        limb_mappings=[LimbMappingSettings(ref(m["bone1"]), ref(m["bone2"]), ref(m["bone3"]),
                                           ref(m["parentBone"]), m["maintainRotationWeight"],
                                           m["weight"])
                       for m in solver["limbMappings"]],
        root_node=ref(solver["rootNode"]), ik_position_weight=solver["IKPositionWeight"],
        iterations=int(solver["iterations"]), spine_stiffness=solver["spineStiffness"],
        pull_body_vertical=solver["pullBodyVertical"],
        pull_body_horizontal=solver["pullBodyHorizontal"])


# --------------------------------------------------------------------------
# Solver parts.
# --------------------------------------------------------------------------

class _Node:
    """IKSolver.Node: one chain bone's solver state."""

    def __init__(self, s: NodeSettings):
        self.transform = s.bone
        self.offset = _vec(s.offset)
        self.length = 0.0
        self.solver_position = _ZERO.copy()
        self.solver_rotation = IDENTITY.copy()
        self.effector_position_weight = 0.0
        self.effector_rotation_weight = 0.0


class _ChildConstraint:
    """FBIKChain.ChildConstraint."""

    def __init__(self, s: ChildConstraintSettings):
        self.s = s
        self.chain1_index = -1
        self.chain2_index = -1
        self.nominal_distance = 0.0
        self.is_rigid = False
        self.cross_fade = 0.0
        self.inverse_cross_fade = 0.0

    def initiate(self, solver: FullBodyBipedIK) -> None:
        """ChildConstraint.Initiate."""
        self.chain1_index = solver.get_chain_index(self.s.bone1)
        self.chain2_index = solver.get_chain_index(self.s.bone2)
        self.on_pre_solve(solver)

    def on_pre_solve(self, solver: FullBodyBipedIK) -> None:
        """ChildConstraint.OnPreSolve: distance from the animated pose, rigid when both elasticities are 0."""
        n1, n2 = solver.get_node(self.chain1_index, 0), solver.get_node(self.chain2_index, 0)
        c1, c2 = solver.chain[self.chain1_index], solver.chain[self.chain2_index]
        self.nominal_distance = _mag(solver.tf.position(n1.transform) - solver.tf.position(n2.transform))
        self.is_rigid = (not self.s.push_elasticity > 0.0) and (not self.s.pull_elasticity > 0.0)
        if self.is_rigid:
            self.cross_fade = 1.0 - (0.5 + (c1.s.pull - c2.s.pull) * 0.5)
        else:
            self.cross_fade = 0.5
        self.inverse_cross_fade = 1.0 - self.cross_fade

    def solve(self, solver: FullBodyBipedIK) -> None:
        """ChildConstraint.Solve."""
        if not self.s.push_elasticity < 1.0 and not self.s.pull_elasticity < 1.0:
            return
        n1 = solver.chain[self.chain1_index].nodes[0]
        n2 = solver.chain[self.chain2_index].nodes[0]
        a, b = _f(n1.solver_position), _f(n2.solver_position)
        d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
        mag = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if mag == self.nominal_distance or mag == 0.0:
            return
        force = 1.0
        if not self.is_rigid:
            elasticity = self.s.pull_elasticity if mag > self.nominal_distance else self.s.push_elasticity
            force = 1.0 - elasticity
        force *= 1.0 - self.nominal_distance / mag
        offset = [c * force for c in d]
        n1.solver_position = np.array([p + o * self.cross_fade for p, o in zip(a, offset)])
        n2.solver_position = np.array([p - o * self.inverse_cross_fade for p, o in zip(b, offset)])


class _BendConstraint:
    """IKConstraintBend: the bend plane of a 3-node limb chain."""

    CLAMP_F = 0.505    # IKConstraintBend..ctor: clampF

    def __init__(self, s: BendSettings):
        self.s = s
        self.bone1 = self.bone2 = self.bone3 = ""
        self.initiated = False
        self.chain_index1 = self.node_index1 = -1
        self.chain_index2 = self.node_index2 = -1
        self.chain_index3 = self.node_index3 = -1
        self.default_local_direction = _ZERO.copy()
        self.default_child_direction = _ZERO.copy()
        self.goal_position: Vec | None = None      # bendGoal.position this frame; None = no goal

    def set_bones(self, bone1: str, bone2: str, bone3: str) -> None:
        """IKConstraintBend.SetBones."""
        self.bone1, self.bone2, self.bone3 = bone1, bone2, bone3

    def _node(self, solver: FullBodyBipedIK, k: int) -> _Node:
        ci, ni = ((self.chain_index1, self.node_index1), (self.chain_index2, self.node_index2),
                  (self.chain_index3, self.node_index3))[k - 1]
        return solver.get_node(ci, ni)

    def initiate(self, solver: FullBodyBipedIK) -> None:
        """IKConstraintBend.Initiate: the bend direction and its bone-local copies from the pose now."""
        tf = solver.tf
        self.chain_index1, self.node_index1 = solver.get_chain_and_node_indexes(self.bone1)
        self.chain_index2, self.node_index2 = solver.get_chain_and_node_indexes(self.bone2)
        self.chain_index3, self.node_index3 = solver.get_chain_and_node_indexes(self.bone3)
        self.s.direction = self.ortho_to_bone1(
            solver, self.ortho_to_limb(solver, tf.position(self.bone2) - tf.position(self.bone1)))
        self.default_local_direction = _q_rot(_q_inv(tf.rotation(self.bone1)), self.s.direction)
        limb = _normalized(tf.position(self.bone3) - tf.position(self.bone1))
        self.default_child_direction = _q_rot(_q_inv(tf.rotation(self.bone3)),
                                              _cross(limb, self.s.direction))
        self.initiated = True

    def limit_bend(self, solver: FullBodyBipedIK, solver_weight: float, position_weight: float) -> None:
        """IKConstraintBend.LimitBend: keep the animated limb from bending the wrong way."""
        if not self.initiated:
            return
        tf = solver.tf
        normal_dir = _q_rot(tf.rotation(self.bone1), -self.default_local_direction)
        direction = tf.position(self.bone3) - tf.position(self.bone2)
        clamped, changed = _clamp_direction(direction, normal_dir, self.CLAMP_F * solver_weight, 0)
        bone3_rot = tf.rotation(self.bone3)
        if changed:
            tf.set_rotation(self.bone2, _q_mul(_q_from_to(direction, clamped), tf.rotation(self.bone2)))
        if position_weight > 0.0:
            upper = tf.position(self.bone2) - tf.position(self.bone1)
            lower = tf.position(self.bone3) - tf.position(self.bone2)
            upper, lower = _ortho_normalize(upper, lower)
            r = _q_from_to(lower, normal_dir)
            b2 = tf.rotation(self.bone2)
            tf.set_rotation(self.bone2, _q_lerp(b2, _q_mul(r, b2), position_weight * solver_weight))
        if changed or position_weight > 0.0:
            tf.set_rotation(self.bone3, bone3_rot)

    def get_dir(self, solver: FullBodyBipedIK) -> Vec:
        """IKConstraintBend.GetDir: the bend direction for the trigonometric step."""
        if not self.initiated:
            return _ZERO.copy()
        w = self.s.weight * solver.s.ik_position_weight
        n1, n3 = self._node(solver, 1), self._node(solver, 3)
        if self.goal_position is not None:
            d = self.goal_position - n1.solver_position
            if not _v_eq(d, _ZERO):
                self.s.direction = d
        if not w < 1.0:
            return _normalized(self.s.direction)
        tf = solver.tf
        limb = n3.solver_position - n1.solver_position
        r = _q_from_to(tf.position(self.bone3) - tf.position(self.bone1), limb)
        bend = _q_rot(r, tf.position(self.bone2) - tf.position(self.bone1))
        if n3.effector_rotation_weight > 0.0:
            from_rot = -_cross(limb, _q_rot(n3.solver_rotation, self.default_child_direction))
            bend = _lerp(bend, from_rot, n3.effector_rotation_weight)
        if not _q_eq(self.s.rotation_offset, IDENTITY):
            off = _q_from_to(_q_rot(self.s.rotation_offset, limb), limb)
            bend = _q_rot(_q_mul(off, self.s.rotation_offset), bend)
        if not w > 0.0:
            return bend
        return _lerp(bend, _normalized(self.s.direction), w)

    def ortho_to_limb(self, solver: FullBodyBipedIK, tangent: Vec) -> Vec:
        """IKConstraintBend.OrthoToLimb."""
        limb = self._node(solver, 3).solver_position - self._node(solver, 1).solver_position
        return _ortho_normalize(limb, tangent)[1]

    def ortho_to_bone1(self, solver: FullBodyBipedIK, tangent: Vec) -> Vec:
        """IKConstraintBend.OrthoToBone1."""
        bone = self._node(solver, 2).solver_position - self._node(solver, 1).solver_position
        return _ortho_normalize(bone, tangent)[1]


def _solve_fabrik_joint(pos1: Vec, pos2: Vec, length: float) -> Vec:
    """FBIKChain.SolveFABRIKJoint / IKMapping.SolveFABRIKJoint: ``pos2 + (pos1 - pos2).normalized * length``."""
    ax, ay, az = _f(pos1)
    bx, by, bz = _f(pos2)
    dx, dy, dz = ax - bx, ay - by, az - bz
    m = math.sqrt(dx * dx + dy * dy + dz * dz)
    nx, ny, nz = (dx / m, dy / m, dz / m) if m > _V_EPS else (0.0, 0.0, 0.0)
    return np.array([bx + nx * length, by + ny * length, bz + nz * length])


def _smooth(v: float, smoothing: int) -> float:
    """FBIKChain.Reach/Push smoothing switch: 1 squares, 2 cubes, anything else leaves it."""
    if smoothing == 1:
        return v * v
    if smoothing == 2:
        return v * (v * v)
    return v


class _Chain:
    """FBIKChain: the body's one-node chain or a limb's three nodes."""

    def __init__(self, s: ChainSettings):
        self.s = s
        self.nodes = [_Node(n) for n in s.nodes]
        self.child_constraints = [_ChildConstraint(c) for c in s.child_constraints]
        self.bend = _BendConstraint(s.bend)
        self.initiated = False
        self.length = 0.0
        self.root_length = 0.0
        self.distance = 0.0
        self.sqr_mag1 = self.sqr_mag2 = self.sqr_mag_dif = 0.0
        self.reach_force = 0.0
        self.pull_parent_sum = 0.0
        self.cross_fades: list = []

    def _child(self, solver: FullBodyBipedIK, i: int) -> _Chain:
        return solver.chain[self.s.children[i]]

    def initiate(self, solver: FullBodyBipedIK) -> None:
        """FBIKChain.Initiate."""
        self.initiated = False
        for n in self.nodes:
            n.solver_position = solver.tf.position(n.transform)
        self.calculate_bone_lengths(solver)
        for c in self.child_constraints:
            c.initiate(solver)
        if len(self.nodes) == 3:
            self.bend.set_bones(self.nodes[0].transform, self.nodes[1].transform, self.nodes[2].transform)
            self.bend.initiate(solver)
        self.cross_fades = [0.0] * len(self.s.children)
        self.initiated = True

    def read_pose(self, solver: FullBodyBipedIK, full_body: bool) -> None:
        """FBIKChain.ReadPose."""
        if not self.initiated:
            return
        for n in self.nodes:
            n.solver_position = solver.tf.position(n.transform) + n.offset
        self.calculate_bone_lengths(solver)
        if not full_body:
            return
        for c in self.child_constraints:
            c.on_pre_solve(solver)
        if len(self.s.children) > 0:
            self._read_cross_fades(solver)
        self.pull_parent_sum = 0.0
        for i in range(len(self.s.children)):
            self.pull_parent_sum += self._child(solver, i).s.pull
        self.pull_parent_sum = _clamp(self.pull_parent_sum, 1.0, _INF)
        if len(self.nodes) == 3:
            self.reach_force = self.s.reach * _clamp(self.nodes[2].effector_position_weight, 0.0, 1.0)
        else:
            self.reach_force = 0.0
        if self.s.push > 0.0 and len(self.nodes) > 1:
            self.distance = _mag(solver.tf.position(self.nodes[0].transform)
                                 - solver.tf.position(self.nodes[-1].transform))

    def _read_cross_fades(self, solver: FullBodyBipedIK) -> None:
        # FBIKChain.ReadPose: each child's share of the end node, by its effector weight and pull
        total = self.nodes[-1].effector_position_weight
        for i in range(len(self.s.children)):
            child = self._child(solver, i)
            total += child.nodes[0].effector_position_weight * child.s.pull
        total = _clamp(total, 1.0, _INF)
        for i in range(len(self.s.children)):
            child = self._child(solver, i)
            self.cross_fades[i] = child.nodes[0].effector_position_weight * child.s.pull / total

    def calculate_bone_lengths(self, solver: FullBodyBipedIK) -> None:
        """FBIKChain.CalculateBoneLengths: from the transforms; a zero length stops it, as in the IL."""
        tf = solver.tf
        self.length = 0.0
        for i in range(len(self.nodes) - 1):
            n = self.nodes[i]
            n.length = _mag(tf.position(n.transform) - tf.position(self.nodes[i + 1].transform))
            self.length += n.length
            if n.length == 0.0:
                warnings.warn(f"Bone {n.transform} - {self.nodes[i + 1].transform} length is zero, "
                              "can not solve.", RuntimeWarning, stacklevel=2)
                return
        for i in range(len(self.s.children)):
            child = self._child(solver, i)
            child.root_length = _mag(tf.position(child.nodes[0].transform)
                                     - tf.position(self.nodes[-1].transform))
            if child.root_length == 0.0:
                return
        if len(self.nodes) == 3:
            self.sqr_mag1 = self.nodes[0].length * self.nodes[0].length
            self.sqr_mag2 = self.nodes[1].length * self.nodes[1].length
            self.sqr_mag_dif = self.sqr_mag1 - self.sqr_mag2

    def reach(self, solver: FullBodyBipedIK) -> None:
        """FBIKChain.Reach: stretch the limb toward its end effector by ``reach``."""
        if not self.initiated:
            return
        for i in range(len(self.s.children)):
            self._child(solver, i).reach(solver)
        if not self.reach_force > 0.0:
            return
        limb = self.nodes[2].solver_position - self.nodes[0].solver_position
        if _v_eq(limb, _ZERO):
            return
        limb_mag = _mag(limb)
        full = _vdiv(limb, limb_mag) * self.length
        r = _clamp(_div(limb_mag, self.length), 1.0 - self.reach_force, 1.0 + self.reach_force) - 1.0
        r = _clamp(r + self.reach_force, -1.0, 1.0)
        r = _smooth(r, self.s.reach_smoothing)
        offset = full * _clamp(r, 0.0, limb_mag)
        n0 = self.nodes[0]
        n0.solver_position = n0.solver_position + offset * (1.0 - n0.effector_position_weight)
        self.nodes[2].solver_position = self.nodes[2].solver_position + offset

    def push(self, solver: FullBodyBipedIK) -> Vec:
        """FBIKChain.Push: returns the push this chain hands its parent."""
        total = _ZERO.copy()
        for i in range(len(self.s.children)):
            child = self._child(solver, i)
            total = total + child.push(solver) * child.s.push_parent
        self.nodes[-1].solver_position = self.nodes[-1].solver_position + total
        if len(self.nodes) < 2:
            return _ZERO.copy()
        if not self.s.push > 0.0:
            return _ZERO.copy()
        limb = self.nodes[2].solver_position - self.nodes[0].solver_position
        limb_mag = _mag(limb)
        if limb_mag == 0.0:
            return _ZERO.copy()
        p = 1.0 - _div(limb_mag, self.distance)
        if not p > 0.0:
            return _ZERO.copy()
        p = _smooth(p, self.s.push_smoothing)
        offset = (-limb * p) * self.s.push
        self.nodes[0].solver_position = self.nodes[0].solver_position + offset
        return offset

    def solve_trigonometric(self, solver: FullBodyBipedIK, calculate_bend_direction: bool) -> None:
        """FBIKChain.SolveTrigonometric: place the middle node on the bend plane."""
        if not self.initiated:
            return
        for i in range(len(self.s.children)):
            self._child(solver, i).solve_trigonometric(solver, calculate_bend_direction)
        if len(self.nodes) != 3:
            return
        n0, n1, n2 = self.nodes
        p0, p2 = _f(n0.solver_position), _f(n2.solver_position)
        d = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]]
        mag = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if mag == 0.0:
            return
        sqr = _clamp(mag, 0.0, self.length * 0.99999)
        direction = [_div(c, mag) * sqr for c in d]
        if calculate_bend_direction and self.bend.initiated:
            bend_direction = self.bend.get_dir(solver)
        else:
            p1 = _f(n1.solver_position)
            bend_direction = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
        to_bend = _f(self.get_dir_to_bend_point(direction, bend_direction, sqr))
        n1.solver_position = np.array([p0[0] + to_bend[0], p0[1] + to_bend[1], p0[2] + to_bend[2]])

    def get_dir_to_bend_point(self, direction: Vec, bend_direction: Vec, direction_mag: float) -> Vec:
        """FBIKChain.GetDirToBendPoint (the law of cosines in the bend plane)."""
        x = _div(_div(direction_mag * direction_mag + self.sqr_mag_dif, 2.0), direction_mag)
        y = math.sqrt(_clamp(self.sqr_mag1 - x * x, 0.0, _INF))
        if _v_eq(direction, _ZERO):
            return _ZERO.copy()
        return _q_rot(_q_look(direction, bend_direction), [0.0, y, x])

    def stage1(self, solver: FullBodyBipedIK) -> None:
        """FBIKChain.Stage1: forward reach, the body pulled by its children."""
        for i in range(len(self.s.children)):
            self._child(solver, i).stage1(solver)
        if len(self.s.children) == 0:
            self.forward_reach(self.nodes[-1].solver_position.copy())
            return
        centroid = self.nodes[-1].solver_position.copy()
        self.solve_child_constraints(solver)
        for i in range(len(self.s.children)):
            child = self._child(solver, i)
            child_pos = child.nodes[0].solver_position.copy()
            if child.root_length > 0.0:
                child_pos = _solve_fabrik_joint(self.nodes[-1].solver_position,
                                                child.nodes[0].solver_position, child.root_length)
            if self.pull_parent_sum > 0.0:
                centroid = centroid + (child_pos - self.nodes[-1].solver_position) * (
                    child.s.pull / self.pull_parent_sum)
        self.forward_reach(_lerp(centroid, self.nodes[-1].solver_position, self.s.pin))

    def stage2(self, solver: FullBodyBipedIK, position: Vec) -> None:
        """FBIKChain.Stage2: backward reach, then the child constraint systems."""
        self.backward_reach(position)
        rounds = int(_clamp(solver.s.iterations, 2, 4))
        if len(self.child_constraints) > 0:
            for _ in range(rounds):
                self.solve_constraint_systems(solver)
        for i in range(len(self.s.children)):
            self._child(solver, i).stage2(solver, self.nodes[-1].solver_position.copy())

    def solve_constraint_systems(self, solver: FullBodyBipedIK) -> None:
        """FBIKChain.SolveConstraintSystems."""
        self.solve_child_constraints(solver)
        for i in range(len(self.s.children)):
            child = self._child(solver, i)
            _solve_linear_constraint(self.nodes[-1], child.nodes[0], self.cross_fades[i],
                                     child.root_length)

    def solve_child_constraints(self, solver: FullBodyBipedIK) -> None:
        """FBIKChain.SolveChildConstraints."""
        for c in self.child_constraints:
            c.solve(solver)

    def forward_reach(self, position: Vec) -> None:
        """FBIKChain.ForwardReach."""
        self.nodes[-1].solver_position = _vec(position)
        for i in range(len(self.nodes) - 2, -1, -1):
            n = self.nodes[i]
            n.solver_position = _solve_fabrik_joint(n.solver_position, self.nodes[i + 1].solver_position,
                                                    n.length)

    def backward_reach(self, position: Vec) -> None:
        """FBIKChain.BackwardReach."""
        if self.root_length > 0.0:
            position = _solve_fabrik_joint(self.nodes[0].solver_position, position, self.root_length)
        self.nodes[0].solver_position = _vec(position)
        for i in range(1, len(self.nodes)):
            prev = self.nodes[i - 1]
            self.nodes[i].solver_position = _solve_fabrik_joint(self.nodes[i].solver_position,
                                                                prev.solver_position, prev.length)


def _solve_linear_constraint(node1: _Node, node2: _Node, cross_fade: float, distance: float) -> None:
    """FBIKChain.SolveLinearConstraint."""
    a, b = _f(node1.solver_position), _f(node2.solver_position)
    d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    mag = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    if distance == mag or mag == 0.0:
        return
    k = 1.0 - distance / mag
    offset = [c * k for c in d]
    node1.solver_position = np.array([p + o * cross_fade for p, o in zip(a, offset)])
    node2.solver_position = np.array([p - o * (1.0 - cross_fade) for p, o in zip(b, offset)])


class _Effector:
    """IKEffector."""

    def __init__(self, s: EffectorSettings):
        self.s = s
        self.chain_index = self.node_index = -1
        self.child_chain_indexes: list = []
        self.child_node_indexes: list = []
        self.local_positions: list = []
        self.use_plane_nodes = False
        self.is_end_effector = False
        self.plane1 = self.plane2 = self.plane3 = (-1, -1)
        self.animated_plane_rotation = IDENTITY.copy()
        self.animated_position = _ZERO.copy()
        self.first_update = False
        self.pos_w = 0.0
        self.rot_w = 0.0

    def node(self, solver: FullBodyBipedIK) -> _Node:
        """IKEffector.GetNode."""
        return solver.get_node(self.chain_index, self.node_index)

    def _child_node(self, solver: FullBodyBipedIK, i: int) -> _Node:
        return solver.get_node(self.child_chain_indexes[i], self.child_node_indexes[i])

    def initiate(self, solver: FullBodyBipedIK) -> None:
        """IKEffector.Initiate: a plane bone makes it an end effector."""
        s, tf = self.s, solver.tf
        s.position = tf.position(s.bone)
        s.rotation = tf.rotation(s.bone)
        self.animated_plane_rotation = IDENTITY.copy()
        self.chain_index, self.node_index = solver.get_chain_and_node_indexes(s.bone)
        idx = [solver.get_chain_and_node_indexes(b) for b in s.child_bones]
        self.child_chain_indexes = [c for c, _ in idx]
        self.child_node_indexes = [n for _, n in idx]
        self.local_positions = [_ZERO.copy() for _ in s.child_bones]
        self.use_plane_nodes = False
        if s.plane_bone1 is None:
            self.is_end_effector = False
            return
        self.plane1 = solver.get_chain_and_node_indexes(s.plane_bone1)
        if s.plane_bone2 is not None:
            self.plane2 = solver.get_chain_and_node_indexes(s.plane_bone2)
            if s.plane_bone3 is not None:
                self.plane3 = solver.get_chain_and_node_indexes(s.plane_bone3)
                self.use_plane_nodes = True
        self.is_end_effector = True

    def reset_offset(self, solver: FullBodyBipedIK) -> None:
        """IKEffector.ResetOffset."""
        self.node(solver).offset = _ZERO.copy()
        for i in range(len(self.child_chain_indexes)):
            self._child_node(solver, i).offset = _ZERO.copy()

    def set_to_target(self, targets: Pose) -> None:
        """IKEffector.SetToTarget: the target transform's pose, when there is a target."""
        if self.s.target is None:
            return
        if self.s.target not in targets:
            raise KeyError(f"effector target '{self.s.target}' is not in the targets")
        p, q = targets[self.s.target]
        self.s.position, self.s.rotation = _vec(p), _quat(q)

    def on_pre_solve(self, solver: FullBodyBipedIK) -> None:
        """IKEffector.OnPreSolve: clamp the weights, hand them and the offset to the node."""
        s, tf, w = self.s, solver.tf, solver.s.ik_position_weight
        s.position_weight = _clamp(s.position_weight, 0.0, 1.0)
        s.rotation_weight = _clamp(s.rotation_weight, 0.0, 1.0)
        s.maintain_relative_position_weight = _clamp(s.maintain_relative_position_weight, 0.0, 1.0)
        self.pos_w = s.position_weight * w
        self.rot_w = s.rotation_weight * w
        node = self.node(solver)
        node.effector_position_weight = self.pos_w
        node.effector_rotation_weight = self.rot_w
        node.solver_rotation = s.rotation.copy()
        node.offset = node.offset + s.position_offset * w
        if s.effect_child_nodes and solver.s.iterations > 0:
            for i, bone in enumerate(s.child_bones):
                self.local_positions[i] = tf.position(bone) - tf.position(s.bone)
                child = self._child_node(solver, i)
                child.offset = child.offset + s.position_offset * w
        if self.use_plane_nodes and s.maintain_relative_position_weight > 0.0:
            p1 = tf.position(s.plane_bone1)
            self.animated_plane_rotation = _q_look(tf.position(s.plane_bone2) - p1,
                                                   tf.position(s.plane_bone3) - p1)
        self.first_update = True

    def on_post_write(self) -> None:
        """IKEffector.OnPostWrite."""
        self.s.position_offset = _ZERO.copy()

    def get_plane_rotation(self, solver: FullBodyBipedIK) -> Quat:
        """IKEffector.GetPlaneRotation."""
        p1 = solver.get_node(*self.plane1).solver_position
        p2 = solver.get_node(*self.plane2).solver_position
        p3 = solver.get_node(*self.plane3).solver_position
        forward, up = p2 - p1, p3 - p1
        if _v_eq(forward, _ZERO):
            warnings.warn("Make sure you are not placing 2 or more FBBIK effectors of the same chain "
                          "to exactly the same position.", RuntimeWarning, stacklevel=2)
            return IDENTITY.copy()
        return _q_look(forward, up)

    def update(self, solver: FullBodyBipedIK) -> None:
        """IKEffector.Update: pull the node (and its child nodes) toward the effector."""
        node = self.node(solver)
        if self.first_update:
            self.animated_position = solver.tf.position(self.s.bone) + node.offset
            self.first_update = False
        start, self.s.plane_rotation_offset = self.get_position(solver)
        node.solver_position = _lerp(start, self.s.position, self.pos_w)
        if not self.s.effect_child_nodes:
            return
        for i in range(len(self.s.child_bones)):
            child = self._child_node(solver, i)
            child.solver_position = _lerp(child.solver_position,
                                          node.solver_position + self.local_positions[i], self.pos_w)

    def get_position(self, solver: FullBodyBipedIK) -> tuple:
        """IKEffector.GetPosition -> (position, planeRotationOffset)."""
        offset_rot = IDENTITY.copy()
        node = self.node(solver)
        if not self.is_end_effector:
            return node.solver_position.copy(), offset_rot
        w = self.s.maintain_relative_position_weight
        if not w > 0.0:
            return self.animated_position.copy(), offset_rot
        tf = solver.tf
        rel = tf.position(self.s.bone) - tf.position(self.s.plane_bone1)
        offset_rot = _q_mul(self.get_plane_rotation(solver), _q_inv(self.animated_plane_rotation))
        p = solver.get_node(*self.plane1).solver_position + _q_rot(offset_rot, rel)
        offset_rot = _q_lerp(IDENTITY, offset_rot, w)
        return _lerp(self.animated_position, p + node.offset, w), offset_rot


class _BoneMap:
    """IKMapping.BoneMap: maps solver nodes back onto one bone."""

    def __init__(self):
        self.transform = ""
        self.chain_index = self.node_index = -1
        self.plane_bone1 = self.plane_bone2 = self.plane_bone3 = ""
        self.plane1 = self.plane2 = self.plane3 = (-1, -1)
        self.length = 0.0
        self.local_swing_axis = _ZERO.copy()
        self.local_twist_axis = _ZERO.copy()
        self.plane_position = _ZERO.copy()
        self.ik_position = _ZERO.copy()
        self.default_local_target_rotation = IDENTITY.copy()
        self.maintain_rotation = IDENTITY.copy()
        self.default_local_position = _ZERO.copy()
        self.default_local_rotation = IDENTITY.copy()

    @property
    def is_node_bone(self) -> bool:
        """BoneMap.isNodeBone."""
        return self.node_index != -1

    def initiate(self, transform: str, solver: FullBodyBipedIK) -> None:
        """BoneMap.Initiate."""
        self.transform = transform
        self.chain_index, self.node_index = solver.get_chain_and_node_indexes(transform)

    def store_default_local_state(self, tf: _Transforms) -> None:
        """BoneMap.StoreDefaultLocalState."""
        self.default_local_position = tf.local_position(self.transform)
        self.default_local_rotation = tf.local_rotation(self.transform)

    def fix_transform(self, tf: _Transforms, position: bool) -> None:
        """BoneMap.FixTransform."""
        if position:
            tf.set_local_position(self.transform, self.default_local_position)
        tf.set_local_rotation(self.transform, self.default_local_rotation)

    def set_length(self, tf: _Transforms, next_bone: _BoneMap) -> None:
        """BoneMap.SetLength."""
        self.length = _mag(tf.position(self.transform) - tf.position(next_bone.transform))

    def set_local_swing_axis(self, tf: _Transforms, bone1: _BoneMap,
                             bone2: _BoneMap | None = None) -> None:
        """BoneMap.SetLocalSwingAxis (both overloads; the one-argument form swings toward ``bone1``)."""
        bone2 = self if bone2 is None else bone2
        self.local_swing_axis = _q_rot(_q_inv(tf.rotation(self.transform)),
                                       tf.position(bone1.transform) - tf.position(bone2.transform))

    def set_local_twist_axis(self, tf: _Transforms, twist_direction: Vec, normal_direction: Vec) -> None:
        """BoneMap.SetLocalTwistAxis."""
        _, twist_direction = _ortho_normalize(normal_direction, twist_direction)
        self.local_twist_axis = _q_rot(_q_inv(tf.rotation(self.transform)), twist_direction)

    def set_plane(self, solver: FullBodyBipedIK, b1: str, b2: str, b3: str) -> None:
        """BoneMap.SetPlane."""
        self.plane_bone1, self.plane_bone2, self.plane_bone3 = b1, b2, b3
        self.plane1 = solver.get_chain_and_node_indexes(b1)
        self.plane2 = solver.get_chain_and_node_indexes(b2)
        self.plane3 = solver.get_chain_and_node_indexes(b3)
        self.update_plane(solver.tf, True, True)

    def update_plane(self, tf: _Transforms, rotation: bool, position: bool) -> None:
        """BoneMap.UpdatePlane: the bone against its plane in the animated pose."""
        t = self.last_animated_target_rotation(tf)
        if rotation:
            self.default_local_target_rotation = _rotation_to_local_space(tf.rotation(self.transform), t)
        if position:
            self.plane_position = _q_rot(_q_inv(t), tf.position(self.transform)
                                         - tf.position(self.plane_bone1))

    def last_animated_target_rotation(self, tf: _Transforms) -> Quat:
        """BoneMap.lastAnimatedTargetRotation."""
        p1 = tf.position(self.plane_bone1)
        if _v_eq(p1, tf.position(self.plane_bone3)):
            return IDENTITY.copy()
        return _q_look(tf.position(self.plane_bone2) - p1, tf.position(self.plane_bone3) - p1)

    def set_ik_position(self, tf: _Transforms) -> None:
        """BoneMap.SetIKPosition."""
        self.ik_position = tf.position(self.transform)

    def read_maintain_rotation(self, tf: _Transforms) -> None:
        """BoneMap.MaintainRotation (stores the rotation to keep)."""
        self.maintain_rotation = tf.rotation(self.transform)

    def set_to_ik_position(self, tf: _Transforms) -> None:
        """BoneMap.SetToIKPosition."""
        tf.set_position(self.transform, self.ik_position)

    def get_plane_position(self, solver: FullBodyBipedIK) -> Vec:
        """BoneMap.GetPlanePosition."""
        return (solver.get_node(*self.plane1).solver_position
                + _q_rot(self.get_target_rotation(solver), self.plane_position))

    def rotate_to_plane(self, solver: FullBodyBipedIK, weight: float) -> None:
        """BoneMap.RotateToPlane."""
        r = _q_mul(self.get_target_rotation(solver), self.default_local_target_rotation)
        self._rotate(solver.tf, r, weight)

    def _rotate(self, tf: _Transforms, r: Quat, weight: float) -> None:
        # BoneMap.RotateToPlane/Swing/Twist tail: set when weight >= 1, else Quaternion.Lerp
        if not weight < 1.0:
            tf.set_rotation(self.transform, r)
        else:
            tf.set_rotation(self.transform, _q_lerp(tf.rotation(self.transform), r, weight))

    def swing(self, tf: _Transforms, swing_target: Vec, weight: float,
              pos2: Vec | None = None) -> None:
        """BoneMap.Swing (both overloads; ``pos2`` defaults to the bone's own position)."""
        pos2 = tf.position(self.transform) if pos2 is None else pos2
        rot = tf.rotation(self.transform)
        r = _q_mul(_q_from_to(_q_rot(rot, self.local_swing_axis), swing_target - pos2), rot)
        self._rotate(tf, r, weight)

    def twist(self, tf: _Transforms, twist_direction: Vec, normal_direction: Vec, weight: float) -> None:
        """BoneMap.Twist."""
        _, twist_direction = _ortho_normalize(normal_direction, twist_direction)
        rot = tf.rotation(self.transform)
        r = _q_mul(_q_from_to(_q_rot(rot, self.local_twist_axis), twist_direction), rot)
        self._rotate(tf, r, weight)

    def rotate_to_maintain(self, tf: _Transforms, weight: float) -> None:
        """BoneMap.RotateToMaintain."""
        if not weight > 0.0:
            return
        tf.set_rotation(self.transform, _q_lerp(tf.rotation(self.transform), self.maintain_rotation, weight))

    def rotate_to_effector(self, solver: FullBodyBipedIK, weight: float) -> None:
        """BoneMap.RotateToEffector."""
        if not self.is_node_bone:
            return
        node = solver.get_node(self.chain_index, self.node_index)
        w = weight * node.effector_rotation_weight
        if not w > 0.0:
            return
        if not w < 1.0:
            solver.tf.set_rotation(self.transform, node.solver_rotation)
            return
        tf = solver.tf
        tf.set_rotation(self.transform, _q_lerp(tf.rotation(self.transform), node.solver_rotation, w))

    def get_target_rotation(self, solver: FullBodyBipedIK) -> Quat:
        """BoneMap.GetTargetRotation: the plane rotation of the solved nodes."""
        p1 = solver.get_node(*self.plane1).solver_position
        p2 = solver.get_node(*self.plane2).solver_position
        p3 = solver.get_node(*self.plane3).solver_position
        if _v_eq(p1, p3):
            return IDENTITY.copy()
        return _q_look(p2 - p1, p3 - p1)


def _rotation_to_local_space(space: Quat, rotation: Quat) -> Quat:
    """QuaTools.RotationToLocalSpace."""
    return _q_inv(_q_mul(_q_inv(space), rotation))


class _SpineMapping:
    """IKMappingSpine: pelvis, spine and the last spine bone from the body and its four limb roots."""

    def __init__(self, s: SpineMappingSettings):
        self.s = s
        self.spine: list = []
        self.left_upper_arm = _BoneMap()
        self.right_upper_arm = _BoneMap()
        self.left_thigh = _BoneMap()
        self.right_thigh = _BoneMap()
        self.root_node_index = -1
        self.use_fabrik = False

    def initiate(self, solver: FullBodyBipedIK) -> None:
        """IKMappingSpine.Initiate."""
        s, tf = self.s, solver.tf
        if s.iterations <= 0:
            s.iterations = 3
        self.spine = [_BoneMap() for _ in s.spine_bones]
        self.root_node_index = -1
        for i, bone in enumerate(s.spine_bones):
            self.spine[i].initiate(bone, solver)
            if self.spine[i].is_node_bone:
                self.root_node_index = i
        self.left_upper_arm.initiate(s.left_upper_arm, solver)
        self.right_upper_arm.initiate(s.right_upper_arm, solver)
        self.left_thigh.initiate(s.left_thigh, solver)
        self.right_thigh.initiate(s.right_thigh, solver)
        for b in self.spine:
            b.set_ik_position(tf)
        root = self.spine[self.root_node_index].transform
        self.spine[0].set_plane(solver, root, self.left_thigh.transform, self.right_thigh.transform)
        self._read_spine_axes(tf)
        self.spine[-1].set_plane(solver, root, self.left_upper_arm.transform, self.right_upper_arm.transform)
        self.spine[-1].set_local_swing_axis(tf, self.left_upper_arm, self.right_upper_arm)
        self.use_fabrik = len(self.spine) > 3 or self.root_node_index != 1   # IKMappingSpine.UseFABRIK

    def _read_spine_axes(self, tf: _Transforms) -> None:
        # IKMappingSpine.Initiate/ReadPose: length, swing and twist axis of every bone but the last
        arms = tf.position(self.left_upper_arm.transform) - tf.position(self.right_upper_arm.transform)
        for i in range(len(self.spine) - 1):
            b, nxt = self.spine[i], self.spine[i + 1]
            b.set_length(tf, nxt)
            b.set_local_swing_axis(tf, nxt)
            b.set_local_twist_axis(tf, arms, tf.position(nxt.transform) - tf.position(b.transform))

    def store_default_local_state(self, tf: _Transforms) -> None:
        """IKMappingSpine.StoreDefaultLocalState."""
        for b in self.spine:
            b.store_default_local_state(tf)

    def fix_transforms(self, tf: _Transforms) -> None:
        """IKMappingSpine.FixTransforms: positions only for the first and the last bone."""
        for i, b in enumerate(self.spine):
            b.fix_transform(tf, i == 0 or i == len(self.spine) - 1)

    def read_pose(self, tf: _Transforms) -> None:
        """IKMappingSpine.ReadPose."""
        self.spine[0].update_plane(tf, True, True)
        self._read_spine_axes(tf)
        self.spine[-1].update_plane(tf, True, True)
        # the last bone's swing axis is refreshed as in the IL; nothing on this path swings it
        self.spine[-1].set_local_swing_axis(tf, self.left_upper_arm, self.right_upper_arm)

    def write_pose(self, solver: FullBodyBipedIK) -> None:
        """IKMappingSpine.WritePose."""
        tf = solver.tf
        first = self.spine[0].get_plane_position(solver)
        root = self.spine[self.root_node_index]
        root_pos = solver.get_node(root.chain_index, root.node_index).solver_position.copy()
        last = self.spine[-1].get_plane_position(solver)
        if self.use_fabrik:
            delta = root_pos - tf.position(root.transform)
            for b in self.spine:
                b.ik_position = tf.position(b.transform) + delta
            for _ in range(self.s.iterations):
                self.forward_reach(last)
                self.backward_reach(first)
                root.ik_position = root_pos.copy()
        else:
            self.spine[0].ik_position = first
            root.ik_position = root_pos.copy()
        self.spine[-1].ik_position = last
        self.map_to_solver_positions(solver)

    def forward_reach(self, position: Vec) -> None:
        """IKMappingSpine.ForwardReach."""
        self.spine[len(self.s.spine_bones) - 1].ik_position = position.copy()
        for i in range(len(self.spine) - 2, -1, -1):
            b = self.spine[i]
            b.ik_position = _solve_fabrik_joint(b.ik_position, self.spine[i + 1].ik_position, b.length)

    def backward_reach(self, position: Vec) -> None:
        """IKMappingSpine.BackwardReach."""
        self.spine[0].ik_position = position.copy()
        for i in range(1, len(self.spine)):
            prev = self.spine[i - 1]
            self.spine[i].ik_position = _solve_fabrik_joint(self.spine[i].ik_position, prev.ik_position,
                                                            prev.length)

    def map_to_solver_positions(self, solver: FullBodyBipedIK) -> None:
        """IKMappingSpine.MapToSolverPositions."""
        tf = solver.tf
        self.spine[0].set_to_ik_position(tf)
        self.spine[0].rotate_to_plane(solver, 1.0)
        for i in range(1, len(self.spine) - 1):
            b, nxt = self.spine[i], self.spine[i + 1]
            b.swing(tf, nxt.ik_position, 1.0)
            if self.s.twist_weight > 0.0:
                w = float(i) / (float(len(self.spine)) - 2.0)
                left = solver.get_node(self.left_upper_arm.chain_index,
                                       self.left_upper_arm.node_index).solver_position
                right = solver.get_node(self.right_upper_arm.chain_index,
                                        self.right_upper_arm.node_index).solver_position
                b.twist(tf, left - right, nxt.ik_position - tf.position(b.transform),
                        w * self.s.twist_weight)
        self.spine[-1].set_to_ik_position(tf)
        self.spine[-1].rotate_to_plane(solver, 1.0)


class _LimbMapping:
    """IKMappingLimb: an arm or a leg (with the clavicle, when it has one)."""

    def __init__(self, s: LimbMappingSettings):
        self.s = s
        self.bone_map_parent = _BoneMap()
        self.bone_map1 = _BoneMap()
        self.bone_map2 = _BoneMap()
        self.bone_map3 = _BoneMap()

    def initiate(self, solver: FullBodyBipedIK) -> None:
        """IKMappingLimb.Initiate: the clavicle's swing axis is read here only."""
        s, tf = self.s, solver.tf
        if s.parent_bone is not None:
            self.bone_map_parent.initiate(s.parent_bone, solver)
        self.bone_map1.initiate(s.bone1, solver)
        self.bone_map2.initiate(s.bone2, solver)
        self.bone_map3.initiate(s.bone3, solver)
        b1, b2, b3 = self.bone_map1.transform, self.bone_map2.transform, self.bone_map3.transform
        self.bone_map1.set_plane(solver, b1, b2, b3)
        self.bone_map2.set_plane(solver, b2, b3, b1)
        if s.parent_bone is not None:
            self.bone_map_parent.set_local_swing_axis(tf, self.bone_map1)

    def store_default_local_state(self, tf: _Transforms) -> None:
        """IKMappingLimb.StoreDefaultLocalState."""
        if self.s.parent_bone is not None:
            self.bone_map_parent.store_default_local_state(tf)
        self.bone_map1.store_default_local_state(tf)
        self.bone_map2.store_default_local_state(tf)
        self.bone_map3.store_default_local_state(tf)

    def fix_transforms(self, tf: _Transforms) -> None:
        """IKMappingLimb.FixTransforms."""
        if self.s.parent_bone is not None:
            self.bone_map_parent.fix_transform(tf, False)
        self.bone_map1.fix_transform(tf, True)
        self.bone_map2.fix_transform(tf, False)
        self.bone_map3.fix_transform(tf, False)

    def read_pose(self, tf: _Transforms) -> None:
        """IKMappingLimb.ReadPose."""
        self.bone_map1.update_plane(tf, True, True)
        self.bone_map2.update_plane(tf, True, False)
        self.s.weight = _clamp(self.s.weight, 0.0, 1.0)
        self.bone_map3.read_maintain_rotation(tf)

    def write_pose(self, solver: FullBodyBipedIK, full_body: bool) -> None:
        """IKMappingLimb.WritePose."""
        w = self.s.weight
        if not w > 0.0:
            return
        if full_body and self.s.parent_bone is not None:
            node = solver.get_node(self.bone_map1.chain_index, self.bone_map1.node_index)
            self.bone_map_parent.swing(solver.tf, node.solver_position, w)
        self.bone_map1.rotate_to_plane(solver, w)
        self.bone_map2.rotate_to_plane(solver, w)
        self.bone_map3.rotate_to_maintain(solver.tf, self.s.maintain_rotation_weight * w
                                          * solver.s.ik_position_weight)
        self.bone_map3.rotate_to_effector(solver, w)


class _BoneMapping:
    """IKMappingBone: a bone (the head) that keeps its animated rotation."""

    def __init__(self, s: BoneMappingSettings):
        self.s = s
        self.bone_map = _BoneMap()

    def initiate(self, solver: FullBodyBipedIK) -> None:
        """IKMappingBone.Initiate."""
        self.bone_map.initiate(self.s.bone, solver)

    def store_default_local_state(self, tf: _Transforms) -> None:
        """IKMappingBone.StoreDefaultLocalState."""
        self.bone_map.store_default_local_state(tf)

    def fix_transforms(self, tf: _Transforms) -> None:
        """IKMappingBone.FixTransforms."""
        self.bone_map.fix_transform(tf, False)

    def read_pose(self, tf: _Transforms) -> None:
        """IKMappingBone.ReadPose."""
        self.bone_map.read_maintain_rotation(tf)

    def write_pose(self, tf: _Transforms, solver_weight: float) -> None:
        """IKMappingBone.WritePose."""
        self.bone_map.rotate_to_maintain(tf, solver_weight * self.s.maintain_rotation_weight)


# --------------------------------------------------------------------------
# The solver.
# --------------------------------------------------------------------------

class FullBodyBipedIK:
    """IKSolverFullBodyBiped after ``IKSolver.Initiate``, ready for ``IKSolver.Update``.

    Constructing it is ``Initiate``: it validates the setup (``IsValid``), runs
    ``OnInitiate`` on the pose given, and stores the default local state that
    :meth:`fix_transforms` restores.  What survives between updates, as in
    FinalIK: the default local state; each bend constraint's local bend
    directions (read once) and its ``direction`` (rewritten whenever a bend goal
    is set and not on the start node); each clavicle's swing axis (read once);
    the limb nodes' bone lengths from the previous frame (``PullBody`` reads them
    before this frame's are measured); effector ``position``/``rotation`` of an
    effector without a target; and the iterations-0 body ``offset``.
    """

    def __init__(self, settings: FBBIKSettings, parents: Mapping[str, str | None], pose: Pose):
        self.s = settings
        self.tf = _Transforms(parents)
        self.chain = [_Chain(c) for c in settings.chains]
        self.effectors = [_Effector(e) for e in settings.effectors]
        self.spine_mapping = _SpineMapping(settings.spine)
        self.bone_mappings = [_BoneMapping(b) for b in settings.bone_mappings]
        self.limb_mappings = [_LimbMapping(m) for m in settings.limb_mappings]
        self.offset = _ZERO.copy()
        self._root_rotation = IDENTITY.copy()
        self._targets: Pose = {}
        self._validate()
        self.tf.load(pose)
        self._on_initiate()
        self._store_default_local_state()

    # -- lookups (IKSolverFullBody) ---------------------------------------
    def get_chain_index(self, transform: str) -> int:
        """IKSolverFullBody.GetChainIndex: the first chain holding ``transform``, or -1."""
        for i, c in enumerate(self.chain):
            if any(n.transform == transform for n in c.nodes):
                return i
        return -1

    def get_chain_and_node_indexes(self, transform: str) -> tuple:
        """IKSolverFullBody.GetChainAndNodeIndexes."""
        ci = self.get_chain_index(transform)
        if ci == -1:
            return -1, -1
        ni = next((k for k, n in enumerate(self.chain[ci].nodes) if n.transform == transform), -1)
        return ci, ni

    def get_node(self, chain_index: int, node_index: int) -> _Node:
        """IKSolverFullBody.GetNode."""
        if chain_index < 0 or node_index < 0:
            raise IndexError(f"no solver node at chain {chain_index}, node {node_index}")
        return self.chain[chain_index].nodes[node_index]

    def get_effector(self, transform: str) -> _Effector | None:
        """IKSolverFullBody.GetEffector(Transform)."""
        return next((e for e in self.effectors if e.s.bone == transform), None)

    def _is_point(self, transform: str | None) -> bool:
        # IKSolverFullBody.GetPoint(t) != null
        return transform is not None and self.get_chain_index(transform) != -1

    # -- IsValid ----------------------------------------------------------
    def _validate(self) -> None:
        """IKSolverFullBodyBiped.IsValid (with the base checks); raises with the IL's message."""
        s = self.s
        if not s.chains:
            raise ValueError("FBIK chain length is 0, can't initiate solver.")
        for c in s.chains:
            if not c.nodes:
                raise ValueError("FBIK chain contains no nodes.")
            if any(n.bone is None for n in c.nodes):
                raise ValueError("Node transform is null in FBIK chain.")
        for e in s.effectors:
            self._validate_effector(e)
        self._validate_spine(s.spine)
        for m in s.limb_mappings:
            for b in (m.bone1, m.bone2, m.bone3):
                if b is None:
                    raise ValueError("IKMappingLimb contains a null reference.")
                if not self._is_point(b):
                    raise ValueError(f"IKMappingLimb is referencing to a bone '{b}' that does not "
                                     "excist in the Node Chain.")
        if any(b.bone is None for b in s.bone_mappings):
            raise ValueError("IKMappingBone's bone is null.")
        if s.root_node is None:
            raise ValueError("Root Node bone is null. FBBIK will not initiate.")
        counts = [len(c.nodes) for c in s.chains]
        if counts != [1, 3, 3, 3, 3] or len(s.effectors) != 9 or len(s.limb_mappings) != 4:
            raise ValueError("Invalid FBBIK setup. Please right-click on the component header and "
                             "select 'Reinitiate'.")
        self._validate_bones_present()

    def _validate_effector(self, e: EffectorSettings) -> None:
        # IKEffector.IsValid
        if e.bone is None:
            raise ValueError("IK Effector bone is null.")
        if any(b is None for b in e.child_bones):
            raise ValueError("IK Effector contains a null reference.")
        for b in [e.bone, *e.child_bones, e.plane_bone1, e.plane_bone2, e.plane_bone3]:
            if b is not None and not self._is_point(b):
                raise ValueError(f"IK Effector is referencing to a bone '{b}' that does not excist "
                                 "in the Node Chain.")

    def _validate_spine(self, m: SpineMappingSettings) -> None:
        # IKMappingSpine.IsValid
        if any(b is None for b in m.spine_bones):
            raise ValueError("Spine bones contains a null reference.")
        if not any(self._is_point(b) for b in m.spine_bones):
            raise ValueError("IKMappingSpine does not contain any nodes.")
        for b, side in ((m.left_upper_arm, "left upper arm"), (m.right_upper_arm, "right upper arm"),
                        (m.left_thigh, "left thigh"), (m.right_thigh, "right thigh")):
            if b is None:
                raise ValueError(f"IKMappingSpine is missing the {side} bone.")
            if not self._is_point(b):
                raise ValueError(f"Full Body IK is missing the {side} node.")

    def _validate_bones_present(self) -> None:
        # Unity would hold live Transform references; here every name has to be a bone we move
        s = self.s
        names = {n.bone for c in s.chains for n in c.nodes}
        names |= {b for e in s.effectors for b in (e.bone, *e.child_bones, e.plane_bone1,
                                                    e.plane_bone2, e.plane_bone3) if b}
        names |= set(s.spine.spine_bones) | {b.bone for b in s.bone_mappings}
        names |= {b for m in s.limb_mappings for b in (m.bone1, m.bone2, m.bone3, m.parent_bone) if b}
        missing = sorted(n for n in names if n not in self.tf.parent)
        if missing:
            raise ValueError(f"solver bones missing from the bone set: {missing}")

    # -- Initiate ---------------------------------------------------------
    def _on_initiate(self) -> None:
        """IKSolverFullBody.OnInitiate."""
        for c in self.chain:
            c.initiate(self)
        for e in self.effectors:
            e.initiate(self)
        self.spine_mapping.initiate(self)
        for b in self.bone_mappings:
            b.initiate(self)
        for m in self.limb_mappings:
            m.initiate(self)

    def _store_default_local_state(self) -> None:
        """IKSolverFullBody.StoreDefaultLocalState."""
        self.spine_mapping.store_default_local_state(self.tf)
        for m in self.limb_mappings:
            m.store_default_local_state(self.tf)
        for b in self.bone_mappings:
            b.store_default_local_state(self.tf)

    def store_default_local_state(self, pose: Pose) -> None:
        """``StoreDefaultLocalState`` again, read from *pose*: what :meth:`fix_transforms` restores from now on.

        For a solver initiated on another shape of the same pose (a caller
        that reads the bend directions on one rig and runs on another).
        """
        self.tf.load(pose)
        self._store_default_local_state()

    # -- per frame --------------------------------------------------------
    def fix_transforms(self, pose: Pose) -> dict:
        """IKSolverFullBody.FixTransforms: the pose with the mapped bones' default local state back.

        The component runs this in ``Update``, before the Animator writes the
        frame's animation, when its ``fixTransforms`` is set.
        """
        self.tf.load(pose)
        if self.s.ik_position_weight > 0.0:
            self.spine_mapping.fix_transforms(self.tf)
            for m in self.limb_mappings:
                m.fix_transforms(self.tf)
            for b in self.bone_mappings:
                b.fix_transforms(self.tf)
        return self.tf.snapshot()

    def default_local_state(self) -> list:
        """``(bone, local position or None, local rotation)`` :meth:`fix_transforms` writes, in its order.

        A position only where ``FixTransform`` is handed ``true`` (the spine's
        ends, each limb's first bone); local values as :class:`_Transforms`
        keeps them.  Empty while ``IKPositionWeight`` is not above 0, as
        ``IKSolverFullBody.FixTransforms`` then writes nothing.  A caller that
        tracks which channels its animation writes reads this instead of
        running :meth:`fix_transforms` on a whole pose.
        """
        if not self.s.ik_position_weight > 0.0:
            return []
        spine = self.spine_mapping.spine
        maps = [(b, i == 0 or i == len(spine) - 1) for i, b in enumerate(spine)]
        for m in self.limb_mappings:
            if m.s.parent_bone is not None:
                maps.append((m.bone_map_parent, False))
            maps += [(m.bone_map1, True), (m.bone_map2, False), (m.bone_map3, False)]
        maps += [(b.bone_map, False) for b in self.bone_mappings]
        return [(b.transform, b.default_local_position.copy() if pos else None, b.default_local_rotation.copy())
                for b, pos in maps]

    def update(self, pose: Pose, targets: Pose, root_rotation: Quat) -> dict:
        """IKSolver.Update on the animated ``pose``; returns the solved world pose of every bone.

        ``targets`` holds the world pose of each effector target and bend goal
        transform named in the settings; ``root_rotation`` is the rotation of
        the transform the solver was initiated with (the component's own
        transform), whose up axis ``PullBody`` uses.
        """
        self.tf.load(pose)
        self._targets = targets
        self._root_rotation = _quat(root_rotation)
        for c in self.chain:
            goal = c.s.bend.bend_goal
            if goal is not None and goal not in targets:
                raise KeyError(f"bend goal '{goal}' is not in the targets")
            c.bend.goal_position = None if goal is None else _vec(targets[goal][0])
        self._on_update()
        return self.tf.snapshot()

    def _on_update(self) -> None:
        """IKSolverFullBody.OnUpdate."""
        if not self.s.ik_position_weight > 0.0:
            for e in self.effectors:
                e.s.position_offset = _ZERO.copy()
            return
        if not self.chain:
            return
        self.s.ik_position_weight = _clamp(self.s.ik_position_weight, 0.0, 1.0)
        self._read_pose()
        self._solve()
        self._write_pose()
        for e in self.effectors:
            e.on_post_write()

    def _read_pose(self) -> None:
        """IKSolverFullBodyBiped.ReadPose."""
        for e in self.effectors:
            e.set_to_target(self._targets)
        self._pull_body()
        stiffness = _clamp(1.0 - self.s.spine_stiffness, 0.0, 1.0)
        self.chain[0].child_constraints[0].s.push_elasticity = stiffness
        self.chain[0].child_constraints[1].s.push_elasticity = stiffness
        self._read_pose_full_body()

    def _read_pose_full_body(self) -> None:
        """IKSolverFullBody.ReadPose."""
        w = self.s.ik_position_weight
        for c in self.chain:
            if c.bend.initiated:
                end = self.get_effector(c.nodes[2].transform)
                if end is None:
                    raise ValueError(f"no effector on '{c.nodes[2].transform}' for its bend limit")
                c.bend.limit_bend(self, w, end.s.position_weight)
        for e in self.effectors:
            e.reset_offset(self)
        for e in self.effectors:
            e.on_pre_solve(self)
        for c in self.chain:
            c.read_pose(self, self.s.iterations > 0)
        if self.s.iterations > 0:
            self.spine_mapping.read_pose(self.tf)
            for b in self.bone_mappings:
                b.read_pose(self.tf)
        for m in self.limb_mappings:
            m.read_pose(self.tf)

    def _pull_body(self) -> None:
        """IKSolverFullBodyBiped.PullBody: the hands drag the body along the root's up axis."""
        s = self.s
        if s.iterations < 1:
            return
        if s.pull_body_vertical == 0.0 and s.pull_body_horizontal == 0.0:
            return
        offset = self._get_body_offset()
        up = _q_rot(self._root_rotation, _UP)
        body = self.effectors[0].s
        body.position_offset = body.position_offset + (
            _extract_vertical(offset, up, s.pull_body_vertical)
            + _extract_horizontal(offset, up, s.pull_body_horizontal))

    def _get_body_offset(self) -> Vec:
        """IKSolverFullBodyBiped.GetBodyOffset."""
        left, right = self.effectors[5], self.effectors[6]
        v = _ZERO + self._get_hand_body_pull(left, self.chain[1], _ZERO) * _clamp(
            left.s.position_weight, 0.0, 1.0)
        return v + self._get_hand_body_pull(right, self.chain[2], v) * _clamp(
            right.s.position_weight, 0.0, 1.0)

    def _get_hand_body_pull(self, effector: _Effector, arm: _Chain, offset: Vec) -> Vec:
        """IKSolverFullBodyBiped.GetHandBodyPull: how far the hand target is past the arm's reach."""
        d = effector.s.position - (self.tf.position(arm.nodes[0].transform) + offset)
        arm_length = arm.nodes[0].length + arm.nodes[1].length
        mag = _mag(d)
        if mag < arm_length:
            return _ZERO.copy()
        return _vdiv(d, mag) * (mag - arm_length)

    def _solve(self) -> None:
        """IKSolverFullBody.Solve."""
        for _ in range(self.s.iterations):
            self._update_effectors(end=True)
            self.chain[0].push(self)
            self.chain[0].reach(self)
            self._update_effectors(end=False)
            self.chain[0].solve_trigonometric(self, False)
            self.chain[0].stage1(self)
            self._update_effectors(end=False)
            self.chain[0].stage2(self, self.chain[0].nodes[0].solver_position.copy())
        self._update_effectors(end=True)
        self._apply_bend_constraints()

    def _update_effectors(self, end: bool) -> None:
        # IKSolverFullBody.Solve: the end effectors' or the other effectors' Update, in order
        for e in self.effectors:
            if e.is_end_effector == end:
                e.update(self)

    def _apply_bend_constraints(self) -> None:
        """IKSolverFullBodyBiped.ApplyBendConstraints, then IKSolverFullBody's."""
        if self.s.iterations > 0:
            for k in range(1, 5):
                self.chain[k].s.bend.rotation_offset = self.effectors[k + 4].s.plane_rotation_offset.copy()
        else:
            body = self.effectors[0]
            moved = body.s.position - (self.tf.position(body.s.bone) + body.s.position_offset)
            self.offset = _lerp(body.s.position_offset, moved, body.s.position_weight)
            for e in self.effectors[:5]:
                node = e.node(self)
                node.solver_position = node.solver_position + self.offset
        self.chain[0].solve_trigonometric(self, True)

    def _write_pose(self) -> None:
        """IKSolverFullBodyBiped.WritePose, then IKSolverFullBody.WritePose."""
        if self.s.iterations == 0:
            pelvis = self.s.spine.spine_bones[0]
            self.tf.set_position(pelvis, self.tf.position(pelvis) + self.offset)
        if not self.s.ik_position_weight > 0.0:
            return
        if self.s.iterations > 0:
            self.spine_mapping.write_pose(self)
            for b in self.bone_mappings:
                b.write_pose(self.tf, self.s.ik_position_weight)
        for m in self.limb_mappings:
            m.write_pose(self, self.s.iterations > 0)


__all__ = [
    "IDENTITY",
    "BendSettings",
    "BoneMappingSettings",
    "ChainSettings",
    "ChildConstraintSettings",
    "EffectorSettings",
    "FBBIKSettings",
    "FullBodyBipedIK",
    "LimbMappingSettings",
    "NodeSettings",
    "SpineMappingSettings",
    "settings_from_serialized",
]

