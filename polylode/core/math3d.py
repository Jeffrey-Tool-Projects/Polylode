"""Minimal 3D math built on numpy.

Convention: matrices are 4x4 numpy arrays in *row-major storage* using the
standard math convention ``M @ v`` where ``v`` is a column vector.  When
uploading to OpenGL pass ``transpose=GL_TRUE`` (the GPU wrapper does this).

Quaternions are stored as ``(x, y, z, w)``.
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-9
DEG = math.pi / 180.0
RAD = 180.0 / math.pi


# --------------------------------------------------------------------------
# vectors
# --------------------------------------------------------------------------


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS)


def length(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def cross(a, b) -> np.ndarray:
    """Cross product; two 3-vectors skip ``np.cross``, same products in the same order.

    ``np.cross`` rebuilds its operands through ``moveaxis`` on every call, which
    is most of its cost on one vector (the eye and neck calcs make ~50 a frame
    per character).  Stacks of vectors still go to ``np.cross``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        return np.cross(a, b)
    a0, a1, a2 = a.tolist()
    b0, b1, b2 = b.tolist()
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def dot(a, b) -> float:
    return float(np.dot(a, b))


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def remap(x, a, b, c, d):
    """Map x from range [a,b] into [c,d]."""
    if abs(b - a) < EPS:
        return c
    return c + (d - c) * (x - a) / (b - a)


# --------------------------------------------------------------------------
# matrices
# --------------------------------------------------------------------------
def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def translate(x, y=None, z=None) -> np.ndarray:
    if y is None:
        x, y, z = x
    m = np.eye(4, dtype=np.float64)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def scale(x, y=None, z=None) -> np.ndarray:
    if y is None:
        if np.isscalar(x):
            x = y = z = float(x)
        else:
            x, y, z = x
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[1, 1], m[2, 2] = x, y, z
    return m


def rotate_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float64)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rotate_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rotate_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float64)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def euler_yxz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Unity-style ZXY order: matrix = Ry @ Rx @ Rz."""
    return rotate_y(ry) @ rotate_x(rx) @ rotate_z(rz)


def compose(t, q, s) -> np.ndarray:
    """Build a TRS matrix from translation, quaternion and (vec3|float) scale."""
    m = quat_to_mat4(q)
    if np.isscalar(s):
        s = (s, s, s)
    m[:3, 0] *= s[0]
    m[:3, 1] *= s[1]
    m[:3, 2] *= s[2]
    m[0, 3], m[1, 3], m[2, 3] = t[0], t[1], t[2]
    return m


def decompose(m: np.ndarray):
    """Split a TRS matrix into (translation, quaternion, scale)."""
    t = m[:3, 3].copy()
    sx = np.linalg.norm(m[:3, 0])
    sy = np.linalg.norm(m[:3, 1])
    sz = np.linalg.norm(m[:3, 2])
    if np.linalg.det(m[:3, :3]) < 0:
        sx = -sx
    r = np.eye(4)
    r[:3, 0] = m[:3, 0] / max(sx, EPS)
    r[:3, 1] = m[:3, 1] / max(sy, EPS)
    r[:3, 2] = m[:3, 2] / max(sz, EPS)
    return t, quat_from_mat4(r), np.array([sx, sy, sz])


def inverse(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(m)


def transform_point(m: np.ndarray, p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return m[:3, :3] @ p + m[:3, 3]


def transform_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts: (N,3) -> (N,3)."""
    return pts @ m[:3, :3].T + m[:3, 3]


# --------------------------------------------------------------------------
# projection / view
# --------------------------------------------------------------------------
def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fovy_deg * DEG * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / max(aspect, EPS)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def ortho(left, right, bottom, top, near, far) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    f = normalize(target - eye)
    upn = np.asarray(up, dtype=np.float64)
    if abs(dot(f, normalize(upn))) > 0.9995:
        upn = np.array([0.0, 0.0, 1.0])
    s = normalize(cross(f, upn))
    u = cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3] = -dot(s, eye)
    m[1, 3] = -dot(u, eye)
    m[2, 3] = dot(f, eye)
    return m


# --------------------------------------------------------------------------
# quaternions  (x, y, z, w)
# --------------------------------------------------------------------------
def quat_identity() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0])


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=np.float64))
    h = angle * 0.5
    s = math.sin(h)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h)])


def quat_from_euler(rx: float, ry: float, rz: float, order: str = "YXZ") -> np.ndarray:
    qx = quat_from_axis_angle((1, 0, 0), rx)
    qy = quat_from_axis_angle((0, 1, 0), ry)
    qz = quat_from_axis_angle((0, 0, 1), rz)
    table = {"XYZ": (qx, qy, qz), "XZY": (qx, qz, qy), "YXZ": (qy, qx, qz),
             "YZX": (qy, qz, qx), "ZXY": (qz, qx, qy), "ZYX": (qz, qy, qx)}
    a, b, c = table[order.upper()]
    return quat_mul(quat_mul(a, b), c)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > EPS else quat_identity()


def quat_rotate(q: np.ndarray, v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    u = q[:3]
    s = q[3]
    return 2.0 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2.0 * s * cross(u, v)


def quat_to_mat4(q: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_normalize(np.asarray(q, dtype=np.float64))
    m = np.eye(4, dtype=np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m[0, 0] = 1 - 2 * (yy + zz); m[0, 1] = 2 * (xy - wz);     m[0, 2] = 2 * (xz + wy)
    m[1, 0] = 2 * (xy + wz);     m[1, 1] = 1 - 2 * (xx + zz); m[1, 2] = 2 * (yz - wx)
    m[2, 0] = 2 * (xz - wy);     m[2, 1] = 2 * (yz + wx);     m[2, 2] = 1 - 2 * (xx + yy)
    return m


def quat_from_mat4(m: np.ndarray) -> np.ndarray:
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([x, y, z, w]))


def quats_from_matrices(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(quaternions (K, 4), scales (K, 3))`` of K matrices' upper 3×3: :func:`decompose` for many.

    The same branches as :func:`quat_from_mat4`, taken per matrix; a mirrored
    matrix has its first scale negative, as there.
    """
    upper = np.asarray(matrices, dtype=np.float64)[:, :3, :3]
    scale = np.linalg.norm(upper, axis=1)
    scale[:, 0] = np.where(np.linalg.det(upper) < 0, -scale[:, 0], scale[:, 0])
    r = upper / np.maximum(scale, EPS)[:, None, :]
    m00, m11, m22 = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    trace = m00 + m11 + m22
    case = np.select([trace > 0.0, (m00 > m11) & (m00 > m22), m11 > m22], [0, 1, 2], default=3)
    under = np.select([case == 0, case == 1, case == 2],
                      [trace + 1.0, 1.0 + m00 - m11 - m22, 1.0 + m11 - m00 - m22], default=1.0 + m22 - m00 - m11)
    span = np.sqrt(np.maximum(under, 0.0)) * 2.0
    span = np.where(span > 0.0, span, 1.0)
    quarter = 0.25 * span
    a, b, c = (r[:, 2, 1] - r[:, 1, 2]) / span, (r[:, 0, 2] - r[:, 2, 0]) / span, (r[:, 1, 0] - r[:, 0, 1]) / span
    d, e, f = (r[:, 0, 1] + r[:, 1, 0]) / span, (r[:, 0, 2] + r[:, 2, 0]) / span, (r[:, 1, 2] + r[:, 2, 1]) / span
    first = [case == 0, case == 1, case == 2]
    quat = np.stack([np.select(first, [a, quarter, d], default=e), np.select(first, [b, d, quarter], default=f),
                     np.select(first, [c, e, f], default=quarter), np.select(first, [quarter, a, b], default=c)], axis=1)
    length = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = np.where(length > EPS, quat / np.where(length > EPS, length, 1.0), np.array([0.0, 0.0, 0.0, 1.0]))
    return quat, scale


def quat_to_euler(q: np.ndarray, order: str = "YXZ") -> np.ndarray:
    """Extract euler angles (radians) in the given order from a quaternion."""
    m = quat_to_mat4(q)
    o = order.upper()
    if o == "YXZ":
        sx = -m[1, 2]
        if abs(sx) < 0.99999:
            rx = math.asin(np.clip(sx, -1, 1))
            ry = math.atan2(m[0, 2], m[2, 2])
            rz = math.atan2(m[1, 0], m[1, 1])
        else:
            rx = math.asin(np.clip(sx, -1, 1))
            ry = math.atan2(-m[2, 0], m[0, 0])
            rz = 0.0
        return np.array([rx, ry, rz])
    # generic XYZ fallback
    sy = np.clip(m[0, 2], -1, 1)
    ry = math.asin(sy)
    if abs(sy) < 0.99999:
        rx = math.atan2(-m[1, 2], m[2, 2])
        rz = math.atan2(-m[0, 1], m[0, 0])
    else:
        rx = math.atan2(m[2, 1], m[1, 1])
        rz = 0.0
    return np.array([rx, ry, rz])


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = quat_normalize(np.asarray(a, dtype=np.float64))
    b = quat_normalize(np.asarray(b, dtype=np.float64))
    d = float(np.dot(a, b))
    if d < 0.0:
        b = -b
        d = -d
    if d > 0.9995:
        return quat_normalize(a + (b - a) * t)
    theta0 = math.acos(np.clip(d, -1.0, 1.0))
    theta = theta0 * t
    s0 = math.sin(theta0 - theta) / math.sin(theta0)
    s1 = math.sin(theta) / math.sin(theta0)
    return quat_normalize(a * s0 + b * s1)


def quat_from_to(a, b) -> np.ndarray:
    """Shortest-arc rotation taking direction a onto direction b."""
    a = normalize(np.asarray(a, dtype=np.float64))
    b = normalize(np.asarray(b, dtype=np.float64))
    d = float(np.dot(a, b))
    if d > 0.999999:
        return quat_identity()
    if d < -0.999999:
        axis = cross(np.array([1.0, 0.0, 0.0]), a)
        if np.linalg.norm(axis) < 1e-6:
            axis = cross(np.array([0.0, 1.0, 0.0]), a)
        return quat_from_axis_angle(axis, math.pi)
    axis = cross(a, b)
    return quat_normalize(np.array([axis[0], axis[1], axis[2], 1.0 + d]))


def quat_look_rotation(forward, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    f = normalize(np.asarray(forward, dtype=np.float64))
    u = np.asarray(up, dtype=np.float64)
    if abs(dot(f, normalize(u))) > 0.9995:
        u = np.array([0.0, 0.0, 1.0])
    r = normalize(cross(u, f))
    u2 = cross(f, r)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = r, u2, f
    return quat_from_mat4(m)


class Transform:
    """Mutable TRS transform."""

    __slots__ = ("translation", "rotation", "scale")

    def __init__(self, translation=None, rotation=None, scale_=None):
        self.translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64).copy()
        self.rotation = quat_identity() if rotation is None else np.asarray(rotation, dtype=np.float64).copy()
        self.scale = np.ones(3) if scale_ is None else (
            np.full(3, float(scale_)) if np.isscalar(scale_) else np.asarray(scale_, dtype=np.float64).copy())

    def matrix(self) -> np.ndarray:
        return compose(self.translation, self.rotation, self.scale)

    def copy(self) -> "Transform":
        return Transform(self.translation, self.rotation, self.scale)

    def __repr__(self):
        return f"Transform(t={self.translation}, r={self.rotation}, s={self.scale})"
