"""Skeletal animation: keyframe curves, clips, pose blending and layers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import math3d as m3

# A Pose uses rot/pos/scale offsets; an optional absolute local "matrix" keeps an imported affine transform.
Pose = dict


@dataclass
class BoneCurve:
    """Keyframed local TRS offsets for one bone."""
    bone: str
    times: np.ndarray
    rots: np.ndarray | None = None       # (N,4) quaternions
    poss: np.ndarray | None = None       # (N,3)
    scales: np.ndarray | None = None     # (N,3)

    def sample(self, t: float) -> dict:
        times = self.times
        n = len(times)
        if n == 0:
            return {}
        if n == 1:
            i0 = i1 = 0
            f = 0.0
        else:
            i1 = int(np.searchsorted(times, t, side="right"))
            i1 = min(max(i1, 1), n - 1)
            i0 = i1 - 1
            span = times[i1] - times[i0]
            f = 0.0 if span <= 1e-9 else float(np.clip((t - times[i0]) / span, 0.0, 1.0))
        out = {}
        if self.rots is not None:
            out["rot"] = m3.quat_slerp(self.rots[i0], self.rots[i1], f)
        if self.poss is not None:
            out["pos"] = self.poss[i0] * (1 - f) + self.poss[i1] * f
        if self.scales is not None:
            out["scale"] = self.scales[i0] * (1 - f) + self.scales[i1] * f
        return out

    def to_dict(self) -> dict:
        d = {"bone": self.bone, "times": np.round(self.times, 5).tolist()}
        for k in ("rots", "poss", "scales"):
            v = getattr(self, k)
            if v is not None:
                d[k] = np.round(v, 6).tolist()
        return d

    @staticmethod
    def from_dict(d: dict) -> BoneCurve:
        get = lambda k: (np.asarray(d[k], dtype=np.float64) if k in d else None)
        return BoneCurve(d["bone"], np.asarray(d["times"], dtype=np.float64),
                         get("rots"), get("poss"), get("scales"))


def _pack_curves(curves: dict):
    """``(times, names, rot, pos, scale)`` stacked, or ``None`` when grids differ.

    Each of the three is ``(indices into names, (B, N, k) array)`` over the
    curves that carry that channel.
    """
    items = list(curves.items())
    if not items:
        return None
    times = items[0][1].times
    if len(times) == 0:
        return None
    for _, c in items[1:]:
        if c.times is not times and not np.array_equal(c.times, times):
            return None
    names = [b for b, _ in items]

    def stack(attr):
        idx = [k for k, (_, c) in enumerate(items) if getattr(c, attr) is not None]
        if not idx:
            return None
        return idx, np.stack([getattr(items[k][1], attr) for k in idx])
    return times, names, stack("rots"), stack("poss"), stack("scales")


def _slerp_rows(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """:func:`math3d.quat_slerp` over rows — the same branches, per row.

    Normalise both (a zero row becomes the identity), take the short way round,
    nlerp above ``d > 0.9995``, else the sin-weighted form.  Agrees with the
    scalar version to rounding (the row sums are not the same BLAS call).
    """
    def norm(q):
        n = np.sqrt(np.einsum("ij,ij->i", q, q))
        out = np.empty_like(q)
        ok = n > m3.EPS
        out[ok] = q[ok] / n[ok, None]
        out[~ok] = (0.0, 0.0, 0.0, 1.0)
        return out
    a = norm(a)
    b = norm(b)
    d = np.einsum("ij,ij->i", a, b)
    neg = d < 0.0
    b = np.where(neg[:, None], -b, b)
    d = np.where(neg, -d, d)
    out = np.empty_like(a)
    near = d > 0.9995
    if near.any():
        out[near] = norm(a[near] + (b[near] - a[near]) * t)
    far = ~near
    if far.any():
        theta0 = np.arccos(np.clip(d[far], -1.0, 1.0))
        theta = theta0 * t
        s0 = np.sin(theta0 - theta) / np.sin(theta0)
        s1 = np.sin(theta) / np.sin(theta0)
        out[far] = norm(a[far] * s0[:, None] + b[far] * s1[:, None])
    return out


def _sample_packed(packed, t: float) -> Pose:
    """:meth:`BoneCurve.sample` for every curve at once (see ``_packed``)."""
    times, names, rot, pos, scl = packed
    n = len(times)
    if n == 1:
        i0 = i1 = 0
        f = 0.0
    else:
        i1 = int(np.searchsorted(times, t, side="right"))
        i1 = min(max(i1, 1), n - 1)
        i0 = i1 - 1
        span = times[i1] - times[i0]
        f = 0.0 if span <= 1e-9 else float(np.clip((t - times[i0]) / span, 0.0, 1.0))
    out: Pose = {b: {} for b in names}
    if rot is not None:
        idx, arr = rot
        r = _slerp_rows(arr[:, i0], arr[:, i1], f)
        for k, row in zip(idx, r):
            out[names[k]]["rot"] = row
    for key, ch in (("pos", pos), ("scale", scl)):
        if ch is not None:
            idx, arr = ch
            v = arr[:, i0] * (1 - f) + arr[:, i1] * f
            for k, row in zip(idx, v):
                out[names[k]][key] = row
    return out


class AnimationClip:
    def __init__(self, name: str, duration: float = 1.0, loop: bool = True):
        self.name = name
        self.duration = max(float(duration), 1e-4)
        self.loop = loop
        self.curves: dict[str, BoneCurve] = {}
        self.events: list[tuple[float, str]] = []
        self.root_motion: np.ndarray | None = None   # (N,3) per keyframe, optional
        self.tags: set[str] = set()

    # -- authoring ---------------------------------------------------------
    def set_curve(self, bone: str, times, rots=None, poss=None, scales=None) -> BoneCurve:
        c = BoneCurve(bone,
                      np.asarray(times, dtype=np.float64),
                      None if rots is None else np.asarray(rots, dtype=np.float64),
                      None if poss is None else np.asarray(poss, dtype=np.float64),
                      None if scales is None else np.asarray(scales, dtype=np.float64))
        self.curves[bone] = c
        self.duration = max(self.duration, float(c.times[-1]) if len(c.times) else 0.0)
        return c

    def set_euler_curve(self, bone: str, times, eulers_deg, order: str = "YXZ") -> BoneCurve:
        q = np.array([m3.quat_from_euler(*np.deg2rad(e), order=order) for e in eulers_deg])
        return self.set_curve(bone, times, rots=q)

    def add_event(self, time: float, name: str) -> None:
        self.events.append((float(time), name))
        self.events.sort()

    def bones(self) -> list[str]:
        return list(self.curves)

    # -- evaluation --------------------------------------------------------
    def wrap(self, t: float) -> float:
        if self.loop:
            return float(t % self.duration)
        return float(np.clip(t, 0.0, self.duration))

    def sample(self, t: float) -> Pose:
        tt = self.wrap(t)
        packed = self._packed()
        if packed is None:
            return {b: c.sample(tt) for b, c in self.curves.items()}
        return _sample_packed(packed, tt)

    def _packed(self):
        """The curves stacked into arrays when they share one time grid.

        Every baked H clip does (one grid per clip), and sampling 161 bones one
        ``BoneCurve.sample`` at a time was most of a frame's clip cost — Python
        per bone, not arithmetic (§7.417).  Keyed on the curves' identities, so
        a clip whose ``curves`` dict is edited rebuilds; ``None`` (per-bone
        path) when the grids differ.
        """
        sig = tuple(id(c) for c in self.curves.values())
        cached = getattr(self, "_pack", None)
        if cached is not None and cached[0] == sig:
            return cached[1]
        packed = _pack_curves(self.curves)
        self._pack = (sig, packed)
        return packed

    def events_between(self, t0: float, t1: float) -> list[str]:
        if not self.events:
            return []
        if self.loop and t1 < t0:
            return ([n for tm, n in self.events if tm >= t0]
                    + [n for tm, n in self.events if tm <= t1])
        return [n for tm, n in self.events if t0 < tm <= t1]

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"name": self.name, "duration": self.duration, "loop": self.loop,
                "curves": [c.to_dict() for c in self.curves.values()],
                "events": self.events, "tags": sorted(self.tags)}

    @staticmethod
    def from_dict(d: dict) -> AnimationClip:
        c = AnimationClip(d.get("name", "clip"), d.get("duration", 1.0), d.get("loop", True))
        for cd in d.get("curves", []):
            bc = BoneCurve.from_dict(cd)
            c.curves[bc.bone] = bc
        c.events = [tuple(e) for e in d.get("events", [])]
        c.tags = set(d.get("tags", []))
        return c

    def __repr__(self):
        return f"<Clip {self.name} {self.duration:.2f}s bones={len(self.curves)}>"


# --------------------------------------------------------------------------
# pose algebra
# --------------------------------------------------------------------------


def copy_pose(p: Pose) -> Pose:
    return {b: dict(v) for b, v in p.items()}


def _entry_blend(a: dict | None, b: dict | None, t: float) -> dict:
    a = a or {}
    b = b or {}
    if "matrix" in a or "matrix" in b:
        # an absolute local (a recorded scene, ``anim.recording``): two of them blend straight; against
        # channels there is nothing to interpolate between, so the heavier side stands.  Dropped, the
        # recording played as no pose at all (D3)
        if "matrix" in a and "matrix" in b:
            ma, mb = np.asarray(a["matrix"], dtype=np.float64), np.asarray(b["matrix"], dtype=np.float64)
            return {"matrix": ma * (1.0 - t) + mb * t}
        return dict(b if t >= 0.5 else a)
    out = {}
    ra, rb = a.get("rot"), b.get("rot")
    if ra is not None or rb is not None:
        ra = m3.quat_identity() if ra is None else ra
        rb = m3.quat_identity() if rb is None else rb
        out["rot"] = m3.quat_slerp(ra, rb, t)
    pa, pb = a.get("pos"), b.get("pos")
    if pa is not None or pb is not None:
        pa = np.zeros(3) if pa is None else np.asarray(pa, dtype=np.float64)
        pb = np.zeros(3) if pb is None else np.asarray(pb, dtype=np.float64)
        out["pos"] = pa * (1 - t) + pb * t
    sa, sb = a.get("scale"), b.get("scale")
    if sa is not None or sb is not None:
        sa = np.ones(3) if sa is None else np.asarray(np.broadcast_to(sa, (3,)), dtype=np.float64)
        sb = np.ones(3) if sb is None else np.asarray(np.broadcast_to(sb, (3,)), dtype=np.float64)
        out["scale"] = sa * (1 - t) + sb * t
    return out


def blend_poses(a: Pose, b: Pose, t: float, mask: set | None = None) -> Pose:
    """Linear/slerp blend; ``mask`` limits which bones take b."""
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.0:
        return copy_pose(a)
    out = copy_pose(a)
    for bone in set(a) | set(b):
        w = t if (mask is None or bone in mask) else 0.0
        if w <= 0.0:
            continue
        out[bone] = _entry_blend(a.get(bone), b.get(bone), w)
    return out


def add_poses(base: Pose, additive: Pose, weight: float = 1.0,
              mask: set | None = None) -> Pose:
    """Apply ``additive`` on top of ``base`` (rotations compose, offsets add)."""
    if weight <= 0.0:
        return copy_pose(base)
    out = copy_pose(base)
    for bone, entry in additive.items():
        if mask is not None and bone not in mask:
            continue
        cur = dict(out.get(bone, {}))
        r = entry.get("rot")
        if r is not None:
            dq = m3.quat_slerp(m3.quat_identity(), r, weight)
            cur["rot"] = m3.quat_mul(cur.get("rot", m3.quat_identity()), dq)
        p = entry.get("pos")
        if p is not None:
            cur["pos"] = np.asarray(cur.get("pos", np.zeros(3)), dtype=np.float64) \
                + np.asarray(p, dtype=np.float64) * weight
        s = entry.get("scale")
        if s is not None:
            base_s = np.asarray(np.broadcast_to(cur.get("scale", 1.0), (3,)), dtype=np.float64)
            cur["scale"] = base_s * (1.0 + (np.asarray(np.broadcast_to(s, (3,)),
                                                       dtype=np.float64) - 1.0) * weight)
        out[bone] = cur
    return out


def scale_pose(p: Pose, weight: float) -> Pose:
    """Scale a pose toward identity — used to build additive layers."""
    out = {}
    for bone, e in p.items():
        n = {}
        if "rot" in e:
            n["rot"] = m3.quat_slerp(m3.quat_identity(), e["rot"], weight)
        if "pos" in e:
            n["pos"] = np.asarray(e["pos"], dtype=np.float64) * weight
        if "scale" in e:
            s = np.asarray(np.broadcast_to(e["scale"], (3,)), dtype=np.float64)
            n["scale"] = 1.0 + (s - 1.0) * weight
        out[bone] = n
    return out


def difference_pose(target: Pose, reference: Pose) -> Pose:
    """target relative to reference — makes an additive clip from two poses."""
    out = {}
    for bone in set(target) | set(reference):
        a = reference.get(bone, {})
        b = target.get(bone, {})
        e = {}
        ra = a.get("rot", m3.quat_identity())
        rb = b.get("rot", m3.quat_identity())
        e["rot"] = m3.quat_mul(m3.quat_conj(ra), rb)
        pa = np.asarray(a.get("pos", np.zeros(3)), dtype=np.float64)
        pb = np.asarray(b.get("pos", np.zeros(3)), dtype=np.float64)
        if np.any(np.abs(pb - pa) > 1e-9):
            e["pos"] = pb - pa
        out[bone] = e
    return out


# --------------------------------------------------------------------------
# layered playback
# --------------------------------------------------------------------------
@dataclass
class AnimationLayer:
    clip: AnimationClip | None = None
    weight: float = 1.0
    speed: float = 1.0
    time: float = 0.0
    additive: bool = False
    mask: set | None = None
    name: str = "layer"
    playing: bool = True

    def advance(self, dt: float) -> list[str]:
        if self.clip is None or not self.playing:
            return []
        t0 = self.clip.wrap(self.time)
        self.time += dt * self.speed
        t1 = self.clip.wrap(self.time)
        return self.clip.events_between(t0, t1)

    def sample(self) -> Pose:
        return self.clip.sample(self.time) if self.clip else {}

    @property
    def finished(self) -> bool:
        return bool(self.clip and not self.clip.loop and self.time >= self.clip.duration)


class Animator:
    """Evaluates a stack of layers into a single pose each frame."""

    def __init__(self, skeleton=None):
        self.skeleton = skeleton
        self.layers: list[AnimationLayer] = []
        self.events: list[str] = []
        self.time_scale = 1.0
        self.pose: Pose = {}

    def add_layer(self, clip=None, weight=1.0, additive=False, mask=None,
                  name="layer", speed=1.0) -> AnimationLayer:
        L = AnimationLayer(clip, weight, speed, 0.0, additive, mask, name)
        self.layers.append(L)
        return L

    def layer(self, name: str) -> AnimationLayer | None:
        for L in self.layers:
            if L.name == name:
                return L
        return None

    def play(self, name: str, clip: AnimationClip, restart: bool = True) -> AnimationLayer:
        L = self.layer(name) or self.add_layer(name=name)
        L.clip = clip
        if restart:
            L.time = 0.0
        L.playing = True
        return L

    def update(self, dt: float) -> Pose:
        dt *= self.time_scale
        self.events = []
        pose: Pose = {}
        for L in self.layers:
            self.events.extend(L.advance(dt))
            if L.clip is None or L.weight <= 1e-4:
                continue
            sampled = L.sample()
            if L.additive:
                pose = add_poses(pose, sampled, L.weight, L.mask)
            else:
                pose = blend_poses(pose, sampled, L.weight, L.mask)
        self.pose = pose
        return pose

    def skinning_matrices(self, pose: Pose | None = None) -> np.ndarray | None:
        if self.skeleton is None:
            return None
        return self.skeleton.skinning_matrices(pose if pose is not None else self.pose)
