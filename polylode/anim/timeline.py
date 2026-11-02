"""Keyframe timeline — the studio's animation editor.

Tracks address scene properties by string path; the owning scene registers
getter/setter pairs so the timeline stays decoupled from the object graph.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from ..core import math3d as m3


FLOAT = "float"
VEC3 = "vec3"
QUAT = "quat"
COLOR = "color"
BOOL = "bool"
INT = "int"


def _ease(t: float, kind: str) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    if kind == "step":
        return 0.0
    if kind == "smooth" or kind == "ease_both":
        return t * t * (3.0 - 2.0 * t)
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return 1.0 - (1.0 - t) ** 2
    return t


@dataclass
class Keyframe:
    time: float
    value: object
    interp: str = "smooth"

    def to_dict(self) -> dict:
        v = self.value
        if isinstance(v, np.ndarray):
            v = v.tolist()
        return {"t": round(float(self.time), 5), "v": v, "i": self.interp}

    @staticmethod
    def from_dict(d: dict) -> "Keyframe":
        return Keyframe(float(d["t"]), d["v"], d.get("i", "smooth"))


class Track:
    def __init__(self, path: str, kind: str = FLOAT, label: str = ""):
        self.path = path
        self.kind = kind
        self.label = label or path
        self.keys: list[Keyframe] = []
        self.enabled = True

    # -- editing -----------------------------------------------------------
    def _times(self) -> list[float]:
        return [k.time for k in self.keys]

    def add_key(self, time: float, value, interp: str = "smooth") -> Keyframe:
        time = float(time)
        for k in self.keys:
            if abs(k.time - time) < 1e-5:
                k.value = value
                k.interp = interp
                return k
        k = Keyframe(time, value, interp)
        bisect.insort(self.keys, k, key=lambda x: x.time)
        return k

    def remove_key(self, time: float, tol: float = 1e-3) -> bool:
        for i, k in enumerate(self.keys):
            if abs(k.time - time) <= tol:
                self.keys.pop(i)
                return True
        return False

    def move_key(self, old_time: float, new_time: float, tol: float = 1e-3) -> bool:
        for k in self.keys:
            if abs(k.time - old_time) <= tol:
                k.time = float(new_time)
                self.keys.sort(key=lambda x: x.time)
                return True
        return False

    def nearest_key(self, time: float) -> Keyframe | None:
        if not self.keys:
            return None
        return min(self.keys, key=lambda k: abs(k.time - time))

    @property
    def duration(self) -> float:
        return self.keys[-1].time if self.keys else 0.0

    # -- sampling ----------------------------------------------------------
    def sample(self, t: float):
        if not self.keys:
            return None
        if len(self.keys) == 1 or t <= self.keys[0].time:
            return self.keys[0].value
        if t >= self.keys[-1].time:
            return self.keys[-1].value
        i = bisect.bisect_right(self._times(), t) - 1
        i = max(0, min(i, len(self.keys) - 2))
        a, b = self.keys[i], self.keys[i + 1]
        span = b.time - a.time
        f = 0.0 if span <= 1e-9 else (t - a.time) / span
        f = _ease(f, a.interp)
        return self._lerp(a.value, b.value, f)

    def _lerp(self, a, b, f: float):
        if self.kind == BOOL:
            return a if f < 1.0 else b
        if self.kind == INT:
            return int(round(a + (b - a) * f))
        if self.kind == QUAT:
            return m3.quat_slerp(np.asarray(a, dtype=np.float64),
                                 np.asarray(b, dtype=np.float64), f)
        if self.kind in (VEC3, COLOR):
            av = np.asarray(a, dtype=np.float64)
            bv = np.asarray(b, dtype=np.float64)
            return av + (bv - av) * f
        return float(a) + (float(b) - float(a)) * f

    def to_dict(self) -> dict:
        return {"path": self.path, "kind": self.kind, "label": self.label,
                "enabled": self.enabled, "keys": [k.to_dict() for k in self.keys]}

    @staticmethod
    def from_dict(d: dict) -> "Track":
        t = Track(d["path"], d.get("kind", FLOAT), d.get("label", ""))
        t.enabled = d.get("enabled", True)
        t.keys = [Keyframe.from_dict(k) for k in d.get("keys", [])]
        return t


class Timeline:
    """Ordered tracks + a playhead. ``resolver`` maps a path to (get, set)."""

    def __init__(self, name: str = "timeline", duration: float = 10.0, fps: float = 30.0):
        self.name = name
        self.tracks: list[Track] = []
        self.duration = float(duration)
        self.fps = float(fps)
        self.time = 0.0
        self.playing = False
        self.loop = True
        self.speed = 1.0
        self.resolver = {}
        self.markers: list[tuple[float, str]] = []
        self.on_event = None

    # -- tracks ------------------------------------------------------------
    def track(self, path: str, kind: str = FLOAT, label: str = "") -> Track:
        for t in self.tracks:
            if t.path == path:
                return t
        t = Track(path, kind, label)
        self.tracks.append(t)
        return t

    def find(self, path: str) -> Track | None:
        for t in self.tracks:
            if t.path == path:
                return t
        return None

    def remove(self, path: str) -> None:
        self.tracks = [t for t in self.tracks if t.path != path]

    def bind(self, path: str, getter, setter, kind: str = FLOAT, label: str = "") -> Track:
        self.resolver[path] = (getter, setter)
        return self.track(path, kind, label)

    def unbind(self, path: str) -> None:
        self.resolver.pop(path, None)

    # -- keying ------------------------------------------------------------
    def key_current(self, path: str, interp: str = "smooth") -> Keyframe | None:
        """Snapshot the bound property's live value at the playhead."""
        entry = self.resolver.get(path)
        tr = self.find(path)
        if entry is None or tr is None:
            return None
        v = entry[0]()
        if isinstance(v, np.ndarray):
            v = v.copy()
        return tr.add_key(self.time, v, interp)

    def key_all(self, interp: str = "smooth") -> int:
        n = 0
        for t in self.tracks:
            if self.key_current(t.path, interp):
                n += 1
        return n

    def delete_keys_at(self, time: float, tol: float = 1e-3) -> int:
        return sum(1 for t in self.tracks if t.remove_key(time, tol))

    # -- playback ----------------------------------------------------------
    def refresh_duration(self) -> float:
        d = max([t.duration for t in self.tracks] + [0.0])
        self.duration = max(self.duration, d)
        return self.duration

    def seek(self, t: float, apply: bool = True) -> None:
        self.time = float(np.clip(t, 0.0, max(self.duration, 1e-4)))
        if apply:
            self.apply()

    def frame_step(self, frames: int = 1) -> None:
        self.seek(self.time + frames / max(self.fps, 1e-3))

    def snap_to_frame(self) -> None:
        self.time = round(self.time * self.fps) / max(self.fps, 1e-3)

    def update(self, dt: float) -> bool:
        if not self.playing:
            return False
        t0 = self.time
        self.time += dt * self.speed
        if self.time >= self.duration:
            if self.loop:
                self.time = self.time % max(self.duration, 1e-4)
            else:
                self.time = self.duration
                self.playing = False
        self._fire_markers(t0, self.time)
        self.apply()
        return True

    def _fire_markers(self, t0: float, t1: float) -> None:
        if not self.markers or self.on_event is None:
            return
        if t1 < t0:
            hit = [n for tm, n in self.markers if tm >= t0] + \
                  [n for tm, n in self.markers if tm <= t1]
        else:
            hit = [n for tm, n in self.markers if t0 < tm <= t1]
        for n in hit:
            self.on_event(n)

    def apply(self) -> int:
        n = 0
        for tr in self.tracks:
            if not tr.enabled or not tr.keys:
                continue
            entry = self.resolver.get(tr.path)
            if entry is None:
                continue
            v = tr.sample(self.time)
            if v is not None:
                entry[1](v)
                n += 1
        return n

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"name": self.name, "duration": self.duration, "fps": self.fps,
                "loop": self.loop, "speed": self.speed,
                "markers": [list(m) for m in self.markers],
                "tracks": [t.to_dict() for t in self.tracks]}

    def from_dict(self, d: dict) -> "Timeline":
        self.name = d.get("name", self.name)
        self.duration = float(d.get("duration", 10.0))
        self.fps = float(d.get("fps", 30.0))
        self.loop = bool(d.get("loop", True))
        self.speed = float(d.get("speed", 1.0))
        self.markers = [tuple(m) for m in d.get("markers", [])]
        self.tracks = [Track.from_dict(t) for t in d.get("tracks", [])]
        return self

    def to_clip(self, skeleton_prefix: str = "bone/"):
        """Bake bone tracks into an AnimationClip for reuse outside the studio."""
        from .clip import AnimationClip
        clip = AnimationClip(self.name, max(self.duration, 1e-3), self.loop)
        per_bone: dict[str, Track] = {}
        for tr in self.tracks:
            if tr.path.startswith(skeleton_prefix) and tr.kind == QUAT:
                per_bone[tr.path[len(skeleton_prefix):]] = tr
        n = max(2, int(self.duration * self.fps))
        times = np.linspace(0.0, self.duration, n)
        for bone, tr in per_bone.items():
            rots = np.array([np.asarray(tr.sample(t), dtype=np.float64) for t in times])
            clip.set_curve(bone, times, rots=rots)
        return clip

    def __repr__(self):
        return f"<Timeline {self.name} tracks={len(self.tracks)} {self.duration:.1f}s>"
