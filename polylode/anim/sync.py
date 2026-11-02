"""Multi-character animation synchronisation.

A ``SyncGroup`` drives several characters from one master clock so paired
animations stay phase-locked no matter how the frame rate jitters.  Members can
carry a phase offset (canon/round dances, marching), a speed ratio (a taller
character taking longer strides), and an optional spatial binding that parents a
follower's root to the leader's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import math3d as m3
from ..core import swallowed


@dataclass
class SyncMember:
    key: str
    animator: object = None            # Animator | StateMachine | MotionMatcher
    phase: float = 0.0                 # 0..1 offset within the shared cycle
    speed: float = 1.0
    weight: float = 1.0
    enabled: bool = True
    # spatial binding
    follow: str | None = None          # key of the leader to parent to
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    face_leader: bool = False
    yaw_offset: float = 0.0

    def clip_duration(self, fallback: float) -> float:
        a = self.animator
        try:
            if hasattr(a, "duration"):
                return float(a.duration)
            if hasattr(a, "states") and getattr(a, "current", None):
                return float(a.states[a.current].duration(getattr(a, "params", {})))
            if hasattr(a, "clip") and a.clip is not None:
                return float(a.clip.duration)
            if hasattr(a, "layers") and a.layers and a.layers[0].clip:
                return float(a.layers[0].clip.duration)
        except Exception as exc:
            swallowed.note("anim.sync.SyncMember.clip_duration: states.duration", exc)
        return fallback


class SyncGroup:
    """One master clock, many followers."""

    def __init__(self, name: str = "sync", cycle: float = 1.0):
        self.name = name
        self.members: dict[str, SyncMember] = {}
        self.order: list[str] = []
        self.cycle = max(float(cycle), 1e-4)
        self.clock = 0.0
        self.speed = 1.0
        self.playing = True
        self.leader: str | None = None
        self.transforms: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def add(self, key: str, animator=None, phase: float = 0.0, speed: float = 1.0,
            follow: str | None = None, offset=(0, 0, 0), face_leader: bool = False,
            yaw_offset: float = 0.0) -> SyncMember:
        m = SyncMember(key, animator, float(phase), float(speed), 1.0, True,
                       follow, np.asarray(offset, dtype=np.float64), face_leader,
                       float(yaw_offset))
        self.members[key] = m
        if key not in self.order:
            self.order.append(key)
        if self.leader is None:
            self.leader = key
        return m

    def remove(self, key: str) -> None:
        self.members.pop(key, None)
        if key in self.order:
            self.order.remove(key)
        if self.leader == key:
            self.leader = self.order[0] if self.order else None
        for m in self.members.values():
            if m.follow == key:
                m.follow = None

    def get(self, key: str) -> SyncMember | None:
        return self.members.get(key)

    # ------------------------------------------------------------------
    def spread_phases(self, mode: str = "even") -> None:
        """Distribute phase offsets — 'even' for canons, 'pairs' for duets."""
        n = len(self.order)
        if n == 0:
            return
        for i, key in enumerate(self.order):
            m = self.members[key]
            if mode == "even":
                m.phase = i / n
            elif mode == "pairs":
                m.phase = 0.0 if i % 2 == 0 else 0.5
            elif mode == "wave":
                m.phase = (i / max(n - 1, 1)) * 0.35
            else:
                m.phase = 0.0

    def align(self) -> None:
        """Snap every member back onto the master clock immediately."""
        for m in self.members.values():
            self._set_member_time(m, self._member_time(m))

    def _member_time(self, m: SyncMember) -> float:
        dur = m.clip_duration(self.cycle)
        return ((self.clock * m.speed) + m.phase * self.cycle) % max(dur, 1e-4)

    @staticmethod
    def _set_member_time(m: SyncMember, t: float) -> None:
        a = m.animator
        if a is None:
            return
        if hasattr(a, "time"):
            a.time = t
        if hasattr(a, "layers"):
            for L in a.layers:
                if L.clip is not None:
                    L.time = t

    # ------------------------------------------------------------------
    def update(self, dt: float) -> dict:
        """Advance the shared clock and every member; returns member times."""
        if self.playing:
            self.clock += dt * self.speed
        times = {}
        for key in self.order:
            m = self.members[key]
            if not m.enabled:
                continue
            t = self._member_time(m)
            times[key] = t
            a = m.animator
            if a is None:
                continue
            # drive by absolute time so drift can never accumulate
            self._set_member_time(m, t)
            if hasattr(a, "update"):
                try:
                    a.update(0.0)
                    self._set_member_time(m, t)
                except TypeError:
                    pass
        self._resolve_bindings()
        return times

    # ------------------------------------------------------------------
    def set_transform(self, key: str, matrix: np.ndarray) -> None:
        self.transforms[key] = np.asarray(matrix, dtype=np.float64).copy()

    def _resolve_bindings(self) -> None:
        """Parent followers to their leader's transform."""
        for _ in range(2):                     # two passes resolve simple chains
            for key in self.order:
                m = self.members[key]
                if not m.follow or m.follow not in self.transforms:
                    continue
                lead = self.transforms[m.follow]
                local = m3.translate(m.offset)
                if m.face_leader:
                    local = local @ m3.rotate_y(math.pi + m.yaw_offset)
                elif abs(m.yaw_offset) > 1e-6:
                    local = local @ m3.rotate_y(m.yaw_offset)
                self.transforms[key] = lead @ local

    def transform(self, key: str) -> np.ndarray:
        return self.transforms.get(key, m3.identity())

    # ------------------------------------------------------------------
    def pair(self, leader: str, follower: str, distance: float = 0.55,
             face: bool = True, phase_offset: float = 0.5) -> None:
        """Set up a two-character interaction (talking, dancing, handing over)."""
        m = self.members.get(follower)
        if m is None:
            return
        m.follow = leader
        m.offset = np.array([0.0, 0.0, distance])
        m.face_leader = face
        m.phase = phase_offset

    def formation(self, shape: str = "line", spacing: float = 0.8) -> None:
        """Arrange members without a follow-binding into a formation."""
        keys = [k for k in self.order if not self.members[k].follow]
        n = len(keys)
        for i, key in enumerate(keys):
            if shape == "line":
                pos = np.array([(i - (n - 1) / 2) * spacing, 0.0, 0.0])
                yaw = 0.0
            elif shape == "circle":
                a = i / max(n, 1) * 2 * math.pi
                r = spacing * max(n, 1) / (2 * math.pi)
                pos = np.array([math.sin(a) * r, 0.0, math.cos(a) * r])
                yaw = a + math.pi
            elif shape == "grid":
                cols = max(1, int(math.ceil(math.sqrt(n))))
                pos = np.array([(i % cols - (cols - 1) / 2) * spacing, 0.0,
                                (i // cols) * spacing])
                yaw = 0.0
            else:
                pos = np.zeros(3)
                yaw = 0.0
            self.transforms[key] = m3.translate(pos) @ m3.rotate_y(yaw)

    def to_dict(self) -> dict:
        return {"name": self.name, "cycle": self.cycle, "speed": self.speed,
                "leader": self.leader, "order": list(self.order),
                "members": {k: {"phase": m.phase, "speed": m.speed,
                                "follow": m.follow, "offset": m.offset.tolist(),
                                "face_leader": m.face_leader,
                                "yaw_offset": m.yaw_offset, "enabled": m.enabled}
                            for k, m in self.members.items()}}

    def from_dict(self, d: dict) -> "SyncGroup":
        self.name = d.get("name", self.name)
        self.cycle = float(d.get("cycle", 1.0))
        self.speed = float(d.get("speed", 1.0))
        self.leader = d.get("leader")
        for k, md in (d.get("members") or {}).items():
            m = self.members.get(k) or self.add(k)
            m.phase = float(md.get("phase", 0.0))
            m.speed = float(md.get("speed", 1.0))
            m.follow = md.get("follow")
            m.offset = np.asarray(md.get("offset", [0, 0, 0]), dtype=np.float64)
            m.face_leader = bool(md.get("face_leader", False))
            m.yaw_offset = float(md.get("yaw_offset", 0.0))
            m.enabled = bool(md.get("enabled", True))
        self.order = [k for k in d.get("order", list(self.members)) if k in self.members]
        return self

    def __repr__(self):
        return f"<SyncGroup {self.name} members={len(self.members)} clock={self.clock:.2f}>"
