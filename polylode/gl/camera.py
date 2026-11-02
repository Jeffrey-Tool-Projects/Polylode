"""Orbit / free camera with focal-length controls and framing presets."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import math3d as m3
from ..core.i18n import N_

SENSOR_MM = 24.0        # vertical sensor size used for the focal-length mapping


@dataclass
class Camera:
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.05, 0.0]))
    distance: float = 2.4
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    fov: float = 30.0
    near: float = 0.02
    far: float = 200.0
    ortho: bool = False
    ortho_size: float = 1.2

    # ---- derived ---------------------------------------------------------
    def direction(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return np.array([math.sin(self.yaw) * cp, math.sin(self.pitch),
                         math.cos(self.yaw) * cp])

    def eye(self) -> np.ndarray:
        return np.asarray(self.target, dtype=np.float64) + self.direction() * self.distance

    def up(self) -> np.ndarray:
        if abs(self.roll) < 1e-6:
            return np.array([0.0, 1.0, 0.0])
        f = -self.direction()
        r = m3.normalize(np.cross(f, np.array([0.0, 1.0, 0.0])))
        return m3.quat_rotate(m3.quat_from_axis_angle(f, self.roll), np.cross(r, f))

    def view(self) -> np.ndarray:
        return m3.look_at(self.eye(), self.target, self.up())

    def projection(self, aspect: float) -> np.ndarray:
        if self.ortho:
            h = self.ortho_size
            return m3.ortho(-h * aspect, h * aspect, -h, h, self.near, self.far)
        return m3.perspective(self.fov, aspect, self.near, self.far)

    # ---- focal length ----------------------------------------------------
    @property
    def focal_length(self) -> float:
        return SENSOR_MM / (2.0 * math.tan(math.radians(self.fov) * 0.5))

    @focal_length.setter
    def focal_length(self, mm: float) -> None:
        mm = max(4.0, float(mm))
        self.fov = math.degrees(2.0 * math.atan(SENSOR_MM / (2.0 * mm)))

    # ---- interaction -----------------------------------------------------
    def orbit(self, dyaw: float, dpitch: float) -> None:
        self.yaw = (self.yaw + dyaw) % (2 * math.pi)
        lim = math.pi * 0.499
        self.pitch = max(-lim, min(lim, self.pitch + dpitch))

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, 0.06, 120.0))
        self.ortho_size = float(np.clip(self.ortho_size * factor, 0.03, 60.0))

    def dolly(self, delta: float) -> None:
        self.distance = float(np.clip(self.distance + delta, 0.06, 120.0))

    def pan(self, dx: float, dy: float) -> None:
        f = -self.direction()
        r = m3.normalize(np.cross(f, np.array([0.0, 1.0, 0.0])))
        u = np.cross(r, f)
        scale = self.distance * math.tan(math.radians(self.fov) * 0.5) * 2.0
        self.target = np.asarray(self.target, dtype=np.float64) + (-r * dx + u * dy) * scale

    def frame(self, lo: np.ndarray, hi: np.ndarray, margin: float = 1.25) -> None:
        center = (np.asarray(lo) + np.asarray(hi)) * 0.5
        radius = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))) * 0.5
        self.target = center
        self.distance = max(0.15, radius * margin / max(math.tan(math.radians(self.fov) * 0.5), 1e-3))
        self.ortho_size = max(0.05, radius * margin)

    def copy(self) -> "Camera":
        c = Camera()
        c.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                           for k, v in self.__dict__.items()})
        return c

    def to_dict(self) -> dict:
        return dict(target=list(map(float, self.target)), distance=self.distance,
                    yaw=self.yaw, pitch=self.pitch, roll=self.roll, fov=self.fov,
                    ortho=self.ortho, ortho_size=self.ortho_size)

    def from_dict(self, d: dict) -> "Camera":
        self.target = np.array(d.get("target", [0, 1.05, 0]), dtype=np.float64)
        self.distance = float(d.get("distance", 2.4))
        self.yaw = float(d.get("yaw", 0.0))
        self.pitch = float(d.get("pitch", 0.0))
        self.roll = float(d.get("roll", 0.0))
        self.fov = float(d.get("fov", 30.0))
        self.ortho = bool(d.get("ortho", False))
        self.ortho_size = float(d.get("ortho_size", 1.2))
        return self


# framing presets: (label, target_height_fraction, vertical span in body heights,
#                   pitch_deg, fov). The distance is derived from the span so a
# preset frames the same amount of the body in any viewport aspect.
FRAMING = {
    "full": (N_("全身"), 0.50, 1.10, -2.0, 30.0),
    "upper": (N_("上半身"), 0.79, 0.46, -3.0, 32.0),
    "face": (N_("臉部"), 0.935, 0.19, -1.0, 30.0),
    "bust": (N_("胸上"), 0.855, 0.32, -4.0, 32.0),
    "legs": (N_("下半身"), 0.27, 0.60, 4.0, 32.0),
    "feet": (N_("腳部"), 0.07, 0.22, 12.0, 32.0),
    "back": (N_("背面"), 0.50, 1.10, -2.0, 30.0),
}


# Unity's ``Camera.main``, which is not a convenience here but part of a
# transcription: ``ChaControl.ChangeLookEyesTarget`` resolves ``eyesTargetType``
# 0 to ``Camera.main.transform``, so "the character looks at the camera" is the
# game's default and every character wants the same one.  Whoever is drawing
# registers it; a screen that never does simply leaves the eyes at rest.
_MAIN: list = []


def set_main_camera(cam: "Camera | None") -> None:
    _MAIN[:] = [cam] if cam is not None else []


def main_camera() -> "Camera | None":
    return _MAIN[0] if _MAIN else None


# ``tools/originshot``'s ``BuildCamera``, transcribed.  The generator writes every
# parameter into the file name precisely so a shot can be reproduced, and half
# of "reproduced" is our side placing its camera by the same rule — otherwise
# the two pictures differ by their framings and nothing measured off them is
# about the geometry (the failure §7.52 / §7.83 / §7.91 all came out of).
#
# ``bounds`` has to be the same thing the plugin fed it: the union of the
# character's **renderer** bounds, not the skeleton's.  A centaur's body is an
# accessory, so a skeleton-framed shot would crop the very thing being looked
# at.  One difference is worth knowing and is far below what it is used for:
# Unity's ``Renderer.bounds`` for a skinned mesh is ``m_LocalAABB`` through the
# root bone, and that box is the union of the base mesh with **every blend
# shape at full weight** (§7.79 三) — 2.171 mm larger than the vertices on
# ``cf_O_face``, i.e. about 0.1% of a full-body frame.
ORIGINSHOT_BOTTOM = {"full": 0.0, "upper": 0.45, "face": 0.86}
ORIGINSHOT_MARGIN = 1.06


def apply_originshot_framing(cam: Camera, lo, hi, view: str = "full",
                         fov: float = 45.0, aspect: float = 512.0 / 768.0,
                         aim=None, distance: float | None = None) -> Camera:
    """Place *cam* the way ``OriginShot`` placed the game's.

    ``aim`` and ``distance`` are the shot's **recorded** camera, and when they
    are there nothing is re-derived — which is the only way the two pictures are
    the same picture.  Re-deriving from geometry does not work and this is
    measured, not assumed: ``Renderer.bounds`` is ``m_LocalAABB`` through the
    root bone, so it carries the blend shapes at full weight and whatever box a
    mod's exporter wrote, and on ``♥Hapi♥`` that box is **40% wider** than the
    mesh (need 2.949 m against 2.103 m, camera 3.773 m against 2.691 m).

    Without them the fallback runs the same arithmetic off our own mesh AABB.
    It is the right shape and the wrong size, so it is for eyeballing only; a
    caller that wants a measurement should re-shoot with a plugin new enough to
    write ``aim`` into the name.
    """
    if view not in ORIGINSHOT_BOTTOM:
        raise ValueError(f"originshot view {view!r} 不在 {sorted(ORIGINSHOT_BOTTOM)}")
    cam.fov = fov
    cam.yaw = 0.0
    cam.pitch = 0.0
    cam.roll = 0.0
    if aim is not None and distance is not None:
        cam.target = np.asarray(aim, dtype=float).copy()
        cam.distance = float(distance)
        cam.ortho_size = (float(distance)
                          * math.tan(math.radians(fov) * 0.5))
        return cam
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    centre = (lo + hi) * 0.5
    top = float(hi[1])
    bottom = float(lo[1] + (hi[1] - lo[1]) * ORIGINSHOT_BOTTOM[view])
    height = max(top - bottom, 0.01)
    mid = (top + bottom) * 0.5
    need = max(height, float(hi[0] - lo[0]) / max(aspect, 1e-3))
    cam.target = np.array([float(centre[0]), mid, float(centre[2])])
    cam.distance = (need * ORIGINSHOT_MARGIN * 0.5
                    / max(math.tan(math.radians(fov) * 0.5), 1e-3))
    cam.ortho_size = need * ORIGINSHOT_MARGIN * 0.5
    return cam


def apply_framing(cam: Camera, key: str, height: float, aspect: float = 1.0) -> Camera:
    if key not in FRAMING:
        return cam
    _, frac, span, pitch, fov = FRAMING[key]
    cam.target = np.array([0.0, height * frac, 0.0])
    cam.fov = fov
    half = span * height * 0.5
    # in a wide viewport the vertical FOV is the binding constraint; in a narrow
    # one the horizontal one is, so widen the required span accordingly
    if aspect < 1.0:
        half /= max(aspect, 0.2)
    cam.distance = half / max(math.tan(math.radians(fov) * 0.5), 1e-3)
    cam.ortho_size = half
    cam.pitch = math.radians(pitch)
    cam.yaw = math.pi if key == "back" else 0.0
    return cam
