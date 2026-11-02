"""Motion matching: a searchable pose/trajectory database over the clip library.

Each database entry is one sampled frame described by
  * pose features   — foot positions and velocities plus hip velocity, all in
                      root-local space
  * trajectory features — where the root will be, and which way it will face,
                      at three future horizons

Each tick the controller builds the same feature vector from the *desired*
trajectory and the *current* pose, finds the nearest entry, and only jumps if
the winner beats the frame it would otherwise play by a margin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .clip import AnimationClip, Pose, blend_poses

HORIZONS = (0.22, 0.45, 0.75)
FOOT_BONES = ("cf_j_leg03_L", "cf_j_leg03_R")
HIP_BONE = "cf_j_hips"

# trajectory leads, pose follows: the desired path is what the player controls
POSE_WEIGHT = 0.6
TRAJ_WEIGHT = 2.2


@dataclass
class MotionEntry:
    """One clip registered into the database with its locomotion metadata."""
    clip: AnimationClip
    velocity: tuple = (0.0, 0.0)      # metres/second in root space (x = right, z = fwd)
    angular: float = 0.0              # radians/second of root yaw
    tag: str = "locomotion"
    cost_bias: float = 0.0            # added to every frame's cost (discourage/prefer)


@dataclass
class MotionFrame:
    clip_index: int
    time: float
    pose_feat: np.ndarray
    traj_feat: np.ndarray


class MotionDatabase:
    def __init__(self, skeleton, sample_rate: float = 20.0):
        self.skeleton = skeleton
        self.sample_rate = sample_rate
        self.entries: list[MotionEntry] = []
        self.frames: list[MotionFrame] = []
        self.pose_matrix: np.ndarray | None = None
        self.traj_matrix: np.ndarray | None = None
        self.pose_scale: np.ndarray | None = None
        self.traj_scale: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    # ------------------------------------------------------------------
    def add(self, clip: AnimationClip, velocity=(0.0, 0.0), angular: float = 0.0,
            tag: str = "locomotion", cost_bias: float = 0.0) -> MotionEntry:
        e = MotionEntry(clip, tuple(velocity), float(angular), tag, float(cost_bias))
        self.entries.append(e)
        return e

    # ------------------------------------------------------------------
    def _pose_features(self, clip: AnimationClip, t: float, dt: float) -> np.ndarray:
        skel = self.skeleton
        w0 = skel.evaluate(clip.sample(t))
        w1 = skel.evaluate(clip.sample(t + dt))
        hip = w0[skel.id(HIP_BONE)][:3, 3]
        feats = []
        for b in FOOT_BONES:
            if not skel.has(b):
                feats.extend([0.0] * 6)
                continue
            p0 = w0[skel.id(b)][:3, 3] - hip
            p1 = w1[skel.id(b)][:3, 3] - w1[skel.id(HIP_BONE)][:3, 3]
            feats.extend(p0.tolist())
            feats.extend(((p1 - p0) / dt).tolist())
        hip1 = w1[skel.id(HIP_BONE)][:3, 3]
        feats.extend(((hip1 - hip) / dt).tolist())
        return np.asarray(feats, dtype=np.float64)

    @staticmethod
    def _trajectory(velocity, angular: float) -> np.ndarray:
        """Integrate constant velocity + yaw rate into the horizon samples."""
        vx, vz = float(velocity[0]), float(velocity[1])
        out = []
        for h in HORIZONS:
            if abs(angular) < 1e-5:
                px, pz = vx * h, vz * h
                yaw = 0.0
            else:
                yaw = angular * h
                # arc integration of a constant-speed turn
                s, c = math.sin(yaw), math.cos(yaw)
                px = (vx * s + vz * (1 - c)) / angular
                pz = (vz * s - vx * (1 - c)) / angular
            out.extend([px, pz, math.sin(yaw), math.cos(yaw)])
        return np.asarray(out, dtype=np.float64)

    # ------------------------------------------------------------------
    def build(self) -> "MotionDatabase":
        self.frames = []
        dt = 1.0 / self.sample_rate
        for ci, e in enumerate(self.entries):
            n = max(2, int(round(e.clip.duration * self.sample_rate)))
            for k in range(n):
                t = e.clip.duration * k / n
                pf = self._pose_features(e.clip, t, dt)
                tf = self._trajectory(e.velocity, e.angular)
                self.frames.append(MotionFrame(ci, t, pf, tf))
        if not self.frames:
            return self
        self.pose_matrix = np.stack([f.pose_feat for f in self.frames])
        self.traj_matrix = np.stack([f.traj_feat for f in self.frames])
        # foot *positions* carry continuity; foot velocities are noisier and
        # would otherwise stop idle->walk transitions from ever winning
        per_foot = np.array([1.0, 1.0, 1.0, 0.45, 0.45, 0.45])
        importance = np.concatenate([per_foot, per_foot, np.full(3, 0.6)])
        self.pose_scale = importance / np.maximum(self.pose_matrix.std(axis=0), 1e-3)
        self.traj_scale = 1.0 / np.maximum(self.traj_matrix.std(axis=0), 1e-3)
        self.bias = np.array([self.entries[f.clip_index].cost_bias for f in self.frames])
        self._pose_norm = self.pose_matrix * self.pose_scale
        self._traj_norm = self.traj_matrix * self.traj_scale
        return self

    # ------------------------------------------------------------------
    def query(self, pose_feat: np.ndarray, traj_feat: np.ndarray,
              pose_weight: float = 0.6, traj_weight: float = 2.2,
              exclude_window: tuple | None = None) -> tuple[int, float]:
        """Return (frame index, cost) of the best match."""
        if not self.frames:
            return -1, float("inf")
        dp = self._pose_norm - pose_feat * self.pose_scale
        dt_ = self._traj_norm - traj_feat * self.traj_scale
        cost = (pose_weight * np.einsum("ij,ij->i", dp, dp)
                + traj_weight * np.einsum("ij,ij->i", dt_, dt_))
        cost = cost + self.bias
        if exclude_window is not None:
            ci, t0, t1 = exclude_window
            for i, f in enumerate(self.frames):
                if f.clip_index == ci and t0 <= f.time <= t1:
                    cost[i] += 1e6
        i = int(np.argmin(cost))
        return i, float(cost[i])

    def features_from_pose(self, clip: AnimationClip, t: float) -> np.ndarray:
        return self._pose_features(clip, t, 1.0 / self.sample_rate)

    def __len__(self):
        return len(self.frames)


class MotionMatcher:
    """Plays the database, re-querying on a fixed cadence and blending on jumps."""

    def __init__(self, db: MotionDatabase, search_interval: float = 0.12,
                 blend_time: float = 0.22, switch_margin: float = 0.55):
        self.db = db
        self.search_interval = search_interval
        self.blend_time = blend_time
        self.switch_margin = switch_margin
        self.frame = 0 if len(db) else -1
        self.time = db.frames[0].time if len(db) else 0.0
        self.prev_frame = -1
        self.prev_time = 0.0
        self.blend = 1.0
        self._search_t = 0.0
        self.pose: Pose = {}
        self.desired_velocity = (0.0, 0.0)
        self.desired_angular = 0.0
        self.last_cost = 0.0
        self.switches = 0

    # ------------------------------------------------------------------
    @property
    def clip(self) -> AnimationClip | None:
        if self.frame < 0:
            return None
        return self.db.entries[self.db.frames[self.frame].clip_index].clip

    @property
    def entry(self) -> MotionEntry | None:
        if self.frame < 0:
            return None
        return self.db.entries[self.db.frames[self.frame].clip_index]

    def set_desired(self, velocity=(0.0, 0.0), angular: float = 0.0) -> None:
        self.desired_velocity = (float(velocity[0]), float(velocity[1]))
        self.desired_angular = float(angular)

    # ------------------------------------------------------------------
    def update(self, dt: float) -> Pose:
        if self.frame < 0:
            return {}
        clip = self.clip
        self.time += dt
        if clip.loop:
            self.time %= clip.duration
        else:
            self.time = min(self.time, clip.duration)

        self._search_t += dt
        if self._search_t >= self.search_interval:
            self._search_t = 0.0
            self._search()

        pose = clip.sample(self.time)
        if self.blend < 1.0 and self.prev_frame >= 0:
            pclip = self.db.entries[self.db.frames[self.prev_frame].clip_index].clip
            self.prev_time += dt
            if pclip.loop:
                self.prev_time %= pclip.duration
            k = self.blend
            pose = blend_poses(pclip.sample(self.prev_time), pose, k * k * (3 - 2 * k))
            self.blend = min(1.0, self.blend + dt / max(self.blend_time, 1e-4))
        self.pose = pose
        return pose

    def _search(self) -> None:
        clip = self.clip
        pose_feat = self.db.features_from_pose(clip, self.time)
        traj_feat = self.db._trajectory(self.desired_velocity, self.desired_angular)
        idx, cost = self.db.query(pose_feat, traj_feat)
        self.last_cost = cost

        cur = self.db.frames[self.frame]
        # cost of simply continuing: the pose already matches itself, so only
        # the trajectory mismatch of the *active clip* counts
        cur_traj = self.db._trajectory(self.entry.velocity, self.entry.angular)
        dt_ = (cur_traj - traj_feat) * self.db.traj_scale
        cur_cost = TRAJ_WEIGHT * float(np.dot(dt_, dt_)) + float(self.db.bias[self.frame])

        nxt = self.db.frames[idx]
        same_spot = (nxt.clip_index == cur.clip_index
                     and abs(nxt.time - self.time) < self.search_interval * 1.5)
        if same_spot or cost > cur_cost - self.switch_margin:
            return
        self.prev_frame = self.frame
        self.prev_time = self.time
        self.frame = idx
        self.time = nxt.time
        self.blend = 0.0
        self.switches += 1

    def _cost_of(self, index: int, pose_feat: np.ndarray, traj_feat: np.ndarray) -> float:
        db = self.db
        dp = (db.pose_matrix[index] - pose_feat) * db.pose_scale
        dt_ = (db.traj_matrix[index] - traj_feat) * db.traj_scale
        return float(POSE_WEIGHT * np.dot(dp, dp) + TRAJ_WEIGHT * np.dot(dt_, dt_)
                     + db.bias[index])

    def root_delta(self, dt: float) -> tuple[np.ndarray, float]:
        """Root motion produced by the active clip this frame."""
        e = self.entry
        if e is None:
            return np.zeros(3), 0.0
        return np.array([e.velocity[0] * dt, 0.0, e.velocity[1] * dt]), e.angular * dt

    def describe(self) -> str:
        e = self.entry
        return (f"{e.clip.name if e else '-'} t={self.time:.2f} "
                f"cost={self.last_cost:.2f} switches={self.switches}")


# --------------------------------------------------------------------------
def build_locomotion_database(skeleton, lib: dict) -> MotionDatabase:
    """Idle, walk, run plus synthesised turning and strafing variants."""
    db = MotionDatabase(skeleton, sample_rate=20.0)
    db.add(lib["idle"], (0.0, 0.0), 0.0, "idle", cost_bias=0.0)
    db.add(lib["idle_soft"], (0.0, 0.0), 0.0, "idle", cost_bias=0.15)
    db.add(lib["walk"], (0.0, 1.15), 0.0, "walk")
    db.add(lib["run"], (0.0, 2.85), 0.0, "run")

    # turning variants: same footwork, different assumed yaw rate
    from .library import make_walk, make_run
    for rate, name in ((0.9, "walk_turn_l"), (-0.9, "walk_turn_r")):
        c = make_walk(name, 1.05)
        db.add(c, (0.0, 1.05), rate, "walk", cost_bias=0.05)
    for rate, name in ((0.8, "run_turn_l"), (-0.8, "run_turn_r")):
        c = make_run(name, 0.68)
        db.add(c, (0.0, 2.6), rate, "run", cost_bias=0.08)
    # strafes: reuse the walk footwork with lateral intent
    for vx, name in ((1.0, "strafe_l"), (-1.0, "strafe_r")):
        c = make_walk(name, 1.15, stride=0.75)
        db.add(c, (vx, 0.0), 0.0, "strafe", cost_bias=0.12)
    return db.build()
