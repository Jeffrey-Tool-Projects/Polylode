"""Inverse kinematics: analytic two-bone, CCD chains, look-at and foot planting.

All solvers read the current world pose from the skeleton, compute a world-space
delta rotation per bone, then convert it back into the local pose entry the
skeleton evaluator expects.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import math3d as m3
from .clip import Pose


def _pose_rot(pose: Pose, bone: str) -> np.ndarray:
    e = pose.get(bone)
    if not e or e.get("rot") is None:
        return m3.quat_identity()
    return np.asarray(e["rot"], dtype=np.float64)


def _set_pose_rot(pose: Pose, bone: str, q: np.ndarray) -> None:
    e = pose.setdefault(bone, {})
    e["rot"] = m3.quat_normalize(q)


def apply_world_delta(skel, pose: Pose, world: np.ndarray, bone: str,
                      delta: np.ndarray) -> None:
    """Compose a world-space rotation ``delta`` onto ``bone``'s local pose."""
    i = skel.id(bone)
    world_rot = m3.quat_from_mat4(world[i])
    cur = _pose_rot(pose, bone)
    w0 = m3.quat_mul(world_rot, m3.quat_conj(cur))          # rotation without the pose part
    local = m3.quat_mul(m3.quat_mul(m3.quat_conj(w0), delta), w0)
    _set_pose_rot(pose, bone, m3.quat_mul(local, cur))


def two_bone_ik(skel, pose: Pose, root: str, mid: str, end: str, target,
                pole=None, weight: float = 1.0, stretch: float = 0.0,
                iterations: int = 2, chain_only: bool = False) -> Pose:
    """Analytic IK for a shoulder/elbow/hand or hip/knee/foot chain.

    Each pass sets the elbow's interior angle from the law of cosines, then aims
    the whole chain at the goal.  Two passes bring the residual under a
    millimetre; the aim step alone cannot fix the bend and vice versa.

    Only the chain's three joints are read from each evaluation.  With
    *chain_only* the skeleton is evaluated down to *end* alone
    (``Skeleton.evaluate(only=…)``: the same rows to within the last bit);
    otherwise whole, as before.  A caller that solves an arm many times a frame
    spent most of the frame in whole evaluations (G12).
    """
    if weight <= 1e-4 or not all(skel.has(b) for b in (root, mid, end)):
        return pose
    target = np.asarray(target, dtype=np.float64)
    only = [skel.id(end)] if chain_only else None
    world = skel.evaluate(pose, only=only)
    pr = world[skel.id(root)][:3, 3]
    pm = world[skel.id(mid)][:3, 3]
    pe = world[skel.id(end)][:3, 3]

    a = float(np.linalg.norm(pm - pr))
    b = float(np.linalg.norm(pe - pm))
    if a < 1e-6 or b < 1e-6:
        return pose
    goal = pr + (target - pr) * weight + (pe - pr) * (1.0 - weight)
    c_len = float(np.linalg.norm(goal - pr))
    if stretch <= 0.0:
        c_len = min(c_len, (a + b) * 0.9995)
    c_len = max(c_len, abs(a - b) + 1e-5)
    clamp = lambda x: float(np.clip(x, -1.0, 1.0))
    tgt_mid = math.acos(clamp((a * a + b * b - c_len * c_len) / (2 * a * b)))

    for _ in range(max(1, iterations)):
        world = skel.evaluate(pose, only=only)
        pr = world[skel.id(root)][:3, 3]
        pm = world[skel.id(mid)][:3, 3]
        pe = world[skel.id(end)][:3, 3]

        u = pr - pm
        v = pe - pm
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu > 1e-9 and nv > 1e-9:
            cur_mid = math.acos(clamp(float(np.dot(u, v)) / (nu * nv)))
            axis = np.cross(u, v)
            if np.linalg.norm(axis) < 1e-7:
                # straight limb: pick a bend plane from the pole or a default
                hint = (np.asarray(pole, dtype=np.float64) - pm) if pole is not None \
                    else np.array([0.0, 0.0, 1.0])
                axis = np.cross(u, hint)
                if np.linalg.norm(axis) < 1e-7:
                    axis = np.cross(u, np.array([0.0, 1.0, 0.0]))
            axis = m3.normalize(axis)
            # rotating v about cross(u, v) by +θ opens the joint
            apply_world_delta(skel, pose, world, mid,
                              m3.quat_from_axis_angle(axis, tgt_mid - cur_mid))

        world = skel.evaluate(pose, only=only)
        pr = world[skel.id(root)][:3, 3]
        pe = world[skel.id(end)][:3, 3]
        apply_world_delta(skel, pose, world, root, m3.quat_from_to(pe - pr, goal - pr))

    if pole is not None:
        _apply_pole(skel, pose, root, mid, end, np.asarray(pole, dtype=np.float64))
    return pose


def _apply_pole(skel, pose: Pose, root: str, mid: str, end: str,
                pole: np.ndarray) -> None:
    """Twist the chain about the root->end axis so the mid joint faces the pole."""
    world = skel.evaluate(pose)
    pr = world[skel.id(root)][:3, 3]
    pm = world[skel.id(mid)][:3, 3]
    pe = world[skel.id(end)][:3, 3]
    axis = pe - pr
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return
    axis = axis / n
    proj = lambda v: v - axis * float(np.dot(v, axis))
    cur = proj(pm - pr)
    want = proj(pole - pr)
    if np.linalg.norm(cur) < 1e-7 or np.linalg.norm(want) < 1e-7:
        return
    cur = m3.normalize(cur)
    want = m3.normalize(want)
    ang = math.atan2(float(np.dot(np.cross(cur, want), axis)),
                     float(np.clip(np.dot(cur, want), -1, 1)))
    apply_world_delta(skel, pose, world, root, m3.quat_from_axis_angle(axis, ang))


def ccd_ik(skel, pose: Pose, chain: list[str], target, iterations: int = 8,
           weight: float = 1.0, tolerance: float = 1e-4) -> Pose:
    """Cyclic coordinate descent for chains of arbitrary length (tails, hair)."""
    if not chain or weight <= 1e-4:
        return pose
    target = np.asarray(target, dtype=np.float64)
    end = chain[-1]
    for _ in range(iterations):
        world = skel.evaluate(pose)
        pe = world[skel.id(end)][:3, 3]
        if float(np.linalg.norm(pe - target)) < tolerance:
            break
        for bone in reversed(chain[:-1]):
            world = skel.evaluate(pose)
            pb = world[skel.id(bone)][:3, 3]
            pe = world[skel.id(end)][:3, 3]
            v0 = pe - pb
            v1 = target - pb
            if np.linalg.norm(v0) < 1e-7 or np.linalg.norm(v1) < 1e-7:
                continue
            d = m3.quat_from_to(v0, v1)
            if weight < 1.0:
                d = m3.quat_slerp(m3.quat_identity(), d, weight)
            apply_world_delta(skel, pose, world, bone, d)
    return pose


def look_at(skel, pose: Pose, bone: str, target, forward=(0.0, 0.0, 1.0),
            weight: float = 1.0, max_angle: float = 70.0) -> Pose:
    """Aim a bone's local ``forward`` axis at a world target (head/eye tracking)."""
    if weight <= 1e-4 or not skel.has(bone):
        return pose
    world = skel.evaluate(pose)
    i = skel.id(bone)
    origin = world[i][:3, 3]
    cur_dir = m3.normalize(world[i][:3, :3] @ np.asarray(forward, dtype=np.float64))
    want = np.asarray(target, dtype=np.float64) - origin
    if np.linalg.norm(want) < 1e-7:
        return pose
    want = m3.normalize(want)
    ang = math.acos(float(np.clip(np.dot(cur_dir, want), -1, 1)))
    limit = math.radians(max_angle)
    if ang > limit:
        want = m3.normalize(cur_dir + (want - cur_dir) * (limit / max(ang, 1e-6)))
    d = m3.quat_from_to(cur_dir, want)
    if weight < 1.0:
        d = m3.quat_slerp(m3.quat_identity(), d, weight)
    apply_world_delta(skel, pose, world, bone, d)
    return pose


def spine_look_at(skel, pose: Pose, target, weight: float = 1.0,
                  chain=("cf_j_spine03", "cf_j_neck", "cf_j_head"),
                  shares=(0.18, 0.32, 0.50)) -> Pose:
    """Distribute a look-at across the chest, neck and head."""
    for bone, share in zip(chain, shares):
        look_at(skel, pose, bone, target, (0, 0, 1), weight * share, 55.0)
    return pose


def ground_feet(skel, pose: Pose, ground_y: float = 0.0, weight: float = 1.0,
                sides=("L", "R")) -> Pose:
    """Plant feet on a ground plane using the leg two-bone chains."""
    world = skel.evaluate(pose)
    for side in sides:
        foot = f"cf_j_leg03_{side}"
        if not skel.has(foot):
            continue
        p = world[skel.id(foot)][:3, 3]
        if p[1] >= ground_y:
            continue
        target = np.array([p[0], ground_y, p[2]])
        knee_pole = world[skel.id(f"cf_j_leg01_{side}")][:3, 3] + np.array([0, 0, 0.5])
        two_bone_ik(skel, pose, f"cf_j_thigh00_{side}", f"cf_j_leg01_{side}",
                    foot, target, knee_pole, weight)
    return pose


class IKRig:
    """Named IK goals evaluated in order after the animation layers."""

    LIMBS = {
        "hand_L": ("cf_j_arm00_L", "cf_j_forearm01_L", "cf_j_hand_L"),
        "hand_R": ("cf_j_arm00_R", "cf_j_forearm01_R", "cf_j_hand_R"),
        "foot_L": ("cf_j_thigh00_L", "cf_j_leg01_L", "cf_j_leg03_L"),
        "foot_R": ("cf_j_thigh00_R", "cf_j_leg01_R", "cf_j_leg03_R"),
    }

    def __init__(self, skeleton, limbs: dict | None = None):
        """*limbs* names the chains of a rig that is not the ``cf_j_*`` one:
        ``{goal: (root, mid, end)}``, replacing :data:`LIMBS` for this rig."""
        self.skeleton = skeleton
        if limbs is not None:
            self.LIMBS = dict(limbs)
        self.goals: dict[str, dict] = {}
        self.look_target = None
        self.look_weight = 0.0

    def set_goal(self, name: str, target, weight: float = 1.0, pole=None) -> None:
        if name not in self.LIMBS:
            return
        self.goals[name] = {"target": np.asarray(target, dtype=np.float64),
                            "weight": float(weight), "pole": pole}

    def clear_goal(self, name: str) -> None:
        self.goals.pop(name, None)

    def clear(self) -> None:
        self.goals.clear()
        self.look_weight = 0.0

    def goal_position(self, name: str, pose: Pose | None = None) -> np.ndarray | None:
        """Current world position of the limb tip (used to seed a new goal)."""
        if name not in self.LIMBS:
            return None
        world = self.skeleton.evaluate(pose or {})
        return world[self.skeleton.id(self.LIMBS[name][2])][:3, 3].copy()

    def solve(self, pose: Pose) -> Pose:
        for name, g in self.goals.items():
            if g["weight"] <= 1e-4:
                continue
            root, mid, end = self.LIMBS[name]
            two_bone_ik(self.skeleton, pose, root, mid, end, g["target"],
                        g.get("pole"), g["weight"])
        if self.look_target is not None and self.look_weight > 1e-4:
            spine_look_at(self.skeleton, pose, self.look_target, self.look_weight)
        return pose
