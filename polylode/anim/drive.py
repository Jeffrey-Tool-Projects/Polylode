"""Which pose a humanoid is asked for this frame, and how it gets there.

One object owns the four ways a character can be driven — a held pose from the
library, a clip, the locomotion state machine, motion matching — plus what is
layered on every one of them: bones turned by hand, and the standing rest the
library's poses are authored against.

It knows nothing about what wears the skeleton.  ``step`` hands back a pose for
:attr:`skeleton`; whoever owns the drive evaluates it, or carries it onto
another rig first (``endfield/retarget.py``).
"""
from __future__ import annotations

import numpy as np

from ..core import math3d as m3
from .clip import Animator, Pose, blend_poses
from .library import (GROUNDED_POSES, GROUNDED_TAGS, build_default_library,
                      compose_rest, standing_rest)
from .motion_matching import MotionMatcher, build_locomotion_database
from .pose import PoseLibrary
from .state_machine import build_locomotion_fsm

DRIVE_POSE = "pose"
DRIVE_CLIP = "clip"
DRIVE_FSM = "fsm"
DRIVE_MOTION_MATCH = "motion"


class PoseDrive:
    """The pose source of one humanoid skeleton.

    ``authored_pose`` is what was asked for without the standing rest under it
    — what a blend starts from and what a saved pose has to hold, since loading
    it back puts the rest underneath again.
    """

    def __init__(self, skeleton, rest: Pose | None = None):
        self.skeleton = skeleton
        self.library = build_default_library()
        self.poses = PoseLibrary()
        self.rest: Pose = standing_rest(skeleton) if rest is None else rest
        self.animator = Animator(skeleton)
        self.animator.play("base", self.library["idle"])
        self.animator.add_layer(self.library["breathe"], 0.55, additive=True,
                                name="breathe")
        self.fsm = build_locomotion_fsm(self.library)
        self.matcher: MotionMatcher | None = None
        self.drive = DRIVE_POSE
        self.pose_name = "relaxed"
        self.base_pose: Pose = {}
        self.manual_pose: Pose = {}
        self.authored_pose: Pose = {}
        self.apply_rest = True
        self.blend_speed = 8.0
        self._blend_from: Pose = {}
        self._blend_t = 1.0
        self._pose_dirty = True

    # -- choosing the source ------------------------------------------------
    def set_pose(self, name: str, immediate: bool = False) -> None:
        """Hold the library pose *name*, blending from what is shown unless *immediate*."""
        self.pose_name = name
        self.drive = DRIVE_POSE
        self.apply_rest = True
        if not immediate:
            self._blend_from = dict(self.authored_pose)
        self.base_pose = self.poses.get(name)
        self._blend_t = 1.0 if immediate else 0.0
        self._pose_dirty = True

    def set_clip(self, clip_name: str) -> None:
        """Play a clip of the library; an unknown name changes nothing."""
        clip = self.library.get(clip_name)
        if clip is None:
            return
        self.drive = DRIVE_CLIP
        self.animator.play("base", clip)

    def use_state_machine(self) -> None:
        self.drive = DRIVE_FSM

    def use_motion_matching(self) -> None:
        if self.matcher is None:
            self.matcher = MotionMatcher(
                build_locomotion_database(self.skeleton, self.library))
        self.drive = DRIVE_MOTION_MATCH

    # -- by hand --------------------------------------------------------------
    def set_bone_rotation(self, bone: str, euler_deg) -> None:
        """Turn one bone by hand: degrees, YXZ, on top of whatever is playing."""
        q = m3.quat_from_euler(*np.deg2rad(np.asarray(euler_deg, dtype=float)),
                               order="YXZ")
        self.manual_pose.setdefault(bone, {})["rot"] = q
        self._pose_dirty = True

    def clear_manual(self) -> None:
        self.manual_pose = {}
        self._pose_dirty = True

    def mark_dirty(self) -> None:
        """Force one more evaluation — call after editing the pose from outside."""
        self._pose_dirty = True

    # -- per frame ------------------------------------------------------------
    @property
    def blending(self) -> bool:
        if self.drive == DRIVE_POSE:
            return self._blend_t < 1.0
        if self.drive == DRIVE_FSM:
            return self.fsm.blend < 1.0
        return False

    @property
    def animated(self) -> bool:
        """Whether :meth:`step` can return something new this frame."""
        return self.drive != DRIVE_POSE or self._blend_t < 1.0 or self._pose_dirty

    def grounded_key(self):
        """What is playing, if its support is not standing feet — else ``None``."""
        if self.drive == DRIVE_POSE:
            return ("pose", self.pose_name) if self.pose_name in GROUNDED_POSES else None
        if self.drive == DRIVE_CLIP:
            clip = getattr(self.animator.layer("base"), "clip", None)
            if clip is not None and GROUNDED_TAGS & set(getattr(clip, "tags", None) or ()):
                return ("clip", clip.name)
            return None
        if self.drive == DRIVE_FSM:
            state = self.fsm.states.get(self.fsm.current)
            if state is not None and state.tag in GROUNDED_TAGS:
                return ("fsm", state.name)
        return None

    def step(self, dt: float) -> Pose:
        """Advance by *dt* and return this frame's pose for :attr:`skeleton`.

        The source's pose, the bones turned by hand over it, and the standing
        rest under both (unless :attr:`apply_rest` is off).  The result is a
        new dict of new entries: a solver may edit it in place.
        """
        if self.drive == DRIVE_POSE:
            pose = self.base_pose
            if self._blend_t < 1.0:
                self._blend_t = min(1.0, self._blend_t + dt * self.blend_speed)
                k = self._blend_t
                pose = blend_poses(self._blend_from, self.base_pose,
                                   k * k * (3 - 2 * k))
            self._pose_dirty = False
        elif self.drive == DRIVE_CLIP:
            pose = self.animator.update(dt)
        elif self.drive == DRIVE_FSM:
            pose = self.fsm.update(dt)
        else:
            pose = self.matcher.update(dt) if self.matcher else {}
        pose = {bone: dict(entry) for bone, entry in pose.items()}
        for bone, entry in self.manual_pose.items():
            pose.setdefault(bone, {}).update(entry)
        self.authored_pose = pose
        out = compose_rest(self.rest, pose) if self.apply_rest else pose
        return {bone: dict(entry) for bone, entry in out.items()}
