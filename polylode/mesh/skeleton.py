"""Humanoid skeleton, bind pose and skinning matrix evaluation.

Bone naming follows the ``cf_j_*`` convention used by Illusion's rigs so that
poses read familiarly, but the hierarchy is rebuilt procedurally from the
character's proportion sliders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import math3d as m3
from ..core.i18n import N_

# No hard armature cap: skinning uploads bones via a float texture
# (see gl/shaders.py).  Keep a soft ceiling only for pathological grafts.
MAX_BONES = 4096


@dataclass
class Bone:
    name: str
    parent: int = -1
    local_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    local_rot: np.ndarray = field(default_factory=m3.quat_identity)
    local_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float64))
    children: list = field(default_factory=list)
    # filled by Skeleton.finalize()
    bind_world: np.ndarray = field(default_factory=m3.identity)
    inv_bind: np.ndarray = field(default_factory=m3.identity)
    world_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    authored_local: np.ndarray | None = None


def affine_matrix(value) -> np.ndarray:
    """Validate and copy an absolute affine local transform, including shear."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("local matrix must be a finite 4x4 affine transform")
    bound = 64 * np.finfo(float).eps * max(1., float(np.abs(matrix).max()))
    if not np.allclose(matrix[3], [0., 0., 0., 1.], atol=bound, rtol=0):
        raise ValueError("local matrix must be affine, without perspective")
    return matrix.copy()


def posed_local(bone: Bone, entry: dict) -> np.ndarray:
    """An absolute matrix takes precedence; channel poses preserve an imported affine basis."""
    if "matrix" in entry:
        return affine_matrix(entry["matrix"])
    rotation = entry.get("rot")
    offset = entry.get("pos")
    scale = entry.get("scale")
    if bone.authored_local is not None:
        local = bone.authored_local.copy()
        if rotation is not None:
            rest = m3.quat_to_mat4(bone.local_rot)[:3, :3]
            turn = m3.quat_to_mat4(m3.quat_mul(bone.local_rot, rotation))[:3, :3]
            local[:3, :3] = turn @ rest.T @ local[:3, :3]
        if scale is not None:
            local[:3, :3] *= np.asarray(scale, dtype=np.float64)
        if offset is not None:
            local[:3, 3] += np.asarray(offset, dtype=np.float64)
        return local
    turn = bone.local_rot if rotation is None else m3.quat_mul(bone.local_rot, rotation)
    position = bone.local_pos if offset is None else bone.local_pos + np.asarray(offset, dtype=np.float64)
    size = bone.local_scale if scale is None else bone.local_scale * np.asarray(scale, dtype=np.float64)
    return m3.compose(position, turn, size)


def _quat_mul_many(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``m3.quat_mul`` row by row for ``(K, 4)`` quaternions."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], axis=1)


def _affine_many(values: list) -> np.ndarray:
    """:func:`affine_matrix` for K matrices at once: the same checks, the same messages."""
    try:
        matrices = np.asarray(values, dtype=np.float64)
    except ValueError:
        matrices = None
    if matrices is None or matrices.shape != (len(values), 4, 4) or not np.isfinite(matrices).all():
        raise ValueError("local matrix must be a finite 4x4 affine transform")
    bound = 64 * np.finfo(float).eps * np.maximum(1., np.abs(matrices).max(axis=(1, 2)))
    if (np.abs(matrices[:, 3] - _AFFINE_ROW) > bound[:, None]).any():
        raise ValueError("local matrix must be affine, without perspective")
    return matrices


def _posed_many(bones: list, entries: list) -> np.ndarray:
    """:func:`posed_local` for K bones and their pose entries at once: ``(K, 4, 4)``.

    A converted character's rig carries an imported affine basis on most of
    its bones and its stand-in hands over absolute matrices, so one bone at a
    time this was about half of an H frame.  Each step is :func:`posed_local`'s
    own: a matrix entry wins and is checked like :func:`affine_matrix`; an
    authored basis is turned by ``turn · restᵀ``, its columns scaled, its
    offset added — only for the entries that carry that channel.
    """
    out = np.empty((len(bones), 4, 4))
    given = [k for k, e in enumerate(entries) if "matrix" in e]
    if given:
        out[given] = _affine_many([entries[k]["matrix"] for k in given])
    rest = [k for k, e in enumerate(entries) if "matrix" not in e]
    if not rest:
        return out
    local = np.stack([bones[k].authored_local for k in rest])
    turned = [j for j, k in enumerate(rest) if entries[k].get("rot") is not None]
    if turned:
        base = np.asarray([bones[rest[j]].local_rot for j in turned], dtype=np.float64)
        by = np.asarray([entries[rest[j]]["rot"] for j in turned], dtype=np.float64)
        zero, one = np.zeros((len(turned), 3)), np.ones((len(turned), 3))
        before = _compose_many(zero, base, one)[:, :3, :3]
        after = _compose_many(zero, _quat_mul_many(base, by), one)[:, :3, :3]
        local[turned, :3, :3] = after @ before.transpose(0, 2, 1) @ local[turned, :3, :3]
    for j, k in enumerate(rest):
        scale, offset = entries[k].get("scale"), entries[k].get("pos")
        if scale is not None:
            local[j, :3, :3] *= np.asarray(scale, dtype=np.float64)
        if offset is not None:
            local[j, :3, 3] += np.asarray(offset, dtype=np.float64)
    out[rest] = local
    return out


def _inverse_bind(world: np.ndarray) -> np.ndarray:
    """Inverse of a bind matrix, including one a zero scale made singular.

    A zero scale on a bone is a real instruction, not bad data: it is how a
    card hides a limb (ABMX writes ``ScaleModifier = (0,0,0)`` on
    ``cf_j_thigh00_*`` to put a human torso on a horse), and Unity handles it
    because a mesh's ``bindposes`` are *authored* — fixed when the mesh was
    bound — so only the runtime matrix goes singular and the vertices collapse
    onto the bone's origin.  Here the bind pose is recomputed from the current
    rest, so the singular matrix reaches the inversion instead and
    ``np.linalg.inv`` raises, which failed the whole character.

    The pseudo-inverse of the linear part is the right substitute, and exactly
    right rather than approximately: with ``A⁺`` Moore-Penrose, ``W · B`` is
    the identity on ``{A u + t}`` — the affine set every vertex bound to this
    bone lies in, whatever the rank, from a flattened axis (``cf_J_ChinLow``
    scale ``(1,0,1)``) down to a full collapse onto the origin.  It is not a
    fallback value standing in for one we could not compute (§一之一): a
    singular matrix has no inverse, and this is what takes its place.
    """
    a = world[:3, :3]
    if abs(float(np.linalg.det(a))) > 1e-12:
        return m3.inverse(world)
    pinv = np.linalg.pinv(a)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = pinv
    out[:3, 3] = -pinv @ world[:3, 3]
    return out


_ONES = np.ones(3)
_NO_TURN = np.array([0.0, 0.0, 0.0, 1.0])
_AFFINE_ROW = np.array([0.0, 0.0, 0.0, 1.0])


def _compose_many(t: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``m3.compose`` for K translations, quaternions and scales at once: ``(K, 4, 4)``."""
    s = np.broadcast_to(s.reshape(len(q), -1), (len(q), 3))
    length = np.sqrt((q * q).sum(axis=1))
    unit = np.where((length > m3.EPS)[:, None], q / np.where(length > m3.EPS, length, 1.0)[:, None],
                    np.array([0.0, 0.0, 0.0, 1.0]))
    x, y, z, w = unit[:, 0], unit[:, 1], unit[:, 2], unit[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.zeros((len(q), 4, 4))
    m[:, 0, 0] = (1 - 2 * (yy + zz)) * s[:, 0]; m[:, 0, 1] = (2 * (xy - wz)) * s[:, 1]; m[:, 0, 2] = (2 * (xz + wy)) * s[:, 2]
    m[:, 1, 0] = (2 * (xy + wz)) * s[:, 0]; m[:, 1, 1] = (1 - 2 * (xx + zz)) * s[:, 1]; m[:, 1, 2] = (2 * (yz - wx)) * s[:, 2]
    m[:, 2, 0] = (2 * (xz - wy)) * s[:, 0]; m[:, 2, 1] = (2 * (yz + wx)) * s[:, 1]; m[:, 2, 2] = (1 - 2 * (xx + yy)) * s[:, 2]
    m[:, :3, 3] = t
    m[:, 3, 3] = 1.0
    return m


class Skeleton:
    def __init__(self, name: str = "skeleton"):
        self.name = name
        self.bones: list[Bone] = []
        self.index: dict[str, int] = {}
        #: The height slider's ``cf_n_height`` scale, which a graft bakes into the bone positions
        #: (:func:`origin.rig.graft_full_skeleton`): a clip's absolute translation is in the unscaled
        #: space and is multiplied by it (``hanim.absolute_pose_to_additive``).
        self.rig_scale: float = 1.0
        #: ``{bone: {axis: value}}`` the body shape writes every frame after the animation
        #: (``ShapeBodyInfoFemale.UpdateAlways``), in this project's space and unscaled; a clip's
        #: translation of that component is replaced by it (``hanim.absolute_pose_to_additive``).
        self.always_pos: dict = {}
        #: ``{bone: offset}`` added to a clip's absolute translation after the height scale
        #: (``hanim.absolute_pose_to_additive``): a converted character's own lengths on the source
        #: game's rig (``origin.converted_rig``).  Empty on a card, whose clips play unchanged.
        self.clip_offsets: dict = {}
        #: ``{clip goal: the joint it is moved with}`` on a converted character's rig
        #: (``origin.converted_rig.remap_goals``, read by ``origin.motion_ik.Body``).  Empty on a card.
        self.goal_joints: dict = {}
        #: bumped whenever a bone's rest changes (:meth:`finalize`, :meth:`set_rest_local`), so a
        #: reader that caches rest locals (``origin.motion_ik.Body``) knows to read them again
        self.rest_revision: int = 0
        #: ``[(bone, dynbone.Collider)]`` the shaft's own ``DynamicBoneCollider``s, in the hierarchy's
        #: order and at this rig's size (``body_head.graft_implant_bones``, §7.1187); empty without a pack.
        self.dan_colliders: list = []
        #: ``{bone index: (bind world or None, inverse bind)}`` set by :meth:`bind_skin`: the binds a
        #: mesh was skinned against where they are not the rest's.  :meth:`finalize` puts them back.
        self.skin_binds: dict = {}
        # ``world_position`` needs ``bind_world``, which only exists after
        # ``finalize``; adding a bone invalidates it again.
        self._finalized = False

    # -- build -------------------------------------------------------------
    def add(self, name: str, parent: str | None, pos, rot=None,
            scale=None) -> int:
        pi = self.index.get(parent, -1) if parent else -1
        b = Bone(name, pi, np.asarray(pos, dtype=np.float64).copy(),
                 m3.quat_identity() if rot is None else np.asarray(rot, dtype=np.float64).copy(),
                 np.ones(3) if scale is None else np.asarray(scale, dtype=np.float64).copy())
        i = len(self.bones)
        self.bones.append(b)
        self.index[name] = i
        self._finalized = False
        if pi >= 0:
            self.bones[pi].children.append(i)
        return i

    def add_world(self, name: str, parent: str | None, world_pos, rot=None) -> int:
        """Add a bone specified in world space (converted to parent-local)."""
        wp = np.asarray(world_pos, dtype=np.float64)
        if parent and parent in self.index:
            pw = self._chain_position(self.index[parent])
            return self.add(name, parent, wp - pw, rot)
        return self.add(name, parent, wp, rot)

    def add_matrix(self, name: str, parent: str | None, matrix) -> int:
        """Add an immutable authored rest matrix without dropping its affine basis.

        Pos/rot/scale properties describe it for existing readers; posing uses
        its exact matrix. Rest edits require replacing the authored matrix.
        """
        local = affine_matrix(matrix)
        position, rotation, scale = m3.decompose(local)
        index = self.add(name, parent, position, rotation, scale)
        self.bones[index].authored_local = local
        return index

    def _chain_position(self, i: int) -> np.ndarray:
        """Sum of ``local_pos`` up the chain — **translations only**.

        Correct while a skeleton is still being built, which is the one place it
        is used (:meth:`add_world` places a bone before anything has been
        finalised).  It is not a world position in general and must not be
        offered as one: see :meth:`world_position`.
        """
        p = np.zeros(3)
        while i >= 0:
            p = p + self.bones[i].local_pos
            i = self.bones[i].parent
        return p

    def world_position(self, name: str) -> np.ndarray:
        """Bind-pose world position.

        Taken from ``bind_world`` once the skeleton has been finalised, because
        summing ``local_pos`` up the chain — which is what this used to do —
        drops every rotation and scale on the way.  On the procedural scaffold
        the two agree, since nothing above a limb is rotated; on the **grafted
        reference armature** they do not, and that is the skeleton every body
        the editor's sliders can drive is built on.  Measured on
        ``cf_j_leg03_L``: the sum says z = −0.0166 where the bone is at
        −0.0398, so the foot was being built 23 mm forward of the ankle it
        hangs off (§7.169).  Every limb sweep and both hands go through here.
        """
        i = self.index[name]
        if self._finalized:
            return self.bones[i].bind_world[:3, 3].copy()
        return self._chain_position(i)

    def set_rest_local(self, i: int, pos, rot, scale) -> None:
        """Move bone *i*'s rest local to *pos* / *rot* / *scale* on a finalized rig (a live reshape).

        Only the local is rewritten: ``bind_world`` and the inverse binds stay
        those the meshes were skinned against.
        """
        b = self.bones[i]
        b.local_pos = np.array(pos, dtype=np.float64)
        b.local_rot = np.array(rot, dtype=np.float64)
        b.local_scale = np.array(scale, dtype=np.float64)
        self._local_bind[i] = m3.compose(b.local_pos, b.local_rot, b.local_scale)
        self.rest_revision += 1

    def bind_skin(self, i: int, inverse, world=None) -> None:
        """Bone *i*'s inverse bind (and with *world* its bind world) is what a mesh was skinned against, not the rest's.

        Kept across :meth:`finalize`, which otherwise derives every bind from
        the rest locals again: a converted character's rig is a card's rest at
        her own binds (``origin.converted_rig``), and an item hung on her hand
        in originact 8,0 re-finalized it onto the card's — her wrist skin
        stood 0.3 m from her hand, 16–44 times stretched, from that row on.
        """
        inverse = np.asarray(inverse, dtype=np.float64).copy()
        if world is None and i in self.skin_binds:
            world = self.skin_binds[i][0]                    # a bind world set before stays the bone's
        world = None if world is None else np.asarray(world, dtype=np.float64).copy()
        self.skin_binds[i] = (world, inverse)
        self._apply_skin_bind(i, world, inverse)

    def _apply_skin_bind(self, i: int, world, inverse) -> None:
        bone = self.bones[i]
        if world is not None:
            bone.bind_world = world.copy()
            bone.world_pos = bone.bind_world[:3, 3].copy()
        bone.inv_bind = inverse.copy()
        if getattr(self, "_inv_bind", None) is not None and i < len(self._inv_bind):
            self._inv_bind[i] = inverse
        self._rest_frames = None

    def finalize(self) -> Skeleton:
        self._finalized = True
        self.rest_revision += 1
        locals_ = []
        for i, b in enumerate(self.bones):
            scl = getattr(b, "local_scale", None)
            if scl is None:
                scl = np.ones(3)
            local = b.authored_local if b.authored_local is not None else m3.compose(b.local_pos, b.local_rot, scl)
            locals_.append(local)
            b.bind_world = local if b.parent < 0 else self.bones[b.parent].bind_world @ local
            b.inv_bind = _inverse_bind(b.bind_world)
            b.world_pos = b.bind_world[:3, 3].copy()
        # caches for the hot path: most bones carry no pose entry on a given
        # frame, so their local matrix is a constant we can look up
        n = len(self.bones)
        self._local_bind = np.stack(locals_) if n else np.zeros((0, 4, 4))
        self._rest_frames = None
        self._inv_bind = np.stack([b.inv_bind for b in self.bones]) if n \
            else np.zeros((0, 4, 4))
        self._parents = np.array([b.parent for b in self.bones], dtype=np.int64)
        self._names = [b.name for b in self.bones]
        # the bones by depth, for evaluating a whole level at a time
        depth = np.zeros(n, dtype=np.int64)
        for i, b in enumerate(self.bones):
            depth[i] = depth[b.parent] + 1 if b.parent >= 0 else 0
        self._levels = [np.flatnonzero(depth == level) for level in range(int(depth.max()) + 1)] if n else []
        for i, (world, inverse) in getattr(self, "skin_binds", {}).items():
            if i < n:
                self._apply_skin_bind(i, world, inverse)
        return self

    # -- queries -----------------------------------------------------------
    def __len__(self):
        return len(self.bones)

    def id(self, name: str, default: int = 0) -> int:
        return self.index.get(name, default)

    def has(self, name: str) -> bool:
        return name in self.index

    def positions(self) -> np.ndarray:
        return np.array([b.world_pos for b in self.bones])

    # The transform Illusion aims the character light off: RefObjKey.HeadParent,
    # which ChaReference.CreateReferenceInfo resolves by this name.  Named here
    # rather than at each call site so the renderer and the offline harnesses
    # cannot drift onto different bones.
    HEAD_LIGHT_BONE = "cf_s_head"

    def head_world(self, world: np.ndarray | None = None):
        """World matrix of :data:`HEAD_LIGHT_BONE`, or None if this rig has none.

        Falls back to the bind pose when no evaluated pose is handed in, which
        is what the still-frame harnesses render anyway.  A scaffold skeleton
        that never got a card grafted onto it has no such bone, and None then
        reads downstream as "not a character".
        """
        i = self.index.get(self.HEAD_LIGHT_BONE, -1)
        if i < 0:
            return None
        return self.bones[i].bind_world if world is None else world[i]

    def chain(self, name: str) -> list[int]:
        out, i = [], self.index[name]
        while i >= 0:
            out.append(i)
            i = self.bones[i].parent
        return out[::-1]

    def descendants(self, name: str) -> list[int]:
        out, stack = [], [self.index[name]]
        while stack:
            i = stack.pop()
            out.append(i)
            stack.extend(self.bones[i].children)
        return out

    # -- evaluation --------------------------------------------------------
    def evaluate(self, pose: dict | None = None, only=None) -> np.ndarray:
        """Return world matrices (N,4,4) for the given pose.

        ``pose`` maps bone name -> dict with optional keys ``rot`` (quaternion),
        ``pos`` (local translation offset) and ``scale`` (vec3 or float).

        *only* (bone indices) restricts the work to those bones and their
        ancestors: the same arithmetic on the same rows, so what it does
        compute is the full evaluation's to within the last bit of a float
        (the full one multiplies a level at a time, :meth:`_evaluate_all`), and
        **every other row is left uninitialised** — the caller reads only what
        it asked for.  A bone's
        parent always has the lower index (the loop below already relies on
        it), so the closure is walked in index order.
        """
        n = len(self.bones)
        if only is None and len(getattr(self, "_levels", ())) and len(self._local_bind) == n:
            return self._evaluate_all(pose or {})
        world = np.empty((n, 4, 4))
        pose = pose or {}
        locals_ = self._local_bind
        parents = self._parents
        names = self._names
        rows = range(n)
        if only is not None:
            need: set = set()
            for j in only:
                j = int(j)
                while j >= 0 and j not in need:
                    need.add(j)
                    j = int(parents[j])
            rows = sorted(need)
        posed = self._posed_local
        for i in rows:
            e = pose.get(names[i]) if pose else None
            local = locals_[i] if e is None else posed(i, e)
            p = parents[i]
            world[i] = local if p < 0 else world[p] @ local
        return world

    #: :meth:`_posed_local` keeps at most this many locals before it starts again
    POSED_LOCALS_KEPT = 8192

    def _posed_local(self, i: int, e: dict) -> np.ndarray:
        """:meth:`local_of` for the partial :meth:`evaluate`, kept by what the entry holds.

        A converted pair's solved H frame evaluated a short chain over a
        hundred times (the arm IK, the aim, the hands, PROGRESS #152): about
        1,100 posed locals a frame, 94 % of them a bone whose entry had not
        changed since the chain before, each a quaternion product and a
        compose.  The same entry gives the same matrix, bit for bit: kept by
        bone and by the bytes of the four channels :meth:`local_of` reads, for
        as long as the rest it was made from (``rest_revision``, the bone's
        rest arrays; an edit in place that no :meth:`finalize` follows is not
        seen here, as it is not by the unposed rows, ``_local_bind``).  Read
        only by :meth:`evaluate`, which copies it into the world it builds.
        """
        b = self.bones[i]
        key = [i, id(b.local_pos), id(b.local_rot), id(getattr(b, "local_scale", None)), id(b.authored_local)]
        for channel in ("matrix", "rot", "pos", "scale"):
            value = e.get(channel)
            if value is None:
                key.append(None)
            else:
                value = np.asarray(value, dtype=np.float64)
                key.append((value.shape, value.tobytes()))
        key = tuple(key)
        memo = self.__dict__.get("_posed_memo")
        if memo is None or memo[0] != self.rest_revision or len(memo[1]) >= self.POSED_LOCALS_KEPT:
            memo = self._posed_memo = (self.rest_revision, {})
        local = memo[1].get(key)
        if local is None:
            local = memo[1][key] = self.local_of(i, e)
        return local

    def local_of(self, i: int, e: dict) -> np.ndarray:
        """Bone *i*'s local matrix under the pose entry *e* (what :meth:`evaluate` multiplies down)."""
        b = self.bones[i]
        if "matrix" in e or b.authored_local is not None:
            return posed_local(b, e)
        rot = e.get("rot")
        q = b.local_rot if rot is None else m3.quat_mul(b.local_rot, rot)
        off = e.get("pos")
        t = b.local_pos if off is None else b.local_pos + np.asarray(off, dtype=np.float64)
        base_s = getattr(b, "local_scale", np.ones(3))
        s = e.get("scale", None)
        s = base_s if s is None else np.asarray(s, dtype=np.float64) * base_s
        return m3.compose(t, q, s)

    def _evaluate_all(self, pose: dict) -> np.ndarray:
        """Every bone's world matrix, the posed locals built together and a level multiplied at a time.

        The same arithmetic as the loop in :meth:`evaluate`, element for
        element; a skeleton of several hundred bones spent most of a frame in
        that loop's per-bone calls (§7.964).
        """
        local = self._local_bind
        if pose:
            index = self.index
            rows, rest, rot, pos, scale = [], [], [], [], []
            affine_rows, affine_bones, affine_entries = [], [], []
            for name, e in pose.items():
                i = index.get(name)
                if i is None or e is None:
                    continue
                b = self.bones[i]
                if "matrix" in e or b.authored_local is not None:
                    affine_rows.append(i)
                    affine_bones.append(b)
                    affine_entries.append(e)
                    continue
                r = e.get("rot")
                off = e.get("pos")
                sc = e.get("scale", None)
                base = getattr(b, "local_scale", _ONES)
                rows.append(i)
                rest.append(b.local_rot)
                rot.append(_NO_TURN if r is None else r)
                pos.append(b.local_pos if off is None else b.local_pos + np.asarray(off, dtype=np.float64))
                scale.append(base if sc is None else np.asarray(sc, dtype=np.float64) * base)
            if rows:
                a, b = np.asarray(rest, dtype=np.float64), np.asarray(rot, dtype=np.float64)
                ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
                bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
                turned = np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                                   aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], axis=1)
                local = local.copy()
                local[rows] = _compose_many(np.asarray(pos, dtype=np.float64), turned,
                                            np.asarray(scale, dtype=np.float64))
            if affine_rows:
                if not rows:
                    local = local.copy()
                local[affine_rows] = _posed_many(affine_bones, affine_entries)
        world = np.empty_like(local)
        parents = self._parents
        levels = self._levels
        world[levels[0]] = local[levels[0]]
        for level in levels[1:]:
            world[level] = world[parents[level]] @ local[level]
        return world

    def skinning_matrices(self, pose: dict | None = None,
                          world: np.ndarray | None = None) -> np.ndarray:
        if world is None:
            world = self.evaluate(pose)
        return world @ self._inv_bind

    def rest_frames(self) -> np.ndarray:
        """Each bone's rest world without its scale, (N, 4, 4); computed once.

        The frame a dual-quaternion palette splits a scaled bone's motion in
        (``mesh/skinning.py``).
        """
        frames = getattr(self, "_rest_frames", None)
        if frames is None or len(frames) != len(self.bones):
            from .skinning import rest_frames
            frames = rest_frames(np.stack([b.bind_world for b in self.bones])) if self.bones \
                else np.zeros((0, 4, 4))
            self._rest_frames = frames
        return frames

    def world_positions(self, pose: dict | None = None) -> np.ndarray:
        return self.evaluate(pose)[:, :3, 3]

    def __repr__(self):
        return f"<Skeleton {self.name} bones={len(self.bones)}>"


# --------------------------------------------------------------------------
# proportions
# --------------------------------------------------------------------------
# Proportions of the reference bind figure, measured in §7.160.  Reproduce with
#
#   python tools/body_profile.py --part trunk --span landmark --slices 24
#   python tools/body_profile.py --limb leg --stations 12
#
# What is written down here is a **table of measurements** — ratios read off a
# figure with a ruler — and the geometry below is generated from them, not
# resampled from anything.  No identifier, table or expression is transcribed;
# the bone names in the right-hand column say *where each number was taken*, so
# that any of them can be re-measured and disagreed with (``著作權.md`` 四:
# numbers, bone names and field names are facts, vertices and keyframes are
# expression, and only the first kind is ours to carry).
#
# The unit is **stature** — sole to the top of the head mesh, 1.7113 m on the
# stock bind figure — and that choice is the whole point.  The obvious
# denominator, the body renderer's own height, is 0.8826 of stature because
# ``o_body_a`` stops at the neck, so a table normalised by it is 13% out of
# step with any figure that has a head.  §7.159 read our stack as "11–16%
# below Origin above the knee" on exactly that mismatch: one column was divided by
# stature and the other was not.  Measured against the same denominator the
# real gaps were different things entirely — the shoulders were 48% too wide,
# the forearm 30% too long, the foot 28% too long, and the knee was fine.
#
# Heights are joints (the ``cf_s_*`` shape bones the body is skinned to; the
# ``cf_j_*`` joints carry no vertices and are absent from the body renderer).
# ``crotch`` and ``trunk_top`` are the exceptions and are read off the
# **surface**: no bone marks either, and they are where the torso begins and
# ends.  Both sexes measure the same here — Origin's male/female difference is in
# the shape sliders, applied after this (§7.153).
# The reference figure's own stature in metres.  Needed as a number, not just
# as the unit the table is in: grafting the reference armature to a requested
# height is ``height / BIND_STATURE_M``, and guessing that ratio anywhere else
# would put the skin and the bones at different sizes.
BIND_STATURE_M = 1.7113

# Where the **sole** sits on the armature, of stature.  Measured on Origin
# (§7.215): in Studio the character's root and ``cf_j_root`` are both at world
# y = 0 and ``o_body_a``'s lowest vertex is at +0.00850 — so Origin's bind sole is
# 14.5 mm off the floor and the toe bones are 26.6 mm up.  ``BIND_REF`` below is
# measured **from the sole**, so every height built on it has to start here.
SOLE_Y = 0.00850

BIND_REF = {
    "ankle": 0.0403,        # cf_s_leg03
    "knee": 0.2944,         # cf_s_leg01
    "thigh": 0.5631,        # cf_s_thigh01 — the hip joint
    "crotch": 0.4762,       # surface: the lowest section with **no midline gap
                            # wider than a slab** — measured that way, and it is
                            # **not** "the lowest section that is not two legs",
                            # which is what this line used to claim: cut the
                            # triangles there and Origin's two thighs are still
                            # 15.7 mm apart (§7.220).  Nothing reads it now.
    "junction": 0.5315,     # surface: the lowest section where **one closed
                            # loop reaches both sides of the midline** — i.e.
                            # the lowest height at which a single ring can
                            # exist at all.  Topological, so it needs no
                            # threshold; the thighs touch before they merge and
                            # Origin's gap runs 1.2 / 0.1 / 0.04 / 0.01 mm over four
                            # consecutive slices (§7.222).
                            #   python tools/body_profile.py --junction
    "junction_m": 0.5265,   # the same on ``body_m``.  The two bodies agree to
                            # the fifth decimal below the chest (§7.176), but
                            # not here: 8.6 mm, and it is the one height the
                            # loft's base has to land on.
    "trunk_base": 0.5144,   # surface: the lowest vertex of the trunk group —
                            # **not** the crotch.  Below this the body is two
                            # thighs and the trunk does not exist, so a loft
                            # taken down to the crotch has to close over the
                            # gap and closes it with a flat slab.
    "pelvis": 0.5777,       # cf_s_waist02
    "spine1_at": 0.4466,    # cf_s_spine01, as a fraction of pelvis→chest
    "spine2_at": 0.7233,    # cf_s_spine02
    "chest": 0.7678,        # cf_s_spine03
    "shoulder": 0.8173,     # cf_s_shoulder02 / cf_s_arm01 (same height, T-pose)
    "neck": 0.8391,         # cf_s_neck
    "trunk_top": 0.8711,    # the highest vertex of the trunk **bone group** —
                            # **not** the surface, whatever this line used to
                            # say.  Same slip as ``trunk_base`` at the other end
                            # (§7.224): the neck's top skin is weighted to
                            # ``cf_s_head`` and so is not in the group, and the
                            # surface goes on to 0.8825.  Read as a height it is
                            # neither end of anything real — it sits 11 mm above
                            # the last closed ring and 20 mm below the rim's top.
                            # Kept because ``TORSO_PROFILE``'s t is measured
                            # against it and that parameterisation is fine; what
                            # must not be built up to it is the loft.
    "neck_cap": 0.8646,     # surface, topological: the **highest** section that
                            # is still one closed ring — i.e. the lowest point of
                            # the neck hole's rim.  Above it every horizontal cut
                            # is an open arc (2 open ends, every slice), so a
                            # ring loft cannot describe it and stops here; the
                            # remaining 30 mm of slant is ``body.NECK_RIM``.
                            # Needs no threshold, exactly like ``junction``.
                            #   python tools/body_profile.py --neck
    "head": 0.8770,         # cf_s_head — where the head renderer is parented
    "head_len": 0.1230,     # that bone to the top of the head mesh
    "hip_x": 0.0485,        # cf_s_thigh01
    "shoulder_x": 0.0641,   # cf_s_shoulder02
    "arm_x": 0.0757,        # cf_s_arm01 — where the arm sweep starts
    "upper_arm": 0.1361,    # cf_s_arm01 → cf_s_forearm01
    "fore_arm": 0.1122,     # cf_s_forearm01 → cf_s_wrist
    "hand": 0.1106,         # wrist to fingertip, off the hand segment
    "foot": 0.1059,         # heel to toe, off the foot segment
}


@dataclass
class Proportions:
    """Skeleton-level measurements, all in metres, derived from sliders."""
    height: float = 1.58
    head_size: float = 1.0
    neck_len: float = 1.0
    shoulder_width: float = 1.0
    shoulder_height: float = 0.0
    arm_len: float = 1.0
    forearm_len: float = 1.0
    hand_size: float = 1.0
    leg_len: float = 1.0
    thigh_len: float = 1.0
    calf_len: float = 1.0
    foot_size: float = 1.0
    hip_width: float = 1.0
    spine_len: float = 1.0
    arm_angle: float = 42.0          # degrees from vertical, A-pose
    leg_spread: float = 3.0
    bust: float = 0.5

    # -- derived landmarks -------------------------------------------------
    def scale(self) -> float:
        return self.height / 1.58

    def landmarks(self) -> dict:
        """Where every joint sits, as measured fractions of stature × height.

        See :data:`BIND_REF`.  Sliders scale the **segments**, not the absolute
        heights, so a longer calf pushes the knee and everything above it up,
        which is what a longer calf does.
        """
        s = self.scale()
        h = self.height
        K = BIND_REF
        head_len = K["head_len"] * h * self.head_size
        # vertical stack.  **From the sole, not from the armature's origin.**
        # ``BIND_REF`` was measured as ``(y - sole) / stature`` on Origin, and Origin's
        # sole is not on the root: in Studio ``chaF_001`` and ``cf_j_root`` are
        # both at world y = 0 while ``o_body_a``'s lowest vertex is 0.00850 of
        # stature up (§7.215).  Starting the stack at 0 therefore put every
        # height derived from this table 14.5 mm low on the shared armature.
        # It was self-consistent — the foot was low by the same amount, so
        # anything measured "from our own sole" agreed — which is why it
        # survived: a global term is invisible to every ruler that normalises
        # it away (§7.98, §7.153, §7.206, §7.213).
        ankle_y = (K["ankle"] + SOLE_Y) * h
        knee_y = ankle_y + (K["knee"] - K["ankle"]) * h * self.calf_len * self.leg_len
        hip_y = knee_y + (K["thigh"] - K["knee"]) * h * self.thigh_len * self.leg_len
        pelvis_y = hip_y + (K["pelvis"] - K["thigh"]) * h
        spine_span = (K["chest"] - K["pelvis"]) * h * self.spine_len
        spine1_y = pelvis_y + spine_span * K["spine1_at"]
        spine2_y = pelvis_y + spine_span * K["spine2_at"]
        chest_y = pelvis_y + spine_span
        neck_y = chest_y + (K["neck"] - K["chest"]) * h * self.neck_len
        head_y = neck_y + (K["head"] - K["neck"]) * h * self.neck_len
        top_y = head_y + head_len
        # The crotch and the top of the neck stub are read off the **surface**,
        # not off a joint, because they are where the torso loft starts and
        # stops and no bone marks either of them.
        crotch_y = (K["crotch"] + SOLE_Y) * h
        trunk_base_y = (K["trunk_base"] + SOLE_Y) * h
        junction_y = ((K["junction"] + SOLE_Y) * h,
                      (K["junction_m"] + SOLE_Y) * h)
        trunk_top_y = neck_y + (K["trunk_top"] - K["neck"]) * h * self.neck_len
        # The loft's own top, which is a different height from the table's t = 1
        # (see ``neck_cap``).  Scaled off ``neck_y`` like ``trunk_top_y`` because
        # it is neck: the slider that lengthens the neck lengthens this too.
        neck_cap_y = neck_y + (K["neck_cap"] - K["neck"]) * h * self.neck_len

        hip_x = K["hip_x"] * h * self.hip_width
        sh_x = K["arm_x"] * h * self.shoulder_width
        sh_y = chest_y + (K["shoulder"] - K["chest"]) * h + 0.03 * s * self.shoulder_height
        return dict(
            s=s, head_len=head_len, ankle_y=ankle_y, knee_y=knee_y, hip_y=hip_y,
            pelvis_y=pelvis_y, spine1_y=spine1_y, spine2_y=spine2_y, chest_y=chest_y,
            neck_y=neck_y, head_y=head_y, top_y=top_y, hip_x=hip_x, sh_x=sh_x, sh_y=sh_y,
            crotch_y=crotch_y, trunk_base_y=trunk_base_y,
            junction_y=junction_y,
            trunk_top_y=trunk_top_y, neck_cap_y=neck_cap_y,
            neck_scale=h * self.neck_len,
            # ``shoulder_x`` used to be handed back here and **nothing read it**
            # (§7.267) — the whole project, tools included.  A key in a dict
            # every generator opens reads as wiring; it was arithmetic on
            # ``BIND_REF["shoulder_x"]``, which stays because it is a
            # measurement (``cf_s_shoulder02``), unlike its derived value.
            upper_arm=K["upper_arm"] * h * self.arm_len,
            fore_arm=K["fore_arm"] * h * self.arm_len * self.forearm_len,
            hand=K["hand"] * h * self.hand_size,
            foot=K["foot"] * h * self.foot_size,
        )


FINGERS = [("thumb", 3), ("index", 3), ("middle", 3), ("ring", 3), ("little", 3)]


def build_humanoid(prop: Proportions | None = None) -> Skeleton:
    """Construct a full humanoid rig (root, spine, arms with fingers, legs, bust)."""
    p = prop or Proportions()
    L = p.landmarks()
    s = L["s"]
    sk = Skeleton("humanoid")

    sk.add("cf_j_root", None, (0, 0, 0))
    sk.add("cf_j_hips", "cf_j_root", (0, L["pelvis_y"], 0))
    sk.add("cf_j_spine01", "cf_j_hips", (0, L["spine1_y"] - L["pelvis_y"], 0))
    sk.add("cf_j_spine02", "cf_j_spine01", (0, L["spine2_y"] - L["spine1_y"], 0))
    sk.add("cf_j_spine03", "cf_j_spine02", (0, L["chest_y"] - L["spine2_y"], 0))
    sk.add("cf_j_neck", "cf_j_spine03", (0, L["neck_y"] - L["chest_y"], 0))
    sk.add("cf_j_head", "cf_j_neck", (0, L["head_y"] - L["neck_y"], 0))
    sk.add("cf_j_head_top", "cf_j_head", (0, L["head_len"], 0))

    # eyes / jaw helpers (used for expressions)
    hs = L["head_len"]
    sk.add("cf_j_eye_L", "cf_j_head", (0.038 * s * p.head_size, hs * 0.42, hs * 0.40))
    sk.add("cf_j_eye_R", "cf_j_head", (-0.038 * s * p.head_size, hs * 0.42, hs * 0.40))
    sk.add("cf_j_jaw", "cf_j_head", (0, hs * 0.15, hs * 0.05))

    # bust chain (secondary deformation for the chest)
    bust_y = L["chest_y"] - L["spine2_y"] + 0.012 * s
    for side, sgn in (("L", 1), ("R", -1)):
        sk.add(f"cf_j_bust00_{side}", "cf_j_spine02",
               (sgn * 0.058 * s, bust_y, 0.030 * s))
        sk.add(f"cf_j_bust01_{side}", f"cf_j_bust00_{side}", (0, 0, 0.02 * s))

    # arms
    ang = np.deg2rad(p.arm_angle)
    d = np.array([np.sin(ang), -np.cos(ang), 0.0])
    for side, sgn in (("L", 1), ("R", -1)):
        sk.add("cf_j_shoulder_" + side, "cf_j_spine03",
               (sgn * 0.045 * s, L["sh_y"] - L["chest_y"], 0))
        sk.add("cf_j_arm00_" + side, "cf_j_shoulder_" + side,
               (sgn * (L["sh_x"] - 0.045 * s), 0, 0))
        sk.add("cf_j_forearm01_" + side, "cf_j_arm00_" + side,
               (sgn * d[0] * L["upper_arm"], d[1] * L["upper_arm"], 0))
        sk.add("cf_j_hand_" + side, "cf_j_forearm01_" + side,
               (sgn * d[0] * L["fore_arm"], d[1] * L["fore_arm"], 0))
        _add_fingers(sk, side, sgn, d, L, s, p)

    # legs
    for side, sgn in (("L", 1), ("R", -1)):
        spread = np.deg2rad(p.leg_spread)
        sk.add("cf_j_thigh00_" + side, "cf_j_hips",
               (sgn * L["hip_x"], L["hip_y"] - L["pelvis_y"], 0))
        th = L["hip_y"] - L["knee_y"]
        sk.add("cf_j_leg01_" + side, "cf_j_thigh00_" + side,
               (sgn * th * np.sin(spread), -th, 0.004 * s))
        cf = L["knee_y"] - L["ankle_y"]
        sk.add("cf_j_leg03_" + side, "cf_j_leg01_" + side,
               (sgn * cf * np.sin(spread * 0.4), -cf, -0.010 * s))
        sk.add("cf_j_toes_" + side, "cf_j_leg03_" + side,
               (0, -L["ankle_y"] * 0.45, L["foot"] * 0.52))

    # genital chain (procedural)
    # Female: between thighs.  Local +Y = toward head, +Z = out of crotch.
    # Order along +Y (matches spread-leg illustration top→bottom reversed in Y):
    #   clitoris (superior) → vaginal entrance → anus (inferior / toward butt).
    sk.add("cf_j_kokan", "cf_j_hips", (0.0, -0.072 * s, 0.052 * s))
    sk.add("cf_j_clitoris", "cf_j_kokan", (0.0, 0.018 * s, 0.012 * s))
    sk.add("cf_j_anus", "cf_j_kokan", (0.0, -0.024 * s, -0.006 * s))
    # Male: pubic front, single shaft along +Z
    sk.add("cf_j_dan00", "cf_j_hips", (0.0, -0.042 * s, 0.092 * s))
    sk.add("cf_j_dan01", "cf_j_dan00", (0.0, -0.006 * s, 0.058 * s))
    sk.add("cf_j_dan02", "cf_j_dan01", (0.0, -0.004 * s, 0.052 * s))

    # skirt / coat helper bones (8 around the hips)
    for i in range(8):
        a = i / 8.0 * 2 * np.pi
        sk.add(f"cf_j_sk_{i:02d}_root", "cf_j_hips",
               (np.sin(a) * L["hip_x"] * 0.9, -0.02 * s, np.cos(a) * L["hip_x"] * 0.75))
        sk.add(f"cf_j_sk_{i:02d}_01", f"cf_j_sk_{i:02d}_root", (0, -0.10 * s, 0))
        sk.add(f"cf_j_sk_{i:02d}_02", f"cf_j_sk_{i:02d}_01", (0, -0.10 * s, 0))

    return sk.finalize()


def _add_fingers(sk: Skeleton, side: str, sgn: int, d: np.ndarray, L: dict,
                 s: float, p: Proportions) -> None:
    hand = L["hand"]
    palm = hand * 0.45
    perp = np.array([-d[1], d[0], 0.0])   # in-plane perpendicular, points "outward"
    spacing = 0.019 * s * p.hand_size
    for fi, (fname, segs) in enumerate(FINGERS):
        if fname == "thumb":
            root = (sgn * (d[0] * palm * 0.35 + perp[0] * -0.012 * s),
                    d[1] * palm * 0.35 + perp[1] * -0.012 * s, 0.020 * s)
            seg_len = hand * 0.17
        else:
            off = (fi - 2.5) * spacing
            root = (sgn * (d[0] * palm + perp[0] * off),
                    d[1] * palm + perp[1] * off, off * 0.15)
            seg_len = hand * (0.19 if fname in ("middle", "index") else 0.165)
        parent = f"cf_j_hand_{side}"
        prev = None
        for si in range(segs):
            bname = f"cf_j_{fname}{si + 1:02d}_{side}"
            if si == 0:
                sk.add(bname, parent, root)
            else:
                sk.add(bname, prev, (sgn * d[0] * seg_len, d[1] * seg_len, 0))
                seg_len *= 0.78
            prev = bname


# --------------------------------------------------------------------------
# convenience groupings used by the pose UI
# --------------------------------------------------------------------------
BONE_GROUPS = {
    "body": ["cf_j_hips", "cf_j_spine01", "cf_j_spine02", "cf_j_spine03",
             "cf_j_neck", "cf_j_head"],
    "arm_l": ["cf_j_shoulder_L", "cf_j_arm00_L", "cf_j_forearm01_L", "cf_j_hand_L"],
    "arm_r": ["cf_j_shoulder_R", "cf_j_arm00_R", "cf_j_forearm01_R", "cf_j_hand_R"],
    "leg_l": ["cf_j_thigh00_L", "cf_j_leg01_L", "cf_j_leg03_L", "cf_j_toes_L"],
    "leg_r": ["cf_j_thigh00_R", "cf_j_leg01_R", "cf_j_leg03_R", "cf_j_toes_R"],
    "bust": ["cf_j_bust00_L", "cf_j_bust01_L", "cf_j_bust00_R", "cf_j_bust01_R"],
}

BONE_LABELS = {
    "cf_j_hips": N_("腰"), "cf_j_spine01": N_("下脊椎"), "cf_j_spine02": N_("中脊椎"),
    "cf_j_spine03": N_("胸"), "cf_j_neck": N_("頸"), "cf_j_head": N_("頭"),
    "cf_j_shoulder_L": N_("左肩"), "cf_j_arm00_L": N_("左上臂"),
    "cf_j_forearm01_L": N_("左前臂"), "cf_j_hand_L": N_("左手"),
    "cf_j_shoulder_R": N_("右肩"), "cf_j_arm00_R": N_("右上臂"),
    "cf_j_forearm01_R": N_("右前臂"), "cf_j_hand_R": N_("右手"),
    "cf_j_thigh00_L": N_("左大腿"), "cf_j_leg01_L": N_("左小腿"),
    "cf_j_leg03_L": N_("左腳"), "cf_j_toes_L": N_("左腳趾"),
    "cf_j_thigh00_R": N_("右大腿"), "cf_j_leg01_R": N_("右小腿"),
    "cf_j_leg03_R": N_("右腳"), "cf_j_toes_R": N_("右腳趾"),
}
