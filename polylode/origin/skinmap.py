"""Transfer Illusion skin weights onto this project's skeleton.

Their rig has ~240 bones (``cf_j_*`` joints plus ``cf_s_*`` scale helpers); ours
has 85.  Mapping is two-stage: an explicit table for the main chain, then a
nearest-bone fallback driven by the reference rig's real bind positions, so
every one of their bones lands somewhere sensible without hand-listing all 240.
"""
from __future__ import annotations

import numpy as np


def skin_normals(mesh, deltas, bi, w, stray=None) -> None:
    """Carry the authored normals through the same blend as the positions.

    **Not** ``compute_normals()``, which is what both skinning paths used to
    end with.  Recomputing throws away the normals the artist shipped, and on a
    head those are not a derivative of the geometry — they are the drawing.
    Measured on ``cf_O_face``: authored against recomputed differ by a mean of
    16.1 degrees (p90 35.5, max 119.6), and 93 front-facing vertices — the
    outer cheek and jaw, ``|x|`` out to 0.112 — sit above ``N·L = 0.5`` with
    the authored normal and below it with the recomputed one.  ``_RampG``
    (``gt_ramp_01``) is a **binary step** at exactly that value, so those
    vertices do not shade slightly differently, they flip to the full shadow
    tint: the hard dark rim that ran down both sides of the jaw and the neck
    and made the face read thin (§7.61, §八 二 A3b).

    Unity blends normals with the same matrix as positions — the plain upper
    3x3, not its inverse transpose — so that is what this does, non-uniform
    face-shape scales included.
    """
    src = np.asarray(mesh.normals, dtype=np.float64)
    if src.shape[0] != mesh.vertex_count or not np.any(src):
        mesh.compute_normals()
        return
    out = np.zeros_like(src)
    for k in range(w.shape[1]):
        wk = w[:, k]
        if not np.any(wk):
            continue
        dk = deltas[bi[:, k]]
        out += np.einsum("nij,nj->ni", dk[:, :3, :3], src) * wk[:, None]
    if stray is not None and np.any(stray):
        out[stray] = src[stray]
    ln = np.linalg.norm(out, axis=1, keepdims=True)
    # A vertex whose blend cancelled out has no direction left to keep; falling
    # back to the authored one is closer than a zero that renders black.
    dead = (ln[:, 0] < 1e-9)
    if np.any(dead):
        out[dead] = src[dead]
        ln = np.linalg.norm(out, axis=1, keepdims=True)
    mesh.normals = (out / np.maximum(ln, 1e-9)).astype(np.float32)


# Illusion bone name -> ours, for everything where the name differs or where a
# whole subtree should collapse onto one of our joints.
EXPLICIT = {
    "cf_j_root": "cf_j_root", "cf_n_height": "cf_j_root",
    "cf_j_hips": "cf_j_hips", "cf_s_waist01": "cf_j_hips",
    "cf_s_waist02": "cf_j_spine01", "cf_j_waist01": "cf_j_hips",
    "cf_j_waist02": "cf_j_spine01",
    "cf_j_spine01": "cf_j_spine01", "cf_s_spine01": "cf_j_spine01",
    "cf_j_spine02": "cf_j_spine02", "cf_s_spine02": "cf_j_spine02",
    "cf_j_spine03": "cf_j_spine03", "cf_s_spine03": "cf_j_spine03",
    "cf_j_neck": "cf_j_neck", "cf_s_neck": "cf_j_neck",
    "cf_j_head": "cf_j_head", "cf_s_head": "cf_j_head",
    "cf_j_chest": "cf_j_spine03", "cf_s_chest": "cf_j_spine03",
}
for _s, _o in (("L", "L"), ("R", "R")):
    EXPLICIT.update({
        f"cf_j_shoulder_{_s}": f"cf_j_shoulder_{_o}",
        f"cf_s_shoulder_{_s}": f"cf_j_shoulder_{_o}",
        f"cf_j_arm00_{_s}": f"cf_j_arm00_{_o}",
        f"cf_s_arm01_{_s}": f"cf_j_arm00_{_o}",
        f"cf_s_arm02_{_s}": f"cf_j_arm00_{_o}",
        f"cf_s_arm03_{_s}": f"cf_j_forearm01_{_o}",
        f"cf_j_forearm01_{_s}": f"cf_j_forearm01_{_o}",
        f"cf_s_forearm01_{_s}": f"cf_j_forearm01_{_o}",
        f"cf_s_forearm02_{_s}": f"cf_j_forearm01_{_o}",
        f"cf_j_hand_{_s}": f"cf_j_hand_{_o}",
        f"cf_s_hand_{_s}": f"cf_j_hand_{_o}",
        f"cf_j_elbo_{_s}": f"cf_j_forearm01_{_o}",
        f"cf_j_wrist_{_s}": f"cf_j_hand_{_o}",
        f"cf_j_thigh00_{_s}": f"cf_j_thigh00_{_o}",
        f"cf_s_thigh01_{_s}": f"cf_j_thigh00_{_o}",
        f"cf_s_thigh02_{_s}": f"cf_j_thigh00_{_o}",
        f"cf_s_thigh03_{_s}": f"cf_j_leg01_{_o}",
        f"cf_j_leg01_{_s}": f"cf_j_leg01_{_o}",
        f"cf_s_leg01_{_s}": f"cf_j_leg01_{_o}",
        f"cf_s_leg02_{_s}": f"cf_j_leg01_{_o}",
        f"cf_s_leg03_{_s}": f"cf_j_leg03_{_o}",
        f"cf_s_kneeB_{_s}": f"cf_j_leg01_{_o}",
        f"cf_s_knee01_{_s}": f"cf_j_leg01_{_o}",
        f"cf_j_leg03_{_s}": f"cf_j_leg03_{_o}",
        f"cf_j_foot_{_s}": f"cf_j_leg03_{_o}",
        f"cf_j_toes_{_s}": f"cf_j_toes_{_o}",
        f"cf_j_ankle_{_s}": f"cf_j_leg03_{_o}",
        f"cf_j_bust00_{_s}": f"cf_j_bust00_{_o}",
        f"cf_j_bust01_{_s}": f"cf_j_bust01_{_o}",
        f"cf_j_bust02_{_s}": f"cf_j_bust01_{_o}",
        f"cf_j_bust03_{_s}": f"cf_j_bust01_{_o}",
        f"cf_d_bust00_{_s}": f"cf_j_bust00_{_o}",
        f"cf_s_bust00_{_s}": f"cf_j_bust00_{_o}",
    })

# their skirt rig fans out; ours has 8 columns x 3 rings
_SKIRT_COL = {"00": 0, "01": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 6, "07": 7}


# Illusion spells facial bones with a capital J (``cf_J_Eye_rz_L``) and body
# bones with a lowercase one (``cf_j_hips``).  The facial rig is authored in
# head-local space, so its bones sit near the origin in the reference armature
# and the nearest-bone fallback would match them against the feet.
#
# BetterPenetration / uncensor packs also use ``cf_J_Vagina_*`` / ``cf_J_Ana_*``
# / ``cf_J_Belly`` on the **body** (not head-local).  Treating those as face
# mapped every crotch weight onto ``cf_j_head`` and Free H could never open
# (§7.122).  Exclude the body BP prefixes first.
_BODY_CF_J_PREFIXES = (
    "cf_J_Vagina", "cf_J_Ana", "cf_J_Belly", "cf_J_Siri", "cf_J_Kokan",
    "cf_J_Kosi", "cf_J_Leg", "cf_J_Spine", "cf_J_Arm", "cf_J_Hand",
    "cf_J_Neck_s",  # careful: face has cf_J_Neck too — only _s midsection list
)


def _is_face_bone(name: str) -> bool:
    if any(name.startswith(p) for p in _BODY_CF_J_PREFIXES):
        return False
    return name.startswith(("cf_J_", "cf_j_hitomi", "cf_J_hitomi"))


def _skirt_map(name: str) -> str | None:
    if not name.startswith("cf_j_sk_"):
        return None
    rest = name[len("cf_j_sk_"):]
    col = rest[:2]
    idx = _SKIRT_COL.get(col)
    if idx is None:
        return None
    tail = rest[3:]
    ring = "root" if tail.startswith("0") and "00" in tail else "01"
    if "02" in tail or "03" in tail or "04" in tail:
        ring = "02"
    return f"cf_j_sk_{idx:02d}_{ring}"


class BoneMapper:
    """Illusion bone name -> index in our skeleton.

    When the skeleton has been grafted from the full Origin armature
    (``exact=True``, default after graft), this matches
    ``AssignedAnotherWeights``: **name equality only** — no collapse of
    ``cf_s_*`` helpers onto a coarser joint.
    """

    def __init__(self, skeleton, reference_rig, scale: float = 1.0,
                 *, exact: bool = False):
        self.skel = skeleton
        self.rig = reference_rig
        self.scale = float(scale)
        self.exact = bool(exact)
        self._cache: dict[str, int] = {}
        self._ref_cache: dict[int, np.ndarray | None] = {}
        self._our_names = [b.name for b in skeleton.bones]
        self._our_pos = np.array([b.world_pos for b in skeleton.bones]) if skeleton.bones else np.zeros((0, 3))

    # ------------------------------------------------------------------
    def index(self, origin_name: str) -> int:
        hit = self._cache.get(origin_name)
        if hit is not None:
            return hit
        idx = self._resolve(origin_name)
        self._cache[origin_name] = idx
        return idx

    def _resolve(self, origin_name: str) -> int:
        # AssignedAnotherWeights: exact name on the shared armature first
        if self.skel.has(origin_name):
            return self.skel.id(origin_name)
        if self.exact:
            # Face bones may be head-local-only; never dump them on the hips.
            if _is_face_bone(origin_name) and self.skel.has("cf_j_head"):
                return self.skel.id("cf_j_head")
            return self.skel.id("cf_j_hips") if self.skel.has("cf_j_hips") else 0

        target = EXPLICIT.get(origin_name)
        if target is None:
            target = _skirt_map(origin_name)
        if target is None and _is_face_bone(origin_name):
            target = "cf_j_head"
        if target and self.skel.has(target):
            return self.skel.id(target)
        p = self.rig.get(origin_name) if self.rig is not None else None
        if p is not None and len(self._our_pos):
            q = np.asarray(p, dtype=np.float64) * self.scale
            d = np.linalg.norm(self._our_pos - q, axis=1)
            return int(np.argmin(d))
        for name in self._our_names:
            if origin_name.startswith(name):
                return self.skel.id(name)
        return self.skel.id("cf_j_hips") if self.skel.has("cf_j_hips") else 0

    # ------------------------------------------------------------------
    def remap(self, bone_names: list[str], skin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """skin: (N,8) as [w0..w3, i0..i3] -> (indices (N,4), weights (N,4))."""
        lut = np.array([self.index(n) for n in bone_names], dtype=np.int32)
        w = skin[:, :4].astype(np.float64)
        i = skin[:, 4:].astype(np.int32)
        i = np.clip(i, 0, len(lut) - 1)
        mapped = lut[i]

        n = len(skin)
        out_i = np.zeros((n, 4), dtype=np.int32)
        out_w = np.zeros((n, 4), dtype=np.float32)
        for row in range(n):
            acc: dict[int, float] = {}
            for k in range(4):
                if w[row, k] <= 1e-6:
                    continue
                b = int(mapped[row, k])
                acc[b] = acc.get(b, 0.0) + float(w[row, k])
            if not acc:
                out_i[row, 0] = 0
                out_w[row, 0] = 1.0
                continue
            top = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
            total = sum(v for _, v in top) or 1.0
            for k, (b, v) in enumerate(top):
                out_i[row, k] = b
                out_w[row, k] = v / total
        return out_i, out_w

    def stats(self) -> dict:
        return {"mapped": len(self._cache)}

    # ------------------------------------------------------------------
    @staticmethod
    def _settle_chain(bone_world: list, strength: float,
                      min_reach: float) -> list:
        """Let a sway-bone chain hang instead of standing out rigidly.

        Trailing sleeves, ribbons and long hair are authored straight and rely
        on Illusion's dynamic bones to fall under gravity.  We resolve them in
        their rest shape, so without this a floor-length sleeve juts sideways
        from the wrist.  Each bone is rotated about the chain's root toward
        straight down, in proportion to how far along the chain it sits, which
        curves the chain rather than pivoting it as a rigid stick.
        """
        pts = [(i, np.asarray(m)[:3, 3]) for i, m in enumerate(bone_world)
               if m is not None]
        if len(pts) < 2 or strength <= 0.0:
            return bone_world
        root = pts[0][1]
        reach = np.array([np.linalg.norm(p - root) for _, p in pts])
        far = float(reach.max())
        if far < min_reach:
            return bone_world              # too short to read as hanging
        direction = pts[int(np.argmax(reach))][1] - root
        n = np.linalg.norm(direction)
        if n < 1e-6:
            return bone_world
        direction = direction / n
        down = np.array([0.0, -1.0, 0.0])
        axis = np.cross(direction, down)
        s = np.linalg.norm(axis)
        if s < 1e-4:
            return bone_world              # already hanging
        axis = axis / s
        total = float(np.arctan2(s, float(np.dot(direction, down)))) * strength

        out = list(bone_world)
        for (i, p), r in zip(pts, reach):
            angle = total * (r / far)
            c, si = np.cos(angle), np.sin(angle)
            K = np.array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
            R = np.eye(3) + si * K + (1.0 - c) * (K @ K)
            M = np.asarray(bone_world[i]).copy()
            M[:3, :3] = R @ M[:3, :3]
            M[:3, 3] = root + R @ (p - root)
            out[i] = M
        return out

    @staticmethod
    def rest_pose(mesh, skin: np.ndarray, bind_poses, bone_world,
                  settle: float = 0.0, min_reach: float = 0.16,
                  bone_rest=None, report: list | None = None,
                  label: str = "") -> bool:
        """Resolve a mesh into its own prefab's rest shape.

        For accessories rigged to their own sway bones there is nothing to
        retarget against -- the body armature has never heard of them.  Skinning
        the piece with its own bone transforms gives the shape the author
        intended, which can then simply be hung off the attach node.

        ``bone_rest`` is the same list after Illusion's own ``DynamicBone``
        solver has been run to steady state (``prefab.sway_rest``).  Where it
        has an entry, it wins: it is the shape the game's physics settles into,
        not an approximation of it.  ``_settle_chain`` stays for the bones no
        chain claims -- mod pieces whose sway components we could not read.

        A bone the renderer names but the track has no matrix for is dropped and
        the vertex's remaining weights re-normalised; a vertex with no other
        bone keeps the position the vertex buffer holds.  That is the least-bad
        thing to *draw*, and it was also completely silent -- which is what let
        ``_collect_worlds``'s depth ceiling hide for fifty sections (§7.96): the
        horse's hind legs stayed 1.2 m above the animal and nothing said a word,
        because untransformed geometry still looks like geometry.  ``report`` is
        ``attach_pieces``'s ``rejected`` list, i.e. ``DressReport.missing``
        (CLAUDE.md 一之一): if a piece comes out partly unposed, that is a
        missing transform, not a shape.
        """
        if skin is None or bind_poses is None or not bone_world:
            return False
        n = min(len(bind_poses), len(bone_world))
        if n == 0:
            return False
        if bone_rest:
            solved = [bone_rest[k] if k < len(bone_rest)
                      and bone_rest[k] is not None else None for k in range(n)]
            if any(m is not None for m in solved):
                bone_world = [solved[k] if solved[k] is not None
                              else bone_world[k] for k in range(n)]
                settle = 0.0
        if settle > 0.0:
            bone_world = BoneMapper._settle_chain(bone_world, settle, min_reach)
        deltas = np.zeros((n, 4, 4))
        usable = np.zeros(n, dtype=bool)
        for k in range(n):
            bw = bone_world[k]
            if bw is None:
                continue
            deltas[k] = np.asarray(bw) @ bind_poses[k]
            usable[k] = True
        if not usable.any():
            return False

        pos = mesh.positions.astype(np.float64)
        w = skin[:, :4].astype(np.float64)
        raw = skin[:, 4:].astype(np.int32)
        # An index past the end of the track is *not* the last bone.  Clamping
        # it there — which this did — binds the vertex to whatever happens to
        # sit at ``n - 1``, and that is a plausible-looking wrong answer of
        # exactly the kind §一之一 rules out.  Out of range means unusable, and
        # unusable is counted below.
        bi = np.clip(raw, 0, n - 1)
        ok = usable[bi] & (raw >= 0) & (raw < n) & (w > 1e-6)
        w = np.where(ok, w, 0.0)
        total = w.sum(axis=1, keepdims=True)
        stray = total[:, 0] < 1e-6
        w = np.divide(w, np.maximum(total, 1e-9))
        out = np.zeros_like(pos)
        for k in range(4):
            wk = w[:, k]
            if not np.any(wk):
                continue
            dk = deltas[bi[:, k]]
            out += (np.einsum("nij,nj->ni", dk[:, :3, :3], pos)
                    + dk[:, :3, 3]) * wk[:, None]
        if np.any(stray):
            out[stray] = pos[stray]
        mesh.positions = out.astype(np.float32)
        skin_normals(mesh, deltas, bi, w, stray)
        if report is not None:
            # Which bones the mesh actually leans on, not which ones the
            # renderer lists: a track may legitimately have a hole where no
            # vertex is weighted to that bone at all (the renderer naming
            # something outside this prefab), and that costs nothing.  What
            # matters is a bone that carries weight and has no matrix.
            weighted = np.zeros(n, dtype=bool)
            for k in range(4):
                idx = raw[:, k]
                inside = (idx >= 0) & (idx < n) & (skin[:, k] > 1e-6)
                if inside.any():
                    weighted[idx[inside]] = True
            lost = int(np.count_nonzero(weighted & ~usable))
            outside = int(np.count_nonzero(
                ((raw < 0) | (raw >= n)) & (skin[:, :4] > 1e-6)))
            if lost or outside or np.any(stray):
                bits = []
                if lost:
                    bits.append(f"{lost} 根有權重的骨頭沒有 transform")
                if outside:
                    bits.append(f"{outside} 筆權重指到骨頭表以外")
                if np.any(stray):
                    bits.append(f"{int(np.count_nonzero(stray))} 個頂點原地未擺放")
                report.append(f"{label or mesh.name}: " + "、".join(bits))
        return True

    def knows(self, bone_names) -> float:
        """Fraction of these bones the reference armature actually has."""
        if not bone_names or self.rig is None:
            return 0.0
        hit = sum(1 for b in bone_names if self.rig.matrix(b) is not None)
        return hit / float(len(bone_names))

    def _scaled(self, name: str) -> np.ndarray | None:
        if self.rig is None:
            return None
        m = self.rig.matrix(name)
        if m is None:
            return None
        m = m.copy()
        m[:3, 3] *= self.scale
        return m

    def to_our_scale(self, bind_world: np.ndarray) -> np.ndarray:
        """Finish the height rescale that the graft only half applies.

        Fitting Illusion's armature to our height scales bone *offsets*
        (``w[:3, 3] *= s``), which chains to ``[R | s·t]`` — the joints move
        closer together but each bone's own frame stays unit size.  A rest
        shape built from that is ``Σ w · [R | s·t] · bind⁻¹``: the skeleton
        shrinks and the flesh does not.  Limbs come out too thick for their
        length, and the head — whose 45 face bones all sit within a few
        centimetres of each other, so scaling their offsets barely moves
        anything — stays at full size on a smaller body.  That was the
        balloon-headed "squashed" look.

        A uniform scale is ``[s·R | s·t]``, so the 3×3 needs the same factor.
        Then ``bind_world⁻¹ · s·bind_world`` is exactly ``s·I``: every vertex
        keeps its bone-local direction and is scaled with the rig, and posing
        still works off the unscaled inverse binds.

        Public because a ``bind_world`` used as a *parent frame* needs the same
        correction for the same reason — hanging a prefab off one leaves the
        prefab at Illusion's native size on a rig that is not (§7.61).
        """
        out = np.asarray(bind_world, dtype=np.float64).copy()
        out[:3, :3] *= self.scale
        return out

    def _ref_for_our(self, idx: int) -> np.ndarray | None:
        """The reference matrix our bone ``idx`` was built from, if any."""
        hit = self._ref_cache.get(idx)
        if hit is not None or idx in self._ref_cache:
            return hit
        from .rigadopt import our_to_their
        m = None
        if self.rig is not None:
            their = our_to_their(self.skel.bones[idx].name, self.rig)
            if their is not None:
                m = self._scaled(their)
        self._ref_cache[idx] = m
        return m

    def _bone_world(self, origin_name: str) -> np.ndarray:
        """Where one of their bones sits in our bind pose.

        Their armature has ~240 bones and ours 85, so several of theirs share
        one of ours.  Using our bone's matrix directly for all of them puts
        every vertex at that one joint's position -- a chest vertex weighted to
        their waist helper ends up placed from our hip.  The remedy is to keep
        our (posed) bone as the frame and re-apply the offset between the bone
        it stands for and the bone actually asked for, which the reference
        armature knows for all 240.  Bones we have one-to-one reduce to our own
        matrix, so this costs nothing where the rigs already agree.

        Facial bones are the exception: their reference matrices are stored in
        head-local space, so they compose onto the head directly.
        """
        idx = self.index(origin_name)
        ours = self.skel.bones[idx].bind_world
        if self.rig is None:
            return ours
        actual = self._scaled(origin_name)
        if actual is None:
            return ours
        if _is_face_bone(origin_name):
            return ours @ actual
        ref = self._ref_for_our(idx)
        if ref is None:
            return ours
        try:
            return ours @ np.linalg.inv(ref) @ actual
        except np.linalg.LinAlgError:
            return ours

    # ------------------------------------------------------------------
    def retarget(self, mesh, bone_names: list[str], skin: np.ndarray,
                 bind_poses: np.ndarray | None = None,
                 *, native: bool = True) -> bool:
        """Place a garment authored on Illusion's rig onto ours.

        **Native path (default, matches the game)**
        ------------------------------------------
        Illusion never remaps bones.  Body and clothes share the full ~240-bone
        armature; vertices are ``Σ w · B_origin · bind · v``.  We do the same with
        the *reference* armature (every ``cf_s_*`` / helper kept), scale to our
        height, then only the *weights* are collapsed onto our 90-bone skeleton
        for animation.  Baking the rest shape from the full rig is what stops
        sailor tops from shredding when three scale-helpers are forced onto one
        arm bone.

        **Legacy path** (``native=False``)
        ----------------------------------
        ``B_ours · bind`` with the collapsed mapping — kept for comparison.
        """
        if not bone_names or bind_poses is None or len(bind_poses) == 0:
            return False
        n_bones = len(bone_names)
        binds = bind_poses
        deltas = np.zeros((n_bones, 4, 4))
        usable = np.zeros(n_bones, dtype=bool)

        if native:
            # AssignedWeights path: rest shape from the shared skeleton's bind
            # (full Origin graft → every name resolves 1:1).  Fall back to the
            # reference rig matrix only when a mod bone is absent.
            for k, name in enumerate(bone_names):
                if k >= len(binds):
                    continue
                if self.skel.has(name):
                    bw = self.skel.bones[self.skel.id(name)].bind_world
                else:
                    bw = self._scaled(name)
                    if bw is None:
                        bw = self.skel.bones[self.index(name)].bind_world
                    elif _is_face_bone(name) and self.skel.has("cf_j_head"):
                        head_w = self.skel.bones[self.skel.id("cf_j_head")].bind_world
                        face = bw.copy()
                        # ref face matrix may still be head-local
                        if abs(float(face[1, 3])) < 0.5:
                            bw = head_w @ face
                deltas[k] = self.to_our_scale(bw) @ binds[k]
                usable[k] = True
        else:
            for k, name in enumerate(bone_names):
                if k < len(binds):
                    deltas[k] = self._bone_world(name) @ binds[k]
                    usable[k] = True
        if not usable.any():
            return False

        pos = mesh.positions.astype(np.float64)
        w = skin[:, :4].astype(np.float64)
        bi = np.clip(skin[:, 4:].astype(np.int32), 0, n_bones - 1)
        ok = usable[bi] & (w > 1e-6)
        w = np.where(ok, w, 0.0)
        total = w.sum(axis=1, keepdims=True)
        stray = total[:, 0] < 1e-6
        w = np.divide(w, np.maximum(total, 1e-9))

        out = np.zeros_like(pos)
        for k in range(4):
            wk = w[:, k]
            if not np.any(wk):
                continue
            dk = deltas[bi[:, k]]
            out += (np.einsum("nij,nj->ni", dk[:, :3, :3], pos)
                    + dk[:, :3, 3]) * wk[:, None]
        if np.any(stray):                            # keep unweighted verts put
            out[stray] = pos[stray]
        mesh.positions = out.astype(np.float32)
        skin_normals(mesh, deltas, bi, w, stray)
        return True
