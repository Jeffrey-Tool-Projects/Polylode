"""Extract Illusion's reference armature from ``chara/oo_base.unity3d``.

The base body bundle carries the whole rig: ~240 ``cf_*`` bones and the 54
``a_n_*`` accessory attach nodes.  Reading their real bind-pose hierarchy is
what lets clothes load the same way ChaControl does — one shared skeleton,
weights assigned by bone *name* (AssignedAnotherWeights), no collapse.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core import math3d as m3
from ..core import swallowed
from ..core.paths import CONFIG_DIR, write_json_atomic

CACHE = CONFIG_DIR / "origin_rig.json"
# 4: the ``p_tang2`` tongue is keyed under its own scope (``ATTACHED_ROOTS``)
# 5: an attached prefab's chain at its Animator's one load frame (``load_frame_time``)
CACHE_VERSION = 5
BASE_BUNDLE = "chara/oo_base.unity3d"

# ``ChaControl.<LoadAsync>`` instantiates ``p_cf_body_bone`` as ``objBodyBone``
# and that *is* the character's skeleton: every mesh loaded afterwards has its
# ``SkinnedMeshRenderer.bones`` re-pointed at this armature by bone **name**
# (``AssignedAnotherWeights.AssignedWeightsLoop``), while its ``bindposes`` are
# left alone.  So where this armature's rest differs from the mesh's bind pose,
# the difference is a deformation the game applies on purpose.
#
# It differs in exactly one place: the whole breast chain, ``cf_d_bust01_*``
# downward — 114 bones, up to 94 mm apart, and a 0.36 difference in the 3×3.
# ``p_cf_body_bone`` holds them **collapsed** (``cf_j_bustNN`` sits exactly on
# ``cf_d_bustNN``, all at x = 0), and the ``cf_a_bustNN_size`` curves are what
# build the breast back out.  ``p_cf_body_00`` and friends hold them at the
# pose the mesh was skinned in, which is a bust the sliders then add to again.
#
# Which prefab wins therefore matters, and it used to be decided by whatever
# order UnityPy happened to enumerate the bundle's roots in — ``p_cf_body_00_hit``
# on this install.  The armature goes first now, and the ``_01`` reference bones
# the sway solver measures against (which live nowhere else) come with it.
PRIMARY_ROOTS = ("p_cf_body_bone",)

# ``RefObjKey.HeadParent`` resolves to ``cf_s_head``; ``cf_j_head`` is the same
# point without the head-size scale and is only a fallback.
FACE_PARENT = ("cf_s_head", "cf_j_head")

# Prefabs ``<LoadAsync>`` instantiates on their own and hangs under a body bone,
# root name → the bone.  ``p_tang2`` is ``objTongueEx``: loaded from oo_base
# after the body's weights are assigned and parented with
# ``SetParent(GetReferenceInfo(0).transform, false)`` (RefObjKey 0 is
# ``cf_s_head``), so its root keeps its identity local there.  It brings a second
# tongue chain under the **same names** as the body armature's own
# (``cf_s_head/cf_j_tang_01‥05``), plus ``cf_j_tang2_L/R_05`` (PROGRESS #30).
ATTACHED_ROOTS = {"p_tang2": "cf_s_head"}
# Bones below an attached root are keyed ``<root>:<game name>``.  The game tells
# the two tongues apart only by path; a name-keyed rig needs a key of its own,
# or the second chain is folded into the first (its ``cf_j_tang2_*`` were filed
# under the body's ``cf_j_tang_05`` with a local from the prefab root and drawn
# at the feet).  The body armature's names stay as they are, so every mesh and
# clip that binds the body's chain by name keeps binding it.
SCOPE_SEP = ":"

# An attached prefab with an ``Animator`` plays exactly **one frame** before
# ``ChaControl.UpdateVisible`` (called from ``LateUpdateForce``, so after that
# frame's animation update) hides it: ``SetActiveControl(objTongueEx,
# visibleAll && tongueState == 2 && …)`` (IL_03a1–03ec), and ``tongueState`` is
# 2 nowhere in the game (``ChangeMouthPtn`` gives 0 or 1).  An inactive object's
# Animator never updates again, so its bones keep that frame's pose for good.
# The frame is the one that ran ``<LoadAsync>``'s bundle loads (IL_10cc–11ea
# instantiates ``p_tang2`` and yields), and a frame that long is clamped by the
# engine to ``Time.maximumDeltaTime``: ``Koikatu_Data/globalgamemanagers``'
# ``TimeManager`` "Maximum Allowed Timestep" is 0.33333334 (float32 of 1/3).
# ``tang2_anime``'s default state ``ptn01`` (speed 1, cycle offset 0, no
# parameters; its 1.333 s clip loops; the only transition leaves at normalised
# 1.57) is therefore 1/3 s in — normalised 0.25 — and that is where all 162
# ``p_tang2`` chains of the H dumps sit (largest rotation apart 0.00001°, PROGRESS #30).
MAX_DELTA_TIME = 0.3333333432674408


def load_frame_time(state: dict, clip_start: float, clip_stop: float,
                    dt: float = MAX_DELTA_TIME) -> float | None:
    """The clip time an ``Animator`` reaches in its default *state* after one frame of *dt* seconds.

    *state* is the controller's ``AnimatorStateConstant`` typetree.  ``None``
    (not transcribed, the caller reports it) when the frame depends on what
    this does not model: a speed or cycle-offset parameter, a transition that
    leaves without exit time, or one whose exit time the frame reaches.
    """
    if int(state.get("m_SpeedParamID", 0) or 0) or int(state.get("m_CycleOffsetParamID", 0) or 0):
        return None
    length = float(clip_stop) - float(clip_start)
    if length <= 0.0:
        return None
    norm = float(state.get("m_CycleOffset", 0.0) or 0.0) + float(state.get("m_Speed", 1.0)) * float(dt) / length
    from .hlist import unwrap
    for tr in unwrap(state.get("m_TransitionConstantArray")) or []:
        tr = unwrap(tr)
        if not tr.get("m_HasExitTime") or norm >= float(tr.get("m_ExitTime", 0.0)):
            return None
    frac = norm % 1.0 if state.get("m_Loop") else min(norm, 1.0)
    return float(clip_start) + frac * length


def _default_state_clip(ctrl_tree: dict) -> tuple[dict, int] | None:
    """``(state typetree, clip index)`` of a one-layer controller's default state playing one clip, else None."""
    from .hlist import unwrap
    ctrl = unwrap(ctrl_tree.get("m_Controller") or {})
    layers = unwrap(ctrl.get("m_LayerArray")) or []
    machines = unwrap(ctrl.get("m_StateMachineArray")) or []
    if len(layers) != 1 or len(machines) != 1:
        return None
    sm = unwrap(machines[0])
    states = unwrap(sm.get("m_StateConstantArray")) or []
    k = int(sm.get("m_DefaultState", 0) or 0)
    if not 0 <= k < len(states):
        return None
    state = unwrap(states[k])
    trees = unwrap(state.get("m_BlendTreeConstantArray")) or []
    nodes = [unwrap(n) for n in unwrap(unwrap(trees[0]).get("m_NodeArray")) or []] if len(trees) == 1 else []
    if len(nodes) != 1:
        return None
    clip = int(nodes[0].get("m_ClipID", -1))
    n_clips = len(ctrl_tree.get("m_AnimationClips") or [])
    return (state, clip) if 0 <= clip < n_clips else None


def scoped_path_map(scope: str, parents: dict) -> dict[int, str]:
    """``{crc32(path): key}`` of the bones under attached root *scope*, paths relative to it (its Animator's)."""
    from .hanim import _unity_path_crc
    out: dict[int, str] = {}
    for name in parents:
        if scope_of(name) != scope:
            continue
        parts, cur = [], name
        while cur and cur != scope:
            parts.append(game_name(cur))
            cur = parents.get(cur) or ""
        if cur == scope:
            out[_unity_path_crc("/".join(reversed(parts)))] = name
    return out


def _play_load_frame(env, by_id: dict, root_tf, scope: str, parents: dict, locals_: dict,
                     mats: dict) -> list[str]:
    """Pose the bones under attached root *scope* where its ``Animator`` leaves them (``MAX_DELTA_TIME``).

    Returns every key under *scope* (their world matrices were recomputed),
    or ``[]`` when the root has no Animator with a controller.  A controller
    this does not transcribe (``load_frame_time``, ``_default_state_clip``) is
    reported and leaves the prefab's own locals.
    """
    go = (root_tf.read_typetree().get("m_GameObject") or {}).get("m_PathID")
    anim = next((o for o in env.objects if o.type.name == "Animator"
                 and (o.read_typetree().get("m_GameObject") or {}).get("m_PathID") == go), None)
    ctl = by_id.get(((anim.read_typetree().get("m_Controller") or {}) if anim is not None else {}).get("m_PathID"))
    if ctl is None:
        return []
    tree = ctl.read_typetree()
    found = _default_state_clip(tree)
    clip_obj = None
    if found is not None:
        ref = (tree.get("m_AnimationClips") or [])[found[1]]
        clip_obj = by_id.get(ref.get("m_PathID"))
    if clip_obj is None:
        swallowed.note("origin.rig._play_load_frame: controller not transcribed",
                       ValueError(f"{scope}: {tree.get('m_Name')}"))
        return []
    clip = clip_obj.read()
    muscle = getattr(clip, "m_MuscleClip", None)
    time = None if muscle is None else load_frame_time(
        found[0], float(getattr(muscle, "m_StartTime", 0.0) or 0.0), float(getattr(muscle, "m_StopTime", 0.0) or 0.0))
    if time is None:
        swallowed.note("origin.rig._play_load_frame: the load frame is not transcribed",
                       ValueError(f"{scope}: {tree.get('m_Name')} / {getattr(clip, 'm_Name', '')}"))
        return []
    from .hanim import sample_transforms
    channels = sample_transforms(clip, time, scoped_path_map(scope, parents))
    if not channels:
        swallowed.note("origin.rig._play_load_frame: the clip reaches none of the bones",
                       ValueError(f"{scope}: {getattr(clip, 'm_Name', '')}"))
        return []
    pose_scope(scope, channels, parents, locals_, mats)
    return [n for n in locals_ if scope_of(n) == scope]


def pose_scope(scope: str, channels: dict, parents: dict, locals_: dict, mats: dict) -> list[str]:
    """Write *channels* (``{key: {"pos"|"rot"|"scale": …}}``, our space) into the locals under *scope*.

    The world matrices under it follow, parents first (*locals_* is in walk
    order).  Returns the keys whose local changed.
    """
    moved = []
    for name, local in locals_.items():
        if scope_of(name) != scope:
            continue
        ch = channels.get(name)
        if ch:
            t, q, s = m3.decompose(local)
            locals_[name] = m3.compose(np.asarray(ch.get("pos", t), dtype=np.float64),
                                       m3.quat_normalize(np.asarray(ch.get("rot", q), dtype=np.float64)),
                                       np.asarray(ch.get("scale", s), dtype=np.float64))
            moved.append(name)
        parent = parents.get(name) or ""
        if parent in mats:
            mats[name] = mats[parent] @ locals_[name]
    return moved


def scoped_name(root: str, name: str) -> str:
    """Our key for the game's bone *name* under the attached prefab *root*."""
    return f"{root}{SCOPE_SEP}{name}"


def game_name(bone: str) -> str:
    """The game's own name of our bone key (the part after the scope, if any)."""
    return bone.rpartition(SCOPE_SEP)[2]


def scope_of(bone: str) -> str:
    """The attached prefab root *bone* is keyed under, or ``""`` for the body armature."""
    root, sep, _name = bone.partition(SCOPE_SEP)
    return root if sep and root in ATTACHED_ROOTS else ""


def name_from_path(path: str) -> str:
    """Our key for the transform at the game's hierarchy *path* (``a/b/c``, as a bone dump writes it).

    The last attached root on the path scopes what is under it; anything else
    is its own last segment.
    """
    parts = [p for p in str(path).split("/") if p]
    if not parts:
        return ""
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] in ATTACHED_ROOTS:
            return scoped_name(parts[i], parts[-1])
    return parts[-1]


def _trs(tree: dict) -> np.ndarray:
    p = tree.get("m_LocalPosition") or {"x": 0, "y": 0, "z": 0}
    r = tree.get("m_LocalRotation") or {"x": 0, "y": 0, "z": 0, "w": 1}
    s = tree.get("m_LocalScale") or {"x": 1, "y": 1, "z": 1}
    pos = np.array([-float(p["x"]), float(p["y"]), float(p["z"])])
    q = m3.quat_normalize(np.array([float(r["x"]), -float(r["y"]),
                                    -float(r["z"]), float(r["w"])]))
    return m3.compose(pos, q, np.array([float(s["x"]), float(s["y"]),
                                        float(s["z"])]))


class ReferenceRig:
    """Full Illusion bind hierarchy from oo_base.

    * ``matrices`` — name → world 4×4 in Illusion metres (X mirrored)
    * ``parents``  — name → parent name (or ``""`` for roots)
    * ``locals``   — name → local 4×4 (parent-relative)
    """

    def __init__(self):
        self.nodes: dict[str, np.ndarray] = {}
        self.matrices: dict[str, np.ndarray] = {}
        self.parents: dict[str, str] = {}
        self.locals: dict[str, np.ndarray] = {}
        self.built = False

    def matrix(self, name: str) -> np.ndarray | None:
        m = self.matrices.get(name)
        return None if m is None else np.asarray(m, dtype=np.float64)

    def parent_of(self, name: str) -> str | None:
        p = self.parents.get(name)
        return p if p else None

    # ------------------------------------------------------------------
    def build(self, install) -> "ReferenceRig":
        import UnityPy
        path = install.bundle(BASE_BUNDLE)
        if not path.exists():
            return self
        env = UnityPy.load(str(path))
        by_id = {o.path_id: o for o in env.objects}

        def deref(ref):
            if not isinstance(ref, dict):
                return None
            inner = ref.get("component", ref)
            return by_id.get(inner.get("m_PathID")) if isinstance(inner, dict) else None

        transforms = [o for o in env.objects if o.type.name == "Transform"]
        roots = []
        for t in transforms:
            tt = t.read_typetree()
            par = deref(tt.get("m_Father"))
            if par is None:
                roots.append(t)

        def root_name(t) -> str:
            tt = t.read_typetree()
            go = by_id.get((tt.get("m_GameObject") or {}).get("m_PathID"))
            try:
                return (go.read().m_Name or "") if go is not None else ""
            except Exception as exc:  # noqa: BLE001
                swallowed.note("origin.rig.ReferenceRig.build.root_name: return go.read().m_Name or '' if go…", exc)
                return ""

        # first name wins, so the armature has to be walked before the prefabs
        # that carry the mesh's bind pose under the same names (see PRIMARY_ROOTS)
        rank = {n: i for i, n in enumerate(PRIMARY_ROOTS)}
        roots.sort(key=lambda t: rank.get(root_name(t), len(rank)))

        out: dict[str, np.ndarray] = {}
        mats: dict[str, np.ndarray] = {}
        parents: dict[str, str] = {}
        locals_: dict[str, np.ndarray] = {}

        def walk(t, parent_world, parent_name: str, scope: str = ""):
            tt = t.read_typetree()
            local = _trs(tt)
            world = parent_world @ local
            go = by_id.get((tt.get("m_GameObject") or {}).get("m_PathID"))
            name = ""
            if go is not None:
                try:
                    name = go.read().m_Name or ""
                except Exception as exc:  # noqa: BLE001
                    swallowed.note("origin.rig.ReferenceRig.build.walk: name = go.read().m_Name or ''", exc)
                    name = ""
            # below an attached prefab's root, a name is keyed under that root
            if name and scope and name != scope:
                name = scoped_name(scope, name)
            if name and name not in out:
                out[name] = world[:3, 3].copy()
                mats[name] = world.copy()
                parents[name] = parent_name
                locals_[name] = local.copy()
            # Descend under this node's own name even when the name was already
            # taken.  The old fallback ("keep the last *new* name") skipped
            # every seen ancestor, so a bone first met in a later prefab was
            # filed under whatever unseen node came before it -- which is how
            # ``cf_j_bust02_L_01`` ended up parented to ``p_cf_body_bone``
            # instead of ``cf_d_bust02_L``, and therefore grafted as a floating
            # root that no shaping ever moved.
            next_parent = name or parent_name
            for ch in tt.get("m_Children", []):
                co = by_id.get(ch.get("m_PathID"))
                if co is not None:
                    walk(co, world, next_parent, scope)

        attached = []
        for r in roots:
            if root_name(r) in ATTACHED_ROOTS:
                attached.append(r)
                continue
            walk(r, m3.identity(), "")
        # After the armature, so the bone each hangs under is known: placed
        # where ``SetParent(…, false)`` puts it, its own local under that bone.
        for r in attached:
            nm = root_name(r)
            host = ATTACHED_ROOTS[nm]
            if host not in mats:
                swallowed.note("origin.rig.ReferenceRig.build: attached root's bone missing",
                               KeyError(f"{nm} -> {host}"))
                continue
            walk(r, mats[host], host, nm)
            for name in _play_load_frame(env, by_id, r, nm, parents, locals_, mats):
                out[name] = mats[name][:3, 3].copy()
        self.nodes = out
        self.matrices = mats
        self.parents = parents
        self.locals = locals_
        self.built = bool(out)
        return self

    # ------------------------------------------------------------------
    def get(self, name: str) -> np.ndarray | None:
        v = self.nodes.get(name)
        return None if v is None else np.asarray(v, dtype=np.float64)

    def head_length(self) -> float:
        top = self.get("a_n_headtop")
        return float(top[1]) if top is not None and top[1] > 0.05 else 0.2368

    def body_height(self) -> float:
        """Standing height of the reference body, in Illusion units.

        ``oo_base`` ships the body at ``cf_n_height`` scale 1.0, which is the
        top of the height slider — every card is some fraction of this.  The
        hat attach node sits head-local, so it doubles as the head length.
        """
        head = self.get("cf_j_head")
        if head is None or float(head[1]) < 0.1:
            return 0.0
        return float(head[1]) + self.head_length()

    def head_local(self, name: str) -> bool:
        """Is this node stored in head space rather than body space?

        ``oo_base`` keeps the head hierarchy under its own root, so every
        transform in that subtree has a "world" matrix that is really
        head-local.  Membership follows the parent chain — ``a_n_megane`` hangs
        off ``cf_J_megane_rx_ear`` and is head-local too, which a name prefix
        would never tell you.
        """
        seen = set()
        n = name
        while n and n not in seen:
            if n.startswith(("cf_J_", "cf_j_hitomi", "cf_hitomi")):
                return True
            seen.add(n)
            n = self.parents.get(n) or ""
        return False

    def attach_offset(self, node: str,
                      relative_to: str = "") -> np.ndarray | None:
        """Where an ``a_n_*`` node sits in *relative_to*'s frame.

        The old version decided between "offset" and "world position" by
        testing ``|y| < 0.6``.  That is wrong in both directions: ``a_n_back``
        is a world position at y 1.38 and was therefore reported as *no offset*
        (the backpack landed on the spine, inside the ribcage), while
        ``a_n_heel`` is a world position at y 0.034 and was reported as an
        offset (a shoe charm moved by its own height).  Which frame a node is
        in is settled by :meth:`head_local`, and a body node is converted with
        the reference's own matrices instead of being guessed at.
        """
        m = self.matrix(node)
        if m is None:
            return None
        if self.head_local(node):
            return m[:3, 3].copy()          # already relative to the head
        base = self.matrix(relative_to) if relative_to else None
        if base is None:
            return None
        return (m3.inverse(base) @ m)[:3, 3]

    def body_bone_names(self) -> list[str]:
        """Bones that belong on the character armature (not prefab / hit GO)."""
        keep = []
        for n in self.matrices:
            # Body, attach nodes, and the facial rig.  ``cf_J_*`` is authored in
            # head-local space under its own root, so it is grafted as a subtree
            # hanging off cf_j_head rather than by world matrix — see
            # graft_full_skeleton.  Without it the head mesh has nowhere to put
            # its weights and InitShapeFace has nothing to move.
            # ``cm_J_dan*`` is the male shaft chain (101 base → 109 tip).  It
            # was excluded with every other ``cm_J_*`` helper, so Free H
            # ``align_penetration`` looked for non-existent ``cf_j_dan00`` and
            # insertion never moved the actor (§7.110).  Keep the dan chain
            # and the small SoS helper bones that hang off the tip.
            # ``k_f_*`` nulls (``k_f_kokan_00``, ``k_f_ana_00``, …) are the
            # Lookat_dan targets named in ``h/list/dan_*`` (§7.117 / §7.118).
            # ``cf_hit_*`` are the 24 ``DynamicBoneCollider`` objects: the body
            # shape pass moves and scales them (``ShapeBodyInfoFemale`` has
            # rows for them, ``cf_a_hit_*``), and the garments' chains are
            # solved against them where the pass and the pose put them (§7.540).
            # ``cf_pv_*`` (under ``cf_n_height``, animated by the clips) and
            # ``cf_t_*`` (``FullBodyBipedIK``'s effector targets) are what the
            # H scene's ``MotionIK`` solves toward (Animation_Progress D14);
            # ``cf_ik_hips`` / ``cf_kk_shoulder`` / ``cf_kk_waist`` are where
            # ``cf_pv_shoulder_*`` / ``cf_pv_waist_*`` hang — without them those
            # four became roots, left at the bind's world spot (§7.1212).
            # An attached prefab (``ATTACHED_ROOTS``: ``p_tang2``) is on the
            # character too: its root, and its bones under their scoped keys
            # by the same name rule as the armature's.
            own = game_name(n) if scope_of(n) else n
            if own.startswith(("cf_j_", "cf_s_", "cf_d_", "cf_n_", "a_n_",
                               "cf_J_", "cf_hitomi", "cm_J_dan", "k_f_", "cf_hit_",
                               "cf_pv_", "cf_t_", "cf_ik_", "cf_kk_")) \
                    or n in ATTACHED_ROOTS:
                keep.append(n)
        # Order: main joints → scale helpers → face → attach nodes
        def rank(n: str) -> tuple:
            if n.startswith("cf_j_"):
                return (0, n)
            if n.startswith("cf_n_"):
                return (1, n)
            if n.startswith("cf_s_"):
                return (2, n)
            if n.startswith("cf_d_"):
                return (3, n)
            if n.startswith("cf_J_") or n.startswith("cf_hitomi"):
                return (4, n)
            if n.startswith("a_n_"):
                return (5, n)
            return (9, n)
        return sorted(keep, key=rank)
    def stats(self) -> dict:
        an = sum(1 for k in self.nodes if k.startswith("a_n_"))
        cf = sum(1 for k in self.nodes if k.lower().startswith("cf_j"))
        return {"nodes": len(self.nodes), "attach": an, "bones": cf,
                "parents": len(self.parents)}

    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = Path(path or CACHE)
        write_json_atomic(path, {
            "version": CACHE_VERSION,
            "nodes": {k: [round(float(x), 6) for x in v]
                      for k, v in self.nodes.items()},
            "matrices": {k: [round(float(x), 6) for x in np.asarray(v).ravel()]
                         for k, v in self.matrices.items()},
            "parents": self.parents,
            "locals": {k: [round(float(x), 6) for x in np.asarray(v).ravel()]
                       for k, v in self.locals.items()},
        })
        return path

    def load(self, path: Path | None = None) -> bool:
        path = Path(path or CACHE)
        if not path.exists():
            return False
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if int(d.get("version", 0)) < CACHE_VERSION or "parents" not in d:
            return False
        self.nodes = {k: np.asarray(v, dtype=np.float64)
                      for k, v in (d.get("nodes") or {}).items()}
        self.matrices = {k: np.asarray(v, dtype=np.float64).reshape(4, 4)
                         for k, v in (d.get("matrices") or {}).items()}
        self.parents = {k: str(v) for k, v in (d.get("parents") or {}).items()}
        self.locals = {k: np.asarray(v, dtype=np.float64).reshape(4, 4)
                       for k, v in (d.get("locals") or {}).items()}
        self.built = bool(self.nodes)
        return self.built


def graft_full_skeleton(skeleton, reference: ReferenceRig, scale: float = 1.0,
                        *, max_bones: int = 4096,
                        head_nodes: dict | None = None) -> int:
    """Replace *skeleton* with Illusion's full bind hierarchy (T-pose).

    Mirrors ChaControl: one armature, every ``cf_*`` / ``a_n_*`` bone kept by
    name so clothing weights need no collapse (AssignedAnotherWeights style).
    Returns number of bones set.

    Face bones (``cf_J_*``) are re-parented under ``cf_s_head`` because oo_base
    stores them in head-local space under a separate root — and ``cf_s_head``
    is the game's own answer, not the nearest plausible bone: ``<LoadAsync>``
    ends the head-bone step with

        objHeadBone.transform.SetParent(
            GetReferenceInfo(RefObjKey.HeadParent).transform, false)

    and ``ChaReference.CreateReferenceInfo`` files key 0 under the name
    ``cf_s_head`` (once, in the body-bone group; nothing else writes index 0).
    ``p_cf_head_bone``'s own root and ``cf_J_N_FaceRoot`` are both identity, so
    the face rig sits at ``cf_s_head``'s origin and **inherits its scale** —
    and that scale is not 1: ``ShapeBodyInfoFemale.Update`` writes
    ``cf_a_head × cf_a_height_aid × correctHeadSize`` there, the term that keeps
    the head from shrinking with the body.  Hanging the face off ``cf_j_head``
    (scale 1, same position) drops that factor and makes the whole head — every
    ``cf_J_*``-weighted mesh, which is all of them — the wrong size.

    ``head_nodes`` is the head **the card wears**, as
    ``name → (parent, local 4×4)`` (``body_head.head_bone_hierarchy``).  The
    head-local subtree is taken from it wherever it has the node, because that
    prefab is the one the game instantiates: ``oo_base``'s ``p_cf_head_bone``
    is a reference rig that never appears on a character.  Left out, the graft
    keeps oo_base's — which agrees on head 0's joints and disagrees on **every**
    ``a_n_*`` attach frame by up to 92.6 mm, so every hat, pin and earring is
    placed off a rig the character does not have (§7.102).
    """
    if not reference.built or not reference.matrices:
        return 0

    names = reference.body_bone_names()
    allowed = set(names)

    # ``oo_base`` holds the head hierarchy under its own root, so everything in
    # that subtree is stored **head-local** while the body's transforms are
    # world.  Which frame a node is in is a property of the hierarchy, not of
    # its name: ``a_n_megane`` and ``a_n_headtop`` hang off face bones and are
    # head-local too, but the old prefix test (``cf_J_*``) missed them, so they
    # were grafted as if 0.09 m were a world height.  That put the glasses
    # attach point at the ankles; the loader papered over it with a "small y
    # means it is an offset" rule, which then mis-read the body's own low
    # attach nodes (``a_n_heel`` at y 0.034 is a world position, not an offset).
    # Face seed is capital-J facial rig only.  BP body bones (``cf_J_Vagina_*``
    # etc.) also use capital J but hang under the pelvis — never re-parent
    # them under ``cf_s_head`` (§7.122).
    def _is_face_seed(n: str) -> bool:
        if n.startswith(("cf_J_Vagina", "cf_J_Ana", "cf_J_Belly", "cf_J_Siri",
                         "cf_J_Kokan", "cf_J_Kosi")):
            return False
        return n.startswith(("cf_J_", "cf_j_hitomi", "cf_hitomi"))

    _face_seed = {n for n in allowed if _is_face_seed(n)}
    _face = set(_face_seed)
    changed = True
    while changed:
        changed = False
        for n in allowed:
            if n in _face:
                continue
            p = reference.parents.get(n) or ""
            if p in _face:
                _face.add(n)
                changed = True

    def is_face(n: str) -> bool:
        return n in _face
    order: list[str] = []
    seen: set[str] = set()

    def visit(n: str) -> None:
        if n in seen or n not in allowed:
            return
        p = reference.parents.get(n) or ""
        # only walk parents that are also character bones (never prefab/hit
        # GOs).  Face bones walk theirs too so the subtree keeps its shape; the
        # chain stops on its own at cf_J_N_FaceRoot, whose parent is a prefab
        # node and therefore not in ``allowed``.
        if p and p in allowed and p not in seen:
            visit(p)
        if n not in seen:
            seen.add(n)
            order.append(n)

    for n in names:
        if not is_face(n):
            visit(n)
    for n in names:
        if is_face(n):
            visit(n)
    if "cf_j_hips" not in seen and "cf_j_root" not in seen:
        return 0

    skeleton.bones.clear()
    skeleton.index.clear()
    worlds: dict[str, np.ndarray] = {}
    scale = float(scale)
    added = 0

    for name in order:
        if added >= max_bones:
            break
        if is_face(name):
            # Head-local subtree: keep the authored parent so a bone three
            # levels down still inherits its ancestors.  Only the subtree root
            # re-parents, onto cf_s_head (see the docstring); cf_j_head is the
            # fallback for an armature that somehow lacks the scale helper --
            # same place, no head scale.
            # The head the card wears wins over the reference rig, node by
            # node — its parent too, because a modded head may hang a frame
            # somewhere else entirely.  A name it does not carry keeps oo_base's
            # (``EyeTargetL/R`` and the ``n_cam_*`` markers are only there).
            worn = (head_nodes or {}).get(name)
            p_name = (worn[0] if worn else reference.parents.get(name)) or ""
            parent = p_name if p_name in skeleton.index else next(
                (b for b in FACE_PARENT if b in skeleton.index), None)
            loc = worn[1] if worn else reference.locals.get(name)
            if loc is None:
                loc = reference.matrices[name]
            loc = loc.copy()
            loc[:3, 3] *= scale
            t, q, s = m3.decompose(loc)
            skeleton.add(name, parent, t, q, s)
            worlds[name] = (worlds[parent] @ loc) if (
                parent and parent in worlds) else loc
            added += 1
            continue

        p_name = reference.parents.get(name) or ""
        if p_name and p_name not in skeleton.index:
            p_name = ""
        w = reference.matrices[name].copy()
        w[:3, 3] *= scale
        if p_name and p_name in worlds:
            local = m3.inverse(worlds[p_name]) @ w
        else:
            local = w
        # decompose, not just position+rotation: a handful of rest bones carry a
        # non-unit bind scale (cf_s_bnip025_* is 1.2), and dropping it here made
        # the shape pass write onto a rest pose the game never had
        t, q, s = m3.decompose(local)
        skeleton.add(name, p_name or None, t, q, s)
        worlds[name] = w
        added += 1

    skeleton.finalize()
    skeleton.rig_scale = float(scale)     # what a clip's absolute translation is multiplied by (§7.1177)
    return added

_rig: ReferenceRig | None = None


def rig(install=None, refresh: bool = False) -> ReferenceRig:
    global _rig
    if _rig is not None and not refresh:
        return _rig
    r = ReferenceRig()
    if not refresh and r.load():
        _rig = r
        return r
    from .install import configured
    install = install or configured()
    if install.valid:
        r.build(install)
        r.save()
    _rig = r
    return r
