"""``BustSoft`` / ``BustGravity`` — the two card sliders that shape the chest.

``bustSoftness`` and ``bustWeight`` sit on every card and, until now, went
nowhere: :mod:`chara.cardimport` mapped them onto parameters of the procedural
body that §7.36 deleted, and the card path never read them at all.  In the game
they do not move a bone directly — they retune the ``DynamicBone_Ver02`` chains
on ``cf_j_bustNN_L`` / ``_R`` (:mod:`..dynbone2`), and the resting shape of
those chains is what the breast looks like when the character is standing
still.

Two classes do it, both driven from ``ChaControl.UpdateBustSoftnessAndGravity``
(softness first, then gravity, then ``ReSetupDynamicBoneBust`` re-selects the
pattern so the new numbers take):

``BustSoft.ReCalc(new int[1])``::

    rate = Clamp(bustSoftness * shapeValueBody[BustSize] + 0.01, 0, 1)
    setSoftParams(0, -1, TreeLerp(bustDamping,    rate),
                         TreeLerp(bustElasticity, rate),
                         TreeLerp(bustStiffness,  rate), true)

``BustGravity.ReCalc(new int[1])``::

    scale = shapeValueBody[BustSize] * bustSoftness * 0.5
    g     = Mathf.Lerp(range[0], range[1], bustWeight) * scale
    setGravity(0, new Vector3(0, g, 0), true)

``range`` is not serialised data — ``BustGravity..ctor`` writes ``range[1] =
-0.005f`` and leaves ``range[0]`` at 0 — so the whole gravity term is

    **g = −0.0025 × bustWeight × BustSize × bustSoftness**

and the three softness tables are C# array literals, which live in ``FieldRVA``
blobs rather than in IL (``origin_il_dump.py --data``)::

    bustDamping    = [0.20, 0.10, 0.10]
    bustElasticity = [0.20, 0.15, 0.05]
    bustStiffness  = [1.00, 0.10, 0.01]

``TreeLerp`` walks them as a two-segment ramp.  **The softness half matters as
much as the gravity half**: a rate of 0 gives stiffness 1.0, which pins the
chain to its rest target no matter how hard gravity pulls, and a rate of 1
gives 0.01, which lets it hang.  §7.42 recorded these three tables as needed
only for animation; they are not, they set the standing shape.

``setSoftParams(0, -1, d, e, s, true)`` writes **one** damping / elasticity /
stiffness across the whole chain — the per-bone values authored in the prefab
are overwritten on every character, and they also go onto pattern 0's blocks so
the following ``setPtn`` does not put them back.

Why this runs at load rather than per frame: our still frames want the shape the
chain hangs in, so the solve is baked into the armature before any mesh is
retargeted onto it — the same thing :func:`..figure.dress._apply_sway` does for
a skirt.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ...core import math3d as m3
from .. import dynbone2 as DB2
from ..prefab import bust_dynamic_bones
from . import shape_cat

BUST_SIZE_SLIDER = shape_cat.BODY_SHAPE_NAMES.index("BustSize")

# ``BustSoft..ctor`` — three-point ramps, read out of the FieldRVA blobs behind
# ``RuntimeHelpers.InitializeArray``.
BUST_DAMPING = (0.20000000298023224, 0.10000000149011612, 0.10000000149011612)
BUST_ELASTICITY = (0.20000000298023224, 0.15000000596046448, 0.05000000074505806)
BUST_STIFFNESS = (1.0, 0.10000000149011612, 0.009999999776482582)

# ``BustGravity..ctor``: ``range = new float[2]; range[1] = -0.005f;``
GRAVITY_RANGE = (0.0, -0.005)

# ``dictDynamicBoneBust`` indices ``ReSetupDynamicBoneBust(false)`` resets —
# the breasts.  2 and 3 are the buttocks, whose components ship with zero
# gravity and zero force and are never handed any, so they have no rest shape
# to solve for.
BUST_INDICES = (0, 1)
#: the buttocks: only the per-frame solve moves them (``bust_live``)
HIP_INDICES = (2, 3)


def tree_lerp(table, t: float) -> float:
    """``BustSoft.TreeLerp`` — a two-segment ramp through three values."""
    if t < 0.5:
        return float(table[0] + (table[1] - table[0]) * min(max(t * 2.0, 0.0), 1.0))
    return float(table[1] + (table[2] - table[1])
                 * min(max((t - 0.5) * 2.0, 0.0), 1.0))


@dataclass
class BustSettings:
    """What the card asks of the two bust chains."""
    softness: float = 0.0
    weight: float = 0.0
    size: float = 0.0
    rate: float = 0.0
    gravity: float = 0.0
    damping: float = 0.0
    elasticity: float = 0.0
    stiffness: float = 0.0


def settings(card) -> BustSettings | None:
    """Read the card and run both ``ReCalc``s; ``None`` if it has no sliders.

    Note which array is read: ``shapeValueBody`` straight off the card, not the
    values :func:`shape_run.slider_values` hands the shaper — the low-poly
    compression and the arousal lerp are the *shaper's* adjustments and neither
    ``BustSoft`` nor ``BustGravity`` goes near them.
    """
    body = (card.body() if hasattr(card, "body") else {}) or {}
    values = body.get("shapeValueBody")
    if not values or len(values) <= BUST_SIZE_SLIDER:
        return None
    try:
        size = float(values[BUST_SIZE_SLIDER])
        soft = float(body.get("bustSoftness") or 0.0)
        weight = float(body.get("bustWeight") or 0.0)
    except (TypeError, ValueError):
        return None
    rate = min(max(soft * size + 0.01, 0.0), 1.0)
    lo, hi = GRAVITY_RANGE
    return BustSettings(
        softness=soft, weight=weight, size=size, rate=rate,
        # Mathf.Lerp clamps its t, which is why a card with bustWeight over 1
        # (the maker allows it) does not fall through the floor
        gravity=(lo + (hi - lo) * min(max(weight, 0.0), 1.0))
        * (size * soft * 0.5),
        damping=tree_lerp(BUST_DAMPING, rate),
        elasticity=tree_lerp(BUST_ELASTICITY, rate),
        stiffness=tree_lerp(BUST_STIFFNESS, rate))


def set_soft_params(spec: DB2.Ver02Spec, ptn_no: int, damping: float,
                    elasticity: float, stiffness: float) -> None:
    """``setSoftParams(ptnNo, -1, d, e, s, true)`` on the pattern's blocks.

    Only the pattern half is needed here: the particles are built after this,
    from these very blocks, so writing the live particles as well would be
    writing to a list that does not exist yet.
    """
    if not (0 <= ptn_no < len(spec.patterns)):
        return
    for prm in spec.patterns[ptn_no].params:
        prm.damping = damping
        prm.elasticity = elasticity
        prm.stiffness = stiffness
    spec.patterns[ptn_no].end_damping = damping
    spec.patterns[ptn_no].end_elasticity = elasticity
    spec.patterns[ptn_no].end_stiffness = stiffness


def set_gravity(spec: DB2.Ver02Spec, ptn_no: int, gravity) -> None:
    """``setGravity(ptnNo, g, true)`` — the pattern's gravity and the field."""
    g = np.asarray(gravity, dtype=np.float64)
    spec.gravity = g.copy()
    if ptn_no < 0:
        for ptn in spec.patterns:
            ptn.gravity = g.copy()
    elif ptn_no < len(spec.patterns):
        spec.patterns[ptn_no].gravity = g.copy()


# ---------------------------------------------------------------------------
# the rig the solver drives
# ---------------------------------------------------------------------------
class SkeletonRig:
    """Unity ``Transform`` semantics over one of our skeletons.

    Two things this has to get right.

    **Rotation is not the matrix.**  ``localToWorldMatrix`` carries the scale
    and places points; ``Transform.rotation`` ignores it and is just the product
    of the local rotations.  Composing a world rotation out of the full matrices
    instead would fold a parent's non-uniform scale into the bone's local
    rotation — and ``cf_d_bustNN`` carries exactly that, the size sliders write
    a per-axis scale onto it.

    **Our armature keeps the height somewhere else.**  ``ChaControl`` scales
    ``cf_n_height`` and lets every descendant inherit it, so in the game a
    bone's world matrix is ``[u·R·S | u·t]``.  ``graft_full_skeleton`` bakes the
    same ``u`` into the translations only, giving ``[R·S | u·t]`` — identical
    positions, but a 3×3 that is ``u`` times too small for anything measured in
    the bone's own frame: ``lossyScale`` (which scales gravity), and the
    ``EndOffset`` that places the chain's free tip 0.2 units past the last bone.
    ``world_scale`` puts it back, and only in :meth:`matrix`; the local writes
    stay in our own convention, where the translations already carry it.
    """

    def __init__(self, skeleton, world_scale: float = 1.0):
        self.skel = skeleton
        self.world_scale = float(world_scale) or 1.0
        self._dirty = False

    # -- lookup ---------------------------------------------------------
    def has(self, key) -> bool:
        return self.skel.has(key)

    def _bone(self, key):
        return self.skel.bones[self.skel.id(key)]

    def _sync(self) -> None:
        if self._dirty:
            self.skel.finalize()
            self._dirty = False

    def matrix(self, key) -> np.ndarray:
        self._sync()
        m = np.asarray(self._bone(key).bind_world, dtype=np.float64).copy()
        m[:3, :3] *= self.world_scale
        return m

    def rotation(self, key) -> np.ndarray:
        q = m3.quat_identity()
        i = self.skel.id(key)
        chain = []
        while i >= 0:
            chain.append(i)
            i = self.skel.bones[i].parent
        for j in reversed(chain):
            q = m3.quat_mul(q, self.skel.bones[j].local_rot)
        return m3.quat_normalize(q)

    def parent_rotation(self, key) -> np.ndarray:
        p = self._bone(key).parent
        return (m3.quat_identity() if p < 0
                else self.rotation(self.skel.bones[p].name))

    # -- writes ---------------------------------------------------------
    def set_rotation(self, key, q) -> None:
        """``Transform.rotation = q`` → ``localRotation = inv(parent) * q``."""
        b = self._bone(key)
        b.local_rot = m3.quat_normalize(
            m3.quat_mul(m3.quat_conj(self.parent_rotation(key)),
                        m3.quat_normalize(np.asarray(q, dtype=np.float64))))
        self._dirty = True

    def set_position(self, key, p) -> None:
        """``Transform.position = p`` → the parent's inverse of it, locally.

        Deliberately the *un*-rescaled ``bind_world``: a world point is the same
        in both conventions, but the local position that reaches it is ``u``
        times larger in ours, which is the one the skeleton stores.
        """
        self._sync()
        b = self._bone(key)
        if b.parent < 0:
            b.local_pos = np.asarray(p, dtype=np.float64).copy()
        else:
            pw = self.skel.bones[b.parent].bind_world
            b.local_pos = m3.transform_point(m3.inverse(pw), p)
        self._dirty = True

    def local(self, key):
        b = self._bone(key)
        return (b.local_pos.copy(), b.local_rot.copy(), b.local_scale.copy())

    def set_local(self, key, pos, rot, scale) -> None:
        b = self._bone(key)
        b.local_pos = np.asarray(pos, dtype=np.float64).copy()
        b.local_rot = m3.quat_normalize(np.asarray(rot, dtype=np.float64))
        b.local_scale = np.asarray(scale, dtype=np.float64).copy()
        self._dirty = True

    def flush(self) -> None:
        self._sync()


# ---------------------------------------------------------------------------
@dataclass
class BustReport:
    label: str = "胸型"
    reason: str = ""
    chains: int = 0
    bones: int = 0
    iterations: int = 0
    drop: float = 0.0            # how far the deepest bone fell, in metres
    settings: BustSettings | None = None
    missing: list = field(default_factory=list)
    # the solved chains themselves: the per-frame solve starts from their drape
    # (``bust_live``, B3)
    chain_objs: list = field(default_factory=list)

    def describe(self) -> str:
        if self.reason:
            return f"{self.label}: {self.reason}"
        s = self.settings
        detail = ""
        if s is not None:
            detail = (f"，g={s.gravity:.5f}、rate={s.rate:.3f}"
                      f"（硬度 {s.stiffness:.3f}／彈性 {s.elasticity:.3f}）")
        out = (f"{self.label}: {self.chains} 條鏈、{self.bones} 根骨頭，"
               f"最深的一根沉降 {self.drop * 1000.0:.2f} mm{detail}")
        if self.missing:
            out += f"，骨架缺 {len(self.missing)} 根"
        return out


def apply_bust_shape(skeleton, card, unit: float = 1.0, *,
                     hi_poly: bool | None = None,
                     iterations: int = 600) -> BustReport:
    """Hang both breast chains under the card's own gravity.

    *unit* is the scale the armature was grafted at — the card's height slider —
    which stands in for ``cf_n_height``'s scale (see :class:`SkeletonRig`).

    Must run **after** the body shape sliders: the chain's rest targets are the
    positions of the ``cf_j_bustNN_*_01`` reference bones, and those are moved
    by ``cf_d_bustNN``'s size curves.  Solving first would hang the default
    body's breast and then resize it.
    """
    report = BustReport()
    cfg = settings(card)
    if cfg is None:
        report.reason = "卡片沒有滑桿陣列"
        return report
    specs = bust_dynamic_bones(hi_poly)
    if not specs:
        report.reason = "找不到 DynamicBone_Ver02（低模骨架本來就沒有）"
        return report

    rig = SkeletonRig(skeleton, world_scale=unit)
    total_drop = 0.0
    for idx in BUST_INDICES:
        cached = specs.get(idx)
        if cached is None:
            report.missing.append(f"dictDynamicBoneBust[{idx}]")
            continue
        # the specs are cached per install; the card's numbers go onto a copy
        # so the next character does not inherit this one's chest
        spec = copy.deepcopy(cached)
        ptn = int(spec.ptn_no)
        set_soft_params(spec, ptn, cfg.damping, cfg.elasticity, cfg.stiffness)
        set_gravity(spec, ptn, (0.0, cfg.gravity, 0.0))
        chain = DB2.setup(spec, rig, ptn)
        if chain is None:
            report.missing.append(spec.comment or f"bust {idx}")
            continue
        # measure the deepest particle that owns a bone, not the chain's tip:
        # the tip is a virtual node whose rest length is ``|EndOffset| *
        # ObjectScale`` while its start is ``TransformPoint(EndOffset)``, and
        # those disagree the moment ``cf_d_bustNN``'s scale is non-uniform, so
        # it snaps on the first step whatever gravity does.  It cannot pull on
        # anything (``UpdateParticles2`` only ever moves the child) and the bone
        # before it has ``IsRotationCalc`` off, so the jump is inert.
        last = max((i for i, p in enumerate(chain.particles)
                    if p.key is not None), default=0)
        was = chain.particles[last].pos.copy()
        report.iterations = max(report.iterations,
                                DB2.solve(chain, iterations=iterations))
        total_drop = max(total_drop,
                         float(np.linalg.norm(chain.particles[last].pos - was)))
        report.bones += DB2.apply_to_transforms(chain, rig)
        report.chains += 1
        chain.index = idx
        report.chain_objs.append(chain)
    rig.flush()
    report.drop = total_drop
    report.settings = cfg
    return report


def hip_chains(skeleton, unit: float = 1.0, *,
               hi_poly: bool | None = None) -> list:
    """The two buttock chains, set up on the shaped armature and not solved.

    Their components ship with zero gravity and zero force and the card hands
    them nothing, so the steady state is the bind itself: what moves them is
    inertia, frame by frame (``bust_live``).  Set up after the shape sliders,
    like the breasts, so the rest offsets are the shaped body's.  An index the
    prefab lacks is simply absent — the low-poly armature has none of the four.
    """
    specs = bust_dynamic_bones(hi_poly)
    rig = SkeletonRig(skeleton, world_scale=unit)
    out = []
    for idx in HIP_INDICES:
        cached = specs.get(idx) if specs else None
        if cached is None:
            continue
        spec = copy.deepcopy(cached)
        chain = DB2.setup(spec, rig, int(spec.ptn_no))
        if chain is not None:
            chain.index = idx
            out.append(chain)
    return out
