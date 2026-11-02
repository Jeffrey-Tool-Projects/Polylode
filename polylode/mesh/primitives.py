"""Parametric mesh primitives: grids, lofts, tubes, spheres, superellipses.

Everything the character generator needs is built from these; the body and head
are lofted from stacks of superellipse cross-sections so that sliders can
reshape them continuously.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import math3d as m3
from .mesh import Mesh


#: How wide the gusset's end is on the mother's **transition edge**, as a
#: fraction of that edge, and with it whether a fork's two children share one
#: crotch curve or get one each.
#:
#: At ``0`` they share it: both loops close on the same chord and its one edge
#: carries one face from each child, back to back.  Above ``0`` the two ends
#: land either side of the crossing :func:`fork_split` found, that far apart,
#: and a band of quads — the gusset — joins the two curves and closes on the
#: short piece of mother edge between each pair.  So the one knob spans both
#: constructions and one build of one ruler measures both (§7.278).
#:
#: **Zero is what ships, and that is measured** (§7.322).  §7.321 read the
#: shared chord's thirteen crotch vertices as a fin — 59-77% of their incident
#: area cancelling in ``compute_normals``' sum (``agree`` 0.23-0.41 where every
#: other row on this body is above 0.82) and their two widest faces 165-179°
#: apart, against Origin's own 0.76-0.92 and 52-91° in the same place — and
#: prescribed replacing the construction.  Splitting the curve **meets both of
#: those targets and misses the point**:
#:
#: ===========================  ==========  =============
#: female, ``--vs-shape``       ``0`` (ship) ``0.05``
#: ===========================  ==========  =============
#: crotch ``agree``             0.23-0.41   **0.83-0.94**
#: crotch ``fold``              165-179°    **98-123°**
#: crotch ``deg p50`` **vs Origin** **46 / 57°** 74 / 83°
#: whole body >30°              **3.0%**    3.4%
#: normal p90                   **16.0°**   16.2°
#: ===========================  ==========  =============
#:
#: The mechanism is in the last two rows: a shared crotch vertex's normal is
#: the **residue** of two opposing walls, which points out of the slit — and
#: that is roughly where Origin's valley normal points, so it is accidentally
#: closer to right than either wall's own honest normal is.  Splitting makes
#: each side shade from its own surface and therefore **further from Origin**,
#: because the surface it shades honestly is the wrong surface.
#:
#: What says so directly is ``--forks``' ``turn`` — the widest angle between
#: the two children's *own* faces either side of the band, which is the one
#: column vertex splitting cannot buy.  It is **unchanged**: 179 / 149 / 126 /
#: 162° on the foot and 169 / 163 / 114° on the hand at every width from 0 to
#: 1, and at 0 it reproduces the shared chord's own dihedral to 0.1°.  The fin
#: is **geometric, not topological** — it is where the two children's first
#: rings are, not what joins them — so it is not for this function to fix.
FORK_GUSSET = 0.0

#: How close to one of the mother ring's own vertices a fork's crotch curve has
#: to land before it **is** that vertex, as a fraction of the edge it sits on.
#: One number, because :func:`fork_faces` places the ends and checks the result
#: against the same tolerance; two would leave a band in which one adds a vertex
#: the other calls coincident (§7.307, and §7.306 for what the band costs).
CROTCH_SNAP = 0.01

#: How finely :func:`fork_faces` divides the two crotch curves, **in the mother
#: ring's own edges** — the unit, because the ring is the density this
#: construction already chose and a length in millimetres would price something
#: that misjudges on a hand and a foot alike (七).  ``0`` is off, i.e. one edge
#: from transition to transition however long it is.
#:
#: **What it was for, and why that reason is gone** (§7.322).  From §7.305 to
#: §7.321 there was **one** chord and both children's loops closed on it, so it
#: carried exactly two faces — 20.7 mm long on the foot's first fork, reaching
#: to **one** vertex of a child ring whose own edges are 2 mm.  §7.310 read that
#: as the cause of the fork ring's normals (the sail is 47-68% of their incident
#: area) and prescribed dividing it; §7.314 priced the prescription and refused
#: it (``--vs-shape``'s >30° 3.3% -> 4.2%, the foot's worst distance 2.24 ->
#: 7.56 mm, and **not one vertex that was already wrong got better**); §7.321
#: said why by construction — divided or not, the two faces on each piece still
#: share an edge and are still back to back.  There is no shared edge any more,
#: so this now only sets how many quads the gusset band is made of.
#:
#: Left switchable so both negative results stay reproducible (§7.278):
#: ``tools/body_profile.py --chord-cells N --chord-pull P`` on any ruler here.
FORK_CHORD_CELLS = 0.0

#: How far each of those divisions is pulled off its straight curve and onto
#: the crotch between the two children, ``0`` straight and ``1`` all the way.
#:
#: A straight crotch curve is a chord of the mother's **section**, so its
#: interior runs through the flesh and not along any surface: measured against
#: Origin at the foot's four forks, the single chord's middle was **7.97 / 5.01 /
#: 1.65 / 0.64 mm** off the skin while both its corners were on it to 0.0-0.4
#: (§7.314).  The target is the midpoint of the station's two nearest points on
#: the two children's first rings — our own geometry, no table, and the one
#: place the crotch can be that is between the two children rather than inside
#: either.  Scored against Origin at the foot's four forks, the interior stations'
#: median distance to the skin was **5.70 / 3.60 / 0.86 / 0.28 mm** straight
#: against **1.77 / 0.55 / 0.46 / 0.62** pulled.
#:
#: It does nothing while :data:`FORK_CHORD_CELLS` is ``0``: with one segment a
#: curve has no interior station to pull.
FORK_CHORD_PULL = 0.0

#: How far a fork's crotch chord is pulled off the two children's bisector and
#: into the **lane** between their first rings — ``0`` the bisector, which
#: shipped from §7.300 to §7.325, and ``1`` the middle of the lane, which is
#: what ships now (§7.326).
#:
#: The chord is where the mother's ring stops being one child's and starts
#: being the other's, so where it sits decides which way each wall of the
#: crotch leans, and :func:`fork_split` used to put it on the bisector of the
#: two children's centroids.  On a fork whose two children are different sizes
#: that lands **inside the bigger one**: on the foot's first fork child a's own
#: first ring reached 8.23 mm past the chord while child b's stopped 8.65 mm
#: short of it, so both walls leaned the same way and ``turn`` was over 90° **by
#: construction** while Origin's own crotch is 52-91° (§7.324/§7.325).
#:
#: The lane is the room between child b's near edge and child a's, measured
#: across the chord in the mother's plane, and §7.325's criterion is
#: ``|off| <= lane/2`` — the chord no further from the middle of that lane than
#: half its width, which is what puts the two walls either side of it.  Six of
#: this body's seven forks failed it by 4.0 to 46 times, and the lanes are
#: narrow enough (0.08-3.67 mm) that **only a pull of ~1 satisfies it**: at
#: anything less the chord is still outside the lane it is being pulled into.
#: So this is not a value fitted to an acceptance column — it is the one
#: setting the criterion admits, and ``off`` reading 0 at it is arithmetic.
#:
#: **What it buys** (female, §7.326; the male body reads the same angles column
#: for column and the same lengths times its stature):
#:
#: ==============================  ==========  ==========
#: ruler                            ``0``       ``1``
#: ==============================  ==========  ==========
#: ``sides``, seven forks           5 same      **7 opp**
#: ``turn`` hand #1 / #2            169 / 163   **104 / 107**
#: ``turn`` foot #2                 149         **111**
#: edges over 120°, foot / hand     6 / 23      **3 / 19**
#: ``--vs-shape`` >30° whole body   3.0%        **2.8%**
#: ``--vs-shape`` >30° foot / hand  6.2 / 6.3%  **5.6 / 5.6%**
#: ``--gaps`` >2 mm foot / hand     1.5 / 1.9%  **0.6 / 1.7%**
#: ``--creases`` over 25°           4082        **4048**
#: ``--foot --toes`` 16 cells v Origin  0.564 mm    **0.509**
#: ==============================  ==========  ==========
#:
#: and **what it costs**: two vertices and four triangles a foot, and one strip
#: of the picture.  Against Origin's own mesh through the same camera, of the pixels
#: this moves the hand is closer on all four angles (492 px on the back view,
#: 324 closer) and so is the foot from the front (1013 px, 609 closer, mean 79
#: -> 52 levels), but the foot's side view is **further** — 446 px of which 406,
#: mean 21 -> 65 levels, and it survives a 25° change of light so it is not a
#: ramp threshold reading.  It is not a new fold either (``--creases`` counts
#: fewer, and the turning radius does not move).  **What it is has not been
#: measured** — a light/dark step across the toe's dorsal surface where Origin's is
#: flat, and two of the three obvious explanations are already ruled out.
#:
#: ``turn`` still does not reach Origin's 52-91° anywhere, and §7.325 said why: the
#: chord is one of two numbers and the other one is the lane's **width**, which
#: is where the two children's first rings are and not where the chord is.
FORK_LANE_PULL = 1.0


# --------------------------------------------------------------------------
# topology helpers
# --------------------------------------------------------------------------
def grid_indices(nu: int, nv: int, wrap_u: bool = False, wrap_v: bool = False) -> np.ndarray:
    """Triangle indices for an (nu x nv) vertex grid, index = i*nv + j."""
    iu = np.arange(nu if wrap_u else nu - 1)
    iv = np.arange(nv if wrap_v else nv - 1)
    if len(iu) == 0 or len(iv) == 0:
        return np.zeros((0, 3), dtype=np.uint32)
    I, J = np.meshgrid(iu, iv, indexing="ij")
    I2 = (I + 1) % nu
    J2 = (J + 1) % nv
    a = (I * nv + J).ravel()
    b = (I2 * nv + J).ravel()
    c = (I2 * nv + J2).ravel()
    d = (I * nv + J2).ravel()
    tris = np.empty((len(a) * 2, 3), dtype=np.uint32)
    tris[0::2, 0], tris[0::2, 1], tris[0::2, 2] = a, b, c
    tris[1::2, 0], tris[1::2, 1], tris[1::2, 2] = a, c, d
    return tris


def _row_extent(pts: np.ndarray) -> float:
    """How far apart a grid row's vertices are -- its bounding box diagonal."""
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def surface(fn, nu: int, nv: int, wrap_u: bool = False, wrap_v: bool = False,
            name: str = "surface", uv_scale=(1.0, 1.0)) -> Mesh:
    """Build a mesh from a vectorised ``fn(U, V) -> (nu, nv, 3)`` parametrisation.

    ``U`` spans [0,1); when ``wrap_u`` is false it spans [0,1].

    **A row the parametrisation collapses is one vertex, not nu of them**
    (§7.313).  A sphere's pole is ``fn(u, 0)`` for every ``u``: the grid puts
    nu coincident vertices there, and every quad touching them comes out as one
    real triangle plus one with two corners in the same place.  Those zero-area
    faces are §7.308's family — invisible to ``compute_normals`` because it is
    area weighted, and read as up to 180° by anything that normalises a face
    before taking a dihedral — and the moment anything welds by position they
    are non-manifold as well: measured on one hand's five nails, the pole's own
    edge is used by **eight** faces and each spoke by **four**, 90 bad edges and
    160 zero-area faces on the shipped body (§7.307/§7.312 counted them).

    Collapsing is the rule ``loft`` has taken for its caps since §7.304 and
    ``fork_split`` for a crossing that lands on a mother vertex since §7.307:
    **a point that is already there is used, not duplicated.**  The cost is the
    pole's UV — one ``u`` instead of nu, so a texture is pinched at the pole
    rather than fanned, which is the standard trade for a UV sphere and matters
    to nothing built here (nails and props carry a flat colour).
    """
    u = (np.arange(nu) / nu) if wrap_u else (np.arange(nu) / max(nu - 1, 1))
    v = (np.arange(nv) / nv) if wrap_v else (np.arange(nv) / max(nv - 1, 1))
    U, V = np.meshgrid(u, v, indexing="ij")
    P = np.asarray(fn(U, V), dtype=np.float64).reshape(nu, nv, 3)
    mesh = Mesh(name)
    mesh.positions = P.reshape(-1, 3).astype(np.float32)
    uvs = np.stack([U * uv_scale[0], V * uv_scale[1]], axis=-1)
    mesh.uvs = uvs.reshape(-1, 2).astype(np.float32)
    n = len(mesh.positions)
    mesh.colors = np.ones((n, 4), dtype=np.float32)
    mesh.bone_idx = np.zeros((n, 4), dtype=np.int32)
    mesh.bone_wt = np.zeros((n, 4), dtype=np.float32)
    mesh.bone_wt[:, 0] = 1.0
    tris = grid_indices(nu, nv, wrap_u, wrap_v).astype(np.int64)
    # **The threshold is not a choice, because the distribution is bimodal**
    # (§7.313 measured it over every shape here that goes through this).
    # Equality alone will not do: one end of a sphere is exact (``sin(0)`` is 0
    # and the whole row is the same float) and the other is not (``sin(pi)`` is
    # 1.2e-16), and a superellipsoid raises that to a fractional power, so its
    # poles land at 4.8e-6 (``round_=0.3``) and 7.2e-5 (``round_=0``).  Against
    # the shape's own bounding box, that is the whole of one group -- 0,
    # 1e-16, 5e-6, 7e-5 -- and **the next row up is 0.164**.  Three orders of
    # magnitude of empty space, so anything from 1e-3 to 1e-2 picks the same
    # set; 1e-3 is 14x above the largest collapsed row and 164x below the
    # smallest real one.
    grid = np.asarray(mesh.positions, dtype=np.float64).reshape(nu, nv, 3)
    span = float(np.linalg.norm(grid.reshape(-1, 3).max(axis=0)
                                - grid.reshape(-1, 3).min(axis=0)))
    tol = 1e-3 * max(span, 1e-30)
    remap = np.arange(nu * nv, dtype=np.int64).reshape(nu, nv)
    if nu > 1:
        for j in range(nv):
            if _row_extent(grid[:, j, :]) <= tol:
                remap[:, j] = remap[0, j]
    if nv > 1:
        for i in range(nu):
            if _row_extent(grid[i, :, :]) <= tol:
                remap[i, :] = remap[i, 0]
    flat = remap.ravel()
    if not np.array_equal(flat, np.arange(nu * nv)):
        tris = flat[tris]
        tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
                    & (tris[:, 2] != tris[:, 0])]
    mesh.indices = tris.ravel().astype(np.uint32)
    mesh.normals = np.zeros((n, 3), dtype=np.float32)
    mesh.drop_unused()
    mesh.compute_normals()
    return mesh


#: **A measurement tap on the stations, off by default.**
#:
#: §7.310 recovered a loft's stations from the mesh alone — a ring is planar, so
#: vertices that share one share an axial coordinate — and §7.316 read the foot
#: with it.  That works where the rings are cut at a **constant** coordinate.
#: They are not, in general: a hand's six tubes point six ways (§7.317 measured
#: 798 rows for 967 vertices along x), and even per-run the ring's plane is
#: perpendicular to the chain's tangent through a centre ``_limb_offset`` has
#: moved (§7.222), so projecting onto the chain blurs a ring by 2.5 mm where the
#: rings are 0.24 mm apart — ten times too coarse, and the ruler says so rather
#: than picking a tolerance that makes it look answered.
#:
#: So the station comes from where stations exist.  When this is a list, every
#: loft appends ``(name, rings)`` to it — the very point sets it was handed, in
#: the order it lays them out — and a consumer maps a finished vertex back by
#: **position**, which survives ``weld``'s trim and per-position weld untouched.
#: Nothing on the drawing path ever turns it on; ``build_body_mesh`` does, only
#: when its caller passed a ``record``.
RING_TAP: list | None = None


#: **A ring's sample that lands on its neighbour is one vertex, not two**
#: (§7.339) — the tolerance, as a fraction of this loft's own bounding-box
#: diagonal.  ``0`` is bit-for-bit the construction before §7.339, so the
#: negative result stays reproducible (§7.278).
#:
#: :func:`surface` has taken the same rule since §7.313 for a **whole row** the
#: parametrisation collapses (a UV sphere's pole); this is the local version —
#: a ring can degenerate along one arc while the rest of it is a real curve, and
#: index space says nothing either way.  The trunk's rim is where it happens:
#: below the junction Origin's body is two separate thighs, so
#: :func:`polylode.mesh.body._bridge_gaps` walks the rays that find no thigh
#: **to the axis** (§7.219, and that claim is measured and right), and the grid
#: then puts one vertex there per ray — four on the female body, eight on the
#: male, all of them at ``(0, y, cz)``.
#:
#: **The threshold is not a choice, because the distribution is bimodal** —
#: measured over all 41 lofts this body builds, every ring-neighbour pair
#: (§7.339): the gaps are **0.0** (the trunk's rim, twice female / six times
#: male) and then **9.32e-4**, which is a real 0.142 mm edge on the foot —
#: §7.266's degenerate ring pair, a defect to be measured and not a duplicate
#: to be welded away.  Nothing in between, so anything from 1e-8 to 1e-4 picks
#: the same set; 1e-5 is 93x below the smallest real gap and about 200x above
#: what ``float32`` can resolve at the trunk's own scale.
#:
#: ``surface``'s 1e-3 would take the foot's pair as well, and that is not an
#: inconsistency: **the two thresholds are on two different quantities** — a
#: whole row's extent there, one neighbour's gap here — so their bimodal gaps
#: are different numbers (七: two things called mm).
LOFT_COLLAPSE = 1e-5

#: **What the collapse above took, so the two kinds of orphan stay apart.**
#:
#: ``build_body_mesh``'s closing ``drop_unused`` already reports what it drops
#: (一之一), and after §7.339 that count has two sources: the vertices
#: :func:`polylode.mesh.weld.weld`'s trim leaves behind, and the ring samples
#: this collapse merged.  One number for two causes is the thing 七 keeps
#: naming — the first is a defect somebody may want to chase and the second is
#: this construction working — so when this is a list every loft appends
#: ``(name, samples merged)`` to it.  Off by default; nothing on the drawing
#: path turns it on.
COLLAPSE_TAP: list | None = None


def _collapse_ring_runs(mesh: Mesh, P: np.ndarray, wrap_ring: bool) -> None:
    """Merge runs of coincident samples **within a ring**; :data:`LOFT_COLLAPSE`.

    ``P`` is ``(k, nv, 3)`` — the layout :func:`grid_indices` addresses as
    ``i*nv + j`` — and this rewrites ``mesh.indices`` in place, dropping the
    triangles the merge degenerates.

    **The vertices stay where they are.**  A caller addresses a loft's grid
    *arithmetically*: :func:`polylode.mesh.body.build_foot` and ``build_hand``
    hand :func:`fork_faces` ``base + i*nv + j`` for the rings a fork joins, so a
    compaction here would shift those — and the symptom is a fork drawn in the
    wrong place, not an error.  The samples that lose their triangles are picked
    up by ``build_body_mesh``'s own ``drop_unused``, which already reports what
    it drops (一之一) and now says which of the two reasons each came from.

    What this does **not** cover is a centroid disc over a collapsed ring:
    ``_fan_cap`` appends its own copy of the rim (§7.304 shares only the apex
    kind), and those copies are past ``k*nv`` so the remap never sees them.
    Measured, no ring this body caps has a coincident pair at all — the only
    ones are the trunk's rim, and the trunk's loft caps neither end.
    """
    tol = float(LOFT_COLLAPSE)
    k, nv = int(P.shape[0]), int(P.shape[1])
    if tol <= 0.0 or k < 2:
        return
    pts = P.reshape(-1, 3)
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    d = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=2)      # (k, nv)
    if not wrap_ring:
        d[-1, :] = np.inf
    join = d < tol * max(span, 1e-30)
    if not join.any():
        return
    remap = np.arange(k * nv, dtype=np.int64).reshape(k, nv)
    for j in np.unique(np.nonzero(join)[1]):
        col = remap[:, j].copy()
        for i in range(k):
            if join[i, j]:
                a, b = int(col[i]), int(col[(i + 1) % k])
                if a != b:
                    col[col == b] = a
        remap[:, j] = col
    flat = remap.ravel()
    if COLLAPSE_TAP is not None:
        COLLAPSE_TAP.append((mesh.name, int((flat != np.arange(k * nv)).sum())))
    tri = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
    grid = tri < k * nv
    tri = np.where(grid, flat[np.clip(tri, 0, k * nv - 1)], tri)
    keep = ((tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2])
            & (tri[:, 2] != tri[:, 0]))
    mesh.indices = tri[keep].ravel().astype(np.uint32)
    mesh.compute_normals()


def loft(rings: list[np.ndarray], wrap_ring: bool = True, cap_start: bool = False,
         cap_end: bool = False, name: str = "loft", vs: np.ndarray | None = None,
         end_apex: np.ndarray | None = None,
         start_apex: np.ndarray | None = None) -> Mesh:
    """Connect a sequence of equal-length rings into a tube.

    ``rings[i]`` is an (K,3) array.  ``vs`` optionally supplies the V texture
    coordinate per ring (defaults to evenly spaced).
    """
    nv = len(rings)
    if nv < 2:
        return Mesh(name)
    k = len(rings[0])
    P = np.stack([np.asarray(r, dtype=np.float64) for r in rings], axis=0)  # (nv, k, 3)
    P = np.transpose(P, (1, 0, 2))                                          # (k, nv, 3)
    mesh = Mesh(name)
    mesh.positions = P.reshape(-1, 3).astype(np.float32)
    u = (np.arange(k) / k) if wrap_ring else (np.arange(k) / max(k - 1, 1))
    v = np.asarray(vs, dtype=np.float64) if vs is not None else np.arange(nv) / max(nv - 1, 1)
    U, V = np.meshgrid(u, v, indexing="ij")
    mesh.uvs = np.stack([U, V], axis=-1).reshape(-1, 2).astype(np.float32)
    n = len(mesh.positions)
    mesh.colors = np.ones((n, 4), dtype=np.float32)
    mesh.bone_idx = np.zeros((n, 4), dtype=np.int32)
    mesh.bone_wt = np.zeros((n, 4), dtype=np.float32)
    mesh.bone_wt[:, 0] = 1.0
    if RING_TAP is not None:
        RING_TAP.append((name, np.transpose(P, (1, 0, 2)).copy()))
    tris = grid_indices(k, nv, wrap_ring, False)
    mesh.indices = tris.ravel().astype(np.uint32)
    mesh.normals = np.zeros((n, 3), dtype=np.float32)
    mesh.compute_normals()

    # **The cap shares the rim it closes** (§7.304).  It used to be a separate
    # ``_fan_cap`` mesh appended on, i.e. its own copy of the ring, and two
    # coincident copies of a vertex do not average their faces: the tube's last
    # ring took its normal from the tube alone and the cone took its from the
    # cone alone, so every capped end in this body carried a hard shading break
    # all the way round it.  That is the seam ``CLAUDE.md`` 七 already names —
    # "a cap's vertices do not share a normal with the ring it closes" — and
    # ``weld.vertex_ids`` has had to weld by position ever since, because the
    # raw index buffer had 71 components where the body has thirteen parts.
    #
    # Sharing costs one vertex per cap instead of ``k + 1`` and leaves the
    # triangles identical — measured, the whole body's triangle soup hashes the
    # same before and after; what changes is that ``compute_normals`` now
    # averages across the join.  Measured on its own (§7.304): overall more than
    # 30° off Origin's normal 5.3 → **4.9%**, ``seg hand`` 10.0 → 9.2%, ``seg foot``
    # 10.8 → 10.4%, and the five toe-tip caps' own 13 bad vertices down to 1.
    # **The ring the cap stands on is a separate defect and did not move** (the
    # cone onto a toe's apex is 1.61 mm across and 0.158 mm long, i.e. almost a
    # flat disc, where Origin's toe ends in a dome) — that is §7.244's purchase for
    # the fingertips, still unmade here.
    #
    # **Only the apex kind shares**, and that is the line ``_fan_cap``'s own
    # docstring already draws: an apex is "a surface that genuinely *ends*
    # there", so its normal belongs to the same smooth surface as the rim's,
    # while a centroid disc closes an *opening* and its rim is a real edge —
    # averaging there would bevel every cylinder in ``mesh/props.py``, which is
    # a change to the scenery nobody asked for.  Every apex-less cap on this
    # body is buried by construction (the arm's root and the leg's two ends sit
    # inside the trunk and the foot, the webs sit between the fingers, the
    # nipple's root inside the breast), so the gate costs the body five visible
    # vertices — the palm's root disc, which §7.230 already allows.
    if cap_start:
        _add_cap(mesh, P[:, 0, :], [i * nv for i in range(k)], True, start_apex)
    if cap_end:
        _add_cap(mesh, P[:, -1, :], [i * nv + nv - 1 for i in range(k)],
                 False, end_apex)
    if cap_start or cap_end:
        mesh.compute_normals()
    # Last, so that it covers the caps' triangles too: ``_cap_onto`` fans over
    # the grid's own ids, and a rim with a coincident pair would otherwise hand
    # the fan a zero-area triangle (§7.339).
    _collapse_ring_runs(mesh, P, wrap_ring)
    return mesh


def _add_cap(mesh: Mesh, ring: np.ndarray, ids: list[int], flip: bool,
             apex: np.ndarray | None) -> None:
    if apex is None:
        mesh.append(_fan_cap(ring, flip=flip))
    else:
        _cap_onto(mesh, ids, flip=flip, apex=apex)


def _cap_onto(mesh: Mesh, rim: list[int], flip: bool = False,
              apex: np.ndarray | None = None) -> None:
    """Close *rim* — vertex ids **already in** *mesh* — with a fan, in place.

    The same fan :func:`_fan_cap` makes, minus the copy of the rim.  Only the
    apex is a new vertex; its UV is the rim's own ``v`` at the rim's mid ``u``,
    which keeps the mesh's UV bounding box exactly what the tube's was (a fan
    over a wrapped rim is degenerate in UV either way, and the copy's UVs were
    all zero, which pinned the box's corner).
    """
    pos = np.asarray(mesh.positions, dtype=np.float64)
    ids = np.asarray(rim, dtype=np.int64)
    center = (pos[ids].mean(axis=0) if apex is None
              else np.asarray(apex, dtype=np.float64))
    c = len(pos)
    mesh.positions = np.vstack([pos, center[None, :]]).astype(np.float32)
    mesh.normals = np.vstack([mesh.normals,
                              np.zeros((1, 3), dtype=np.float32)])
    uv = np.asarray(mesh.uvs, dtype=np.float64)
    mesh.uvs = np.vstack([uv, [[0.5, uv[ids, 1].mean()]]]).astype(np.float32)
    mesh.colors = np.vstack([mesh.colors, np.ones((1, 4), dtype=np.float32)])
    mesh.bone_idx = np.vstack([mesh.bone_idx,
                               np.zeros((1, 4), dtype=np.int32)])
    wt = np.zeros((1, 4), dtype=np.float32)
    wt[0, 0] = 1.0
    mesh.bone_wt = np.vstack([mesh.bone_wt, wt])
    for s, width in Mesh.OPTIONAL_STREAMS:
        cur = getattr(mesh, s)
        if len(cur) == c:
            setattr(mesh, s, np.vstack([cur, np.zeros((1, width),
                                                      dtype=np.float32)]))
    n = len(ids)
    tris = [((c, int(ids[i]), int(ids[(i + 1) % n])) if not flip
             else (c, int(ids[(i + 1) % n]), int(ids[i]))) for i in range(n)]
    mesh.indices = np.concatenate(
        [np.asarray(mesh.indices, dtype=np.uint32),
         np.asarray(tris, dtype=np.uint32).ravel()]).astype(np.uint32)


def _fan_cap(ring: np.ndarray, flip: bool = False,
             apex: np.ndarray | None = None) -> Mesh:
    """Close a ring with a fan, at its centroid or at a given *apex*.

    Lifting the apex into a funnel was tried for the crotch cap and does not
    help: apex up, ring down and outward is a **bowl**, and a bowl's outward
    face still points down.  Measured, the downward area went 122.80 → 129.60
    cm² and the picture did not change (§7.184).

    *apex* is for the opposite case — a surface that genuinely **ends** there,
    where the centroid is the wrong point because the end is not in the ring's
    plane.  Then the fan is a cone onto the point the surface reaches, and the
    ring is where the last measurement is, not where the sweep ran out of
    table: :data:`polylode.mesh.body.DIGIT_TIP` closes all ten fingertips
    this way.  A flat disc there was the hand's worst row on every column —
    66-89% of the cap's normals more than 30° off Origin's, because a disc's normal
    is the axis while the surface it stands in for is a dome (§7.244).

    Both ends take one, because a foot has two of them: :data:`FOOT_APEX` is the
    back of the heel and the tip of the big toe, and neither is on the axis the
    sections are centred about — the toe's is 8.9 mm to the side (§7.245).
    """
    center = ring.mean(axis=0) if apex is None else np.asarray(apex,
                                                              dtype=np.float64)
    pts = np.vstack([center[None, :], ring])
    k = len(ring)
    tris = []
    for i in range(k):
        j = (i + 1) % k
        tris.append((0, i + 1, j + 1) if not flip else (0, j + 1, i + 1))
    m = Mesh("cap")
    m.positions = pts.astype(np.float32)
    m.uvs = np.zeros((len(pts), 2), dtype=np.float32)
    m.colors = np.ones((len(pts), 4), dtype=np.float32)
    m.bone_idx = np.zeros((len(pts), 4), dtype=np.int32)
    m.bone_wt = np.zeros((len(pts), 4), dtype=np.float32)
    m.bone_wt[:, 0] = 1.0
    m.indices = np.asarray(tris, dtype=np.uint32).ravel()
    m.normals = np.zeros((len(pts), 3), dtype=np.float32)
    m.compute_normals()
    return m


def fork_split(mother: np.ndarray, child_a: np.ndarray, child_b: np.ndarray):
    """Split a closed ring between two children — the crotch of a fork.

    A dict, always: ``ok`` plus ``owner``/``cut``/``frac`` when the mother's
    vertices fall into **two contiguous arcs**, and ``n_a``/``n_b``/``cuts``
    either way.  A partition with four transitions, or with one child owning
    nothing, is one this construction cannot realise — and the counts are
    reported rather than a single word, because "not two arcs" has three causes
    and they need different work (一之一, and §7.298 is the same lesson about
    ``_ray_ring``'s one-word ``None``).

    **That refusal is a sanity check, not the fork criterion**, and the
    difference matters because it would be easy to read as one.  Two Voronoi
    cells split a convex ring into two arcs *whatever* the two children are:
    handed two rings sitting **inside** the mother's own outline — §7.283's
    disc-with-holes, which is precisely the case a fork cannot describe — this
    returns a perfectly tidy two-arc split and a nonsense surface.  Nothing here
    can tell those apart, because both are the same partition; what tells them
    apart is that the section **pinches**, which is a fact about the station and
    is measured elsewhere: ``--fork`` bisects the stations where the cut stops
    being one closed curve (§7.284) and prices which of them one ring can carry
    (§7.298/§7.299).  The caller supplies a fork; this draws it.

    *owner* is ``True`` for the vertices nearest *child_a* — nearest that
    child's **centroid**, which is the bisector, and the test ``--fork``'s
    ``on`` column makes on the digit axes.  Nearest *surface* is a different
    question and the wrong one; the comment below measures what it answers.

    :data:`FORK_LANE_PULL` then moves the chord off that bisector and into the
    lane between the two children's first rings, which is what ships (§7.326).
    At ``0`` the lane pass does not run at all and this is the bisector alone —
    the construction that shipped to §7.325, bit for bit, so its readings stay
    reproducible (§7.278).

    *frac* is one number per transition: where along that edge the bisector
    crosses, by linear interpolation of the signed difference ``d(a) - d(b)``.
    It is where the two children pull apart — on a hand, the dorsal and palmar
    ends of the web's valley.

    **It is a position on the edge, not a vertex** (§7.322).  Until §7.321 this
    returned the crossing *point*, and both children's loops closed on the chord
    between the two of them: one edge with one face from each child, back to
    back, and thirteen vertices whose normals were what survived the
    cancellation (``agree`` 0.23-0.41 against Origin's 0.76-0.92 in the same place).
    :func:`fork_faces` now gives each child its own curve and puts a band
    between them, so what it needs from here is where to straddle — and the
    snap this used to return with it (§7.307: a crossing that lands on a mother
    vertex **is** that vertex) moved to the two ends it places, where the same
    rule still applies to each of them.
    """
    M = np.asarray(mother, dtype=np.float64)
    A = np.asarray(child_a, dtype=np.float64)
    B = np.asarray(child_b, dtype=np.float64)
    if len(M) < 3 or len(A) < 3 or len(B) < 3:
        return {"ok": False, "why": "a ring with fewer than three vertices"}
    # **The two children's centrelines, not their surfaces** — the distance to
    # the nearest vertex of a child *ring* is not the question.  Measured on the
    # hand's first fork: the trunk's remaining lobe is a 23 mm ring a fraction of
    # a millimetre away while the little finger's is a 5.6 mm ring 31 mm off to
    # the side, so nearest-surface hands **every** mother vertex to the trunk and
    # the fork refuses (§7.300).  Which child a vertex belongs to is which
    # centreline it is nearer — the bisector — which is the test ``--fork``'s
    # ``on`` column makes on the digit axes, and it is what makes the two rulers
    # the same question.
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    da = np.linalg.norm(M - ca[None], axis=1)
    db = np.linalg.norm(M - cb[None], axis=1)
    n = len(M)

    def split_on(g):
        """``(owner, cut, frac)`` for a signed field, ``> 0`` being child a.

        One spelling for the two fields this function can be asked about — the
        bisector's ``db - da`` and the lane's lateral — because the arcs and the
        crossings have to mean the same thing whichever placed the chord, and a
        rule stated at two call sites is a rule not stated (七).
        """
        own = g > 0.0
        cs = [i for i in range(n) if own[i] != own[(i + 1) % n]]
        if len(cs) != 2 or bool(own.all()) or not bool(own.any()):
            return None, cs
        fr = []
        for i in cs:
            j = (i + 1) % n
            g0, g1 = float(g[i]), float(g[j])
            fr.append(0.5 if abs(g1 - g0) < 1e-15
                      else float(np.clip(-g0 / (g1 - g0), 0.0, 1.0)))
        return (own, cs, fr), cs

    got, cut = split_on(db - da)
    out = {"ok": False, "n_a": int((da < db).sum()), "n_b": int((da >= db).sum()),
           "cuts": len(cut)}
    if got is None:
        out["why"] = ("one child owns the whole ring"
                      if len(cut) < 2 else f"{len(cut)} transitions, not two")
        return out
    owner, cut, frac = got

    # ---- the lane (:data:`FORK_LANE_PULL`) ----------------------------------
    # **Where the chord sits across the mother, rather than which child each
    # vertex is nearer.**  The bisector answers the second question, and on a
    # fork whose children differ in size the answer to it runs straight through
    # the bigger child: §7.325 measured the chord 0.39-8.57 mm off the middle of
    # the lane between the two children's first rings, against lane half-widths
    # of 0.04-1.84, so six of the seven forks have both walls of the crotch the
    # same side of it and ``turn > 90°`` by construction.
    #
    # The frame is the chord's own: the lateral is the direction across it in
    # the mother's plane, pointing at child a, which is the frame ``--forks``'
    # ``off``/``lane`` columns are printed in, so the two are the same number.
    # Both ends of the chord are at lateral 0 there by construction, so pulling
    # the split to ``pull * midline`` moves ``off`` by exactly that much and at
    # ``1`` leaves it at 0 — the calibration, and it can fail.
    #
    # **One pass, not a solve.**  The new chord lies on the level set of the
    # very lateral it was measured with, so it comes out parallel to the old
    # one and the frame it would be measured in again is the same frame.
    pull = float(FORK_LANE_PULL)
    if pull > 0.0:
        ends = np.stack([M[cut[k]] + (M[(cut[k] + 1) % n] - M[cut[k]])
                         * frac[k] for k in (0, 1)])
        u = ends[1] - ends[0]
        mc = M.mean(axis=0)
        nrm = np.cross(M - mc, np.roll(M, -1, axis=0) - mc).sum(axis=0)
        nu, nn = float(np.linalg.norm(u)), float(np.linalg.norm(nrm))
        w = np.cross(nrm / nn, u / nu) if nu > 0.0 and nn > 0.0 else u * 0.0
        nw = float(np.linalg.norm(w))
        if nw <= 0.0:
            out["lane_why"] = "the chord and the mother's plane are degenerate"
        else:
            mid = ends.mean(axis=0)
            w = w / nw
            if float(np.dot(ca - mid, w)) < 0.0:
                w = -w
            near_a = float(((A - mid) @ w).min())
            near_b = float(((B - mid) @ w).max())
            # **Reported as the bisector left them** — how wide the lane is
            # and how far outside it the chord started, which is the reason to
            # move and not the result of moving.  What the built surface ends
            # up with is ``--forks``' own two columns, measured off the mesh.
            out.update(lane=near_a - near_b, off=-0.5 * (near_a + near_b))
            moved, cs = split_on((M - mid) @ w - pull * 0.5 * (near_a + near_b))
            if moved is None:
                # The bisector's split stays — a fork that cannot be drawn at
                # all is worse than one drawn where it has always been — but it
                # is **said** (一之一), because a knob that silently does
                # nothing on some forks is a knob nobody can price.
                out["lane_why"] = (f"the lane's chord leaves {len(cs)} "
                                   f"transitions, not two")
            else:
                owner, cut, frac = moved
                out.update(n_a=int(owner.sum()), n_b=int((~owner).sum()),
                           cuts=len(cut))

    out.update(ok=True, owner=owner, cut=cut, frac=frac)
    return out


def _near_on_loop(p: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """The closest point to *p* on a closed polyline.

    A child's loft **starts** at the ring it is given here, so for a query
    proximal of it the closest point on the whole child is on this ring — which
    is why :func:`fork_faces` can ask this question with the ids it already
    has, instead of needing the child's surface.
    """
    a = np.asarray(ring, dtype=np.float64)
    ab = np.roll(a, -1, axis=0) - a
    t = np.clip(((p - a) * ab).sum(axis=1)
                / np.maximum((ab * ab).sum(axis=1), 1e-30), 0.0, 1.0)
    q = a + t[:, None] * ab
    return q[int(np.argmin(((q - p) ** 2).sum(axis=1)))]


def _drop_repeats(loop: list[int]) -> list[int]:
    """A closed loop with no vertex named twice running (§7.307).

    Cyclic, because the caller's loops are closed: ``_stitch`` walks them with
    ``% len``, so the last entry's neighbour is the first.
    """
    out = [i for k, i in enumerate(loop) if i != loop[k - 1]]
    return out if len(out) != 1 else loop[:1]


def _fork_arcs(owner, cut, n: int):
    """The two index runs a :func:`fork_split` leaves, in the mother's order."""
    i0, i1 = cut
    arc0 = [(i0 + 1 + k) % n for k in range((i1 - i0) % n)]
    arc1 = [(i1 + 1 + k) % n for k in range((i0 - i1) % n)]
    # ``arc0`` starts just past the first transition, so its owner is the one at
    # ``i0 + 1``; the loops are built in the mother's own winding either way.
    return (arc0, arc1) if owner[(i0 + 1) % n] else (arc1, arc0)


def _cut_edge(faces: list, u: int, v: int, w: int) -> None:
    """Split every face using edge ``(u, v)`` at *w*, in place then appended.

    *w* is a vertex **on** that edge, so this removes the T-junction it makes
    rather than covering it: the triangle count grows by one per face and the
    surface does not move at all (§7.312 — a face laid over the junction has
    three collinear corners by construction, no normal, and is a column nobody
    can read).

    **In place, then append** — every face already in the mesh keeps its index,
    so a caller holding a range from an earlier fork (there are three on a hand)
    still holds the same faces.  Growing in the middle would slide all of them.
    """
    for i in range(len(faces)):
        t = faces[i]
        if u in t and v in t:
            # Read the face in its own winding first, then split the side that
            # runs a -> b, so nothing here has to know which way the mother's
            # loft was wound.
            i0 = t.index(u)
            if t[(i0 + 1) % 3] == v:
                a, b, x = u, v, t[(i0 + 2) % 3]
            else:
                j0 = t.index(v)
                a, b, x = v, u, t[(j0 + 2) % 3]
            faces[i] = (a, w, x)
            faces.append((w, b, x))


def fork_faces(mesh: Mesh, mother_ids, child_a_ids, child_b_ids,
               info: dict | None = None):
    """Join one ring of *mesh* to two, in place — the trousers surface.

    Each child gets **its own crotch curve** across the mother's section, and a
    band of quads — the gusset — joins the two curves and closes on a piece of
    each transition edge.  So the surface between the two children is a real
    strip with a width, and no edge anywhere on the fork carries a face from
    both children.

    **That is the whole of §7.322, and it is what the shared chord was.**  From
    §7.305 the two loops both closed on one chord between the two crotch points:
    one edge, one face from each child, back to back.  Thirteen vertices (six on
    a hand, seven on a foot) therefore had 59-77% of their incident area cancel
    in ``compute_normals``' sum — ``agree`` 0.23-0.41 where every other row on
    this body is above 0.82 — and their two widest faces were 165-179° apart,
    while Origin's own surface at the same place reads 0.76-0.92 and 52-91° (§7.321
    measured both with one function).  Their **positions** were right the whole
    time (0.11 / 0.35 mm from Origin's skin), which is why no ruler scored in
    millimetres could see it and why four separate knobs came back negative:
    dividing the chord (§7.314), pulling it onto the crotch (§7.314), walking
    the children forward along their own axes (§7.320) and dropping the fork's
    faces from the sum (§7.318) all leave one edge with two faces on it.

    :data:`FORK_GUSSET` sets how much of a transition edge the band's end takes.
    The two curve ends land either side of the crossing, and each of them
    **snaps to the mother's own vertex** when it lands within
    :data:`CROTCH_SNAP` of it (§7.307: a crossing that lands on a vertex is that
    vertex, and a second one microns away is a zero-area triangle and a
    non-manifold edge as soon as anything welds by position).  What is left
    interior cuts the transition edge with :func:`_cut_edge`, which is the same
    closure §7.312 put in and for the same reason.

    Every vertex this adds is stated as **a weight over vertices the mesh
    already has**, so uvs, weights, colours and labels come along by
    construction — ``weld``'s reason for stitching between existing vertices,
    and the reason this can run before any of those are assigned.

    Returns ``(triangles added, why)`` — ``(0, reason)`` and nothing added when
    :func:`fork_split` refuses, when a loop comes out with fewer than three
    vertices, or when neither winding of a piece agrees with the surface.

    The strips and the band are wound by counting **directed-edge clashes**
    against the triangles already there (§7.240): a boundary edge is used once
    in one direction, so a face that closes onto it has to use it the other way,
    and a piece that is inside out shows up as a nonzero count rather than as a
    hole nobody can tell from a back-facing face.  The band is settled after the
    strips and against them, because it shares the two curves with them.
    """
    from .weld import _agree, _orient, _stitch

    pos = np.asarray(mesh.positions, dtype=np.float64)
    ma = np.asarray(mother_ids, dtype=int)
    n = len(ma)
    split = fork_split(pos[ma], pos[np.asarray(child_a_ids, dtype=int)],
                       pos[np.asarray(child_b_ids, dtype=int)])
    if not split["ok"]:
        return 0, f"{split.get('why')} ({split.get('n_a')}/{split.get('n_b')})"
    owner, cut, frac = split["owner"], split["cut"], split["frac"]
    arc_a, arc_b = _fork_arcs(owner, cut, n)
    if info is not None:
        info.update(arc_a=len(arc_a), arc_b=len(arc_b),
                    n_a=split["n_a"], n_b=split["n_b"],
                    lane=split.get("lane"), off=split.get("off"))
    if len(arc_a) < 1 or len(arc_b) < 1:
        return 0, "an empty arc"

    # ``ka`` is the transition arc_a starts just past, ``kb`` the one it ends
    # on; child b's arc is the other way round, which is what makes the band's
    # two ends the two transition edges and nothing else.
    ka = 0 if arc_a[0] == (cut[0] + 1) % n else 1
    kb = 1 - ka

    base = mesh.vertex_count
    new_pos: list = []
    new_src: list = []

    def point_at(k: int, t: float):
        """``(id, snapped)`` for the point at *t* along transition edge *k*."""
        u = int(ma[cut[k]])
        v = int(ma[(cut[k] + 1) % n])
        if t <= CROTCH_SNAP:
            return u, True
        if t >= 1.0 - CROTCH_SNAP:
            return v, True
        new_pos.append(pos[u] + (pos[v] - pos[u]) * t)
        new_src.append({u: 1.0 - t, v: t})
        return base + len(new_pos) - 1, False

    def at(i: int) -> np.ndarray:
        return pos[i] if i < base else new_pos[i - base]

    half = 0.5 * min(1.0, max(0.0, float(FORK_GUSSET)))
    pa: dict = {}
    pb: dict = {}
    to_cut: list = []
    snapped = 0
    for k in (ka, kb):
        s = float(frac[k])
        t_lo, t_hi = max(0.0, s - half), min(1.0, s + half)
        i_lo, s_lo = point_at(k, t_lo)
        # **At zero width the two ends are one point, and that is the
        # construction this replaced** — not a special case to keep out of the
        # way but the far end of the one knob (§7.278: a candidate that has
        # been ruled out has to stay askable, or the negative result is not
        # reproducible next time).  With ``FORK_GUSSET`` 0 both children's
        # loops close on the same chord, the band's quads are degenerate and
        # dropped, and what is left is exactly what shipped from §7.305 to
        # §7.321 — which is what lets one build of one ruler measure both.
        i_hi, s_hi = ((i_lo, s_lo) if t_hi <= t_lo else point_at(k, t_hi))
        # Ends, not calls: with no width the two are one end and counting it
        # twice would make the column read four where the fork has two.
        snapped += int(s_lo) + (0 if i_hi == i_lo else int(s_hi))
        # ``ka`` is where arc_a begins, so a owns that edge's far end; at ``kb``
        # arc_a ends, so a owns the near one.
        pa[k], pb[k] = ((i_hi, i_lo) if k == ka else (i_lo, i_hi))
        to_cut.append((int(ma[cut[k]]), int(ma[(cut[k] + 1) % n]),
                       None if s_lo else i_lo,
                       None if (s_hi or i_hi == i_lo) else i_hi))

    # ---- the two curves, divided (:data:`FORK_CHORD_CELLS`) -----------------
    seg0 = np.linalg.norm(pos[np.roll(ma, -1)] - pos[ma], axis=1)
    cell = float(np.median(seg0)) if len(seg0) else 0.0
    span = float(np.linalg.norm(at(pa[kb]) - at(pa[ka])))
    n_cut = (max(1, int(math.ceil(span / (FORK_CHORD_CELLS * cell) - 1e-9)))
             if FORK_CHORD_CELLS > 0.0 and cell > 0.0 else 1)
    kid_a = pos[np.asarray(child_a_ids, dtype=int)]
    kid_b = pos[np.asarray(child_b_ids, dtype=int)]
    src_of = {i: (dict(new_src[i - base]) if i >= base else {i: 1.0})
              for i in set(pa.values()) | set(pb.values())}
    mids_a: list = []
    mids_b: list = []
    # One curve when the band has no width: the two children share it, the way
    # they shared the chord before this.
    one = pa[ka] == pb[ka] and pa[kb] == pb[kb]
    for step in range(1, n_cut):
        s = step / n_cut                    # 0 at the ``kb`` end, 1 at ``ka``
        for lst, e0, e1 in (((mids_a, pa[kb], pa[ka]),) if one else
                            ((mids_a, pa[kb], pa[ka]),
                             (mids_b, pb[kb], pb[ka]))):
            q = at(e0) + s * (at(e1) - at(e0))
            if FORK_CHORD_PULL:
                crot = 0.5 * (_near_on_loop(q, kid_a) + _near_on_loop(q, kid_b))
                q = q + FORK_CHORD_PULL * (crot - q)
            w: dict = {}
            for d, g in ((src_of[e0], 1.0 - s), (src_of[e1], s)):
                for v_, h in d.items():
                    w[v_] = w.get(v_, 0.0) + g * h
            lst.append(base + len(new_pos))
            new_pos.append(q)
            new_src.append(w)

    if one:
        mids_b = mids_a
    curve_a = [pa[kb]] + mids_a + [pa[ka]]
    curve_b = [pb[kb]] + mids_b + [pb[ka]]

    # Each loop is its arc of the mother, the piece of transition edge at each
    # end, and its own crotch curve back — closed, in the mother's own winding,
    # and sharing no edge with the other child's.
    loop_a = _drop_repeats([pa[ka]] + [int(ma[i]) for i in arc_a] + [pa[kb]]
                           + mids_a)
    loop_b = _drop_repeats([pb[kb]] + [int(ma[i]) for i in arc_b] + [pb[ka]]
                           + mids_b[::-1])
    if len(loop_a) < 3 or len(loop_b) < 3:
        return 0, ("an arc of fewer than three vertices once the ends land "
                   f"({len(loop_a)}/{len(loop_b)})")

    extra = Mesh("crotch")
    if new_pos:
        extra.positions = np.asarray(new_pos, dtype=np.float32)

        def blend(src):
            return np.stack([
                sum(f * src[v].astype(np.float64) for v, f in w.items())
                for w in new_src]).astype(np.float32)

        for attr, width in (("uvs", 2), ("colors", 4), ("normals", 3)):
            src = getattr(mesh, attr)
            setattr(extra, attr, blend(src) if len(src)
                    else np.zeros((len(new_pos), width), dtype=np.float32))
        # Skin is **taken from one source, not mixed**: a per-slot lerp mixes
        # one chain's bone index with another's and lands on a third bone
        # (``weld._blend_skin`` exists for the junctions that have to), and
        # every source here is on one ring of one part.  Ties go to the first,
        # which is the mother vertex the edge runs from.
        pick = [max(w.items(), key=lambda kv: kv[1])[0] for w in new_src]
        extra.bone_idx = np.stack([mesh.bone_idx[v] for v in pick])
        extra.bone_wt = np.stack([mesh.bone_wt[v] for v in pick])
        for s_, width in Mesh.OPTIONAL_STREAMS:
            src = getattr(mesh, s_)
            setattr(extra, s_, blend(src) if len(src) == mesh.vertex_count
                    else np.zeros((0, width), dtype=np.float32))

    p2 = (np.vstack([pos, np.asarray(new_pos, dtype=np.float64)])
          if new_pos else pos)
    faces = [(int(a), int(b), int(c)) for a, b, c in
             np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)]
    before = len(faces)
    for u, v, i_lo, i_hi in to_cut:
        left = u
        if i_lo is not None:
            _cut_edge(faces, left, v, i_lo)
            left = i_lo
        if i_hi is not None:
            _cut_edge(faces, left, v, i_hi)

    strip_a = _stitch(loop_a, [int(i) for i in child_a_ids], p2)
    strip_b = _stitch(loop_b, [int(i) for i in child_b_ids], p2)
    strips = strip_a + strip_b
    band = []
    for i in range(len(curve_a) - 1):
        a0, a1 = curve_a[i], curve_a[i + 1]
        b0, b1 = curve_b[i], curve_b[i + 1]
        if a0 == b0 and a1 == b1:      # zero width — the two curves are one
            continue
        band.append((a0, a1, b1))
        band.append((a0, b1, b0))

    used = _orient(faces)
    clash = _agree(strips, used)
    if clash:
        flip = [(a, c, b) for a, b, c in strips]
        n2 = _agree(flip, used)
        if n2:
            return 0, f"both windings clash ({clash} and {n2} edges)"
        strips = flip
    used |= _orient(strips)
    clash = _agree(band, used) if band else 0
    if clash:
        flip = [(a, c, b) for a, b, c in band]
        n2 = _agree(flip, used)
        if n2:
            return 0, f"both gusset windings clash ({clash} and {n2} edges)"
        band = flip
    tris = strips + band
    if info is not None:
        info.update(snapped=snapped, curve_a=tuple(curve_a),
                    curve_b=tuple(curve_b), gusset=len(band),
                    strip_a=len(strip_a), strip_b=len(strip_b),
                    tri=(len(faces), len(faces) + len(tris)))

    # **The strip counts its own edges.**  A fork can be built and still not be a
    # surface — an undirected edge used by three faces renders, so nothing
    # downstream reports it, and it is the same class §7.240 had to count rather
    # than look at.  Cheap here and nowhere else: the two strips, the band and
    # the mother faces the ends cut are the only faces that touch these loops.
    seen: dict = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            k = (min(u, v), max(u, v))
            seen[k] = seen.get(k, 0) + 1
    for u, v in list(_orient(faces)):
        k = (min(u, v), max(u, v))
        if k in seen:
            seen[k] += 1
    bad = sum(1 for v in seen.values() if v > 2)

    # **And two vertices that coincide are the same failure one step earlier**
    # (§7.306, generalised in §7.307).  The count above is in *index* space,
    # where two coincident vertices are two vertices; the mesh only becomes
    # non-manifold once anything welds by position — which is what
    # ``weld.vertex_ids`` does for every topology test in this project, and what
    # a DCC or an exporter does.  A pair 0.006 mm apart gives a zero-area
    # triangle and an edge used four times, and every one of those is invisible:
    # the strip still closes, still renders, and ``why`` stayed empty while the
    # foot went from 0 non-manifold edges to 2.
    #
    # The snap above removes the one source this construction knows about — an
    # end that lands on a mother vertex — so what is left is **anything else
    # that happens to land on top of something**, and the honest check is over
    # the vertices the new faces actually use rather than over those ends alone:
    # a child's first ring can be moved (that is the knob §7.320 priced) until
    # one of *its* vertices meets a mother's.  The tolerance is
    # :data:`CROTCH_SNAP` of the mother ring's own edge — the same number the
    # snap uses, relative to the same thing (七: millimetres misjudge at another
    # scale), and coarser than ``weld.vertex_ids``' 1e-5 of stature so it speaks
    # a little *before* a weld would merge the pair.
    tol = CROTCH_SNAP * cell
    touch = ""
    uniq = sorted({int(i) for t in tris for i in t})
    if tol > 0 and len(uniq) > 1:
        q = p2[np.asarray(uniq, dtype=int)]
        d = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        if float(d.min()) < tol:
            i_, j_ = np.unravel_index(int(np.argmin(d)), d.shape)
            touch = (f"two of the fork's vertices coincide "
                     f"({1e3 * float(d[i_, j_]):.3f} mm apart, "
                     f"{100.0 * float(d[i_, j_]) / tol * CROTCH_SNAP:.2f}% of "
                     f"a mother edge)")
    if new_pos:
        mesh.append(extra)
    mesh.indices = np.concatenate(
        [np.asarray(faces, dtype=np.uint32).ravel(),
         np.asarray(tris, dtype=np.uint32).ravel()]).astype(np.uint32)
    why = ""
    if bad:
        why = f"{bad} edges used more than twice"
    elif touch:
        why = touch
    # A lane pull that could not place the chord is reported even when the fork
    # is otherwise clean: it drew, and it drew somewhere the knob did not ask
    # for (一之一).
    if split.get("lane_why"):
        why = f"{split['lane_why']}{'; ' + why if why else ''}"
    # The count is the mesh's, not the strips': cutting a mother face adds one
    # triangle each time, and the caller asked how many were added.
    return len(tris) + len(faces) - before, why


def ensure_outward(mesh: Mesh, center=None) -> Mesh:
    """Flip winding if the surface normals mostly point at the centroid."""
    if mesh.tri_count == 0:
        return mesh
    c = mesh.positions.mean(axis=0) if center is None else np.asarray(center, dtype=np.float32)
    d = mesh.positions - c
    ln = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(ln, 1e-6)
    if float(np.sum(d * mesh.normals)) < 0.0:
        mesh.flip_winding()
        mesh.compute_normals()
    return mesh


# --------------------------------------------------------------------------
# rings / profiles
# --------------------------------------------------------------------------
def superellipse_ring(k: int, rx: float, rz: float, n: float = 2.0,
                      center=(0.0, 0.0, 0.0), phase: float = 0.0,
                      polar: bool = False) -> np.ndarray:
    """Ring of k points in the XZ plane; n=2 is an ellipse, larger is boxier.

    ``polar`` spaces the points by **polar angle about the centre** instead of
    along the curve's own parameter.  The two are the same curve and agree only
    for a circle: measured on the hip section (rx 137.4, rz 83.5, n 4.874 mm),
    one parameter step in the middle of the flat back face covers **39.58°** of
    polar angle, so a 34-point ring has no vertex anywhere between 230.42° and
    270.00° — and whatever the outline table carries in there becomes a single
    straight chord.  At the gluteal cleft that chord cost 45 mm of buttock
    (§7.208).  Ratio tables are measured on rays at uniform polar angle, so a
    loft that consumes one has to sample the same way.
    """
    t = np.arange(k) / k * 2.0 * math.pi + phase
    ct, st = np.cos(t), np.sin(t)
    if polar:
        # the same implicit curve |x/rx|^n + |z/rz|^n = 1, solved for the
        # radius along each ray instead of stepped along the parameter
        r = (np.abs(ct / max(rx, 1e-12)) ** n
             + np.abs(st / max(rz, 1e-12)) ** n) ** (-1.0 / n)
        x, z = r * ct, r * st
    else:
        e = 2.0 / max(n, 0.2)
        x = rx * np.sign(ct) * np.abs(ct) ** e
        z = rz * np.sign(st) * np.abs(st) ** e
    ring = np.stack([x, np.zeros_like(x), z], axis=-1)
    return ring + np.asarray(center, dtype=np.float64)


def circle_ring(k: int, r: float, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    return superellipse_ring(k, r, r, 2.0, center)


def smooth_profile(keys: list[tuple[float, float]], t: np.ndarray) -> np.ndarray:
    """Interpolate (x, y) key points with a **slope-continuous** curve.

    This was piecewise smoothstep, and smoothstep has **zero slope at both
    ends** — so every key was a stationary point.  Twenty-four keys therefore
    became twenty-four little plateaus with steep runs between them, and where
    the profile turns over (the trunk table has a local maximum at t = 0.1042:
    0.09230 → 0.09690 → 0.09584) the two runs meet at an angle.  Measured on
    the hip, the sharpest face-to-face angle on our body was **53.3°** at 0.549
    of stature, and that is what shades as the line reported under the buttocks
    (§7.184).

    Catmull-Rom gives each key the slope of the chord through its neighbours,
    so the curve carries on through a key instead of flattening at it.  Ends
    keep the one-sided slope; outside the range it still clamps, which is what
    ``body._extend_to_zero`` relies on.

    **The slopes are then limited so the curve cannot leave the data**
    (Fritsch–Carlson).  A chord slope is a fact about the neighbours, not about
    the span, so where the profile bends sharply the cubic can bulge past both of
    its own keys — and the trunk table's depth falls 37% over its last span.

    **Measured, this changes nothing on the tables we have**: at most 0.05% on
    any of the twenty-four trunk stations (§7.196 三).  It went in while chasing
    a +16.6% ring at the neck and did not explain it — that was the outline
    table, not this — so it is a guard against a class of error, not a fix for an
    observed one, and nothing may be credited to it.  It introduces no tuned
    number (the condition is the necessary-and-sufficient monotonicity region for
    a Hermite cubic) and it leaves §7.184's reason for dropping smoothstep
    untouched: slopes stay continuous, so no key becomes a plateau.
    """
    ks = sorted(keys)
    xs = np.array([k[0] for k in ks], dtype=np.float64)
    ys = np.array([k[1] for k in ks], dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = len(xs)
    if n < 2:
        return np.full(np.shape(t), ys[0] if n else 0.0)
    idx = np.clip(np.searchsorted(xs, t) - 1, 0, n - 2)
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    h = np.maximum(x1 - x0, 1e-9)
    lt = np.clip((t - x0) / h, 0.0, 1.0)

    # one-sided at the ends, centred elsewhere — the chord slope through the
    # neighbours, scaled into this span's parameterisation
    slope = np.zeros(n)
    slope[1:-1] = (ys[2:] - ys[:-2]) / np.maximum(xs[2:] - xs[:-2], 1e-9)
    slope[0] = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-9)
    slope[-1] = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-9)

    # Fritsch–Carlson: on a span where the data does not turn over, keep the
    # cubic inside it.  A flat span pins both of its slopes to zero; otherwise
    # the pair is pulled back onto the circle of radius 3 in units of the span's
    # own secant, which is the necessary-and-sufficient monotonicity region.
    sec = (ys[1:] - ys[:-1]) / np.maximum(xs[1:] - xs[:-1], 1e-9)
    for k in range(n - 1):
        if abs(sec[k]) < 1e-12:
            slope[k] = slope[k + 1] = 0.0
            continue
        a, b = slope[k] / sec[k], slope[k + 1] / sec[k]
        if a < 0.0:
            a = 0.0
        if b < 0.0:
            b = 0.0
        q = a * a + b * b
        if q > 9.0:
            tau = 3.0 / (q ** 0.5)
            a, b = tau * a, tau * b
        slope[k], slope[k + 1] = a * sec[k], b * sec[k]
    m0 = slope[idx] * h
    m1 = slope[idx + 1] * h

    t2, t3 = lt * lt, lt * lt * lt
    return ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + lt) * m0
            + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * m1)


# --------------------------------------------------------------------------
# solids
# --------------------------------------------------------------------------


def ellipsoid(rx: float, ry: float, rz: float, segments: int = 24, rings: int = 16,
              name: str = "ellipsoid") -> Mesh:
    def fn(U, V):
        theta = U * 2.0 * math.pi
        phi = V * math.pi
        return np.stack([rx * np.sin(phi) * np.cos(theta),
                         ry * np.cos(phi),
                         rz * np.sin(phi) * np.sin(theta)], axis=-1)
    return ensure_outward(surface(fn, segments, rings, wrap_u=True, name=name), (0, 0, 0))


def box(sx: float = 1.0, sy: float = 1.0, sz: float = 1.0, name: str = "box") -> Mesh:
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    p = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float64)
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    verts, tris, uvs = [], [], []
    for f in faces:
        base = len(verts)
        for i, vi in enumerate(f):
            verts.append(p[vi])
            uvs.append([(i in (1, 2)) * 1.0, (i in (2, 3)) * 1.0])
        tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return Mesh.from_arrays(np.array(verts), np.array(tris, dtype=np.uint32).ravel(),
                            uvs=np.array(uvs), name=name)


def bevel_box(sx: float, sy: float, sz: float, bevel: float = 0.01,
              name: str = "bbox") -> Mesh:
    """A box with its edges chamfered by *bevel* — flat faces, so the toon
    shading and the outline catch every edge the way a plain box hides them."""
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    r = float(min(bevel, hx * 0.49, hy * 0.49, hz * 0.49))
    if r <= 0.0:
        return box(sx, sy, sz, name)

    def corner(a, b, c):
        return (np.array([a * hx, b * (hy - r), c * (hz - r)]),      # on the X face
                np.array([a * (hx - r), b * hy, c * (hz - r)]),      # on the Y face
                np.array([a * (hx - r), b * (hy - r), c * hz]))      # on the Z face

    v = {(a, b, c): corner(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)}
    quads, tris = [], []
    for a in (-1, 1):                    # the six faces
        quads.append([v[(a, -1, -1)][0], v[(a, 1, -1)][0], v[(a, 1, 1)][0], v[(a, -1, 1)][0]])
        quads.append([v[(-1, a, -1)][1], v[(1, a, -1)][1], v[(1, a, 1)][1], v[(-1, a, 1)][1]])
        quads.append([v[(-1, -1, a)][2], v[(1, -1, a)][2], v[(1, 1, a)][2], v[(-1, 1, a)][2]])
    for a in (-1, 1):                    # the twelve chamfers
        for b in (-1, 1):
            quads.append([v[(a, b, -1)][0], v[(a, b, -1)][1], v[(a, b, 1)][1], v[(a, b, 1)][0]])
            quads.append([v[(a, -1, b)][0], v[(a, -1, b)][2], v[(a, 1, b)][2], v[(a, 1, b)][0]])
            quads.append([v[(-1, a, b)][1], v[(-1, a, b)][2], v[(1, a, b)][2], v[(1, a, b)][1]])
    for key, (px, py, pz) in v.items():  # the eight corners
        tris.append([px, py, pz])
    verts, idx = [], []
    centre = np.zeros(3)
    for poly in quads + tris:
        base = len(verts)
        pts = [np.asarray(q, dtype=np.float64) for q in poly]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        mid = np.mean(pts, axis=0)
        if np.dot(n, mid - centre) < 0.0:          # wind every face outward
            pts = pts[::-1]
        verts.extend(pts)
        for k in range(1, len(pts) - 1):
            idx += [base, base + k, base + k + 1]
    return Mesh.from_arrays(np.array(verts), np.array(idx, dtype=np.uint32), name=name)


def rounded_box(sx: float, sy: float, sz: float, round_: float = 0.3,
                segments: int = 20, rings: int = 14, name: str = "rbox") -> Mesh:
    """A superellipsoid — useful for chunky shapes like shoes and bags."""
    n = m3.lerp(8.0, 2.0, np.clip(round_, 0.0, 1.0))

    def fn(U, V):
        theta = U * 2.0 * math.pi
        phi = V * math.pi - math.pi * 0.5
        e = 2.0 / n
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        f = lambda a, p: np.sign(a) * np.abs(a) ** p
        return np.stack([sx * 0.5 * f(cp, e) * f(ct, e),
                         sy * 0.5 * f(sp, e),
                         sz * 0.5 * f(cp, e) * f(st, e)], axis=-1)
    return ensure_outward(surface(fn, segments, rings, wrap_u=True, name=name), (0, 0, 0))


def cylinder(radius: float, height: float, segments: int = 20, name: str = "cyl",
             radius_top: float | None = None, caps: bool = True) -> Mesh:
    rt = radius if radius_top is None else radius_top
    rings = [circle_ring(segments, radius, (0, 0, 0)),
             circle_ring(segments, rt, (0, height, 0))]
    return ensure_outward(loft(rings, True, caps, caps, name), (0, height * 0.5, 0))


def torus(major: float, minor: float, seg_major: int = 32, seg_minor: int = 12,
          name: str = "torus") -> Mesh:
    def fn(U, V):
        a = U * 2 * math.pi
        b = V * 2 * math.pi
        r = major + minor * np.cos(b)
        return np.stack([r * np.cos(a), minor * np.sin(b), r * np.sin(a)], axis=-1)
    return surface(fn, seg_major, seg_minor, wrap_u=True, wrap_v=True, name=name)


def plane(sx: float = 1.0, sz: float = 1.0, nu: int = 2, nv: int = 2,
          name: str = "plane") -> Mesh:
    def fn(U, V):
        return np.stack([(U - 0.5) * sx, np.zeros_like(U), (V - 0.5) * sz], axis=-1)
    return surface(fn, nu, nv, name=name)


# --------------------------------------------------------------------------
# swept tubes with parallel-transport frames
# --------------------------------------------------------------------------
def frames_along(points: np.ndarray, up_hint=(0.0, 0.0, 1.0)):
    """Parallel-transport frames: returns (tangents, normals, binormals)."""
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    t = np.zeros_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    t = m3.normalize(t)
    nor = np.zeros_like(p)
    ref = np.asarray(up_hint, dtype=np.float64)
    if abs(np.dot(ref, t[0])) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    nor[0] = m3.normalize(ref - t[0] * np.dot(ref, t[0]))
    for i in range(1, n):
        v = nor[i - 1]
        v = v - t[i] * np.dot(v, t[i])
        ln = np.linalg.norm(v)
        nor[i] = v / ln if ln > 1e-8 else m3.normalize(np.cross(t[i], np.array([0.0, 1.0, 0.0])))
    bi = np.cross(t, nor)
    return t, nor, bi


def sweep_rings(points: np.ndarray, radii, sides: int = 12, squash=None,
                twist=None, up_hint=(0.0, 0.0, 1.0), ratio=None,
                frame_pts=None, polar: bool = True,
                frames=None) -> list[np.ndarray]:
    """The rings :func:`tube_along` would loft, without lofting them.

    Split out because a *neighbouring* part sometimes has to be asked what its
    own section is at a station — "does this cap hide inside that tube" is a
    question about the tube's **ring**, and a scalar radius cannot answer it
    once the section carries a squash and an outline ratio (§7.230's clamp
    compared two numbers and let three rays out; §7.289 measured 0.82 mm).
    One recipe with one owner, so the answer is the same geometry the sweep
    will actually build rather than a second spelling of it.

    ``polar`` spaces the ring by **polar angle about the centre** rather than
    along the ellipse's own parameter — :func:`superellipse_ring`'s flag, on
    this family of rings, and the same finding (§7.208).  It was fixed for the
    trunk's loft and not here, which is ``CLAUDE.md`` 七's "a trap that belongs
    to a class, fixed at one call site, is not fixed".

    The two spacings agree only on a circle, and they diverge by
    ``atan((1 - squash) / 2)``: on a leg or a digit (squash 0.85-0.95) that is
    2-4° against the tables' 15° rays and costs nothing, but the palm's fused
    section is the flattest on this body — ``r2/r1`` runs to **0.38** — and
    there the ring's twenty-four vertices land up to **24°** off the twenty-four
    rays the ratio table was measured on.  Only four of them sit on a ray at
    all; the other twenty read an interpolation of a table whose neighbouring
    entries run 0.24 and 1.19 (the first web space against the knuckles), so the
    extremes are cut.  Measured at ``PALM_SECTIONS``' last fused station, the
    ring came back **12.7% narrower than its own table** while Origin's own cut
    reconstructed on the same 24 rays keeps **98.7%** of it (``--fork``'s ray
    price list) — i.e. the loss was the sample *positions*, not the ray count,
    which is exactly the shape §7.208 diagnosed on the gluteal cleft ("an error
    that is zero at the sample points and large between them is a sample in the
    wrong place").

    A ring with no squash is unaffected by construction, so every circular tube
    on this body (and in ``props.py``) is bit-identical either way.

    ``frames`` supplies ``(nor, bi)`` per station instead of transporting them
    from *up_hint*, and it is what makes ``polar`` worth anything: uniform polar
    spacing puts the vertices on rays of **this** frame, and a ratio table's
    rays are in the frame it was measured in.  The two are not the same one —
    measured on the palm, the transported ``nor`` sits **171.8°** from the
    table's ``e1`` where anti-parallel would be 180, i.e. 8.2° out, which is
    0.55 of a ray step: every vertex between rays, and the ellipse's own axes
    turned 8.2° off the semi-axes the table stores.  Transport exists to keep a
    tube from twisting along a curved axis; a frame taken from the axis itself
    (``body._limb_frame``) does not accumulate twist either, and it is the one
    the numbers are in.
    """
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    r = (np.full(n, float(radii)) if np.isscalar(radii)
         else np.asarray(radii, dtype=np.float64))
    sq = np.ones(n) if squash is None else np.asarray(squash, dtype=np.float64)
    tw = np.zeros(n) if twist is None else np.asarray(twist, dtype=np.float64)
    fp = p if frame_pts is None else np.asarray(frame_pts, dtype=np.float64)
    if frames is None:
        _, nor, bi = frames_along(fp, up_hint)
    else:
        f = np.asarray(frames, dtype=np.float64)     # (n, 2, 3)
        nor, bi = f[:, 0, :], f[:, 1, :]
    a = np.arange(sides) / sides * 2 * math.pi
    rings = []
    for i in range(n):
        ang = a + tw[i]
        if polar:
            # The same ellipse, solved for its radius along each ray instead of
            # stepped along its parameter.
            ra, rb = max(abs(r[i]), 1e-12), max(abs(r[i] * sq[i]), 1e-12)
            rr = ((np.cos(ang) / ra) ** 2 + (np.sin(ang) / rb) ** 2) ** -0.5
            offs = (np.cos(ang)[:, None] * nor[i][None, :] * rr[:, None]
                    + np.sin(ang)[:, None] * bi[i][None, :] * rr[:, None])
        else:
            offs = (np.cos(ang)[:, None] * nor[i][None, :] * r[i]
                    + np.sin(ang)[:, None] * bi[i][None, :] * r[i] * sq[i])
        if ratio is not None:
            offs = offs * np.asarray(ratio(i, offs), dtype=np.float64)[:, None]
        rings.append(p[i][None, :] + offs)
    return rings


def tube_along(points: np.ndarray, radii, sides: int = 12, name: str = "tube",
               squash=None, caps=True, twist=None,
               up_hint=(0.0, 0.0, 1.0), ratio=None, frame_pts=None,
               end_apex=None, polar: bool = True, frames=None) -> Mesh:
    """Sweep a (possibly elliptical) ring along a polyline.

    ``radii``  scalar or per-point radius.
    ``caps``   ``True``/``False`` for both ends, or ``(start, end)``.  The two
               ends of a limb are not the same question and one of them can be
               unanswerable: a cap is only allowed where nothing can see it
               (§7.230), and where **two** tubes meet end to end neither cap can
               be hidden, because each one needs the other part to be wider than
               itself at the very station where they are the same section
               (§7.285 measured that at the wrist).  ``loft`` has always taken
               the two separately; this signature was what collapsed them.
    ``squash`` optional per-point Z/X ratio (1.0 = circular).
    ``twist``  optional per-point twist in radians.
    ``ratio``  optional ``fn(i, offs) -> (sides,)`` multiplier, applied radially
               to ring *i*'s offsets from the centreline.  It is handed the
               finished offsets rather than the ring angle because an ellipse's
               ring parameter is **not** its polar angle (they differ by up to
               ``atan((1-squash)/2)``), and a table of residuals against the
               ellipse is indexed by the polar one.  Passing the vectors also
               keeps the caller from having to guess this function's frame:
               ``frames_along`` is parallel-transported from *up_hint* and is
               not the frame anything is measured in.
    ``polar``  see :func:`sweep_rings` — the ring is spaced by polar angle, so
               its vertices land on the rays ``ratio``'s table was measured on
               instead of between them.

    ``frame_pts`` takes the ring **orientations** from a second polyline, while
    the ring **centres** stay on *points*.  They are two different questions and
    a limb answers them with two different curves: the sections are measured on
    a plane perpendicular to the bone chain, but their centres are not on it —
    a thigh's flesh sits ~12 mm behind the femur, and how far behind changes
    from station to station.  Sweeping along the offset centres makes the
    frames follow the *derivative of the offset*, so a stretch where the centre
    moves fast tilts every ring there: measured at the crotch, the centre moves
    7.2 mm across in 13 mm of height, and the resulting 29-degree tilt smeared
    each ring +-41 mm up and down the leg (§7.222).  Same class as §7.198's
    "measured along one polyline, consumed along another".
    """
    rings = sweep_rings(points, radii, sides, squash=squash, twist=twist,
                        up_hint=up_hint, ratio=ratio, frame_pts=frame_pts,
                        polar=polar, frames=frames)
    p = np.asarray(points, dtype=np.float64)
    cap_start, cap_end = (caps if isinstance(caps, (tuple, list))
                          else (caps, caps))
    mesh = loft(rings, True, bool(cap_start), bool(cap_end), name,
                end_apex=end_apex)
    return ensure_outward(mesh, p.mean(axis=0))


def ribbon_along(points: np.ndarray, widths, name: str = "ribbon",
                 up_hint=(0.0, 0.0, 1.0), thickness: float = 0.0) -> Mesh:
    """A flat strip swept along a polyline — used for hair strands and straps."""
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    w = np.full(n, float(widths)) if np.isscalar(widths) else np.asarray(widths, dtype=np.float64)
    _, nor, bi = frames_along(p, up_hint)
    if thickness <= 0.0:
        left = p + nor * w[:, None] * 0.5
        right = p - nor * w[:, None] * 0.5
        rings = [np.stack([left[i], right[i]]) for i in range(n)]
        return loft(rings, wrap_ring=False, name=name)
    rings = []
    for i in range(n):
        a = nor[i] * w[i] * 0.5
        b = bi[i] * thickness * 0.5
        rings.append(np.stack([p[i] + a + b, p[i] - a + b, p[i] - a - b, p[i] + a - b]))
    return ensure_outward(loft(rings, True, True, True, name), p.mean(axis=0))


# --------------------------------------------------------------------------
# deformers
# --------------------------------------------------------------------------


def smooth_field(mesh: Mesh, field: np.ndarray, iterations: int = 1,
                 factor: float = 0.5) -> np.ndarray:
    """Laplacian smoothing of a per-vertex vector *field* over *mesh*'s adjacency.

    Split out from :func:`smooth_mesh` because **what gets diffused matters**:
    a surface lofted from a measured table is already continuous, and diffusing
    *it* only rounds the table's own extremes off (``CLAUDE.md`` 七, §7.280 on
    ``smooth_weights``).  What a relax is usually wanted for is the ridge a
    region deformer leaves at its falloff boundary — and that ridge lives in the
    **displacement** the deformer added, not in the surface.  Smoothing the
    displacement and adding it back takes the ridge out and leaves the loft
    where the table put it (§7.293 measured 4.2 mm of gluteal cleft on the
    difference).
    """
    if mesh.tri_count == 0 or iterations <= 0:
        return np.asarray(field, dtype=np.float64)
    tri = mesh.indices.reshape(-1, 3)
    e0 = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    e1 = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    src = np.concatenate([e0, e1])
    dst = np.concatenate([e1, e0])
    n = mesh.vertex_count
    v = np.array(field, dtype=np.float64, copy=True)
    counts = np.bincount(src, minlength=n).astype(np.float64)
    counts[counts == 0] = 1.0
    for _ in range(iterations):
        acc = np.zeros_like(v)
        np.add.at(acc, src, v[dst])
        avg = acc / counts[:, None]
        v = v + (avg - v) * factor
    return v
