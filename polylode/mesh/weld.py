"""Turn a pile of intersecting parts into **one** surface.

The body is built as separate closed volumes — a trunk loft, four limb tubes, a
palm with five finger tubes, two feet — and until now they were joined by simply
**intersecting**.  That is the whole of N25 (c), and §7.237 gave it a number:
132 of Origin's 176 deltoid vertices lie inside *neither* our arm nor our trunk,
because the flesh at a junction is exactly what two crossing tubes do not have.

The construction here is the cheapest thing that is actually a manifold:

1. **Trim** — a triangle with any corner inside another part is interior, and
   goes.  Parity, not signed distance: a surface that folds answers the second
   one wrong (§7.168), and this is decided per **closed piece** for the same
   reason §7.228 had to (a hand is six tubes under one label).
2. **Stitch** — trimming leaves two boundary loops per junction, one on each
   part, a ring's width apart.  Connect them with a triangle strip.

**No new vertices.**  The strip is built between vertices that already exist, so
UVs, skin weights, vertex colours and the ``seg`` labels come along unchanged —
which is what makes this a post-process rather than a redesign of every part.
The cost is that the seam is a band as wide as the parts' own ring spacing; at a
limb junction that band *is* the fillet, which is what Origin has there.

What this deliberately does **not** do is a general CSG: no triangle is split,
so the trimmed opening is a little larger than the true intersection curve and
the strip spans it.  An exact boolean would need retriangulation, and every
attribute would then have to be interpolated onto vertices nothing measured.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh

#: Directions the parity test shoots along.  Odd for a majority is "inside".
#: Five, because a single ray through a shared edge or a coincident face is a
#: coin toss and the parts here touch by construction.
_DIRS = np.array([[1.0, 0.2, 0.3], [0.3, 1.0, -0.2], [-0.4, 0.1, 1.0],
                  [0.0, -1.0, 0.4], [-1.0, -0.3, -0.5]])
_DIRS /= np.linalg.norm(_DIRS, axis=1, keepdims=True)


def vertex_ids(pos: np.ndarray, scale: float) -> np.ndarray:
    """Group index per vertex, welding coincident ones.

    **Topology first, attributes never.**  Every capped tube in this body is two
    index-disjoint pieces — ``primitives._fan_cap`` builds the cap with its own
    copy of the rim — so the raw index buffer has **71** components where the
    body has thirteen parts, and a parity test against "the tube" would be a
    test against a surface open at both ends.  Welding by position fixes the
    topology; the triangles keep their original vertices, so the cap's UVs and
    the ring's stay whatever they were (they differ — a fan cap's are zero).
    """
    key = np.round(np.asarray(pos, dtype=np.float64) / (scale * 1e-5))
    _u, inv = np.unique(key.astype(np.int64), axis=0, return_inverse=True)
    return inv


def components(tri: np.ndarray, n_vert: int) -> np.ndarray:
    """Connected-component label per vertex, over the triangles' edges."""
    parent = np.arange(n_vert)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    for a, b, c in tri:
        for u, v in ((a, b), (b, c)):
            ra, rb = find(u), find(v)
            if ra != rb:
                parent[ra] = rb
    return np.array([find(i) for i in range(n_vert)])


def _inside(pts: np.ndarray, pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Parity test of *pts* against the closed surface ``pos``/``tri``."""
    if not len(tri) or not len(pts):
        return np.zeros(len(pts), dtype=bool)
    a = pos[tri[:, 0]]
    e1 = pos[tri[:, 1]] - a
    e2 = pos[tri[:, 2]] - a
    odd = np.zeros(len(pts), dtype=np.int64)
    for d in _DIRS:
        h = np.cross(d, e2)
        det = (e1 * h).sum(axis=1)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        for k, o in enumerate(pts):
            s = o - a
            u = (s * h).sum(axis=1) * inv
            q = np.cross(s, e1)
            v = (d * q).sum(axis=1) * inv
            t = (e2 * q).sum(axis=1) * inv
            odd[k] += int((ok & (u >= 0.0) & (u <= 1.0) & (v >= 0.0)
                           & (u + v <= 1.0) & (t > 1e-7)).sum()) % 2
    return odd >= 3


def boundary_loops(tri: np.ndarray) -> list[list[int]]:
    """Ordered vertex loops along edges used by exactly one triangle."""
    seen: dict[tuple[int, int], int] = {}
    for a, b, c in tri:
        for u, v in ((a, b), (b, c), (c, a)):
            k = (min(u, v), max(u, v))
            seen[k] = seen.get(k, 0) + 1
    nbr: dict[int, list[int]] = {}
    for (u, v), n in seen.items():
        if n != 1:
            continue
        nbr.setdefault(u, []).append(v)
        nbr.setdefault(v, []).append(u)
    loops, used = [], set()
    for start in nbr:
        if start in used:
            continue
        loop, cur, prev = [start], start, -1
        used.add(start)
        while True:
            nxt = [w for w in nbr.get(cur, ()) if w != prev and w not in used]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            used.add(cur)
            loop.append(cur)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _orient(tri: np.ndarray) -> set:
    """Every directed edge the surviving triangles use."""
    out = set()
    for a, b, c in tri:
        out.add((int(a), int(b)))
        out.add((int(b), int(c)))
        out.add((int(c), int(a)))
    return out


def _agree(tris, used: set) -> int:
    """How many of *tris*' directed edges **clash** with the surface's.

    A boundary edge is used once, in one direction; a triangle that closes the
    hole has to use it the **other** way.  Counting the clashes is what tells a
    strip it is inside out — and that is the whole of §7.239's holes: the
    geometry was there and back-facing, so it was culled and read as a hole
    (a hole and a flipped face look the same from outside, and only one of them
    is missing).
    """
    n = 0
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            if (u, v) in used:
                n += 1
    return n


def band_frame(loop_a, loop_b):
    """``(origin, axis, e1, e2)`` — the cylinder a limb junction lives in.

    A junction's two openings both encircle the limb's axis, so the band
    between them is an annulus about it and the natural coordinates are that
    cylinder's: how far inboard, which way round, how far out.  The axis is
    loop A's own best-fit plane normal pointing at loop B, and ``e1`` is world
    up projected onto the ring — no bone is read, so the tool that measures the
    band and the generator that builds it get the same frame from the same two
    loops (``CLAUDE.md`` 七: one owner for a quantity).

    Mirrored sides come out mirrored: the axis points inboard on both, ``e1``
    is up on both, and ``e2 = axis x e1`` flips with the axis — which is what
    makes the left band's ray *k* the right band's ray *-k*, and that pairing
    is this measurement's noise floor.
    """
    a = np.asarray(loop_a, dtype=np.float64)
    c = a.mean(axis=0)
    _u, _s, vt = np.linalg.svd(a - c, full_matrices=False)
    n = vt[2] / max(float(np.linalg.norm(vt[2])), 1e-12)
    if float(n @ (np.asarray(loop_b, dtype=np.float64).mean(axis=0) - c)) < 0.0:
        n = -n
    up = np.array([0.0, 1.0, 0.0])
    e1 = up - n * float(up @ n)
    if float(np.linalg.norm(e1)) < 1e-9:
        e1 = np.array([1.0, 0.0, 0.0]) - n * float(n[0])
    e1 = e1 / float(np.linalg.norm(e1))
    return c, n, e1, np.cross(n, e1)


def cylindrical(pts, frame):
    """``(d, theta, r)`` of *pts* in a :func:`band_frame`."""
    c, n, e1, e2 = frame
    q = np.asarray(pts, dtype=np.float64) - c
    x, z = q @ e1, q @ e2
    return q @ n, np.arctan2(z, x) % (2.0 * np.pi), np.hypot(x, z)


def ray_crossing(loop, frame, theta):
    """Where a closed *loop* crosses each half-plane — ``(d, r, edge, frac)``.

    Walking the loop's own polyline rather than taking the nearest vertex: the
    two loops of a junction have different vertex counts (24 against 35 at the
    shoulder), so "the same ray on both" only exists if one of them is
    interpolated.  ``edge``/``frac`` say which of the loop's edges was crossed
    and where along it, which is what lets an attribute (a UV, a weight) be
    interpolated to the same place the position was.
    """
    d, a, r = cylindrical(loop, frame)
    m = len(d)
    th = np.asarray(theta, dtype=np.float64)
    out_d = np.full(len(th), np.nan)
    out_r = np.full(len(th), np.nan)
    out_e = np.zeros(len(th), dtype=np.int64)
    out_f = np.zeros(len(th))
    for k, t in enumerate(th):
        for i in range(m):
            j = (i + 1) % m
            step = (a[j] - a[i] + np.pi) % (2.0 * np.pi) - np.pi
            if abs(step) < 1e-12:
                continue
            f = (((t - a[i] + np.pi) % (2.0 * np.pi)) - np.pi) / step
            if not (0.0 <= f <= 1.0):
                continue
            out_d[k] = d[i] + f * (d[j] - d[i])
            out_r[k] = r[i] + f * (r[j] - r[i])
            out_e[k], out_f[k] = i, f
            break
    return out_d, out_r, out_e, out_f


def single_valued(loop, frame) -> np.ndarray:
    """Per vertex: is its own ray the **only** place it meets the loop?

    The condition a band table has to have before it can place this loop at
    all.  A table is ``r(theta, s)``, so two vertices that share a ray share a
    radius — measured on the shoulder, three consecutive rim vertices sit at
    theta 159.3 / 160.5 / 160.8 with radii 51.2 / 41.1 / 38.3 mm (the trim's
    staircase running *along* a ray instead of across it), and placing them by
    the table puts all three at 48.2 and leaves 0.73 mm edges: a sliver, which
    ``--creases`` reads as a near-180 degree dihedral (§7.258 四之一).

    Counted rather than compared: asking whether ``ray_crossing`` hands back
    the vertex itself answers "did the walk find another crossing *first*",
    which depends on where the loop's index origin is.  The number of crossings
    does not, and it is the same admission :data:`body.SHOULDER_BAND` uses for
    its own rays — exactly one crossing or the axial coordinate is the wrong
    parameter for that direction (§7.208's family).  Half-open in the edge's
    fraction so a ray through a vertex is one crossing, not two.
    """
    d, a, _r = cylindrical(loop, frame)
    m = len(d)
    out = np.ones(m, dtype=bool)
    for k in range(m):
        n = 0
        for i in range(m):
            j = (i + 1) % m
            step = (a[j] - a[i] + np.pi) % (2.0 * np.pi) - np.pi
            if abs(step) < 1e-12:
                continue
            f = (((a[k] - a[i] + np.pi) % (2.0 * np.pi)) - np.pi) / step
            if 0.0 <= f < 1.0:
                n += 1
        out[k] = n == 1
    return out


def _stitch(loop_a: list[int], loop_b: list[int], pos: np.ndarray) -> list:
    """Triangles joining two loops, walking both by nearest advance.

    Both loops are oriented the same way first (the strip crosses itself
    otherwise), and the walk starts from the closest pair — a fixed start makes
    the strip spiral by however far the two loops' index origins differ, which
    on a limb ring is up to half a turn.
    """
    A = pos[loop_a]
    B = pos[loop_b]
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    # Same winding: compare the two loops' area vectors about their centroids.
    na = np.cross(A - ca, np.roll(A, -1, axis=0) - ca).sum(axis=0)
    nb = np.cross(B - cb, np.roll(B, -1, axis=0) - cb).sum(axis=0)
    if float(na @ nb) < 0.0:
        loop_b = loop_b[::-1]
        B = pos[loop_b]
    j0 = int(np.argmin(((B - A[0]) ** 2).sum(axis=1)))
    na_, nb_ = len(loop_a), len(loop_b)
    out, i, j = [], 0, 0
    while i < na_ or j < nb_:
        ai, aj = loop_a[i % na_], loop_b[(j + j0) % nb_]
        an, bn = loop_a[(i + 1) % na_], loop_b[(j + 1 + j0) % nb_]
        # **Neither side may fall a whole step behind in proportion**, and that
        # is a correctness bound rather than a preference: the shorter-diagonal
        # test is local, and on a very unequal pair it can leave one side parked
        # while the other walks its **whole** circuit — the fan from that one
        # vertex then reaches the same opposite vertex at both ends of the turn,
        # so its spoke is used twice by the fan and twice more by the take_a
        # steps either side of it.  Four faces on one edge, and nothing else
        # reports it because the strip still closes and still renders (§7.301
        # found it on the hand's little fork: a 2-vertex arc plus two crotch
        # points against an 8-vertex finger ring, one edge used 4 times).
        # ``(i + 1) * nb_ <= j * na_`` is "a is behind by one of its own steps".
        take_a = (j >= nb_) or (i < na_ and
                                ((i + 1) * nb_ <= j * na_
                                 or np.linalg.norm(pos[an] - pos[aj])
                                 <= np.linalg.norm(pos[bn] - pos[ai])))
        if take_a:
            out.append((ai, an, aj))
            i += 1
        else:
            # The quad's **other** triangle, so consecutive strip faces share
            # their edge in opposite directions.  ``(ai, bn, aj)`` is the fan
            # form and it repeats ``bj -> ai`` in both, which is two faces
            # meeting the same way round — non-manifold by construction.
            out.append((aj, ai, bn))
            j += 1
    return out


def _blend_skin(idx_a, wt_a, idx_b, wt_b, s: float):
    """One vertex's influences from two, mixed ``(1-s) : s``.

    Summed per bone and then cut to the buffer's width, because the two ends of
    a junction are bound to different chains and a straight per-slot lerp would
    mix an arm bone's index with a trunk bone's and skin the new vertex to a
    third bone entirely.
    """
    acc: dict[int, float] = {}
    for idx, wt, f in ((idx_a, wt_a, 1.0 - s), (idx_b, wt_b, s)):
        for b, w in zip(np.asarray(idx).ravel(), np.asarray(wt).ravel()):
            if w > 0.0:
                acc[int(b)] = acc.get(int(b), 0.0) + float(w) * f
    top = sorted(acc.items(), key=lambda kv: -kv[1])[:len(np.ravel(idx_a))]
    tot = sum(w for _b, w in top) or 1.0
    out_i = np.zeros(len(np.ravel(idx_a)), dtype=np.int32)
    out_w = np.zeros(len(np.ravel(idx_a)), dtype=np.float32)
    for k, (b, w) in enumerate(top):
        out_i[k], out_w[k] = b, w / tot
    return out_i, out_w


def weld(mesh: Mesh, keep_open: set[tuple] | None = None,
         skip: np.ndarray | None = None,
         max_span: float = 2.0, collapse: float = 0.0,
         labels=None, band=None, record: dict | None = None) -> dict:
    """Trim interpenetrating parts and stitch the openings, **in place**.

    ``skip`` marks vertices the trim may not touch.  Two kinds so far, and the
    second is the interesting one:

    * the **nails** — an applied feature, not skin (Origin has none at all; its
      nails are a texture channel, §7.144), and ellipsoids whose poles go
      non-manifold when welded by position (90 of the first pass's 94 bad
      edges);
    * the trunk's **closure rings** — §7.224 built the bottom of the loft to
      finish *inside* the thighs on purpose ("each lobe of the figure-eight is
      bounded by the tangent chords, i.e. inside the tube").  That is not a
      junction waiting to be welded, it is a closure that is **meant** to be
      interior, and trimming it is what left the ragged 14-vertex hole §7.241
      found on one hip.  A piece that is deliberately inside another must be
      told so; the parity test cannot know the difference.


    ``keep_open`` is the set of boundary loops the body is meant to have — the
    neck's rim and the trunk's base — identified by their rounded centroid, so
    the welder does not try to sew the body's own openings shut.

    ``band`` is what lets a junction have a **shape**.  Called with the two
    loops' positions and their ``labels``, it either declines (``None`` — the
    plain one-row strip) or hands back interior rings; the strip is one row of
    quads and a row of quads is *straight by construction*, which is the whole
    of §7.247: measured against Origin's own ``o_body_a``, the shoulder's 43 mm of
    band falls on a line to within 0.04 mm per step while Origin is a saddle with
    4.5 mm of relief.  Interior rings are the only way to carry that, and they
    are the one thing the "no new vertices" contract above forbids — so the new
    vertices' UV, colour and skin weights are interpolated from the two ends
    (:func:`_blend_skin`), which is the least this can cost.

    A band may also hand back ``tail_pos``, which **moves the far loop** — the
    trimmed part's own rim — onto the surface its table describes.  That is not
    a new vertex and not a shape the strip can carry: it is the one end of the
    junction the table could not place, and leaving it where the trim put it is
    what made the last row a step rather than a taper (§7.257).

    ``collapse`` is the other side of that: a table cannot carry a step of the
    loop it is not a function over, so placing one leaves duplicates, and step 3
    below merges them.  **Trimming those vertices away instead was measured and
    it is not the answer** — retreating until every ray meets the loop exactly
    once (:func:`single_valued`) converges in one pass and costs only 4/7
    triangles a side, but the row it exposes is denser still: the shortest rim
    edge comes back 0.73 → 0.68 mm and ``--creases`` trades y=0.795 (32 → 22
    edges) for a new ``<<`` at y=0.800 (10 → 18) and one more ``seg deltoid``
    fold, which is the signature §7.258 四之一 rejected (§7.259 二).

    ``tail_rows`` and ``tail_field`` carry that on **behind** the rim: the rows
    of the trimmed part's own surface next to it, placed by the same table.
    Moving the rim alone put its whole displacement into one row of faces, and
    a displacement that dies in one row is a fold (§7.258).  Rows are counted
    along the surface, and that is the point of doing it here rather than in
    the band: the band's axis is nearly horizontal, so *any* metric window on
    it is a plane that grazes the flank and picks up the hip.  The welder is
    the one that knows what is next to what.

    Returns a report: pieces, trimmed triangles, loops stitched, whether the
    result is closed apart from *keep_open*, and ``added`` — the label of each
    vertex appended, in order, so the caller's per-vertex ``seg`` can grow with
    the mesh.  ``record`` collects every junction that was sewn (both loops'
    positions and labels), which is what a ruler needs to measure the band
    without a second copy of the trim living in the tool.
    """
    pos = mesh.positions.astype(np.float64)
    tri = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
    scale = float(np.ptp(pos[:, 1])) or 1.0
    vid = vertex_ids(pos, scale)
    vpos = np.zeros((int(vid.max()) + 1, 3))
    vpos[vid] = pos
    vtri = vid[tri]
    label = components(vtri, len(vpos))
    parts = np.unique(label)
    rep = {}
    for p in parts:
        keep_t = (label[vtri] == p).all(axis=1)
        rep[p] = vtri[keep_t]

    frozen = (np.zeros(len(vpos), dtype=bool) if skip is None
              else np.bincount(vid, weights=np.asarray(skip, float),
                               minlength=len(vpos)) > 0)
    pre_loops = boundary_loops(vtri)

    # --- 1. trim -----------------------------------------------------------
    # Only vertices inside another part's bounding box can be inside it, and at
    # a junction that is a few hundred out of eleven thousand — without the
    # filter this is 11k x 20k ray casts.
    inside = np.zeros(len(vpos), dtype=bool)
    for p in parts:
        t = rep[p]
        if len(t) < 4 or frozen[t[0, 0]]:
            continue
        lo, hi = vpos[t.reshape(-1)].min(axis=0), vpos[t.reshape(-1)].max(axis=0)
        cand = np.flatnonzero((label != p) & ~frozen
                              & (vpos >= lo - 1e-9).all(axis=1)
                              & (vpos <= hi + 1e-9).all(axis=1)
                              & ~inside)
        if not len(cand):
            continue
        inside[cand] = _inside(vpos[cand], vpos, t)

    keep_tri = ~inside[vtri].any(axis=1)
    trimmed = int((~keep_tri).sum())
    if record is not None:
        # The parts as they were, before anything was cut away.  A band's error
        # has two possible owners — the part's own shape or the trim that ate
        # into it — and they are told apart by asking the untrimmed part.
        #
        # **Positions as well as indices.**  A band that moves the trimmed
        # part's own vertices (``tail_pos``, ``tail_rows``) moves them in
        # ``mesh.positions``, so pre-trim triangles read on the finished mesh
        # are the moved surface — the one thing this record exists not to be.
        # §7.257 already had that hole: its ``raw`` column was described as
        # "before the trim and before the move" while the rim it moved is a
        # trunk vertex like any other.
        record["pre_indices"] = tri.copy()
        record["pre_positions"] = mesh.positions.copy()
    tri, vtri = tri[keep_tri], vtri[keep_tri]

    # --- 2. stitch ---------------------------------------------------------
    # A welded vertex may have several original indices (a ring and the cap that
    # sat on it).  Stitch with whichever of them the **surviving** triangles
    # still use most — when a cap is trimmed away its copy has no triangles
    # left, so the ring's vertex wins on its own and the strip inherits real
    # UVs instead of a fan cap's zeros.
    votes = np.bincount(tri.reshape(-1), minlength=len(pos))
    pick = np.zeros(len(vpos), dtype=np.int64)
    order = np.argsort(votes, kind="stable")
    pick[vid[order]] = order

    def tag(loop):
        return tuple(np.round(vpos[loop].mean(axis=0), 3))

    # **The openings the body already had are not junctions.**  Detected rather
    # than declared: whatever was a boundary loop *before* trimming is one the
    # parts were built with — the neck's rim and the trunk's base — and sewing
    # those two to each other would close the body into a bag.
    #
    # Matched by **membership, not by centroid**.  Trimming eats into these
    # loops, so their centroid moves: measured, the trunk's base went from 42
    # vertices at (0.002, 0.845) to a 14-vertex remnant at (+0.123, 0.952), the
    # tag stopped matching, and the welder sewed the body's own opening shut and
    # left the remnant as a hole at the hip instead — a hole nobody would look
    # for, because it is between the leg and the trunk.
    pre_vertices = {v for lp in pre_loops for v in lp}
    declared = set(keep_open or ())

    dir_used = _orient(vtri)
    capped = []
    def was_open(loop):
        return sum(v in pre_vertices for v in loop) > 0.5 * len(loop)

    free = [lp for lp in boundary_loops(vtri)
            if tag(lp) not in declared and not was_open(lp)
            and not frozen[lp].any()]
    # **Pairs taken in order of distance, not in order of index.**  Greedy by
    # index lets a big rim reach past its own junction and take a small rim's
    # partner: measured, the trunk's base (r 97 mm, so a 194 mm reach) is rim 0
    # and it grabbed the left hip's partner at 47 mm, leaving that hip's own
    # 16 mm pair unmade — which is §7.241's hole, and why it was **asymmetric**
    # on a symmetric body (whichever side the scan reached second lost).
    #
    # Mutual-nearest was tried for the same asymmetry and is worse (openings
    # 10 → 15): a junction's two rims are often not each other's nearest.
    # Sorting the candidates fixes the order without demanding that.
    #
    # **And the score itself has to be symmetric, which it was not** (§7.282).
    # It was "mean over A of the nearest distance to B" with A the loop of lower
    # index — so which rim the boundary walk happened to reach first decided the
    # number, by 4.4x on the palm's own junction (the 26-vertex hub reads 23.07
    # mm to a finger ring and the finger reads 5.22 mm back, because most of a
    # 21 mm hub is far from any one finger).  Letting the hand in, 29 pairs come
    # out differently depending on the order.  Over the **union** of both rims'
    # nearest distances there is no side to pick: it is the same statistic with
    # the conditioning removed, and it is what "how far apart are these two
    # rims" means.  Measured on the shipping build it changes nothing — all four
    # junctions read the same under either rule — so this is the fault removed,
    # not a new opinion about which junctions exist.
    span_of = {}
    for i, lp in enumerate(free):
        A = vpos[lp]
        span_of[i] = max_span * float(np.linalg.norm(A - A.mean(axis=0),
                                                     axis=1).mean())
    cand = []
    for i in range(len(free)):
        A = vpos[free[i]]
        for j in range(i + 1, len(free)):
            B = vpos[free[j]]
            dd = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
            d = float(np.concatenate([dd.min(axis=1), dd.min(axis=0)]).mean())
            if d <= min(span_of[i], span_of[j]):
                cand.append((d, i, j))
    cand.sort()

    lab_v = None
    if labels is not None:
        lab_v = np.empty(len(vpos), dtype=object)
        lab_v[vid] = np.asarray(labels, dtype=object)
    if record is not None:
        # Every opening the trim left, before any pairing — what §7.243 needed
        # to see and had to rebuild the trim in a throwaway script to get.
        record["free"] = [{"pos": vpos[lp].copy(),
                           "label": None if lab_v is None else list(lab_v[lp])}
                          for lp in free]

    # New vertices live past the end of the welded arrays; ``ext_*`` is the
    # extension of ``vpos``/``pick`` so :func:`_stitch` and :func:`_agree` keep
    # working on one index space.
    ext_pos: list = []
    ext_orig: list = []
    new_lab: list = []
    grow: list = []                     # (position, uv, colour, idx, wt)

    def _spawn(p, a_orig, b0, b1, f, s, lab):
        """A new vertex on the band: position given, everything else blended."""
        wa, wb = 1.0 - f, f
        uv = (mesh.uvs[a_orig] * (1.0 - s)
              + (mesh.uvs[b0] * wa + mesh.uvs[b1] * wb) * s)
        col = (mesh.colors[a_orig] * (1.0 - s)
               + (mesh.colors[b0] * wa + mesh.colors[b1] * wb) * s)
        bi = np.concatenate([mesh.bone_idx[b0], mesh.bone_idx[b1]])
        bw = np.concatenate([mesh.bone_wt[b0] * wa, mesh.bone_wt[b1] * wb])
        idx, wt = _blend_skin(mesh.bone_idx[a_orig], mesh.bone_wt[a_orig],
                              bi, bw, s)
        orig = len(pos) + len(grow)
        grow.append((p, uv, col, idx, wt))
        ext_pos.append(np.asarray(p, dtype=np.float64))
        ext_orig.append(orig)
        new_lab.append(lab)
        return len(vpos) + len(ext_pos) - 1

    def _pick(v):
        return int(pick[v]) if v < len(vpos) else int(ext_orig[v - len(vpos)])

    def _place(ids, new):
        """Move welded vertices the band placed, in both position arrays.

        ``vpos`` because the stitch and the record read it, and **every**
        original vertex welded onto that id because that is what ships — a
        welded id can stand for several of them (a ring and the cap that sat on
        it), and moving only the one the stitch picked leaves the others behind
        at the old place.
        """
        ids = np.asarray(ids, dtype=np.int64)
        vpos[ids] = new
        sel = np.isin(vid, ids)
        # a new array, not an in-place write: ``Mesh.bounds`` keys its cache
        # on the array's identity (§7.420)
        pos = mesh.positions.copy()
        pos[sel] = vpos[vid[sel]].astype(pos.dtype)
        mesh.positions = pos

    # Who is next to whom on the **trimmed** surface — a band that places rows
    # of the part behind its rim counts them here, where the topology is.
    nbr: dict = {}
    for _a, _b, _c in vtri:
        for _u, _v in ((_a, _b), (_b, _c), (_c, _a)):
            nbr.setdefault(int(_u), set()).add(int(_v))
            nbr.setdefault(int(_v), set()).add(int(_u))

    def _emit(tris, out):
        """Faces, wound so their shared edges oppose the surface's."""
        if _agree(tris, dir_used) > _agree([(c, b, a) for a, b, c in tris],
                                           dir_used):
            tris = [(c, b, a) for a, b, c in tris]
        for a, b, c in tris:
            dir_used.update({(a, b), (b, c), (c, a)})
            out.append((_pick(a), _pick(b), _pick(c)))

    used, added, placed = set(), [], []
    for _d, i, j in cand:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        la = lab_v[free[i]] if lab_v is not None else None
        lb = lab_v[free[j]] if lab_v is not None else None
        spec = None if band is None else band(vpos[free[i]], vpos[free[j]],
                                              la, lb)
        head, tail, hl, tl = free[i], free[j], la, lb
        if spec is not None and spec.get("head") == "b":
            # Which loop the rings are sampled on is the band's call, not the
            # pairing order's: at the shoulder loop A is the trunk's *trimmed*
            # hole (35 irregular vertices) and loop B is the arm's own ring
            # (24 even ones), and a band wants the even one.
            head, tail, hl, tl = free[j], free[i], lb, la
        moved = False
        if spec is not None and spec.get("tail_pos") is not None:
            # A band may place its **far** loop as well, and then that loop is
            # not a new vertex — it is the trimmed part's own rim, which the
            # band moves to where its table says the surface is.  Both position
            # arrays have to follow: ``vpos`` because the stitch and the record
            # read it, and every original vertex welded onto that id because
            # that is what ships.
            tp = np.asarray(spec["tail_pos"], dtype=np.float64)
            if tp.shape == (len(tail), 3):
                # Kept **before** the move, because the only edges step 3 may
                # merge are the ones the move itself shortened.
                placed.append((list(tail), vpos[tail].copy()))
                _place(np.asarray(tail), tp)
                moved = True
        rows = 0 if spec is None else int(spec.get("tail_rows") or 0)
        field = None if spec is None else spec.get("tail_field")
        if moved and rows and field is not None:
            # The rows behind the rim, counted **along the surface**: breadth
            # first from the rim over the trimmed part's own triangles.  A ring
            # of vertices, not a distance — see the docstring.
            seen, front, grown = set(tail), set(tail), []
            for _ in range(rows):
                front = {int(w) for v in front for w in nbr.get(int(v), ())
                         if w not in seen and label[w] == label[tail[0]]}
                if not front:
                    break
                seen |= front
                grown.extend(sorted(front))
            if grown:
                idx = np.asarray(grown, dtype=np.int64)
                np_ = np.asarray(field(vpos[idx]), dtype=np.float64)
                take = np.isfinite(np_).all(axis=1)
                if take.any():
                    _place(idx[take], np_[take])
                if record is not None:
                    record.setdefault("tail_rows", []).append(
                        {"asked": len(idx), "moved": int(take.sum())})
        if record is not None:
            # **After** the move, not before: a ruler that reads this record and
            # then measures the mesh would otherwise be comparing the rim to a
            # place it no longer is.
            record.setdefault("junctions", []).append(
                {"a": vpos[free[i]].copy(), "b": vpos[free[j]].copy(),
                 "label_a": None if la is None else list(la),
                 "label_b": None if lb is None else list(lb),
                 "tail_moved": moved})
        if spec is not None:
            # Rings first, then the last row falls back to the pairwise walk:
            # the interior rings carry the head loop's own ray count, so every
            # row but the last is a 1:1 quad strip and only the step onto the
            # other loop's different vertex count needs nearest-advance.
            P, ss = np.asarray(spec["pos"], dtype=np.float64), spec["s"]
            edge, frac = np.asarray(spec["edge"]), np.asarray(spec["frac"])
            m = len(head)
            base = head
            for k in range(len(ss)):
                ring = []
                for q in range(m):
                    e = int(edge[q])
                    b0 = _pick(tail[e])
                    b1 = _pick(tail[(e + 1) % len(tail)])
                    lab = spec.get("label")
                    if lab is None:
                        lab = (hl[q] if hl is not None and ss[k] < 0.5
                               else (tl[e] if tl is not None else None))
                    ring.append(_spawn(P[k, q], _pick(base[q]), b0, b1,
                                       float(frac[q]), float(ss[k]), lab))
                _emit([t for q in range(m)
                       for t in ((head[q], head[(q + 1) % m], ring[q]),
                                 (head[(q + 1) % m], ring[(q + 1) % m],
                                  ring[q]))], added)
                head = ring
        vpos_all = (vpos if not ext_pos
                    else np.vstack([vpos, np.asarray(ext_pos)]))
        _emit(_stitch(list(head), tail, vpos_all), added)
    capped.extend(free[i] for i in range(len(free)) if i not in used)

    for lp in capped:
        _emit([(lp[0], lp[k], lp[k + 1]) for k in range(1, len(lp) - 1)], added)

    if ext_pos:
        vpos = np.vstack([vpos, np.asarray(ext_pos)])
    if grow:
        mesh.positions = np.vstack(
            [mesh.positions, np.asarray([g[0] for g in grow],
                                        dtype=np.float32)])
        mesh.normals = np.vstack([mesh.normals,
                                  np.zeros((len(grow), 3), dtype=np.float32)])
        mesh.uvs = np.vstack([mesh.uvs, np.asarray([g[1] for g in grow],
                                                   dtype=np.float32)])
        mesh.colors = np.vstack([mesh.colors,
                                 np.asarray([g[2] for g in grow],
                                            dtype=np.float32)])
        mesh.bone_idx = np.vstack([mesh.bone_idx,
                                   np.asarray([g[3] for g in grow],
                                              dtype=np.int32)])
        mesh.bone_wt = np.vstack([mesh.bone_wt,
                                  np.asarray([g[4] for g in grow],
                                             dtype=np.float32)])
        vid = np.concatenate([vid, np.arange(len(vpos),
                                             len(vpos) + len(grow))])
    if added:
        tri = np.vstack([tri, np.asarray(added, dtype=np.int64)])

    # --- 3. the duplicates a placed rim is left with ------------------------
    # A table is ``r(theta, s)``, so it cannot carry a step of the loop that is
    # **radial**: the trim's staircase alternates between running around the
    # part's rings and running along them, and the second kind is two samples of
    # one point of the true intersection curve — an artifact of not splitting
    # triangles, which the docstring at the top of this file says is the deal.
    # Placing both by the table puts them within a millimetre of each other, and
    # a pair of faces on a sub-millimetre edge is what ``--creases`` reads as a
    # near-180 degree dihedral (§7.258 四之一 measured 177.7).
    #
    # So the surface does not move here: one of the two goes.  Two conditions,
    # and the first is what keeps this from being a mesh decimator:
    #
    # * the **placement** shortened that edge.  A trimmed loop has short edges
    #   of its own — the staircase's corners — and they are the part's own
    #   tessellation, nothing here has an opinion about them;
    # * it is now under a fraction of **that loop's own median edge**, for
    #   §7.240's reason: a length in millimetres is a different claim on a 45 mm
    #   ring than on an 8 mm one.
    #
    # Scoped to the loops a band placed either way, because "collapse short
    # edges" over the whole body would eat the fingers, whose real features are
    # that size (``seg hand`` turns inside 0.5 mm).
    if placed and collapse > 0.0:
        remap: dict = {}
        gone = 0
        for lp, was in placed:
            P = vpos[lp]
            e = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
            before = np.linalg.norm(was - np.roll(was, -1, axis=0), axis=1)
            lim = collapse * float(np.median(e))
            eaten: set = set()
            for q in np.argsort(e):
                if e[q] >= lim:
                    break
                if e[q] >= before[q]:
                    continue          # the trim's own corner, not the move's
                a, b = int(lp[int(q)]), int(lp[(int(q) + 1) % len(lp)])
                # One merge per vertex, so a run of three collapses to two
                # rather than to a point: the shortest edge goes first and its
                # neighbours are then spoken for.
                if a in eaten or b in eaten:
                    continue
                eaten.add(a)
                eaten.add(b)
                gone += 1
                # **Every** original index welded onto the vertex that goes, not
                # just the one the stitch picked — a welded id can stand for
                # several of them, which is the same trap :func:`_place` names.
                for o in np.flatnonzero(vid == b):
                    remap[int(o)] = int(pick[a])
        if remap:
            flat = tri.reshape(-1).copy()
            for b, a in remap.items():
                flat[flat == b] = a
            tri = flat.reshape(-1, 3)
            keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) \
                & (tri[:, 2] != tri[:, 0])
            if record is not None:
                record["collapsed"] = {"vertices": gone,
                                       "triangles": int((~keep).sum())}
            tri = tri[keep]
    mesh.indices = tri.reshape(-1).astype(np.uint32)
    mesh.compute_normals()

    left = [lp for lp in boundary_loops(vid[tri])
            if tag(lp) not in declared and not was_open(lp)]
    return {"pieces": int(len(parts)), "trimmed": trimmed,
            "stitched": len(used) // 2, "capped": len(capped),
            "open_left": len(left), "inside": int(inside.sum()),
            "added": new_lab}
