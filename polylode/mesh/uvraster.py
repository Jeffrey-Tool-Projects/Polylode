"""Rasterise per-vertex attributes into a mesh's UV0 texel grid.

Some shaders read a *second* coordinate set: the surface is laid out on UV0 for
its albedo, but a decal on top of it is placed with UV1/UV2/UV3, or gated by a
vertex colour.  Compositing that decal into the albedo map — which is what a
CPU bake does — needs the value of those attributes **at each texel of the UV0
layout**, and the only thing that relates the two is the mesh itself.

So: draw the mesh once into UV0 space, interpolating whatever per-vertex
arrays the caller hands over.  The result is a texel-aligned lookup the bake
can read with the same indexing it uses for every other map.

Nothing here knows what the attributes mean; the Origin-specific part (which slot
takes which set, and what the shader does with it) lives in
``figure/body_head.py``.

The texel convention is :func:`figure.create_tex._grid`'s, and it has to be:
row 0 is v = 1.  A vertex at (u, v) therefore lands at pixel (u * w,
(1 - v) * h), and the centre of texel (x, y) is at (x + 0.5, y + 0.5).
"""
from __future__ import annotations

import numpy as np

# Texels a UV island's edge gets grown by after rasterising.  A texel whose
# *centre* falls outside every triangle still gets read on the GPU: bilinear
# filtering at an island's border mixes it into the texels that are on the
# surface.  Leaving those undefined puts a hairline of un-composited albedo
# along every seam, which is the classic lightmap-bake artefact and is why
# every baker pads.  Three is enough for the 2048² body map: the widest gap a
# bilinear tap can reach across is one texel, and mip 0 is all the game reads
# for these.
DILATE = 3

# How many triangles are turned into candidate texels at once.  The candidate
# list is the sum of the triangles' bounding boxes, so doing all 14k of the
# body's at once allocates a few hundred MB of temporaries for no gain.
_CHUNK = 2048


def rasterise(uv0, indices, attrs: dict, shape, *, dilate: int = DILATE):
    """``{name: (h, w, k)}`` of *attrs* interpolated over the UV0 layout.

    *attrs* maps a name to a ``(vertices, k)`` array.  Returns those maps plus
    a ``(h, w)`` bool coverage mask, which is what tells the caller whether a
    texel's value means anything (before dilation grew it).

    Overlapping UV islands resolve to whichever triangle is written last.
    Illusion's character meshes do not overlap — the layouts are authored for
    exactly this kind of per-texel work — and a real overlap has no single
    right answer anyway.
    """
    h, w = int(shape[0]), int(shape[1])
    uv0 = np.asarray(uv0, dtype=np.float32).reshape(-1, 2)
    tri = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    out = {k: np.zeros((h, w, np.asarray(v).shape[1]), dtype=np.float32)
           for k, v in attrs.items()}
    covered = np.zeros((h, w), dtype=bool)
    if len(tri) == 0 or len(uv0) == 0 or h < 1 or w < 1:
        return out, covered

    px = np.stack([uv0[:, 0] * w, (1.0 - uv0[:, 1]) * h], axis=1).astype(np.float64)
    a, b, c = px[tri[:, 0]], px[tri[:, 1]], px[tri[:, 2]]
    # Twice the signed area.  A degenerate triangle has no interior, and its
    # reciprocal is what would turn the barycentrics into infinities.
    area = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    live = np.abs(area) > 1e-12
    if not live.any():
        return out, covered

    lo = np.floor(np.minimum(np.minimum(a, b), c) - 0.5).astype(np.int64)
    hi = np.ceil(np.maximum(np.maximum(a, b), c) + 0.5).astype(np.int64)
    x0 = np.clip(lo[:, 0], 0, w)
    y0 = np.clip(lo[:, 1], 0, h)
    x1 = np.clip(hi[:, 0], 0, w)
    y1 = np.clip(hi[:, 1], 0, h)
    bw = np.maximum(x1 - x0, 0)
    bh = np.maximum(y1 - y0, 0)
    live &= (bw > 0) & (bh > 0)
    keep = np.nonzero(live)[0]

    values = {k: np.asarray(v, dtype=np.float32).reshape(len(uv0), -1)
              for k, v in attrs.items()}
    for start in range(0, len(keep), _CHUNK):
        sel = keep[start:start + _CHUNK]
        _draw(sel, tri, a, b, c, area, x0, y0, bw, bh, values, out, covered, w)

    if dilate > 0:
        _dilate(out, covered, dilate)
    return out, covered


def _draw(sel, tri, a, b, c, area, x0, y0, bw, bh, values, out, covered, w):
    n = bw[sel] * bh[sel]
    total = int(n.sum())
    if total == 0:
        return
    who = np.repeat(np.arange(len(sel), dtype=np.int64), n)
    base = np.concatenate([[0], np.cumsum(n)[:-1]])
    k = np.arange(total, dtype=np.int64) - np.repeat(base, n)
    wid = bw[sel][who]
    xs = x0[sel][who] + k % wid
    ys = y0[sel][who] + k // wid

    t = sel[who]
    cx = xs + 0.5
    cy = ys + 0.5
    inv = 1.0 / area[t]
    # Barycentric weights of the texel centre, straight from the edge
    # functions; ``>= 0`` on all three is inside.
    wa = ((b[t, 0] - cx) * (c[t, 1] - cy) - (b[t, 1] - cy) * (c[t, 0] - cx)) * inv
    wb = ((c[t, 0] - cx) * (a[t, 1] - cy) - (c[t, 1] - cy) * (a[t, 0] - cx)) * inv
    wc = 1.0 - wa - wb
    inside = (wa >= 0.0) & (wb >= 0.0) & (wc >= 0.0)
    if not inside.any():
        return
    t, xs, ys = t[inside], xs[inside], ys[inside]
    wa, wb, wc = (wa[inside].astype(np.float32), wb[inside].astype(np.float32),
                  wc[inside].astype(np.float32))
    i0, i1, i2 = tri[t, 0], tri[t, 1], tri[t, 2]
    flat = ys * w + xs
    for name, val in values.items():
        blend = (val[i0] * wa[:, None] + val[i1] * wb[:, None]
                 + val[i2] * wc[:, None])
        dst = out[name].reshape(-1, out[name].shape[2])
        dst[flat] = blend
    covered.reshape(-1)[flat] = True


# (destination, source) slice pairs for the four 4-neighbours.  Slices rather
# than ``np.roll`` because roll copies the whole map per direction per step —
# 33 MB a piece on the 2048² body, forty-eight times over.
_NEIGHBOURS = (
    ((slice(1, None), slice(None)), (slice(0, -1), slice(None))),
    ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
    ((slice(None), slice(1, None)), (slice(None), slice(0, -1))),
    ((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
)


def distance_inside(mask: np.ndarray, limit: int = 64) -> np.ndarray:
    """Texels from each covered texel to the nearest uncovered one.

    Erosion by one ring per step rather than a per-texel chamfer: an island's
    interior is a few tens of texels deep, so this is a few tens of vectorised
    passes where the scalar version takes minutes on a 2048² sheet (scipy is
    not a dependency).  *limit* caps the depth, and a texel deeper than that
    reports *limit* — every reader here normalises by the max, so a cap that
    bites shows up as a flat plateau rather than as a wrong gradient.

    The sheet's border counts as uncovered.  An island running off the edge of
    the map is a layout the game never ships, and treating the edge as interior
    would make its distance depend on where the sheet happens to end.
    """
    cur = np.asarray(mask, dtype=bool)
    d = cur.astype(np.float32)
    for _ in range(int(limit)):
        if not cur.any():
            break
        e = cur.copy()
        e[1:, :] &= cur[:-1, :]
        e[:-1, :] &= cur[1:, :]
        e[:, 1:] &= cur[:, :-1]
        e[:, :-1] &= cur[:, 1:]
        e[0, :] = False
        e[-1, :] = False
        e[:, 0] = False
        e[:, -1] = False
        cur = e
        d += cur
    return d


def label_islands(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """``(labels, count)`` — 4-connected components of a coverage mask.

    ``-1`` where the mask is clear; labels are ``0..count-1`` ordered by
    **descending area**, because every caller wants "the big islands" and an
    arbitrary order makes that a second pass.

    Run-length union-find rather than iterative neighbour propagation: a
    propagation converges in as many passes as the island is long, and a
    garment's waistband is two thousand texels long on a 2048² sheet.  Runs per
    row are a few, so this is linear in the ink.
    """
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    parent: list[int] = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    runs_prev: list[tuple[int, int, int]] = []
    rows: list[list[tuple[int, int, int]]] = []
    for y in range(h):
        row = m[y]
        if not row.any():
            rows.append([])
            runs_prev = []
            continue
        # run starts/ends without a Python scan over every texel
        d = np.diff(row.astype(np.int8))
        starts = np.flatnonzero(d == 1) + 1
        ends = np.flatnonzero(d == -1) + 1
        if row[0]:
            starts = np.concatenate([[0], starts])
        if row[-1]:
            ends = np.concatenate([ends, [w]])
        runs: list[tuple[int, int, int]] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            rid = len(parent)
            parent.append(rid)
            for ps, pe, pid in runs_prev:
                if ps < e and s < pe:
                    union(rid, pid)
            runs.append((s, e, rid))
        rows.append(runs)
        runs_prev = runs

    roots: dict[int, int] = {}
    area: dict[int, int] = {}
    for runs in rows:
        for s, e, rid in runs:
            r = find(rid)
            roots[rid] = r
            area[r] = area.get(r, 0) + (e - s)
    order = sorted(area, key=lambda r: -area[r])
    rank = {r: i for i, r in enumerate(order)}
    out = np.full((h, w), -1, dtype=np.int32)
    for y, runs in enumerate(rows):
        for s, e, rid in runs:
            out[y, s:e] = rank[roots[rid]]
    return out, len(order)


def _dilate(out: dict, covered: np.ndarray, steps: int) -> None:
    """Grow every island outwards by *steps* texels, in place.

    One step copies from any already-covered 4-neighbour.  Which neighbour wins
    does not matter: the point is that a texel a bilinear tap can reach carries
    the surface's value rather than zero, and inside a step of the border the
    four candidates are the same value to within a texel of interpolation.
    """
    grown = covered.copy()
    for _ in range(int(steps)):
        frozen = grown.copy()          # one texel per step, not four
        if frozen.all():
            return
        for dst, src in _NEIGHBOURS:
            take = frozen[src] & ~grown[dst]
            if not take.any():
                continue
            for arr in out.values():
                arr[dst][take] = arr[src][take]
            grown[dst][take] = True

def distance_from(seed: np.ndarray, inside: np.ndarray,
                  limit: int = 96) -> np.ndarray:
    """Texels from each covered texel to the nearest *seed*, inside *inside*.

    :func:`distance_inside` measures to the nearest border of any kind, which
    is the wrong question for a trim: a skirt's island has a waist edge and a
    hem edge, and a band at "depth 4" from either is two bands.  Seeding the
    walk on one chosen edge gives an offset from **that** edge, which is what a
    hem, a cuff or a collar stripe is.

    Unreached texels come back as *limit*, which the caller can test for; a
    silent 0 would put them on the seed.
    """
    inside = np.asarray(inside, dtype=bool)
    cur = np.asarray(seed, dtype=bool) & inside
    out = np.full(inside.shape, float(limit), dtype=np.float32)
    out[cur] = 0.0
    for step in range(1, int(limit) + 1):
        nxt = np.zeros_like(cur)
        for dst, src in _NEIGHBOURS:
            nxt[dst] |= cur[src]
        nxt &= inside & (out > step - 0.5)
        if not nxt.any():
            break
        out[nxt] = step
        cur = nxt
    return out


def border_of(mask: np.ndarray) -> np.ndarray:
    """The one-texel rim of a coverage mask."""
    m = np.asarray(mask, dtype=bool)
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    e[0, :] = False
    e[-1, :] = False
    e[:, 0] = False
    e[:, -1] = False
    return m & ~e
