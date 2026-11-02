"""Garment, hair and skin maps, generated **against the stock ones**.

The sibling of :mod:`polylode.mesh.facetex`, which draws the face sheets.
``PROGRESS.md`` F0-3b / F0-6.

``CLAUDE.md`` 一之四 is the rule this follows, and it has two halves.  *Every
sheet here is drawn* — flat fills, distance fields, quantile ramps, at our own
resolution — but *nothing about them is invented*: every level, ratio and
region comes from the garment in hand, not from a rule that seemed reasonable.
Where a thing cannot be derived from the mesh, it is **read off the stock map**
rather than made up, and the colour mask is the case where that means the
regions themselves: which island is a sailor collar is not a fact about a UV
layout.  So this is not "none of Illusion's pixels" — it is "none of
Illusion's pixels *shipped*, and none of the layout guessed".  The earlier
framing (``著作權.md`` 四, measurements only) held until the first render
showed what a guessed layout costs: a navy collar the colour of its shirt.

What ``tools/part_tex_profile.py`` found, over 90 garment and 43 hair
renderers sampled across every clothing and hair category, is why this module
is short:

**Two thirds of garment ``_MainTex`` sheets are one flat value — and the rest
are drawings.**  Measured inside the texels the piece's own UV0 covers, and
among those only the opaque ones (the transparent ones keep compressed
garbage, §7.88).  Over the **whole** library, 772 renderers with an albedo:
the luminance spread is bimodal — **63.9% under 1** and **15.6% over 40** —
and only **50.6%** are 99% white.  So a flat sheet reproduces the majority
exactly, and whitening the rest **erases the garment's design**: the china
dress's navy floral, a maid apron's lace, a knight's panel lines.  The
decision is therefore per piece, from that piece's own spread, and the flat
ones are filled with **their own median colour** rather than 255 —
``cf_o_bot_knightSK_d`` is 233 everywhere and ``cf_o_glove_yumikake`` 237
(§7.155; the earlier 90-piece sample's median could not see a 16% tail).

**A garment's ``_ColorMask`` is binary, and its three channels overlap.**  The
fraction of covered texels that are neither off nor on is **0.0007**, so the
regions have hard edges and are redrawn as such.  They are *not* exclusive
though: ``create_topN`` lerps R→G→B in sequence, so a trim sits **over** its
base and both channels are on there — ``top/25`` has whole islands at
``R 1.00 G 1.00`` and ``bottom/1`` puts its entire G region inside its single
island.  ``purity`` (the largest channel's share per island) reads 1.00 on
everything and was misread as "one island, one slot" for two sections; the
column that answers the question is ``ovlp`` (§7.154).

**``_DetailMask.g`` hugs the island borders.**  It averages 33.5/255 over the
covered texels and is **1.57×** stronger over the outer 5% of each island's
depth than over the inner half.  That is the seam-and-fold shading, and it is
the one garment map with real structure — the two other channels measure 0.

**Hair is the colour mask.**  Only **87 of 473** hair renderers (18.4%) bind a
``_MainTex`` at all, while **473 of 473** bind a colour mask, and unlike the
garment ones it is *not* binary (mid fraction 0.133): it is a gradient down the
strand.  Which way, measured rather than assumed — the mask's blue channel
against the vertex height, on three styles from three families:

    cf_hair_b_18_00   r(y) = -0.86      cf_hair_s_03_00  r(y) = -0.94
    cf_hair_f_14      r(y) = -0.87      (green: +0.30 … +0.56)

Colour 3 is the **tips**, colour 2 the **roots**.  ``uv1`` — the strand layout
``_HairGloss`` runs along — correlates less well (-0.48 … -0.54), so height is
what this is drawn against.

**The body's albedo is pure white**, 2048², on both sexes: ``min`` is 255 on
all three channels.  ``cf_body_00_t`` was never a picture (``CLAUDE.md`` 七);
the skin colour is the card's.  Its colour mask is R everywhere except the
nails, which are B — and *that* is derivable from our own data, because a nail
is at the end of a digit: the texels Origin marks blue are dominated by
``cf_j_toes*``, ``cf_j_ring03_R`` and ``cf_j_thumb03_R``.  Green measures
0.0002 and is not drawn.

What is **not** reproduced, said plainly rather than papered over:

* the alpha cut-out, and this is the expensive one.  A garment's own alpha is
  inside ``main_opaque``'s clip test, so on lace and ribbon it *is* the shape —
  ``o_bra_a`` is 79% holes.  The median piece is fully opaque, but **42% of the
  89 measured are under 0.9**, so this is a large minority rather than a corner
  case: those garments come out solid.  :func:`lost_cutout` measures it per
  piece so a caller can say which ones.
* nothing about the *layout* — that is read off the garment's own stock maps
  rather than invented (``CLAUDE.md`` 一之四).  Which island carries the trim
  colour, how much groove shading this fabric has, how far down a style's tips
  are dyed: each is one number per piece, taken from the sheet in hand.  The
  medians above are only the fallback for a build that has no sheet to read,
  and the first cut used them everywhere and called the difference art
  direction — which turned a sailor top's navy collar the colour of its shirt.
* normal maps.  94% of garment renderers bind one; inventing bumps is not a
  measurement, so the slot is left unbound, which the shader's own property
  table already defines as no effect.
* accessories (``main_item``) and the head's ``face_t`` / ``face_mc``, which
  are their own families and are not answered here.

Colour space is chosen, not guessed, and three of the four maps do not care:
a byte that is only ever 0 or 255 decodes to 0 or 1 in **both** spaces, which
covers the white albedo and the binary garment mask.  The two that carry
intermediate values — the hair mask's gradient and ``_DetailMask.g`` — are
weights, so they are authored and declared **linear**, which is also what 98%
of the stock hair masks and 92% of the stock garment detail maps are.
"""
from __future__ import annotations

import numpy as np

from . import uvraster as UR
from ..core import swallowed

# ---------------------------------------------------------------------------
# measured targets — medians over 90 garment and 43 hair renderers, sampled
# equidistantly through every clothing and hair list
# (``tools/part_tex_profile.py --per 8 --pieces 2``)
# ---------------------------------------------------------------------------
#: ``_ColorMask`` channel coverage on a stock garment, as a fraction of the
#: texels its own mesh covers.  R is the dominant channel on **89 of 89** — a
#: garment is one colour with small trim regions, never three colours in
#: thirds, and that is what makes an island-labelling generator the right shape
#: rather than a convenient one.
GARMENT_MASK_G = 0.056
GARMENT_MASK_B = 0.007
#: ``_DetailMask.g``: mean over covered texels, and outer-5% over inner-half.
GARMENT_MD_LEVEL = 33.5 / 255.0
GARMENT_MD_EDGE = 1.57
HAIR_MD_LEVEL = 85.8 / 255.0
HAIR_MD_EDGE = 1.38
#: Hair ``_ColorMask``: how much of the strand takes colour 3 (tips) and
#: colour 2 (roots), and how much of the sheet is between values.  R is the
#: dominant channel on 39 of 39 here too.
HAIR_TIP_COVER = 0.121
HAIR_ROOT_COVER = 0.042
HAIR_MASK_MID = 0.133
#: Body ``_ColorMask.b`` — the nails.
NAIL_COVER = 0.0225

#: Default sheet sizes for a build that has no stock map to match.  Stock
#: garment maps are 512²–1024², hair masks 1024², the body albedo 2048².
DEFAULT_SIZE = 1024
SKIN_ALBEDO_SIZE = 2048
SKIN_MASK_SIZE = 1024

#: Bones a nail can be on.  Not a name pattern: ``ChaReference`` names the
#: distal joints of the four fingers ``*03``, the thumb the same, and the toes
#: are their own chain — so this is the list of *ends*, which is what a nail is
#: at.  A body whose rig omits them simply gets no nail region, which is the
#: honest outcome rather than a blue patch somewhere plausible.
NAIL_BONE_MARKS = ("03_l", "03_r", "toes")


def _lin(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _u8(x) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64) * 255.0 + 0.5,
                   0, 255).astype(np.uint8)


def _rgba(r, g, b, a, shape) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for i, ch in enumerate((r, g, b, a)):
        out[:, :, i] = ch if np.ndim(ch) else np.full((h, w), int(ch), np.uint8)
    return out


# ---------------------------------------------------------------------------
# the mesh's own footprint
# ---------------------------------------------------------------------------
_CACHE: dict = {}
_CACHE_MAX = 64


def _cache_key(mesh, shape) -> tuple | None:
    """A key that is stable across loads without hashing megabytes.

    Not ``hash()`` on anything textual — Python salts string hashes per
    process, so a cache keyed on one never hits twice (``CLAUDE.md`` 七).  The
    name plus the triangle and vertex counts identifies a renderer inside a
    run, and a collision would only reuse a coverage mask between two meshes
    with the same name and size, which is the same mesh.
    """
    name = getattr(mesh, "name", None)
    if not name:
        return None
    try:
        return (str(name), int(mesh.vertex_count), int(len(mesh.indices)),
                int(shape[0]), int(shape[1]))
    except (TypeError, ValueError, AttributeError):
        return None


def footprint(mesh, shape, attrs: dict | None = None):
    """``(coverage, {name: map})`` — the mesh drawn into its own UV0 grid.

    Cached per (mesh, size): every generator below needs it and a garment is
    asked for three maps in a row.
    """
    key = _cache_key(mesh, shape) if not attrs else None
    if key is not None and key in _CACHE:
        return _CACHE[key]
    uv = getattr(mesh, "uvs", None)
    if uv is None or getattr(mesh, "indices", None) is None:
        return None, {}
    maps, cov = UR.rasterise(uv, mesh.indices, attrs or {}, shape)
    if key is not None:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = (cov, maps)
    return cov, maps


def _edge_field(cov: np.ndarray, level: float, edge_ratio: float,
                scale: float = 0.25) -> np.ndarray:
    """A value that is *edge_ratio* times stronger at an island's rim.

    Normalised so the mean over the covered texels comes out at *level*
    whatever the falloff's shape is — the two measured numbers are a mean and
    a ratio, and hard-coding a gain instead would silently stop matching either
    of them the moment the shape is touched.
    """
    d = UR.distance_inside(cov)
    dn = d / max(float(d.max()), 1.0)
    shape = np.exp(-dn / max(scale, 1e-3))
    if not cov.any():
        return np.zeros_like(shape, dtype=np.float32)
    rim = cov & (dn <= 0.05)
    core = cov & (dn >= 0.5)
    s_rim = float(shape[rim].mean()) if rim.any() else 1.0
    s_core = float(shape[core].mean()) if core.any() else 0.0
    # gain k such that (1 + k*s_rim) / (1 + k*s_core) == edge_ratio
    denom = s_rim - edge_ratio * s_core
    k = (edge_ratio - 1.0) / denom if abs(denom) > 1e-6 else 0.0
    k = float(np.clip(k, 0.0, 40.0))
    field = 1.0 + k * shape
    mean = float(field[cov].mean()) or 1.0
    return _lin(field * (level / mean))


def _smooth1(v: np.ndarray, k: int = 2) -> np.ndarray:
    """Box-smooth a 1-D profile, wrapping — the angular one is a circle."""
    if k <= 0 or len(v) < 3:
        return v
    pad = np.concatenate([v[-k:], v, v[:k]])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(pad, ker, mode="valid")


def coord_field(ref_g, cov, coords, shape_bins, smooth=(2, 1)):
    """Redraw a map as a smooth function of two **geometric** coordinates.

    The stock ``_DetailMask`` is folds, seams and strand grooves, and §7.153's
    border-distance field reproduced its mean and edge ratio while explaining
    only **R² 0.19** of a garment's variance and **0.06** of a hairstyle's.
    Both are far better described by coordinates the mesh itself supplies:

    * garments — the angle **around the body** crossed with the distance to the
      island border: R² 0.45–0.59 on the sampled skirts;
    * hair — the strand layout on **UV1**: R² 0.59–0.93.

    So the map is redrawn as a smoothed profile over those two coordinates.
    It is a function evaluated per texel, at our resolution, from the mesh's
    own geometry — not a resample of anybody's pixels — and it is what
    ``own_assets`` is for: **remade**, referencing Origin closely (一之四).
    """
    nb0, nb1 = shape_bins
    a = np.clip((coords[0] * nb0).astype(int), 0, nb0 - 1)
    b = np.clip((coords[1] * nb1).astype(int), 0, nb1 - 1)
    table = np.zeros((nb0, nb1), dtype=np.float64)
    seen = np.zeros((nb0, nb1), dtype=bool)
    flat = a * nb1 + b
    idx = flat[cov]
    vals = ref_g[cov]
    sums = np.bincount(idx, weights=vals, minlength=nb0 * nb1)
    cnts = np.bincount(idx, minlength=nb0 * nb1)
    live = cnts > 0
    table.reshape(-1)[live] = sums[live] / cnts[live]
    seen.reshape(-1)[live] = True
    # Fill the bins nothing landed in from their row's mean, then smooth: a
    # hole would otherwise draw as a black notch across the garment.
    for i in range(nb0):
        row = table[i]
        if seen[i].any():
            row[~seen[i]] = row[seen[i]].mean()
        table[i] = _smooth1(row, smooth[1]) if smooth[1] else row
    for j in range(nb1):
        col = table[:, j]
        table[:, j] = _smooth1(col, smooth[0]) if smooth[0] else col
    return table[a, b]


#: How far from a fold the ink profile is measured, in texels, and how many
#: bands the along-the-line amplitude is split into.  Eight covers it: on
#: ``o_bot_skirt01_a`` the stock ink is 104 on the border texel and 1 by the
#: seventh (§7.156).
LINE_KMAX = 8
LINE_BANDS = (1, 8)
#: A **fold** is an edge whose two triangles turn by at least this much.  Not a
#: taste constant: a garment's dihedral histogram is bimodal — the sampled
#: renderers measure a median of 4.8–15° (flat cloth) against a p90 of 76–104°
#: (the pleats) — and 30° is the gap between the two humps.
FOLD_DEG = 30.0


def _mesh_edges(mesh, weld: float = 1e-5):
    """``(open boundary, valley fold, ridge fold)`` edge corner-pairs.

    All three are topology, not taste.  An **open boundary** edge has one
    triangle: the garment's hem, waist or cuff.  A **valley** edge has two
    triangles that turn *inwards* — a groove, which is the thing that
    self-shadows — and a **ridge** turns outwards; the sign is what tells them
    apart, and they alternate along a pleat.

    Vertices are welded by rounded position first: a UV seam splits one 3-D
    edge into two index pairs, and reading those as boundary would call every
    seam an open edge.
    """
    raw = getattr(mesh, "positions", None)
    idx = getattr(mesh, "indices", None)
    if raw is None or idx is None or len(raw) == 0:
        return None, None, None
    pos = np.asarray(raw, dtype=np.float64)
    tri = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
    if len(tri) == 0:
        return None, None, None
    _, wid = np.unique(np.round(pos / weld).astype(np.int64), axis=0,
                       return_inverse=True)
    c0 = tri[:, [0, 1, 2]].ravel()
    c1 = tri[:, [1, 2, 0]].ravel()
    face = np.repeat(np.arange(len(tri)), 3)
    key = np.sort(np.stack([wid[c0], wid[c1]], axis=1), axis=1)
    uid, inv, cnt = np.unique(key, axis=0, return_inverse=True,
                              return_counts=True)
    open_sel = cnt[inv] == 1
    two = np.nonzero(cnt == 2)[0]
    valley_sel = np.zeros(len(inv), dtype=bool)
    ridge_sel = np.zeros(len(inv), dtype=bool)
    if len(two):
        # the two entries of an edge, in face order: stable sort keeps the
        # lower face first, which is what makes the sign reproducible
        order = np.argsort(inv, kind="stable")
        offs = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        e0 = order[offs[two]]
        e1 = order[offs[two] + 1]
        n = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]],
                     pos[tri[:, 2]] - pos[tri[:, 0]])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.where(ln > 0, ln, 1.0)
        n0, n1 = n[face[e0]], n[face[e1]]
        v = pos[c1[e0]] - pos[c0[e0]]
        lv = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.where(lv > 0, lv, 1.0)
        ang = np.degrees(np.arctan2(np.sum(np.cross(n0, n1) * v, axis=1),
                                    np.sum(n0 * n1, axis=1)))
        hit = ang <= -FOLD_DEG
        valley_sel[e0[hit]] = True
        valley_sel[e1[hit]] = True
        out = ang >= FOLD_DEG
        ridge_sel[e0[out]] = True
        ridge_sel[e1[out]] = True
    return ((c0[open_sel], c1[open_sel]),
            (c0[valley_sel], c1[valley_sel]),
            (c0[ridge_sel], c1[ridge_sel]))


def _draw_uv_edges(pairs, uv, shape) -> np.ndarray:
    """Edge corner-pairs drawn into the texel grid of *shape*."""
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), dtype=bool)
    if pairs is None:
        return out
    a, b = pairs
    if len(a) == 0:
        return out
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    x0, y0 = uv[a, 0] * (w - 1), (1.0 - uv[a, 1]) * (h - 1)
    x1, y1 = uv[b, 0] * (w - 1), (1.0 - uv[b, 1]) * (h - 1)
    span = np.maximum(np.abs(x1 - x0), np.abs(y1 - y0))
    steps = int(np.clip(np.ceil(float(span.max())) * 2 + 2, 2, 512))
    t = np.linspace(0.0, 1.0, steps)
    for lo in range(0, len(a), 4096):           # chunked: steps x edges
        hi = min(lo + 4096, len(a))
        xs = np.round(x0[lo:hi, None] + (x1 - x0)[lo:hi, None] * t)
        ys = np.round(y0[lo:hi, None] + (y1 - y0)[lo:hi, None] * t)
        out[np.clip(ys, 0, h - 1).astype(np.int32).ravel(),
            np.clip(xs, 0, w - 1).astype(np.int32).ravel()] = True
    return out


def _grow(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    for _ in range(int(steps)):
        g = m.copy()
        g[1:, :] |= m[:-1, :]
        g[:-1, :] |= m[1:, :]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        m = g
    return m


def fold_maps(mesh, shape, cov) -> dict:
    """``{open, valley, ridge}`` texel masks, **undilated**, clipped to *cov*.

    Undilated because :func:`across_coord` has to tell the two folds apart and
    they are a median of **two texels** away from each other on a pleated
    skirt: growing the valley by one swallows the ridge, and the ratio then
    reads 0.07 on a ridge instead of 1.00 — which is why the first cut could
    not darken the ridge side at all (§7.156 六之四).
    """
    uv = getattr(mesh, "uvs", None)
    if uv is None:
        return {}
    opens, valleys, ridges = _mesh_edges(mesh)
    if opens is None:
        return {}
    out = {}
    for tag, pairs in (("open", opens), ("valley", valleys), ("ridge", ridges)):
        m = _draw_uv_edges(pairs, uv, cov.shape) & cov
        if m.any():
            out[tag] = m
    return out


def line_seeds(mesh, shape, cov, folds: dict | None = None) -> dict:
    """Where a drawn line could start: the island rim, minus the open edge.

    A garment's ``_DetailMask`` line is on the **fold**, and the three
    candidates differ in how much of the rim they call one: everything, the
    part that is not the hem or the waist, and the part that is a valley.  Which
    of them the piece actually uses is decided by the fit, not here — on the
    pleated skirt the stock ink measures 98/255 on the vertical borders against
    15 in the interior and 15 on the horizontal ones, but a pair of trousers
    puts 3% of its ink near a border at all (§7.156).
    """
    rim = UR.border_of(cov) & cov
    out = {"rim": rim}
    if not rim.any():
        return out
    if folds is None:
        folds = fold_maps(mesh, shape, cov)
    if not folds:
        return out
    inner = rim & ~_grow(folds.get("open", np.zeros_like(cov)), 1)
    if inner.any():
        out["fold"] = inner
        val = folds.get("valley")
        if val is not None:
            # **Grown by one here, unlike :func:`fold_maps`.**  The line's
            # profile wants every texel the crease actually covers, and the
            # rasterised edge misses some to aliasing: undilated, the same
            # model fits worse (R² 0.54 against 0.74).  Telling the two folds
            # apart is a different question and uses the raw masks.
            deep = inner & _grow(val, 1)
            if deep.any():
                out["valley"] = deep
    return out


def across_coord(cov, folds):
    """0 on the valley fold, 1 on the ridge one — or ``None``.

    A distance is symmetric and the stock ink is not: on ``o_bot_skirt01_a``
    one texel to one side of a fold measures 109 and one texel to the other 43,
    and the two folds are a **median of two texels apart** in UV, so no profile
    of "distance from the fold" can hold them apart.  A ratio can: it says
    which fold a texel belongs to as well as how far along it is, and it is the
    coordinate a pleat's cross-section actually runs on (§7.156).

    Takes the **raw** fold masks: a one-texel dilation of the valley swallows
    the ridge two texels away, and the ratio then reads 0.07 on a ridge instead
    of 1.00 — a coordinate that has stopped separating the thing it exists to
    separate, silently.
    """
    live_v, live_r = folds.get("valley"), folds.get("ridge")
    if live_v is None or live_r is None:
        return None
    if not (live_v & cov).any() or not (live_r & cov).any():
        return None
    dv = UR.distance_from(live_v & cov, cov, limit=32).astype(np.float32)
    dr = UR.distance_from(live_r & cov, cov, limit=32).astype(np.float32)
    return dv / np.maximum(dv + dr, 1e-6)


def side_factor(ref_g, cov, field, across, nb: int = 8):
    """*field* rescaled per ``across`` band so each band's mean matches *ref_g*.

    Eight numbers, and they are what turns a symmetric profile into the
    asymmetric one the sheet has: the same line model measures 35/255 on the
    ridge side against the stock's 14 without this, and 15 with it (§7.156).
    """
    xb = np.clip((np.asarray(across) * nb).astype(int), 0, nb - 1)
    out = np.array(field, dtype=np.float32, copy=True)
    for j in range(nb):
        m = cov & (xb == j)
        if not m.any():
            continue
        have = float(field[m].mean())
        if have > 1e-6:
            out[m] = field[m] * (float(ref_g[m].mean()) / have)
    return out


def line_depth(cov, seed, kmax: int = LINE_KMAX):
    """Texels from *seed*, capped at *kmax* — or ``None`` if the seed is empty.

    Hoisted out of :func:`line_field` because the amplitude is fitted at more
    than one band count and the distance does not depend on it: computing it
    per candidate was half the cost of the whole candidate set.
    """
    live = np.asarray(seed, dtype=bool) & cov
    if not live.any():
        return None
    d = UR.distance_from(live, cov, limit=kmax + 1)
    return np.clip(np.round(d).astype(int), 0, kmax)


def line_field(ref_g, cov, seed, along, kmax: int = LINE_KMAX, nb: int = 8,
               dk=None):
    """``P(distance from the seed) x A(along the seed)`` — both off *ref_g*.

    A binned function of two smooth coordinates cannot draw a hard line: it is
    low-pass by construction, and the pleat creases came out as soft wide bands
    (the render differed from the stock one by 3.36/255 over the skirt, 9.8% of
    its pixels by more than 12 levels).  This is the other shape a map can have
    — ink at a measured distance from a line the *mesh* supplies — and it costs
    about sixteen numbers: a cross-profile of nine and an amplitude of eight.
    """
    if dk is None:
        if seed is None:
            return None
        dk = line_depth(cov, seed, kmax)
        if dk is None:
            return None
    prof = np.zeros(kmax + 1)
    for k in range(kmax + 1):
        m = cov & (dk == k)
        if m.any():
            prof[k] = float(ref_g[m].mean())
    band = cov & (dk <= 1)
    hb = np.clip((np.asarray(along) * nb).astype(int), 0, nb - 1)
    amp = np.full(nb, np.nan)
    for j in range(nb):
        m = band & (hb == j)
        if m.any():
            amp[j] = float(ref_g[m].mean())
    if not np.isfinite(amp).any():
        return None
    mid = float(np.nanmean(amp)) or 1.0
    amp = np.where(np.isfinite(amp), amp, mid) / max(mid, 1e-6)
    return np.where(cov, prof[dk] * amp[hb], 0.0).astype(np.float32)


def _r2(ref_g, cov, field) -> float:
    v = float(ref_g[cov].var())
    if v <= 1e-9:
        return 1.0
    return float(1.0 - ((ref_g - field)[cov]).var() / v)


def best_coord_field(ref_g, cov, candidates, bins=(40, 24)):
    """Redraw *ref_g* over whichever candidate coordinate pair explains it best.

    One pair does not fit every piece: on ``cf_hair_b_14_00`` the map runs
    almost entirely along the strand (R² 0.92 on height) while its neighbours
    run across it (0.54–0.64 on UV1's u).  Picking per piece, by fit, is the
    difference between a map with grooves and a flat one — and the score is
    printed by ``tools/part_tex_profile.py`` so the choice is checkable.

    The result is renormalised to the reference's own mean and spread: binning
    and smoothing both pull towards the middle, and the mean is what the shading
    term actually multiplies.
    """
    best, best_score = None, -1e9
    for coords in candidates:
        if coords is None:
            continue
        f = coord_field(ref_g, cov, coords, bins)
        score = _r2(ref_g, cov, f)
        if score > best_score:
            best, best_score = f, score
    if best is None:
        return None, -1.0
    m0, s0 = float(ref_g[cov].mean()), float(ref_g[cov].std())
    m1, s1 = float(best[cov].mean()), float(best[cov].std())
    if s1 > 1e-6 and s0 > 1e-6:
        best = (best - m1) * (s0 / s1) + m0
    else:
        best = best - m1 + m0
    return np.clip(best, 0.0, 255.0), best_score


def geometry_coords(mesh, shape, cov):
    """``(theta, height, border depth, uv1 u, uv1 v)`` for a piece, 0..1 each.

    Everything here comes from the mesh: where a texel sits around the body,
    how far up it is, how deep into its island, and where it lands on the
    strand layout.  ``None`` for a coordinate this mesh cannot supply.
    """
    out = {}
    verts = getattr(mesh, "positions", None)
    attrs = {}
    if verts is not None and len(verts):
        attrs["p"] = np.asarray(verts, dtype=np.float32)
    raw = getattr(mesh, "uv1", None)
    if raw is not None and len(raw):
        uv1 = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
        attrs["a"] = uv1[:, 0:1]
        attrs["b"] = uv1[:, 1:2]
    if not attrs:
        return out
    _, maps = footprint(mesh, shape, attrs)

    def norm(x):
        lo, hi = float(x[cov].min()), float(x[cov].max())
        return (x - lo) / max(hi - lo, 1e-6)

    if "p" in maps:
        pos = maps["p"]
        c = np.median(pos[cov], axis=0)
        out["theta"] = (np.arctan2(pos[..., 2] - c[2], pos[..., 0] - c[0])
                        + np.pi) / (2.0 * np.pi)
        out["height"] = norm(pos[..., 1])
    if "a" in maps:
        out["u1"] = norm(maps["a"][..., 0])
        out["v1"] = norm(maps["b"][..., 0])
    d = UR.distance_inside(cov)
    out["depth"] = d / max(float(d.max()), 1.0)
    return out


def _quantile_region(value: np.ndarray, cov: np.ndarray, cover: float,
                     high: bool, soft: float) -> np.ndarray:
    """A soft 0..1 region holding *cover* of the covered texels.

    A threshold picked as a quantile rather than as a constant, so the measured
    coverage is hit on **any** mesh instead of on the one the constant was
    fitted to.  *soft* is the ramp's width in the same units, and what sets it
    is the measured fraction of samples between values
    (:data:`HAIR_MASK_MID`, 0.133).
    """
    if not cov.any() or cover <= 0.0:
        return np.zeros(cov.shape, dtype=np.float32)
    v = value[cov]
    q = np.percentile(v, 100.0 * (1.0 - cover) if high else 100.0 * cover)
    if high:
        out = (value - q) / max(soft, 1e-6)
    else:
        out = (q - value) / max(soft, 1e-6)
    out = np.clip(out + 0.5, 0.0, 1.0)
    return _lin(np.where(cov, out, 0.0))


# ---------------------------------------------------------------------------
# garments
# ---------------------------------------------------------------------------
#: Luminance spread (p95 − p5 over the opaque texels the mesh covers) above
#: which a garment's ``_MainTex`` is a **drawing** rather than a flat fill.
#: The library is bimodal — 63.9% of the 772 renderers with an albedo measure
#: under 1, and 15.6% are over 40 — so the cut sits in the gap between them.
#: Below it the sheet is one value and we redraw it exactly; above it there is
#: a picture on it (the china dress's navy floral, a maid apron's lace, a
#: knight's panel lines) and whitening the sheet **erases the garment's
#: design** (§7.155).
GARMENT_DRAWN_SPREAD = 8.0


#: When set, a drawn albedo is **redrawn** rather than kept.  Off by default,
#: because ``CLAUDE.md`` 一之四 says a design is not something a measurement
#: replaces — but a caller composing an original character wants the base
#: garment without the crest, the buttons or the print that came with it, and
#: that is a deliberate choice rather than an accident.
REDRAW = {"on": False, "trim": 0.0}

#: Where generated sheets are written, and how small a colour-mask region has
#: to be before it is folded back into the base.  Both off by default: the
#: generator's product is normally an array handed straight to the composite,
#: and nothing reaches the disk.
DUMP = {"dir": None, "n": 0}
#: Mask regions under this share of a piece's footprint are dropped — buttons,
#: studs, small badges.  0 keeps every one of them.
MASK_MIN_REGION = {"v": 0.0}


def own_assets_dir():
    """``own_assets/`` — the one place anything **we** author is written.

    Kept apart from ``assets_baked`` (converted Illusion items) and from
    ``preview`` (throwaway renders) so that "what is ours" is a folder rather
    than an argument.
    """
    from ..core.paths import ROOT
    return ROOT / "own_assets"


def set_dump(path, min_region: float = 0.0) -> None:
    """Write every generated sheet as it is drawn.  ``True`` = the default dir.

    The generator normally produces arrays, not files — this is what makes the
    output inspectable and editable, and it is also the honest answer to "is
    this a new texture or the old one": open the folder.
    """
    from pathlib import Path as _P
    if path is None or path is False:
        DUMP["dir"] = None
    elif path is True:
        DUMP["dir"] = own_assets_dir() / "tex"
    else:
        DUMP["dir"] = _P(path)
    DUMP["n"] = 0
    if DUMP["dir"] is not None:
        DUMP["dir"].mkdir(parents=True, exist_ok=True)
    MASK_MIN_REGION["v"] = float(min_region)


def _dump(category: str, name: str, slot: str, px) -> None:
    if DUMP["dir"] is None or px is None:
        return
    try:
        from PIL import Image
        def safe(t):
            return "".join(c if (c.isalnum() or c in "._-") else "_" for c in t)
        folder = DUMP["dir"] / safe(category or "misc")
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(px)).save(folder / f"{safe(name)}_{slot}.png")
        DUMP["n"] += 1
    except Exception as exc:                                 # noqa: BLE001
        swallowed.note("mesh.parttex._dump: safe", exc)


def set_redraw(on: bool, trim: float = 0.0) -> None:
    """Redraw drawn albedos, optionally with an edge trim of *trim* strength.

    The trim is a light line just inside each UV island's border — a piped
    edge.  It is drawn from the mesh's own layout, so it lands on seams and
    hems rather than anywhere a rule happened to put it.
    """
    REDRAW["on"] = bool(on)
    REDRAW["trim"] = float(trim)


def garment_main(shape, ref=None, mesh=None):
    """``_MainTex`` — the piece's own flat value, or ``None`` to keep the stock.

    Two thirds of the stock sheets are a single value over the texels their own
    mesh samples, and for those this reproduces it exactly.  ``None`` is the
    answer for the rest: an albedo that carries a drawing is not something a
    measurement replaces, and the honest outcome is to leave it and say so
    (``CLAUDE.md`` 一之四, and the same verdict §7.152 reached for accessories).

    The value is the piece's **own median**, not 255.  Some flat sheets are
    tinted — ``cf_o_bot_knightSK_d`` is 233 everywhere, ``cf_o_glove_yumikake``
    237 — and painting those white is an 8.6% brightness error on a garment
    that measured "flat".

    The alpha is 255 because we have no way to know a cut-out; see
    :func:`lost_cutout` for what that costs on which garment.
    """
    stats = albedo_stats(ref)
    if stats is None:
        rgb = [255.0, 255.0, 255.0]
    else:
        rgb, spread = stats
        if spread >= GARMENT_DRAWN_SPREAD and not REDRAW["on"]:
            return None
        if spread >= GARMENT_DRAWN_SPREAD:
            # Redrawing a *drawn* sheet: its median is the ink's median, which
            # on a printed garment is the print rather than the cloth.  Take
            # the bright end instead, which is the fabric the print sits on.
            a = np.asarray(ref, dtype=np.float64)
            al = a[..., 3] if a.shape[2] >= 4 else np.full(a.shape[:2], 255.0)
            op = al >= 128.0
            rgb = [float(np.percentile(a[..., c][op], 75)) for c in range(3)]
    out = _rgba(int(rgb[0]), int(rgb[1]), int(rgb[2]), 255, shape)
    trim = float(REDRAW["trim"])
    if trim > 0.0 and mesh is not None:
        cov, _ = footprint(mesh, shape)
        if cov is not None and cov.any():
            d = UR.distance_inside(cov)
            # A line a couple of texels in from the border, not at it: at the
            # border it is eaten by the bilinear tap that already blends the
            # island's edge with whatever is outside.
            band = cov & (d >= 1.5) & (d <= 1.5 + max(1.0, shape[0] / 256.0))
            lift = np.clip(np.asarray(rgb) + 255.0 * trim, 0, 255)
            for c in range(3):
                out[:, :, c] = np.where(band, int(lift[c]), out[:, :, c])
    return out


def albedo_stats(ref):
    """``((r, g, b) median, luminance spread)`` of a stock albedo, or ``None``.

    Over the **opaque** texels only: a compressed albedo keeps garbage RGB
    where it is transparent, and including those measured 9% of a plain white
    shirt as "dark" (§7.88, §7.144 一).
    """
    if ref is None or np.ndim(ref) != 3:
        return None
    a = np.asarray(ref, dtype=np.float64)
    if a.shape[2] < 3:
        return None
    al = a[..., 3] if a.shape[2] >= 4 else np.full(a.shape[:2], 255.0)
    op = al >= 128.0
    if not op.any():
        return None
    lum = a[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    spread = float(np.percentile(lum[op], 95) - np.percentile(lum[op], 5))
    rgb = [float(np.median(a[..., c][op])) for c in range(3)]
    return rgb, spread


def _close(mask, inside, steps: int = 1):
    """Morphological close — dilate then erode — clipped to *inside*."""
    m = np.asarray(mask, dtype=bool)
    for _ in range(int(steps)):
        g = m.copy()
        g[1:, :] |= m[:-1, :]
        g[:-1, :] |= m[1:, :]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        m = g & inside
    for _ in range(int(steps)):
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        m = e | (mask & inside)
    return m


def _interval(region, cov, x):
    lo, hi = float(np.percentile(x[region], 3)), float(np.percentile(x[region], 97))
    return cov & (x >= lo) & (x <= hi)


def _interval_along(region, cov, x, by, nb: int = 32):
    """An interval in *x* whose two limits follow *by*.

    A pleated skirt's hem is a band at a fixed height **that zigzags**, so one
    flat height interval either misses half of it or swallows the pleats above
    it — the straight version scored IoU 0.05.  Letting the limits vary with
    the angle around the body is what makes the band follow the hem.
    """
    b = np.clip((by * nb).astype(int), 0, nb - 1)
    lo = np.full(nb, np.nan)
    hi = np.full(nb, np.nan)
    for k in range(nb):
        sel = region & (b == k)
        if sel.any():
            lo[k] = np.percentile(x[sel], 3)
            hi[k] = np.percentile(x[sel], 97)
    if np.all(np.isnan(lo)):
        return cov & False
    med_lo, med_hi = np.nanmedian(lo), np.nanmedian(hi)
    lo = np.where(np.isnan(lo), med_lo, lo)
    hi = np.where(np.isnan(hi), med_hi, hi)
    lo, hi = _smooth1(lo, 1), _smooth1(hi, 1)
    return cov & (x >= lo[b]) & (x <= hi[b])


def best_band(region, cov, coords: dict):
    """Redraw a mask region as an interval — or a box — in the mesh's own axes.

    A sailor collar's stripe, a cuff, a hem: each is a line drawn *parallel to
    an edge of the garment*, so it is an interval of some coordinate rather
    than an arbitrary shape, and drawing it as one is how we make our own
    rather than copy theirs.

    Which coordinate has to be chosen per region, not assumed.  Distance from
    the island border is the obvious candidate and it is wrong on a skirt: the
    island has a waist edge **and** a hem edge, so the hem band's depth
    interval selects both and scores IoU 0.05.  Height alone finds it.  A
    collar stripe needs depth *and* height together.  So every single axis and
    every pair is tried and the best kept, with the score returned for the
    caller to gate on.
    """
    if not region.any() or not coords:
        return None, 0.0
    names = [k for k, v in coords.items() if v is not None]
    best, score = None, 0.0

    def take(cand):
        nonlocal best, score
        inter = float((cand & region).sum())
        union = float((cand | region).sum()) or 1.0
        if inter / union > score:
            best, score = cand, inter / union

    for a in names:
        take(_interval(region, cov, coords[a]))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            take(_interval(region, cov, coords[a])
                 & _interval(region, cov, coords[b]))
    for a in names:
        for b in names:
            if a != b:
                take(_interval_along(region, cov, coords[a], coords[b]))
    # An offset from **the edge this trim runs along**.  Distance to the
    # nearest border of any kind cannot express it: a skirt island has a waist
    # edge and a hem edge, so "depth 4" is two bands and the hem scored 0.05.
    # Seeding the walk on the rim next to the region itself gives the offset
    # that a hem, a cuff or a collar stripe actually is.
    rim = UR.border_of(cov)
    grown = region.copy()
    for _ in range(3):
        g2 = grown.copy()
        g2[1:, :] |= grown[:-1, :]
        g2[:-1, :] |= grown[1:, :]
        g2[:, 1:] |= grown[:, :-1]
        g2[:, :-1] |= grown[:, 1:]
        grown = g2
    seed = rim & grown
    if seed.any():
        dd = UR.distance_from(seed, cov)
        take(_interval(region, cov, dd))
    # A collar carries **two or three** parallel stripes, and one band cannot
    # be two lines: fitted whole, the sailor stripe scored 0.43.  Each
    # connected piece is its own line, so each gets its own band and the union
    # is the answer.
    lab, count = UR.label_islands(region)
    if 1 < count <= 12:
        parts = []
        for isl in range(count):
            sub = lab == isl
            if sub.sum() < 24:
                parts.append(sub)
                continue
            got, _ = best_band(sub, cov, coords)
            parts.append(got if got is not None else sub)
        merged = np.zeros_like(region)
        for q in parts:
            merged |= q
        take(merged)
    return best, score


def garment_mask(mesh, shape, ref=None, *, g_cover: float = GARMENT_MASK_G,
                 b_cover: float = GARMENT_MASK_B) -> np.ndarray | None:
    """``_ColorMask`` — whole islands labelled with a colour slot.

    R over the whole footprint (and over the background, as the stock sheets
    have it, so a bilinear tap at a seam never reads an unassigned texel), and
    the other two channels on the islands that carry them.

    **Which island carries which is read off the garment's own stock mask**
    (*ref*), one integer per island: rasterise this piece's UV, label the
    islands, and take the dominant channel of the stock sheet inside each.
    The form is ours — binary, one slot per island, which is what the
    population measures (mid 0.0007, purity 1.00) — and the *layout* is the
    garment's own, which is not something a rule can supply.

    The first cut promoted the **smallest** islands instead and called it art
    direction.  It is not: the label is a measurement, and skipping it turned a
    sailor top's navy collar the same colour as its shirt and left the 48% of
    garment renderers that are a single island entirely one colour
    (``CLAUDE.md`` 一之四).

    Without *ref* — a build with no stock sheet to read — every island takes
    colour 1 and the caller is told, rather than being handed an invented trim.
    """
    cov, _ = footprint(mesh, shape)
    if cov is None or not cov.any():
        return None
    src = _as_mask(ref, shape)
    if src is None:
        # Nothing to read: one colour, and the caller is told rather than
        # handed an invented trim.
        return _rgba(255, 0, 0, 255, shape)
    # **The three channels are not exclusive**, which the first two cuts both
    # had wrong.  ``create_topN`` lerps R then G then B *in sequence*, so a
    # texel can be in two of them and the later one wins: ``top/25`` has whole
    # islands measuring ``R 1.00 G 1.00``, and ``bottom/1`` puts its entire G
    # region **inside** its one island.  Nor is R always the base — ``top/11``
    # measures 0.21/0.44/0.35.  So all three regions are read, per texel.
    on = [cov & (src[..., c] > 128) for c in range(3)]
    # **Trim is redrawn, not copied.**  A collar stripe, a cuff and a hem are
    # all lines parallel to the garment's edge, so each is an interval of the
    # distance-to-border field — ours is drawn from that field at the depth the
    # stock one occupies, and only where the fit says the region really is
    # edge-parallel (``CLAUDE.md`` 一之四: remade, referencing closely).
    axes = geometry_coords(mesh, shape, cov)
    for c in (1, 2):
        band, iou = best_band(on[c], cov, axes)
        if band is not None and iou >= 0.6:
            # Close it first: an interval fitted to a thin trim is right down
            # the middle and ragged at the ends, and a hem that is 82% right is
            # a hem with gaps in it — which reads as a dashed line rather than
            # as a band.
            on[c] = _close(band, cov, 2)
    small = float(MASK_MIN_REGION["v"])
    if small > 0.0:
        # Fold anything smaller than *small* of the footprint back into the
        # base channel: on a school uniform those regions are the buttons and
        # the studs, and a design without them is a design decision the caller
        # is making, not a measurement going wrong.
        total = float(cov.sum())
        # **Holes only.**  A texel in no channel composites to ``c = 1`` —
        # white, whatever the card asked for — which is how a school uniform's
        # buttons are drawn: a small gap in the base, one point in UV repeated
        # eight times on the mesh.  The *regions* (G and B) are the trim: a
        # skirt's hem band is one of them, and it is what makes the pleats
        # visible at all, so an earlier cut that dropped small regions too
        # turned the whole skirt into a black silhouette — in Origin's own render
        # the pleats are that white line following their zigzag and nothing
        # else.
        gap = cov & ~on[0] & ~on[1] & ~on[2]
        if gap.any():
            lab, count = UR.label_islands(gap)
            for isl in range(count):
                s = lab == isl
                if float(s.sum()) / total < small:
                    on[0] = on[0] | s
    return _rgba(np.where(on[0] | ~cov, 255, 0).astype(np.uint8),
                 np.where(on[1], 255, 0).astype(np.uint8),
                 np.where(on[2], 255, 0).astype(np.uint8), 255, shape)


def _as_mask(ref, shape):
    """A stock map resampled onto *shape*, or ``None``.

    Nearest-neighbour on purpose: this is only ever read as "which channel is
    on here", and interpolating a binary label invents values between the two.
    """
    if ref is None or np.ndim(ref) != 3 or np.asarray(ref).shape[2] < 3:
        return None
    a = np.asarray(ref)
    h, w = int(shape[0]), int(shape[1])
    if a.shape[0] == h and a.shape[1] == w:
        return a.astype(np.float32)
    yi = np.clip((np.arange(h) + 0.5) * a.shape[0] / h, 0,
                 a.shape[0] - 1).astype(int)
    xi = np.clip((np.arange(w) + 0.5) * a.shape[1] / w, 0,
                 a.shape[1] - 1).astype(int)
    return a[np.ix_(yi, xi)].astype(np.float32)


def garment_detail(mesh, shape, ref=None, *, level: float = GARMENT_MD_LEVEL,
                   edge: float = GARMENT_MD_EDGE) -> np.ndarray | None:
    """``_DetailMask`` — G is the seam shading, R and B measure zero.

    ``shade = min(1 - _LineMask.b, 1 - _DetailMask.g)`` in ``main_opaque``'s
    forward base, so G is a *darkening*.  §7.144 measured it as border-hugging
    (1.57× over the outer 5% of an island's depth) and generated a distance
    field to match — which reproduced that ratio and the mean exactly while
    throwing the map away: fitted against the stock sheets, the border model
    explains **R² 0.19** of a garment's variance and **0.06** of a hairstyle's,
    23 of 23 and 25 of 26 under 0.5.  What is actually on those maps is folds,
    seams and per-strand grooves, and on hair the generated version measured
    std 10 against the stock's 60–83 — visibly flatter hair.

    An edge ratio is a *marginal* statistic and matching a marginal is not
    reproducing the picture (the same mistake as reading ``purity`` for
    exclusivity in §7.154).  So the map is **redrawn** — not kept, because
    ``own_assets`` is for assets we remade — and there are two shapes it can
    take, because garments come in two kinds:

    * a smooth function of two coordinates the mesh supplies (the angle around
      the body, the height, the depth into the island).  That is what trousers
      and jumpers carry — soft folds all over: ``o_bot_pants01_a`` puts **3%**
      of its ink within two texels of a border;
    * a **line**, drawn at a measured distance from a fold the mesh names.
      That is what a pleated skirt carries: **95%** of ``o_bot_skirt01_a``'s
      ink is within two texels of an island border and it measures 98/255 there
      against 15 in the interior.  A binned smooth field is low-pass by
      construction and draws that as a soft wide band, which is what "the
      pleat lines are missing" looked like (§7.156).

    Which one is not chosen here — every candidate is scored against the
    garment's own sheet and the best kept, the same per-piece decision
    :func:`garment_main` makes for a drawn albedo.  The chosen model is
    recorded in :data:`DETAIL_MODEL` so a tool can print it.
    """
    cov, _ = footprint(mesh, shape)
    if cov is None or not cov.any():
        return None
    src = _as_mask(ref, shape)
    if src is None:
        return _rgba(0, _u8(_edge_field(cov, level, edge)), 0, 255, shape)
    ref_g = src[..., 1]
    co = geometry_coords(mesh, shape, cov)
    cands = []
    for a, b in (("theta", "depth"), ("theta", "height"), ("height", "depth")):
        if a in co and b in co:
            cands.append((f"{a}x{b}",
                          coord_field(ref_g, cov, (co[a], co[b]), (48, 16)),
                          True))
    along = co.get("height")
    if along is None:
        along = co.get("theta", np.zeros(cov.shape, dtype=np.float32))
    folds = fold_maps(mesh, shape, cov)
    seeds = line_seeds(mesh, shape, cov, folds)
    across = across_coord(cov, folds)
    if across is not None:
        # No smoothing on this pair: the wrap-around box filter the coordinate
        # tables use would blend the valley bin into the ridge bin, which is
        # the very distinction this coordinate exists to make.
        cands.append(("across", coord_field(ref_g, cov, (across, along),
                                            (16, 8), smooth=(0, 0)), False))
    for tag, seed in seeds.items():
        dk = line_depth(cov, seed)
        if dk is None:
            continue
        for nb in LINE_BANDS:
            f = line_field(ref_g, cov, seed, along, nb=nb, dk=dk)
            if f is None:
                continue
            cands.append((f"line:{tag}/{nb}", f, False))
            if across is not None:
                cands.append((f"line:{tag}/{nb}+side",
                              side_factor(ref_g, cov, f, across), False))
    g, name = _best_field(ref_g, cov, cands,
                          str(getattr(mesh, "name", "") or ""))
    if g is None:
        return _rgba(0, _u8(_edge_field(cov, level, edge)), 0, 255, shape)
    return _rgba(0, g.astype(np.uint8), 0, 255, shape)


#: What :func:`garment_detail` picked, per renderer: ``{name: (model, R²)}``.
#: A generator that chooses per piece has to say which choice it made, or
#: "ours" stops meaning one thing.
DETAIL_MODEL: dict = {}


def _best_field(ref_g, cov, cands, name: str):
    """The candidate that explains *ref_g* best, renormalised if it is a field.

    A coordinate table is renormalised to the reference's own mean and spread —
    binning and smoothing both pull towards the middle.  A line is **not**:
    its levels are measured means already, and shifting them would lift the
    unlit cloth off zero, which is the whole of the line's contrast.
    """
    best, best_score, best_name = None, -1e9, ""
    for tag, field, renorm in cands:
        if field is None:
            continue
        score = _r2(ref_g, cov, field)
        if score > best_score:
            best, best_score, best_name = (field, renorm), score, tag
    if best is None:
        return None, ""
    field, renorm = best
    if renorm:
        m0, s0 = float(ref_g[cov].mean()), float(ref_g[cov].std())
        m1, s1 = float(field[cov].mean()), float(field[cov].std())
        field = ((field - m1) * (s0 / s1) + m0) if s1 > 1e-6 and s0 > 1e-6 \
            else field - m1 + m0
    if name:
        DETAIL_MODEL[name] = (best_name, round(float(best_score), 3))
    return np.clip(field, 0.0, 255.0), best_name


# ---------------------------------------------------------------------------
# hair
# ---------------------------------------------------------------------------
def hair_mask(mesh, shape, ref=None, *, tip: float = HAIR_TIP_COVER,
              root: float = HAIR_ROOT_COVER) -> np.ndarray | None:
    """``_ColorMask`` — colour 1 overall, colour 3 at the tips, 2 at the roots.

    ``create_hair`` lerps R then G then B, so blue wins where it overlaps; the
    gradient is in the channel rather than in the geometry, which is why this
    is the one mask that is *not* binary.  The axis is the piece's own vertex
    height, which is what the stock masks correlate with (-0.86 … -0.94).
    """
    verts = getattr(mesh, "positions", None)
    if verts is None or len(verts) == 0:
        return None
    y = np.asarray(verts, dtype=np.float32)[:, 1:2]
    cov, maps = footprint(mesh, shape, {"y": y})
    if cov is None or not cov.any():
        return None
    # How much of *this* style takes colour 2 and colour 3, off its own stock
    # mask.  A twin-tail with dyed tips and a bob do not share a number, and
    # the population median is neither of them (``CLAUDE.md`` 一之四).
    mid_target = HAIR_MASK_MID
    src = _as_mask(ref, shape)
    if src is not None and cov.any():
        root = float(np.mean(src[..., 1][cov] > 128)) or root
        tip = float(np.mean(src[..., 2][cov] > 128)) or tip
        mid_target = float(np.mean((src[..., :3][cov] > 32)
                                   & (src[..., :3][cov] < 223)))
    hy = maps["y"][..., 0]
    span = float(hy[cov].max() - hy[cov].min()) or 1.0
    # The ramp's width is set by the measured "fraction of texels between
    # values" (0.15), not by a constant that looked right on one style: how far
    # a given height band reaches depends on how the strands are laid out, so a
    # fixed width lands somewhere different on every piece.  Searched rather
    # than solved because the relation runs through the mesh's own height
    # histogram; six thresholdings of an array already in hand.
    #
    # ``mid`` is counted the way ``tools/part_tex_profile.py`` counts it —
    # over the **three channel samples** of each texel, not per texel — because
    # that is the ruler :data:`HAIR_MASK_MID` was read off.  Counting per texel
    # instead makes the same sheet measure three times higher and picks a ramp
    # a third as wide, which is a gradient nobody asked for.
    best, best_err = None, 1e9
    for k in (0.06, 0.12, 0.20, 0.32, 0.50, 0.75, 1.10):
        b = _quantile_region(hy, cov, tip, high=False, soft=k * span)
        g = _quantile_region(hy, cov, root, high=True, soft=k * span)
        ones = np.ones_like(b)
        chans = np.stack([ones, g, b], axis=-1)[cov]
        mid = float(np.mean((chans > 0.125) & (chans < 0.875)))
        err = abs(mid - mid_target)
        if err < best_err:
            best, best_err = (g, b), err
    g, b = best
    return _rgba(255, _u8(g), _u8(b), 255, shape)


def hair_detail(mesh, shape, ref=None, *, level: float = HAIR_MD_LEVEL,
                edge: float = HAIR_MD_EDGE) -> np.ndarray | None:
    """``_DetailMask`` for ``main_hair``: R = 1-G, A = (1-G)(1-B) downstream.

    Redrawn over the **strand layout on UV1** — the same coordinate
    ``_HairGloss`` runs along.  That is where the grooves are: fitted against
    the stock sheets, UV1 explains **R² 0.59–0.93** of their variance while the
    border field explains 0.06, and a border field drew hair at std 10 against
    the stock's 60–83 — visibly flat (:func:`coord_field`).
    """
    cov, _ = footprint(mesh, shape)
    if cov is None or not cov.any():
        return None
    src = _as_mask(ref, shape)
    if src is None:
        return _rgba(0, _u8(_edge_field(cov, level, edge)), 0, 255, shape)
    co = geometry_coords(mesh, shape, cov)
    pairs = [(co[a], co[b]) for a, b in
             (("u1", "v1"), ("height", "u1"), ("height", "theta"),
              ("theta", "depth"))
             if a in co and b in co]
    g, _score = best_coord_field(src[..., 1], cov, pairs, (48, 48))
    if g is None:
        return _rgba(0, _u8(_edge_field(cov, level, edge)), 0, 255, shape)
    return _rgba(0, g.astype(np.uint8), 0, 255, shape)


# ---------------------------------------------------------------------------
# skin
# ---------------------------------------------------------------------------
def skin_albedo(size: int = SKIN_ALBEDO_SIZE) -> np.ndarray:
    """``cf_body_00_t`` — white.  Measured ``min`` 255 on every channel."""
    return _rgba(255, 255, 255, 255, (size, size))


def nail_region(mesh, skin, bones, shape) -> np.ndarray | None:
    """The texels a nail sits on, from the rig — or ``None`` if it cannot tell.

    Candidates are the texels whose dominant skin bone is the *end* of a digit;
    among them the ones bound hardest to that bone are kept, until the measured
    2.25% of the body's footprint is covered.  Weight is the ordering because a
    nail is at the very tip, where the distal bone owns the vertex outright,
    while the texels further back share with the joint above.
    """
    if skin is None or not bones or mesh is None:
        return None
    skin = np.asarray(skin)
    if skin.ndim != 2 or skin.shape[1] < 8:
        return None
    w4, i4 = skin[:, 0:4], skin[:, 4:8]
    pick = np.argmax(w4, axis=1)
    rows = np.arange(len(skin))
    top_w = w4[rows, pick].astype(np.float32)
    top_i = i4[rows, pick].astype(np.int32)
    marks = np.zeros(len(bones), dtype=bool)
    for k, name in enumerate(bones):
        low = str(name).lower()
        marks[k] = any(m in low for m in NAIL_BONE_MARKS)
    if not marks.any():
        return None
    is_nail = marks[np.clip(top_i, 0, len(bones) - 1)].astype(np.float32)
    cov, maps = footprint(mesh, shape,
                          {"nail": np.stack([is_nail, top_w], axis=1)})
    if cov is None or not cov.any():
        return None
    cand = cov & (maps["nail"][..., 0] > 0.5)
    if not cand.any():
        return None
    want = int(NAIL_COVER * float(cov.sum()))
    if want <= 0:
        return None
    wgt = maps["nail"][..., 1]
    vals = wgt[cand]
    if len(vals) > want:
        cut = float(np.partition(vals, len(vals) - want)[len(vals) - want])
        cand = cand & (wgt >= cut)
    return cand


def skin_mask(mesh=None, skin=None, bones=None, size: int = SKIN_MASK_SIZE
              ) -> np.ndarray:
    """``cf_body_00_mc`` — B on the nails, G unused, R unread.

    ``create_body`` is the authority on what each channel is, and it is not the
    garment shader's three colours::

        main.rgb *= lerp(1, _Color2, mask.g * _Color2.a)    # shading tone
        …paint layers…
        o.rgb     = lerp(col, col * _Color5, mask.b)        # the nail colour

    — **red is never sampled**.  It is written as 255 here only because the
    stock sheet has it that way (99% of the map), so a side-by-side comparison
    is not confused by a channel neither shader reads.

    G is left at 0 because the stock mask leaves it there: 0.0002 of the
    covered texels on the female body, all of them on ``cf_s_ana``.
    """
    shape = (size, size)
    nails = nail_region(mesh, skin, bones, shape) if mesh is not None else None
    if nails is None:
        return _rgba(255, 0, 0, 255, shape)
    return _rgba(np.where(nails, 0, 255).astype(np.uint8), 0,
                 np.where(nails, 255, 0).astype(np.uint8), 255, shape)


# ---------------------------------------------------------------------------
# what a generated set costs
# ---------------------------------------------------------------------------
#: The combo-top sub-part lists: garments in every respect but their name.
COMBO_CATS = ("cpo_sailor_a", "cpo_sailor_b", "cpo_sailor_c",
              "cpo_jacket_a", "cpo_jacket_b", "cpo_jacket_c")
HAIR_CATS = frozenset(("hair_back", "hair_front", "hair_side", "hair_option"))


def _garment_cats() -> frozenset:
    """Which categories this module answers for — the wearer's own slot list.

    By category rather than by shader family, because ``main_item`` is *also*
    what accessories use and they are not measured here: a hat with a badge
    painted on it is a drawing, and handing it a white sheet would be an
    invention rather than a match.

    Taken from ``outfit.CLOTHES_SLOT_NAMES`` rather than restated, because a
    restated list is a list that drifts — the first cut spelled seven of the
    nine and the shoes silently kept Illusion's maps while every count said the
    character was fully reskinned.
    """
    hit = _CATS.get("v")
    if hit is None:
        names = ()
        try:
            from ..origin.figure.outfit import CLOTHES_SLOT_NAMES as names
        except Exception as exc:                             # noqa: BLE001
            swallowed.note("mesh.parttex._garment_cats: from ..origin.figure.outfit import CLOTHES_SLO", exc)
            names = ("top", "bottom", "bra", "panty", "gloves", "legwear",
                     "socks", "shoes_inner", "shoes_outer")
        hit = _CATS["v"] = frozenset(tuple(names) + COMBO_CATS)
    return hit


_CATS: dict = {}

#: Maps we take over, per family.  A name mapped to ``None`` is **unbound**,
#: which the shaders' own property tables define as no effect — that is a
#: different statement from "we have no opinion", and the two have to be
#: distinguishable or an un-generated slot silently keeps Illusion's pixels.
_GARMENT_SLOTS = ("texture", "mask", "detail", "normal", "line_mask")
_HAIR_SLOTS = ("texture", "mask", "detail", "normal")


def _family(shader) -> int:
    """Which of Origin's shader families this renderer draws with, or 0.

    The gate on every generator below, because a *category* is the slot an item
    fills and an item can hold renderers of several kinds: the chest badge
    rides inside the garment, and a hair prefab carries its ribbons and clips as
    ``main_item``.  Those run no create pass we transcribed — the badge's map is
    the card's emblem straight through, an accessory takes ``_Color`` from
    ``ChangeAccessoryColor`` — so handing either a generated sheet replaces a
    drawing with an invention (``CLAUDE.md`` 一之一).  The first cut gated on
    the category alone and drew a blank white rectangle on the shirt.
    """
    sh = shader or {}
    if str(sh.get("_clothRole") or "").startswith("emblem"):
        return 0
    try:
        from ..origin.figure import toon as TOON
    except Exception as exc:                                 # noqa: BLE001
        swallowed.note("mesh.parttex._family: from ..origin.figure import toon as TOON", exc)
        return 0
    return int(TOON.family(sh))


def _shape_for(piece, slot: str, default: int = DEFAULT_SIZE):
    """Match the stock map's size when there is one, else a default.

    Matching costs nothing and keeps a re-baked library comparable map for map;
    the default is what a build with no Illusion textures at all falls back to.
    """
    arr = getattr(piece, slot, None)
    if arr is not None and np.ndim(arr) >= 2:
        return (int(arr.shape[0]), int(arr.shape[1]))
    return (default, default)


#: ``{category: [renderers answered, renderers left to Illusion]}``.  "A
#: character drawn with our own textures" is a claim with a number behind it,
#: and without the number it is easy to render a whole figure and not notice
#: that its clothes are still somebody else's (§7.143 十二).
STATS: dict = {}


#: ``{category: [albedos we replaced, of those how many had a cut-out, worst
#: fraction lost, that piece's name]}``.  A flat sheet is written at alpha 255
#: because no measurement says where a lace hole is (:func:`garment_main`), so
#: every piece whose stock albedo *did* cut something is drawn solid — and that
#: is reported here rather than assumed away (``CLAUDE.md`` 一之一).  §7.350
#: wired it: :func:`lost_cutout` had carried that docstring with no caller.
CUTOUT: dict = {}


def reset_stats() -> None:
    STATS.clear()
    CUTOUT.clear()


def _note_cutout(cat: str, piece) -> None:
    """Tally what replacing *piece*'s albedo with a flat sheet threw away."""
    row = CUTOUT.setdefault(cat or "?", [0, 0, 0.0, ""])
    row[0] += 1
    lost = lost_cutout(piece)
    if lost > 0.0:
        row[1] += 1
        if lost > row[2]:
            row[2] = lost
            row[3] = str(getattr(piece, "name", "") or "")


def _count(cat: str, ours: bool, partial: bool = False) -> None:
    """Tally a renderer.  *partial* = its maps are ours but its albedo is not.

    Three states, not two: a garment whose ``_MainTex`` carries a design keeps
    that sheet while everything else about it is generated, and folding that
    into either "ours" or "theirs" is what would let a count of 16/19 stand for
    a character still wearing somebody's floral print (§7.155).
    """
    row = STATS.setdefault(cat or "?", [0, 0, 0])
    while len(row) < 3:
        row.append(0)
    if not ours:
        row[1] += 1
    elif partial:
        row[2] += 1
    else:
        row[0] += 1


def piece_maps(category: str, piece) -> dict | None:
    """``{slot: (pixels | None, srgb)}`` for one renderer, or ``None``.

    ``None`` means "not ours to answer" — the caller keeps whatever it had, and
    a category we do not draw is reported rather than replaced by something
    invented (``CLAUDE.md`` 一之一).
    """
    cat = str(category or "")
    mesh = getattr(piece, "mesh", None)
    if mesh is None or getattr(mesh, "uvs", None) is None or \
            len(getattr(mesh, "uvs", ())) == 0:
        _count(cat, False)
        return None
    # Both halves of the gate: the slot the item fills **and** the family the
    # renderer draws with.  Neither alone is enough — see :func:`_family`.
    from ..origin.figure import toon as TOON
    fam = _family(getattr(piece, "shader", None))
    tex = getattr(piece, "texture", None)
    if cat in _garment_cats() and fam == TOON.FAMILY_CLOTH:
        out = {
            "mask": (garment_mask(mesh, _shape_for(piece, "mask"),
                                  getattr(piece, "mask", None)), True),
            # a weight, authored and declared linear
            "detail": (garment_detail(mesh, _shape_for(piece, "detail"),
                                      getattr(piece, "detail", None)), False),
        }
        # 255 and 0 decode the same in either space, so the albedo's flag is
        # free; sRGB matches the family the composite expects.
        main = garment_main(_shape_for(piece, "texture"), tex, mesh)
        if main is not None:
            out["texture"] = (main, True)
            _note_cutout(cat, piece)
        slots = _GARMENT_SLOTS
    elif cat in HAIR_CATS and fam == TOON.FAMILY_HAIR:
        out = {
            "mask": (hair_mask(mesh, _shape_for(piece, "mask"),
                               getattr(piece, "mask", None)), False),
            "detail": (hair_detail(mesh, _shape_for(piece, "detail"),
                                   getattr(piece, "detail", None)), False),
        }
        # 82% of the stock hair renderers bind no albedo at all and take their
        # colour wholly from the card, so unbinding matches the majority — but
        # the 18% that do bind one carry the strand shading drawn along it, and
        # that is a drawing.  Same test as the garment's.
        if tex is None:
            out["texture"] = (None, True)
        else:
            stats = albedo_stats(tex)
            if stats is not None and stats[1] < GARMENT_DRAWN_SPREAD:
                out["texture"] = (_rgba(*[int(c) for c in stats[0]], 255,
                                        _shape_for(piece, "texture")), True)
        slots = _HAIR_SLOTS
    else:
        _count(cat, False)
        return None
    # Slots this module owns and did not fill are **unbound**, which the
    # shaders' property tables define as no effect.  ``texture`` is not on that
    # list when the stock one is a drawing: absent from *out* means "keep it".
    _count(cat, True, partial="texture" not in out and tex is not None)
    for slot in slots:
        if slot == "texture" and "texture" not in out:
            continue
        out.setdefault(slot, (None, False))
    # A generator that could not answer (no UV, no islands) must not be read as
    # "unbind": that would drop a garment's colour mask and paint it white.
    for slot, (px, srgb) in list(out.items()):
        if px is None and slot in ("mask", "detail") and \
                getattr(piece, slot, None) is not None:
            del out[slot]
    return out


def reskin(piece, category: str):
    """*piece* wearing our own maps.  Returns the piece unchanged if not ours.

    One call site rather than one per consumer: the albedo composite, the
    packed shading mask and the raw ``_DetailMask`` all read the piece, so
    swapping it once is what keeps them from drifting apart (the same reason
    :func:`origin.bake.set_texture_provider` sits where it does).
    """
    got = piece_maps(category, piece)
    if not got:
        return piece
    from ..origin import colorspace as CS
    cs = dict(getattr(piece, "map_cs", None) or {})
    fields = {}
    for slot, (px, srgb) in got.items():
        fields[slot] = px
        if px is None:
            cs.pop(slot, None)
        else:
            cs[slot] = CS.SRGB if srgb else CS.LINEAR
    fields["map_cs"] = cs
    name = str(getattr(getattr(piece, "mesh", None), "name", "") or "piece")
    for slot, px in fields.items():
        if slot != "map_cs" and px is not None:
            _dump(category, name, slot, px)
    return piece._replace(**fields)


def skin_sheets(piece=None, *, albedo_size: int = SKIN_ALBEDO_SIZE,
                mask_size: int = SKIN_MASK_SIZE):
    """``(albedo, colour mask)`` for a body, both as ``colorspace.Map``.

    *piece* is ``o_body_a`` — the one the nails are read off.  Without it the
    mask is R everywhere, which loses the nail colour and nothing else.
    """
    from ..origin import colorspace as CS
    mesh = getattr(piece, "mesh", None) if piece is not None else None
    mask = skin_mask(mesh, getattr(piece, "skin", None),
                     list(getattr(piece, "bones", None) or []), mask_size)
    # The albedo is white, so its flag cannot change a pixel; the mask is
    # binary for the same reason.  Declared to match the stock rows anyway.
    return CS.Map(skin_albedo(albedo_size), True), CS.Map(mask, False)


_MODE: dict = {}


def mode() -> str:
    """``startup.own_part_tex`` — ``auto`` (default) / ``always`` / ``never``.

    Same three states and the same reasoning as ``own_face_tex`` (§7.143 十三):
    ``auto`` uses the generated maps only when the bake has no garment items to
    offer, which is the build ``PROGRESS.md`` F0 exists for, and ``always`` is
    how to look at them on a machine that does have the library.  Which art a
    build ships is a product decision, not one this module makes.

    ``ISLANDLIFE_OWN_PART_TEX`` overrides the setting, so the regression tools
    can be pointed at either path without editing a config file — a check that
    only ever runs against the default is a check that never sees the new code.
    """
    hit = _MODE.get("v")
    if hit is None:
        import os
        hit = os.environ.get("ISLANDLIFE_OWN_PART_TEX")
        if not hit:
            try:
                from ..core.paths import Config
                hit = str(Config("startup").get("own_part_tex", "auto")
                          or "auto")
            except Exception as exc:                         # noqa: BLE001
                swallowed.note("mesh.parttex.mode: from ..core.paths import Config", exc)
                hit = "auto"
        _MODE["v"] = str(hit)
    return _MODE["v"]


def enabled() -> bool:
    """Whether :func:`reskin` should be applied on this build."""
    m = mode()
    if m == "never":
        return False
    if m == "always":
        return True
    try:                      # auto: only when the bake cannot dress anybody
        from ..origin import bake
        items = bake.manifest()["items"]
        return not any(k.split("/", 1)[0] in _garment_cats()
                   for k in items)
    except Exception as exc:                                 # noqa: BLE001
        swallowed.note("mesh.parttex.enabled: from ..origin import bake", exc)
        return True


def set_mode(value: str | None) -> None:
    """Override the setting for one process (tools, tests)."""
    if value is None:
        _MODE.pop("v", None)
    else:
        _MODE["v"] = str(value)


# ---------------------------------------------------------------------------
# texture-only lists: pattern, hairgloss, nose, underhair
#
# These have no mesh — the PNG *is* the item — so they arrive through
# ``bake.read_texture`` rather than off a piece.  They are what a dressed
# character still asked Illusion for after §7.144, which
# ``tools/own_tex_preview.py`` counts.
# ---------------------------------------------------------------------------
#: What each stock ``mt_pattern`` row **is**, measured rather than guessed:
#: ``(kind, period_u, period_v, duty)``, periods as fractions of the sheet,
#: from ``tools/face_tex_profile.py --tiles pattern``.  ``create_topN`` reads
#: only ``.r`` — as the weight between a slot's two colours — so one channel is
#: the whole content of these sheets.
#:
#: A table rather than a rule like "id mod 5", because the *kind* is what makes
#: a plaid skirt a plaid: 17 of the 42 are grids, 11 dot fields, 8 stripes
#: (four upright each way), 3 checks, 2 diamond checks and 1 diagonal stripe.
#: Which id is which is a fact about the list, like the ramp id or a map's
#: colour space, and 42 (kind, period, duty) triples are the measurement —
#: not the drawing (``著作權.md`` 四).
PATTERN_TABLE = {
    1: ("stripe_u", 0.2500, 0.2500, 0.500),
    2: ("stripe_v", 0.2500, 0.2500, 0.500),
    3: ("grid", 0.2812, 0.2812, 0.803),
    4: ("grid", 0.2500, 0.2500, 0.926),
    5: ("grid", 0.4922, 0.4922, 0.862),
    6: ("dots", 0.3281, 0.3203, 0.497),
    7: ("grid", 0.5000, 0.5391, 0.881),
    8: ("grid", 0.4922, 0.4922, 0.830),
    9: ("grid", 0.8047, 0.7734, 0.722),
    10: ("check", 0.2500, 0.2500, 0.500),
    11: ("grid", 0.1250, 0.1250, 0.781),
    12: ("dots", 0.5000, 0.5000, 0.393),
    13: ("check", 0.5000, 0.2422, 0.426),
    14: ("dots", 0.7344, 0.5078, 0.494),
    15: ("grid", 0.2500, 0.2500, 0.820),
    16: ("grid", 0.2500, 0.2500, 0.930),
    17: ("diag_check", 0.4766, 0.4766, 0.500),
    18: ("dots", 0.4922, 0.2500, 0.629),
    19: ("dots", 0.0469, 0.2500, 0.608),
    20: ("check", 0.7109, 0.1172, 0.682),
    21: ("stripe_u", 0.2500, 0.5000, 0.523),
    22: ("dots", 0.3203, 0.8281, 0.674),
    23: ("dots", 0.2578, 0.2656, 0.616),
    24: ("dots", 0.5000, 0.9141, 0.500),
    25: ("grid", 0.4844, 0.5391, 0.856),
    26: ("grid", 0.5000, 0.5078, 0.797),
    27: ("stripe_d", 0.5000, 0.5000, 0.740),
    28: ("stripe_v", 0.8438, 0.8438, 0.508),
    29: ("grid", 0.5000, 0.4961, 0.854),
    30: ("stripe_u", 0.2500, 0.2500, 0.969),
    31: ("stripe_v", 0.2500, 0.2500, 0.969),
    32: ("dots", 0.2500, 0.6250, 0.531),
    33: ("grid", 0.8984, 0.8555, 0.725),
    34: ("grid", 0.0664, 0.5000, 0.789),
    35: ("dots", 0.5000, 0.5000, 0.250),
    36: ("stripe_u", 0.5000, 0.5000, 0.570),
    37: ("stripe_v", 0.5000, 0.5000, 0.570),
    38: ("grid", 0.9180, 0.9180, 0.780),
    39: ("grid", 0.4297, 0.4766, 0.843),
    200: ("dots", 0.4922, 0.4922, 0.457),
    201: ("diag_check", 0.4766, 0.4531, 0.494),
    202: ("grid", 0.5000, 0.0469, 1.000),
}
PATTERN_SIZE = 128
#: A row the table has no entry for — a zipmod's own pattern.  Drawn as the
#: commonest kind at the median period and duty rather than refused, because
#: ``create_topN`` treats an unbound pattern as "no pattern at all" and a
#: garment that asked for one would come out flat.
PATTERN_DEFAULT = ("grid", 0.5, 0.5, 0.8)

#: ``_HairGloss``: every stock sheet is black but for one band across the
#: middle — the strand highlight ``main_hair`` walks with the view angle.
#: Medians over the 8 rows (``--band hairgloss``).
GLOSS_PEAK_V = 0.505
GLOSS_HALF_W = 0.060
GLOSS_PEAK = 112.0
GLOSS_COVER = 0.048
#: Along the band the autocorrelation is **weak** (0.14–0.41), so these are not
#: a period: they are counts of lit runs (12–35 per sheet, median run 0.043 of
#: the width).  The generator places them with a fixed integer sequence, so the
#: same row draws the same sheet in every process.
GLOSS_SEGMENTS = 15
GLOSS_SEG_W = 0.043

#: ``nose``: a black stroke on the face sheet's midline.  ``(v_top, v_bottom,
#: width)`` per stock row, all as fractions of the sheet.  The three base rows
#: are the nose-size options — same bottom at 0.842, starting higher the longer
#: the row — and every one of them peaks at **alpha 255**, which is what the
#: generator draws with: taking the *median* alpha for the peak (17 on row 0)
#: put the whole stroke under any ink threshold and measured zero coverage.
#: Row 3 is a different mark: 0.133 wide and 0.006 tall, i.e. horizontal.
#:
#: The fourth number is the row's **ink coverage** (alpha > 90), and it is what
#: the taper is solved against: a stroke of the measured width and length drawn
#: at full strength covers 1.6× what row 1 does, because the stock line fades
#: towards the top (its median ink alpha is 159 against a peak of 255).
NOSE_TABLE = {0: (0.768, 0.842, 0.0137, 0.00077),
              1: (0.629, 0.842, 0.0156, 0.00190),
              2: (0.516, 0.842, 0.0156, 0.00264),
              3: (0.887, 0.893, 0.1328, 0.00020),
              200: (0.797, 0.910, 0.0254, 0.00167),
              201: (0.688, 0.900, 0.0156, 0.00190)}
NOSE_DEFAULT = (0.700, 0.842, 0.0145, 0.00150)
NOSE_U = 0.499

#: ``underhair``: coverage runs 0.005–0.167 over the 13 stock rows, ink centred
#: near (0.5, 0.45).  The median is used for a row not in the table.
UNDERHAIR_COVER = {1: 0.070, 2: 0.069, 3: 0.016, 4: 0.021, 5: 0.005,
                   6: 0.072, 7: 0.167, 8: 0.123}
UNDERHAIR_DEFAULT = 0.055


def _grid_uv(w: int, h: int):
    ys, xs = np.mgrid[0:h, 0:w]
    return (xs + 0.5) / w, (ys + 0.5) / h


def _wave(t: np.ndarray, duty: float) -> np.ndarray:
    """A 0/1 square wave of *duty* over each unit of *t*."""
    return ((t - np.floor(t)) < duty).astype(np.float64)


def _solve_param(measure, target: float, lo: float, hi: float,
                 iters: int = 12) -> float:
    """The parameter whose *measure* comes out at *target*, by bisection.

    Every sheet below has one free number — a line width, a disc radius, a
    taper — and one measurement it has to reproduce.  Solving beats deriving:
    the analytic form is only right when the sheet holds a whole number of
    periods and the edges are not antialiased, and neither holds in general.
    Deriving the grid's gap from ``1 - sqrt(duty)`` drew id 38 at 0.660 against
    a measured 0.780 for exactly that reason.

    Assumes *measure* is monotonic in the parameter and reads its direction off
    the two ends, so a generator does not have to declare which way it runs.
    """
    f_lo, f_hi = float(measure(lo)), float(measure(hi))
    rising = f_hi >= f_lo
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        got = float(measure(mid))
        if (got < target) == rising:
            lo = mid
        else:
            hi = mid
    # These measures are **step** functions — a stripe's on-fraction moves in
    # whole texels — so the bracket's midpoint can sit on the wrong tread even
    # after it is narrow.  Taking the better of the three endpoints costs two
    # evaluations and stops id 30 coming out at 1.000 against a measured 0.969.
    return min((lo, 0.5 * (lo + hi), hi),
               key=lambda x: abs(float(measure(x)) - target))


def pattern(ident=0, size: int = PATTERN_SIZE) -> np.ndarray:
    """One ``mt_pattern`` row, drawn from its measured kind, period and duty.

    Grey and opaque like the stock sheets: the shader reads ``.r`` and nothing
    else, and every stock row has alpha 255.
    """
    try:
        key = int(ident)
    except (TypeError, ValueError):
        key = -1
    kind, pu, pv, duty = PATTERN_TABLE.get(key, PATTERN_DEFAULT)
    u, v = _grid_uv(size, size)
    pu = max(float(pu), 2.0 / size)
    pv = max(float(pv), 2.0 / size)
    duty = float(np.clip(duty, 0.0, 1.0))
    diag = 1.41421356

    if kind in ("stripe_u", "stripe_v", "stripe_d"):
        t = (u / pu if kind == "stripe_u" else
             v / pv if kind == "stripe_v" else (u + v) / (pu * diag))
        # Solved like the others: with fewer than two periods across the sheet
        # (id 28's is 0.84 of it) the realised on-fraction is not the duty.
        d = _solve_param(lambda x: _wave(t, x).mean(), duty, 0.0, 1.0)
        on = _wave(t, d)
    elif kind == "check":
        on = ((np.floor(u / pu) + np.floor(v / pv)) % 2.0 > 0.5).astype(float)
        if duty < 0.45:                 # a check whose dark squares are bigger
            on = on * _wave(u / pu, min(1.0, duty * 2.0))
    elif kind == "diag_check":
        on = ((np.floor((u + v) / (pu * diag))
               + np.floor((u - v) / (pv * diag))) % 2.0 > 0.5).astype(float)
    elif kind == "dots":
        # Discs on a lattice; over half duty it is the gaps that are round.
        fu = (u / pu) % 1.0 - 0.5
        fv = (v / pv) % 1.0 - 0.5
        d = np.hypot(fu, fv)
        flip = duty > 0.5

        def discs(r):
            got = (d < r).astype(np.float64)
            return 1.0 - got if flip else got

        r = _solve_param(lambda x: discs(x).mean(), duty, 0.0, 0.75)
        on = discs(r)
    else:
        # Grid: thin lines both ways.  The gap is solved rather than taken as
        # ``1 - sqrt(duty)`` — that closed form assumes the sheet holds a whole
        # number of periods, and the long-period rows do not, which drew id 38
        # at 0.660 against a measured 0.780.
        def lines(line):
            return 1.0 - np.maximum(_wave(u / pu, line), _wave(v / pv, line))

        line = _solve_param(lambda x: lines(x).mean(), duty, 0.99, 0.0)
        on = lines(line)
    val = np.clip(on * 255.0, 0, 255).astype(np.uint8)
    return _rgba(val, val, val, 255, (size, size))


def _segments(count: int, width: float, seed: int) -> list:
    """Deterministic segment centres in 0..1 — no ``random``, no ``hash``.

    A fixed integer recurrence rather than a PRNG, so the same row draws the
    same sheet in every process (``CLAUDE.md`` 七 on salted string hashes) and
    two builds differ by nothing instead of by noise.
    """
    out = []
    x = (int(seed) * 2654435761 + 1013904223) & 0xFFFFFFFF
    n = max(int(count), 1)
    for i in range(n):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        jitter = ((x >> 8) & 0xFFFF) / 65535.0 - 0.5
        out.append(((i + 0.5) / n + jitter * width) % 1.0)
    return out


def hairgloss(ident=0, size: int = 1024) -> np.ndarray:
    """``_HairGloss`` — one broken band across the middle of a black sheet."""
    u, v = _grid_uv(size, size)
    # ``GLOSS_HALF_W`` is the full width at half maximum, so the Gaussian's
    # sigma is it over ``2·sqrt(2 ln 2)`` — 0.6 of it, which the first cut used,
    # draws a band 0.084 wide against the measured 0.060.
    band = np.exp(-0.5 * ((v - GLOSS_PEAK_V) / (GLOSS_HALF_W * 0.4247)) ** 2)
    try:
        seed = int(ident)
    except (TypeError, ValueError):
        seed = 0
    along = np.zeros_like(u)
    half = GLOSS_SEG_W * 0.5
    for c in _segments(GLOSS_SEGMENTS, GLOSS_SEG_W, seed):
        d = np.abs(((u - c + 0.5) % 1.0) - 0.5)
        along = np.maximum(along, np.clip(1.0 - (d / max(half, 1e-6)) ** 2,
                                          0.0, 1.0))
    val = np.clip(GLOSS_PEAK * band * along, 0, 255).astype(np.uint8)
    return _rgba(val, val, val, 255, (size, size))


def nose(ident=0, size: int = 512) -> np.ndarray:
    """The nose line: a black stroke on the face sheet's midline.

    Vertical unless the row is wider than it is tall, which is how row 3 — a
    short horizontal mark — comes out of the same two numbers rather than out
    of a special case.
    """
    from . import facetex as FT
    try:
        key = int(ident)
    except (TypeError, ValueError):
        key = -1
    v0, v1, width, cover = NOSE_TABLE.get(key, NOSE_DEFAULT)
    span = abs(float(v1) - float(v0))
    if span >= float(width):
        pts = np.stack([np.full(32, NOSE_U),
                        np.linspace(float(v0), float(v1), 32)], axis=-1)
        w = float(width)
    else:                              # wider than tall: lay it across instead
        mid = 0.5 * (float(v0) + float(v1))
        half = 0.5 * float(width)
        pts = np.stack([np.linspace(NOSE_U - half, NOSE_U + half, 32),
                        np.full(32, mid)], axis=-1)
        w = max(span, 2.0 / size)

    def draw(k):
        return FT.stroke(size, size, pts, w, taper=(0.30, 0.10), feather=1.4,
                         profile=lambda t: t ** k)

    k = _solve_param(lambda x: float(np.mean(draw(x) > 0.353)), float(cover),
                     0.02, 6.0)
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[:, :, 3] = _u8(draw(k))
    return out


def underhair(ident=0, size: int = 512) -> np.ndarray:
    """A soft patch, sized by the row's measured coverage.

    White with the shape in alpha, which is the stencil form the stock rows
    use: ``create_body`` multiplies the layer by the card's own colour.
    """
    try:
        key = int(ident)
    except (TypeError, ValueError):
        key = -1
    cover = float(UNDERHAIR_COVER.get(key, UNDERHAIR_DEFAULT))
    u, v = _grid_uv(size, size)

    def draw(scale):
        x = (u - 0.5) / max(scale * 1.35, 1e-6)
        y = (v - 0.42) / max(scale * 1.85, 1e-6)
        taper = 1.0 + 0.9 * np.clip(y, 0.0, 1.0)
        return np.clip((1.0 - np.hypot(x * taper, y)) * 3.0, 0.0, 1.0)

    # Solved, not derived: the taper makes the patch bigger than ``sqrt(A/π)``
    # says, which drew every row at 1.72× its measured coverage.
    s = _solve_param(lambda x: float(np.mean(draw(x) > 0.353)), cover,
                     0.005, 0.6)
    return _rgba(255, 255, 255, _u8(draw(s)), (size, size))


# ---------------------------------------------------------------------------
# the head's two base maps
# ---------------------------------------------------------------------------
#: ``cf_face_00_t`` is **white plus one solid rectangle**, and that is the whole
#: sheet: measured over all 1024², the non-white texels are 0.0602 of it, their
#: bounding box is below, and the box's **fill ratio is 1.0000**.  Inside it the
#: colour is a flat dark red — the mouth interior — and ``cf_face_00_mc``'s blue
#: channel is the *same* rectangle (IoU 0.914), which is what stops the block
#: being dyed with a skin tone: ``create_head`` ends on ``col = max(b, col)``,
#: so blue 1 pins the colour at white and the albedo's own pixels come through.
#:
#: The first cut expected teeth and a tongue drawn in there and measured each
#: head mesh's footprint separately looking for them — every one of the ten
#: reported the same ``[89, 33, 33]``.  There is nothing to draw: it is a block.
FACE_BLOCK_T = (0.3232, 0.0049, 0.6758, 0.1758)     # u0, v0, u1, v1
FACE_BLOCK_MC = (0.3271, 0.0000, 0.6699, 0.1699)
FACE_BLOCK_RGB = (87, 33, 33)
FACE_SIZE = 1024


def _rect(u0: float, v0: float, u1: float, v1: float, shape) -> np.ndarray:
    u, v = _grid_uv(shape[1], shape[0])
    return (u >= u0) & (u < u1) & (v >= v0) & (v < v1)


def face_albedo(size: int = FACE_SIZE) -> np.ndarray:
    """``cf_face_00_t`` — white, with the mouth block."""
    box = _rect(*FACE_BLOCK_T, (size, size))
    out = np.full((size, size, 4), 255, dtype=np.uint8)
    for c in range(3):
        out[:, :, c] = np.where(box, FACE_BLOCK_RGB[c], 255)
    return out


def face_mask(size: int = FACE_SIZE) -> np.ndarray:
    """``cf_face_00_mc`` — R is the skin, B is the block, G is not drawn.

    ``create_head`` reads ``col = (1 + r*(c1-1)) * (1 + g*(c2-1))`` then
    ``max(b, col)``, so R and G are two independent tints and B is a floor.  G
    measures **0.31% on** (5.1% part-way) over the face's own footprint and is
    left at 0 for the same reason the body's is: the stock sheet barely uses
    it, and the honest generated value for something that was not drawn is the
    one that does nothing.
    """
    box = _rect(*FACE_BLOCK_MC, (size, size))
    return _rgba(np.where(box, 0, 255).astype(np.uint8), 0,
                 np.where(box, 255, 0).astype(np.uint8), 255, (size, size))


def face_sheets(*, size: int = FACE_SIZE):
    """``(albedo, colour mask)`` for the head, both as ``colorspace.Map``."""
    from ..origin import colorspace as CS
    return CS.Map(face_albedo(size), True), CS.Map(face_mask(size), False)


#: The texture-only lists this module answers, and the size it draws each at.
SHEETS = {
    "pattern": (pattern, PATTERN_SIZE),
    "hairgloss": (hairgloss, 1024),
    "nose": (nose, 512),
    "underhair": (underhair, 512),
}

#: Categories deliberately **not** answered, with the reason.  Data rather than
#: an omission, so the gap is visible in one place:
#:
#: * ``emblem`` / ``body_paint`` / ``face_paint`` — a crest and a tattoo are
#:   *designs*.  No measurement turns into somebody's insignia.
#: * ``sunburn`` — the tan line is a swimsuit silhouette, also a design, and
#:   not answering is already the neutral: ``create_body`` reads
#:   ``t = max(Sunburn.r, 1 - _Color4.a)``, so an absent map leaves the skin at
#:   ``_Color``, untanned.  Returning a white sheet would say the same thing
#:   while pretending to have drawn it.
NOT_DRAWN = {
    "emblem": "a crest is a design, not a measurement",
    "body_paint": "a tattoo is a design",
    "face_paint": "a tattoo is a design",
    "sunburn": "a tan line is a swimsuit silhouette; absent already means none",
}

#: ``{category: [rows answered, rows left to Illusion]}`` for the sheet half.
SHEET_STATS: dict = {}


def provide_sheet(category: str, ident):
    """A generated texture-only sheet as a ``colorspace.Map``, or ``None``.

    Chained after :func:`polylode.mesh.facetex.provide` on the single
    ``bake.set_texture_provider`` seam, and gated by the same switch as the
    rest of this module (:func:`enabled`).

    Colour space is the **row's own flag** where the bake still has one, with
    the category majority only as the fallback — §7.143 十's lesson, and it
    applies here too: ``pattern`` splits 35 sRGB / 7 linear.

    A row the list calls 「なし」 is declined, the same way and from the same
    fact as :func:`facetex.draws_nothing`.  All four of these categories have
    one and **two of them are not id 0** — ``hairgloss`` calls **5** none and
    ``nose`` calls **4** — so this cannot be a local "reject 0" rule; the id
    comes from ``lists.draws_nothing``, which reads the installed lists, falls
    back to ``lists.NONE_IDS`` when there are none to read, and also covers the
    second way Illusion writes なし (a row naming an *empty* asset).  None of
    these four use that second form today; asking the shared predicate rather
    than one half of it is so that they do not have to be revisited when one
    does.
    """
    cat = str(category or "")
    row = SHEET_STATS.setdefault(cat, [0, 0])
    hit = SHEETS.get(cat)
    if hit is None:
        row[1] += 1
        return None
    from ..origin import lists as L
    if L.draws_nothing(cat, ident):
        row[1] += 1
        return None
    from ..origin import colorspace as CS
    srgb = cat != "underhair"        # measured: 13 of 13 underhair rows linear
    try:
        from ..origin import bake
        entry = bake.manifest()["items"].get(bake.key_of(cat, int(ident)))
        if entry and "cs" in entry:
            srgb = int(entry["cs"]) == CS.SRGB
    except Exception as exc:                                 # noqa: BLE001
        swallowed.note("mesh.parttex.provide_sheet: bake.manifest", exc)
    row[0] += 1
    draw, size = hit
    return CS.Map(draw(ident, size), srgb)


# ---------------------------------------------------------------------------
# what a generated set costs
# ---------------------------------------------------------------------------
def lost_cutout(piece) -> float:
    """How much of this piece's albedo alpha a white sheet throws away.

    The fraction of the covered texels whose stock alpha is under 0.5 — 0 for
    most garments, 0.79 for a ribbon bra.  Reported rather than assumed away:
    it is the one thing on this path that a measurement cannot give back.
    """
    tex = getattr(piece, "texture", None)
    mesh = getattr(piece, "mesh", None)
    if tex is None or mesh is None or np.ndim(tex) != 3 or tex.shape[2] < 4:
        return 0.0
    # Most stock albedos are opaque everywhere; asking that first keeps the
    # rasterisation below off the dress path for every piece that has nothing
    # to lose (F0-6 (g): this path is already the slow one).
    if not bool((np.asarray(tex)[..., 3] < 128).any()):
        return 0.0
    cov, _ = footprint(mesh, tex.shape[:2])
    if cov is None or not cov.any():
        return 0.0
    return float(np.mean(tex[..., 3][cov] < 128))
