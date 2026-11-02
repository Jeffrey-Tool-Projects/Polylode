"""Procedural face and skin sheets — our own pixels, drawn from measurements.

``PROGRESS.md`` F0-3.  Everything here is painted from parameters; nothing is
resampled, traced or filtered from anyone else's image.  The parameters come
from :mod:`tools.face_tex_profile`, which reports *where* the ink is, *how
wide* it is and *how much* of the sheet it covers — facts about a drawing, not
the drawing (``著作權.md`` 四).

The measurement that made this tractable: **most of these sheets are stencils.**
``eyebrow`` is a constant RGB 223 with the whole shape in alpha; ``lipline``
and ``mole`` are constant 255; ``lip`` / ``cheek`` / ``eyeshadow`` are constant
white washes.  The colour arrives at draw time from the card, so there is no
painting to reproduce — only a shape to draw.  Only ``eyeline_up`` and the iris
carry real RGB variation.

Targets each generator is written against (Origin's stock sheets, measured):

============ ========= ================================ ========= ==========
sheet        size      ink bbox (u0,v0,u1,v1)           coverage  stroke
============ ========= ================================ ========= ==========
eyebrow      512²      0.121 0.465 0.789 0.559          0.0157    8 px
eyeline_up   512²      0.172 0.104 0.877 0.641          0.107     30 px
eyeline_down 512²      0.281 0.713 0.674 0.803          0.0033    4 px
lipline      128×64    0.352 0.328 0.633 0.391          0.0145    2 px
lip          512²      0.186 0.336 0.815 0.512          0.0835    36 px
mole         512²      0.494 0.494 0.510 0.510          0.0002    4 px
iris (eye)   512²      0.301 0.207 0.691 0.783          0.176     —
============ ========= ================================ ========= ==========

**Mirror IoU says how many of a thing to draw.**  ``cf_O_mayuge``'s two halves
read the *same* full window, so the sheet holds **one** brow and the mesh
mirrors it — draw two and the face gets four.  ``eyeline_up`` is the same
(one eye).  ``cheek`` and ``eyeshadow`` measure 0.97 mirror IoU and really do
carry both sides.  The eyes are a third case: ``cf_Ohitomi_L`` samples
u 0.230–0.845 while ``cf_Ohitomi_R`` samples 0.155–0.770 — *shifted* windows
of one sheet, which is why the iris is centred at 0.496 rather than per-eye.
"""
from __future__ import annotations

import numpy as np

from ..core import swallowed


# --------------------------------------------------------------------------
# stroke rasteriser
# --------------------------------------------------------------------------
def _grid(w: int, h: int):
    ys, xs = np.mgrid[0:h, 0:w]
    return (xs + 0.5) / w, (ys + 0.5) / h


def _polyline_field(w: int, h: int, pts: np.ndarray):
    """(distance in u-units, parameter t along the path) for every pixel.

    Exact per-segment distance rather than a splat: a taper is a function of
    *t*, so the rasteriser has to know how far along the stroke each pixel is,
    and a splat only knows how far from a centre.
    """
    U, V = _grid(w, h)
    P = np.stack([U, V], axis=-1)
    pts = np.asarray(pts, dtype=np.float64)
    seg = pts[1:] - pts[:-1]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1]) or 1.0

    best_d = np.full(P.shape[:2], 1e9)
    best_t = np.zeros(P.shape[:2])
    for i in range(len(seg)):
        a, ab = pts[i], seg[i]
        denom = float(ab @ ab) or 1e-12
        rel = P - a
        s = np.clip((rel[..., 0] * ab[0] + rel[..., 1] * ab[1]) / denom, 0.0, 1.0)
        closest = a + s[..., None] * ab
        d = np.hypot(P[..., 0] - closest[..., 0], P[..., 1] - closest[..., 1])
        take = d < best_d
        best_d = np.where(take, d, best_d)
        best_t = np.where(take, (cum[i] + s * seg_len[i]) / total, best_t)
    return best_d, best_t


def stroke(w: int, h: int, pts, width, taper=(0.15, 0.15), feather: float = 1.6,
           profile=None):
    """Alpha for one tapered stroke.  *width* is in u-units, at the middle.

    ``taper`` is how much of each end fades to nothing.  Ends that stop dead
    are what makes procedural line work read as a decal rather than a drawn
    line, and it costs one ``smoothstep`` to avoid.

    ``profile`` is an optional ``f(t) -> multiplier`` for the width along the
    path, and it is not decoration: the symmetric taper alone draws a **bar**,
    and a brow is not a bar.  What shape it should be is a per-caller question
    and has to come from the whole category, not one sheet — ``eyebrow`` first
    used ``(1-t)**0.85`` because *one* stock row thins monotonically, while the
    17 stacked rows hold a plateau and then stop.
    """
    d, t = _polyline_field(w, h, pts)
    t0, t1 = taper
    ramp_in = np.clip(t / max(t0, 1e-6), 0.0, 1.0) if t0 > 0 else 1.0
    ramp_out = np.clip((1.0 - t) / max(t1, 1e-6), 0.0, 1.0) if t1 > 0 else 1.0
    prof = ramp_in * ramp_out
    prof = prof * prof * (3.0 - 2.0 * prof)          # smoothstep both ends
    if profile is not None:
        prof = prof * np.clip(profile(t), 0.0, 1.0)
    half = 0.5 * width * prof
    edge = feather / w
    a = np.clip((half - d) / max(edge, 1e-6) + 0.5, 0.0, 1.0)
    return a


def stroke_frame(ink: np.ndarray):
    """(s, q) over the whole sheet, in a stroke's **own** frame.

    ``s`` runs 0..1 along the ink's u extent, ``q`` runs 0..1 across the local
    stroke, 0 at its top edge.  It is defined everywhere — the centre line and
    the local height are interpolated across columns the ink misses — so the
    same frame can carry a table onto a sheet we are drawing and read one off
    a sheet Illusion drew.

    It lives here rather than in the measuring tool because both sides have to
    agree on it: a table measured in one frame and applied in another is a
    different drawing with the same numbers, which is the failure §7.148 十五
    spent a round on before the frame was named.
    """
    h, w = ink.shape
    cols = np.nonzero(ink.any(axis=0))[0]
    if len(cols) == 0:
        return np.zeros(ink.shape), np.full(ink.shape, 0.5)
    cen = np.empty(len(cols), np.float64)
    hgt = np.empty(len(cols), np.float64)
    for k, x in enumerate(cols):
        c = np.nonzero(ink[:, x])[0]
        cen[k] = (c.min() + c.max() + 1) / 2.0
        hgt[k] = max(c.max() - c.min() + 1, 1)
    xs = np.arange(w, dtype=np.float64)
    cen_f = np.interp(xs, cols, cen)
    hgt_f = np.maximum(np.interp(xs, cols, hgt), 1.0)
    u0, u1 = float(cols.min()), float(cols.max() + 1)
    s = np.clip((xs - u0) / max(u1 - u0, 1.0), 0.0, 1.0)
    ys = np.arange(h, dtype=np.float64)[:, None]
    q = np.clip(0.5 + (ys - cen_f[None, :]) / hgt_f[None, :], 0.0, 1.0)
    return np.broadcast_to(s[None, :], ink.shape).copy(), q


def _arc(u0, v0, u1, v1, bow: float, n: int = 96):
    """Points along a quadratic arc from (u0,v0) to (u1,v1), bowed by *bow*."""
    t = np.linspace(0.0, 1.0, n)
    mx, my = (u0 + u1) * 0.5, (v0 + v1) * 0.5
    dx, dy = u1 - u0, v1 - v0
    L = float(np.hypot(dx, dy)) or 1.0
    cx, cy = mx - dy / L * bow, my + dx / L * bow
    u = (1 - t) ** 2 * u0 + 2 * (1 - t) * t * cx + t ** 2 * u1
    v = (1 - t) ** 2 * v0 + 2 * (1 - t) * t * cy + t ** 2 * v1
    return np.stack([u, v], axis=-1)


def _rgba(alpha: np.ndarray, rgb) -> np.ndarray:
    h, w = alpha.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, 0] = int(rgb[0])
    out[:, :, 1] = int(rgb[1])
    out[:, :, 2] = int(rgb[2])
    out[:, :, 3] = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------
# sheets
# --------------------------------------------------------------------------
def eyebrow(size: int = 512, *, thickness: float = 0.058, arch: float = 0.12,
            tilt: float = 0.0, u0: float = 0.150, u1: float = 0.655,
            rise: float = 0.04, fall: float = 0.26, fall_pow: float = 0.65,
            rgb=(223, 223, 223)) -> np.ndarray:
    """Two brows in the band ``cf_O_mayuge`` samples (v 0.352–0.632).

    Fitted, not chosen (``tools/fit_face_tex.py``): 0.514 -> 0.631 against
    ``eyebrow/11``, which is the row the default card names through
    ``eyebrowId`` — every earlier comparison here used ``eyebrow/0``, the first
    file in the directory, which is the same mistake already made with the iris
    (``eye#0`` for a card using ``eye#10``) and the blush (``cheek/1`` for
    ``cheek/2``).  Three times is a rule: **look up the row the card names.**


    **One** brow, not a pair: both halves of ``cf_O_mayuge`` read the *same*
    full window (0.064–0.816 × 0.352–0.632), so the mesh mirrors one drawing
    and a sheet with two brows on it puts two on each side of the face.  The
    measured mirror IoU said so before the render did — 0.125 for this sheet
    against 0.97 for ``cheek``, which really does carry both.
    """
    # Both ends sit at nearly the same height and the **bow** makes the
    # arch.  Sloping the endpoints instead drew a line rising steadily to
    # the outer corner, which overlaying our alpha on Origin's inside the
    # rasterised UV island showed at once: theirs peaks in the middle and
    # comes back down, ours only went up.  Both sheets agreed on every
    # marginal profile — coverage per column and per row — because a
    # projection cannot see a curve's direction.
    # **A plateau, then a quick cut — not a steady thinning.**  The width
    # profile here used to be ``(1-t)**0.85``, fitted to ``eyebrow/11`` alone.
    # Stacked over all 17 rows the family's thickness by column is
    #
    #     0.054 0.065 0.065 0.064 0.057 0.037 0.011 0.013
    #
    # i.e. nearly constant for two thirds of its length and then gone, which a
    # monotone decay cannot draw: ours was 0.029 where Origin is 0.054 and still
    # 0.025 where Origin has stopped.  Against the stacked mask that took the IoU
    # from 0.532 to 0.729 (at the 0.2 membership level).
    def _profile(t):
        r = np.clip(t / max(rise, 1e-3), 0.0, 1.0)
        r = r * r * (3.0 - 2.0 * r)
        return r * np.clip((1.0 - t) / max(fall, 1e-3), 0.0, 1.0) ** fall_pow

    pts = _arc(u0, 0.527 + tilt * 0.15, u1, 0.533 - tilt * 0.15, -arch)
    a = stroke(size, size, pts, thickness, taper=(0.03, 0.02), profile=_profile)
    return _rgba(a, rgb)


# The eye aperture — the hole in ``cf_O_eyeline``'s UV island — measured off
# our own rasterised footprint (``mesh.uvraster``).  The lash is defined
# *relative to this*, because that is what it is: a crescent laid along the
# opening's edge.
APERTURE = (0.4751, 0.5411, 0.2578, 0.2471)     # cu, cv, ru, rv


# The upper lash line, measured off the whole category rather than modelled.
#
# ``著作權.md`` 四: a measurement is a fact, the pixels are the expression.
# These are **aggregate statistics over all 62 stock sheets**, taken in the
# stroke's own frame — position along the sweep (s) by fraction of the local
# width (q, 0 at the eye) — so no sheet's drawing survives in them and no
# sheet's resolution does either.  What they replace is an analytic basis that
# could not express the category: `along**p + across**q` fitted the family's
# median value field at a correlation of only **0.254** (§7.148 十四), and
# every further parameter sweep was arguing with the basis, not with the data.
#
# Two assumptions died when this was measured in the right frame:
#
#  * the value is **darkest against the eye** (q = 0: 27-99) and lightest at
#    the outer edge (q = 1: 127-193) — the generator had no across-stroke term
#    at all, and before that had the ramp the other way round;
#  * the darkest place along the sweep is **s ~ 0.35**, not the inner end.
#
# §7.143 二十三 reached the opposite conclusion from a whole-sheet statistic
# ("the dark centroid sits above the ink centroid"), which is confounded: the
# crescent wraps, so "up in the sheet" is the eye side at one end and the
# outer side at the other.  A stroke-local frame has no such ambiguity.
#
# Width is the **median of each sheet's own 95th-percentile extent**, not the
# extent of the family's core mask: the core is where 62 drawings agree, which
# is narrower than any of them (the same trap as fitting the median image).
LASH_W = np.array([
    0.2442, 0.3957, 0.5278, 0.5410, 0.4162, 0.3083, 0.2790, 0.2639,
    0.2503, 0.2396, 0.2713, 0.3111, 0.3208, 0.2587, 0.1156, 0.0368])
LASH_V = np.array([
    [128.0, 141.0, 166.0, 177.0, 186.0, 193.0],
    [112.5, 142.0, 159.5, 173.0, 181.0, 187.0],
    [ 82.0, 108.0, 139.5, 148.0, 163.2, 147.8],
    [ 52.5,  78.0, 102.0, 120.2, 133.0, 132.0],
    [ 38.0,  58.0,  82.5, 102.5, 122.0, 129.5],
    [ 27.0,  45.5,  63.0,  84.5, 100.5, 127.2],
    [ 37.5,  47.5,  71.0,  90.8, 107.0, 135.5],
    [ 51.5,  68.8,  87.2, 102.5, 122.0, 144.0],
    [ 60.8,  83.5, 100.5, 116.8, 137.2, 154.5],
    [ 71.0,  89.2, 110.8, 125.5, 142.2, 157.8],
    [ 75.0,  99.5, 117.0, 135.0, 154.0, 169.2],
    [ 81.8, 107.0, 129.0, 147.5, 169.0, 178.5],
    [ 88.0, 117.0, 140.0, 164.0, 177.0, 191.0],
    [ 96.0, 117.0, 137.0, 159.0, 173.5, 186.0],
    [ 97.0, 111.0, 121.2, 127.0, 136.0, 143.5],
    [ 99.5, 103.0, 112.0, 119.5, 126.0, 130.0]])


def _table2d(tab, s, q):
    """Bilinear sample of a table laid out along (s, q), both 0..1."""
    ns, nq = tab.shape
    fs = np.clip(s, 0.0, 1.0) * (ns - 1)
    fq = np.clip(q, 0.0, 1.0) * (nq - 1)
    i0 = np.clip(fs.astype(np.int32), 0, ns - 1)
    j0 = np.clip(fq.astype(np.int32), 0, nq - 1)
    i1 = np.minimum(i0 + 1, ns - 1)
    j1 = np.minimum(j0 + 1, nq - 1)
    a, b = fs - i0, fq - j0
    return ((tab[i0, j0] * (1 - a) + tab[i1, j0] * a) * (1 - b)
            + (tab[i0, j1] * (1 - a) + tab[i1, j1] * a) * b)


def eyeline_up(size: int = 512, *, spikes: int = 6, a0: float = 189.0,
               a1: float = 404.0,
               span_k: float = 1.0, width_scale: float = 1.0,
               end_taper: float = 0.30, end_taper_in: float = 0.13, split_at: float = 0.17,
               split_gap: float = 0.42, split_pow: float = 1.0,
               split_drift: float = 0.20,
               lum_lo: float = 27.5, lum_hi: float = 191.0,
               lum_gamma: float = 1.30,
               spike_len: float = 0.16, spike_w: float = 0.0072,
               spike_lum: float = 118.0, spike_out: float = 0.20,
               spike_bow: float = 0.05,
               inner_a: tuple[float, float] = (221.7, 251.9),
               inner_n: int = 0) -> np.ndarray:
    """Upper lash: a crescent along the eye opening, in Origin's cel idiom.

    **These defaults are fitted, not chosen.**  ``tools/fit_face_tex.py`` runs
    coordinate descent over all fifteen of them against the stock row the card
    names, scoring IoU inside the mesh's UV island plus the luminance
    histogram, the opaque fraction and the row/column profiles: 0.722 -> 0.831.
    Every earlier value here was set by looking at a render and guessing which
    constant was wrong, which is how the lashes ended up under the eye and the
    blush ended up a stripe.  Change them by re-running the fit, not by eye.


    Four earlier cuts fitted an arc and kept getting the *placement* closer
    while the face read as angry.  ``tools/uv_overlay.py`` showed the ink hugs
    the aperture rim, and compositing Origin's own sheet over white showed the rest
    of it — the part a histogram cannot say:

    * the edge **against the eye is hard**, the edge **away from it fades** over
      a long distance.  Ours was soft on the inside and cut off abruptly on the
      outside, which is what turns a lash into smoky eyeshadow.
    * it does not stop at the outer corner — it **sweeps down the outer side**
      of the opening and ends in a point.
    * there are a few discrete **lash spikes** past the outline, two at the
      inner corner and a flick at the outer one.  Those read as *drawn* rather
      than airbrushed and are most of what "Origin style" means here.

    So this is parameterised by the angle around the aperture rather than by u:
    every pixel gets its normalised radius and angle, ink lives just outside
    the rim, and the width varies along the sweep.
    """
    U, V = _grid(size, size)
    cu, cv, ru, rv = APERTURE
    x, y = (U - cu) / ru, (V - cv) / rv
    r = np.hypot(x, y)
    ang = np.degrees(np.arctan2(y, x)) % 360.0     # 0 outer, 90 low, 180 inner, 270 top

    # Sweep from just past the inner corner, over the top, round to the outer
    # underside.  ``s`` runs 0 -> 1 along it.
    aa = np.where(ang < 90.0, ang + 360.0, ang)
    s = np.clip((aa - a0) / (a1 - a0), 0.0, 1.0)
    on_sweep = (aa >= a0) & (aa <= a1)

    # Width of the crescent, in units of the aperture radius.
    #
    # **The peak has to be a parameter, because Origin's lash line is front-loaded.**
    # Measured off the family core (the texels at least half of the 62 sheets
    # ink), thickness against position along the sweep is
    #
    #     s     0.04  0.12  0.21  0.29  0.38  0.46  0.54  0.62  0.71  0.79  0.96
    #     Origin   0.236 0.310 0.361 0.337 0.278 0.263 0.241 0.235 0.260 0.272 0.183
    #
    # — thickest at **s ≈ 0.2**, near the inner corner, then a long shallow
    # plateau that never reaches zero.  The first basis was a symmetric
    # ``sin(pi*s)`` bump, which can only peak in the middle: it drew 0.147
    # where Origin has 0.236 and 0.396 where Origin has 0.263, and no setting of its
    # three parameters could do better than rms 0.057 because the shape it can
    # express is the wrong shape.  A skewed (Beta-like) bump whose peak sits at
    # ``peak_at`` fits the same twelve numbers at **rms 0.021**.
    w = width_scale * np.interp(np.clip(s, 0.0, 1.0),
                                np.linspace(0.0, 1.0, len(LASH_W)), LASH_W)
    # Both ends come to a point.  ``LASH_W`` is a median over 62 sheets, and a
    # median cannot end sharply: sheets start and stop at slightly different
    # angles, so the first bin averages drawn ink with blank and reports a
    # blunt 0.244.  The taper is the shape each individual sheet has.
    # Asymmetric: the outer end has the split to carry it and can afford a
    # long taper, the inner end only has its own width and went thin.
    _t = np.clip(np.minimum(s / max(end_taper_in, 1e-3),
                            (1.0 - s) / max(end_taper, 1e-3)), 0.0, 1.0)
    w = w * (_t * _t * (3.0 - 2.0 * _t))
    # The outer end tapers over 4% of the sweep, not 10%: at s = 0.96 Origin still
    # measures 0.183 of an aperture radius, so a longer taper cuts off a piece
    # of lash line that is really there.
    w = np.maximum(w, 0.0) * np.clip(np.minimum(s / 0.05, (1.0 - s) / 0.04),
                                     0.0, 1.0)

    # **The falloff belongs in the value, not in the alpha.**  Origin's sheet is
    # 93.7% fully opaque with the shading carried in RGB; the first crescent
    # put the fade in alpha instead (18.9% opaque) and read as a translucent
    # smudge tinting the skin rather than ink laid on it.  The luminance
    # histograms can match perfectly while that is wrong — they say how dark
    # the ink is, not whether it is ink.
    e = r - 1.0                                   # distance outside the rim
    # The opaque region ends exactly where the ramp reaches white.  At
    # 1.35x it overshot and a third of the sheet was a saturated plateau —
    # 47% of the ink over lum 200 against Origin's 4.5% — which drew as a tan
    # ring around the eye.
    span = w * span_k
    px = 1.5 / size
    inside_edge = np.clip((e + px) / px, 0.0, 1.0)              # hard, at the eye
    outside_edge = np.clip((span - e) * size / 2.0, 0.0, 1.0)   # 1-texel AA
    a = np.where(on_sweep, inside_edge * outside_edge, 0.0)

    # The drawn lash spikes.  Enlarging Origin's own sheet at both corners shows
    # what they are, and it is not what the first two cuts drew:
    #
    # * **nearly tangential**, not radial — long blades sweeping *along* the
    #   crescent and curving away from the eye, not spikes poking straight out
    #   of the rim;
    # * **needles** — several times longer than they are wide at the base, and
    #   tapering to a point rather than a stub;
    # * **lighter than the crescent's core**, mid-grey rather than black, so
    #   they read as separate hairs instead of as lumps on the outline;
    # * at **both** corners, and **most of the length is on the inner half**.
    #   Measured as how far the family's sparse ink (texels at least 4 of the
    #   62 sheets touch) reaches past its core, by angle round the rim:
    #
    #       deg     201  213  226  238  251  263  276  288  301  313  326  338
    #       over   0.20 0.22 0.38 0.38 0.50 0.43 0.17 0.13 0.17 0.15 0.18 0.27
    #
    #   The overhang peaks at **251 deg** and stays 0.38-0.50 across 226-263;
    #   over the top it is a third of that.  The first version put two short
    #   lashes at 204-215 and three long ones at 331-378 — the long ones were
    #   at the end where Origin's are middling, and the region Origin draws longest had
    #   nothing in it.
    #
    # They also have to be tracked separately from the crescent: its value is
    # a function of the distance outside the rim, and a blade reaches far past
    # it, so an earlier version drew every lash at lum 255 — white ink.
    # The terminals.  Enlarging Origin's own sheets at both ends shows that the
    # branches are **the lash line itself dividing**, not hairs — and not
    # extra arcs laid alongside it either, which is what the previous revision
    # drew.  The band's tail **splits**: thin white wedges open *inside* the
    # ink, starting at nothing part-way along and widening to the tip, leaving
    # slivers that share one base, curve with the eyelid because they are still
    # the same arc, and each close to a needle.
    #
    # So this is a **subtraction**, not an addition.  Anything added alongside
    # has its own base and its own outer edge; only cutting into the band gives
    # branches that grow out of it.
    # **The two ends are drawn differently**, which is why neither
    # construction alone ever looked right.  Laying whole stock islands side by
    # side (rows 10, 34, 6) rather than cropping a tip:
    #
    #   inner end  — two or three **detached** thin arcs riding *above* the
    #                band, roughly parallel to its upper edge, sweeping back
    #                and tapering to points.  They are separate strokes.
    #   outer end  — the band **itself splits**: a wedge opens inside the ink
    #                near the tip and the tail runs on as two slivers.
    #
    # Cutting both ends (the previous revision) striped the whole crescent;
    # adding blades at both ends left the outer tip with a seam that has no
    # base.  So: subtract at the outer end, add at the inner one.
    q_here = np.clip(e / np.maximum(w, 1e-6), 0.0, 1.0)
    d = np.clip((s - (1.0 - split_at)) / max(split_at, 1e-6), 0.0, 1.0)
    for k in range(1, max(2, max(0, spikes) // 3 + 1)):
        qk = k / (max(2, max(0, spikes) // 3 + 1)) - split_drift * d
        half = 0.5 * split_gap * d ** split_pow
        a = np.where(on_sweep & (d > 0.0) & (np.abs(q_here - qk) < half), 0.0, a)

    # **``spike_len`` / ``spike_bow`` / ``spike_out`` were orphans.**  They were
    # declared, documented and set by callers, and this loop used hard-wired
    # constants instead — five different values for all three drew a
    # bit-identical sheet (1254 spike texels, r_max 1.712, 191-220 deg, every
    # time).  §7.148 十七 rewrote the lashes and replaced the reads with the
    # numbers it had just fitted, which is the ``anim_speed`` shape in
    # ``CLAUDE.md`` 七: the giveaway is that a parameter has writers and no
    # reader, and only a measurement finds it.
    #
    #  * ``spike_len``  — how far round the lid a blade sweeps, as a fraction of
    #    the whole arc.  A longer lash follows the lid, it does not just stick
    #    out further; that is what makes it read as a hair rather than a spine.
    #  * ``spike_out``  — how far past the crescent's outer edge it sits.
    #  * ``spike_bow``  — how much it curves away from the eye along its length.
    #  * ``inner_a``    — the window the detached arcs live in, in **absolute**
    #    degrees.  It used to be a declared parameter nobody read, with the
    #    starts written as ``a0 + 9 - 7f`` instead — tied to where the sweep
    #    begins rather than to where the ink is, which is the shape
    #    ``CLAUDE.md`` 七 records as "位置不要寫成某段弧的百分之幾".  Measured:
    #    every one of the 62 stock rows carries a detached stroke (median 1
    #    piece, 4.7% of the row's ink) at **221.7 to 251.9 degrees**, r 1.473 to
    #    1.608.  Anchored on ``a0`` they came out at 190-225 and r 0.98-1.50 —
    #    down in the corner, below the rim, which on screen is a line running
    #    down the side of the eye rather than lashes on the lid (§7.205).
    #  * ``inner_n``    — how many, and it is **0 by default: a product answer,
    #    not a measurement**.  The family draws them (62/62 rows) and the window
    #    above is where; shipping them puts a fork with a break in it above the
    #    lid, because our own lid crease (``eyeline_shadow``) occupies the same
    #    band from 250 degrees out.  Asked which drawing to ship, the answer was
    #    one clean crease — so the converted rows get the same treatment in
    #    ``provide`` (largest component only) and this draws none.  The
    #    measurement stays here because turning them back on is this one number.
    spike_a = np.zeros_like(a)
    n_in = max(0, inner_n)
    for k in range(n_in):
        f = k / max(n_in - 1, 1)
        th0 = inner_a[0] + (inner_a[1] - inner_a[0]) * 0.18 * f
        # The inner blades are shorter than the outer ones; the spread is kept
        # so a single ``spike_len`` still gives a fan rather than a comb.
        th1 = th0 + (a1 - a0) * spike_len * (0.55 + 0.45 * f)
        t = np.linspace(0.0, 1.0, 40)
        th = np.radians(th0 + (th1 - th0) * t)
        sh = np.clip((np.degrees(th) - a0) / max(a1 - a0, 1e-6), 0.0, 1.0)
        wl = width_scale * np.interp(sh, np.linspace(0.0, 1.0, len(LASH_W)),
                                     LASH_W)
        off = wl * (1.0 + spike_out * (0.6 + 0.4 * f)) + spike_bow * t ** 1.5
        pts = list(zip(cu + ru * (1.0 + off) * np.cos(th),
                       cv + rv * (1.0 + off) * np.sin(th)))
        spike_a = np.maximum(spike_a, stroke(
            size, size, pts, spike_w * (1.7 + 0.5 * (1.0 - f)),
            taper=(0.0, 0.0), feather=1.0,
            profile=lambda z: np.clip(1.05 - np.clip(z, 0, 1) ** 0.85, 0.0, 1.0)))
    a = np.maximum(a, spike_a)

    out = np.zeros((size, size, 4), dtype=np.uint8)
    # **Densest away from the eye, lifting toward it.**  The first draft had
    # this the other way round on the anime-eye intuition that the lash line
    # against the opening is the blackest part, and the rendered lid came out
    # brown while Origin's is black — with the sheets agreeing on every luminance
    # quantile (ours p5/p50/p95 21/115/200 against a family 27/107/190) and on
    # how much of the ink is a solid core (0.241 against a family median
    # 0.224).  What differed was **where that core sits**: across all 62 stock
    # rows the dark centroid is 0.027 of the sheet *above* the ink's own
    # centroid, and ours was 0.019 below it.  So the visible edge — the one
    # over the eye — was taking the pale end of the ramp.  A distribution can
    # match on every quantile and still be drawn upside down.
    # Two terms, and the **along-arc** one is the big one.  Per-texel median
    # luminance over the 62 stock sheets is a smooth ramp running *along* the
    # crescent — darkest at the inner end, lifting steadily to the outer tip —
    # with only a mild lift across the stroke.  Ours had a strong across-stroke
    # ramp and nothing along, which is a different drawing with the same
    # histogram: every luminance quantile matched (§7.143 二十三) while the
    # 2-D residual was rms 0.337.
    lum = _table2d(LASH_V, s, e / np.maximum(w, 1e-6))
    # The blades are mid-grey: black ones read as blobs stuck on the outline.
    # Only **outside** the crescent — lightening their roots as well cut a
    # notch out of the outline where each one leaves it.
    lum = np.where((spike_a > 0.25) & (e > w), np.minimum(lum, spike_lum), lum)

    # **Pattern from the family, contrast from the family — separately.**
    # The per-texel median of 62 different lash drawings is smoother than any
    # one of them, so fitting values straight to it produced a flat mid-grey
    # band: 2-D rms fell 0.337 -> 0.151 while the dark core went 0.220 -> 0.004
    # and p95 191 -> 127, i.e. a sheet no member of the family looks like.  So
    # the terms above are fitted for *shape* only (correlation, which does not
    # see scale) and the ink is then mapped so its 5th and 95th percentiles sit
    # on the family medians.  Both rulers pass; either alone does not.
    _ink = a > 0.35
    if _ink.any():
        lo, hi = np.percentile(lum[_ink], 5), np.percentile(lum[_ink], 95)
        if hi - lo > 1e-6:
            t = np.clip((lum - lo) / (hi - lo), 0.0, 1.0) ** lum_gamma
            lum = lum_lo + t * (lum_hi - lum_lo)
        lum = np.clip(lum, 0.0, 255.0)
    out[:, :, 0] = out[:, :, 1] = out[:, :, 2] = np.clip(lum, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def eyeline_shadow(size: int = 512, *, a0: float = 222.4, a1: float = 249.3,
                   r0: float = 1.508, r1: float = 1.664,
                   lum: float = 174.0, feather: float = 0.12,
                   soft: float = 1.0) -> np.ndarray:
    """``EyelineShadowTex`` — the shadow the upper lid casts on the skin.

    A second map named by the **same** ``mt_eyeline_up`` row as the lash line,
    drawn by ``cf_O_eyeline``'s second material (``cf_m_eyeline_kage``) and
    tinted with the card's ``skinMainColor``: it is a shadow on skin, not more
    lash.  ``body_head`` has read it since §7.148 and there was never a
    generator for it, so a machine with no converted textures had **no eye
    socket line at all** — which is what the report of "the eye needs a socket
    line" is.

    Measured over all **62** stock rows — every one carries one — in the same
    aperture polar frame :func:`eyeline_up` uses.  This map has no converted
    category of its own (it is named by the lash row, so the converter never
    sees it), so the ruler reads the install directly:
    ``tools/face_tex_family.py --socket``.

    **Per row, then the median across rows** — this describes one drawing:

        coverage   0.0048  [p10 0.0032 .. p90 0.0101]
        radius     1.508 / 1.594 / 1.664   (p5 / median / p95 of the ink)
        angle      222.4 .. 249.3          (p3 / p97) — a 28-degree arc
        luminance  174     [172 .. 179]    mid-grey, not black
        peak alpha 255     on every row

    **Pooled over the family** it is r 1.381 / 1.605 / 1.839 and 216.1 to 272.7
    degrees — the spread of where 62 artists put a short arc, not an arc anyone
    drew.  The two aggregations differ by more than rounding and quoting the
    wrong one is a different claim; the defaults above are the per-row set, so
    this function draws **one row of the family**, not its envelope.

    So it is a **short arc well clear of the rim** — the lash line's body ends
    at r 1.30-1.40 across the lid — over the inner half of the upper lid.  Not
    a crease drawn against the lashes, a soft band of shade further up.

    The reference illustration this was reported from carries a much longer
    crease running the whole lid.  **The family has no measurement for that**
    (28 degrees median), so a longer line is a product choice; it is made by
    :func:`socket_placement`, which moves ``a0``/``a1``/``r0`` per card and
    keeps the clearance this family holds.  Called with the defaults this still
    draws the family's own short arc.

    The defaults were once walked off these numbers by four rounds of tuning —
    an 89-degree sweep at luminance 45 and 2.5x the family's p90 coverage — and
    every one of those rounds was arguing with a map that **could not reach the
    screen at all** (§7.168): the lid shadow is material 1 of ``cf_O_eyeline``
    and that slot was drawing a stale copy of the geometry, buried under the
    skin.  **"The picture did not change" is not evidence about a parameter
    until the map is known to be drawn.**
    """
    U, V = _grid(size, size)
    cu, cv, ru, rv = APERTURE
    x, y = (U - cu) / ru, (V - cv) / rv
    r = np.hypot(x, y)
    ang = np.degrees(np.arctan2(y, x)) % 360.0
    aa = np.where(ang < 90.0, ang + 360.0, ang)

    # Across the band: a smooth ridge between r0 and r1, so it reads as shade
    # rather than as a second drawn line.  The stock rows are fully opaque at
    # the core (peak alpha 255 on every one) with the softness at the edges.
    # **One radius, not a curve following the lash contour** (§7.183).  Letting
    # the band track the ink edge per angle is what "the socket line outlines
    # the lashes" was: it walks into the notch between every spike, and on a
    # spiky row that is a scalloped line rather than a crease.  §7.207 tried it
    # again to hold a constant clearance over the longer window and got the
    # same picture back; the clearance is held by the *choice* of radius
    # instead (:func:`socket_placement`).
    s = (aa - a0) / max(a1 - a0, 1e-6)
    mid, half = 0.5 * (r0 + r1), np.maximum(0.5 * (r1 - r0), 1e-6)
    # ``soft`` is how much of the half-width the fade takes.  1.0 is the ridge
    # the family draws — and a ridge is **98% semi-transparent**, so over skin
    # every texel lands on a different blend and the band reads as a colour
    # gradient rather than a line.  That is fine for Origin's 28-degree smudge and
    # wrong for a crease running the whole lid, which has to be one flat tone
    # (§7.183).  A small value gives a plateau with the softness only at the
    # two borders.
    t = np.clip((1.0 - np.abs(r - mid) / half) / max(soft, 1e-6), 0.0, 1.0)
    a = t * t * (3.0 - 2.0 * t)
    # And fade in and out over ``feather`` of the span, so the ends do not
    # stop dead.
    e = np.clip(np.minimum(s, 1.0 - s) / max(feather, 1e-6), 0.0, 1.0)
    a = a * (e * e * (3.0 - 2.0 * e)) * ((s >= 0.0) & (s <= 1.0))

    out = np.zeros((size, size, 4), dtype=np.uint8)
    v = int(np.clip(lum, 0, 255))
    out[:, :, 0] = out[:, :, 1] = out[:, :, 2] = v
    out[:, :, 3] = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


# The lower lash line's value field, measured over all 21 inked stock rows in
# the stroke's own frame (``tools/face_tex_family.py --stroke eyeline_down``):
# 12 steps along the sweep by 4 across it, q = 0 at the top edge — which is the
# eye side, the same way round as ``LASH_V``.
#
# It replaces a **flat 151**, and both halves of that were wrong: 151 is the
# family's 98th percentile, not its middle (median 132), so the whole stroke
# sat on the family's ceiling; and a constant has no pattern at all, where the
# 21 rows agree on two (family correlation 0.8653 against our "flat").  The
# shape is a **dip in the middle** along the sweep (108 at s ~ 0.5 against 146
# at both ends) and a monotone lift across it (108 -> 137 top to bottom).
#
# ``--value`` reported this as rms 0.1718 against a member floor of 0.0814 for
# six sections without anyone reading it, because a flat sheet loses nothing on
# any marginal: coverage, bbox, stroke width and the core IoU (0.594 against a
# member 0.286) were all in family.  A value ruler is the only one that sees a
# drawing that is the right shape in the wrong ink.
ELD_V = np.array([
    [146.0, 148.0, 149.5, 150.0],
    [138.0, 142.1, 144.0, 148.0],
    [129.0, 132.9, 138.1, 143.0],
    [118.4, 126.8, 134.9, 139.7],
    [109.5, 114.1, 126.2, 134.1],
    [110.7, 116.3, 126.0, 136.4],
    [108.4, 114.2, 130.2, 137.4],
    [111.1, 119.5, 126.2, 135.0],
    [117.2, 125.2, 133.9, 140.1],
    [129.1, 134.1, 139.7, 144.4],
    [139.5, 140.9, 144.2, 148.0],
    [144.7, 142.8, 146.5, 148.5]])


def eyeline_down(size: int = 512, *, thickness: float = 0.0175,
                 u0: float = 0.324, u1: float = 0.621,
                 v0: float = 0.764, v1: float = 0.796,
                 bow: float = 0.030, lum_scale: float = 1.0) -> np.ndarray:
    """Lower lash line — a thin arc, 0.33% of the sheet."""
    # The eyeline island is a **ring** — the eye aperture is a hole in the
    # middle of it — and this stroke was drawn just inside that hole, where the
    # mesh samples nothing: ``tools/uv_overlay.py`` reported only 6.4% of our
    # ink landing on the island and an IoU of 0.003.  It has to sit on the
    # lower rim, not on the edge of the opening.
    pts = _arc(u0, v0, u1, v1, bow)
    a = stroke(size, size, pts, thickness, taper=(0.25, 0.35))
    s, q = stroke_frame(a > 0.35)
    lum = _table2d(ELD_V, s, q) * lum_scale
    out = np.zeros((size, size, 4), dtype=np.uint8)
    for ch in range(3):
        out[:, :, ch] = np.clip(lum, 0.0, 255.0).astype(np.uint8)
    out[:, :, 3] = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def lipline(w: int = 128, h: int = 64, *, thickness: float = 0.068,
            u0: float = 0.355, u1: float = 0.630, v: float = 0.352,
            bow: float = 0.016) -> np.ndarray:
    """The mouth line: a small, near-symmetric sheet (mirror IoU 0.86–0.91)."""
    pts = _arc(u0, v, u1, v, bow)
    a = stroke(w, h, pts, thickness, taper=(0.22, 0.22))
    return _rgba(a, (255, 255, 255))


def lip(size: int = 512, *, spread: float = 0.155) -> np.ndarray:
    """The lip wash — flat white, the shape entirely in a soft alpha."""
    pts = _arc(0.200, 0.424, 0.800, 0.424, 0.012)
    a = stroke(size, size, pts, spread, taper=(0.28, 0.28), feather=26.0)
    return _rgba(a * 0.85, (255, 255, 255))


def mole(size: int = 512, *, u: float = 0.502, v: float = 0.502,
         radius: float = 0.008) -> np.ndarray:
    """A single dot; Origin's stock sheet puts it at (0.494–0.510)²."""
    U, V = _grid(size, size)
    d = np.hypot(U - u, V - v)
    a = np.clip((radius - d) * size / 1.6 + 0.5, 0.0, 1.0)
    return _rgba(a, (255, 255, 255))


def iris(size: int = 512, *, outer: float = 0.1845, aspect: float = 1.4514,
         base: float = 0.9650, lid_d: float = 0.6267, lid_s: float = 0.98,
         lid_w: float | None = None,
         floor_d: float = 0.0, floor_a: float = 0.95, pup_d: float = 0.325,
         pup_r: float = 0.50, pup_s: float = 0.35, pup_aspect: float = 1.0,
         limb_d: float = 0.85,
         ring: float = 0.20, ring_open: float = 0.48, ring_pow: float = 6.0,
         prim_r: float = 0.290, prim_w: float = 0.045, prim_d: float = 0.0,
         fib: float = 0.0, spokes: int = 84) -> np.ndarray:
    """The iris disc.

    Only ``.r`` and ``.a`` reach the shader (``create_eye`` takes the hue from
    the card's ``baseColor`` / ``subColor``), so this is a single-channel
    pattern — but it is **not** flat, and an earlier note in this file saying
    the stock sheet has no pupil was measured on **one row** and is wrong.
    Profiled across all 95 rows, normalised per sheet and taken as a median:

        island w/h        0.689  [0.643 .. 0.908]     a standing rounded oval
        vertical  (top->) 0.37 0.57 0.67 0.69 0.75 0.90 1.00 0.73
        horizontal        0.58 0.92 0.98 0.93 0.93 0.99 0.94 0.61
        radial (mid->rim) 0.62 0.74 0.91 0.98 0.95 0.53

    Three separate structures, none of which a coverage number or a bounding
    box can see: a **lid shadow** across the top (0.37 against a peak of 1.00),
    a **limbal ring** at the rim (0.53, and 0.58/0.61 at the left and right
    edges), and a **dark centre** — the pupil, right there in the sheet at 0.62.
    Drawing the disc flat is what made the generated eye read as a plastic
    bead: every quantile of its luminance was defensible and it had no
    structure at all.

    The dark centre is **wide** — half its depth spans 0.312 of the radius,
    against 0.094 for the thin annulus this sheet drew instead until §7.173.
    **No marginal caught that**, including the ones invented to: the annulus's
    dip width sat inside the family (p10 0.062, and 21 of the 95 rows are at or
    below 0.094), its coverage and bbox were in range, and the darkness right
    around its own radius was too (1.000 of angles dark against a family median
    of 0.583, p90 1.000).  What placed it was the **residual by radius against
    the per-texel median value map** — +0.34, +0.24, +0.14 over the three
    innermost bands with a notch at t 0.27, a residual with a shape to it.

    ``outer`` and ``aspect`` come from the **outline**: the alpha masks of all
    95 sheets, resampled and stacked, give a family core whose IoU against ours
    is **0.943** with an area ratio of 0.992 and every bbox edge within 0.01.
    The silhouette was already right — an earlier eyeball reading of "our iris
    is much bigger" was wrong, and the difference on screen is entirely the
    shading.

    The shading parameters are a coordinate-descent fit against the **per-texel
    median luminance** of the same 95 sheets (each normalised to its own peak),
    which took the error from rms 0.121 to **0.0717** and the mean from +0.043
    to +0.001.  Fitting the three band profiles instead — what the first cut
    did — leaves a residual with an obvious shape to it: too dark along the top
    band, 0.10 too bright through the middle, and 0.20-0.23 too dark along the
    bottom rim, which is the brightest part of a Origin iris.

    **``base`` is the one thing that fit could not see, and the composite
    multiplies by it.**  Every family comparison here normalises each sheet to
    its own peak — which is right for judging a *drawing* and removes exactly
    one degree of freedom: the absolute level.  Left unconstrained it sat at
    0.8833, putting our ink's p95 at **213** against the family's **233**
    [p10 210 .. p90 248], and ``create_eye`` is ``tint * lum``, so the whole
    eye came out that much darker on screen.  0.9650 puts the default sheet at
    232 (§7.185).  A per-sheet-normalised score will never move this number,
    so it has to be set against the family's absolute one.

    Those rms figures are **not** comparable with the ones §7.173 quotes
    (0.1219 -> 0.0833 for the pupil): this fit was hand-rolled, and the ruler
    that now lives in ``face_tex_family --value`` normalises each row to its
    own 98th percentile and scores over the family core.  Same idea, different
    denominator — read either series against its own ``member`` floor, which
    for ``eye`` is 0.2104.

    **``outer`` and ``aspect`` are held fixed during that fit.**  Left free the
    search widened the disc to 1.55, because a score computed over the overlap
    can always trade silhouette for shading: it improved rms by 0.006 while
    giving up a verified outline (``CLAUDE.md`` 七).
    """
    U, V = _grid(size, size)
    cu, cv = 0.496, 0.489
    # ``aspect`` is v-radius / u-radius; the family's island is 0.689 wide.
    dx, dy = U - cu, (V - cv) / aspect
    r = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    t = np.clip(r / max(outer, 1e-6), 0.0, 1.0)
    # **The pupil and its rim get their own aspect.**  ``t`` is the island's,
    # which is right for the limbal ring — that one follows the outline — and
    # wrong for the pupil as soon as the pupil is not the island's shape.  It
    # need not be: a pupil that has to be a *different colour* is drawn as the
    # hole in ``eye_gradation`` (the iris sheet is single-channel and only
    # scales the tint it is handed), and that hole has its own ``pupil_aspect``.
    # Drawing the rim on the island's aspect then crosses the pupil at the top
    # and bottom instead of hugging it — measured on a tall pupil: the hole is
    # 1.89 high per unit wide on screen, the island 0.89.
    # ``1.0`` reproduces the previous term exactly, so every existing sheet is
    # bit-identical (§7.158).
    tp = t if abs(pup_aspect - 1.0) < 1e-9 else np.clip(
        np.hypot(dx, dy / max(pup_aspect, 1e-6)) / max(outer, 1e-6), 0.0, 1.0)
    ry = outer * aspect
    # Position down the island, 0 at the top: the lid shadow is a function of
    # height, not of radius, so it cannot be folded into the radial term.
    sv = np.clip((V - (cv - ry)) / max(2.0 * ry, 1e-6), 0.0, 1.0)

    def _ss(x: np.ndarray) -> np.ndarray:
        y = np.clip(x, 0.0, 1.0)
        return y * y * (3.0 - 2.0 * y)

    # ``lid_s`` is where the shadow ends and ``lid_w`` is how long it takes to
    # get there.  They were the same number until §7.177, i.e. a ramp that
    # starts fading the moment it leaves the top edge — and a Origin lid is a
    # **fast drop and then a plateau**, which that shape cannot draw at any
    # depth.  ``None`` reproduces the old single-parameter form exactly.
    lid = lid_d * _ss((lid_s - sv) / max(lid_s if lid_w is None else lid_w, 1e-6))
    floor = floor_d * _ss((sv - floor_a) / max(1.0 - floor_a, 1e-6))
    # **The pupil is a wide soft darkening of the middle, and it is measured.**
    # Fitted against the family's per-texel median value map with the outline
    # and ``limb_d`` held fixed: ``pup_d`` 0.325, ``pup_r`` 0.50, ``pup_s``
    # 0.35 — rms 0.1219 -> 0.0833, correlation 0.874 -> 0.938.  What that
    # replaced was a thin dark annulus (``prim_d`` below), which on screen was
    # a black hoop around a pale disc, and **no summary statistic said so**:
    # the annulus matched the family on coverage, bbox, radial mean by band,
    # dip width and angular spread alike.  Only the residual against the
    # per-texel median map has a shape to it — +0.34 / +0.24 / +0.14 over the
    # three innermost bands (§7.173).  Half-depth dip width is still the
    # honest *description* of the change, 0.094 -> 0.312 against a family
    # median of 0.312; it is not what detected it.
    pup = pup_d * _ss((pup_r - tp) / max(pup_s, 1e-6))
    # A dark rim around the pupil, and **off by default because the family
    # does not draw one**.  The mechanism is right — a rim has to live in
    # *this* sheet rather than the gradation map, which only chooses *which*
    # of the two card colours a texel takes, so a ring drawn there would be
    # another patch of ``subColor`` and not a darker edge — but shipping it at
    # 0.88 with no pupil behind it put a black hoop around a pale disc on
    # screen.  It is kept because a card that wants a drawn rim has nowhere
    # else to put it; it is not the category's grammar (§7.173).
    if prim_d > 0.0:
        pup = np.maximum(pup, prim_d * _ss(
            1.0 - np.abs(tp - prim_r) / max(prim_w, 1e-6)))
    limbal = limb_d * _ss((t - (1.0 - ring)) / max(ring, 1e-6))
    # **The limbal ring opens at the bottom.**  Applied uniformly by radius it
    # darkens the one place the family keeps brightest: per-texel, the median
    # of the 95 sheets is near-white along the lower rim, and ours came out
    # 0.20-0.23 darker there while being 0.10 too bright through the middle.
    limbal = limbal * (1.0 - ring_open * sv ** ring_pow)
    # Off by default: the eye is a flat colour with a highlight, so the
    # fibres are set to zero amplitude rather than removed — the family does
    # carry them (angular ripple 0.103) and a card that wants them can turn
    # ``fib`` back up.
    #
    # **Irregular, not a comb.**  Measured on 95 stock sheets, the angular
    # ripple left after removing the smooth radial and vertical trends is
    # 0.103 [0.046..0.274] of the mean; ours was **0.020** — five times less.
    # So the visible stripes were never an excess of amplitude, they were an
    # excess of *regularity*: 84 evenly spaced spokes beat against the texel
    # grid and read as moire.  Three incommensurate harmonics with fixed
    # offsets keep the bake deterministic and never repeat.
    #
    # **The amplitude stays low even though the family's is higher.**
    # Raising ``fib`` to reach 0.103 made the render visibly worse:
    # Origin's ripple is fine-grained texture, ours is coherent streaks that
    # run the whole radius, so the same number means two different
    # things.  A statistic that has been shown not to distinguish the
    # thing being judged does not get to overrule the picture.
    fibres = (0.5 + 0.36 * np.cos(ang * spokes * 0.73)
              + 0.34 * np.cos(ang * spokes + 1.31)
              + 0.22 * np.cos(ang * spokes * 1.51 + 2.62)) / 1.42
    body = base * (1.0 - lid) * (1.0 - floor) * (1.0 - pup) * (1.0 - limbal)         * (1.0 + fib * fibres * t)

    a = np.clip((outer - r) * size / 2.0 + 0.5, 0.0, 1.0)
    out = np.zeros((size, size, 4), dtype=np.uint8)
    val = np.clip(body * 255.0, 0, 255).astype(np.uint8)
    out[:, :, 0] = val
    out[:, :, 1] = val
    out[:, :, 2] = val
    out[:, :, 3] = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def eye_gradation(size: int = 512, *, centre: float = 0.47,
                  half: float = 0.22, pupil: float = 0.030,
                  pupil_u: float = 0.5, pupil_v: float = 0.652,
                  pupil_aspect: float = 2.30,
                  top: float = 1.0,
                  bottom: float = 1.0,
                  radial: float = 0.0, rad_r: float = 0.860,
                  rad_pow: float = 0.850) -> np.ndarray:
    """``_ColorMask`` for ``create_eye`` — the ramp the iris tint follows.

    ``tint = lerp(_Color2, _Color, g.r)``, so this map is **where the card's
    two eye colours meet**: high ``.r`` takes ``baseColor``, low takes
    ``subColor``.  Leaving it to Origin meant a generated iris sitting under
    Illusion's ramp, which is not an eye we own.

    Family (4 rows): everything is invariant — full coverage, opaque, mirrored
    — except whether the RGB varies at all, which is free.

    The stock row is a **pure vertical ramp**: measured by band it runs
    0 0 11 84 181 248 255 255 down the sheet while every column band is the
    same 129, i.e. **dark at the top, light at the bottom**.  Through the lerp
    that means the iris takes ``subColor`` at the top and ``baseColor`` at the
    bottom — the dark upper iris an anime eye has, and where the pupil on
    screen actually comes from.  The first draft ramped the other way and had
    a small range, which is why the generated eye came out as a flat green
    disc with no pupil at all.

    ``create_eye`` samples it through ``uv * 2 - 0.5`` at ``_grad = 0``, a 2x
    zoom about the centre, so the whole sheet's ramp lands across the middle
    half of the iris.
    """
    _U, V = _grid(size, size)
    t = np.clip((V - (centre - half)) / max(2.0 * half, 1e-6), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    val = top + (bottom - top) * t
    # **One of the four stock rows is radial, not a vertical ramp** — bright at
    # the sheet's centre and dark at the rim (measured, 16 radial bands: 0.967
    # down to 0.002), and it is the row the default **male** card names
    # (``gradMaskId`` 3; the female card names 2).  Handing every card the
    # vertical row's drawing is the cost of ignoring ``ident``, and it lands on
    # the sex whose face this project already cannot judge.
    #
    # ``(1 - r/rad_r)**rad_pow`` fits those 16 numbers at rms **0.023** with
    # ``rad_r`` 0.860, ``rad_pow`` 0.850, where r is in half-sheets.
    #
    # This one does **not** fight the pupil hole below, which is why it is drawn
    # and the vertical rows still are not: both put ``subColor`` away from the
    # centre, so the result reads as a dark pupil, a coloured iris and a dark
    # rim.  A vertical ramp puts ``subColor`` along the upper edge *as well as*
    # in the hole, which is the conflict recorded further down.
    if radial > 0.0:
        _Ur, _Vr = _grid(size, size)
        rr = np.hypot(_Ur * 2.0 - 1.0, _Vr * 2.0 - 1.0)
        val = val * (1.0 - radial) + radial * (
            np.clip(1.0 - rr / max(rad_r, 1e-6), 0.0, 1.0) ** max(rad_pow, 1e-6))
    # **The pupil is a region of this map, not a dark patch in the iris.**
    # ``create_eye`` does ``tint = lerp(_Color2, _Color, g.r)``, so wherever
    # this map is 0 the eye takes ``subColor`` outright.  A pupil that has to
    # be a *different colour* from the iris — rather than a darker version of
    # it — can only be drawn here: the iris sheet is single-channel and only
    # scales whatever tint it is handed.
    # Flat at 1 with a hole: the stock map is a top-to-bottom ramp, which puts
    # ``subColor`` along the **upper** rim — correct for Origin's two-tone iris and
    # wrong when ``subColor`` is meant to be the pupil.  The lid shadow is
    # already in the iris sheet, so nothing is lost by flattening it.
    if pupil > 0.0:
        _U2, _V2 = _grid(size, size)
        # **The hole goes where the sampler will fetch the iris centre, not
        # at the middle of the map.**  ``create_eye`` reads this through
        # ``uv * 2 - 0.5`` *and* the card's ``_grad`` ST
        # (``gradOffsetY``/``gradScale``: -0.208 and 0.629 on the default
        # card), so a hole drawn at v 0.5 came out over the iris's upper rim.
        # ``pupil_v`` is that pre-image, calibrated with two renders rather
        # than derived: a hole at map v 0.5 landed at iris 0.30 and one at
        # 0.752 at iris 0.72, so map_v = 0.60 * iris_v + 0.32 and the centre
        # wants 0.62.  Two points beat one derivation here because the ST is
        # applied on top of ``uv * 2 - 0.5`` and the sign convention was the
        # thing in doubt.  Nudged 0.620 -> 0.680 afterwards: the two-point
        # fit put the pupil a little high on screen, and the same two points
        # give the correction (0.60 per unit of iris height).  0.680 overshot low,
        # so the centred value is 0.652 — three renders, bracketed.
        #
        # **That 0.60 belongs to the stock ``outer``/``aspect``, not to this
        # map.**  The chain is map -> ``uv*2-0.5`` -> the card's ``_grad`` ST ->
        # the *iris disc*, so changing the disc changes the slope: with
        # ``outer=0.21, aspect=0.97`` two renders give 0.600 -> iris v 0.351 and
        # 0.689 -> 0.691, i.e. **3.82** per unit, six times steeper (§7.158).
        # Recalibrate with two renders whenever the disc moves; do not carry the
        # constant across.
        # Taller than wide: an anime pupil is an upright ellipse.  The v
        # radius is scaled here rather than after the fact because the card's
        # ``_grad`` ST already stretches this map on the way in.
        r = np.hypot((_U2 - pupil_u) / pupil,
                     (_V2 - pupil_v) / max(pupil * pupil_aspect, 1e-6))
        val = val * np.clip((r - 1.0) * size / 6.0 + 0.5, 0.0, 1.0)
    val = np.clip(val * 255.0, 0, 255).astype(np.uint8)
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[:, :, 0] = out[:, :, 1] = out[:, :, 2] = val
    out[:, :, 3] = 255
    return out


def eye_hi_up(size: int = 512, *, u: float = 0.454, v: float = 0.424,
              ru: float = 0.072, rv: float = 0.050,
              tilt: float = -0.35) -> np.ndarray:
    """``_overtex1`` — the upper catchlight.  An alpha stencil.

    Family (53 rows): ``rgb_is_flat`` is **invariant 1** — always a stencil,
    never a painted highlight — and *everything else is free*: coverage runs
    0.011 ± 0.016, the bounding box wanders over a quarter of the sheet and
    the mirror IoU spans the whole range.  So the grammar is "a small opaque
    shape somewhere over the iris", and pinning the position to one row's
    would be copying that row's choice rather than joining the family.  These
    defaults sit at the family medians.
    """
    U, V = _grid(size, size)
    du, dv = U - u, V - v
    c, sn = np.cos(tilt), np.sin(tilt)
    x, y = (du * c + dv * sn) / ru, (-du * sn + dv * c) / rv
    r = np.hypot(x, y)
    a = np.clip((1.0 - r) * size / 3.0, 0.0, 1.0)
    return _rgba(a, (255, 255, 255))


def eye_hi_down(size: int = 512, *, u: float = 0.6045, v: float = 0.6289,
                radius: float = 0.0391, aspect: float = 1.0) -> np.ndarray:
    """``_overtex2`` — the lower catchlight, a small round dot.

    **This drew nothing at all for six sections**, on a justification that a
    median over a bimodal family produced.  The reasoning was: the family's
    median position is u 0.45-0.54, that is inside our pupil (drawn at 0.94 so
    a grey eye reads as having one), the card tints the dot cyan, so a cyan blob
    in a black pupil is worse than no dot — turn it off.  Every step follows
    except the first.  Split by position, the 26 main rows are **13 at u 0.404
    and 13 at u 0.604 with nothing between them**; the nine 200-series rows are
    a separate centred set.  "u 0.496" is the midpoint of a gap, and it is the
    one place the main family never draws.  ``CLAUDE.md`` 七 already records
    that medians erase the ends of a distribution — here it invented a position.

    So the defaults are the **right lobe's** own medians (13 rows: u 0.6045,
    v 0.6289, bbox 0.078 x 0.070, ``bh/bw`` **1.000** — round, not the 1.15
    upright ellipse this drew before), which agree with row 7, the row the
    default female card names, to within 0.02.  On our own iris that lands at
    t = 0.89 of the disc radius — out on the limbal ring, not in the pupil, so
    the reason for switching it off does not apply to where Origin actually puts it.

    Same grammar as :func:`eye_hi_up`: a stencil (invariant across all 35 rows),
    position and size free — and *free* here means the family has **no core at
    all** (``face_tex_family --core``: the rows agree on no texel, peak 34%),
    which is why a lobe median is the strongest statement available.

    ``radius <= 0`` still turns it off, for a card that wants no lower
    catchlight.
    """
    if radius <= 0.0:
        return np.zeros((size, size, 4), dtype=np.uint8)
    U, V = _grid(size, size)
    r = np.hypot(U - u, (V - v) / max(aspect, 1e-6))
    a = np.clip((radius - r) * size / 2.5 + 0.5, 0.0, 1.0)
    return _rgba(a, (255, 255, 255))


def eye_white(size: int = 512, *, cu: float = 0.500, cv: float = 0.682,
              ru: float = 0.334, rv: float = 0.283, soft: float = 0.30,
              shadow: float = 1.0) -> np.ndarray:
    """``cw_t_sirome_kage`` — the **shadow stencil** for the sclera.

    ``create_eyewhite`` is ``lerp(_Color2, _Color, MainTex.r)``, so this map
    chooses between the card's ``whiteBaseColor`` (lit) and ``whiteSubColor``
    (the shade the lid casts).  Bright here = lit sclera.

    **This category has two kinds and we shipped the rare one.**  Row 0 is an
    8x8 pure white — that is how "no shadow" is spelled — and rows 1-4 are 512
    sheets carrying a bright blob low on the midline.  A flat sheet of ours was
    recorded as in-family, which it is: measured against the category's median
    value map row 0 scores **0.7145** and so did ours, while the shaded rows
    score 0.13-0.22.  What that reading missed is **which kind the cards ask
    for**: over 150 female cards ``whiteId`` is 1: 37%, 2: 32%, 3: 12%, 4: 9%
    and **0: 5%**, and the default card names row 1.  So 95% of characters were
    getting a sclera with no shade in it (§7.185).

    The blob, measured on the four shaded rows (bright = above half the sheet's
    99th percentile):

        centre    u 0.500-0.505      v 0.590-0.744
        extents  ru 0.291-0.406     rv 0.274-0.333
        softness   1% .. 60% of the bright area below full brightness

    ``shadow`` is the amplitude, and **0 reproduces row 0 exactly** whatever
    the blob is — a row measured as having no dark texels solves to it, so the
    two kinds come out of one generator instead of a special case.
    """
    U, V = _grid(size, size)
    t = np.hypot((U - cu) / max(ru, 1e-6), (V - cv) / max(rv, 1e-6))
    a = np.clip((1.0 - t) / max(soft, 1e-6), 0.0, 1.0)
    a = a * a * (3.0 - 2.0 * a)
    val = 255.0 * (1.0 - float(np.clip(shadow, 0.0, 1.0)) * (1.0 - a))
    out = np.zeros((size, size, 4), dtype=np.uint8)
    v = np.clip(val + 0.5, 0, 255).astype(np.uint8)
    out[:, :, 0] = out[:, :, 1] = out[:, :, 2] = v
    out[:, :, 3] = 255
    return out


def nip(size: int = 512, *, radius: float = 0.155, areola: float = 0.075,
        gloss: float = 1.0,
        hi_r: float = 0.64, hi_th: float = 91.0, hi_sr: float = 0.170,
        hi_st: float = 0.105, hi_peak: float = 243.0,
        lo_r: float = 0.50, lo_th: float = 274.0, lo_sr: float = 0.130,
        lo_st: float = 0.105, lo_peak: float = 113.0,
        dot_r: float = 0.13, dot_s: float = 0.026, dot_peak: float = 226.0
        ) -> np.ndarray:
    """Areola disc: a soft outer field with a firmer centre, plus its gloss.

    The colour is **measured**, not chosen.  The first version ramped a beige
    (228 → 184, with green and blue at 0.90/0.88 of red) on the reasoning that
    an areola is a darker skin tone; Origin's seven stock sheets are nothing like
    it — red sits at 242–249 across the whole disc while green and blue stay
    near zero, i.e. a saturated pink whose *hue* is the point:

        core (0.25 r)   ours (184, 165, 161)   Origin median (242,  23,  29)
        ring (0.60 r)   ours (228, 205, 200)   Origin median (249,   9,  11)

    Coverage was already in the family (0.083 against Origin's 0.064–0.082), so
    the shape was right and only the colour was invented — which is exactly
    what ``CLAUDE.md`` 一之四 forbids, and a case a coverage check cannot see.

    **The three channels are not three colours.**  ``main_skin``'s ``_overtex1``
    block reads this sheet as::

        nip = lerp(t.rgb * _overcolor1.rgb,
                   t.r * (_overcolor1.rgb + 0.33 * t.g * _nip_specular),
                   saturate(_tex1mask))

    and ``_tex1mask`` is authored **1** on both ``cf_m_body`` and ``cm_m_body``
    with nothing in ``ChaControl`` ever assigning it, so the branch that runs is
    the right-hand one: **R is a weight, G is a specular mask, and B is never
    sampled at all**.  ``_nip_specular`` is the card's ``nipGlossPower``, which
    is non-zero on 1,375 of this install's 1,388 cards, so the green channel is
    a live input and not a slot nobody fills.

    So the seven stock rows carry a **painted highlight** in green — 5.5% to
    10.0% of the ink (median 6.8%), in a few pieces rather than a speckle:

        main lobe    area 3.9%  r 0.64  theta 51-109  peak 243  0.25 x 0.15 R
        second lobe  area 1.5%  r 0.50  theta 250-300 peak 113  0.13 x 0.11 R
        two dots     area 0.15% r 0.13                peak 226  0.04 R

    Both lobes are elongated **along their own radius** (measured axis within
    a few degrees of the radial direction on every row), which is what a
    highlight on a dome looks like.  Ours had a flat green of 9-23 and no
    pieces at all, which ``--value`` reported as "the whole sheet is too
    bright" — the luminance it stacks mixes a weight with a highlight, and the
    finding only appears once the channels are read for what the shader does
    with them.  Forcing G 0 -> 255 moves 1,591 texels of the body albedo by up
    to 42 levels, so this reaches the frame (§7.193).

    ``theta`` is measured up-positive from the sheet's +u axis; the median of
    all the lobes is not usable (they are bimodal, upper against lower — the
    same trap as ``eye_hi_down``'s two lobes in §7.161), hence two terms.
    """
    U, V = _grid(size, size)
    r = np.hypot(U - 0.5055, V - 0.5)
    a = np.clip((radius - r) * size / 26.0 + 0.5, 0.0, 1.0)
    core = np.clip((areola - r) * size / 12.0 + 0.5, 0.0, 1.0)
    out = np.zeros((size, size, 4), dtype=np.uint8)
    # Green's **floor** is per band, not one number: taken as a median over the
    # rows at each radius it is 18-22 in the middle and 0-2 from t = 0.45 out
    # (rows 3 and 4 are 0 throughout).  The 9 that used to sit in the ring came
    # from a whole-disc statistic and is a sixth of the highlight's own median —
    # enough on its own to lift the whole sheet 0.06-0.13 above the family in
    # ``--value``, because luminance weights green 0.7152.
    for ch, (ring_v, core_v) in enumerate(((249.0, 242.0), (1.0, 19.0),
                                           (11.0, 29.0))):
        out[:, :, ch] = np.clip(ring_v + (core_v - ring_v) * core,
                                0, 255).astype(np.uint8)
    # ---- the specular channel, in units of the disc's own radius ----------
    dx = (U - 0.5055) / max(radius, 1e-6)
    dy = (V - 0.5) / max(radius, 1e-6)

    def _lobe(rr, th, sr, st, peak):
        ang = np.radians(th)
        ur, vr = np.cos(ang), -np.sin(ang)          # +v is down in the sheet
        px, py = dx - rr * ur, dy - rr * vr
        along = px * ur + py * vr                   # radial
        across = -px * vr + py * ur
        return peak * np.exp(-0.5 * ((along / sr) ** 2 + (across / st) ** 2))

    g = np.zeros_like(U)
    for piece in (_lobe(hi_r, hi_th, hi_sr, hi_st, hi_peak),
                  _lobe(lo_r, lo_th, lo_sr, lo_st, lo_peak),
                  _lobe(dot_r, hi_th, dot_s, dot_s, dot_peak),
                  _lobe(dot_r, lo_th, dot_s, dot_s, dot_peak)):
        g = np.maximum(g, piece)
    out[:, :, 1] = np.clip(np.maximum(out[:, :, 1], g * gloss),
                           0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def cheek(w: int = 1024, h: int = 256, *, cu: float = 0.330, cv: float = 0.326,
          ru: float = 0.170, rv: float = 0.250, peak: float = 0.58,
          falloff: float = 0.95) -> np.ndarray:
    """Blush — two soft patches, flat white with the shape in alpha.

    Built as ellipses rather than as strokes: a stroke's width is measured
    across it, so matching Origin's 22 px median gave a *thin* smear where the
    stock row (``cheek/2``, which is the one the default card names through
    ``baseMakeup.cheekId``) is a broad patch — coverage 0.051 against our
    0.017, and 0.23 of the sheet tall against our 0.066.  The right primitive
    for a blob is a blob.

    And it is a **faint** blob: Origin's sheet never exceeds alpha **148** and
    91% of its non-zero alpha is under 120, spread over four row bands
    (3.4 / 22.8 / 45.5 / 26.2 / 3.5).  Matching only the coverage gave a peak
    of 234 packed into one band (0 / 4.4 / 66.3 / 14.4 / 0) — the same ink in
    the same place, reading as a painted stripe instead of a flush.

    ``falloff`` is how fast it fades, and it is per row (§7.211).  It was a
    literal 0.95 for eight sections, which is a shape and not a measurement:
    across the nine stock rows the share of a patch's ink that is *soft*
    (alpha under 0.35 of full) runs **0.56 to 0.81**, and holding one exponent
    put rows 6 and 7 at 0.55 against their own 0.80 — twice the mid-tone the
    row carries, over the right outline.  The acceptance score could not see it
    either: it was a mask IoU taken at that same 0.35, so everything the
    exponent controls fell outside what was being scored.
    """
    U, V = _grid(w, h)
    a = np.zeros((h, w), dtype=np.float64)
    for c in (cu, 1.0 - cu):
        r = np.hypot((U - c) / ru, (V - cv) / rv)
        a = np.maximum(a, np.clip(1.0 - r, 0.0, 1.0) ** max(falloff, 1e-3))
    return _rgba(a * peak, (255, 255, 255))


# One eyeshadow patch, measured in the patch's own frame over the whole
# category — the same construction as ``LASH_V`` and for the same reason.
#
# Each of the 9 stock rows is split at the midline, the right half mirrored onto
# the left, each patch cropped to its **own** ink bbox and resampled to this
# grid, normalised by its own peak alpha; 18 patches, per-cell median.  Rows
# down = nv (top of the patch to bottom), columns across = nu (the sheet's
# **outer** side to the inner one).  Nothing of any row's resolution or
# placement survives that, which is the point (``著作權.md`` 四).
#
# What it says that the marginals did not: the patch is a **rounded blob**,
# brightest just inside the middle (nu 0.55) and falling off on all four sides.
# The old generator drew two feathered strokes, which is a **bar** — flat-topped
# at 0.945 across six of ten columns with hard shoulders — and it was the wrong
# aspect as well: ours measured 0.289 wide by 0.211 tall against the family's
# 0.213 by 0.286, i.e. a wide flat smear where the category draws an upright
# patch.  ``cheek`` already carries this lesson in its own docstring ("the right
# primitive for a blob is a blob") and this sheet did not get it.
SHADOW_A = np.array([
    [0.012, 0.175, 0.427, 0.427, 0.604, 0.633, 0.314, 0.180, 0.094, 0.008],
    [0.186, 0.506, 0.769, 0.843, 0.929, 0.957, 0.910, 0.596, 0.302, 0.027],
    [0.318, 0.733, 0.953, 0.945, 0.976, 0.988, 0.957, 0.894, 0.682, 0.247],
    [0.506, 0.788, 0.922, 0.980, 0.976, 0.957, 0.878, 0.867, 0.686, 0.435],
    [0.498, 0.694, 0.914, 0.969, 0.965, 0.922, 0.898, 0.816, 0.673, 0.455],
    [0.437, 0.647, 0.733, 0.880, 0.969, 0.961, 0.949, 0.922, 0.847, 0.498],
    [0.067, 0.506, 0.357, 0.522, 0.831, 0.918, 0.886, 0.776, 0.502, 0.125],
    [0.000, 0.051, 0.090, 0.216, 0.463, 0.573, 0.449, 0.224, 0.051, 0.008]])

# A ring of zeros so the sampler fades out over one cell instead of clamping to
# the border value: the table is cropped **at** the ink bbox, so its edge cells
# are around the 0.35 ink threshold, not at nothing.  Clamping there would end
# the patch on a hard step at half opacity.
_SHADOW_PAD = np.pad(SHADOW_A, 1)


def eyeshadow(w: int = 512, h: int = 256, *, cu: float = 0.2178,
              cv: float = 0.6387, ru: float = 0.1065, rv: float = 0.1433,
              peak: float = 1.0, falloff: float = 1.0) -> np.ndarray:
    """Lid shadow — two patches, flat white with the shape in the alpha.

    **Placement came from the family, not from a guess.**  The previous
    version's ``v = 0.450`` was never measured: stacking all nine rows on a
    common grid and taking the texels at least half of them ink gives a core at
    v **0.484–0.793**, and ours sat at 0.387–0.496 — an IoU of **0.003**, ink
    a seventh of the sheet above where the category puts it.  Every marginal
    said it was close (coverage 0.0895 against 0.0985, bbox u within 0.05):
    coverage, stroke width and row/column profiles are all projections and none
    of them can see a translation.  That is what ``face_tex_family --core`` is
    for, and this sheet is why it exists.

    Size is the **median of each patch's own extent** (0.213 x 0.287 of the
    sheet) rather than the core's, which is narrower than any member — the same
    distinction ``LASH_W`` records.  Centre is the median of each patch's own
    centre (u 0.218, v 0.639); the core's centre is biased outward by exactly
    that narrowing.

    Peak alpha is **255**: the family's median peak is 255 and its lowest row
    248, while the old wash multiplied by 0.85 for no measured reason.

    ``falloff`` bends :data:`SHADOW_A` — the family's **median** profile — so a
    row can be flatter or softer than the middle of its family without any
    other row's pixels entering (§7.211).  ``peak`` cannot do that job: it
    scales the whole patch, so the *shape* stayed the median's on every row and
    the median alpha of our ink came out **0.573-0.576 on all nine**, a
    constant, against Origin's 0.384 to 1.000 — rows 4 and 5 are all but solid and
    we drew the family's soft bump over their outline.  This is the fourth time
    a shared model has been frozen into a per-row table (§7.171, §7.174,
    §7.187): the tell is a column of the same number where the family spans a
    range.
    """
    U, V = _grid(w, h)
    ns, nq = SHADOW_A.shape
    a = np.zeros((h, w), dtype=np.float64)
    # Mirrored: ``mirror_iou`` is an invariant 1.000 across the family, so the
    # sheet really does carry both sides (unlike the brow and the lash line).
    # ``sign`` keeps nu = 0 on the sheet's outer edge for both patches, which is
    # the frame the table was measured in.
    for centre, sign in ((cu, 1.0), (1.0 - cu, -1.0)):
        nu = 0.5 + sign * (U - centre) / (2.0 * max(ru, 1e-6))
        nv = 0.5 + (V - cv) / (2.0 * max(rv, 1e-6))
        # Map the patch box onto the padded table's interior; the sampler's own
        # clipping then lands on the zero ring outside it.
        q = (1.0 + nu * (nq - 1)) / (nq + 1)
        s = (1.0 + nv * (ns - 1)) / (ns + 1)
        a = np.maximum(a, _table2d(_SHADOW_PAD, s, q))
    if abs(falloff - 1.0) > 1e-6:
        a = a ** max(falloff, 1e-3)
    return _rgba(a * peak, (255, 255, 255))


def detail_mask(size: int = 1024, *, outline: float = 0.0) -> np.ndarray:
    """``_DetailMask``: R specular, G rim, B outline-width scale, A the gate.

    Measured on the stock rows: ``face_detail/1`` is **all zero** with alpha
    255 (R reaches 82 on 1% of texels and nothing else is non-zero), while
    ``body_detail/1`` carries a B median of 21 and ``body_detail/0`` one of 68.
    B is the *outline width scale* through ``1 - B`` (``CLAUDE.md`` 七), so
    zero means full-width outlines and the stock body rows only trim them
    slightly.  There is nothing to draw here — the honest generated mask is a
    constant, and saying so is better than inventing a pattern.

    The nail / lip region (``A < 0.5``, 1.17% of the body) is **not** generated:
    it needs UV positions we have not measured, so alpha stays 255 and the
    nails simply get the same specular as the skin.  That is a visible
    simplification, recorded rather than papered over.
    """
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[:, :, 2] = int(np.clip(outline * 255.0, 0, 255))
    out[:, :, 3] = 255
    return out


def face_detail(size: int = 1024) -> np.ndarray:
    return detail_mask(size, outline=0.0)


def body_detail(size: int = 1024) -> np.ndarray:
    return detail_mask(size, outline=21.0 / 255.0)


# Where each sheet belongs on disk.  The generated set is written into a tree,
# not a flat folder: a directory of sixteen loose PNGs says nothing about which
# of them are one feature, and the eye alone is five of them.
GROUPS = {
    "eye": "face/eye", "eye_gradation": "face/eye", "eye_hi_up": "face/eye",
    "eye_hi_down": "face/eye", "eye_white": "face/eye",
    "eyeline_up": "face/eyeline", "eyeline_down": "face/eyeline",
    "eyeline_shadow": "face/eyeline",
    "eyebrow": "face/brow",
    "cheek": "face/makeup", "eyeshadow": "face/makeup", "lip": "face/makeup",
    "lipline": "face/makeup", "mole": "face/makeup",
    "face_detail": "face/detail",
    "nip": "body", "body_detail": "body/detail",
}


# Per-row parameters, one entry per stock row, measured by
# ``tools/face_tex_perid.py`` — **regenerate it, do not hand-edit**.
#
# Each value is a dimensionless measurement of that row: where the motif sits,
# how far it extends, how wide the stroke is, how strong the wash is — and, for
# ``eye``, **how dark the lid shadow across the top is**, which is a value and
# not a shape.  It is in here because a filled disc's outline says nothing
# about its shading, so solving only the alpha handed every card the family's
# median lid while the rows span 0.25 to 0.89 (§7.174).  A few numbers per row
# against a 512² sheet describes *where a shape is and roughly how dark* and
# does not reconstruct the drawing, which is the same footing as
# ``parttex.PATTERN_TABLE`` and ``LASH_V`` (``著作權.md`` 四: a measurement is a
# fact, the pixels are the expression).  Nothing here is resampled or traced,
# and ``tools/release_audit.py`` still governs what may ship.
#
# **A row is present only when its own numbers beat the category default** on
# that row, scored by the tool.  Rows where the generator has no grammar for the
# drawing — a brow with a break in it, a lower catchlight drawn as a hollow ring
# — are absent on purpose and fall back to the default sheet.
#
# A row drawn as **several** marks is measured twice, once from the union of its
# ink and once from its biggest piece alone, and keeps whichever agrees with it
# better (§7.199).  So the freckle *bands* in the mole list are in here after
# all, at 0.057 and 0.031: one dot where that card has a freckle beats one dot
# where she has nothing, and the band grammar is still missing.  ``--rows``
# prints ``[biggest piece]`` against those entries, because how much of a row an
# entry answers for is part of what it says.
PER_ID: dict = {
    "eyeshadow": {
        1: {"cu": 0.2266, "cv": 0.5137, "ru": 0.1183, "rv": 0.1608, "peak": 1.0000},
        2: {"cu": 0.2530, "cv": 0.4512, "ru": 0.1065, "rv": 0.1119, "peak": 0.9922, "falloff": 1.2523},
        3: {"cu": 0.2149, "cv": 0.7871, "ru": 0.1377, "rv": 0.1164, "peak": 0.9725, "falloff": 5.1876},
        4: {"cu": 0.1416, "cv": 0.6407, "ru": 0.0774, "rv": 0.1381, "peak": 1.0000},
        5: {"cu": 0.2256, "cv": 0.7500, "ru": 0.1520, "rv": 0.1206, "peak": 1.0000},
        6: {"cu": 0.1944, "cv": 0.5274, "rv": 0.1568, "peak": 1.0000, "falloff": 0.4519},
        7: {"cu": 0.2393, "cv": 0.8399, "rv": 0.0908, "peak": 0.9961, "falloff": 2.8904},
        8: {"cu": 0.1905, "cv": 0.6797, "ru": 0.0828, "rv": 0.2115, "peak": 1.0000},
        9: {"cu": 0.1387, "cv": 0.6172, "ru": 0.0528, "rv": 0.1276, "peak": 1.0000},
    },
    "cheek": {
        1: {"cu": 0.2846, "cv": 0.5037, "ru": 0.1499, "rv": 0.4026, "peak": 0.8510},
        2: {"cu": 0.3095, "cv": 0.3260, "ru": 0.1866, "rv": 0.2783, "peak": 0.5804},
        3: {"cu": 0.1879, "cv": 0.3944, "ru": 0.1693, "rv": 0.4529, "peak": 1.0000},
        4: {"cu": 0.3046, "cv": 0.3358, "ru": 0.1337, "rv": 0.2252, "peak": 0.9725, "falloff": 1.0859},
        5: {"cu": 0.3036, "cv": 0.3299, "ru": 0.0704, "rv": 0.2184, "peak": 0.8745},
        6: {"cu": 0.3041, "ru": 0.1472, "rv": 0.2819, "peak": 0.9922, "falloff": 2.1795},
        7: {"cu": 0.3100, "ru": 0.1629, "rv": 0.2518, "peak": 0.9686, "falloff": 2.1268},
        8: {"cu": 0.2607, "cv": 0.4725, "ru": 0.2066, "rv": 0.5294, "peak": 1.0000},
        9: {"cu": 0.2773, "cv": 0.5467, "ru": 0.0616, "rv": 0.5942, "peak": 1.0000},
    },
    "eye_hi_up": {
        0: {"u": 0.3837, "v": 0.4054, "ru": 0.0538, "rv": 0.0473},
        1: {"u": 0.3651, "v": 0.4347, "ru": 0.0670, "rv": 0.0211},
        2: {"u": 0.6405, "v": 0.4484, "ru": 0.0872},
        3: {"u": 0.4296, "v": 0.3410, "ru": 0.0488},
        4: {"u": 0.6220, "v": 0.3410, "ru": 0.0172, "rv": 0.0157},
        5: {"u": 0.4501, "v": 0.3527, "ru": 0.1065, "rv": 0.0862},
        6: {"u": 0.3515, "v": 0.5119, "ru": 0.0346, "rv": 0.0353},
        7: {"u": 0.3925, "v": 0.4084, "rv": 0.0819},
        8: {"u": 0.3905, "v": 0.3664, "ru": 0.0407},
        9: {"u": 0.4335, "v": 0.3869, "ru": 0.0609},
        10: {"u": 0.4130, "v": 0.4045, "rv": 0.0893},
        11: {"u": 0.3974, "v": 0.4045, "ru": 0.0782, "rv": 0.0892},
        12: {"u": 0.4149, "v": 0.3918, "rv": 0.0715},
        13: {"u": 0.4130, "v": 0.4006, "rv": 0.0743},
        14: {"u": 0.5936, "v": 0.3849, "ru": 0.0429, "rv": 0.0634},
        15: {"u": 0.4149, "v": 0.4035, "ru": 0.0639},
        16: {"u": 0.6444, "v": 0.3888, "ru": 0.0101, "rv": 0.0093},
        17: {"u": 0.3827, "v": 0.3986, "ru": 0.0933},
        18: {"u": 0.3964, "v": 0.4260, "ru": 0.0508},
        19: {"u": 0.3749, "v": 0.5363, "ru": 0.0690},
        20: {"u": 0.4042, "v": 0.4211, "ru": 0.0812, "rv": 0.0789},
        21: {"u": 0.6122, "v": 0.4074, "ru": 0.0645, "rv": 0.0960},
        22: {"u": 0.6366, "v": 0.4113, "ru": 0.0554, "rv": 0.0886},
        23: {"u": 0.4032, "v": 0.3879, "ru": 0.0417},
        24: {"u": 0.5624, "v": 0.3459, "ru": 0.0406, "rv": 0.0408},
        25: {"u": 0.3612, "v": 0.4152, "ru": 0.0750, "rv": 0.0454},
        26: {"u": 0.4052, "v": 0.3879, "ru": 0.0598},
        27: {"u": 0.6210, "v": 0.3615, "ru": 0.0346, "rv": 0.0353},
        28: {"u": 0.6259, "v": 0.4133, "ru": 0.0502, "rv": 0.0755},
        29: {"u": 0.3690, "v": 0.4904, "ru": 0.0411, "rv": 0.0691},
        30: {"u": 0.3847, "v": 0.3830, "ru": 0.0467, "rv": 0.0464},
        31: {"u": 0.5692, "v": 0.3927, "ru": 0.0558},
        32: {"u": 0.3769, "v": 0.4289, "ru": 0.0651, "rv": 0.0772},
        33: {"u": 0.3915, "v": 0.4640, "ru": 0.0923},
        34: {"u": 0.3778, "v": 0.4904, "ru": 0.0558, "rv": 0.0576},
        35: {"u": 0.4901, "v": 0.3478, "ru": 0.0548, "rv": 0.0297},
        36: {"u": 0.3749, "v": 0.4972, "ru": 0.0487, "rv": 0.0216},
        37: {"u": 0.3720, "v": 0.4504, "ru": 0.0679, "rv": 0.0327},
        38: {"u": 0.3700, "v": 0.5158, "ru": 0.0740},
        39: {"u": 0.4735, "v": 0.3849, "ru": 0.1227, "rv": 0.1038},
        40: {"u": 0.6102, "v": 0.3918, "ru": 0.0600, "rv": 0.0735},
        200: {"u": 0.3671, "v": 0.4592, "ru": 0.0791, "rv": 0.0574},
        201: {"u": 0.4979, "v": 0.4640, "ru": 0.0710, "rv": 0.0677},
        202: {"u": 0.4999, "v": 0.4601, "rv": 0.0921},
        203: {"u": 0.4999, "v": 0.4982, "ru": 0.0954, "rv": 0.0853},
        204: {"u": 0.4999, "v": 0.4650, "ru": 0.0936, "rv": 0.1097},
        205: {"u": 0.4999, "v": 0.4826, "ru": 0.0698, "rv": 0.1119},
        206: {"u": 0.4999, "v": 0.4806, "ru": 0.0702, "rv": 0.1269},
        207: {"u": 0.5028, "v": 0.4943, "ru": 0.0335, "rv": 0.0241},
        210: {"u": 0.4960, "v": 0.4894, "ru": 0.0875, "rv": 0.1088},
    },
    "eye_hi_down": {
        0: {"u": 0.6123, "v": 0.6172, "radius": 0.0334, "aspect": 1.1165},
        1: {"u": 0.3457, "v": 0.5566, "radius": 0.0477, "aspect": 0.8963},
        2: {"u": 0.4365, "v": 0.7129, "radius": 0.0334, "aspect": 0.9143},
        3: {"u": 0.4092, "v": 0.6152, "radius": 0.0238, "aspect": 1.0889},
        4: {"u": 0.6455, "v": 0.5830, "radius": 0.0334},
        6: {"u": 0.6279, "v": 0.6680, "radius": 0.0544, "aspect": 0.3115},
        7: {"u": 0.6221, "v": 0.6162, "radius": 0.0334},
        8: {"u": 0.5830, "v": 0.6289, "radius": 0.0677, "aspect": 0.8245},
        9: {"u": 0.5820, "v": 0.6592, "radius": 0.0572, "aspect": 1.2669},
        10: {"u": 0.5820, "v": 0.6894, "radius": 0.0210, "aspect": 1.0083},
        11: {"u": 0.5918, "v": 0.6592, "radius": 0.0534, "aspect": 0.5587},
        12: {"u": 0.3877, "v": 0.6172, "radius": 0.0334, "aspect": 1.2250},
        13: {"u": 0.3887, "v": 0.6533, "radius": 0.0496, "aspect": 0.5682},
        14: {"u": 0.6045, "v": 0.6768, "radius": 0.0238},
        15: {"u": 0.3740, "v": 0.6406, "radius": 0.0601, "aspect": 0.2893},
        16: {"u": 0.3770, "v": 0.6309, "radius": 0.0458, "aspect": 0.6250},
        17: {"u": 0.4043, "v": 0.6289, "radius": 0.0458, "aspect": 1.1879},
        18: {"u": 0.4082, "v": 0.6260, "radius": 0.0324, "aspect": 1.2493},
        19: {"u": 0.3994, "v": 0.6592, "radius": 0.0486, "aspect": 1.2726},
        20: {"u": 0.6289, "v": 0.6465, "aspect": 1.2500},
        21: {"u": 0.6397, "v": 0.6113, "radius": 0.0219, "aspect": 0.8100},
        22: {"u": 0.5889, "v": 0.6338, "radius": 0.0505, "aspect": 1.0360},
        23: {"u": 0.3887, "v": 0.6348, "radius": 0.0267},
        24: {"u": 0.3877, "v": 0.6221, "radius": 0.0410, "aspect": 1.1823},
        25: {"u": 0.6006, "v": 0.6221, "radius": 0.0563, "aspect": 1.1570},
        200: {"u": 0.4902, "v": 0.4980, "radius": 0.0267},
        201: {"u": 0.4971, "v": 0.4639, "radius": 0.0353, "aspect": 0.9280},
        202: {"u": 0.4961, "v": 0.4687, "aspect": 1.1500},
        203: {"u": 0.4961, "v": 0.4668, "radius": 0.0458, "aspect": 0.9308},
        204: {"u": 0.4961, "v": 0.4678, "aspect": 1.3268},
        205: {"u": 0.4971, "v": 0.4678, "radius": 0.0372, "aspect": 1.4269},
        206: {"u": 0.4961, "v": 0.4687, "radius": 0.0381},
        208: {"u": 0.4961, "v": 0.4678, "aspect": 1.3268},
    },
    "mole": {
        1: {"u": 0.5020, "v": 0.5020, "radius": 0.0080},
        2: {"u": 0.7803, "v": 0.5762, "radius": 0.0051},
        3: {"u": 0.8311, "v": 0.5665, "radius": 0.0032},
    },
    "eye": {
        0: {"outer": 0.1942, "aspect": 1.4769, "lid_d": 0.5042, "lid_w": 0.5500},
        1: {"outer": 0.1942, "aspect": 1.5242, "lid_d": 0.4162, "lid_w": 0.7500},
        2: {"outer": 0.2059, "aspect": 1.2792, "lid_d": 0.9735, "lid_w": 0.5500},
        3: {"outer": 0.1942, "aspect": 1.5295, "lid_d": 0.7974, "lid_w": 0.7500},
        4: {"outer": 0.1942, "aspect": 1.5295, "lid_d": 0.8787, "lid_w": 0.7500},
        5: {"outer": 0.1942, "aspect": 1.5295, "lid_d": 0.5173, "lid_w": 0.5500},
        6: {"outer": 0.1942, "aspect": 1.5295, "lid_d": 0.5870, "lid_w": 0.5500},
        7: {"outer": 0.1942, "aspect": 1.5295, "lid_d": 0.2164, "lid_w": 0.5500},
        8: {"outer": 0.1942, "aspect": 1.5295, "lid_d": 0.4840, "lid_w": 0.5500},
        9: {"outer": 0.1942, "aspect": 1.5295, "lid_d": 0.3572, "lid_w": 0.7500},
        10: {"outer": 0.1942, "aspect": 1.5295, "lid_d": 0.9268, "lid_s": 0.8000},
        11: {"outer": 0.1767, "aspect": 1.4070},
        12: {"outer": 0.1787, "aspect": 1.4158, "lid_d": 0.3339, "lid_w": 0.4500},
        13: {"outer": 0.1787, "aspect": 1.5427, "lid_d": 0.4771, "lid_w": 0.4500},
        14: {"outer": 0.1787, "aspect": 1.3736, "lid_d": 0.5573, "lid_w": 0.7500},
        15: {"outer": 0.1534, "aspect": 1.4491, "lid_d": 0.6433},
        17: {"outer": 0.1787, "aspect": 1.5427, "lid_d": 0.4989, "lid_w": 0.5500},
        18: {"outer": 0.1787, "aspect": 1.5427},
        20: {"outer": 0.1806, "aspect": 1.4519, "lid_d": 0.6596, "lid_w": 0.3500},
        21: {"outer": 0.1806, "aspect": 1.4519, "lid_d": 0.4555, "lid_w": 0.7500},
        22: {"outer": 0.1806, "aspect": 1.4573, "lid_d": 0.8374, "lid_w": 0.3500},
        23: {"outer": 0.1806, "aspect": 1.5629, "lid_d": 0.1644, "lid_w": 0.4500},
        24: {"outer": 0.1952, "aspect": 1.7377, "lid_d": 0.8105, "lid_w": 0.7500},
        25: {"outer": 0.1728, "aspect": 1.3459, "lid_d": 0.7745, "lid_w": 0.3500},
        26: {"outer": 0.1806, "aspect": 1.5629, "lid_d": 0.7038},
        27: {"outer": 0.2097, "aspect": 1.1771},
        28: {"outer": 0.1806, "aspect": 1.4836, "lid_d": 0.6000, "lid_s": 0.8000},
        29: {"outer": 0.1806, "aspect": 1.5629, "lid_d": 0.6110, "lid_w": 0.4500},
        30: {"outer": 0.2097, "aspect": 1.1771, "lid_d": 0.5416, "lid_w": 0.3500},
        32: {"outer": 0.1923, "aspect": 1.7713, "lid_d": 0.5045, "lid_w": 0.5500},
        33: {"outer": 0.1670, "aspect": 1.5600, "lid_d": 0.2550, "lid_w": 0.7500},
        34: {"outer": 0.1806, "aspect": 1.5629, "lid_d": 0.5024, "lid_w": 0.7500},
        35: {"outer": 0.1923, "aspect": 1.2849, "lid_d": 0.2797, "lid_w": 0.7500},
        37: {"outer": 0.1923, "aspect": 1.2947},
        38: {"outer": 0.1845, "aspect": 1.5572, "lid_d": 0.3870, "lid_w": 0.4500},
        39: {"outer": 0.1806, "aspect": 1.5629, "lid_d": 0.6383, "lid_w": 0.7500},
        40: {"outer": 0.1806, "aspect": 1.5629, "lid_d": 0.6905},
        41: {"outer": 0.2097, "aspect": 1.1771, "lid_d": 0.5484, "lid_w": 0.3500},
        42: {"outer": 0.1728, "aspect": 1.3939, "lid_d": 0.5681, "lid_w": 0.4500},
        43: {"outer": 0.1903, "aspect": 1.3365, "lid_d": 0.7081},
        44: {"outer": 0.1884, "aspect": 1.2356, "lid_d": 0.3448, "lid_w": 0.3500},
        45: {"outer": 0.1903, "aspect": 1.3365, "lid_d": 0.3199},
        46: {"outer": 0.2156, "aspect": 1.1025},
        47: {"outer": 0.2020, "aspect": 1.1333},
        49: {"outer": 0.1728, "aspect": 1.3939, "lid_d": 0.5748, "lid_w": 0.3500},
        50: {"outer": 0.1923, "aspect": 1.2845},
        51: {"outer": 0.1903, "aspect": 1.3365, "lid_d": 0.5030, "lid_s": 0.9000},
        52: {"outer": 0.1855, "aspect": 1.5149, "lid_d": 0.6515, "lid_s": 0.8000},
        53: {"outer": 0.1903, "aspect": 1.3365},
        54: {"outer": 0.1903, "aspect": 1.3365, "lid_d": 0.7480, "lid_w": 0.7500},
        55: {"outer": 0.1903, "aspect": 1.2046, "lid_d": 0.5941, "lid_w": 0.5500},
        56: {"outer": 0.1903, "aspect": 1.2046, "lid_d": 0.6416},
        57: {"outer": 0.1855, "aspect": 1.5149, "lid_d": 0.9603},
        58: {"outer": 0.1903, "aspect": 1.2097, "lid_d": 0.7504, "lid_w": 0.5500},
        59: {"outer": 0.1903, "aspect": 1.2046, "lid_d": 0.5659, "lid_w": 0.7500},
        60: {"outer": 0.1903, "aspect": 1.2046, "lid_d": 0.5092, "lid_w": 0.4500},
        61: {"outer": 0.1767, "aspect": 1.3696},
        62: {"outer": 0.2156, "aspect": 1.1077},
        63: {"outer": 0.1884, "aspect": 1.5046, "lid_d": 0.6185, "lid_s": 0.9000},
        64: {"outer": 0.1884, "aspect": 1.5046, "lid_d": 0.6034, "lid_s": 0.9000},
        65: {"outer": 0.1874, "aspect": 1.4995, "lid_d": 0.3467, "lid_s": 0.6000},
        66: {"outer": 0.1874, "aspect": 1.4995, "lid_d": 0.3467, "lid_s": 0.6000},
        67: {"outer": 0.1874, "aspect": 1.4995, "lid_d": 0.3808, "lid_s": 0.8000},
        68: {"outer": 0.1884, "aspect": 1.5046, "lid_d": 0.3466, "lid_w": 0.5500},
        69: {"outer": 0.1884, "aspect": 1.5046, "lid_d": 0.6225},
        70: {"outer": 0.1874, "aspect": 1.4995, "lid_d": 0.6188, "lid_s": 0.9000},
        71: {"outer": 0.1884, "aspect": 1.5046, "lid_d": 0.6885, "lid_s": 0.7000},
        72: {"outer": 0.1923, "aspect": 1.2002},
        73: {"outer": 0.1748, "aspect": 1.3873, "lid_d": 0.3249},
        74: {"outer": 0.1884, "aspect": 1.5046, "lid_d": 0.5839, "lid_w": 0.5500},
        75: {"outer": 0.1806, "aspect": 2.1015, "lid_d": 0.2513, "lid_w": 0.4500},
        76: {"outer": 0.1534, "aspect": 1.4491, "lid_d": 0.0204, "lid_w": 0.3500},
        200: {"outer": 0.2068, "aspect": 1.0452, "lid_d": 0.4262, "lid_w": 0.5500},
        201: {"outer": 0.2020, "aspect": 1.4543, "lid_d": 0.5540, "lid_w": 0.5500},
        202: {"outer": 0.2020, "aspect": 1.4591, "lid_d": 0.5932, "lid_w": 0.5500},
        203: {"outer": 0.2020, "aspect": 1.4591, "lid_d": 0.4276, "lid_w": 0.4500},
        204: {"outer": 0.2020, "aspect": 1.4591, "lid_d": 0.5464, "lid_w": 0.5500},
        205: {"outer": 0.2020, "aspect": 1.4543, "lid_d": 0.5355},
        206: {"outer": 0.2020, "aspect": 1.4591, "lid_d": 0.7802, "lid_w": 0.4500},
        207: {"outer": 0.1748, "aspect": 1.6673, "lid_d": 0.4804, "lid_w": 0.5500},
        208: {"outer": 0.2068, "aspect": 1.0343, "lid_d": 0.2067, "lid_w": 0.5500},
        209: {"outer": 0.1913, "aspect": 1.3268, "lid_d": 0.2388, "lid_s": 0.6000},
        210: {"outer": 0.2068, "aspect": 1.0343, "lid_d": 0.2642, "lid_s": 0.6000},
        211: {"outer": 0.2068, "aspect": 1.0343, "lid_d": 0.3189, "lid_s": 0.9000},
        212: {"outer": 0.2068, "aspect": 1.0343, "lid_d": 0.3639, "lid_s": 0.6000},
        213: {"outer": 0.1913, "aspect": 1.3268, "lid_d": 0.1969, "lid_w": 0.4500},
        214: {"outer": 0.2068, "aspect": 1.0343, "lid_d": 0.4472, "lid_s": 0.8000},
        215: {"outer": 0.2068, "aspect": 1.0344, "lid_d": 0.2438, "lid_s": 0.6000},
        216: {"outer": 0.2068, "aspect": 1.0344, "lid_d": 0.2694, "lid_w": 0.4500},
        217: {"outer": 0.2068, "aspect": 1.0344, "lid_d": 0.2452, "lid_s": 0.9000},
    },
    "eyebrow": {
        0: {"thickness": 0.0265, "arch": 0.1047, "u1": 0.7859},
        1: {"thickness": 0.0186, "arch": 0.1593, "u1": 0.7859},
        2: {"thickness": 0.0911, "arch": 0.1351, "u1": 0.6003},
        3: {"thickness": 0.0186, "arch": 0.1342, "u1": 0.5339},
        4: {"thickness": 0.0265, "arch": 0.1285, "u1": 0.5534},
        5: {"arch": 0.1315, "u0": 0.0895, "u1": 0.7155},
        6: {"arch": 0.1199, "u0": 0.0875, "u1": 0.7761},
        7: {"thickness": 0.0994, "arch": 0.1545, "u0": 0.0855, "u1": 0.5632},
        8: {"arch": 0.1288, "u0": 0.1676, "u1": 0.6921},
        9: {"thickness": 0.0265, "arch": 0.1361, "u1": 0.7175},
        10: {"thickness": 0.0497, "arch": 0.1152, "u1": 0.8132},
        11: {"thickness": 0.0426, "arch": 0.0927, "u1": 0.5241},
        12: {"thickness": 0.1160, "arch": 0.1365, "u0": 0.0914, "u1": 0.7019},
        13: {"thickness": 0.1003, "arch": 0.0370, "u0": 0.1246, "u1": 0.3445},
        200: {"thickness": 0.0497, "arch": 0.1129, "u1": 0.7937},
        201: {"thickness": 0.0345, "arch": 0.1567, "u1": 0.7507},
    },
    "eyeline_down": {
        0: {"thickness": 0.0117, "u0": 0.2810, "v0": 0.7480, "v1": 0.7743, "bow": 0.0636},
        1: {"thickness": 0.0117, "u0": 0.4295, "u1": 0.6464, "v0": 0.7883, "v1": 0.7626, "bow": 0.0383},
        2: {"thickness": 0.0525, "u0": 0.2810, "v0": 0.7405, "v1": 0.7783, "bow": 0.0352},
        3: {"thickness": 0.0438, "u0": 0.2888, "u1": 0.6327, "v0": 0.7367, "v1": 0.7621, "bow": 0.0597},
        4: {"thickness": 0.0175, "u0": 0.6053, "u1": 0.6288, "v0": 0.7586, "v1": 0.7867, "bow": -0.0008},
        5: {"thickness": 0.0280, "u0": 0.2576, "u1": 0.7284, "v0": 0.7207, "v1": 0.6600, "bow": 0.2064},
        6: {"u0": 0.3533, "u1": 0.4569, "v0": 0.7682, "v1": 0.7942, "bow": 0.0094},
        7: {"thickness": 0.0175, "u0": 0.4666, "u1": 0.4726, "v0": 0.7908, "v1": 0.7898, "bow": 0.0270},
        8: {"thickness": 0.0438, "u0": 0.3748, "u1": 0.5136, "v0": 0.7649, "v1": 0.7781, "bow": 0.0158},
        9: {"thickness": 0.0263, "u0": 0.3357, "u1": 0.5448, "v1": 0.7862, "bow": 0.0257},
        10: {"u0": 0.5838, "u1": 0.6835, "v0": 0.7751, "v1": 0.7064, "bow": 0.0099},
        11: {"thickness": 0.0117, "u0": 0.6638, "u1": 0.7011, "v0": 0.7034, "v1": 0.7463, "bow": -0.0182},
        12: {"u0": 0.5252, "u1": 0.6522, "v0": 0.7878, "v1": 0.7270, "bow": 0.0378},
        13: {"thickness": 0.0175, "u0": 0.3045, "u1": 0.5858, "v0": 0.7557, "v1": 0.7858, "bow": 0.0324},
        14: {"u0": 0.4529, "u1": 0.5800, "v0": 0.7837, "v1": 0.7725, "bow": 0.0217},
        15: {"thickness": 0.0350, "u0": 0.3689, "u1": 0.5917, "v0": 0.7644, "v1": 0.7781, "bow": 0.0472},
        16: {"u0": 0.2713, "u1": 0.5878, "v0": 0.7407, "v1": 0.7899, "bow": 0.0467},
        17: {"thickness": 0.0263, "u0": 0.2908, "u1": 0.5761, "v0": 0.7575, "v1": 0.8047},
        18: {"thickness": 0.0263, "u0": 0.3592, "u1": 0.5351, "v1": 0.7940, "bow": 0.0310},
        19: {"thickness": 0.0263, "u0": 0.3025, "u1": 0.6190, "v1": 0.7898, "bow": 0.0318},
        20: {"thickness": 0.0263, "u0": 0.3474, "u1": 0.5526, "v0": 0.7712, "v1": 0.7938},
    },
    "lipline": {
        1: {"u0": 0.3628, "u1": 0.6300, "v": 0.3338, "bow": 0.0238},
        2: {"thickness": 0.0680, "u0": 0.3550, "u1": 0.6300, "bow": 0.0160},
        3: {"thickness": 0.0340, "u0": 0.3003, "u1": 0.6925, "v": 0.3358, "bow": 0.0316},
        4: {"thickness": 0.0907, "u0": 0.3081, "u1": 0.6769, "bow": 0.0160},
        5: {"thickness": 0.0340, "u0": 0.4175, "u1": 0.5675, "v": 0.3384, "bow": 0.0082},
        6: {"u0": 0.3550, "u1": 0.6378, "v": 0.3408, "bow": 0.0082},
    },
    "eye_gradation": {
        0: {"top": 1.0000, "bottom": 1.0000, "radial": 0.0000},
        1: {"centre": 0.5441, "half": 0.4818, "top": 0.0229, "bottom": 0.9386, "radial": 0.0000},
        2: {"centre": 0.4918, "top": 0.0000, "bottom": 1.0000, "radial": 0.0000},
        3: {"radial": 1.0000},
    },
    "eye_white": {
        0: {"cu": 0.5000, "cv": 0.6820, "ru": 0.3340, "rv": 0.2830, "shadow": 0.0000},
        1: {"cu": 0.5050, "cv": 0.7579, "ru": 0.3674, "rv": 0.3575, "shadow": 1.0000},
        2: {"cu": 0.5000, "cv": 0.6855, "ru": 0.3432, "rv": 0.3428, "shadow": 1.0000},
        3: {"cv": 0.6809, "ru": 0.4631, "rv": 0.3959, "shadow": 0.8000},
        4: {"cu": 0.5000, "cv": 0.5898, "ru": 0.4186, "rv": 0.3231, "shadow": 0.9804},
    },
    "eyeline_up": {
        0: {"a0": 199.5959, "width_scale": 1.3393},
        1: {"a0": 203.4882, "a1": 412.4276, "width_scale": 1.0909},
        3: {"a0": 199.7525, "a1": 400.3534, "width_scale": 1.3393},
        4: {"a0": 202.8295, "a1": 411.1601, "width_scale": 1.1736},
        5: {"a0": 202.5497, "a1": 415.1672},
        6: {"a0": 208.0172, "width_scale": 1.1736},
        7: {"a0": 194.1386, "a1": 420.4325},
        8: {"a0": 197.1947, "width_scale": 1.1736},
        9: {"a0": 203.2796, "a1": 397.5742, "width_scale": 0.6667},
        10: {"a0": 201.0197, "a1": 413.8185},
        11: {"a0": 195.8996, "a1": 406.0685},
        12: {"a0": 202.1584, "a1": 399.0720, "width_scale": 0.9167},
        13: {"a0": 201.1138, "a1": 403.2073, "width_scale": 1.1736},
        14: {"a1": 415.5597, "width_scale": 0.7500},
        15: {"a0": 200.7302, "a1": 412.1536, "width_scale": 1.0909},
        16: {"a0": 196.7239, "a1": 417.0457},
        17: {"a0": 190.4494, "a1": 419.0534},
        18: {"a0": 194.8268, "a1": 421.2729, "width_scale": 0.8333},
        19: {"a0": 198.2646, "a1": 427.0630},
        20: {"a0": 197.0623, "width_scale": 1.0909},
        21: {"a0": 181.9956, "a1": 414.1590, "width_scale": 0.8333},
        22: {"a0": 198.9625, "a1": 420.7327, "width_scale": 0.8333},
        23: {"a0": 182.4131, "a1": 413.4091, "width_scale": 0.6667},
        24: {"a0": 181.0995, "a1": 410.4529, "width_scale": 0.7500},
        25: {"a0": 193.0930, "width_scale": 0.7500},
        26: {"a0": 205.6819, "a1": 389.2623, "width_scale": 0.8333},
        27: {"a1": 413.0315, "width_scale": 0.6667},
        28: {"a0": 200.5442, "width_scale": 0.8333},
        29: {"a0": 199.5975, "a1": 418.7638, "width_scale": 0.8333},
        30: {"a0": 196.1729, "a1": 410.7868},
        31: {"a0": 197.3141, "width_scale": 0.7500},
        32: {"a0": 196.7599, "width_scale": 1.0909},
        33: {"a1": 409.5555, "width_scale": 0.6667},
        34: {"a0": 200.6313, "a1": 422.5609, "width_scale": 1.0909},
        35: {"a0": 200.6816, "a1": 370.7225},
        36: {"a0": 197.3554, "width_scale": 1.1736},
        37: {"a0": 197.1354, "a1": 421.5998, "width_scale": 0.6667},
        38: {"a0": 196.9662, "width_scale": 1.1736},
        39: {"a0": 195.1952, "width_scale": 1.3393},
        41: {"a0": 199.1812, "width_scale": 0.7500},
        42: {"a1": 419.3198, "width_scale": 0.7500},
        43: {"a0": 194.1540, "a1": 409.2683, "width_scale": 0.8333},
        44: {"a0": 199.3373, "a1": 415.6887},
        45: {"a0": 201.8044, "a1": 407.3118},
        46: {"a0": 195.6892, "a1": 406.9837, "width_scale": 0.6667},
        47: {"a0": 197.4892, "a1": 396.8668, "width_scale": 0.8333},
        48: {"a0": 196.6706, "a1": 405.1874},
        49: {"a1": 402.4587, "width_scale": 0.7500},
        50: {"a0": 194.9744, "a1": 402.3469},
        51: {"a0": 200.5592, "a1": 399.5145},
        52: {"a0": 194.4281, "a1": 403.8682},
        200: {"a1": 378.8032, "width_scale": 0.5000},
        201: {"a1": 382.3146, "width_scale": 0.6667},
        202: {"a1": 396.8129, "width_scale": 0.4167},
        203: {"a1": 409.3637, "width_scale": 0.5833},
        204: {"a1": 395.6776, "width_scale": 0.5000},
        205: {"a0": 180.2265, "a1": 398.5255},
        206: {"a0": 179.3157, "a1": 385.4925},
        207: {"a1": 405.2149, "width_scale": 0.5000},
    },
}


def group_of(category: str) -> str:
    """Sub-directory for *category*, or ``misc`` for one we have not placed."""
    return GROUPS.get(str(category), "misc")


SHEETS = {
    "eyebrow": eyebrow,
    "eyeline_up": eyeline_up,
    "eyeline_down": eyeline_down,
    "eyeline_shadow": eyeline_shadow,
    "lipline": lipline,
    "lip": lip,
    "cheek": cheek,
    "eyeshadow": eyeshadow,
    "mole": mole,
    "eye": iris,
    "eye_gradation": eye_gradation,
    "eye_hi_up": eye_hi_up,
    "eye_hi_down": eye_hi_down,
    "eye_white": eye_white,
    "nip": nip,
    "face_detail": face_detail,
    "body_detail": body_detail,
}

# Colour space per category, read off the stock rows' ``cs`` flags rather than
# assumed: a mask is a lerp weight and reading it in the wrong space turns byte
# 128 into 0.216 instead of 0.502 (``CLAUDE.md`` 七).
#
# **This is only the fallback.**  ``provide`` asks the bake for the *row's own*
# flag first, because the flag is per texture and a category majority is a
# guess wearing a measurement's clothes: ``eye`` is 76 sRGB / 19 linear on disk
# and the default female card uses row **10, which is linear**.  Handing that
# row back as sRGB decodes byte 163 to 0.38 instead of 0.64, and since
# ``create_eye``'s base term is ``tint * m.r`` the whole iris comes out dark —
# which is exactly what the render showed while the sheet's own radial profile
# matched Origin's to within a few counts (§7.143 九).
_SRGB = {"eyebrow", "eyeline_up", "eyeline_down", "eyeline_shadow",
         "lipline", "lip",
         "eye_white", "cheek", "eyeshadow", "eye", "eye_gradation",
         "eye_hi_up", "eye_hi_down"}
_LINEAR = {"mole", "nip", "face_detail", "body_detail"}


# Categories the generator declines even though it can draw them, so the
# caller falls through to whatever the local conversion produced.
#
# **This is a per-machine preference, not a distribution decision.**  Converted
# Origin sheets live under ``core.paths.derived_root()`` on the machine that ran
# the converter, and nothing here changes what a release may contain: the line
# in ``CLAUDE.md`` 四之一 is drawn by *content*, not by format or by how small
# a drawing is, and ``tools/release_audit.py`` still refuses to let
# ``assets_baked/`` into a build.  Deferring here is exactly the model that
# section describes — the converter runs on the user's machine and its output
# stays there.
#
# **Empty by default: we draw every sheet ourselves.**  This briefly defaulted
# to the lash line, because ours was in-family on shape (§7.148 九) and still
# did not read like Origin's.  That is a reason to finish the drawing, not to stop
# drawing it — the mechanism stays because it is genuinely a per-machine
# preference, but the product answer is our own art.
# **Empty again as of §7.166.**  §7.148 二十六 put the whole eye chain in here
# because our own eye was in-family on every statistic and still did not look
# like Origin's.  That answer does not survive a request to *change* the eye shape:
# with the chain deferred, ``iris()`` is not what the renderer draws, so
# ``aspect`` 1.556 -> 1.100 produced a **bit-identical** render — the parameter
# was not wired out of the picture, the whole sheet was.  A generator whose
# output never reaches a frame cannot be tuned by looking at frames.
_DEFER_TO_BAKE: set = {"eye", "eye_gradation", "eye_hi_up", "eye_hi_down",
                       "eye_white"}


def defer_to_bake(categories=None) -> set:
    """Read or replace the set of categories left to the local conversion.

    Passing ``None`` reads it.  ``startup.own_face_tex_defer`` overrides the
    default at load; an empty list means "draw everything ourselves".
    """
    global _DEFER_TO_BAKE
    if categories is not None:
        _DEFER_TO_BAKE = {str(c) for c in categories}
    return set(_DEFER_TO_BAKE)


# Categories taken from the local conversion but **filtered** on the way
# through: the sheet is the user's own converted copy, and this drops the parts
# of it that do not belong on the drawing.
#
# ``eyeline_up``: Origin's sheet carries the lash line *and* two or three blades
# that float clear of it at the inner corner.  Everything joined to the line is
# the lash; the detached pieces are what our own generator has never been able
# to place (§7.148 二十五).  Keeping only the component the line belongs to
# gives a usable sheet and a reference to draw against.
# **Empty as of §7.166.**  ``provide`` checks this set *first*, so while
# ``eyeline_up`` was in it our own lash generator was never called at all — the
# sheet always came from the converted Origin row with its detached pieces removed.
# Every lash change made against it (§7.148 十七, §7.165 三) was therefore
# invisible, including the three parameters §7.165 had just un-orphaned.  A
# category can be filtered from the bake or drawn by us, and this set silently
# won that argument.
#
# **``eyeline_up`` is filtered again as of §7.205, but by its own branch in
# ``provide``, not by this set** — and the difference is the failure above: that
# branch falls **through** when there is no converted row, so a machine with no
# conversion still reaches our own drawing.  Membership here does not.
_FILTER_FROM_BAKE: set = set()


def draws_nothing(category: str, ident) -> bool:
    """Is *ident* the row the list itself calls 「なし」?

    Illusion's optional layers make "no blush", "no mole", "no lip line" real
    list rows that name no asset, and it draws them as **nothing**.  This
    generator was drawing them as a whole blush, because ``_list_texture`` asks
    the provider *before* it asks the list, so whatever answers here is the
    last word.  Measured on that path with this check removed: 11,818 texels of
    blush, 14,696 of eyeshadow, 18,970 of lip, 118 of lip line and 58 of mole
    on a card that asked for none of them, against ``None`` for every one down
    the Illusion path — and 194 of 200 sampled female cards get at least one.

    It is the write side of 一之一: not "we could not find it, so here is
    something plausible" but "we were told to draw nothing and drew something".

    The answer belongs to :func:`origin.lists.draws_nothing`, which reads the
    installed lists and falls back to ``lists.NONE_IDS`` when there are none to
    read — the case F0 exists for.  Asking it rather than keeping a second copy
    here is why ``parttex`` gets the same fix from the same fact (its 「なし」
    ids are ``hairgloss`` **5** and ``nose`` **4**, so a local "id 0" rule would
    have been wrong twice), and why the **second form** of なし — a row that
    names an empty asset rather than none at all — arrives here for free.
    """
    from ..origin import lists as L
    return L.draws_nothing(str(category), ident)


def _largest_component(alpha: np.ndarray, thresh: int = 24) -> np.ndarray:
    """Mask of the biggest 8-connected blob in *alpha*.

    Iterative dilation from the largest row's seed rather than a labelling
    pass: the sheets are a few hundred kilopixels and this needs no scipy.
    """
    ink = alpha >= thresh
    if not ink.any():
        return ink
    # seed at the densest row's densest column — the lash line's thick middle
    rows = ink.sum(axis=1)
    y = int(np.argmax(rows))
    xs = np.nonzero(ink[y])[0]
    seed = np.zeros_like(ink)
    seed[y, int(xs[len(xs) // 2])] = True
    cur = seed
    for _ in range(4 * max(ink.shape)):
        nxt = cur.copy()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        nxt[1:, 1:] |= cur[:-1, :-1]
        nxt[1:, :-1] |= cur[:-1, 1:]
        nxt[:-1, 1:] |= cur[1:, :-1]
        nxt[:-1, :-1] |= cur[1:, 1:]
        nxt &= ink
        if nxt.sum() == cur.sum():
            break
        cur = nxt
    return cur


def _bake_row(category: str, ident):
    """The converted row as it was drawn, or ``None`` if it was not converted.

    Read here rather than through :func:`origin.bake.read_texture` because
    the caller *is* that function's provider — going back through it recurses.
    """
    from ..origin import bake, colorspace as CS
    try:
        entry = bake.manifest()["items"].get(bake.key_of(category, int(ident)))
    except (TypeError, ValueError):
        return None
    if not entry or entry.get("kind") != "texture" or "cs" not in entry:
        return None
    px = bake._read_png(bake.bake_dir() / entry["file"])
    if px is None or px.ndim != 3 or px.shape[2] != 4:
        return None
    return CS.Map(px, int(entry["cs"]) == CS.SRGB)


def _bake_has(category: str, ident) -> bool:
    """Is there a converted row for this id?

    The manifest entry alone — deciding whether to defer must not cost a PNG
    decode of the sheet we are about to decline to draw and hand back to
    ``read_texture`` to read for itself.
    """
    from ..origin import bake
    try:
        entry = bake.manifest()["items"].get(bake.key_of(category, int(ident)))
    except (TypeError, ValueError):
        return False
    return bool(entry) and entry.get("kind") == "texture" and "cs" in entry


def _filtered_from_bake(category: str, ident):
    """The converted sheet with its detached pieces removed, or ``None``.

    **This drops drawing, so it is only for a caller that wants the body.**
    On the lash sheets the detached pieces are not stray ink: all 62 stock rows
    carry one (median 1 piece, 4.7% of the row's ink) at 221.7-251.9 degrees and
    r 1.473-1.608 — the fine arcs riding above the crescent that §7.148 二十五
    described.  ``provide`` used to serve the filtered sheet and graft our own
    blades on; that deleted Origin's arcs and drew ours 30 degrees further into the
    corner (§7.205), which is what the report of "a line down the side of the
    eye" was.
    """
    got = _bake_row(category, ident)
    if got is None:
        return None
    from ..origin import colorspace as CS
    out = np.array(got.px, copy=True)
    keep = _largest_component(out[:, :, 3])
    out[:, :, 3] = np.where(keep, out[:, :, 3], 0)
    return CS.Map(out, got.srgb)


def _ink_edge(alpha: np.ndarray, nbins: int = 360) -> np.ndarray:
    """Outer radius of the ink, per angle, in the aperture frame.

    Indexed by ``(aa - 90) / 360 * nbins`` where ``aa`` is the same unwrapped
    angle :func:`eyeline_up` sweeps.  Angles the sheet does not ink stay at 1.0
    (the rim itself), so a lash anchored there starts at the opening.
    """
    size = alpha.shape[0]
    U, V = _grid(size, size)
    cu, cv, ru, rv = APERTURE
    x, y = (U - cu) / ru, (V - cv) / rv
    r = np.hypot(x, y)
    ang = np.degrees(np.arctan2(y, x)) % 360.0
    aa = np.where(ang < 90.0, ang + 360.0, ang)
    ink = alpha > 90
    idx = np.clip(((aa - 90.0) / 360.0 * nbins).astype(np.int32), 0, nbins - 1)
    edge = np.ones(nbins)
    np.maximum.at(edge, idx[ink], r[ink])
    # A per-degree maximum is noisy at the ends of a stroke; a short circular
    # box smooth keeps the lashes from stepping off a one-bin spike.
    k = 5
    pad = np.concatenate([edge[-k:], edge, edge[:k]])
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), "valid")


def _body_edge(alpha: np.ndarray, nbins: int = 360) -> np.ndarray:
    """Outer contour of the lash line's **body**, per angle.

    :func:`_ink_edge` on the whole sheet answers with whatever is furthest out
    at that angle, and on these rows that is the detached arcs (r 1.473-1.608)
    and the terminal slivers — so a crease placed against it jumps by up to
    0.50 of an aperture radius from one degree to the next.  The crescent
    itself is smooth: the same contour taken on the largest component alone
    steps at most 0.032 per degree over the drawn window.
    """
    keep = _largest_component(alpha)
    return _ink_edge(np.where(keep, alpha, 0), nbins)


#: Band width and edge softness for the lid crease, in aperture radii.
#:
#: The **length** of this line is a product decision (§7.180 — the family draws
#: a 28-degree smudge and the request is a crease along the whole lid).  Its
#: **weight** is not, so the width is set where the coverage lands back on the
#: family: 62 stock rows measure 0.0048 [p10 0.0032 .. p90 0.0101], and a line
#: three times longer has to be correspondingly thinner to stay there.
#: 0.045 lands coverage at **0.0072** on the stock row (0.0058 .. 0.0073 over
#: the four rows spot-checked since §7.207 moved the window inwards and made it
#: a little longer), with 71% of the band fully opaque against 3% for the ridge
#: it replaced.
SOCKET_WIDTH = 0.045
SOCKET_SOFT = 0.30
#: How far the crease is allowed to fall below its own top before it stops
#: being a crease on the lid, in fractions of the sheet, and how much of each
#: end fades out.
#:
#: **The lash's full sweep is not the upper lid** (§7.183): an arc over the
#: whole sweep hangs far enough down at both ends that it reads as a dark line
#: running down the *side* of the eye.  The criterion §7.183 states is
#: "keep the columns within a threshold of the arc's own **top**"; what it
#: stored was that criterion's answer *on one card* — the middle 0.62 of the
#: sweep, **centred on the sweep's midpoint**.  Those are not the same window,
#: because the band's top is not the sweep's midpoint: the band is held at a
#: near-constant radius, so it peaks near 270 degrees, while the sweep runs
#: further round the outer corner and its midpoint is a median **25 degrees**
#: further out (§7.207).  So the summary walked the window outwards, and the
#: family's own arc — 222.4 .. 249.3, the only place ``EyelineShadowTex`` is
#: ever drawn — fell off its inner end:
#:
#:     how much of that row's own Origin arc the window covers, 62 stock rows
#:        middle 0.62 of the sweep   all of it  0/62   none of it 32/62
#:        within this drop of the top   all of it 47/62   none of it  0/62
#:
#: The threshold is set so the card §7.183 decided the **length** on keeps it
#: (0.62 of that row's sweep is 101.8 degrees; this gives 101).  The length
#: stays a product decision (§7.180 — the family draws a 28-degree smudge and
#: the request was a crease along the whole lid); what changes is its *form*,
#: from a fraction of the sweep to a drop, and a drop is the more stable of the
#: two across rows: 76..119 degrees against 78..150 (the sweep itself runs
#: 125.6 to 241.1, so a fraction of it is a different line on every card).
#:
#: The height is read off the **lash contour**, not off the band: the band's
#: radius is chosen from the window (:func:`socket_placement`), so reading the
#: window off the band would be circular.  It costs nothing — a constant-radius
#: band peaks at 270 degrees by construction and its window has a closed form,
#: and that window's ends differ from these by a median 1.4 / 2.2 degrees.
SOCKET_DROP = 0.125
SOCKET_FEATHER = 0.06

#: How far outside the lash line the crease sits, in aperture radii, and how
#: close it is ever allowed to come — **both measured off Origin's own shadow maps**
#: (``tools/face_tex_family.py --socket``, all 62 stock rows).
#:
#: Per degree, against the same row's lash **body** contour (:func:`_body_edge`
#: — against the whole ink it would be measuring the detached arcs, which sit
#: at the same radius as the shadow), Origin's band starts
#:
#:     inner edge above the lash   p10 +0.111   median +0.178   p90 +0.291
#:     bins where it touches the lash                     0 of 2366  (0.0%)
#:
#: — a clearance, held over a short arc, never zero.  A constant radius held
#: that for Origin because its own arc is 28 degrees long over a flat stretch of
#: lid; ours runs 102 degrees (§7.180, a product decision) across a contour
#: that moves 1.30 to 1.67, so the same constant gave +0.209 at one end and
#: **-0.165** at the other — the crease crossing the lash's terminal slivers,
#: which is the report of "the socket line is still too close to the eye"
#: (§7.205).
#:
#: **Which of the two the radius is built from is decided by the band being
#: one radius** (§7.183 — see :func:`socket_placement` for why it has to be).
#: A constant can hold the *minimum* clearance over a whole lid or the median,
#: not both: built from ``SOCKET_GAP`` at the contour's middle it is inside the
#: lash on 35 of 6490 bins, built from ``SOCKET_GAP_MIN`` at the contour's
#: **peak** it touches on none and its median comes out +0.201 against the
#: family's +0.178.  So ``SOCKET_GAP`` is what the result is checked against
#: and ``SOCKET_GAP_MIN`` is what it is built from.
#:
#: Cross-check, two measurements of the same thing that were taken separately:
#: the family's per-row median ink radius is **1.594**, and holding its median
#: clearance puts our band's centre on the stock row at **1.571** (the band is
#: a curve since §7.207, r 1.487..1.594 over the window) — 0.023 of an aperture
#: radius apart, under two screen pixels at the preview's framing.  Over the
#: family's own 28-degree window the two agree more closely still, because that
#: is the stretch of lid the constant was fitted on.
SOCKET_GAP = 0.178
SOCKET_GAP_MIN = 0.111


def _lid_sweep(alpha: np.ndarray) -> tuple:
    """The angular span the lash ink covers, in the aperture polar frame.

    p3/p97 rather than the extremes: the sheets carry a few stray texels at the
    corners and the ends of this band are what a viewer reads as "where the
    crease starts".  On the stock row it comes out 218.9 to 383.6 — the whole
    upper lid — against the 28-degree arc the socket family draws.
    """
    h, w = alpha.shape[:2]
    cu, cv, ru, rv = APERTURE
    U, V = np.mgrid[0:h, 0:w]
    x, y = (V / w - cu) / ru, (U / h - cv) / rv
    ink = alpha > 32
    if ink.sum() < 64:
        return None
    ang = np.degrees(np.arctan2(y[ink], x[ink])) % 360.0
    aa = np.where(ang < 90.0, ang + 360.0, ang)
    return float(np.percentile(aa, 3)), float(np.percentile(aa, 97))


def _run_around(ok: np.ndarray, c: int):
    """The run of ``True`` containing index *c*, or ``None`` if *c* is False.

    Both windows below are runs, not thresholds applied element-wise: a crease
    has to be one arc.  A stray angle that passes the test out beyond the
    corner is not a second piece of lid.
    """
    if not ok[c]:
        return None
    lo = hi = c
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return lo, hi


def socket_placement(alpha: np.ndarray) -> tuple | None:
    """Where the crease goes on a lid whose lash sheet has this *alpha*.

    Returns ``(a0, a1, r0)`` in the aperture polar frame, or ``None`` if there
    is not enough ink to read a lid off.  Split out from :func:`_socket_on_lash`
    so the ruler that checks it against Origin measures the same arithmetic the
    generator draws.

    Two measured pieces:

    * **the lid** — the run of angles, around the lash contour's **own top**,
      over which it stays within :data:`SOCKET_DROP` of that top (§7.183's
      stated criterion; §7.207 for why the middle of the sweep is a different
      window);
    * **the radius** — **one** radius (§7.183: a band that follows the ink edge
      per angle walks into the notch between every lash spike, which is what
      "the socket line outlines the lashes" was), set to the **highest** the
      contour reaches over that window plus :data:`SOCKET_GAP_MIN`.

    That is the smallest constant that holds the family's own minimum clearance
    everywhere on this lid, and with the window moved inwards it is the choice
    that has to carry the clearance, because nothing else can.  Over 62 stock
    rows:

        radius from                 r0 median   clearance p10/med/p90   touches
        median contour + GAP        1.476       +0.093 +0.178 +0.238    36/6506
        p90 contour + GAP_MIN       1.481       +0.111 +0.178 +0.270     3/6506
        max contour + GAP_MIN       1.509       +0.123 +0.201 +0.315     0/6506
        Origin's own bands              1.594       +0.111 +0.178 +0.291     0/2366

    The first is §7.205's arithmetic, and it is the one the longer window
    breaks: the contour moves a median **0.174** of an aperture radius across a
    100-degree lid, so a radius set from its middle is inside the lash at the
    top.  §7.205 answered that by trimming the ends back to
    :data:`SOCKET_GAP_MIN`, which with this window cuts a median 19 degrees and
    one row to **11** — always off the inner end, the only place the family
    ever draws this map.  Raising the radius instead costs 0.02 of an aperture
    radius against the family's median clearance and needs no trim at all.
    """
    sweep = _lid_sweep(alpha)
    if sweep is None:
        return None
    edge = _body_edge(alpha)
    nb = len(edge)
    deg = np.arange(np.floor(sweep[0]), np.ceil(sweep[1]) + 1.0)
    here = np.interp(((deg - 90.0) / 360.0 * nb) % nb,
                     np.arange(nb), edge, period=nb)
    # Angles the body has no ink at all carry _ink_edge's 1.0 placeholder, and
    # reading a lid height off that would be a window drawn from a missing
    # sample.  Carry it across from the angles either side instead (§7.186).
    ink = here > 1.0001
    if not ink.any():
        return None
    here = np.interp(np.arange(len(deg)), np.flatnonzero(ink), here[ink])
    # Height of the lash contour, in fractions of the sheet — the units the
    # drop threshold is in.  y runs down the sheet, so up is negative.
    h = -here * np.sin(np.radians(deg)) * APERTURE[3]
    lo, hi = _run_around(h >= h.max() - SOCKET_DROP, int(np.argmax(h)))
    return (float(deg[lo]), float(deg[hi]),
            float(here[lo:hi + 1].max()) + SOCKET_GAP_MIN)


def _socket_on_lash(ident):
    """The lid shadow drawn over this character's own lid, or ``None``.

    The lash sheet is read for **where the lid is** — its sweep and its outer
    contour — off whichever sheet this character actually gets: the converted
    row where one exists, our own drawing where it does not.  A crease has to
    know that; it is per card and cannot be a constant.

    **The length is a product choice, not Origin's drawing** (§7.180): the family
    draws a 28-degree smudge over the inner half of the lid and the request was
    for a crease along the whole lid.  Where it sits across the lash is
    measured — see :func:`socket_placement`.  ``eyeline_shadow()`` called with
    its own defaults still draws the family's own short arc.
    """
    src = _bake_row("eyeline_up", ident)
    alpha = None
    if src is not None and src.px.ndim == 3 and src.px.shape[2] == 4:
        alpha = src.px[:, :, 3]
    if alpha is None or alpha.shape[0] != alpha.shape[1]:
        own = eyeline_up(**params_for("eyeline_up", ident))
        alpha = own[:, :, 3]
    placed = socket_placement(alpha)
    if placed is None:
        return None
    a0, a1, r0 = placed
    p = dict(params_for("eyeline_shadow", ident))
    for k in ("a0", "a1", "r0", "r1", "edge", "feather"):
        p.pop(k, None)
    width = p.pop("width", SOCKET_WIDTH)
    return eyeline_shadow(size=alpha.shape[0], a0=a0, a1=a1,
                          r0=r0, r1=r0 + width,
                          soft=SOCKET_SOFT, feather=SOCKET_FEATHER, **p)


def params_for(category: str, ident) -> dict:
    """Per-row parameters for *ident*, or ``{}`` for the category default.

    An id we have no measurement for — a modded slot, or a row whose grammar
    the generator does not have — gets the category default, which is still a
    whole drawing of ours.  It does **not** get interpolated or nearest-neighbour
    parameters: inventing a variant for an id nobody measured is the read side
    of ``CLAUDE.md`` 一之一.
    """
    try:
        return dict(PER_ID.get(str(category), {}).get(int(ident), {}))
    except (TypeError, ValueError):
        return {}


def provide(category: str, ident):
    """A generated sheet as a ``colorspace.Map``, or ``None``.

    Installed into :func:`origin.bake.set_texture_provider`, this is what
    lets a build with no Illusion textures still put a face on a character:
    the sheet is ours, and everything downstream — the create passes, the
    shading — was already ours.  Returning ``None`` for a category we do not
    draw is deliberate: the caller then falls through to whatever it has, and a
    missing sheet is reported rather than replaced by something invented.

    Categories in :data:`_DEFER_TO_BAKE` return ``None`` the same way, so on a
    machine that has run the converter they come from it instead.

    **``ident`` is honoured.**  It used to be accepted and dropped, so one sheet
    answered for a whole category and every character wore the same brow, the
    same lash line and the same catchlight.  That is not what the maker does:
    it offers a numbered row per slot and the card stores the one the player
    picked.  Counted over 400 local female cards those slots hold 138 distinct
    lash ids, 119 iris ids, 82 upper-catchlight ids and 61 brow ids, and the
    most common value covers only 9-26% of cards — so a single sheet per
    category put a drawing the character did not choose on three quarters of
    them.  :data:`PER_ID` carries one row's worth of measured geometry per id.
    """
    from ..origin import colorspace as CS
    from ..origin import bake
    cat = str(category)
    # 「なし」 first, before any of the drawing paths below — see
    # :func:`draws_nothing`.  Which of the two answers depends on which form
    # the row uses: a row that names **no asset** resolves to nothing, so
    # declining is what the loader wants; a row that names an **empty asset**
    # has content, and standing in for it means producing that content.
    # Declining there would leave the renderer on the prefab's own ``_MainTex``
    # — white, on a cut-out lash quad — wherever there is no conversion to fall
    # through to.
    if draws_nothing(cat, ident):
        from ..origin import lists as L
        try:
            empty = int(ident) in L.EMPTY_ASSET_IDS.get(cat, ())
        except (TypeError, ValueError):
            empty = False
        if not empty:
            return None
        return CS.Map(np.zeros((16, 16, 4), dtype=np.uint8), cat in _SRGB)
    if cat == "eyeline_up":
        # **The converted row's crescent, and nothing grafted onto it.**
        #
        # This used to hand back the filtered row with our own blades composited
        # where it was transparent.  Measured, the graft's whole visible output
        # was **one** connected piece of 587 texels at 190-225 degrees — the
        # inner corner, below the rim — because Origin's crescent covers the rest of
        # the sweep; on screen that is a line down the side of the eye, which is
        # what it was reported as.  It is not where the family puts this ink
        # either: all 62 stock rows carry a detached stroke at **221.7-251.9
        # degrees, r 1.473-1.608** (median 1 piece, 4.7% of the row's ink).
        # So the graft is gone (§7.205).
        #
        # **The filter stays, and that is a product decision, not a measurement.**
        # Serving the row *as drawn* puts those detached strokes back on the lid
        # and they read as a fork: they run 221.0-252.1 on the stock row, where
        # one is a tapering stroke and the other a constant-width band.  When
        # this was written the two merely abutted (the crease started at 250);
        # since §7.207 the crease covers **all 31.1 degrees** of them, so they
        # would now lie inside it rather than beside it — the same decision, and
        # less to argue with.  Asked which drawing to ship, the answer was one
        # clean crease, so the detached strokes are dropped here and
        # :func:`eyeline_up` draws none of its own (``inner_n`` 0).
        got = _filtered_from_bake(cat, ident)
        if got is not None:
            return got
    if cat == "eyeline_shadow" and cat not in _DEFER_TO_BAKE:
        got = _socket_on_lash(ident)
        if got is not None:
            return CS.Map(got, cat in _SRGB)
    if cat in _FILTER_FROM_BAKE:
        got = _filtered_from_bake(cat, ident)
        if got is not None:
            return got
        return None
    if cat in _DEFER_TO_BAKE and _bake_has(cat, ident):
        # **Only where there is something to defer to.**  Declining outright is
        # what the set used to do, and on a machine with no conversion — the F0
        # target, the whole reason this module exists — that left the five eye
        # categories with no map at all rather than with ours: `provide('eye')`
        # and `read_texture('eye')` both answered ``None`` against an empty bake
        # (§7.205).  The comment in ``bake._install_default_provider`` has always
        # described the conditional behaviour; now it is the behaviour.
        return None
    fn = SHEETS.get(cat)
    if fn is None:
        return None
    srgb = cat in _SRGB
    try:                      # the row's own flag beats the category default
        entry = bake.manifest()["items"].get(bake.key_of(cat, int(ident)))
        if entry and "cs" in entry:
            srgb = int(entry["cs"]) == CS.SRGB
    except Exception as exc:                                 # noqa: BLE001
        swallowed.note("mesh.facetex.provide: bake.manifest", exc)
    return CS.Map(fn(**params_for(cat, ident)), srgb)
