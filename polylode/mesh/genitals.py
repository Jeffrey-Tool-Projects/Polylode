"""Procedural genital meshes for the humanoid ``cf_j_*`` rig.

Anime-readable but anatomically suggested: female vulva (labia, cleft, clitoris
with hood, vestibule) and male shaft (tapered tube, corona, glans, scrotum).
Sized and placed relative to head-scale so they stay outside the torso shell.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import math3d as m3
from . import detail, primitives as prim
from .mesh import Material, Mesh
from ..core import swallowed


def _sv(p: dict, key: str, default: float = 0.5) -> float:
    try:
        return float(p.get(key, default))
    except (TypeError, ValueError):
        return default


def enabled(params: dict) -> bool:
    try:
        return int(params.get("body.genitals", 1)) != 0
    except (TypeError, ValueError):
        return True


def skin_tint(params: dict) -> tuple:
    c = params.get("body.skin_color") or (0.96, 0.78, 0.72)
    return tuple(float(x) for x in c[:3])


def _mucosa_mat(params: dict, name: str = "mucosa", deep: float = 0.0) -> Material:
    base = np.array(skin_tint(params), dtype=np.float64)
    pink = base * np.array([1.05, 0.62 - 0.12 * deep, 0.70 - 0.08 * deep])
    pink = np.clip(pink, 0.0, 1.0)
    return Material(
        name=name,
        base_color=(*pink, 1.0),
        shadow_color=(pink[0] * 0.72, pink[1] * 0.5, pink[2] * 0.55, 1.0),
        specular_strength=0.55 + 0.15 * deep,
        specular_power=56.0,
        rim_strength=0.28,
        outline_width=0.45,
        cast_outline=True,
        two_sided=True,
        shade_feather=0.06,
    )


def _skin_mat(params: dict, name: str = "genital_skin") -> Material:
    base = np.array(skin_tint(params), dtype=np.float64)
    return Material(
        name=name,
        base_color=(*base, 1.0),
        shadow_color=(base[0] * 0.78, base[1] * 0.7, base[2] * 0.72, 1.0),
        specular_strength=0.28,
        specular_power=72.0,
        rim_strength=0.22,
        outline_width=0.5,
    )


def _place_on_bone(mesh: Mesh, skel, bone: str, local_offset=(0, 0, 0),
                   local_euler_deg=(0, 0, 0)) -> Mesh:
    if not skel.has(bone):
        bone = "cf_j_hips"
    i = skel.id(bone)
    W = skel.bones[i].bind_world
    rx, ry, rz = (math.radians(float(a)) for a in local_euler_deg)
    R = m3.euler_yxz(rx, ry, rz)
    T = m3.translate(*local_offset)
    mesh.transform(W @ T @ R)
    mesh.set_bone(i)
    return mesh


def _skin_chain(mesh: Mesh, skel, bones: list[str]) -> Mesh:
    ids = [skel.id(b) for b in bones if skel.has(b)]
    if not ids:
        mesh.set_bone(skel.id("cf_j_hips") if skel.has("cf_j_hips") else 0)
        return mesh
    if len(ids) == 1:
        mesh.set_bone(ids[0])
        return mesh
    origins = np.array([skel.bones[i].bind_world[:3, 3] for i in ids], dtype=np.float64)
    p = mesh.positions.astype(np.float64)
    a, b = origins[0], origins[-1]
    ab = b - a
    ab2 = float(np.dot(ab, ab)) + 1e-12
    t = np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0)
    n = len(ids)
    mesh.bone_idx[:] = 0
    mesh.bone_wt[:] = 0.0
    for vi in range(len(p)):
        u = float(t[vi]) * (n - 1)
        i0 = int(min(n - 2, max(0, math.floor(u))))
        f = u - i0
        mesh.bone_idx[vi, 0] = ids[i0]
        mesh.bone_idx[vi, 1] = ids[i0 + 1]
        mesh.bone_wt[vi, 0] = 1.0 - f
        mesh.bone_wt[vi, 1] = f
    return mesh.normalize_weights()


def _paint_vertex_colors(mesh: Mesh, colors_rgb: np.ndarray, a: float = 1.0) -> Mesh:
    """Per-vertex RGB → mesh.colors (shader multiplies by uBaseColor)."""
    n = mesh.vertex_count
    c = np.ones((n, 4), dtype=np.float32)
    c[:, :3] = np.asarray(colors_rgb, dtype=np.float32).reshape(n, 3)
    c[:, 3] = a
    mesh.colors = c
    return mesh


def _vulva_pad(length: float, half_w: float, thick: float,
               segs_y: int = 28, segs_x: int = 22) -> tuple[Mesh, np.ndarray]:
    """Continuous soft mound with a vertical cleft — illustration-style unit.

    Returns (mesh in local space, per-vertex pink weights 0..1 for colouring).
    Local: +Y superior (clit end), -Y inferior (anus end), +Z out of body.
    """
    def fn(U, V):
        # U across X (0 left → 1 right), V along Y (0 superior → 1 inferior)
        x = (U * 2.0 - 1.0) * half_w
        y = (0.5 - V) * length
        # outer silhouette: rounded diamond / almond (anime crotch)
        nx = np.abs(x) / max(half_w, 1e-6)
        ny = np.abs(y) / max(length * 0.5, 1e-6)
        edge = np.clip(1.0 - (nx ** 1.6 + ny ** 1.8), 0.0, 1.0)
        edge = edge ** 0.85
        # central cleft: deep groove on midline
        groove = np.exp(-((x / max(half_w * 0.22, 1e-6)) ** 2))
        # slightly deeper mid-cleft (vaginal zone ~ V 0.35–0.65)
        zone = np.exp(-(((V - 0.48) / 0.22) ** 2))
        z = thick * (0.55 * edge + 0.45 * edge ** 1.4)
        z = z - thick * 0.55 * groove * (0.45 + 0.55 * zone)
        # gentle front curve
        z = z + thick * 0.08 * (1.0 - ny)
        return np.stack([x, y, z], axis=-1)

    mesh = prim.surface(fn, segs_x, segs_y, wrap_u=False, wrap_v=False, name="vulva_pad")
    mesh = prim.ensure_outward(mesh, (0, 0, thick * 0.2))
    # pink weight: higher in the groove / inner cleft
    P = mesh.positions.reshape(segs_x, segs_y, 3)
    x = P[:, :, 0]
    y = P[:, :, 1]
    groove = np.exp(-((x / max(half_w * 0.28, 1e-6)) ** 2))
    ny = np.abs(y) / max(length * 0.5, 1e-6)
    pink_w = np.clip(groove * (0.55 + 0.45 * (1.0 - ny * 0.5)), 0.0, 1.0)
    return mesh, pink_w.reshape(-1)


def _inner_petals(length: float, half_w: float, thick: float,
                  segs_y: int = 20, segs_x: int = 12) -> tuple[Mesh, np.ndarray]:
    """Inner labia as a continuous soft V-shape inside the cleft."""
    def fn(U, V):
        t = V  # 0 top → 1 bottom
        # width grows then tapers
        span = half_w * (0.25 + 0.75 * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.9)
        x = (U * 2.0 - 1.0) * span
        y = (0.5 - t) * length
        # petals sit proud of the groove floor
        nx = np.abs(U * 2.0 - 1.0)
        z = thick * (0.35 + 0.45 * (1.0 - nx) + 0.2 * np.sin(np.pi * t))
        # meet at top (clit) and bottom
        taper = np.clip(np.sin(np.pi * t), 0.05, 1.0)
        x = x * taper
        return np.stack([x, y, z], axis=-1)

    mesh = prim.surface(fn, segs_x, segs_y, wrap_u=False, wrap_v=False, name="inner_petals")
    mesh = prim.ensure_outward(mesh, (0, 0, thick * 0.3))
    pink = np.full(mesh.vertex_count, 0.85, dtype=np.float64)
    return mesh, pink


def _clitoris_glans(rx: float, ry: float, rz: float, segs: int = 18) -> Mesh:
    """The glans — generated, like the rest of this module."""
    def fn(U, V):
        th = U * 2.0 * math.pi
        ph = V * math.pi
        x = rx * np.sin(ph) * np.cos(th)
        y = ry * np.cos(ph) * 0.9
        z = rz * np.sin(ph) * np.sin(th) * 0.75 + rz * 0.4 * np.sin(ph)
        return np.stack([x, y, z], axis=-1)

    m = prim.surface(fn, segs, max(10, segs // 2), wrap_u=True, wrap_v=False, name="clitoris")
    return prim.ensure_outward(m, (0, 0, rz * 0.25))


def _orifice(rx: float, ry: float, depth: float, segs: int = 18) -> Mesh:
    """Dark oval opening (vaginal) — rim + recessed centre."""
    def fn(U, V):
        th = U * 2.0 * math.pi
        r = V  # 0 centre → 1 rim
        x = rx * r * np.cos(th)
        y = ry * r * np.sin(th)
        # dish inward (−Z into body)
        z = -depth * (1.0 - r * r) ** 0.9
        return np.stack([x, y, z], axis=-1)

    m = prim.surface(fn, segs, max(8, segs // 2), wrap_u=True, wrap_v=False, name="orifice")
    return prim.ensure_outward(m, (0, 0, -0.1 * depth))


def _anus_pucker(R: float, r: float, segs: int = 16) -> Mesh:
    return _entrance_ring(R, r, segs)


# Female layout — spread-leg illustration order (superior → inferior):
#   clitoris → vaginal opening → anus
# Keep Z offsets small: snap already puts kokan on the skin surface; large
# z0 made the whole stack float in front of the body.
def female_layout(s: float, size: float = 1.0) -> dict:
    z0 = 0.0035 * s
    return {
        "z0": z0,
        "pad": (0.0, -0.002 * s, z0),
        "inner": (0.0, -0.003 * s, z0 + 0.0015 * s),
        "clitoris": (0.0, 0.0, 0.0008 * s),
        "entrance": (0.0, -0.005 * s, z0 + 0.002 * s),
        "anus": (0.0, 0.0, 0.0005 * s),
        "pad_len": 0.048 * s * size,
        "pad_w": 0.020 * s * size,
        "pad_t": 0.010 * s * size,
    }


# --------------------------------------------------------------------------
# female
# --------------------------------------------------------------------------
def build_female(params: dict, skel, prop) -> list[tuple[str, Mesh, Material]]:
    """Higher-detail anime vulva (still procedural — not a 2D painting)."""
    s = prop.landmarks()["s"]
    size = 0.95 + 0.55 * _sv(params, "body.genital_size", 0.55)
    # more segments than body clothes — silhouette is close-up critical
    segs = detail.n(28, 16)
    lay = female_layout(s, size)
    out = []

    skin = np.array(skin_tint(params), dtype=np.float64)
    pink_soft = np.clip(skin * np.array([1.04, 0.78, 0.82]), 0, 1)
    pink_hot = np.clip(np.array([0.96, 0.48, 0.58]), 0, 1)
    pink_deep = np.clip(np.array([0.72, 0.28, 0.38]), 0, 1)

    # --- continuous outer pad (labia majora + mons blend) ---
    pad, pink_w = _vulva_pad(lay["pad_len"], lay["pad_w"], lay["pad_t"],
                             segs_y=segs, segs_x=max(18, segs - 4))
    cols = pink_soft[None, :] * (1.0 - pink_w[:, None]) + pink_hot[None, :] * pink_w[:, None]
    _paint_vertex_colors(pad, cols)
    _place_on_bone(pad, skel, "cf_j_kokan", local_offset=lay["pad"])
    mat_pad = _skin_mat(params, "vulva_pad")
    mat_pad.base_color = (1.0, 1.0, 1.0, 1.0)  # vertex colour carries tint
    mat_pad.specular_strength = 0.42
    mat_pad.shade_feather = 0.08
    out.append(("gen_vulva", pad, mat_pad))

    # --- inner petals (labia minora) ---
    inn, pw = _inner_petals(lay["pad_len"] * 0.72, lay["pad_w"] * 0.38,
                            lay["pad_t"] * 0.45,
                            segs_y=max(16, segs - 6), segs_x=max(12, segs // 2))
    cols_i = pink_hot[None, :] * 0.55 + pink_deep[None, :] * 0.45
    cols_i = np.broadcast_to(cols_i, (inn.vertex_count, 3)).copy()
    _paint_vertex_colors(inn, cols_i)
    _place_on_bone(inn, skel, "cf_j_kokan", local_offset=lay["inner"])
    mat_i = _mucosa_mat(params, "inner", 0.35)
    mat_i.base_color = (1.0, 1.0, 1.0, 1.0)
    mat_i.specular_strength = 0.7
    out.append(("gen_inner", inn, mat_i))

    # --- vaginal orifice (mid-cleft, insertion target) ---
    orifice = _orifice(0.0050 * s * size, 0.0075 * s * size, 0.0055 * s * size,
                       segs=segs)
    cols_o = np.broadcast_to(pink_deep, (orifice.vertex_count, 3)).copy()
    # rim lighter
    r = np.linalg.norm(orifice.positions[:, :2], axis=1)
    r = r / (r.max() + 1e-9)
    cols_o = cols_o * (0.65 + 0.35 * r[:, None]) + pink_hot * (0.35 * (1 - r[:, None]))
    _paint_vertex_colors(orifice, cols_o)
    _place_on_bone(orifice, skel, "cf_j_kokan", local_offset=lay["entrance"])
    mat_e = _mucosa_mat(params, "entrance", 0.7)
    mat_e.base_color = (1.0, 1.0, 1.0, 1.0)
    mat_e.specular_strength = 0.55
    out.append(("gen_entrance", orifice, mat_e))

    # --- clitoris ---
    cs = 0.9 + 0.45 * _sv(params, "body.clitoris_size", 0.55)
    crx = 0.0038 * s * size * cs
    cry = 0.0030 * s * size * cs
    crz = 0.0036 * s * size * cs
    clit = _clitoris_glans(crx, cry, crz, segs=max(16, segs - 4))
    cols_c = np.broadcast_to(
        np.clip(pink_hot * 1.08, 0, 1), (clit.vertex_count, 3)).copy()
    _paint_vertex_colors(clit, cols_c)
    _place_on_bone(clit, skel, "cf_j_clitoris",
                   local_offset=(0.0, 0.0, 0.0003 * s))
    mat_c = _mucosa_mat(params, "clitoris", 0.1)
    mat_c.base_color = (1.0, 1.0, 1.0, 1.0)
    mat_c.specular_strength = 0.85
    out.append(("gen_clitoris", clit, mat_c))

    # --- anus: small pucker inferior to vagina ---
    anus_R = 0.0036 * s * size
    anus = _anus_pucker(anus_R, anus_R * 0.36, segs=max(14, segs - 6))
    cols_a = np.broadcast_to(pink_deep * 0.9, (anus.vertex_count, 3)).copy()
    _paint_vertex_colors(anus, cols_a)
    anus_bone = "cf_j_anus" if skel.has("cf_j_anus") else "cf_j_kokan"
    anus_off = (0.0, 0.0, 0.001 * s) if anus_bone == "cf_j_anus" else (
        0.0, -0.024 * s, lay["pad"][2] * 0.4)
    _place_on_bone(anus, skel, anus_bone, local_offset=anus_off,
                   local_euler_deg=(90.0, 0.0, 0.0))
    mat_a = _mucosa_mat(params, "anus", 0.75)
    mat_a.base_color = (1.0, 1.0, 1.0, 1.0)
    out.append(("gen_anus", anus, mat_a))

    return out


def entrance_local_offset(skel, prop=None) -> np.ndarray:
    """Kokan-local insertion target — vaginal opening, not clitoris or anus."""
    if prop is not None:
        s = prop.landmarks()["s"]
    else:
        s = 1.0
        if skel is not None and skel.has("cf_j_hips"):
            s = float(skel.bones[skel.id("cf_j_hips")].bind_world[1, 3]) / 0.88
            s = max(0.7, min(1.4, s))
    return np.asarray(female_layout(s)["entrance"], dtype=np.float64)


def _entrance_ring(R: float, r: float, segs: int) -> Mesh:
    segs = max(8, segs)
    rings = max(8, segs // 2)

    def fn(U, V):
        theta = U * 2.0 * math.pi
        phi = V * 2.0 * math.pi
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        return np.stack([x, y, z], axis=-1)

    return prim.ensure_outward(
        prim.surface(fn, segs, rings, wrap_u=True, wrap_v=True, name="entrance"),
        (0, 0, 0))


# --------------------------------------------------------------------------
# male
# --------------------------------------------------------------------------
def build_male(params: dict, skel, prop) -> list[tuple[str, Mesh, Material]]:
    """Single penis mesh only (shaft + integrated tip) — no scrotum / corona / extras."""
    s = prop.landmarks()["s"]
    size = 0.9 + 0.5 * _sv(params, "body.genital_size", 0.55)
    chain = 0.110 * s
    length = chain * (0.92 + 0.25 * _sv(params, "body.penis_length", 0.5)) * size / 1.15
    length = max(0.08 * s, min(0.16 * s, length))
    r0 = 0.0135 * s * size * (0.85 + 0.35 * _sv(params, "body.penis_girth", 0.5))
    segs = detail.even(16, 10)

    # one continuous tube: base → shaft → rounded tip (no separate glans part)
    n_pts = 16
    zs = np.linspace(0.0, length, n_pts)
    radii = []
    for z in zs:
        t = z / max(length, 1e-6)
        r = r0 * (1.02 - 0.10 * t + 0.06 * math.sin(t * math.pi))
        # soft tip taper (last 15%)
        if t > 0.85:
            u = (t - 0.85) / 0.15
            r *= max(0.15, 1.0 - 0.75 * u * u)
        radii.append(r)
    pts = np.stack([np.zeros(n_pts), np.zeros(n_pts), zs], axis=1)
    pts[:, 1] = -0.010 * s * (zs / max(length, 1e-6)) ** 1.5
    squash = np.full(n_pts, 0.94)
    penis = prim.tube_along(pts, radii, sides=segs, name="penis",
                            squash=squash, caps=True, up_hint=(0, 1, 0))
    if skel.has("cf_j_dan00"):
        W = skel.bones[skel.id("cf_j_dan00")].bind_world
        penis.transform(W)
        _skin_chain(penis, skel, ["cf_j_dan00", "cf_j_dan01", "cf_j_dan02"])
    else:
        _place_on_bone(penis, skel, "cf_j_hips")
    return [("gen_penis", penis, _skin_mat(params, "penis"))]


def build_junction(params: dict, skel, prop, sex: int = 0) -> list[tuple[str, Mesh, Material]]:
    """No extra junction ring — female entrance / male penis already define contact."""
    return []


# --------------------------------------------------------------------------
def attach_genitals(built) -> None:
    p = built.params
    if not enabled(p):
        return
    skel = built.skeleton
    if skel is None or len(skel) < 4:
        return
    # drop any previous genital parts (e.g. re-attach after glTF swap)
    for part in list(built.model.by_group("genitals")):
        built.model.remove(part.name)
    sex = int(getattr(built, "sex", 0) or 0)
    prop = built.prop
    pieces = build_female(p, skel, prop) if sex == 0 else build_male(p, skel, prop)
    pieces += build_junction(p, skel, prop, sex)
    for name, mesh, mat in pieces:
        if mesh.vertex_count == 0:
            continue
        mesh.compute_normals()
        part = built.model.add_mesh(name, mesh, mat, group="genitals", layer=3)
        part.dynamic = False
        part.skinned = True
    for part in built.model.by_group("genitals"):
        part.visible = False


def _is_male_genital_part(part) -> bool:
    """Origin ``origin_o_dankon`` / ``origin_o_dan_f`` or legacy procedural ``gen_penis``."""
    if part is None or part.group != "genitals" or part.mesh is None:
        return False
    n = str(part.name or "").lower()
    return any(k in n for k in (
        "dankon", "dan_f", "gen_penis", "penis", "o_dan", "gomu",
    )) and "mnp" not in n  # never female pads; the condom shows only while worn (``set_condom``)


def strip_to_penis_only(built) -> None:
    """Hide every part except male penis meshes (Free H default for actor)."""
    if built is None or built.model is None:
        return
    for part in built.model.parts:
        keep = _is_male_genital_part(part)
        part.visible = keep
        part.dynamic = False
        if keep:
            part.skinned = True


def set_character_visible(built, visible: bool) -> None:
    """Show/hide every draw part on *built* (e.g. hide male on female solo H)."""
    if built is None or built.model is None:
        return
    on = bool(visible)
    for part in built.model.parts:
        part.visible = on
        if on and part.group == "genitals":
            # leave genital default off until undress / Free H forces them
            pass


def apply_male_h_display(built, *, mode: str = "genitals") -> None:
    """Free H male draw mode: ``hidden`` | ``genitals`` | ``full``.

    * ``hidden`` — female solo / lesbian rows: no male mesh at all.
    * ``genitals`` — default dual H: only ``o_dankon`` / glans family.
    * ``full`` — whole body (clothes still controlled separately).
    """
    if built is None or built.model is None:
        return
    m = (mode or "genitals").lower()
    if m == "hidden":
        set_character_visible(built, False)
        return
    if m == "genitals":
        strip_to_penis_only(built)
        return
    # full
    for part in built.model.parts:
        if part.group == "genitals":
            continue
        part.visible = True
        part.skinned = True
    set_genitals_visible(built, True)


def set_genitals_visible(built, visible: bool) -> None:
    """Show/hide genital *shell* parts (``o_mnpa`` / ``o_mnpb`` / ``o_dankon``…).

    Origin does **not** drive a per-frame “vagina open” morph on these meshes
    (§7.121).  Female uncensor puts the real crotch on ``o_body_a``; vanilla
    ``o_mnpa``/``o_mnpb`` stay ~48/69-vert censor stubs.  Forcing those stubs
    visible over an uncensor body seals the open body geometry.  When the body
    already has uncensor density, keep the tiny female stubs hidden.
    """
    if built is None or built.model is None:
        return
    show = bool(visible)
    if show:
        # Re-bind the body's **current** albedo onto the Origin genital parts: they
        # were handed its store key at attach time and have to follow every
        # re-store or they resolve to a 4×4 white (§7.104).
        #
        # This used to call ``restore_body_skin``, which also puts the *unmasked*
        # sheet on the body — and this runs every frame from Free H, so wearing
        # a top became impossible: ``refit_body_mask`` cut the holes and the
        # next tick washed them out.  Showing genitals says nothing about
        # whether a garment is cutting the skin; that is the top's business
        # (§7.133).
        try:
            from ..origin.figure.body_head import sync_genital_skin
            sync_genital_skin(built)
        except Exception as exc: # noqa: BLE001
            swallowed.note("mesh.genitals.set_genitals_visible: sync_genital_skin", exc)
    # Uncensor body density floor (same constant as body bake acceptance).
    # one owner for the number (a second literal here was a copy that could drift)
    from ..origin.figure.body_head import UNCENSOR_BODY_A_MIN
    body_min = int(UNCENSOR_BODY_A_MIN)
    body = built.model.get("body") if hasattr(built.model, "get") else None
    body_n = (body.mesh.vertex_count
              if body is not None and body.mesh is not None else 0)
    uncensor_body = body_n >= body_min
    sex = int(getattr(built, "sex", 0) or 0)

    for part in built.model.by_group("genitals"):
        on = show
        # Female vanilla stubs over uncensor body → hide (crotch is on body).
        if (on and uncensor_body and sex == 0 and part.mesh is not None
                and part.mesh.vertex_count < 200
                and ("mnpa" in part.name or "mnpb" in part.name)):
            on = False
        part.visible = on
        part.dynamic = False
        part.skinned = True
        if on and part.material is not None:
            part.material.alpha_test = 0.0


def _store_rest_poses(built) -> None:
    """Cache bind positions so opening morph can restore each frame."""
    for part in built.model.by_group("genitals"):
        if part.mesh is None:
            continue
        if getattr(part, "_rest_pos", None) is None:
            part._rest_pos = part.mesh.positions.copy()
            part._rest_nrm = (part.mesh.normals.copy()
                              if part.mesh.normals is not None else None)


def apply_genital_opening(built, depth: float, wave: float = 0.5) -> None:
    """Legacy vertex push for **procedural** ``gen_*`` shells only.

    Origin has no matching IL: no body blendshape open, no H-clip labia curves,
    no per-frame morph on ``o_mnpa``/``o_mnpb`` (§7.121).  Free H parts are
    named ``origin_o_mnpa``/``origin_o_mnpb`` and this function leaves them unchanged
    (measured delta 0).  Uncensor open look is the **body** mesh, not this.
    """
    if built is None or built.model is None:
        return
    if int(getattr(built, "sex", 0) or 0) != 0:
        return
    depth = float(np.clip(depth, 0.0, 1.0))
    wave = float(np.clip(wave, 0.0, 1.0))
    # visible gape: mostly depth, boosted mid-to-deep thrust
    open_t = float(np.clip(depth * (0.35 + 0.65 * wave), 0.0, 1.0))
    # mild idle parting when connected at all
    if depth > 0.05:
        open_t = max(open_t, 0.12 + 0.2 * depth)

    _store_rest_poses(built)
    s = 1.0
    if getattr(built, "prop", None) is not None:
        s = float(built.prop.landmarks().get("s", 1.0) or 1.0)

    for part in built.model.by_group("genitals"):
        rest = getattr(part, "_rest_pos", None)
        if rest is None or part.mesh is None:
            continue
        p = rest.astype(np.float64).copy()
        c = p.mean(axis=0)
        name = part.name
        # Origin bake names never match — intentional no-op (§7.121).

        if name == "gen_entrance":
            # dilate the orifice in the pad plane (X/Y), recess deeper (−Z)
            sx = 1.0 + 2.2 * open_t
            sy = 1.0 + 1.4 * open_t
            p[:, 0] = c[0] + (p[:, 0] - c[0]) * sx
            p[:, 1] = c[1] + (p[:, 1] - c[1]) * sy
            p[:, 2] = c[2] + (p[:, 2] - c[2]) * (1.0 + 0.35 * open_t) - 0.004 * s * open_t

        elif name == "gen_inner":
            # labia minora peel outward
            lat = np.sign(p[:, 0] - c[0])
            lat[lat == 0] = 1.0
            p[:, 0] += lat * (0.010 * s * open_t) * (
                0.4 + 0.6 * np.clip(1.0 - np.abs(p[:, 1] - c[1]) / (0.03 * s + 1e-9), 0, 1))
            p[:, 2] += 0.002 * s * open_t

        elif name == "gen_vulva":
            # majora part along midline groove
            lat = np.sign(p[:, 0] - c[0])
            lat[lat == 0] = 1.0
            # stronger open near the central cleft (small |x|)
            mid = np.exp(-((p[:, 0] - c[0]) / (0.012 * s + 1e-9)) ** 2)
            p[:, 0] += lat * (0.007 * s * open_t) * (0.35 + 0.65 * mid)
            p[:, 2] -= 0.0015 * s * open_t * mid  # deepen cleft

        elif name == "gen_clitoris":
            # hood retracts superiorly as the pad opens (not the glans “opening”)
            p[:, 1] += 0.004 * s * open_t
            p[:, 2] += 0.001 * s * open_t

        part.mesh.positions = p.astype(np.float32)
        part.mesh.compute_normals()
        part.dynamic = False
        part.skinned = True


def freeze_cloth_parts(built) -> None:
    """Force clothes/hair/import bottoms back to rigid skeleton skinning."""
    if built is None or built.model is None:
        return
    for part in built.model.parts:
        if part.group in ("clothes", "hair", "genitals"):
            part.dynamic = False
            if part.group == "clothes":
                part.skinned = True


def drive_genital_pose(pose: dict, *, sex: int, depth: float, wave: float,
                       erect: float = 1.0, angle: float = 0.0) -> dict:
    depth = float(np.clip(depth, 0.0, 1.0))
    wave = float(np.clip(wave, 0.0, 1.0))
    erect = float(np.clip(erect, 0.0, 1.0))
    if sex == 1:
        # Origin shaft is ``cm_J_dan100/101 → 109``; procedural ``cf_j_dan*`` is gone.
        base_pitch = math.radians(14.0 * (1.0 - erect))
        shaft_bend = math.radians(2.5 * depth * (wave - 0.5) * 2.0)
        tip_bend = math.radians(1.5 * depth * wave)
        ang = math.radians(angle * 6.0)
        q_base = m3.quat_from_euler(base_pitch, ang, 0.0)
        q_shaft = m3.quat_from_euler(shaft_bend, 0.0, 0.0)
        q_tip = m3.quat_from_euler(tip_bend, 0.0, 0.0)
        for name, q in (
                ("cm_J_dan100_00", q_base),
                ("cm_J_dan101_00", q_shaft),
                ("cm_J_dan109_00", q_tip),
                ("cf_j_dan00", q_base),
                ("cf_j_dan01", q_shaft),
                ("cf_j_dan02", q_tip),
        ):
            pose[name] = {"rot": q}
    else:
        # Mild pelvis/clit tilt only — no labia bone offsets (BP open experiments
        # warped the mesh and were withdrawn).
        open_ = math.radians(8.0 * depth + 6.0 * wave)
        pose["cf_j_kokan"] = {"rot": m3.quat_from_euler(open_ * 0.35, 0.0, 0.0)}
        pose["cf_j_clitoris"] = {
            "rot": m3.quat_from_euler(-open_ * 0.15, 0.0, 0.0)}
    return pose
