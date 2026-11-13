"""GF2's eye blend layers baked into the eye's own map, for the formats that cannot multiply or add a layer onto
what is behind it (glTF, FBX; owner 2026-11-02, §7.1455).

The layers' rules were read from their shaders (``export.coverage_fields``): ``eyeblend_add`` adds the layer
(its map times ``_MainColor``, at ``_SpecularIntensity``, times the main light's 0.2 + 0.8 · N·L) and
``eyeblend_multiply`` multiplies by lerp(white, its map times ``_MainColor``, ``_SpecularIntensity``); the add
layer draws before the multiply one (queues +458, +459), both over the opaque eye (+14).  Here every texel of
the eye's map that its triangles cover is taken to the eye's surface, the closest point of each layer's
surface is found there (the layers are shells just in front of it), the layer's map is read at that point,
and the two rules are applied in that order, in linear light.  The main light's term depends on a scene, so
the add layer is taken at its fully lit value — as the ``.blend`` takes it.  Texels no layer lies over are
left as they were; a texel the two eyes share is the mean of what each gives.

The ``.blend`` keeps the layers as layers (it can blend them); this writes a copy of the scene document whose
eye material reads the baked map, for the glTF and FBX writers only.

The eye's own program also lights a specular map (``_Specularmap``; 5 of the 168 exported eyes set one, the
shader's default is black) by N·L: it is laid on fully lit as well (IslandLife §7.1491).  Each eye keeps its
unbaked map under its own name and says, in ``eye``, the program's terms and the add and multiply layers as maps on
its own UV, for a reader that lights the eye itself (the add layer by 0.2 + 0.8 · N·L, not by a ramp).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .export import EYE, EYE_ADD, EYE_MULTIPLY

#: how far in front of the eye a layer may lie and still be the eye's (the layers are shells a fraction of a
#: millimetre out; a surface further away is another part of the face)
REACH = 0.003


def _linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def _texels(uv: np.ndarray, record: dict, size: tuple[int, int], wrap: bool = True) -> np.ndarray:
    """Unity UVs (v up) through the map's scale and offset, as fractional pixel positions (row 0 at the top);
    without *wrap* they run past the map's edge, for a triangle to be rasterised whole."""
    scale = np.asarray(record.get("scale") or (1.0, 1.0), np.float64)
    offset = np.asarray(record.get("offset") or (0.0, 0.0), np.float64)
    at = uv * scale + offset
    if wrap:
        at = np.mod(at, 1.0)
    return np.stack([at[:, 0] * size[0], (1.0 - at[:, 1]) * size[1]], axis=1)


def _surfaces(document: dict, arrays, names: set[str]) -> dict[str, tuple]:
    """Each material in *names*: its triangles' corners (3D, metres) and UVs, over every shown part."""
    found: dict[str, list] = {name: [] for name in names}
    for index, part in enumerate(document["parts"]):
        if part.get("hidden"):
            continue
        p = f"p{index}_"
        tri, slot = arrays[p + "tri"], arrays[p + "slot"]
        for number, material in enumerate(part["materials"]):
            if material in found:
                chosen = tri[slot == number]
                if len(chosen):
                    found[material].append((arrays[p + "co"][chosen], arrays[p + "uv0"][chosen]))
    return {name: (np.concatenate([c for c, _ in parts]), np.concatenate([u for _, u in parts]))
            for name, parts in found.items() if parts}


def _closest(points: np.ndarray, corners: np.ndarray, tree: cKDTree, k: int = 8):
    """For each point, the nearest of its *k* nearest triangles (by centroid): distance and barycentrics."""
    k = min(k, len(corners))
    _d, near = tree.query(points, k=k)
    near = near.reshape(len(points), k)
    best = np.full(len(points), np.inf)
    best_bary = np.zeros((len(points), 3))
    best_tri = np.zeros(len(points), np.int64)
    for column in range(k):
        a, b, c = (corners[near[:, column], i] for i in range(3))
        # the point's projection on the triangle's plane, its barycentrics clamped back onto the triangle
        ab, ac, ap = b - a, c - a, points - a
        d00, d01, d11 = (ab * ab).sum(1), (ab * ac).sum(1), (ac * ac).sum(1)
        d20, d21 = (ap * ab).sum(1), (ap * ac).sum(1)
        denominator = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-20, 1e-20, d00 * d11 - d01 * d01)
        v = (d11 * d20 - d01 * d21) / denominator
        w = (d00 * d21 - d01 * d20) / denominator
        bary = np.clip(np.stack([1 - v - w, v, w], axis=1), 0, None)
        bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
        at = bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:] * c
        distance = np.linalg.norm(points - at, axis=1)
        better = distance < best
        best[better], best_bary[better], best_tri[better] = distance[better], bary[better], near[better, column]
    return best, best_bary, best_tri


def _sample(image: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The image (H × W × C, floats) at fractional pixel positions, bilinear, wrapping."""
    h, w = image.shape[:2]
    x, y = at[:, 0] - 0.5, at[:, 1] - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    pick = lambda xx, yy: image[np.mod(yy, h), np.mod(xx, w)]
    return ((pick(x0, y0) * (1 - fx) + pick(x0 + 1, y0) * fx) * (1 - fy)
            + (pick(x0, y0 + 1) * (1 - fx) + pick(x0 + 1, y0 + 1) * fx) * fy)


def _eye_texels(corners: np.ndarray, uvs: np.ndarray, record: dict, size: tuple[int, int]):
    """Every texel centre of the eye's map inside one of its triangles: (row, column) and its 3D point."""
    rows, columns, points = [], [], []
    for tri3, triuv in zip(corners, uvs):
        px = _texels(triuv, record, size, wrap=False)          # whole; its texels wrap below
        x0, x1 = int(np.floor(px[:, 0].min())), int(np.ceil(px[:, 0].max()))
        y0, y1 = int(np.floor(px[:, 1].min())), int(np.ceil(px[:, 1].max()))
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        centres = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
        a, b, c = px
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (centres[:, 0] - c[0]) + (c[0] - b[0]) * (centres[:, 1] - c[1])) / denominator
        l2 = ((c[1] - a[1]) * (centres[:, 0] - c[0]) + (a[0] - c[0]) * (centres[:, 1] - c[1])) / denominator
        l3 = 1 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not inside.any():
            continue
        weights = np.stack([l1, l2, l3], axis=1)[inside]
        columns.append(np.mod(xs.ravel()[inside], size[0]))
        rows.append(np.mod(ys.ravel()[inside], size[1]))
        points.append(weights @ tri3)
    if not points:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3))
    return np.concatenate(rows), np.concatenate(columns), np.concatenate(points)


#: the maps a baked eye adds beside its own (which stays under its name, the program's unbaked map)
BAKED_KEY, ADD_KEY, MULTIPLY_KEY = "EyeLayers_BaseColor", "EyeLayers_Add", "EyeLayers_Multiply"
#: the eye program's terms, read from its compiled fragment (IslandLife §7.1491; ``gf_shader/pbr/character/eye``,
#: pass 0, the variant without keywords, UnityPerMaterial by byte offset): what a reader that lights the eye
#: itself needs, and what the formats without the program are given instead
PROGRAM_TERMS = {
    "diffuse": "_MainTex × (the ambient's colour at N, scaled to the ambient's mean luminance + _MainLightColor × "
               "lerp(_ShadowIntensity, 1, saturate(N·L))): no ramp, no cast shadow",
    "specular": "_Specularmap × _SpecularIntensity × saturate(N·L) × _MainLightColor (the shader's default map is "
                "black: no term without one)",
    "final": "× _FinalTint",
    "add_layer": "eyeblend_add (Blend One One, queue +458): its map × _MainColor × _SpecularIntensity × "
                 "(0.2 + 0.8 × saturate(N·L) × the main light's shadow) — the light's direction, not its colour",
    "multiply_layer": "eyeblend_multiply (Blend DstColor Zero, queue +459): lerp(1, its map × _MainColor, "
                      "_SpecularIntensity), unlit",
    "parallax": "the vertex program's offset × _CorneaParallax (base) and × _SpecularParallax (specular): not "
                "worked out here (head-on)"}


def _uv_of(rows, columns, record: dict, size) -> np.ndarray:
    """The mesh's UV (Unity's, v up) at texel centres of a map read through *record*'s scale and offset."""
    scale = np.asarray(record.get("scale") or (1.0, 1.0), np.float64)
    offset = np.asarray(record.get("offset") or (0.0, 0.0), np.float64)
    at = np.stack([(columns + 0.5) / size[0], 1.0 - (rows + 0.5) / size[1]], axis=1)
    return (at - offset) / np.where(scale == 0, 1.0, scale)


def _program(eye: dict, image_of, rows, columns, size) -> tuple[np.ndarray | None, dict]:
    """The eye program's specular term at the eye's texels (fully lit, head-on) and the values it reads."""
    floats, colours = eye.get("floats") or {}, eye.get("colors") or {}
    stated = {"_ShadowIntensity": floats.get("_ShadowIntensity"), "_SpecularIntensity": floats.get("_SpecularIntensity"),
              "_FinalTint": colours.get("_FinalTint")}
    spec = None
    if "_Specularmap" in eye.get("maps", {}):
        record = eye["maps"]["_Specularmap"]
        image = image_of(record)
        at = _texels(_uv_of(rows, columns, eye["maps"][eye["base"]], size), record, (image.shape[1], image.shape[0]))
        spec = _linear(_sample(image, at)[:, :3]) * float(floats.get("_SpecularIntensity", 1.5))
    return spec, stated


def bake(document_path: Path, log=print) -> Path:
    """A copy of the scene document at *document_path* whose eye materials read their maps with the eye program's
    specular map (fully lit) and the eye blend layers baked in; the document itself when it has no eye.  The
    baked maps go beside the originals; each eye keeps its own map under its name and states, in ``eye``, the
    program's terms and the maps a reader that lights the eye itself needs: the add layer's colour (at the
    scale ``add_scale``) and the multiply layer's factor, both on the eye's own UV."""
    document_path = Path(document_path)
    folder = document_path.parent
    document = json.loads(document_path.read_text(encoding="utf-8"))
    materials = document["materials"]
    eyes = {n for n, m in materials.items() if m.get("shader") == EYE and m.get("base")}
    layers = {n: m for n, m in materials.items() if m.get("shader") in (EYE_ADD, EYE_MULTIPLY) and m.get("base")}
    if not eyes:
        return document_path
    arrays = np.load(folder / document["arrays"])
    surfaces = _surfaces(document, arrays, eyes | set(layers))
    # draw order: the add layer (+458) before the multiply one (+459)
    order = sorted((n for n in layers if n in surfaces), key=lambda n: materials[n]["shader"] != EYE_ADD)
    trees = {n: cKDTree(surfaces[n][0].mean(axis=1)) for n in order}
    images: dict[Path, np.ndarray] = {}

    def image_of(record: dict) -> np.ndarray:
        path = (folder / record["file"]).resolve()
        if path not in images:
            with Image.open(path) as opened:
                images[path] = np.asarray(opened.convert("RGBA"), np.float64) / 255.0
        return images[path]

    changed = False
    for eye in sorted(n for n in eyes if n in surfaces):
        source = materials[eye]["base"]
        record = materials[eye]["maps"][source]
        base = image_of(record)
        size = (base.shape[1], base.shape[0])
        rows, columns, points = _eye_texels(*surfaces[eye], record, size)
        if not len(points):
            continue
        colour = _linear(base[rows, columns, :3])
        spec, program = _program(materials[eye], image_of, rows, columns, size)
        if spec is not None:
            colour = colour + spec                                   # the main light's terms taken as 1
        added = np.zeros_like(colour)
        factor = np.ones_like(colour)
        laid = []
        for name in order:
            layer = materials[name]
            corners, uvs = surfaces[name]
            distance, bary, tri = _closest(points, corners, trees[name])
            near = distance <= REACH
            if not near.any():
                continue
            layer_record = layer["maps"][layer["base"]]
            layer_image = image_of(layer_record)
            layer_uv = (bary[near, :, None] * uvs[tri[near]]).sum(axis=1)
            at = _texels(layer_uv, layer_record, (layer_image.shape[1], layer_image.shape[0]))
            value = _linear(_sample(layer_image, at)[:, :3]) * _linear(np.asarray(layer.get("tint", [1, 1, 1])[:3]))
            strength = float(layer.get("strength", 1.0))
            if layer["shader"] == EYE_ADD:
                # every add layer draws before every multiply one (+458, +459): the adds sum, then the factors apply
                added[near] += value * strength
            else:
                factor[near] *= 1.0 + strength * (value - 1.0)
            laid.append({"material": name, "blend": "add" if layer["shader"] == EYE_ADD else "multiply",
                         "strength": strength, "tint": [float(v) for v in layer.get("tint", [1, 1, 1, 1])],
                         "texels": int(near.sum())})
            log(f"{eye}: {name} over {int(near.sum())} of {len(points)} texels")
        baked = (colour + added) * factor                            # the add layer's light term taken as 1
        flat = rows * size[0] + columns
        done, mean = _mean(flat, size, _srgb(baked))
        out = base.copy()
        out.reshape(-1, 4)[done, :3] = mean
        stem = Path(record["file"]).stem
        target = Path(record["file"]).with_name(f"{stem}_eyelayers.png")
        Image.fromarray(np.round(out * 255).astype(np.uint8), "RGBA").save(folder / target)
        entries = {BAKED_KEY: dict(record, file=target.as_posix(), srgb=True)}
        statement = {"game": "gf2", "status": "baked", "program": EYE, "terms": PROGRAM_TERMS, "values": program,
                     "source_base": source, "baked_base": BAKED_KEY, "layers": laid,
                     "baked": ("the program's specular map (fully lit) and the layers (the add layer's light term "
                               "1) laid on the eye's map for glTF and FBX"),
                     "maps": {"source_base": source} | ({"specular": "_Specularmap"} if spec is not None else {})}
        if laid:
            # the layers on the eye's own UV, for a reader that lights the add layer itself
            scale = max(1.0, float(added.max()))
            _done, add_mean = _mean(flat, size, _srgb(added / scale))
            _done, mul_mean = _mean(flat, size, np.clip(factor, 0.0, 1.0))
            add_map = np.zeros((size[1] * size[0], 4))
            add_map[:, 3] = 1.0
            add_map[done, :3] = add_mean
            mul_map = np.ones((size[1] * size[0], 4))
            mul_map[done, :3] = mul_mean
            for key, image, srgb in ((ADD_KEY, add_map, True), (MULTIPLY_KEY, mul_map, False)):
                path = Path(record["file"]).with_name(f"{stem}_{key.lower()}.png")
                pixels = np.round(image.reshape(size[1], size[0], 4) * 255).astype(np.uint8)
                Image.fromarray(pixels, "RGBA").save(folder / path)
                entries[key] = {k: v for k, v in record.items() if k != "sha256"} | {"file": path.as_posix(),
                                                                                     "srgb": srgb}
            statement["maps"].update(add=ADD_KEY, multiply=MULTIPLY_KEY)
            statement["add_scale"] = scale
            statement["multiply_clipped_texels"] = int(((factor < 0.0) | (factor > 1.0)).any(1).sum())
        materials[eye] = dict(materials[eye], base=BAKED_KEY, maps=dict(materials[eye]["maps"], **entries),
                              eye=statement, **({"eye_layers_baked": sorted(r["material"] for r in laid)}
                                                if laid else {}))
        changed = True
    if not changed:
        return document_path
    out_path = document_path.with_name(document_path.stem + "_eyebaked.json")
    out_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return out_path


def _mean(flat: np.ndarray, size, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A texel the two eyes (or two levels of detail) share is the mean of what each gives: the texels written
    (a mask over the flattened map) and their values."""
    totals = np.zeros((size[0] * size[1], values.shape[1]))
    counts = np.zeros(size[0] * size[1])
    np.add.at(totals, flat, values)
    np.add.at(counts, flat, 1)
    done = counts > 0
    return done, totals[done] / counts[done, None]
