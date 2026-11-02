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


def bake(document_path: Path, log=print) -> Path:
    """A copy of the scene document at *document_path* whose eye materials read their maps with the eye blend
    layers baked in; the document itself when it has no eye layer.  The baked maps go beside the originals."""
    document_path = Path(document_path)
    folder = document_path.parent
    document = json.loads(document_path.read_text(encoding="utf-8"))
    materials = document["materials"]
    eyes = {n for n, m in materials.items() if m.get("shader") == EYE and m.get("base")}
    layers = {n: m for n, m in materials.items() if m.get("shader") in (EYE_ADD, EYE_MULTIPLY) and m.get("base")}
    if not eyes or not layers:
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

    for eye in sorted(n for n in eyes if n in surfaces):
        record = materials[eye]["maps"][materials[eye]["base"]]
        base = image_of(record)
        size = (base.shape[1], base.shape[0])
        rows, columns, points = _eye_texels(*surfaces[eye], record, size)
        if not len(points):
            continue
        colour = _linear(base[rows, columns, :3])
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
                colour[near] = colour[near] + value * strength          # the main light's term taken as 1
            else:
                colour[near] = colour[near] * (1.0 + strength * (value - 1.0))
            log(f"{eye}: {name} over {int(near.sum())} of {len(points)} texels")
        # a texel the two eyes share is the mean of what each gives
        out = base.copy()
        flat = rows * size[0] + columns
        totals = np.zeros((size[0] * size[1], 3))
        counts = np.zeros(size[0] * size[1])
        np.add.at(totals, flat, _srgb(colour))
        np.add.at(counts, flat, 1)
        done = counts > 0
        out.reshape(-1, 4)[done, :3] = totals[done] / counts[done, None]
        source = Path(record["file"])
        baked = source.with_name(f"{source.stem}_eyelayers{source.suffix if source.suffix == '.png' else '.png'}")
        Image.fromarray(np.round(out * 255).astype(np.uint8), "RGBA").save(folder / baked)
        materials[eye] = dict(materials[eye], maps=dict(materials[eye]["maps"],
                                                       **{materials[eye]["base"]: dict(record, file=baked.as_posix())}),
                              eye_layers_baked=sorted(order))
    out_path = document_path.with_name(document_path.stem + "_eyebaked.json")
    out_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return out_path
