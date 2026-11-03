"""Wuthering Waves' parallax fur as geometry: layers of strands above the fur cards, which any format can draw.

The fur shader (``UseFurryParallax``, read from the base pass of 心's fox tail) fakes depth on a fur card — the
vertices whose colour alpha is in 0.4–0.6 (``UseVertexAMask``).  The card is the ground: the strands of a height
field — ``HeightMap``.R tiled ``HeightMapSize`` (T) times over the card's UV — stand up to ``Height`` × 0.01 UV
units above it.  The shader walks the view ray down from that height to the card in n steps (n =
lerp(``MaxStep``, ``MinStep``, |N·V|): ``MinStep`` seen head-on); at layer height t the ray is at uv + P·t,
P = (V·T, V·B) / (V·N) × depth, and the first layer where a strand stands taller than t (height × the rim fade)
gives the pixel its look.  Where none does, the ray reaches the card and it shows as it is.

Two things move the strands as they rise:

* **the rim fade** — every height is scaled by smoothstep(``ParallaxRimFadeMin``, ``ParallaxRimFadeMax``, N·V):
  seen at a grazing angle the strands sink toward the card;
* **the flow map** — at step j (layer height t = 1 − j/n, j ≥ 1) both the height and the colour are looked up
  at an offset of m_j × (``FlowmapMap``.RG × 2 − 1) × ``FlowMapInt`` / n height-map tiles, the flow map read at
  ``FlowmapScale`` × uv, and m_j = ``FurRootInv`` × (n + 2 − 2(j − 1)) + (j − 1): the strands lean along the flow.

glTF and FBX cannot march a ray, so the same field is built: the card stays, and above it a layer at each height
t = k/n (k = 1 … n − 1) keeps the strands taller than t (``<base>_strands.png``: the base colour with the strand
height as alpha where the card covers; layer k cuts at k/n).  The flow's lean is each layer's UVs moved by its
offset at that vertex (exact where the flow map is smooth across a triangle).  The rim fade depends on the view:
each layer records it (``fur_rim_fade``) and the ``.blend`` applies it in its shader; glTF and FBX draw the
strands at their full height from every side.  The depth in world units is the UV depth times each vertex's
world length per UV unit.  A material with ``Plane`` ≠ 0 walks differently and is left as it is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

#: the vertex colour alpha band of the cards the march runs on
BAND = (0.4, 0.6)
#: the layer map's size: the height field tiles HeightMapSize times over it
LAYER_MAP = 4096


def is_fur(record: dict) -> bool:
    native = record.get("native_material") or {}
    return ((native.get("Switches") or {}).get("UseFurryParallax") is True and "HeightMap" in record.get("maps", {})
            and record.get("base") is not None)


def _image(path: Path, srgb: bool = False) -> np.ndarray:
    with Image.open(path) as opened:
        data = np.asarray(opened.convert("RGBA"), np.float32) / 255.0
    if srgb:                                           # an sRGB texture is read linear on the GPU
        rgb = data[..., :3]
        data[..., :3] = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return data


def _sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear, wrapping, at Unreal UVs (v down, row 0 at the top)."""
    h, w = image.shape[:2]
    x, y = uv[..., 0] * w - 0.5, uv[..., 1] * h - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]

    def pick(xx, yy):
        return image[np.mod(yy, h), np.mod(xx, w)]
    return ((pick(x0, y0) * (1 - fx) + pick(x0 + 1, y0) * fx) * (1 - fy)
            + (pick(x0, y0 + 1) * (1 - fx) + pick(x0 + 1, y0 + 1) * fx) * fy)


def _layer_map(record: dict, case: Path, tiles: float) -> Path:
    """The base map's colour with, as alpha, each strand's height where the card covers (0 elsewhere)."""
    maps = record["maps"]
    base = _image(case / maps[record["base"]]["file"])
    heights = _image(case / maps["HeightMap"]["file"], bool(maps["HeightMap"].get("srgb")))[..., 0]
    size = min(LAYER_MAP, max(base.shape[:2]) * 4)
    v, u = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size, indexing="ij")
    uv = np.stack([u, v], axis=-1)
    colour = _sample(base, uv)
    strand = _sample(heights[..., None], uv * tiles)[..., 0]
    alpha = np.where(colour[..., 3] >= 0.5, np.maximum(strand, 0.0), 0.0)
    layer = np.concatenate([colour[..., :3], alpha[..., None]], axis=-1)
    folder = case / "fur"
    folder.mkdir(exist_ok=True)
    target = folder / f"{Path(maps[record['base']]['file']).stem}_strands.png"
    Image.fromarray(np.round(np.clip(layer, 0, 1) * 255).astype(np.uint8), "RGBA").save(target)
    return target


def _world_per_uv(co: np.ndarray, uv: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Each vertex's world length per UV unit: the mean over its triangles of edge length ratios."""
    total, count = np.zeros(len(co)), np.zeros(len(co))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        world = np.linalg.norm(co[tri[:, a]] - co[tri[:, b]], axis=1)
        texture = np.linalg.norm(uv[tri[:, a]] - uv[tri[:, b]], axis=1)
        ratio = np.divide(world, texture, out=np.zeros_like(world), where=texture > 1e-9)
        for corner in (a, b):
            np.add.at(total, tri[:, corner], ratio)
            np.add.at(count, tri[:, corner], ratio > 0)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def lean(step: int, steps: int, root_inverted: float) -> float:
    """m_j: how many flow offsets the look-up at step *step* (≥ 1) is moved by."""
    return root_inverted * (steps + 2 - 2 * (step - 1)) + (step - 1)


def build(document: dict, arrays: dict, case: Path, log=print) -> tuple[dict, dict]:
    """*document* and *arrays* with each parallax-fur material's strand layers added above its cards, as a part
    of their own."""
    case = Path(case)
    for name, record in list(document["materials"].items()):
        if not is_fur(record):
            continue
        scalars = (record.get("native_material") or {}).get("Scalars") or {}
        if scalars.get("Plane", 0.0) != 0.0:
            log(f"{name}：Plane 不是 0 的視差毛髮走法不同，沒有讀過，毛絲層不做")
            continue
        steps = int(np.floor(scalars.get("MinStep", 8.0)))             # seen head-on
        depth = float(scalars.get("Height", 0.0)) * 0.01                # UV units (the shader's Height × 0.01)
        tiles = float(scalars.get("HeightMapSize", 1.0))
        flow_scale = float(scalars.get("FlowmapScale", 1.0))
        flow_strength = float(scalars.get("FlowMapInt", 0.0))
        root_inverted = float(scalars.get("FurRootInv", 0.0))
        fade = (float(scalars.get("ParallaxRimFadeMin", 0.0)), float(scalars.get("ParallaxRimFadeMax", 1.0)))
        if steps < 2 or depth <= 0 or tiles <= 0:
            continue
        flow = None
        if flow_strength and "FlowmapMap" in record["maps"]:
            flow_map = record["maps"]["FlowmapMap"]
            flow = _image(case / flow_map["file"], bool(flow_map.get("srgb")))[..., :2] * 2.0 - 1.0
        layer_map = None
        for index, part in enumerate(list(document["parts"])):
            if name not in part["materials"]:
                continue
            p = f"p{index}_"
            if p + "col" not in arrays:
                continue
            slot = part["materials"].index(name)
            tri, slots = arrays[p + "tri"], arrays[p + "slot"]
            alpha = arrays[p + "col"][:, 3]
            card = (alpha >= BAND[0]) & (alpha <= BAND[1])
            chosen = (slots == slot) & card[tri].all(axis=1)
            if not chosen.any():
                continue
            if layer_map is None:
                layer_map = _layer_map(record, case, tiles)
                for k in range(1, steps):
                    shell = dict(record, maps=dict(record["maps"]), base=record["base"] + "_Strands")
                    shell["maps"][shell["base"]] = dict(record["maps"][record["base"]],
                                                        file=layer_map.relative_to(case).as_posix())
                    shell.pop("blend", None)
                    # the layer k/n above the card keeps the strands taller than that
                    shell.update(cutoff=k / steps, alpha_from="a", fur_rim_fade=list(fade),
                                 coverage=f"fur strand layer {k} of {steps}: the strands taller than {k / steps:.3f}")
                    document["materials"][f"{name}_Strands{k}"] = shell
            uv0 = arrays[p + "uv0"].astype(np.float64) * [1, -1] + [0, 1]   # Unreal UVs, as the depth is
            scale = _world_per_uv(arrays[p + "co"].astype(np.float64), uv0, tri[chosen])
            used = np.unique(tri[chosen])
            remap = np.full(len(arrays[p + "co"]), -1)
            remap[used] = np.arange(len(used))
            normal = arrays[p + "no"][used].astype(np.float64)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            reach = (depth * scale[used])[:, None] * normal              # the whole depth, along the normal
            # the flow offset per vertex, in height-map tiles per unit of m_j
            offset = (np.zeros((len(used), 2)) if flow is None
                      else _sample(flow, uv0[used] * flow_scale) * flow_strength / steps)
            q = f"p{len(document['parts'])}_"
            per_vertex = [key for key in arrays if key.startswith(p) and key[len(p):] not in ("tri", "slot")
                          and not key[len(p):].startswith("s") and len(arrays[key]) == len(arrays[p + "co"])]
            stacks = {key: [] for key in per_vertex}
            triangles, triangle_slots = [], []
            for k in range(1, steps):
                shift = offset * lean(steps - k, steps, root_inverted) / tiles       # base UV units, v down
                for key in per_vertex:
                    values = arrays[key][used]
                    if key == p + "co":
                        values = (values + reach * (k / steps)).astype(arrays[key].dtype)
                    elif key == p + "uv0":
                        values = (values + shift * [1, -1]).astype(arrays[key].dtype)   # back to v up
                    stacks[key].append(values)
                triangles.append(remap[tri[chosen]] + (k - 1) * len(used))
                triangle_slots.append(np.full(int(chosen.sum()), k - 1))
            for key, values in stacks.items():
                arrays[q + key[len(p):]] = np.concatenate(values)
            arrays[q + "tri"] = np.concatenate(triangles)
            arrays[q + "slot"] = np.concatenate(triangle_slots)
            document["parts"].append(dict(part, name=part["name"] + "_FurLayers", shapes=[],
                                          materials=[f"{name}_Strands{k}" for k in range(1, steps)]))
            log(f"{name}：{int(chosen.sum())} 個毛髮卡片三角形做成 {steps - 1} 層毛絲")
    return document, arrays


def apply(case: Path, log=print) -> None:
    """The scene in *case* with its parallax fur built as layers (the document and arrays rewritten)."""
    case = Path(case)
    document = json.loads((case / "source_document.json").read_text(encoding="utf-8"))
    if not any(is_fur(r) for r in document["materials"].values()):
        return
    arrays = dict(np.load(case / document["arrays"]))
    document, arrays = build(document, arrays, case, log)
    np.savez_compressed(case / document["arrays"], **arrays)
    (case / "source_document.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
