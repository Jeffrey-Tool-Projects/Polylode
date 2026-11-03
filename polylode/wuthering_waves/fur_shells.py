"""Wuthering Waves' parallax fur as geometry: layers of strands below the fur cards, which any format can draw.

The fur shader (``UseFurryParallax``, read from the fox tail of 心's compiled shader) fakes depth: on a fur
card — the vertices whose colour alpha is in 0.4–0.6 (``UseVertexAMask``) — it marches the view ray down from
the surface, ``MaxStep`` steps (8), through a height field — ``HeightMap``.R tiled ``HeightMapSize`` times (35)
over the card's UV — that lies ``Height`` × 0.01 UV units deep, and takes the card's look from where the ray
meets a strand (a strand of height h stands up to h of the depth from the ground).  Where no strand stands the
ray reaches the ground and the card shows as it is: the march never cuts holes, it only moves what is seen.

glTF and FBX cannot march a ray, so the same field is built: the cards are moved down to the ground (as they
are: their own coverage), and above them a layer at each step k = 1 … MaxStep − 1, k/MaxStep of the depth up,
keeps the strands taller than that (``HeightMap`` > k / MaxStep, where the card covers).  Seen head-on the
layers fall on the ground's own texels — the card as before; seen at an angle the strands stand out.  The
depth in world units is the UV depth times each vertex's world length per UV unit.  What is not built: the
height's fade at the rim (taken at 1, facing) and the flow map's sideways nudge of the strands (``FlowMapInt``,
0.1 of a step).
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


def _image(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGBA"), np.float32) / 255.0


def _sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x, y = uv[..., 0] * w - 0.5, uv[..., 1] * h - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    pick = lambda xx, yy: image[np.mod(yy, h), np.mod(xx, w)]
    return ((pick(x0, y0) * (1 - fx) + pick(x0 + 1, y0) * fx) * (1 - fy)
            + (pick(x0, y0 + 1) * (1 - fx) + pick(x0 + 1, y0 + 1) * fx) * fy)


def _layer_map(record: dict, case: Path, height_tiles: float) -> Path:
    """The base map's colour with, as alpha, each strand's height where the card covers (0 elsewhere)."""
    maps = record["maps"]
    base = _image(case / maps[record["base"]]["file"])
    heights = _image(case / maps["HeightMap"]["file"])[..., 0]
    size = min(LAYER_MAP, max(base.shape[:2]) * 4)
    v, u = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size, indexing="ij")
    uv = np.stack([u, v], axis=-1)                                     # Unreal UVs: v down, row 0 at the top
    colour = _sample(base, uv)
    strand = _sample(heights[..., None], uv * height_tiles)[..., 0]
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


def build(document: dict, arrays: dict, case: Path, log=print) -> tuple[dict, dict]:
    """*document* and *arrays* with each parallax-fur material's cards moved to its ground and its strand
    layers added above them, as a part of their own."""
    case = Path(case)
    for name, record in list(document["materials"].items()):
        if not is_fur(record):
            continue
        scalars = (record.get("native_material") or {}).get("Scalars") or {}
        steps = round(scalars.get("MaxStep", 8))
        depth = float(scalars.get("Height", 0.0)) * 0.01                # UV units (the shader's Height × 0.01)
        tiles = float(scalars.get("HeightMapSize", 1.0))
        if steps < 2 or depth <= 0:
            continue
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
                    shell["maps"][shell["base"]] = dict(record["maps"][record["base"]], file=str(layer_map.relative_to(case)))
                    shell.pop("blend", None)
                    # the layer k/steps above the ground keeps the strands taller than that
                    shell.update(cutoff=k / steps, alpha_from="a",
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
            new = len(document["parts"])
            q = f"p{new}_"
            per_vertex = [k for k in arrays if k.startswith(p) and k[len(p):] not in ("tri", "slot")
                          and not k[len(p):].startswith("s") and len(arrays[k]) == len(arrays[p + "co"])]
            layers = [k / steps - 1.0 for k in range(steps)]              # the ground (−1 of the depth), then up
            stacks = {key: [] for key in per_vertex}
            triangles, triangle_slots = [], []
            for k, lift in enumerate(layers):
                for key in per_vertex:
                    values = arrays[key][used]
                    if key == p + "co":
                        values = (values + reach * lift).astype(arrays[key].dtype)
                    stacks[key].append(values)
                triangles.append(remap[tri[chosen]] + k * len(used))
                triangle_slots.append(np.full(int(chosen.sum()), k))
            for key, values in stacks.items():
                arrays[q + key[len(p):]] = np.concatenate(values)
            arrays[q + "tri"] = np.concatenate(triangles)
            arrays[q + "slot"] = np.concatenate(triangle_slots)
            document["parts"].append(dict(part, name=part["name"] + "_FurLayers", shapes=[],
                                          materials=[name] + [f"{name}_Strands{k}" for k in range(1, steps)]))
            # the cards themselves are now the ground layer: off their own part
            arrays[p + "tri"], arrays[p + "slot"] = tri[~chosen], slots[~chosen]
            log(f"{name}：{int(chosen.sum())} 個毛髮卡片三角形做成 {steps} 層毛絲")
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
