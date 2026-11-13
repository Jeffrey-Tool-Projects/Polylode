"""Endfield's iris highlight baked into the iris's map, for the formats that cannot run the eye program.

The iris's highlight is two terms of its own program ``HGRP/CharacterNPR_Eye`` (IslandLife §7.566 read it,
§7.1490 lays it on in IslandLife's approximate shading), not another material:

* under ``_EYE_HIGHLIGHT``, the iris mesh lays small quads outside its UV disc ``|frac(uv) − ½| < ½`` over the
  highlight sprite its base map draws there; the program multiplies the base colour outside the disc by
  ``_EyeHighLightColor`` (HDR) and everywhere by ``lerp(1, _EyeScatteringColor, base.a)``;
* under ``_MATCAP_ON``, the cornea's dome normal — ``(q · −_MatcapNormalScale, √(1 − |q|²))`` in tangent space,
  ``q = frac(uv)·2 − 1`` — turned to view space reads ``_MatcapTex``, ``rgb · _MatcapColor.a + a ·
  _MatcapColor.rgb``, added under the specular light.

glTF and FBX have neither term, so they are worked out for every texel of the base map and the result written
as the map those files carry, the program's way seen head-on: the parallax walks no distance, the camera's
right and up lie along the map's u and v (the matcap's view space is the tangent space), and the lights are 1.
What a texel cannot hold above 1 is clipped and counted.  The source map stays among the maps and the material
carries ``eye`` (the terms, their values, the maps and what was assumed), for a reader that runs the program
itself (IslandLife's approximate shading does: it reads the unbaked map, ``eye.source_base``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

PROGRAM = "HGRP/CharacterNPR_Eye"
#: each keyword's values the program reads (floats, colours)
TERMS = {"_EYE_HIGHLIGHT": (("_ParallaxScale",), ("_EyeHighLightColor", "_EyeScatteringColor")),
         "_MATCAP_ON": (("_MatcapNormalScale",), ("_MatcapColor",))}
#: the map a baked iris is drawn with in glTF and FBX
BAKED_KEY = "EyeProgram_BaseColor"


class EyeError(ValueError):
    pass


def _linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def _read(path: Path, srgb: bool) -> np.ndarray:
    with Image.open(path) as opened:
        image = np.asarray(opened.convert("RGBA"), np.float64) / 255.0
    if srgb:
        image[..., :3] = _linear(image[..., :3])
    return image


def terms(record: dict) -> dict | None:
    """The program's highlight keywords on *record* and the values they read, or None for another material; a
    value the program reads that the record does not hold refuses it."""
    keywords = [key for key in TERMS if key in (record.get("keywords") or ())]
    if record.get("shader") != PROGRAM or not keywords:
        return None
    floats, colours = record.get("floats") or {}, record.get("colors") or {}
    out = {"keywords": keywords, "floats": {}, "colors": {}}
    for keyword in keywords:
        for key in TERMS[keyword][0]:
            if not isinstance(floats.get(key), (int, float)):
                raise EyeError(f"{keyword} without {key}")
            out["floats"][key] = float(floats[key])
        for key in TERMS[keyword][1]:
            if len(colours.get(key) or ()) != 4:
                raise EyeError(f"{keyword} without {key}")
            out["colors"][key] = [float(v) for v in colours[key]]
    if "_MATCAP_ON" in keywords and "_MatcapTex" not in (record.get("maps") or {}):
        raise EyeError("_MATCAP_ON without its _MatcapTex")
    return out


def bake_map(base: np.ndarray, record: dict, stated: dict, matcap: np.ndarray | None) -> tuple[np.ndarray, int]:
    """The iris's base map (H × W × 4, linear colour) with the program's terms laid on, head-on; and how many
    texels went past 1 and were clipped."""
    h, w = base.shape[:2]
    entry = record["maps"][record["base"]]
    scale = np.asarray(entry.get("scale") or (1.0, 1.0), np.float64)
    offset = np.asarray(entry.get("offset") or (0.0, 0.0), np.float64)
    # each texel's centre in the map (Unity's v up), back to the mesh's UV through the map's scale and offset
    u = (np.arange(w) + 0.5) / w
    v = 1.0 - (np.arange(h) + 0.5) / h
    uv = np.stack(np.meshgrid(u, v), axis=-1)
    uv = (uv - offset) / np.where(scale == 0, 1.0, scale)
    p = uv - np.floor(uv) - 0.5
    inside = (p * p).sum(-1) < 0.25
    colour = base[..., :3].copy()
    colours = stated["colors"]
    if "_EYE_HIGHLIGHT" in stated["keywords"]:
        highlight = np.where(inside[..., None], 1.0, np.asarray(colours["_EyeHighLightColor"][:3]))
        scattering = 1.0 + (np.asarray(colours["_EyeScatteringColor"][:3]) - 1.0) * base[..., 3:4]
        colour *= highlight * scattering
    if "_MATCAP_ON" in stated["keywords"]:
        q = (uv - np.floor(uv)) * 2.0 - 1.0
        dome = np.concatenate([q * -stated["floats"]["_MatcapNormalScale"],
                               np.sqrt(np.maximum(1.0 - np.minimum((q * q).sum(-1, keepdims=True), 1.0), 0.0))],
                              axis=-1)
        dome /= np.maximum(np.linalg.norm(dome, axis=-1, keepdims=True), 1e-12)
        at = dome[..., :2] * 0.5 + 0.5
        mh, mw = matcap.shape[:2]
        x = np.clip(at[..., 0] * mw - 0.5, 0, mw - 1)
        y = np.clip((1.0 - at[..., 1]) * mh - 0.5, 0, mh - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, mw - 1), np.minimum(y0 + 1, mh - 1)
        fx, fy = (x - x0)[..., None], (y - y0)[..., None]
        sample = ((matcap[y0, x0] * (1 - fx) + matcap[y0, x1] * fx) * (1 - fy)
                  + (matcap[y1, x0] * (1 - fx) + matcap[y1, x1] * fx) * fy)
        tint = np.asarray(colours["_MatcapColor"])
        colour += sample[..., :3] * tint[3] + sample[..., 3:4] * tint[:3]
    clipped = int((colour > 1.0).any(-1).sum())
    return np.concatenate([np.clip(colour, 0.0, 1.0), base[..., 3:4]], axis=-1), clipped


def bake(document_path: Path, log=print) -> Path:
    """A copy of the scene document at *document_path* whose irises read their maps with the program's highlight
    terms baked in (glTF and FBX); the document itself when it has none.  The baked maps go beside the originals."""
    document_path = Path(document_path)
    folder = document_path.parent
    document = json.loads(document_path.read_text(encoding="utf-8"))
    drawn = {name for part in document["parts"] if not part.get("hidden") for name in part["materials"]}
    changed = False
    for name, record in document["materials"].items():
        if name not in drawn or not record.get("base"):
            continue
        try:
            stated = terms(record)
        except EyeError as refusal:
            record["eye"] = {"game": "endfield", "status": None, "reason": f"{refusal}: the map is written unbaked"}
            log(f"{name}：{refusal}，眼睛照原底色圖輸出")
            changed = True
            continue
        if stated is None:
            continue
        source = record["base"]
        entry = record["maps"][source]
        base = _read(folder / entry["file"], bool(entry.get("srgb", True)))
        matcap = None
        if "_MATCAP_ON" in stated["keywords"]:
            m = record["maps"]["_MatcapTex"]
            matcap = _read(folder / m["file"], bool(m.get("srgb")))
        colour, clipped = bake_map(base, record, stated, matcap)
        target = Path(entry["file"]).with_name(f"{Path(entry['file']).stem}_{name}_eyeprogram.png")
        rgba = np.concatenate([_srgb(colour[..., :3]), colour[..., 3:]], axis=-1)
        Image.fromarray(np.round(rgba * 255).astype(np.uint8), "RGBA").save(folder / target)
        record["maps"][BAKED_KEY] = {k: v for k, v in entry.items() if k != "sha256"} | {"file": target.as_posix(),
                                                                                         "srgb": True}
        record["base"] = BAKED_KEY
        maps = {"source_base": source} | ({"matcap": "_MatcapTex"} if matcap is not None else {})
        record["eye"] = {"game": "endfield", "status": "baked", "program": PROGRAM, **stated, "maps": maps,
                         "source_base": source, "baked_base": BAKED_KEY,
                         "baked": ("the program's highlight sprite tint (_EYE_HIGHLIGHT) and matcap (_MATCAP_ON) "
                                   "laid on the base map for glTF and FBX"),
                         "view": ("head-on: no parallax, the camera's right and up along the map's u and v, "
                                  "the diffuse and specular light 1"),
                         "clipped_texels": clipped}
        log(f"{name}：虹膜的高光貼片與 matcap 烘進底色圖（{clipped} 個貼圖像素超過 1 被截斷）")
        changed = True
    if not changed:
        return document_path
    out_path = document_path.with_name(document_path.stem + "_eyebaked.json")
    out_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return out_path
