"""Source material descriptors for the shared Qt character preview.

Imported Qt models follow the glTF scene's depth-first node order. The QML
adapter checks every primitive count before replacing any material.
"""
from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

STYLES = (("studio", "攝影棚"), ("endfield", "終末地角色著色"),
          ("gf2", "少女前線 2 角色著色"), ("wuwa", "鳴潮角色著色"))
LIMITS = "角色著色預覽改編；不含遊戲場景後製、投影與眼睛／頭髮專用效果"
WUWA_PARAMETERS = {"backShadowOffset": ("BackShadowProcessOffset", -.1),
                   "frontShadowOffset": ("FrontShadowProcessOffset", .4),
                   "shadowProcess": ("ShadowProcess", .5), "shadowWidth": ("ShadowWidth", .01),
                   "pbrRate": ("PBRRateIntensity", 1.), "specularPower": ("SpecularPower", 5.),
                   "metalSpecularPower": ("MetalSpecularPower", 5000.)}


def descriptors(path: Path) -> list:
    try:
        return _descriptors(path)
    except (OSError, ValueError, KeyError, IndexError, struct.error):
        # Ordinary FBX/OBJ and third-party glTF still use their imported materials.
        return []


def _descriptors(path: Path) -> list:
    if path.suffix.lower() not in (".glb", ".gltf"):
        return []
    from .gltf_runtime import mesh_nodes
    document, blob, materials = read_materials(path)
    if any("NORMAL" not in primitive["attributes"]
           for mesh in document.get("meshes", []) for primitive in mesh.get("primitives", [])):
        return []
    return [[materials[p["material"]] if "material" in p else {}
             for p in document["meshes"][document["nodes"][node]["mesh"]]["primitives"]]
            for node in mesh_nodes(document, blob)]


def read_materials(path: Path, texture_dir: Path | None = None) -> tuple[dict, bytes, list]:
    from .gltf_runtime import read
    document, blob = read(path)

    def texture(entry):
        if not entry:
            return ""
        if "texture" in entry:
            image = document["images"][document["textures"][entry["texture"]]["source"]]
            if "uri" in image:
                uri = image["uri"]
                return uri if uri.startswith("data:") else (path.parent / uri).resolve().as_uri()
            if "bufferView" in image and blob:
                view = document["bufferViews"][image["bufferView"]]
                start = view.get("byteOffset", 0)
                pixels = blob[start:start + view["byteLength"]]
                from ..core.paths import derived_root
                folder = texture_dir if texture_dir is not None else derived_root() / "assets_cache" / "extractor" / "shader-textures"
                folder.mkdir(parents=True, exist_ok=True)
                suffix = ".png" if image["mimeType"] == "image/png" else ".jpg"
                target = folder / (hashlib.sha256(pixels).hexdigest() + suffix)
                if not target.exists():
                    temporary = target.with_suffix(f".{os.getpid()}.tmp")
                    temporary.write_bytes(pixels)
                    temporary.replace(target)
                return target.resolve().as_uri()
        candidate = path.parent / entry.get("file", "")
        return candidate.resolve().as_uri() if candidate.is_file() else ""

    materials = []
    for index, material in enumerate(document.get("materials", [])):
        native = material.get("extras", {})
        pbr = material.get("pbrMetallicRoughness", {})
        base_texture = pbr.get("baseColorTexture", {})
        transform = base_texture.get("extensions", {}).get("KHR_texture_transform", {})
        values = native.get("floats", {})
        maps = native.get("maps", {})
        shader = native.get("shader", "")
        family = ("endfield" if shader.startswith("HGRP/") else "gf2" if shader.startswith("gf_shader/")
                  else "wuwa" if native.get("native_material") else "")
        ramp = maps.get("_DiffRampMap" if family == "endfield" else "_RampMap")
        if (family == "endfield" and "_DIFF_RAMP_ON" not in native.get("keywords", [])) \
                or (family == "gf2" and values.get("_UseRampMap", 0) <= 0):
            ramp = None
        scalars = native.get("native_material", {}).get("Scalars", {})
        surface = maps.get("_MetallicGlossMap" if family == "endfield" else "_RMOTex")
        if family == "endfield" and "_METALLICSPECGLOSSMAP" not in native.get("keywords", []):
            surface = None
        materials.append({"id": index, "name": material.get("name", str(index)),
                          "baseColor": pbr.get("baseColorFactor", [1, 1, 1, 1]),
                          "alphaMode": material.get("alphaMode", "OPAQUE"),
                          "alphaCutoff": material.get("alphaCutoff", .5),
                          "doubleSided": material.get("doubleSided", False),
                          "preserveOriginal": ("KHR_materials_unlit" in material.get("extensions", {})
                                               or shader.endswith("_VFX") or "/vfx/" in shader.lower()),
                          "family": family, "ramp": texture(ramp),
                          "rampSrgb": bool(ramp and ramp.get("srgb")),
                          "surfaceSrgb": bool(surface and surface.get("srgb")),
                          "shadowLutSrgb": bool(maps.get("_ShadowLutTex", {}).get("srgb")),
                          "baseTexture": texture({"texture": base_texture["index"]}) if "index" in base_texture else "",
                          "baseUv": [*transform.get("scale", [1, 1]), *transform.get("offset", [0, 0])],
                          "baseRotation": transform.get("rotation", 0),
                          "shadowLut": texture(maps.get("_ShadowLutTex")) if "_SHADOW_LUT_TEX" in native.get("keywords", []) else "",
                          **{key: scalars.get(prop, default) for key, (prop, default) in WUWA_PARAMETERS.items()},
                          "surface": texture(surface),
                          "emission": values.get("_EmissiveIntensity", 0.),
                          "shadowBrightness": values.get("_ShadowColorBrightness", .5),
                          "shadowSaturation": values.get("_ShadowColorSaturation", 1.),
                          "smoothness": values.get("_Smoothness", .5) if family == "endfield" else 1 - pbr.get("roughnessFactor", .5),
                          "metallic": values.get("_Metallic", 0.) if family == "endfield" else pbr.get("metallicFactor", 0.),
                          "specularLevel": values.get("_Specular", 0.)})
    return document, blob, materials


def check_styles(view, output: Path | None = None) -> dict:
    """Exercise the real shader pipeline with animation held, without a window."""
    import time
    import numpy as np
    from PySide6.QtCore import qInstallMessageHandler
    from PySide6.QtGui import QImage
    messages = []
    previous = qInstallMessageHandler(lambda _kind, _context, text: messages.append(text))
    images = {}
    stack = [view.root]
    while stack:
        item = stack.pop()
        if item.metaObject().className() == "QQuickTimelineAnimation":
            item.setProperty("running", False)
        stack.extend(child for child in item.children()
                     if child.metaObject().className() not in ("QQuickKeyframeGroup", "QQuickKeyframe"))
    try:
        for style in [key for key, _label in STYLES] + ["studio"]:
            view.root.setProperty("characterStyle", style)
            start, stable, last = time.monotonic(), 0, ""
            while time.monotonic() - start < 5:
                image = view.image().convertToFormat(QImage.Format_RGBA8888)
                pixels = np.asarray(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
                digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                stable = stable + 1 if digest == last else 0
                last = digest
                if stable >= 3 and time.monotonic() - start > .5:
                    break
                time.sleep(.02)
            record = {"sha256": digest, "painted": int(np.count_nonzero(np.any(pixels != pixels[0, 0], axis=2)))}
            if style == "studio" and style in images:
                restored = record == images[style]
            else:
                images[style] = record
                if output is not None:
                    output.mkdir(parents=True, exist_ok=True)
                    image.save(str(output / f"{style}.png"))
        errors = [message for message in messages if "Failed to" in message or "ERROR:" in message]
        if view.root.property("styleStatus"):
            errors.append("the imported mesh/material layout does not match glTF")
        return {"modes": images, "restored": restored, "errors": errors,
                "ok": restored and not errors and all(item["painted"] for item in images.values())
                and len({item["sha256"] for item in images.values()}) == len(STYLES)}
    finally:
        qInstallMessageHandler(previous)
