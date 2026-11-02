"""A staged ``Material`` and the textures it binds.

Nothing here interprets a property: every bound texture is kept under the
shader's own property name with its tiling, and every float, colour and keyword
is carried through.  Which property a renderer draws as its albedo is the
shader's business and is decided where the pieces are built
(:mod:`.convert`), not from what a texture happens to be called.

A texture's colour space and wrap mode are not in the exporter's decoded
``Texture2D``; they are read from its type-tree dump, which carries every
serialized field.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .cabmap import CabIndex, Ref


class MaterialError(ValueError):
    """A staged material or texture dump missing a field the bake needs."""


@dataclass(frozen=True)
class TexBinding:
    """One bound entry of ``m_SavedProperties.m_TexEnvs``."""
    prop: str
    ref: Ref
    scale: tuple
    offset: tuple


@dataclass
class UnityMaterial:
    name: str
    shader: Ref | None
    keywords: list
    queue: int                 # ``m_CustomRenderQueue``; -1 = the shader's own
    tags: dict
    disabled_passes: list
    floats: dict
    colors: dict               # name → (r, g, b, a)
    textures: list             # TexBinding, bound entries only


@dataclass(frozen=True)
class TextureInfo:
    name: str
    srgb: bool                 # ``m_ColorSpace`` == 1
    wrap_u: int                # 0 Repeat, 1 Clamp, 2 Mirror, 3 MirrorOnce
    wrap_v: int


def read_material(doc: dict, cabs: CabIndex, cab: str) -> UnityMaterial:
    """The exporter's ``Material`` JSON from serialized file *cab*."""
    saved = doc.get("m_SavedProperties") or {}
    textures = []
    for prop, env in sorted((saved.get("m_TexEnvs") or {}).items()):
        ref = cabs.pointer(cab, env.get("m_Texture"))
        if ref is None:
            continue
        scale, offset = env["m_Scale"], env["m_Offset"]
        textures.append(TexBinding(
            prop, ref, (float(scale["X"]), float(scale["Y"])),
            (float(offset["X"]), float(offset["Y"]))))
    colors = {name: (float(c["r"]), float(c["g"]), float(c["b"]), float(c["a"]))
              for name, c in (saved.get("m_Colors") or {}).items()}
    floats = {name: float(v) for name, v in (saved.get("m_Floats") or {}).items()}
    floats.update({name: float(v)
                   for name, v in (saved.get("m_Ints") or {}).items()})
    return UnityMaterial(
        name=str(doc.get("m_Name") or ""),
        shader=cabs.pointer(cab, doc.get("m_Shader")),
        keywords=sorted(str(k) for k in doc.get("m_ValidKeywords") or []),
        queue=int(doc.get("m_CustomRenderQueue", -1)),
        tags={str(k): str(v) for k, v in (doc.get("m_StringTagMap") or {}).items()},
        disabled_passes=[str(p) for p in doc.get("m_DisabledShaderPasses") or []],
        floats=floats, colors=colors, textures=textures)


def _field(text: str, kind: str, name: str) -> str:
    found = re.search(rf"^\s*{kind} {name} = (.*?)\s*$", text, re.M)
    if found is None:
        raise MaterialError(f"texture dump has no '{kind} {name}'")
    return found.group(1)


def read_texture_info(text: str) -> TextureInfo:
    """Colour space and wrap modes out of a ``Texture2D`` type-tree dump."""
    return TextureInfo(
        name=_field(text, "string", "m_Name").strip('"'),
        srgb=int(_field(text, "int", "m_ColorSpace")) == 1,
        wrap_u=int(_field(text, "int", "m_WrapU")),
        wrap_v=int(_field(text, "int", "m_WrapV")))
