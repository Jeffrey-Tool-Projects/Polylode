"""A converted character's body skin drawn the way its face is: without painted occlusion.

The face and the body are drawn by the same skin program, which reads the base
map's alpha as occlusion.  The game paints the two maps apart: the face's
alpha is 1 everywhere, while the body's carries occlusion painted for the body
under its outfit — dark wherever a garment sits.  Undressed, that shade is on
bare skin: Avywen's chest drew at an occlusion of 0.34, dark brown under a face
drawn at 1, and her neck met her jaw as a second skin (§7.1113).

So every body skin surface — a skin-program part without the face's light map
— is drawn with its occlusion at 1, as the face is.  Its colour is kept: the
lent body's tints stay matched to it along the seams.
"""
from __future__ import annotations

from ..gl import shaders_endfield as SE
from ..origin import textures as TEX
from .actor import ActorError


def _flags(material) -> int:
    for _kind, name, value in material.extra_uniforms or ():
        if name == "uEfFlags":
            return int(value[0])
    return 0


def body_skin(parts) -> list:
    """The skin-program parts among *parts* that are not the face's (no SDF light map: the face and its brows)."""
    return [part for part in parts
            if part.material is not None and part.material.toon_family == SE.FAMILY_SKIN
            and not _flags(part.material) & SE.FLAGS["_SDFLIGHTMAP"]]


def without_occlusion(material) -> bool:
    """Draw *material* with its base map's alpha (the program's occlusion) at 1; whether it had any below 1.

    The map is copied; the original stays for whatever else draws with it.
    """
    pixels = TEX.pixels_of(material.texture)
    if pixels is None:
        raise ActorError(f"skin map of {material.name} is not in memory")
    if pixels.shape[-1] != 4 or (pixels[..., 3] == 255).all():
        return False
    opaque = pixels.copy()
    opaque[..., 3] = 255
    kind, _id, srgb = material.texture
    material.texture = TEX.store(opaque, srgb=bool(srgb), repeat=kind == "originasset_repeat")
    return True


def match_face(parts) -> list:
    """Draw the body skin among *parts* without its painted occlusion; the names of the parts that had some."""
    done, changed = set(), []
    for part in body_skin(parts):
        if id(part.material) in done:
            continue
        done.add(id(part.material))
        if without_occlusion(part.material):
            changed.append(part.name)
    return changed
