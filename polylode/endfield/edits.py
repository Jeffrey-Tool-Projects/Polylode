"""A converted character's edits, kept per character: colours and the outfit it is put on in.

The character maker edits a converted character the way it edits a card's
(owner, 2026-10-16), but a converted character has no card to write them
into.  They are kept in ``UserData/config/actor_edits/<ident>.json`` and laid
on the character wherever it is loaded (``ActorController``), so the maker,
the studio and Free H show the same person.

- **Colours** multiply the base colour of the parts drawn by one program:
  the skin program (body and face), the hair program, the eye program —
  told by the material, not by the part's name (``ActorController.piece_group``).
  The textures are not touched.
- **The outfit** is the set of pieces worn (``ActorController.wear``),
  kept as the pieces of the parts the character has now; a piece the
  character no longer has is dropped when it is laid on.
- **The body** is the card's 44 body sliders, played the way a card plays
  them (``origin/figure/shape_apply``) on the joints of the lent body that
  the character keeps (``body.helper_nodes``: each the source joint, fitted,
  under the character's own bone).  Each such joint stands where the fit put
  the source body's — the reference rig shaped by the source card (§7.1109:
  run on the reference rig, the card's sliders put the source's joints within
  3.8 mm median of the stored ones, the bust's 90 mm off unshaped) — so an edit
  moves it by what the sliders change on the reference rig, in the joint's
  own frame (:func:`shape_offsets`), and the same sliders change nothing.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..core import math3d as m3
from ..core.paths import CONFIG_DIR, write_json_atomic

#: what a colour edit tints, by the program that draws it
COLOR_TARGETS = ("skin", "hair", "eyes")
# ChaFileBody.MemberInit IL_002b: editor controls start at 0.5. Measured H
# dimensions are separate, and are not a shape state for the donor's bones.
BODY_DEFAULT = 0.5
BODY_EDIT_VERSION = 2


def edits_path(ident: str) -> Path:
    """Where *ident*'s edits are kept."""
    return CONFIG_DIR / "actor_edits" / f"{ident}.json"


def load(ident: str) -> dict:
    """*ident*'s edits (``{}`` when it has none).  A file that cannot be read is an error, not no edits."""
    path = edits_path(ident)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: not an edits table")
    return data


def save(ident: str, data: dict) -> None:
    """Keep *data* as *ident*'s edits; an empty table removes the file."""
    path = edits_path(ident)
    if not data:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, data, indent=1)


def color_target(part) -> str:
    """Which of :data:`COLOR_TARGETS` *part* is tinted with, or ``""``: by the program that draws it."""
    from ..gl import shaders_endfield as SE
    declared = getattr(part.material, "edit_color_target", None)
    if declared in COLOR_TARGETS:
        return declared
    group = getattr(part, "group", "")
    if group == "endfield_native_skin":
        return "skin"
    if (group != "endfield_outfit"
            and getattr(part.material, "declared_role", None) in ("face", "native_skin")):
        return "skin"
    family = getattr(part.material, "toon_family", 0)
    if family == SE.FAMILY_SKIN:
        return "skin"
    if family == SE.FAMILY_HAIR:
        return "hair"
    if family == SE.FAMILY_EYE:
        return "eyes"
    return ""


def tinted(original: tuple, tint) -> tuple:
    """*original* (RGBA) times *tint* (RGB, 1 = as the game made it); alpha kept."""
    r, g, b = (float(v) for v in tint)
    return (original[0] * r, original[1] * g, original[2] * b, original[3])


def valid_tint(tint) -> bool:
    """Whether *tint* is three finite non-negative numbers."""
    try:
        values = [float(v) for v in tint]
    except (TypeError, ValueError):
        return False
    return len(values) == 3 and all(math.isfinite(v) and v >= 0.0 for v in values)


@lru_cache(maxsize=1)
def _reference_rest() -> tuple:
    """``(names, parents, locals)`` of the reference rig (``origin/rig``), parents before children."""
    from ..origin import rig as R
    ref = R.rig()
    order, seen = [], set()

    def visit(name):
        if name in seen or name not in ref.locals:
            return
        parent = ref.parents.get(name, "")
        if parent and parent != name:
            visit(parent)
        seen.add(name)
        order.append(name)
    for name in ref.locals:
        visit(name)
    parents = tuple(ref.parents.get(name, "") for name in order)
    return tuple(order), parents, tuple(np.asarray(ref.locals[name], dtype=np.float64) for name in order)


def _reference_skeleton():
    """A fresh reference rig at rest, to shape."""
    from ..mesh.skeleton import Skeleton
    names, parents, locals_ = _reference_rest()
    if not names:
        raise LookupError("the reference rig has not been converted yet")
    skeleton = Skeleton("reference")
    known = set()
    for name, parent, local in zip(names, parents, locals_):
        pos, rot, scale = m3.decompose(local)
        skeleton.add(name, parent if parent in known else None, pos, rot, scale)
        known.add(name)
    return skeleton.finalize()


class _Sliders:
    """A card with other body slider values: what ``shape_apply.apply_body_shape`` reads of it."""

    def __init__(self, card, values):
        self._card, self._values = card, list(values)

    def body(self) -> dict:
        return {**(self._card.body() or {}), "shapeValueBody": list(self._values)}

    def __getattr__(self, name):
        return getattr(self._card, name)


def body_values(card) -> list[float]:
    """The card's 44 body slider values."""
    return [float(v) for v in (card.body() or {}).get("shapeValueBody") or ()]


def _shaped_worlds(card, values) -> dict:
    from ..origin.figure import shape_apply
    skeleton = _reference_skeleton()
    report = shape_apply.apply_body_shape(skeleton, _Sliders(card, values), 1.0)
    if report.reason:
        raise LookupError(report.describe())
    return {bone.name: bone.bind_world for bone in skeleton.bones}


def shape_offsets(card, values) -> dict:
    """``{joint: 4×4}``: how body sliders *values* move each joint of the reference rig from where *card*'s put it.

    In the joint's own frame (``W_card⁻¹ · W_values``), so a joint fitted
    anywhere moves the same way relative to itself.  Joints that do not move
    are left out.
    """
    source, edited = _shaped_worlds(card, body_values(card)), _shaped_worlds(card, values)
    out = {}
    for name, world in edited.items():
        offset = np.linalg.inv(source[name]) @ world
        if not np.allclose(offset, np.eye(4), atol=1e-7):
            out[name] = offset
    return out
