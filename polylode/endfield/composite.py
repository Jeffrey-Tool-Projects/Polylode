"""Reversible outfit coverage, with permanent added/native skin exclusions."""
from __future__ import annotations

import numpy as np

from .actor import ActorError


def validate_manifest(manifest: dict, triangle_count: int) -> None:
    """Reject stale topology and malformed face indices rather than guessing."""
    if manifest.get("version") != 1:
        raise ValueError("unsupported Endfield composite manifest version")
    if manifest.get("body_triangles") != triangle_count:
        raise ValueError("composite body topology does not match stored source")
    for key in ("native_excluded_faces", "covered_faces"):
        faces = manifest.get(key)
        if not isinstance(faces, list) or any(type(f) is not int for f in faces):
            raise ValueError(f"{key} must contain integer face indices")
        if len(set(faces)) != len(faces) or any(f < 0 or f >= triangle_count for f in faces):
            raise ValueError(f"{key} contains duplicate or invalid face indices")


class CompositeVisibility:
    """Swap prebuilt render meshes; the editable source remains untouched."""

    def __init__(self, body, outfit: list, manifest: dict):
        validate_manifest(manifest, body.mesh.tri_count)
        self.body, self.outfit = body, outfit
        self.source = body.mesh
        faces = self.source.indices.reshape(-1, 3)
        permanent = np.ones(len(faces), dtype=bool)
        permanent[manifest["native_excluded_faces"]] = False
        dressed = permanent.copy()
        dressed[manifest["covered_faces"]] = False
        self._permanent = permanent
        self._dressed = dressed
        #: body faces to leave out, measured piece by piece (:meth:`set_cover`); None: the stored list
        self.cover: np.ndarray | None = None
        self._meshes = {}
        for visible, keep in ((False, permanent), (True, dressed)):
            mesh = self.source.copy()
            mesh.indices = faces[keep].ravel().copy()
            self._meshes[visible] = mesh
        self._original_visibility = [part.visible for part in outfit]
        #: the outfit parts taken off one by one, by ``id`` (:meth:`set_hidden`)
        self.hidden: frozenset = frozenset()
        #: body faces a pose put outside what is worn (:meth:`set_posed`); None: none
        self.posed: np.ndarray | None = None
        #: the source faces the body draws now, in the order its mesh has them (:meth:`drawn_faces`)
        self._drawn = np.flatnonzero(permanent)
        #: the genitals' faces (:meth:`set_son`); None: no genitals
        self._son: np.ndarray | None = None
        #: whether what is worn stands across the genitals, and whether a scene shows them anyway
        self.son_covered = False
        self.son_out = False
        self.outfit_visible = True
        self.set_outfit_visible(True)

    def set_outfit_visible(self, visible: bool) -> None:
        """Retain native-skin exclusion in both garment visibility states."""
        self.outfit_visible = bool(visible)
        self._apply()

    def replace_source(self, mesh) -> None:
        """A live shape of the same body: rebuild cached coverage meshes and notify the renderer."""
        if (mesh.vertex_count != self.source.vertex_count
                or not np.array_equal(mesh.indices, self.source.indices)):
            raise ActorError("a live body shape must retain its topology")
        self.source = mesh
        faces = mesh.indices.reshape(-1, 3)
        self._meshes = {}
        for visible, keep in ((False, self._permanent), (True, self._dressed)):
            shaped = mesh.copy()
            shaped.indices = faces[keep].ravel().copy()
            self._meshes[visible] = shaped
        self.body.stale = True
        self._apply()

    def set_hidden(self, parts) -> None:
        """Take outfit parts off one by one (their ``id``); the rest stay as the outfit has them."""
        self.hidden = frozenset(parts)
        self._apply()

    def worn_parts(self) -> list:
        """The parts that are put on the body (everything but the body itself)."""
        return list(self.outfit)

    def body_surfaces(self) -> list:
        """``(part, whole mesh, faces never drawn)`` of the body's surfaces."""
        return [(self.body, self.source, ~self._permanent)]

    def set_cover(self, masks: dict | None) -> None:
        """Leave out the body faces *masks* names (``id(part) → (F,) bool``) while dressed.

        ``None`` goes back to the stored whole-outfit list.  Used when a piece
        of the outfit is off: what only it covered is back in view, what the
        rest covers stays out (``endfield/pieces.py``).
        """
        self.cover = None if masks is None else np.asarray(masks[id(self.body)], dtype=bool)
        self._apply()

    def drawable(self) -> dict:
        """``id(part) → whether the item draws it in some state``."""
        return {id(self.body): True,
                **{id(part): bool(original)
                   for part, original in zip(self.outfit, self._original_visibility)}}

    def set_posed(self, faces: np.ndarray | None) -> None:
        """Also leave out, while dressed, the body faces *faces* (``(F,) bool``) that poses put outside what is worn.

        The cover is measured at rest; a pose moves the body and what it wears
        by their own weights, and the lent skin came out through the shorts
        (``pieces.through``; Typhoea standing with a leg raised, 316 faces up
        to 30 mm, §7.1090).  ``ActorController`` gathers them as a clip plays.
        """
        self.posed = None if faces is None else np.asarray(faces, dtype=bool)
        self._apply()

    def set_son(self, faces: np.ndarray | None, covered: bool) -> None:
        """Name the genitals' faces (``(F,) bool``; None: there are none) and whether what is worn is in their way.

        Koikatsu's ``ChaControl.UpdateVisible`` draws a man's genitals under a
        bottom or underwear that is fully on only when ``visibleSon`` is set,
        which ``HSceneProc.ChangeAnimator`` sets from the row (``numMaleSon``:
        1 on every row with a man's clip, 0 on the rest, §7.1115) and which is
        off otherwise.  A converted outfit has no garment kinds; *covered* says
        whether a worn piece stands across the shaft (``ActorController``).
        """
        self._son = None if faces is None or not np.any(faces) else np.asarray(faces, dtype=bool)
        self.son_covered = bool(covered)
        self._apply()

    def set_son_out(self, out: bool) -> None:
        """Show the genitals whatever is worn (an H scene's row), or by the outfit again."""
        out = bool(out)
        if out != self.son_out:
            self.son_out = out
            self._apply()

    def son_hidden(self) -> bool:
        """Whether the genitals are left out now: dressed, something worn across them, and no H row showing them."""
        return self._son is not None and self.outfit_visible and self.son_covered and not self.son_out

    def drawn_faces(self) -> np.ndarray:
        """The source faces the body's mesh draws now, row for row."""
        return self._drawn

    def _apply(self) -> None:
        # the body's covered faces are left out only under the whole outfit:
        # with one part of it off, what that part covered has to be there
        whole = self.outfit_visible and not any(
            id(part) in self.hidden for part, original
            in zip(self.outfit, self._original_visibility) if original)
        faces = self.source.indices.reshape(-1, 3)
        keep = self._permanent.copy()
        if whole:
            keep &= self._dressed
        if self.outfit_visible and self.cover is not None:
            keep = self._permanent & ~self.cover
        if self.outfit_visible and self.posed is not None:
            keep &= ~self.posed
        hidden = self.son_hidden()
        if hidden:
            keep &= ~self._son
        self._drawn = np.flatnonzero(keep)
        if self.posed is None and not hidden and (not self.outfit_visible or self.cover is None):
            self.body.mesh = self._meshes[whole]
        else:
            mesh = self.source.copy()
            mesh.indices = faces[keep].ravel().copy()
            self.body.mesh = mesh
        for part, original in zip(self.outfit, self._original_visibility):
            part.visible = original and self.outfit_visible and id(part) not in self.hidden


def split_original_parts(actor, manifest: dict) -> list:
    """Split runtime parts by serialized source face groups, preserving skin."""
    entries = manifest.get("parts") or {}
    parts, outfit = [], []
    for original in actor.model.parts:
        groups = entries.get(original.name)
        if groups is None:
            raise ActorError(f"composite classification missing for {original.name}")
        triangles = original.mesh.indices.reshape(-1, 3)
        used = []
        for role, faces in groups.items():
            if role not in ("head", "native_skin", "outfit"):
                raise ActorError(f"unknown composite part role {role}")
            if any(type(f) is not int or f < 0 or f >= len(triangles) for f in faces):
                raise ActorError(f"invalid composite faces for {original.name}")
            used.extend(faces)
            part = original.copy()
            part.name = f"{original.name}|{role}"
            part.mesh.indices = triangles[faces].ravel().copy()
            part.group = f"endfield_{role}"
            parts.append(part)
            if role == "outfit":
                outfit.append(part)
        if sorted(used) != list(range(len(triangles))):
            raise ActorError(f"incomplete or duplicate composite faces for {original.name}")
    actor.model.parts = parts
    return outfit
