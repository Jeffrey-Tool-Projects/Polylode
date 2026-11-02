"""A converted face that moves by its own shape keys, answering what an H row asks of a face.

Endfield's faces move by bone controls (:mod:`.morph`).  A face converted from
a model file — a converted source's declared face and head renderers
(``part_roles.DECLARED_ROLE``) — moves by the shape keys those renderers carry:
one control per key name, every head renderer that has a key of that name
moving with it (the source split one morph over its face, eyes and lashes by
giving each part the same name).

It answers an H row's face (``origin.h_face.face_weights``) the way a bone face
does (:class:`.hface.Answer`): marks measured on the face, each control's
movement read at them, each reference shape answered by the nearest few.  The
marks are read off the face's own geometry and keys, never off their names:

* the lips: vertices resting across the mouth from each other that some key
  parts (``hface._reference_lips``, the reference head's own rule), among the
  keys that leave the eyes' lid rings where they are (the lids meet at their
  corners too, and a blink parts them there);
* the lid rings: the face's openings that come in a mirrored pair, the pair
  with the most points — its eyes;
* a brow: the face's points above one lid ring and within its width that some
  key moves.

The voice mouth (``lip_sync``) reaches the face through the same row weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import hface
from .part_roles import DECLARED_ROLE

#: The roles whose renderers' shape keys are the face's controls.
HEAD_ROLES = ("face", "head")
#: Two openings are a mirrored pair when their centres mirror within this (m).
MIRROR_M = 2e-3


class ShapeFaceError(LookupError):
    """The item declares no face with shape keys, or its face has no marks to measure."""


@dataclass
class ShapeFace:
    """Shape-key controls over a model's head parts: ``{control: [(part index, rows, delta)]}``."""
    parts: list
    controls: dict
    rest: dict = field(default_factory=dict)
    #: the declared face's part index (its marks are measured on it)
    face: int = -1

    @classmethod
    def of(cls, parts: list, pieces: list) -> ShapeFace | None:
        """The face of *parts* (an actor's model) from *pieces* (its item's), ``None`` when none is declared."""
        by_name = {f"{(p.shader or {}).get('_rendName')}#{p.mat_slot}": p for p in pieces}
        controls: dict = {}
        face = -1
        for index, part in enumerate(parts):
            piece = by_name.get(str(part.name).split("|", 1)[0])
            role = (getattr(piece, "shader", None) or {}).get(DECLARED_ROLE)
            if role not in HEAD_ROLES:
                continue
            if role == "face":
                face = index
            for name, rows, delta, *_ in piece.shapes or ():
                controls.setdefault(str(name), []).append((index, np.asarray(rows, dtype=np.int64),
                                                           np.asarray(delta, dtype=np.float64)))
        if face < 0 or not controls:
            return None
        rest = {index: parts[index].mesh.positions.astype(np.float64).copy()
                for index in {index for rows in controls.values() for index, _r, _d in rows}}
        return cls(parts, controls, rest, face)

    def known(self, name: str) -> bool:
        """Is *name* one of this face's controls?"""
        return name in self.controls

    def missing(self, weights: dict) -> list:
        """The controls *weights* names that this face does not have."""
        return sorted(name for name in weights if name not in self.controls)

    def moved(self, name: str) -> np.ndarray:
        """How far control *name* at full weight moves each point of the face part."""
        out = np.zeros_like(self.rest[self.face]) if self.face in self.rest else \
            np.zeros((self.parts[self.face].mesh.vertex_count, 3))
        for index, rows, delta in self.controls.get(name, ()):
            if index == self.face:
                out[rows] += delta
        return out

    def apply(self, weights: dict) -> None:
        """Lay *weights* on the head parts' rest shapes (a control not named is at rest)."""
        shaped = {index: rest.copy() for index, rest in self.rest.items()}
        for name, weight in weights.items():
            for index, rows, delta in self.controls.get(name, ()):
                shaped[index][rows] += float(weight) * delta
        for index, positions in shaped.items():
            part = self.parts[index]
            part.mesh.positions = positions.astype(np.float32)
            part.stale = True

    def face_points(self) -> np.ndarray:
        """The face part's rest points."""
        return self.rest.get(self.face, self.parts[self.face].mesh.positions.astype(np.float64))

    def mouth_controls(self) -> frozenset:
        """The controls that move the face's own skin and leave its eyes' lid rings still (:func:`mouth_keys`)."""
        if not hasattr(self, "_mouth"):
            try:
                rings = np.concatenate(lid_rings(self.parts[self.face].mesh))
            except ShapeFaceError:
                rings = None
            out = set()
            for name in self.controls:
                moved = self.moved(name)
                size = np.linalg.norm(moved, axis=1)
                if size.max() > STILL_M and (rings is None or size[rings].max() <= STILL_M):
                    out.add(name)
            self._mouth = frozenset(out)
        return self._mouth

    def capped(self, weights: dict) -> dict:
        """*weights* with its mouth controls scaled so the mouth moves no farther than one of its keys does.

        An answer can hold several mouth keys at full weight at once, and a
        face's keys add: three of Lappland's opened her mouth 30 mm where the
        most any one of them moves it is 16.5, and the lower lip came out
        under the chin.  So the mouth stays inside what the face was made to
        do: the farthest point of the mouth controls' sum is held to the
        farthest any single mouth control moves a point at full weight.
        """
        mouth = [name for name in weights if name in self.mouth_controls()]
        if len(mouth) < 2:
            return dict(weights)
        reach = max(float(np.linalg.norm(self.moved(name), axis=1).max()) for name in self.mouth_controls())
        total = sum((float(weights[name]) * self.moved(name) for name in mouth), np.zeros_like(self.moved(mouth[0])))
        far = float(np.linalg.norm(total, axis=1).max())
        if far <= reach or far <= 0.0:
            return dict(weights)
        scale = reach / far
        return {name: (weight * scale if name in mouth else weight) for name, weight in weights.items()}

    def blush_control(self) -> str | None:
        """The control that lays a blush on this face (:func:`blush_overlay`), measured once; None without one."""
        if not hasattr(self, "_blush"):
            try:
                rings = lid_rings(self.parts[self.face].mesh)
            except ShapeFaceError:
                rings = None
            self._blush = None if rings is None else blush_overlay(self, rings)
        return self._blush


#: A key moves a lid ring when one of its points goes farther than this (m): it is not a mouth key then.
STILL_M = 1e-4


def overlay_colour(part, rows: np.ndarray) -> np.ndarray | None:
    """``(r, g, b)`` 0–1 of *part*'s base map over its triangles touching *rows*, weighed by the map's alpha.

    Sampled at each triangle's UV centre; None when the part has no map in
    memory or no alpha over those triangles.
    """
    from ..origin.textures import pixels_of
    material = getattr(part, "material", None)
    pixels = pixels_of(getattr(material, "texture", None)) if material is not None else None
    uvs = getattr(part.mesh, "uvs", None)
    if pixels is None or uvs is None or not len(uvs) or pixels.ndim != 3 or pixels.shape[2] < 4:
        return None
    triangles = part.mesh.indices.reshape(-1, 3)
    touched = triangles[np.isin(triangles, rows).any(axis=1)]
    if not len(touched):
        return None
    centre = np.asarray(uvs, dtype=np.float64)[touched].mean(axis=1) % 1.0
    high, wide = pixels.shape[:2]
    x = np.clip((centre[:, 0] * (wide - 1)).astype(int), 0, wide - 1)
    y = np.clip(((1.0 - centre[:, 1]) * (high - 1)).astype(int), 0, high - 1)
    sample = pixels[y, x].astype(np.float64) / 255.0
    cover = float(sample[:, 3].sum())
    return None if cover <= 0.0 else (sample[:, :3] * sample[:, 3:4]).sum(axis=0) / cover


def blush_overlay(face: ShapeFace, rings: list) -> str | None:
    """The control that brings the reddest overlay onto the face below its eyes; None when none is a blush.

    A face built of shape keys carries its blush as an overlay (a renderer of
    its own, drawn with its expression sheet's alpha) that a key moves out
    onto the cheeks.  Which key it is, is read off the model: of the keys
    moving a part other than the face's own skin, the one whose moved points
    end below the eyes (:func:`lid_rings`) with the reddest map under them —
    the rule the converted skin's emotion sheet is read by
    (``shading.blush_cell``: an overlay redder than it is green or blue by
    ``BLUSH_RED``).
    """
    from .shading import BLUSH_RED
    points = face.face_points()
    eyes = float(np.mean([points[ring][:, 1].mean() for ring in rings]))
    best, reddest = None, BLUSH_RED
    for name, entries in sorted(face.controls.items()):
        for index, rows, delta in entries:
            if index == face.face:
                continue
            moved = np.linalg.norm(delta, axis=1) > STILL_M
            if not moved.any():
                continue
            end = face.rest[index][rows[moved]] + delta[moved]
            if float(end[:, 1].mean()) >= eyes:
                continue
            colour = overlay_colour(face.parts[index], rows[moved])
            if colour is None:
                continue
            red = float(colour[0] - max(colour[1], colour[2]))
            if red > reddest:
                best, reddest = name, red
    return best


def mouth_keys(keys: list, rings: list) -> list:
    """The keys (movements per face point) that leave both lid rings still: the mouth's, not the eyes'."""
    rows = np.concatenate(rings)
    return [moved for moved in keys if float(np.linalg.norm(moved[rows], axis=1).max()) <= STILL_M]


def lips(points: np.ndarray, keys: list) -> tuple[np.ndarray, np.ndarray]:
    """``(upper, lower)`` vertex rows of the lips: the reference head's own rule (``hface._reference_lips``)."""
    return hface._reference_lips(points, keys)


def lips_centre(points: np.ndarray, keys: list) -> np.ndarray:
    """The middle of the lips: where the mouth an oral row aims at is on this face."""
    upper, lower = lips(points, keys)
    return points[np.concatenate([upper, lower])].mean(axis=0)


def lid_rings(mesh) -> list[np.ndarray]:
    """The face's two eye openings (the +x one first): its mirrored pair of openings with the most points."""
    from .seams import _opening_loops
    loops, error = _opening_loops(mesh, allow_fins=True, split_pinches=True)
    if error:
        raise ShapeFaceError(f"the face's openings cannot be traced: {error}")
    points = mesh.positions.astype(np.float64)
    centres = [points[loop].mean(axis=0) for loop in loops]
    best = None
    for a, first in enumerate(loops):
        for b in range(a + 1, len(loops)):
            mirrored = centres[b] * np.array([-1.0, 1.0, 1.0])
            if (abs(centres[a][0]) > MIRROR_M and np.linalg.norm(centres[a] - mirrored) <= MIRROR_M
                    and len(first) == len(loops[b])):
                size = len(first)
                if best is None or size > best[0]:
                    best = (size, first, loops[b])
    if best is None:
        raise ShapeFaceError("the face has no mirrored pair of openings for its eyes")
    rings = [np.asarray(best[1]), np.asarray(best[2])]
    return sorted(rings, key=lambda ring: -points[ring][:, 0].mean())


def brows(points: np.ndarray, rings: list, stirred: np.ndarray) -> list[np.ndarray]:
    """Per eye, the points above its lid ring and within its width that some control moves."""
    out = []
    for ring in rings:
        around = points[ring]
        low, high = around.min(axis=0), around.max(axis=0)
        width = high[0] - low[0]
        above = (points[:, 1] > high[1]) & (np.abs(points[:, 0] - around[:, 0].mean()) <= width) & stirred
        out.append(np.flatnonzero(above))
    if any(len(rows) < 2 for rows in out):
        raise ShapeFaceError("no control moves the face above its eyes: it has no brows to measure")
    return out


class ShapeAnswer(hface.Answer):
    """:class:`.hface.Answer` for a :class:`ShapeFace`: the same matching, its marks measured off its geometry."""

    def __init__(self, face: ShapeFace):
        self.face = face
        self.controls = sorted(face.controls)
        points = face.face_points()
        moved = {name: face.moved(name) for name in self.controls}
        stirred = np.any([np.linalg.norm(m, axis=1) > 0 for m in moved.values()], axis=0)
        self.measured: dict = {}
        mesh = face.parts[face.face].mesh
        try:
            rings = lid_rings(mesh)
            self.measured["eyes"] = [self._columns(moved, lambda gone, r=ring: hface.ring(points[r], gone[r]))
                                     for ring in rings]
            self.measured["eyebrow"] = [self._columns(moved, lambda gone, r=rows: hface.brow(points[r], gone[r]))
                                        for rows in brows(points, rings, stirred)]
        except ShapeFaceError:
            pass
        try:
            rings = lid_rings(mesh)
            upper, lower = lips(points, mouth_keys(list(moved.values()), rings))
            self.measured["mouth"] = [self._columns(moved, lambda gone: hface.mouth(
                (points[upper], gone[upper]), (points[lower], gone[lower])))]
        except (hface.FaceError, ShapeFaceError):
            pass
        self.without = [part for part in hface.MOST if part not in self.measured]
        self._answers: dict = {}
        self.left: dict = {}

    def _amounts(self, control: str) -> np.ndarray:
        """Every movement *control* makes, without its sign, smallest first (its mirror is told by them)."""
        return np.sort(np.abs(np.concatenate([delta.ravel() for _i, _r, delta in self.face.controls[control]])))


def measured_lips(pieces: list, skeleton) -> np.ndarray | None:
    """``(1, 3)``: the middle of the lips of *pieces*' declared face at rest, ``None`` without one.

    The face's own points and keys (:func:`lips_centre`); what an oral row's
    aim is set against on a face with no lip joints (``origin_rig.gather``).
    """
    from .actor import _rest_mesh
    face = next((p for p in pieces if (p.shader or {}).get(DECLARED_ROLE) == "face" and p.shapes), None)
    if face is None:
        return None
    points = _rest_mesh(face, skeleton).positions.astype(np.float64)
    keys = []
    for _name, rows, delta, *_ in face.shapes:
        moved = np.zeros_like(points)
        moved[np.asarray(rows, dtype=np.int64)] = np.asarray(delta, dtype=np.float64)
        keys.append(moved)
    try:
        return lips_centre(points, mouth_keys(keys, lid_rings(_rest_mesh(face, skeleton))))[None]
    except (hface.FaceError, ShapeFaceError):
        return None
