"""Blend shapes (morph targets) with sparse vertex deltas.

Shapes are stored sparsely because facial morphs touch a few hundred vertices
out of tens of thousands.  ``BlendShapeSet.apply`` accumulates every non-zero
weight in one pass and writes straight into a Part's mesh.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh, Part


@dataclass
class BlendShape:
    name: str
    indices: np.ndarray               # (K,) int32 vertex ids
    deltas: np.ndarray                # (K,3) float32 position offsets
    normal_deltas: np.ndarray | None = None
    label: str = ""

    @staticmethod
    def from_meshes(name: str, base: Mesh, target: Mesh, tol: float = 1e-5,
                    label: str = "") -> "BlendShape":
        """Difference between two meshes with identical topology."""
        d = target.positions.astype(np.float64) - base.positions.astype(np.float64)
        mag = np.linalg.norm(d, axis=1)
        idx = np.nonzero(mag > tol)[0].astype(np.int32)
        nd = None
        if target.normals.shape == base.normals.shape:
            nd = (target.normals.astype(np.float32) - base.normals.astype(np.float32))[idx]
        return BlendShape(name, idx, d[idx].astype(np.float32), nd, label or name)

    @staticmethod
    def from_deform(name: str, base: Mesh, fn, tol: float = 1e-5,
                    label: str = "") -> "BlendShape":
        """Build a shape by running ``fn(mesh_copy)`` as the target."""
        tgt = base.copy()
        fn(tgt)
        return BlendShape.from_meshes(name, base, tgt, tol, label)

    @property
    def count(self) -> int:
        return len(self.indices)

    def to_dict(self) -> dict:
        return {"name": self.name, "label": self.label,
                "indices": self.indices.tolist(),
                "deltas": np.round(self.deltas, 6).tolist()}

    @staticmethod
    def from_dict(d: dict) -> "BlendShape":
        return BlendShape(d["name"], np.asarray(d["indices"], dtype=np.int32),
                          np.asarray(d["deltas"], dtype=np.float32),
                          None, d.get("label", d["name"]))


class BlendShapeSet:
    """All morph targets for one Part, plus their current weights."""

    def __init__(self, part: Part | None = None):
        self.part = part
        self.base_positions: np.ndarray | None = (
            part.mesh.positions.copy() if part else None)
        self.base_normals: np.ndarray | None = (
            part.mesh.normals.copy() if part else None)
        self.shapes: dict[str, BlendShape] = {}
        self.weights: dict[str, float] = {}
        # Extra names for a shape that is already in ``shapes``.  **Not** a
        # second entry: ``apply`` sums every weight, so registering one shape
        # twice under two names would apply its delta twice as soon as both
        # were driven — which is exactly what happens when a group suffix
        # (``def_cl``, what the blink drives) and a full Origin name
        # (``eye_face.f00_def_cl``, what an expression pattern drives) name the
        # same channel.  One shape, one weight slot (§7.131).
        self._alias: dict[str, str] = {}
        self._dirty = True

    def rebind(self, part: Part) -> None:
        self.part = part
        self.base_positions = part.mesh.positions.copy()
        self.base_normals = part.mesh.normals.copy()
        self._dirty = True

    def add(self, shape: BlendShape) -> BlendShape:
        self.shapes[shape.name] = shape
        self.weights.setdefault(shape.name, 0.0)
        self._alias.pop(shape.name, None)
        return shape

    def alias(self, name: str, target: str) -> bool:
        """Let *name* drive the shape stored under *target*."""
        if not name or name == target or target not in self.shapes:
            return False
        if name in self.shapes:          # a real shape wins over an alias
            return False
        self._alias[name] = target
        return True

    def resolve(self, name: str) -> str | None:
        if name in self.shapes:
            return name
        return self._alias.get(name)

    def has(self, name: str) -> bool:
        return name in self.shapes or name in self._alias

    def names(self) -> list[str]:
        return list(self.shapes) + list(self._alias)

    def set(self, name: str, weight: float) -> None:
        key = self.resolve(name)
        if key is not None:
            w = float(np.clip(weight, -1.0, 2.0))
            if abs(self.weights.get(key, 0.0) - w) > 1e-5:
                self.weights[key] = w
                self._dirty = True

    def set_many(self, values: dict) -> None:
        for k, v in values.items():
            self.set(k, v)

    def clear(self) -> None:
        if any(abs(v) > 1e-6 for v in self.weights.values()):
            self._dirty = True
        for k in self.weights:
            self.weights[k] = 0.0

    @property
    def dirty(self) -> bool:
        return self._dirty

    def apply(self, force: bool = False, recompute_normals: bool = False) -> bool:
        """Write the blended result into the bound Part's mesh.

        Normals are **kept**, not rebuilt.  A Unity blend shape frame carries a
        normal delta beside the vertex delta, so a morphed mesh keeps the
        normals the artist drew, nudged; rebuilding them from the triangles
        instead replaces the whole head's shading with the geometry's own — and
        on a Origin head those two differ by a mean of 16 degrees, enough to push
        the outer cheek across ``_RampG``'s binary step and paint a hard shadow
        rim down the jaw (§7.61).  A resting character still comes through
        here: ``def_cl`` holds the lids at 8%, which is a morph on the eyelids
        and no reason to redraw the face.

        The frame's normal deltas are added the same way and the touched
        normals renormalised, which is what Unity's blend-shape pass does
        (§7.379); a shape without them leaves those normals at the base.
        ``recompute_normals`` is left as an opt-in for a caller whose mesh has
        no authored normals to keep.
        """
        if self.part is None or self.base_positions is None:
            return False
        if not self._dirty and not force:
            return False
        pos = self.base_positions.copy()
        nrm = (self.base_normals.copy()
               if self.base_normals is not None else None)
        touched = False
        bent: list[np.ndarray] = []
        for name, w in self.weights.items():
            if abs(w) < 1e-5:
                continue
            sh = self.shapes.get(name)
            if sh is None or sh.count == 0:
                continue
            np.add.at(pos, sh.indices, sh.deltas * w)
            touched = True
            nd = sh.normal_deltas
            if (nrm is not None and nd is not None
                    and len(nd) == len(sh.indices)):
                np.add.at(nrm, sh.indices, nd * w)
                bent.append(sh.indices)
        self.part.mesh.positions = pos
        if touched and recompute_normals:
            self.part.mesh.compute_normals()
        elif nrm is not None:
            if bent:
                rows = np.unique(np.concatenate(bent))
                ln = np.linalg.norm(nrm[rows], axis=1, keepdims=True)
                nrm[rows] = nrm[rows] / np.maximum(ln, 1e-8)
            self.part.mesh.normals = nrm
        self._dirty = False
        return True


class MorphRig:
    """Blend shape sets for a whole character, addressed as ``part/shape``."""

    def __init__(self):
        self.sets: dict[str, BlendShapeSet] = {}

    def register(self, part: Part) -> BlendShapeSet:
        s = self.sets.get(part.name)
        if s is None:
            s = BlendShapeSet(part)
            self.sets[part.name] = s
        else:
            s.rebind(part)
        return s

    def set(self, path: str, weight: float) -> None:
        if "/" in path:
            part, shape = path.split("/", 1)
            s = self.sets.get(part)
            if s:
                s.set(shape, weight)
        else:                     # broadcast to every set that has the shape
            for s in self.sets.values():
                if s.has(path):   # ``has``, not ``in shapes`` — aliases count
                    s.set(path, weight)

    def set_many(self, values: dict) -> None:
        for k, v in values.items():
            self.set(k, v)

    def clear(self) -> None:
        for s in self.sets.values():
            s.clear()

    def apply(self, force: bool = False) -> list[str]:
        """Apply every dirty set; returns the names of parts that changed."""
        changed = []
        for name, s in self.sets.items():
            if s.apply(force=force):
                changed.append(name)
        return changed

    def all_shapes(self) -> list[str]:
        out = []
        for part, s in self.sets.items():
            out.extend(f"{part}/{n}" for n in s.names())
        return out


# --------------------------------------------------------------------------
def rest_blink(built) -> float:
    """``def_cl``'s weight when nothing is closing the eye — ``1 - correctOpenMax``.

    Illusion never opens the lids all the way (see
    ``figure.body_head.eyes_open_max``), so a character standing still is not
    at ``def_cl = 0``.  Leaving it there is what made our eyes read as wider
    than the game's.

    ``eyes_open_correct`` is what ``FaceBlendShape.LateUpdate`` wrote for the
    gaze being driven this frame; without one, the same function with the gaze
    at zero gives the resting cap ``min(1 - EyeLookUpCorrect, OpenMax)``.  Both
    paths go through ``face_morph.gaze_open_max`` so there is one transcription
    of that line, not two.
    """
    v = getattr(built, "eyes_open_correct", None)
    if v is None:
        from ..origin.figure.face_morph import gaze_open_max
        look = (getattr(built, "eye_look", None) or [None])[0]
        v = gaze_open_max(getattr(built, "eyes_open_max", 1.0), 0.0, 0.0, look)
    return float(max(0.0, 1.0 - float(v)))


def build_face_morphs(built) -> MorphRig:
    """Register the expression blend shapes the card's head brought with it.

    Illusion ships expressions as blend shapes on the head meshes themselves
    (``cf_O_face`` alone has 81), so there is nothing to generate — a face
    comes from a card, and its morphs come with it.  A head with none simply
    gets an empty rig rather than a set of shapes authored for some other mesh.

    The rig comes back with the lids already at their resting weight, and it is
    cached on *built*: a second one would snapshot the *posed* mesh as its base
    and every weight after that would stack on top of the first rig's output.
    """
    cached = getattr(built, "face_morphs", None)
    if isinstance(cached, MorphRig):
        return cached
    rig = MorphRig()
    for part in built.model.parts:
        shapes = getattr(part, "origin_shapes", None)
        if shapes:
            from ..origin.figure import face_morph
            face_morph.register(rig, part, shapes)
    rest = rest_blink(built)
    if rest > 0.0:
        # Scoped to ``FBSCtrlEyes``'s own meshes: the bare suffix would take
        # ``cf_O_mayuge`` with it and close the eyebrows 8% too (§7.137).
        from ..origin.figure.face_morph import set_eye_close
        set_eye_close(built, rig, rest)
        rig.apply()
    try:
        built.face_morphs = rig
    except AttributeError:                # a caller passing something minimal
        pass
    return rig
