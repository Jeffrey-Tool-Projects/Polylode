"""A converted face answering the face an H row asks for.

An H row is weights on the expression shapes of the head the clips were made
for (``origin.h_face.face_weights``: one closed and one open shape for the
eyebrows, the eyes and the mouth, mixed by how far each is open).  A converted
character's face is its own game's: bone controls, each one a whole look of
one part of the face.  Neither side's names say which answers which, so both
are **measured the same way** and matched on the numbers:

* the lids are a ring of joints round each eye on both rigs.  A shape or a
  control is how far the ring moves at eight places round the eye, in eye
  heights;
* a brow is how far its inner, middle and outer third move, in brow lengths;
* the mouth is the same three thirds of the upper and of the lower lip from
  the middle to a corner, in half mouth widths.

The tongue is no mark.  The reference head's own tongue rests parked out of
sight (its plain open-mouth shape alone carries the tip a whole half mouth
width down), and the tongue a licking row shows is another object, on the
body, switched in by the row's mouth number (``ChaControl.UpdateVisible``:
``S_TongueB`` for ``S_TongueF``) — neither is a movement from a rest that a
converted face's tongue joints could be compared with.

Every displacement is from the face's own rest, in that face's own unit, so
"shut" on one face is "shut" on the other whatever their sizes.  A shape is
then answered by the few controls whose measured movement, weighted between
0 and 1, comes nearest to it (:func:`nearest`), and a row's face is the sum
of its shapes' answers at the row's weights: both rigs add their parts up.

The reference head is read from what this machine converted
(``body_head.load_head_prefab``); nothing of it is kept but the measured
movements.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: The reference head's meshes that carry the lids and lips, and the brows.
FACE_MESH, BROW_MESH = "cf_O_face", "cf_O_mayuge"
#: Its ring of eight joints round an eye (``%`` the place 1–8 and the side).
REFERENCE_LIDS = "cf_J_Eye%02d_s_%s"
#: What the names of a converted face's joints begin with: round each eye, along each brow …
LIDS = ("eyeRt", "eyeLf")
BROWS = ("browRt", "browLf")
#: … and along the lips.
LIP = "lip"

#: Places round an eye a lid's movement is read at, and the thirds of a brow or of half a lip.
STATIONS = 8
THIRDS = 3
#: The vertices of the reference face that stand for one of its lid joints: the nearest few.
NEAR = 3
#: Two vertices of the reference face are across the lips from each other when they rest this
#: close (m) and some mouth shape parts them by this much (m).
LIP_REACH, LIP_PART = 0.002, 0.006
#: How many controls may answer one shape, per part of the face (for an eye or a brow, a side).
MOST = {"eyes": 2, "eyebrow": 2, "mouth": 3}
#: A shape whose two sides differ by less than this share of its movement is the same on both.
EVEN = 0.05
#: A further control has to take this share of the shape's movement off what is left.
GAIN = 0.05
#: The controls tried at each step: those that would take the most off on their own.
SHORTLIST = 6


class FaceError(LookupError):
    """The reference head is not converted, or is not the rig this module measures."""


# -- measuring: the same for both faces ---------------------------------------------
def ring(rest: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """One eye's lid ring: its movement at :data:`STATIONS` places round the eye, in eye heights.

    *rest* is where the ring's points stand and *moved* how far each has
    gone.  The places are angles round the middle of the ring, counted from
    the outer corner on a ring squared up to its own width and height, so two
    rigs whose joints sit at different places round the eye are read at the
    same ones.  Sideways movement is counted away from the middle of the face.
    """
    low, high = rest[:, :2].min(axis=0), rest[:, :2].max(axis=0)
    centre, half = (low + high) / 2.0, (high - low) / 2.0
    side = 1.0 if centre[0] >= 0.0 else -1.0
    angle = np.arctan2((rest[:, 1] - centre[1]) / half[1], side * (rest[:, 0] - centre[0]) / half[0])
    order = np.argsort(angle)
    at = np.linspace(-np.pi, np.pi, STATIONS, endpoint=False)
    return np.concatenate([np.interp(at, angle[order], values[order], period=2.0 * np.pi)
                           for values in (side * moved[:, 0], moved[:, 1])]) / (2.0 * half[1])


def thirds(along: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The mean of *values* over each third of *along* (0–1); an empty third takes its nearest point."""
    out = np.empty((THIRDS, values.shape[1]))
    for k in range(THIRDS):
        inside = (along >= k / THIRDS) & ((along < (k + 1) / THIRDS) | (k == THIRDS - 1))
        if not inside.any():
            inside = np.abs(along - (k + 0.5) / THIRDS) <= np.abs(along - (k + 0.5) / THIRDS).min() + 1e-12
        out[k] = values[inside].mean(axis=0)
    return out.ravel()


def brow(rest: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """One brow: how far its inner, middle and outer third move (outward, up), in brow lengths."""
    side = 1.0 if rest[:, 0].mean() >= 0.0 else -1.0
    across = side * rest[:, 0]
    length = across.max() - across.min()
    values = np.stack([side * moved[:, 0], moved[:, 1]], axis=1) / length
    return thirds((across - across.min()) / length, values)


def mouth(upper: tuple, lower: tuple) -> np.ndarray:
    """The lips: each lip's thirds from the middle to a corner (outward, up, forward), in half mouth widths.

    *upper* and *lower* are ``(rest, moved)`` of the points along each lip,
    both sides together: a side's sideways movement is counted outward, so the
    two sides fall on each other.
    """
    half = max(np.abs(upper[0][:, 0]).max(), np.abs(lower[0][:, 0]).max())
    out = []
    for rest, moved in (upper, lower):
        outward = np.where(rest[:, 0] >= 0.0, 1.0, -1.0) * moved[:, 0]
        out.append(thirds(np.abs(rest[:, 0]) / half, np.stack([outward, moved[:, 1], moved[:, 2]], axis=1) / half))
    return np.concatenate(out)


def nearest(have: np.ndarray, wanted: np.ndarray, most: int) -> tuple[dict, float]:
    """``({column: weight}, share of the movement left over)``: *wanted* out of *have*'s columns.

    Weights are held between 0 and 1 (what a control is made for).  Columns
    are taken one at a time, each the one that leaves the least over with
    the ones already taken refitted, until *most* are in or the next one would
    take less than :data:`GAIN` of the whole movement off.
    """
    whole = float(wanted @ wanted)
    if whole < 1e-12:
        return {}, 0.0
    lengths = (have * have).sum(axis=0)
    taken: list = []
    weights = np.zeros(0)
    left = whole
    for _ in range(most):
        pull = have.T @ (wanted - have[:, taken] @ weights)
        # what each column would take off alone, at its own best weight inside 0–1
        weight = np.clip(pull / np.maximum(lengths, 1e-12), 0.0, 1.0)
        alone = weight * (2.0 * pull - weight * lengths)
        alone[taken] = 0.0
        best = None
        for column in np.argsort(alone)[::-1][:SHORTLIST]:
            if alone[column] <= 0.0:
                break
            tried = _held(have[:, taken + [int(column)]], wanted)
            over = wanted - have[:, taken + [int(column)]] @ tried
            if best is None or float(over @ over) < best[0]:
                best = (float(over @ over), int(column), tried)
        if best is None or left - best[0] < GAIN * whole:
            break
        left, taken, weights = best[0], taken + [best[1]], best[2]
    return {column: float(weight) for column, weight in zip(taken, weights) if weight > 1e-3}, left / whole


def _held(columns: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Least squares of *wanted* on a few *columns* with every weight held in 0–1."""
    weights = np.zeros(columns.shape[1])
    lengths = (columns * columns).sum(axis=0)
    for _ in range(60):
        for k in range(len(weights)):
            over = wanted - columns @ weights + columns[:, k] * weights[k]
            weights[k] = min(1.0, max(0.0, float(columns[:, k] @ over) / lengths[k]))
    return weights


# -- the reference head ----------------------------------------------------------------
@dataclass(frozen=True)
class Reference:
    """What each expression shape of the reference head does to its marks.

    ``shapes`` is ``{"mesh/shape": measured movement}`` and ``part`` which
    part of the face each belongs to (``eyes`` / ``eyebrow`` / ``mouth``); the
    keys are the ones ``h_face.face_weights`` hands out.
    """
    shapes: dict
    part: dict


def _fields(piece) -> dict:
    """``{shape: (vertex indices, movement)}`` of one piece of the reference head."""
    return {str(name): (np.asarray(index), np.asarray(delta, dtype=np.float64))
            for name, index, delta, *_ in piece.shapes or ()}


def _moved(fields: dict, name: str, count: int) -> np.ndarray:
    """Shape *name* as a movement per vertex; a shape the mesh does not carry moves nothing."""
    out = np.zeros((count, 3))
    if name in fields:
        index, delta = fields[name]
        out[index] = delta
    return out


def _reference_lids(piece, points: np.ndarray) -> list:
    """Per eye (the +x one first), the vertices standing for each of its eight lid joints."""
    where = {str(name): np.asarray(world, dtype=np.float64).reshape(4, 4)[:3, 3]
             for name, world in zip(piece.bones, piece.bone_world or ())}
    eyes = []
    for side in "LR":
        names = [REFERENCE_LIDS % (place, side) for place in range(1, 9)]
        if not all(name in where for name in names):
            raise FaceError(f"the reference face has no lid joints {names[0]}…")
        eyes.append([np.argsort(np.linalg.norm(points - where[name], axis=1))[:NEAR] for name in names])
    return sorted(eyes, key=lambda eye: -points[eye[0]][:, 0].mean())


def _reference_lips(points: np.ndarray, shapes: list) -> tuple[np.ndarray, np.ndarray]:
    """The vertices along the upper and along the lower lip of the reference face.

    The lips rest shut on each other: two vertices that rest within
    :data:`LIP_REACH` and that some mouth shape moves :data:`LIP_PART` apart
    are across the lips from each other, and under the shape that parts the
    most pairs the farthest, the one left higher is on the upper lip.
    """
    stirred = np.where(np.max([np.linalg.norm(moved, axis=1) for moved in shapes], axis=0) > 0.0)[0]
    apart = np.linalg.norm(points[stirred][:, None] - points[stirred][None], axis=2)
    first, second = np.where(np.triu(apart < LIP_REACH, 1))
    first, second = stirred[first], stirred[second]
    parted = np.stack([np.abs(moved[first, 1] - moved[second, 1]) for moved in shapes])
    across = parted.max(axis=0) > LIP_PART
    if not across.any():
        raise FaceError("no mouth shape of the reference face parts its lips")
    first, second = first[across], second[across]
    widest = shapes[int(parted[:, across].sum(axis=1).argmax())]
    higher = widest[first, 1] > widest[second, 1]
    upper = set(np.where(higher, first, second).tolist())
    lower = set(np.where(higher, second, first).tolist())
    return np.array(sorted(upper - lower)), np.array(sorted(lower - upper))


@lru_cache(maxsize=1)
def reference() -> Reference:
    """Measure the reference head; :class:`FaceError` when this machine has not converted it."""
    from ..origin.figure import body_head, face_morph
    try:
        pieces = body_head.load_head_prefab()[0]
    except (OSError, LookupError, ValueError) as exc:
        raise FaceError(f"the reference head is not converted: {exc}") from exc
    owned = {str(piece.mesh.name): piece for piece in reversed(pieces) if piece.mesh is not None and piece.shapes}
    tables = {part: face_morph.pattern_table(part) for part in MOST}
    if any(name not in owned for name in (FACE_MESH, BROW_MESH)) or not all(tables.values()):
        raise FaceError("the reference head has no expression shapes")
    shapes, part = {}, {}

    face = owned[FACE_MESH]
    points = np.asarray(face.mesh.positions, dtype=np.float64)
    fields = _fields(face)

    def used(table: dict, mesh: str) -> list:
        """Per pattern of a part, its (closed, open) shape names on *mesh*."""
        return [pair for _number, pair in sorted((table["by_target"].get(mesh) or {}).items())]

    lids = _reference_lids(face, points)
    for name in sorted({name for pair in used(tables["eyes"], FACE_MESH) for name in pair if name}):
        moved = _moved(fields, name, len(points))
        shapes[f"{FACE_MESH}/{name}"] = np.concatenate(
            [ring(np.stack([points[near].mean(axis=0) for near in eye]),
                  np.stack([moved[near].mean(axis=0) for near in eye])) for eye in lids])
        part[f"{FACE_MESH}/{name}"] = "eyes"

    brows = owned[BROW_MESH]
    hairs = np.asarray(brows.mesh.positions, dtype=np.float64)
    brow_fields = _fields(brows)
    for name in sorted({name for pair in used(tables["eyebrow"], BROW_MESH) for name in pair if name}):
        moved = _moved(brow_fields, name, len(hairs))
        shapes[f"{BROW_MESH}/{name}"] = np.concatenate(
            [brow(hairs[side], moved[side]) for side in (hairs[:, 0] > 0.0, hairs[:, 0] < 0.0)])
        part[f"{BROW_MESH}/{name}"] = "eyebrow"

    lips = used(tables["mouth"], FACE_MESH)
    names = sorted({name for pair in lips for name in pair if name})
    upper, lower = _reference_lips(points, [_moved(fields, name, len(points)) for name in names])
    for name in names:
        moved = _moved(fields, name, len(points))
        shapes[f"{FACE_MESH}/{name}"] = mouth((points[upper], moved[upper]), (points[lower], moved[lower]))
        part[f"{FACE_MESH}/{name}"] = "mouth"
    return Reference(shapes, part)


# -- a converted face ----------------------------------------------------------------------
def numbered(face, stem: str) -> list:
    """The skeleton rows of *face*'s joints named *stem* + two digits + ``Joint`` (a lid ring, a brow)."""
    skeleton = face.skeleton
    return [skeleton.index[name] for name in face.bones
            if skeleton.has(name) and name.startswith(stem)
            and name[len(stem):] == name[len(stem):len(stem) + 2] + "Joint" and name[len(stem):len(stem) + 2].isdigit()]


class Answer:
    """One converted face's controls, measured on its own marks, answering reference shapes.

    *face* is the character's :class:`~polylode.endfield.morph.Face`.  A
    part of the face whose joints it does not have is listed in ``without``
    and answers nothing, which leaves that part at rest.
    """

    def __init__(self, face):
        self.face = face
        self.controls = sorted(face.controls)
        skeleton = face.skeleton
        held = [name for name in face.bones if skeleton.has(name)]

        lids = [rows for rows in (numbered(face, stem) for stem in LIDS) if len(rows) >= 4]
        brows = [rows for rows in (numbered(face, stem) for stem in BROWS) if len(rows) >= 2]
        lips = [skeleton.index[name] for name in held if name.startswith(LIP) and name.endswith("Joint")]
        rows = sorted({row for group in lids + brows for row in group} | set(lips))

        def stand(weights: dict) -> np.ndarray:
            """Where the marks stand under *weights*; a row that is no mark stays at the origin."""
            out = np.zeros((len(skeleton.bones), 3))
            if rows:
                out[rows] = skeleton.evaluate(face.pose(weights), only=rows)[rows, :3, 3]
            return out

        rest = stand({})
        moved = {name: stand({name: 1.0}) - rest for name in self.controls}
        self.without = []
        self.measured: dict = {}
        if len(lids) == 2:
            lids.sort(key=lambda group: -rest[group][:, 0].mean())
            self.measured["eyes"] = [self._columns(moved, lambda gone, group=group: ring(rest[group], gone[group]))
                                     for group in lids]
        if len(brows) == 2:
            brows.sort(key=lambda group: -rest[group][:, 0].mean())
            self.measured["eyebrow"] = [self._columns(moved, lambda gone, group=group: brow(rest[group], gone[group]))
                                        for group in brows]
        if len(lips) >= 4:
            lips = np.array(lips)
            # the control that parts the lips the most says which joints are on which lip
            parting = max(self.controls, key=lambda name: float(np.ptp(moved[name][lips, 1])), default=None)
            lift = moved[parting][lips, 1] if parting is not None else np.zeros(len(lips))
            if np.ptp(lift) > 1e-6:
                above = lift > (lift.max() + lift.min()) / 2.0
                upper, lower = lips[above], lips[~above]
                self.measured["mouth"] = [self._columns(moved, lambda gone: mouth(
                    (rest[upper], gone[upper]), (rest[lower], gone[lower])))]
        self.without = [part for part in MOST if part not in self.measured]
        self._answers: dict = {}
        #: ``{"mesh/shape": share of its movement no control answers}`` of the shapes asked for so far
        self.left: dict = {}

    def _mirror(self, name: str, there: np.ndarray, here: np.ndarray) -> int:
        """The column of the control that moves this side as control *name* moves the other.

        Several controls can move one side's marks alike and still differ on
        the joints that are no marks (the lashes): of those, the one whose
        table holds the same amounts as *name*'s, whatever their signs — a
        mirrored control turns the mirrored joints as far the other way.
        """
        apart = np.linalg.norm(here - there[:, [self.controls.index(name)]], axis=0)
        mine = self._amounts(name)

        def unlike(column: int) -> tuple:
            theirs = self._amounts(self.controls[column])
            same = len(theirs) == len(mine)
            return (not same, float(np.abs(theirs - mine).sum()) if same else 0.0)

        return int(min(np.where(apart <= apart.min() + 1e-6)[0], key=unlike))

    def _amounts(self, control: str) -> np.ndarray:
        """Every number in *control*'s table, without its sign, smallest first."""
        return np.sort(np.abs(np.concatenate([delta for _row, delta in self.face.controls[control]])))

    def _columns(self, moved: dict, measure) -> np.ndarray:
        """Every control's measured movement, a column each (in :attr:`controls` order)."""
        return np.stack([measure(moved[name]) for name in self.controls], axis=1)

    def shape(self, path: str) -> dict:
        """``{control: weight}`` answering reference shape *path* at full weight; ``{}`` when nothing does."""
        if path not in self._answers:
            known = reference()
            part = known.part.get(path)
            if part is None or part not in self.measured:
                self._answers[path] = {}
            else:
                self._answers[path], self.left[path] = self._sides(
                    self.measured[part], np.split(known.shapes[path], len(self.measured[part])), MOST[part])
        return self._answers[path]

    def _sides(self, sides: list, wanted: list, most: int) -> tuple[dict, float]:
        """Answer a shape side by side: ``({control: weight}, share of its movement left over)``.

        A shape that is the same on both sides (:data:`EVEN`) is answered on
        the first and given to the second through each control's **mirror**:
        the control that moves the second side the way this one moves the
        first (itself, for a control that moves both).  Fitted apart, two
        nearly equal controls would be picked differently on the two sides and
        the face would come out lopsided.
        """
        whole = sum(float(side @ side) for side in wanted)
        if whole < 1e-12:
            return {}, 0.0
        even = len(sides) == 2 and float((wanted[0] - wanted[1]) @ (wanted[0] - wanted[1])) < EVEN * whole
        out: dict = {}
        left = 0.0
        for k, (have, want) in enumerate(zip(sides, wanted)):
            if even and k == 1:
                first = out
                columns = {self._mirror(name, sides[0], have): weight for name, weight in first.items()}
                out = dict(first)
            else:
                columns, _share = nearest(have, want, most)
            over = want - sum((have[:, column] * weight for column, weight in columns.items()), np.zeros(len(want)))
            left += float(over @ over)
            for column, weight in columns.items():
                out[self.controls[column]] = max(out.get(self.controls[column], 0.0), weight)
        return out, left / whole

    def weights(self, shapes: dict) -> dict:
        """A face given as ``{"mesh/shape": weight}`` (``h_face.face_weights``) → ``{control: weight}``."""
        out: dict = {}
        for path, weight in shapes.items():
            for control, share in self.shape(path).items():
                out[control] = out.get(control, 0.0) + float(weight) * share
        return {control: min(1.0, weight) for control, weight in out.items() if weight > 1e-3}
