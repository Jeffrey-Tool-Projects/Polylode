"""Body sliders read off a drawn body (PROGRESS #169 stage 5, design note §3).

A converted character has no card, but whatever reads a card's body sliders
(MotionIK's shape-sized corrections, the clips' ``Breast`` layers, the size
blend) asks for them all the same, and a slider left out plays as 0 — the
thinnest body there is, not a neutral one.  So each slider is **measured**:
the value at which the source game's body has this body's dimensions at the
bones that slider drives.

What a slider drives is read off the game's own shape program, not guessed:
``cf_custombody`` lists the curves a slider plays (``shape_cat``) and the
transcribed ``ShapeBodyInfoFemale.Update`` (``shape_update.BODY_UPDATE``) which
bone channel each curve reaches (:func:`slider_channels`).  Each channel is
measured on the drawn body by one rule:

==============  ==================================================================
``Scale``       the spread (RMS about its centroid) of the skin that hangs mostly on
                that bone, along the bone's own axis
``Position``    the bone's place from its nearest *placed* ancestor, in that
                ancestor's frame — only where both stand where the body put them
                (a converted rig's anchors and landmarks); a bone the rig only
                carried was not measured on this body, and is not read
``Rotation``    not measured: a placed bone's frame is the rig's alignment, not the
                body's own turn, so a turn read off it would be the rig's
==============  ==================================================================

and the sliders are the values at which the source game's body (its skin
under the shape program at those values) shows the same numbers: fitted
together, by bounded least squares over every number any of them reaches, in
metres at the source game's size, over the range SliderUnlocker opens
(:data:`SLIDER_LIMITS`, :func:`measure`).  Every length on the drawn body is
divided by its scale ``s`` (the body is ``s`` times the source game's size, as
a card at that height is), so the two compare at one size.  A slider none of
whose channels can be read this way, or that the body does not tell from the
others, is **not measured** and is said so (:class:`Reading`); the caller keeps
what it had.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

#: The body sliders a card has (``ChaFileBody.shapeValueBody``).
SLIDER_COUNT = 44
#: The height slider: the body's size ``s``, which every length here is divided by — not measured here.
HEIGHT = 0
#: The range a slider can take: SliderUnlocker's limits on the reference install
#: (``BepInEx/config/com.bepis.bepinex.sliderunlocker.cfg``: minimum −100, maximum 200).
SLIDER_LIMITS = (-1.0, 2.0)
#: The fewest skin points a spread is read from: a 3-D spread needs four points not on one plane.
MIN_POINTS = 4
#: Setters by the kind of channel they write (``shape_run._SETTERS``).
SCALE_SETTERS = {"Scale": (0, 1, 2), "ScaleX": (0,), "ScaleY": (1,), "ScaleZ": (2,)}
POSITION_SETTERS = {"Position": (0, 1, 2), "PositionX": (0,), "PositionY": (1,), "PositionZ": (2,)}
#: The fit's finite-difference step (relative, ``scipy.optimize.least_squares``): a hundredth of the
#: editor's 1 % step, inside one segment of a slider's curve (keys are 1/(n−1) apart, n ≤ 11 here).
DIFF_STEP = 1e-4
#: The fit stops when a step moves no slider by more than this share of it.
XTOL = 1e-8
#: The editor's step: a slider is set in hundredths.
STEP = 0.01
#: How precisely the stored body is kept: float32 positions about a metre from the origin
#: (2^-24 of a metre, in millimetres).  A slider whose own step moves the body less than this is not
#: told from the others by it.
PRECISION_MM = 1000.0 * 2.0 ** -24


@dataclass
class Body:
    """A body at rest, as the measurement reads it.

    *positions*: skin points (N×3, world); *owner*: per point, the bone it
    hangs on most (by name); *frames*: bone → its 4×4 world (only its turn and
    place are read, a scale in it is left out); *parents*: bone → parent;
    *placed*: the bones whose place is this body's own (all of them on the
    source game's body; a converted rig's anchors and landmarks);
    *scale*: the body's size against the source game's (``s``).
    """
    positions: np.ndarray
    owner: np.ndarray
    frames: dict
    parents: dict
    placed: set
    scale: float = 1.0
    #: ``{bone: point indices}`` (:func:`regions`), made from *owner* when not given
    by_bone: dict | None = None

    def region(self, bone: str) -> np.ndarray:
        """The skin points that hang mostly on *bone*."""
        if self.by_bone is None:
            self.by_bone = regions(self.owner)
        rows = self.by_bone.get(bone)
        return self.positions[rows] if rows is not None else self.positions[:0]


def regions(owner) -> dict:
    """``{bone: indices of the points it owns}``."""
    owner = np.asarray(owner, dtype=object)
    out: dict = {}
    for i, name in enumerate(owner.tolist()):
        out.setdefault(name, []).append(i)
    return {name: np.asarray(rows, dtype=np.int64) for name, rows in out.items()}


@dataclass(frozen=True)
class Channel:
    """One channel a slider reaches: ``(bone, kind, axis)``, *kind* ``scale`` / ``position`` / ``rotation``."""
    bone: str
    kind: str
    axis: int


def _curves_read(expr, out: set) -> None:
    if expr[0] == "s":
        out.add(expr[1])
        return
    for child in expr[1:]:
        if isinstance(child, tuple):
            _curves_read(child, out)


def slider_channels(rows, program) -> dict:
    """``{slider: [Channel, …]}``: every bone channel the curves of each slider reach.

    *rows*: the ``cf_custombody`` table (``shape_cat.ShapeRow``); *program*:
    ``shape_update.BODY_UPDATE``.  A channel is listed for a slider when the
    expression the program writes it with reads a curve that slider plays.
    """
    plays: dict[int, set] = {}
    for row in rows:
        plays.setdefault(int(row.category), set()).add(row.curve)
    out: dict[int, list] = {}
    for bone, setter, exprs, _bits in program:
        if setter in SCALE_SETTERS:
            kind, axes = "scale", SCALE_SETTERS[setter]
        elif setter in POSITION_SETTERS:
            kind, axes = "position", POSITION_SETTERS[setter]
        else:
            kind, axes = "rotation", {"Rotation": (0, 1, 2), "RotationX": (0,), "RotationY": (1,),
                                      "RotationZ": (2,)}[setter]
        for axis, expr in zip(axes, exprs):
            read: set = set()
            _curves_read(expr, read)
            for slider, curves in plays.items():
                if read & curves:
                    channel = Channel(bone, kind, axis)
                    listed = out.setdefault(slider, [])
                    if channel not in listed:
                        listed.append(channel)
    return out


def _turn(frame) -> np.ndarray:
    """A frame's rotation, its columns' scale left out."""
    m = np.asarray(frame, dtype=np.float64)[:3, :3]
    return m / np.maximum(np.linalg.norm(m, axis=0), 1e-12)


def spread(body: Body, bone: str, axis: int) -> float | None:
    """RMS of *bone*'s skin about its centroid along the bone's *axis*, over the body's scale.

    None when fewer than :data:`MIN_POINTS` points hang on it.
    """
    points = body.region(bone)
    if len(points) < MIN_POINTS or bone not in body.frames:
        return None
    along = (points - points.mean(axis=0)) @ _turn(body.frames[bone])[:, axis]
    return float(np.sqrt(np.mean(along * along))) / body.scale


def placed_ancestor(body: Body, bone: str) -> str | None:
    """The nearest ancestor of *bone* that :attr:`Body.placed` holds, or None."""
    parent = body.parents.get(bone)
    seen = set()
    while parent is not None and parent not in seen:
        if parent in body.placed:
            return parent
        seen.add(parent)
        parent = body.parents.get(parent)
    return None


def place(body: Body, bone: str, axis: int, ancestor: str) -> float | None:
    """*bone*'s place along *ancestor*'s *axis*, from the ancestor, over the body's scale."""
    if bone not in body.frames or ancestor not in body.frames:
        return None
    a = np.asarray(body.frames[ancestor], dtype=np.float64)
    b = np.asarray(body.frames[bone], dtype=np.float64)
    return float(_turn(a)[:, axis] @ (b[:3, 3] - a[:3, 3])) / body.scale


@dataclass(frozen=True)
class Target:
    """One number of the drawn body the sliders are fitted to: how it is read and its value (metres)."""
    channel: Channel
    #: the ancestor a place is read from (``position``), else None
    ancestor: str | None
    value: float

    def read(self, body: Body) -> float | None:
        if self.channel.kind == "scale":
            return spread(body, self.channel.bone, self.channel.axis)
        return place(body, self.channel.bone, self.channel.axis, self.ancestor)


def targets(drawn: Body, reference: Body, channels: list) -> tuple[list, list]:
    """``(targets, unread)``: the channels of one slider *drawn* can be read at, and why each other one is not.

    *reference*: the source game's body, to check that it has the same skin
    and bones to read.
    """
    out, unread = [], []
    for channel in channels:
        if channel.kind == "rotation":
            unread.append((channel, "a turn is not read off a placed bone"))
            continue
        if channel.kind == "scale":
            value = spread(drawn, channel.bone, channel.axis)
            if value is None or spread(reference, channel.bone, channel.axis) is None:
                unread.append((channel, f"fewer than {MIN_POINTS} skin points hang on it"))
                continue
            out.append(Target(channel, None, value))
            continue
        if channel.bone not in drawn.placed:
            unread.append((channel, "the body did not place this bone"))
            continue
        ancestor = placed_ancestor(drawn, channel.bone)
        if ancestor is None or ancestor not in reference.frames or channel.bone not in reference.frames:
            unread.append((channel, "no placed ancestor to read its place from"))
            continue
        out.append(Target(channel, ancestor, place(drawn, channel.bone, channel.axis, ancestor)))
    return out, unread


def misfit(body: Body, wanted: list) -> np.ndarray:
    """*body*'s readings less the *wanted* targets' values, in metres at the source game's size."""
    out = np.empty(len(wanted))
    for i, t in enumerate(wanted):
        got = t.read(body)
        if got is None:
            raise ValueError(f"the source game's body has nothing to read at {t.channel}")
        out[i] = got - t.value
    return out


@dataclass
class Reading:
    """What was found for one slider."""
    slider: int
    value: float | None
    #: the channels it was fitted to, and the ones it could not be read at (with why)
    read: list = field(default_factory=list)
    unread: list = field(default_factory=list)
    #: RMS of its targets' misfit at the value, in millimetres at the source game's size
    residual_mm: float | None = None
    #: how far its targets move per unit of the slider that no other slider can move them (mm):
    #: the column of the fit's Jacobian less its projection on the others'
    alone_mm: float | None = None
    #: the value is at one end of :data:`SLIDER_LIMITS`: the body is past what the slider reaches
    at_limit: bool = False
    #: why it was given no value, when it was not
    why: str = ""

    @property
    def measured(self) -> bool:
        return self.value is not None


def _alone(jac: np.ndarray) -> np.ndarray:
    """Per column of *jac*, its length after the part the other columns can make is taken off."""
    out = np.zeros(jac.shape[1])
    for j in range(jac.shape[1]):
        others = np.delete(jac, j, axis=1)
        column = jac[:, j]
        if others.shape[1]:
            fit, *_rest = np.linalg.lstsq(others, column, rcond=None)
            column = column - others @ fit
        out[j] = float(np.linalg.norm(column))
    return out


def heaviest(skin: np.ndarray, names: list) -> np.ndarray:
    """Per skin point, the name of the bone it hangs on most (an archived ``[weights, indices]`` skin)."""
    from ..mesh.skin import split_archive_skin
    weights, indices = split_archive_skin(np.asarray(skin, dtype=np.float64), bone_count=len(names))
    pick = indices[np.arange(len(indices)), np.argmax(weights, axis=1)]
    return np.asarray(names, dtype=object)[pick]


class OriginBody:
    """The source game's body under a slider array: its skin shaped as a woman's card with those sliders.

    *piece*: the source game's body skin (``o_body_a``: mesh, bone names,
    archived skin, binds); *reference*: the reference rig at scale 1 holding
    every bone the skin hangs on.  Called with an array, it runs the game's body shape
    program on the reference rig put back to its rest (``shape_apply.apply_body_shape``,
    the height's ``cf_n_height`` left to the scale as on a card, §7.9) and skins
    the piece with it.  The bust's hang under its own gravity
    (``bust.apply_bust_shape``) is not run: it turns the chain, which a
    spread along the bones' own axes does not read, and it is 600 solver steps
    a call.
    """

    def __init__(self, piece, reference, rows=None):
        """*rows*: the skin points to keep (the ones the drawn body shows, by the piece's own order); all without."""
        from ..mesh.skeleton import Skeleton
        from ..mesh.skin import split_archive_skin
        missing = sorted(n for n in piece.bones if not reference.has(n))
        if missing:
            raise ValueError(f"the reference rig lacks the skin's bones {missing[:5]}")
        # one rig, put back to its rest before every run (a fresh one per run was 112 ms a call)
        self._sk = Skeleton("shape_measure")
        for b in reference.bones:
            self._sk.add(b.name, reference.bones[b.parent].name if b.parent >= 0 else None,
                         b.local_pos, b.local_rot, np.asarray(b.local_scale, dtype=np.float64))
        self._sk.finalize()
        self._rest = [(b.local_pos.copy(), b.local_rot.copy(), np.asarray(b.local_scale, dtype=np.float64).copy())
                      for b in self._sk.bones]
        # a piece may list a bone past its binds (a man's ``o_body_a``: 101 bones, 100 binds, the last
        # ``cf_d_bust00``); only the bound ones skin it (``hrole.StandIn.source_edges``)
        self.names = list(piece.bones[:len(piece.binds)])
        weights, indices = split_archive_skin(np.asarray(piece.skin, dtype=np.float64), bone_count=len(piece.binds))
        keep = np.arange(len(weights)) if rows is None else np.asarray(rows, dtype=np.int64)
        self._weights, self._indices = weights[keep], indices[keep]
        self._binds = np.stack([np.asarray(m, dtype=np.float64) for m in piece.binds])
        self._skin_rows = np.array([self._sk.index[n] for n in self.names], dtype=np.int64)
        self._points = np.c_[np.asarray(piece.mesh.positions, dtype=np.float64)[keep], np.ones(len(keep))]
        self.owner = heaviest(piece.skin, self.names)[keep]
        self.parents = {b.name: (self._sk.bones[b.parent].name if b.parent >= 0 else None) for b in self._sk.bones}
        self._regions = regions(self.owner)

    def worlds(self, values) -> np.ndarray:
        """Every bone's world (K×4×4, the rig's order) with the body shaped by *values*."""
        from types import SimpleNamespace

        from ..mesh.skeleton import _compose_many
        from .figure import shape_apply
        sk = self._sk
        for bone, (pos, rot, scale) in zip(sk.bones, self._rest):
            bone.local_pos, bone.local_rot, bone.local_scale = pos.copy(), rot.copy(), scale.copy()
        card = SimpleNamespace(body=lambda: {"shapeValueBody": list(values)}, sex=1, parameter={"sex": 1},
                               sections={})
        report = shape_apply.apply_body_shape(sk, card, 1.0, finalize=False)
        if report.reason:
            raise ValueError(report.describe())
        local = _compose_many(np.stack([b.local_pos for b in sk.bones]), np.stack([b.local_rot for b in sk.bones]),
                              np.stack([np.broadcast_to(b.local_scale, 3) for b in sk.bones]))
        world = local.copy()
        parents = sk._parents
        for rows in sk._levels[1:]:
            world[rows] = world[parents[rows]] @ local[rows]
        return world

    def __call__(self, values) -> Body:
        world = self.worlds(values)
        skin = world[self._skin_rows] @ self._binds
        blend = np.einsum("vk,vkij->vij", self._weights, skin[self._indices])
        points = np.einsum("vij,vj->vi", blend, self._points)[:, :3]
        frames = dict(zip(self._sk._names, world))
        return Body(points, self.owner, frames, self.parents, set(frames), 1.0, self._regions)


def _fit(drawn_targets: list, model: Callable[[list], Body], values: list, free: list) -> tuple:
    """``(values, Jacobian in metres per unit of each free slider)``: the free sliders fitted together."""
    from scipy.optimize import least_squares

    def fill(x) -> list:
        out = list(values)
        for k, v in zip(free, x):
            out[k] = float(v)
        return out

    found = least_squares(lambda x: misfit(model(fill(x)), drawn_targets), np.array([values[k] for k in free]),
                          bounds=SLIDER_LIMITS, diff_step=DIFF_STEP, xtol=XTOL)
    return fill(found.x), np.asarray(found.jac, dtype=np.float64)


def _reach(model: Callable[[list], Body], values: list, k: int, mine: list) -> float:
    """How far the whole range of slider *k* moves its targets (RMS, mm), the others as in *values*."""
    ends = []
    for end in SLIDER_LIMITS:
        trial = list(values)
        trial[k] = end
        ends.append(misfit(model(trial), mine))
    return float(np.sqrt(np.mean((ends[1] - ends[0]) ** 2))) * 1000.0


def measure(drawn: Body, model: Callable[[list], Body], start: list, channels: dict, *,
            sliders=None) -> tuple[list, list]:
    """``(values, readings)``: the sliders of *sliders* at which ``model(values)`` reads as *drawn* does.

    *model* is the source game's body under a full slider array (its skin
    and bones shaped by the game's program); *start* the array it starts
    from (the height already in it — it is the body's size, not measured
    here); *channels* :func:`slider_channels`.  The sliders are fitted
    together (bounded least squares over every target any of them reaches,
    in metres): one channel is shared by several (``cf_a_height_aid``
    multiplies many, ``cf_s_bust00``'s place is three sliders'), and one at
    a time they stuck where a shared target could not be met by any alone.
    A slider is **not measured** — kept at *start*, :attr:`Reading.why` says
    why — when none of its channels can be read on this body, or when one
    editor step of it moves its targets, apart from what the other sliders
    can do, by less than the stored body's own precision (:data:`PRECISION_MM`):
    the body then does not tell it from the others — or when the fit leaves
    its targets further from the body than the slider's whole range moves
    them (:func:`_reach`): the source game's body does not take that shape
    there, and a value would say nothing of it.  Either way the others are
    fitted again without it.
    """
    values = [float(v) for v in start]
    if len(values) != SLIDER_COUNT:
        raise ValueError(f"{len(values)} sliders, not {SLIDER_COUNT}")
    sliders = [k for k in (range(SLIDER_COUNT) if sliders is None else sliders) if k != HEIGHT]
    reference = model(values)
    readings: dict[int, Reading] = {}
    wanted: dict[Target, list] = {}
    for k in sliders:
        found, unread = targets(drawn, reference, channels.get(k, []))
        readings[k] = Reading(k, None, [t.channel for t in found], unread)
        if not found:
            readings[k].why = "none of its channels can be read on this body"
        for t in found:
            wanted.setdefault(t, []).append(k)
    free = [k for k in sliders if readings[k].read]
    while free:
        drawn_targets = [t for t, owners in wanted.items() if any(k in free for k in owners)]
        values, jac = _fit(drawn_targets, model, values, free)
        miss = misfit(model(values), drawn_targets) * 1000.0
        alone = _alone(jac) * 1000.0
        weak = []
        for j, k in enumerate(free):
            mine = [i for i, t in enumerate(drawn_targets) if k in wanted[t]]
            reading = readings[k]
            reading.alone_mm = float(alone[j])
            reading.residual_mm = float(np.sqrt(np.mean(miss[mine] ** 2)))
            reach = _reach(model, values, k, [drawn_targets[i] for i in mine])
            if reading.alone_mm * STEP < PRECISION_MM:
                reading.why = (f"not determined: one step moves its targets {reading.alone_mm * STEP:.2e} mm "
                               f"apart from the other sliders, under the body's {PRECISION_MM:.0e} mm")
            elif reading.residual_mm > reach:
                # left further from the body than the whole slider moves it: the source game's body
                # does not take this shape there, and the value says nothing about it
                reading.why = (f"the source game's body does not take this body's shape there: "
                               f"{reading.residual_mm:.2f} mm left, the whole slider moves it {reach:.2f} mm")
            else:
                continue
            weak.append(k)
        if not weak:
            break
        for k in weak:
            values[k] = float(start[k])
            readings[k].residual_mm = None
        free = [k for k in free if k not in weak]
    for k in free:
        readings[k].value = values[k]
        readings[k].at_limit = min(abs(values[k] - end) for end in SLIDER_LIMITS) < STEP
    return values, [readings[k] for k in sliders]
