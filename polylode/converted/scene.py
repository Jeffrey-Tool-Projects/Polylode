"""Source-qualified native actor batches in the real scene renderer.

Backends must prepare every visible original material slot and both rebuilt
surfaces before the renderer writes a frame. This routing layer does not invent
materials, certify a backend, or change warehouse/card acceptance.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self

import numpy as np

from .clock import FrameTime

if TYPE_CHECKING:
    from ..gl.renderer import RenderScene
    from .controller import ActorController, NativeFrame

SOURCES = frozenset(('gf2', 'wuthering_waves'))
Surface = tuple[str, int, int]


class NativeSceneError(ValueError):
    """An actual source pipeline or required native surface/input is unavailable."""


@dataclass
class NativeItem:
    """A source controller on stage, independent of generic DrawItem/Model materials."""
    controller: ActorController
    visible: bool = True


@dataclass(frozen=True)
class SceneView:
    """The renderer's actual camera matrices and scene settings for this frame."""
    scene: RenderScene
    width: int
    height: int
    view: np.ndarray
    projection: np.ndarray
    eye: np.ndarray
    clock: FrameTime | None = None

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int or min(self.width, self.height) <= 0:
            raise NativeSceneError('native scene dimensions must be positive integers')
        if self.clock is not None and not isinstance(self.clock, FrameTime):
            raise NativeSceneError('native scene frame needs an immutable host clock snapshot')
        for name, shape in (('view', (4, 4)), ('projection', (4, 4)), ('eye', (3,))):
            matrix = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if matrix.shape != shape or not np.isfinite(matrix).all():
                raise NativeSceneError('native scene camera requires complete finite matrices and eye')
            matrix.setflags(write=False)
            object.__setattr__(self, name, matrix)


@dataclass(frozen=True)
class SceneTarget:
    """The actual bound scene destination; a backend owns any additional source MRTs."""
    framebuffer: int
    samples: int
    srgb: bool


@dataclass(frozen=True)
class PlacedActor:
    """One frame snapshot and its exact required surface/material identities."""
    index: int
    ident: str
    frame: NativeFrame
    surfaces: frozenset[Surface]


class NativeBatch(Protocol):
    """One source's prepared actors, with explicit material/scene readiness and draw lifetime."""
    source: str
    surfaces: Mapping[int, frozenset[Surface]]
    missing_inputs: tuple[str, ...]

    def opaque(self, target: SceneTarget) -> None:
        """Draw source opaque passes, retaining the caller's GPU state."""

    def transparent(self, target: SceneTarget) -> None:
        """Draw the source's ordered transparent/composite passes and restore GPU state."""

    def close(self) -> None:
        """Release this frame's acquired resources, including after preparation/draw failure."""


class NativeBackend(Protocol):
    """A verified source implementation; no implicit alias or generic material fallback."""
    source: str

    def prepare(self, actors: tuple[PlacedActor, ...], view: SceneView) -> NativeBatch:
        """Acquire complete geometry/material/scene inputs without writing any destination pixels."""

    def release(self) -> None:
        """Release the backend's persistent source caches while the owning GL context is current."""


class PreparedScene:
    """All source batches prepared atomically; both draw phases use the same pose snapshots."""

    def __init__(self, batches: tuple[NativeBatch, ...], lifetime: ExitStack):
        self._batches = batches
        self._lifetime = lifetime
        self._closed = False

    def __enter__(self) -> Self:
        if self._closed:
            raise NativeSceneError('native scene preparation has already been released')
        return self

    @property
    def active(self) -> bool:
        """Whether this prepared frame contains a native source batch."""
        return bool(self._batches)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def opaque(self, target: SceneTarget) -> None:
        """Run the prepared source opaque phases; a closed batch cannot draw."""
        self._draw('opaque', target)

    def transparent(self, target: SceneTarget) -> None:
        """Run the prepared source transparent phases, with no new pose evaluation."""
        self._draw('transparent', target)

    def _draw(self, phase: str, target: SceneTarget) -> None:
        if self._closed:
            raise NativeSceneError('native scene preparation has already been released')
        for batch in self._batches:
            getattr(batch, phase)(target)

    def close(self) -> None:
        """Release all prepared batches once, even when a later source or draw fails."""
        if not self._closed:
            self._closed = True
            self._lifetime.close()


def _snapshot(index: int, item: NativeItem) -> tuple[str, PlacedActor]:
    controller = item.controller
    frame = controller.frame()
    metadata = frame.actor.geometry.metadata
    if (controller.source not in SOURCES or metadata['source'] != controller.source
            or metadata['ident'] != controller.ident):
        raise NativeSceneError('native scene controller and warehouse have different source identities')
    needed = {('donor', 0, 0), ('inner', 0, 0)}
    for mesh_index, _name, visible in controller.parts():
        if visible:
            mesh = frame.actor.meshes[mesh_index]
            needed.update(('native', mesh_index, int(slot)) for slot in np.unique(mesh.slots))
    return controller.source, PlacedActor(index, controller.ident, frame, frozenset(needed))


class NativeStage:
    """Route actual warehouse sources; incomplete backends fail before any scene draws."""

    def __init__(self) -> None:
        self._backends: dict[str, NativeBackend] = {}

    def register(self, source: str, backend: NativeBackend) -> None:
        """Install a source's explicit backend once; reject aliases and accidental replacement."""
        if source not in SOURCES or backend.source != source or source in self._backends:
            raise NativeSceneError('invalid, aliased or duplicate native scene backend')
        self._backends[source] = backend

    def place(self, controller: ActorController, view: SceneView, *, replace: NativeItem | None = None) -> NativeItem:
        """Preflight the whole prospective scene before inserting/replacing an actor.

        Use with the renderer's context current, just like prepare. Candidates
        for appearance review may use this same route without changing their
        warehouse/card acceptance. A failed load leaves the current scene intact.
        """
        scene = view.scene
        items = list(scene.native_items)
        item = NativeItem(controller)
        if replace is None:
            items.append(item)
        else:
            matches = [index for index, entry in enumerate(items) if entry is replace]
            if len(matches) != 1:
                raise NativeSceneError('replaced native scene item is missing or duplicated')
            items[matches[0]] = item
        with self.prepare(items, view):
            pass
        scene.native_items[:] = items
        return item

    @staticmethod
    def remove(item: NativeItem, view: SceneView) -> None:
        """Remove exactly the selected placed instance, retaining other same-source/same-id actors."""
        items = view.scene.native_items
        matches = [index for index, entry in enumerate(items) if entry is item]
        if len(matches) != 1:
            raise NativeSceneError('removed native scene item is missing or duplicated')
        del items[matches[0]]

    def release(self) -> None:
        """Release all source backends once when the renderer's context is current."""
        with ExitStack() as lifetime:
            for backend in self._backends.values():
                lifetime.callback(backend.release)
            self._backends.clear()

    def prepare(self, items: Sequence[NativeItem], view: SceneView) -> PreparedScene:
        """Snapshot poses and validate all required source batches before returning a draw lifetime."""
        groups: dict[str, list[PlacedActor]] = {}
        for index, item in enumerate(items):
            if item.visible:
                source, actor = _snapshot(index, item)
                groups.setdefault(source, []).append(actor)
        missing = groups.keys() - self._backends.keys()
        if missing:
            raise NativeSceneError('complete native scene backend unavailable: ' + ', '.join(sorted(missing)))
        batches = []
        with ExitStack() as lifetime:
            for source, actors in groups.items():
                batch = self._backends[source].prepare(tuple(actors), view)
                lifetime.callback(batch.close)
                expected = {actor.index: actor.surfaces for actor in actors}
                if batch.source != source or dict(batch.surfaces) != expected:
                    raise NativeSceneError('native batch does not cover its own complete actor/material surfaces')
                if batch.missing_inputs:
                    raise NativeSceneError(source + ': ' + '; '.join(batch.missing_inputs))
                batches.append(batch)
            return PreparedScene(tuple(batches), lifetime.pop_all())
