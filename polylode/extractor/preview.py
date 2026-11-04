"""The preview before an export: the model converted to ``.glb`` by the same job an export runs, shown turnable.

A preview is an export of one format, ``.glb`` (the one Qt Quick 3D reads,
through ``RuntimeLoader``), into a cache of this program's own
(:func:`cache_folder`), without Blender's read-back: it is for looking at,
the files a user keeps come from the export queue.  A model previewed once is
shown from the cache the next time.  ``preview.qml`` frames the model by its
bounds; the mouse turns (left drag), pans (right drag) and zooms (wheel).  A model with display poses (its glTF
animations) plays the first, and the panel lists them to switch between (``view_host.Poses``).
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import json

from PySide6.QtCore import QObject, QProcess, Qt, QUrl, Signal
from PySide6.QtGui import QVector3D, QWindow
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..core.paths import derived_root
from . import settings as S
from .sources import Model
from .view_host import encode

QML = Path(__file__).with_name("preview.qml")
#: the writers' rules a cached preview was made with: raised when what a .glb holds changes, so an earlier
#: preview is not shown again (2: overlays as their tint by coverage; 3: GF2 eye layers and transparency,
#: WuWa translucency, §7.1455; 5: vertex colours kept out of COLOR_0, which hid
#: GF2's characters, and GF2's lower LOD levels hidden; 6: Wuthering Waves split by material; 7: Wuthering
#: Waves' cut-outs from the game's shader, parallax fur as strand layers; 8: GF2's spare outfit parts and
#: dormitory variants hidden; 9: every dormitory part of a model that is not the dormitory one; 10: every display
#: pose, not only the held ones; 11: a preview carries the poses, to switch between, GF2's and Endfield's poses
#: their scale, and no cutscene clip that pulls the body apart)
CACHE_VERSION = 11


def cache_folder(model: Model) -> Path:
    """Where *model*'s preview is written: one folder per source and model."""
    safe = re.sub(r"[^A-Za-z0-9._@-]", "_", model.ident)
    return derived_root() / "assets_cache" / "extractor" / "preview" / f"v{CACHE_VERSION}" / model.source / safe


def cached(model: Model) -> Path | None:
    """The ``.glb`` an earlier preview of *model* wrote, when there is one."""
    return next(iter(sorted(cache_folder(model).rglob("*.glb"))), None)


def fresh_folder(model: Model) -> Path:
    """Where a preview made again is written, beside the cache: the earlier one stays until it is replaced."""
    folder = cache_folder(model)
    return folder.with_name(folder.name + ".new")


def adopt(model: Model, written: Path) -> Path:
    """The preview made again in :func:`fresh_folder` (*written* in it) made *model*'s cached one: the earlier
    folder removed and the new one put in its place.  When the earlier one cannot be removed (a file in it
    held open), the new one is shown from where it was written and the cache keeps the earlier."""
    import shutil
    folder, fresh = cache_folder(model), fresh_folder(model)
    try:
        if folder.exists():
            shutil.rmtree(folder)
        fresh.replace(folder)
    except OSError:
        return Path(written)
    return folder / Path(written).relative_to(fresh)


def preview_settings(settings: S.Settings) -> S.Settings:
    """*settings* for a preview job: ``.glb`` only, no Blender (its read-back takes minutes).  The poses follow the
    settings: decoding a character's clips adds tens of seconds, and with them the preview switches between them."""
    return replace(settings, formats=["glb"], tools={k: v for k, v in settings.tools.items() if k != "blender"})


#: the poses of :func:`triangle_glb`'s triangle: what :func:`load_check` finds the view reading
TRIANGLE_POSES = ("rise", "slide")


def triangle_glb(path: Path) -> Path:
    """A one-triangle ``.glb`` at *path*, moved by two animations (:data:`TRIANGLE_POSES`): what
    :func:`load_check` reads."""
    import json
    import struct
    points = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    times = struct.pack("<2f", 0, 1)
    moves = [struct.pack("<6f", 0, 0, 0, 0, 1, 0), struct.pack("<6f", 0, 0, 0, 1, 0, 0)]
    positions = points + times + b"".join(moves)
    views = [(0, len(points)), (len(points), len(times))] + \
        [(len(points) + len(times) + 24 * k, 24) for k in range(len(moves))]
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "buffers": [{"byteLength": len(positions)}],
                "bufferViews": [{"buffer": 0, "byteOffset": offset, "byteLength": size} for offset, size in views],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                               "min": [0, 0, 0], "max": [1, 1, 0]},
                              {"bufferView": 1, "componentType": 5126, "count": 2, "type": "SCALAR",
                               "min": [0], "max": [1]}]
                + [{"bufferView": 2 + k, "componentType": 5126, "count": 2, "type": "VEC3"} for k in range(len(moves))],
                "animations": [{"name": name, "samplers": [{"input": 1, "output": 2 + k, "interpolation": "LINEAR"}],
                                "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}]}
                               for k, name in enumerate(TRIANGLE_POSES)]}
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(positions), 0x004E4942) + positions
    Path(path).write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
    return Path(path)


def use_opengl() -> None:
    """Have Qt Quick draw with OpenGL, with a depth and stencil buffer: what :func:`load_check` renders into.

    Called before the application is made; with Qt Quick on its default (Direct3D) the
    OpenGL render target of :func:`load_check` crashed the process.
    """
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
    surface = QSurfaceFormat()
    surface.setDepthBufferSize(24)
    surface.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(surface)
    QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGL)


class Offscreen:
    """``preview.qml`` drawn with no window at all (``QQuickRenderControl`` into an OpenGL framebuffer), the way
    the window draws it.  Needs a ``QGuiApplication``; :attr:`failure` says why it could not start."""

    def __init__(self, width: int = 64, height: int = 64):
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QOffscreenSurface, QOpenGLContext
        from PySide6.QtOpenGL import QOpenGLFramebufferObject
        from PySide6.QtQml import QQmlComponent, QQmlEngine
        from PySide6.QtQuick import QQuickGraphicsDevice, QQuickRenderControl, QQuickRenderTarget, QQuickWindow
        self.failure, self.root = "", None
        self.context = QOpenGLContext()
        if not self.context.create():
            self.failure = "no OpenGL context"
            return
        self.surface = QOffscreenSurface()
        self.surface.setFormat(self.context.format())
        self.surface.create()
        self.control = QQuickRenderControl()
        self.window = QQuickWindow(self.control)
        self.window.setGraphicsDevice(QQuickGraphicsDevice.fromOpenGLContext(self.context))
        self.context.makeCurrent(self.surface)
        if not self.control.initialize():
            self.failure = "the render control did not start"
            return
        size = QSize(width, height)
        self.target = QOpenGLFramebufferObject(size, QOpenGLFramebufferObject.CombinedDepthStencil)
        self.window.setRenderTarget(QQuickRenderTarget.fromOpenGLTexture(self.target.texture(), size))
        self.window.setGeometry(0, 0, width, height)
        self.engine = QQmlEngine()
        # kept: a component collected when this returns takes the item it made with it
        self.component = QQmlComponent(self.engine, QUrl.fromLocalFile(str(QML)))
        self.root = self.component.create()
        if self.root is None:
            self.failure = "; ".join(e.toString() for e in self.component.errors()) or "no root item"
            return
        self.root.setParentItem(self.window.contentItem())
        self.root.setWidth(width)
        self.root.setHeight(height)

    def frame(self) -> None:
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.instance().processEvents()
        self.context.makeCurrent(self.surface)
        self.control.polishItems()
        self.control.beginFrame()
        self.control.sync()
        self.control.render()
        self.control.endFrame()

    def load(self, path: Path, seconds: float = 30) -> str:
        """*path* shown, framed as the window frames it; the loader's state once it has settled."""
        import time
        box = model_bounds(path)
        self.root.setProperty("framed", box is not None)
        if box is not None:
            self.root.setProperty("frameLow", QVector3D(*box[0]))
            self.root.setProperty("frameHigh", QVector3D(*box[1]))
        self.root.setProperty("source", QUrl.fromLocalFile(str(path)))
        deadline = time.monotonic() + seconds
        while True:
            self.frame()
            state = self.root.property("state_")
            if state not in ("", "loading") or time.monotonic() > deadline:
                return state
            time.sleep(0.02)

    def image(self):
        """The frame now drawn, as a ``QImage`` (a few frames first: the view settles its framing)."""
        for _ in range(6):
            self.frame()
        self.context.makeCurrent(self.surface)
        return self.target.toImage()


def load_check(glb: Path, seconds: float = 30) -> dict:
    """``preview.qml`` loading *glb* the way the window does, drawn off-screen (:class:`Offscreen`): its loader
    state, size and the poses the view's process would list (``view_host.Poses``).

    The packaged program's ``--selftest`` reads a triangle with it: a missing Qt Quick 3D module or
    importer plugin only shows when a file is actually loaded, and the poses need Qt Quick Timeline, which the
    loader makes them with.  Needs a ``QGuiApplication``.
    """
    from .view_host import Poses
    view = Offscreen()
    if view.failure:
        return {"state": view.failure, "size": 0.0, "poses": []}
    state = view.load(glb, seconds)
    poses = Poses().read(view.root) if state == "ready" else []
    return {"state": state, "size": float(view.root.property("size")), "poses": poses}


def snapshot(path: Path, out: Path, width: int, height: int, seconds: float = 120) -> Path:
    """*path* drawn as the preview shows it, from the front and framed whole, into the image *out*.
    Raises ``ValueError`` when the file does not load.  Needs a ``QGuiApplication``."""
    view = Offscreen(width, height)
    if view.failure:
        raise ValueError(view.failure)
    state = view.load(shown_copy(path), seconds)
    if state != "ready":
        raise ValueError(f"{Path(path).name} did not load: {state}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not view.image().save(str(out)):
        raise ValueError(f"{out} could not be written")
    return out


#: display copies kept (:func:`shown_copy`): the newest few, a viewer goes back and forth between files
SHOWN_KEPT = 8


#: characters Qt's model loader cannot open a file through (its URL takes them as markers): measured — a path
#: with "@", "#" or "%" loads nothing, with spaces, CJK, "&", "+" or ";" it draws
UNSAFE = frozenset("@#%")


def shown_copy(path: Path) -> Path:
    """*path* as Qt's loader can draw it, as a viewer honouring ``KHR_node_visibility`` would.

    Two things make the loader draw a file wrongly: hidden mesh nodes, which it draws all the same (GF2's lower
    LOD levels over the most detailed one), and a path holding ``UNSAFE`` characters, which it opens as nothing
    (the preview cache's ``<model>@<hash>`` folders: a GF2 preview came out blank once its ``.glb`` had no hidden
    part and was shown from where it was written).  Either way the file is shown from a copy beside the cache,
    named by a hash: a ``.glb`` with hidden nodes taking them off (only its JSON chunk changes), any other ``.glb``
    or ``.fbx`` as it is.  The copy is kept by the file's path, size and time, so showing it again does not write
    it again; the file itself is left alone.  Anything else is shown as it is.
    """
    import hashlib
    import json
    import struct
    path = Path(path)
    suffix = path.suffix.lower()
    unsafe = any(c in str(path.resolve()) for c in UNSAFE)
    document, hidden, data, length = None, [], b"", 0
    if suffix == ".glb":
        try:
            data = path.read_bytes()
            length = struct.unpack_from("<I", data, 12)[0]
            document = json.loads(data[20:20 + length])
        except (OSError, ValueError, struct.error):
            return path
        hidden = [node for node in document.get("nodes", []) if "mesh" in node
                  and not (node.get("extensions") or {}).get("KHR_node_visibility", {}).get("visible", True)]
    if not hidden and not (unsafe and suffix in (".glb", ".fbx")):
        return path
    stat = path.stat()
    key = hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:20]
    folder = derived_root() / "assets_cache" / "extractor" / "shown"
    copy = folder / f"{key}{suffix}"
    if copy.is_file():
        return copy
    folder.mkdir(parents=True, exist_ok=True)
    partial = copy.with_suffix(".part")
    if hidden:
        for node in hidden:
            del node["mesh"]
            node.pop("skin", None)
            node.pop("weights", None)
        text = json.dumps(document, ensure_ascii=False).encode("utf-8")
        text += b" " * (-len(text) % 4)
        rest = data[20 + length:]
        partial.write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(text) + len(rest))
                            + struct.pack("<II", len(text), 0x4E4F534A) + text + rest)
    else:
        import shutil
        shutil.copyfile(path, partial)
    partial.replace(copy)
    kept = sorted([*folder.glob("*.glb"), *folder.glob("*.fbx")], key=lambda p: p.stat().st_mtime, reverse=True)
    for old in kept[SHOWN_KEPT:]:
        old.unlink(missing_ok=True)
    return copy
    for node in hidden:
        del node["mesh"]
        node.pop("skin", None)
        node.pop("weights", None)
    text = json.dumps(document, ensure_ascii=False).encode("utf-8")
    text += b" " * (-len(text) % 4)
    rest = data[20 + length:]
    folder.mkdir(parents=True, exist_ok=True)
    partial = copy.with_suffix(".part")
    partial.write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(text) + len(rest))
                        + struct.pack("<II", len(text), 0x4E4F534A) + text + rest)
    partial.replace(copy)
    for old in sorted(folder.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True)[SHOWN_KEPT:]:
        old.unlink(missing_ok=True)
    return copy


def gltf_bounds(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """The box a ``.glb``／``.gltf`` file's meshes fill, from its own data: each ``POSITION`` accessor's
    ``min``／``max`` (required by the glTF spec) through its node's world transform.  A skinned mesh's
    positions are taken as they are: its node's transform does not apply (the joints place it), and at rest
    they are where the skin puts them.  None when the file is not glTF or carries no positions.

    The preview frames a model with it: Qt's loader reports an empty box for a skinned model, and the
    camera then sat 2 cm from the model's middle (every GF2 preview looked empty).
    """
    import json
    import struct
    import numpy as np
    path = Path(path)
    try:
        if path.suffix.lower() == ".glb":
            data = path.read_bytes()
            if data[:4] != b"glTF":
                return None
            document = json.loads(data[20:20 + struct.unpack_from("<I", data, 12)[0]])
        elif path.suffix.lower() == ".gltf":
            document = json.loads(path.read_text(encoding="utf-8"))
        else:
            return None
    except (OSError, ValueError, struct.error):
        return None
    nodes = document.get("nodes", [])
    parent = {child: i for i, node in enumerate(nodes) for child in node.get("children", [])}

    def local(node) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], float).reshape(4, 4).T
        x, y, z, w = node.get("rotation", (0, 0, 0, 1))
        matrix = np.eye(4)
        matrix[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]) \
            * np.array(node.get("scale", (1, 1, 1)), float)
        matrix[:3, 3] = node.get("translation", (0, 0, 0))
        return matrix

    def world(index: int) -> np.ndarray:
        matrix = local(nodes[index])
        while index in parent:
            index = parent[index]
            matrix = local(nodes[index]) @ matrix
        return matrix

    corners = []
    for index, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        place = np.eye(4) if "skin" in node else world(index)
        for primitive in document["meshes"][node["mesh"]].get("primitives", []):
            accessor = document["accessors"][primitive["attributes"]["POSITION"]] \
                if "POSITION" in primitive.get("attributes", {}) else {}
            if "min" not in accessor or "max" not in accessor:
                continue
            low, high = accessor["min"], accessor["max"]
            box = np.array([[a, b, c, 1.0] for a in (low[0], high[0]) for b in (low[1], high[1]) for c in (low[2], high[2])])
            corners.append((place @ box.T).T[:, :3])
    if not corners:
        return None
    every = np.vstack(corners)
    return tuple(map(float, every.min(0))), tuple(map(float, every.max(0)))


def fbx_bounds(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """The box a binary FBX's geometry fills: each ``Geometry``'s ``Vertices`` through the ``Lcl``
    transforms of the ``Model`` chain it hangs from.  Other node properties are not read (a file from
    another tool may carry pivots, which move it a little; the box only frames the view).

    Qt's loader ignores ``UnitScaleFactor``: a file of ours (100) and the same file set to 1 drew at the
    same size, both at the raw box.  None for a file this cannot read (FBX 7.5's 64-bit records, text FBX).
    """
    import struct
    import numpy as np
    from ..endfield import fbxfile as F
    try:
        data = Path(path).read_bytes()
        top = F._top(data)
        geometries, models, parent = {}, {}, {}
        for name, props, start, stop in F._records(data, *top[b"Objects"]):
            if name == b"Geometry":
                for held, values, _a, _b in F._records(data, start, stop):
                    if held == b"Vertices":
                        geometries[int(props[0][1])] = F._array(data, values[0]).reshape(-1, 3)
            elif name == b"Model":
                lcl = {}
                for held, _p, inner, inner_stop in F._records(data, start, stop):
                    if held == b"Properties70":
                        for _q, values, _c, _d in F._records(data, inner, inner_stop):
                            key = values[0][1].decode("utf-8", "replace")
                            if key.startswith("Lcl "):
                                lcl[key] = [float(v[1]) for v in values[4:7]]
                models[int(props[0][1])] = lcl
        for _name, props, _start, _stop in F._records(data, *top[b"Connections"]):
            if props[0][1] == b"OO":
                parent.setdefault(int(props[1][1]), int(props[2][1]))
    except (OSError, F.FbxError, KeyError, IndexError, ValueError, struct.error):
        return None

    def local(lcl: dict) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = F.compose(lcl.get("Lcl Rotation", (0, 0, 0))) * np.array(lcl.get("Lcl Scaling", (1, 1, 1)))
        matrix[:3, 3] = lcl.get("Lcl Translation", (0, 0, 0))
        return matrix

    corners = []
    for uid, vertices in geometries.items():
        if not len(vertices):
            continue
        matrix, node = np.eye(4), parent.get(uid)
        while node in models:
            matrix = local(models[node]) @ matrix
            node = parent.get(node)
        placed = np.c_[vertices, np.ones(len(vertices))] @ matrix.T
        corners.append(np.array([placed[:, :3].min(0), placed[:, :3].max(0)]))
    if not corners:
        return None
    every = np.vstack(corners)
    return tuple(map(float, every.min(0))), tuple(map(float, every.max(0)))


def obj_bounds(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """The box of an ``.obj``'s ``v`` lines (an OBJ has no transforms); None when it has none."""
    low = high = None
    try:
        with open(path, encoding="utf-8", errors="replace") as lines:
            for line in lines:
                if line.startswith("v "):
                    point = [float(value) for value in line.split()[1:4]]
                    low = point if low is None else [min(a, b) for a, b in zip(low, point)]
                    high = point if high is None else [max(a, b) for a, b in zip(high, point)]
    except (OSError, ValueError):
        return None
    return (tuple(low), tuple(high)) if low is not None and len(low) == 3 else None


def model_bounds(path: Path):
    """The box the view frames *path* by: from the file's own data, by its kind; None when not measured."""
    suffix = Path(path).suffix.lower()
    if suffix in (".glb", ".gltf"):
        return gltf_bounds(path)
    if suffix == ".fbx":
        return fbx_bounds(path)
    if suffix == ".obj":
        return obj_bounds(path)
    return None


def load_errors() -> list[str]:
    """What keeps ``preview.qml`` from loading (a Qt Quick 3D module the program does not carry); ``[]`` when it loads.

    Needs a ``QGuiApplication``; the packaged program's ``--selftest`` asks it.
    """
    from PySide6.QtQml import QQmlComponent, QQmlEngine
    engine = QQmlEngine()
    component = QQmlComponent(engine, QUrl.fromLocalFile(str(QML)))
    errors = [error.toString() for error in component.errors()]
    if not errors and component.create() is None:
        errors = ["preview.qml did not create its root item"]
    return errors


class _Done(QObject):
    """Carries a background step's result back to the window's thread."""
    finished = Signal(object, object)


def background(owner: QObject, work, done) -> None:
    """*work* () run on a thread of its own, then *done* (result, error) on *owner*'s thread: file reading the
    window must not wait for (a model's display copy and box, a viewer's listing and file summary)."""
    import threading
    carrier = _Done(owner)
    carrier.finished.connect(lambda result, error: (done(result, error), carrier.deleteLater()), Qt.QueuedConnection)

    def run():
        try:
            result, error = work(), None
        except Exception as failure:  # noqa: BLE001 - handed to *done*, which says what went wrong
            result, error = None, failure
        carrier.finished.emit(result, error)

    threading.Thread(target=run, daemon=True, name="polylode-background").start()


class _RemoteRoot:
    """Where ``preview.qml``'s root object stood, now in the view's own process (:mod:`view_host`): a property set is
    sent there and remembered here (read back from here, and set again in a view that has to be started anew),
    ``look`` is sent."""

    DEFAULTS = {"hint": "選一個模型後按「預覽」", "hintSource": "", "loadingText": "載入中…",
                "errorText": "讀不進這個檔案："}

    def __init__(self, panel: "PreviewPanel"):
        self.panel = panel
        self.values: dict = {}
        #: every command sent, in order (what a test reads; a running view has applied them)
        self.sent: list = []

    def setProperty(self, name: str, value) -> None:
        self.values[name] = value
        self.panel._send({"op": "set", "name": name, "value": encode(value)})
        if name == "hint":
            self.panel.placeholder.setText(str(value))

    def property(self, name: str):
        return self.values.get(name, self.DEFAULTS.get(name))

    def look(self, yaw: float, pitch: float) -> None:
        self.panel._send({"op": "look", "yaw": float(yaw), "pitch": float(pitch)})


class _HostedView:
    """``rootObject()`` as ``QQuickWidget`` had it, for the pages that turn and style the view."""

    def __init__(self, root: _RemoteRoot):
        self._root = root

    def rootObject(self) -> _RemoteRoot:
        return self._root


class PreviewPanel(QWidget):
    """The turnable view of one model and a line saying what is shown.

    The view runs in a process of its own (:mod:`view_host`), started the first time a model is shown, and its
    window is shown inside this one: Qt Quick 3D reads a model on the thread that renders it, in one go, and a large
    one held the whole window for seconds.  What is read before the view is given a file (its display copy and box)
    is read on a thread (:func:`background`); a result for a file no longer asked for is dropped.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.root = _RemoteRoot(self)
        self.view = _HostedView(self.root)
        #: stands where the view's window goes until its process has given it
        self.placeholder = QLabel(self.root.property("hint"))
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setObjectName("muted")
        self.placeholder.setWordWrap(True)
        self.placeholder.setMinimumSize(200, 200)
        self.placeholder.setStyleSheet("background: #1e2129;")
        layout.addWidget(self.placeholder, 1)
        self.container: QWidget | None = None
        self.host: QProcess | None = None
        self._host_lines = b""
        self._asked = 0
        from .stepbar import StepBar
        # the conversion a preview waits for (the file's own load is the view's bar)
        self.steps = StepBar()
        layout.addWidget(self.steps)
        # the shown model's poses, as the view's process lists them once it has loaded the file (hidden with none)
        self.pose_row = QWidget()
        row = QHBoxLayout(self.pose_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("姿勢"))
        self.pose_list = QComboBox()
        self.pose_list.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.pose_list.setMinimumContentsLength(12)
        self.pose_list.currentIndexChanged.connect(self._pose_chosen)
        row.addWidget(self.pose_list, 1)
        self.pose_row.hide()
        layout.addWidget(self.pose_row)
        #: the file shown (the view may read a display copy of it, :func:`shown_copy`)
        self.current: Path | None = None
        self.caption = QLabel("左鍵拖曳旋轉、右鍵平移、滾輪縮放")
        self.apply_texts()
        self.caption.setObjectName("muted")
        self.caption.setWordWrap(True)
        # a path has nowhere to break: the caption takes the width it is given rather than widening the panel
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.caption)

    # -- the view's process -----------------------------------------------------------------------------------
    def _ensure_host(self) -> None:
        if self.host is not None:
            return
        from . import jobs
        self.host = QProcess(self)
        self.host.setProcessChannelMode(QProcess.SeparateChannels)
        self.host.readyReadStandardOutput.connect(self._host_output)
        self.host.finished.connect(self._host_ended)
        program, *arguments = jobs.view_host_command()
        self.host.start(program, arguments)
        # a view started anew (its process ended) takes up where the last one was
        for name, value in self.root.values.items():
            self._write({"op": "set", "name": name, "value": encode(value)})

    def _write(self, command: dict) -> None:
        if self.host is not None:
            self.host.write((json.dumps(command) + "\n").encode("utf-8"))

    def _send(self, command: dict) -> None:
        self.root.sent.append(command)
        self._write(command)

    def _host_output(self) -> None:
        self._host_lines += bytes(self.host.readAllStandardOutput())
        *lines, self._host_lines = self._host_lines.split(b"\n")
        for raw in lines:
            self._host_said(raw.decode("utf-8", "replace"))

    def _host_said(self, line: str) -> None:
        """One line the view's process wrote: its window, the poses of the model it loaded, or why it cannot start."""
        kind, _, text = line.strip().partition(" ")
        if kind == "WINID" and self.container is None:
            window = QWindow.fromWinId(int(text))
            self.container = QWidget.createWindowContainer(window, self)
            self.container.setMinimumSize(200, 200)
            self.layout().replaceWidget(self.placeholder, self.container)
            self.placeholder.hide()
            self._write({"op": "show"})
        elif kind == "POSES":
            try:
                names = json.loads(text)
            except ValueError:
                names = []
            self._list_poses([str(name) for name in names] if isinstance(names, list) else [])
        elif kind == "ERROR":
            self.placeholder.setText(text)

    def _host_ended(self, *_status) -> None:
        if self.container is not None:
            self.layout().replaceWidget(self.container, self.placeholder)
            self.container.deleteLater()
            self.container = None
        self.placeholder.show()
        self.host = None
        self._list_poses([])

    # -- poses ---------------------------------------------------------------------------------------------------
    def _list_poses(self, names: list[str]) -> None:
        """*names* listed to choose from, the first (the one the view plays) chosen; the row hidden with none."""
        self.pose_list.blockSignals(True)
        self.pose_list.clear()
        self.pose_list.addItems(names)
        self.pose_list.setCurrentIndex(0 if names else -1)
        self.pose_list.blockSignals(False)
        self.pose_row.setVisible(bool(names))

    def _pose_chosen(self, index: int) -> None:
        if index >= 0:
            self._send({"op": "pose", "index": index})

    # -- what is shown -----------------------------------------------------------------------------------------
    def apply_texts(self) -> None:
        """The view's own texts (the hint, loading, a load error) in the chosen language."""
        from .i18n import tr
        root = self.root
        root.setProperty("loadingText", tr("載入中…"))
        root.setProperty("errorText", tr("讀不進這個檔案："))
        hint = root.property("hintSource") or root.property("hint")
        root.setProperty("hintSource", hint)
        root.setProperty("hint", tr(hint))

    def show_file(self, path: Path, caption: str) -> None:
        """*path* shown: its display copy and box read on a thread, then given to the view (framed by the box of
        the parts it draws as soon as the load is done)."""
        from .i18n import tr
        self.current = Path(path)
        self._asked += 1
        asked = self._asked
        self._list_poses([])
        self._ensure_host()
        named = caption if Path(path).name in caption else f"{caption}\n{Path(path).name}"
        self.caption.setText(tr(f"{named}\n載入中…"))
        self.caption.setToolTip(str(path))

        def prepare():
            shown = shown_copy(path)
            return shown, model_bounds(shown)

        def ready(result, error) -> None:
            if asked != self._asked:
                return                                   # another file was asked for meanwhile
            if error is not None:
                self.caption.setText(tr(f"{named}\n讀不進這個檔案：") + str(error))
                return
            shown, box = result
            self.root.setProperty("framed", box is not None)
            if box is not None:
                self.root.setProperty("frameLow", QVector3D(*box[0]))
                self.root.setProperty("frameHigh", QVector3D(*box[1]))
            self.root.setProperty("source", QUrl.fromLocalFile(str(shown)))
            unmeasured = "" if box is not None else "\n" + tr("量不到這個檔案的大小，視角放在原點；用滾輪縮放")
            self.caption.setText(tr(f"{named}\n左鍵拖曳旋轉、右鍵平移、滾輪縮放") + unmeasured)

        background(self, prepare, ready)

    def show_message(self, text: str) -> None:
        self.current = None
        self._asked += 1
        self._list_poses([])
        self.caption.setToolTip("")
        if self.root.values.get("source"):
            self.root.setProperty("source", QUrl())
        from .i18n import tr
        self.caption.setText(tr(text))
