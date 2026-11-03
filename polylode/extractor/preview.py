"""The preview before an export: the model converted to ``.glb`` by the same job an export runs, shown turnable.

A preview is an export of one format, ``.glb`` (the one Qt Quick 3D reads,
through ``RuntimeLoader``), into a cache of this program's own
(:func:`cache_folder`), without Blender's read-back: it is for looking at,
the files a user keeps come from the export queue.  A model previewed once is
shown from the cache the next time.  ``preview.qml`` frames the model by its
bounds; the mouse turns (left drag), pans (right drag) and zooms (wheel).
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QVector3D
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..core.paths import derived_root
from . import settings as S
from .sources import Model

QML = Path(__file__).with_name("preview.qml")
#: the writers' rules a cached preview was made with: raised when what a .glb holds changes, so an earlier
#: preview is not shown again (2: overlays as their tint by coverage; 3: GF2 eye layers and transparency,
#: WuWa translucency, §7.1455; 5: vertex colours kept out of COLOR_0, which hid
#: GF2's characters, and GF2's lower LOD levels hidden; 6: Wuthering Waves split by material; 7: Wuthering
#: Waves' cut-outs from the game's shader, parallax fur as strand layers; 8: GF2's spare outfit parts and
#: dormitory variants hidden)
CACHE_VERSION = 8


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
    """*settings* for a preview job: ``.glb`` only, and no Blender (its read-back takes minutes)."""
    return replace(settings, formats=["glb"], tools={k: v for k, v in settings.tools.items() if k != "blender"})


def triangle_glb(path: Path) -> Path:
    """A one-triangle ``.glb`` at *path*: what :func:`load_check` reads."""
    import json
    import struct
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "buffers": [{"byteLength": len(positions)}],
                "bufferViews": [{"buffer": 0, "byteLength": len(positions)}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                               "min": [0, 0, 0], "max": [1, 1, 0]}]}
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


def load_check(glb: Path, seconds: float = 30) -> dict:
    """``preview.qml`` loading *glb* the way the window does, drawn off-screen with no window at all
    (``QQuickRenderControl`` into an OpenGL framebuffer): its loader state and size.

    The packaged program's ``--selftest`` reads a triangle with it: a missing Qt Quick 3D module or
    importer plugin only shows when a file is actually loaded.  Needs a ``QGuiApplication``.
    """
    import time
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext
    from PySide6.QtOpenGL import QOpenGLFramebufferObject
    from PySide6.QtQml import QQmlComponent, QQmlEngine
    from PySide6.QtQuick import QQuickGraphicsDevice, QQuickRenderControl, QQuickRenderTarget, QQuickWindow
    app = QGuiApplication.instance()
    context = QOpenGLContext()
    if not context.create():
        return {"state": "no OpenGL context", "size": 0.0}
    surface = QOffscreenSurface()
    surface.setFormat(context.format())
    surface.create()
    control = QQuickRenderControl()
    window = QQuickWindow(control)
    window.setGraphicsDevice(QQuickGraphicsDevice.fromOpenGLContext(context))
    context.makeCurrent(surface)
    if not control.initialize():
        return {"state": "the render control did not start", "size": 0.0}
    size = QSize(64, 64)
    target = QOpenGLFramebufferObject(size, QOpenGLFramebufferObject.CombinedDepthStencil)
    window.setRenderTarget(QQuickRenderTarget.fromOpenGLTexture(target.texture(), size))
    window.setGeometry(0, 0, size.width(), size.height())
    engine = QQmlEngine()
    component = QQmlComponent(engine, QUrl.fromLocalFile(str(QML)))
    root = component.create()
    if root is None:
        return {"state": "; ".join(e.toString() for e in component.errors()) or "no root item", "size": 0.0}
    root.setParentItem(window.contentItem())
    root.setProperty("source", QUrl.fromLocalFile(str(glb)))
    deadline = time.monotonic() + seconds
    while True:
        app.processEvents()
        context.makeCurrent(surface)
        control.polishItems()
        control.beginFrame()
        control.sync()
        control.render()
        control.endFrame()
        state = root.property("state_")
        if state not in ("", "loading") or time.monotonic() > deadline:
            return {"state": state, "size": float(root.property("size"))}
        time.sleep(0.02)


#: display copies kept (:func:`shown_copy`): the newest few, a viewer goes back and forth between files
SHOWN_KEPT = 8


def shown_copy(path: Path) -> Path:
    """*path* as a viewer that honours ``KHR_node_visibility`` draws it: Qt's loader does not, and drew every
    hidden part (GF2's lower LOD levels over the most detailed one, a converter's alternate parts).

    A ``.glb`` with hidden mesh nodes is shown from a copy whose hidden nodes carry no mesh: only the JSON
    chunk changes, the binary one is copied as it is, and the file itself is left alone.  The copy is kept
    by the file's path, size and time, so showing the file again does not write it again.  Any other file,
    or a ``.glb`` without hidden meshes, is shown as it is.
    """
    import hashlib
    import json
    import struct
    path = Path(path)
    if path.suffix.lower() != ".glb":
        return path
    try:
        data = path.read_bytes()
        length = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20:20 + length])
    except (OSError, ValueError, struct.error):
        return path
    hidden = [node for node in document.get("nodes", []) if "mesh" in node
              and not (node.get("extensions") or {}).get("KHR_node_visibility", {}).get("visible", True)]
    if not hidden:
        return path
    stat = path.stat()
    key = hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:20]
    folder = derived_root() / "assets_cache" / "extractor" / "shown"
    copy = folder / f"{key}.glb"
    if copy.is_file():
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


class PreviewPanel(QWidget):
    """The turnable view of one ``.glb`` and a line saying what is shown."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = QQuickWidget()
        self.view.setResizeMode(QQuickWidget.SizeRootObjectToView)
        self.view.setSource(QUrl.fromLocalFile(str(QML)))
        self.view.setMinimumSize(200, 200)
        layout.addWidget(self.view, 1)
        from .stepbar import StepBar
        # the conversion a preview waits for (the file's own load is the view's bar)
        self.steps = StepBar()
        layout.addWidget(self.steps)
        #: the file shown (the view may read a display copy of it, :func:`shown_copy`)
        self.current: Path | None = None
        self.caption = QLabel("左鍵拖曳旋轉、右鍵平移、滾輪縮放")
        self.apply_texts()
        self.caption.setObjectName("muted")
        self.caption.setWordWrap(True)
        # a path has nowhere to break: the caption takes the width it is given rather than widening the panel
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.caption)

    def apply_texts(self) -> None:
        """The view's own texts (the hint, loading, a load error) in the chosen language."""
        from .i18n import tr
        root = self.view.rootObject()
        if root is not None:
            root.setProperty("loadingText", tr("載入中…"))
            root.setProperty("errorText", tr("讀不進這個檔案："))
            hint = root.property("hintSource") or root.property("hint")
            root.setProperty("hintSource", hint)
            root.setProperty("hint", tr(hint))

    def show_file(self, path: Path, caption: str) -> None:
        self.current = Path(path)
        root = self.view.rootObject()
        if root is not None:
            # what a viewer honouring hidden parts shows, framed before it is given: the view looks at
            # the box of the parts it draws as soon as the load is done
            shown = shown_copy(path)
            box = model_bounds(shown)
            root.setProperty("framed", box is not None)
            if box is not None:
                root.setProperty("frameLow", QVector3D(*box[0]))
                root.setProperty("frameHigh", QVector3D(*box[1]))
            root.setProperty("source", QUrl.fromLocalFile(str(shown)))
        from .i18n import tr
        unmeasured = "" if root is None or box is not None else "\n" + tr("量不到這個檔案的大小，視角放在原點；用滾輪縮放")
        # the file's name; its whole path on the caption's tooltip
        named = caption if Path(path).name in caption else f"{caption}\n{Path(path).name}"
        self.caption.setText(tr(f"{named}\n左鍵拖曳旋轉、右鍵平移、滾輪縮放") + unmeasured)
        self.caption.setToolTip(str(path))

    def show_message(self, text: str) -> None:
        self.current = None
        self.caption.setToolTip("")
        root = self.view.rootObject()
        if root is not None:
            root.setProperty("source", QUrl())
        from .i18n import tr
        self.caption.setText(tr(text))
