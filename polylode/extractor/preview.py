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
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..core.paths import derived_root
from . import settings as S
from .sources import Model

QML = Path(__file__).with_name("preview.qml")
#: the writers' rules a cached preview was made with: raised when what a .glb holds changes, so an earlier
#: preview is not shown again (2: overlays as their tint by coverage; 3: GF2 eye layers and transparency,
#: WuWa translucency, §7.1455)
CACHE_VERSION = 4


def cache_folder(model: Model) -> Path:
    """Where *model*'s preview is written: one folder per source and model."""
    safe = re.sub(r"[^A-Za-z0-9._@-]", "_", model.ident)
    return derived_root() / "assets_cache" / "extractor" / "preview" / f"v{CACHE_VERSION}" / model.source / safe


def cached(model: Model) -> Path | None:
    """The ``.glb`` an earlier preview of *model* wrote, when there is one."""
    return next(iter(sorted(cache_folder(model).rglob("*.glb"))), None)


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
        self.caption = QLabel("左鍵拖曳旋轉、右鍵平移、滾輪縮放")
        self.apply_texts()
        self.caption.setObjectName("muted")
        self.caption.setWordWrap(True)
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
        root = self.view.rootObject()
        if root is not None:
            root.setProperty("source", QUrl.fromLocalFile(str(path)))
        from .i18n import tr
        self.caption.setText(tr(f"{caption}\n{path}\n左鍵拖曳旋轉、右鍵平移、滾輪縮放"))

    def show_message(self, text: str) -> None:
        root = self.view.rootObject()
        if root is not None:
            root.setProperty("source", QUrl())
        from .i18n import tr
        self.caption.setText(tr(text))
