"""The 3D view in a process of its own (``--view-host``), its window shown inside the program's.

Qt Quick 3D's ``RuntimeLoader`` reads a model on the thread that renders the view, in one go and with no way to do
it elsewhere: a 21 MB model held one frame for 2.4 s (measured), and with the view in the window's own process the
whole window stopped answering for as long — lists, buttons, the log.  So the view runs here: ``preview.qml`` in a
frameless window whose id is printed (``WINID <id>``); the program embeds that window in its own
(:class:`preview.PreviewPanel`, ``QWindow.fromWinId``), so the mouse reaches the view directly, and a load that holds
this process holds nothing of the program's.

Commands come one JSON object a line on stdin: ``{"op": "set", "name": …, "value": …}`` sets a property of the view's
root (a value ``{"url": …}`` is a ``QUrl``, ``{"vec3": [x, y, z]}`` a ``QVector3D``), ``{"op": "look", "yaw": …,
"pitch": …}`` turns it, ``{"op": "show"}`` shows the window once it is embedded.  The process ends when stdin
does: the program closing (or dying) takes its view with it.
"""
from __future__ import annotations

import json
import sys
import threading


def decode(value):
    """A command's value as the view takes it."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QVector3D
    if isinstance(value, dict) and "url" in value:
        return QUrl.fromLocalFile(value["url"]) if value["url"] else QUrl()
    if isinstance(value, dict) and "vec3" in value:
        return QVector3D(*value["vec3"])
    return value


def encode(value):
    """A value the program sets, as a command carries it (:func:`decode` reverses it)."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QColor, QVector3D
    if isinstance(value, QUrl):
        return {"url": value.toLocalFile()}
    if isinstance(value, QColor):
        return value.name(QColor.HexArgb)
    if isinstance(value, QVector3D):
        return {"vec3": [value.x(), value.y(), value.z()]}
    return value


def apply(root, view, text: str) -> None:
    """One command line applied to the view (*root* its root object, *view* its window); an unreadable line is
    passed over."""
    try:
        command = json.loads(text)
    except ValueError:
        return
    op = command.get("op") if isinstance(command, dict) else None
    if op == "set":
        root.setProperty(command["name"], decode(command.get("value")))
    elif op == "look":
        root.look(float(command.get("yaw", 0)), float(command.get("pitch", 0)))
    elif op == "show":
        view.show()


def run() -> int:
    from PySide6.QtCore import QObject, Qt, QUrl, Signal
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQuick import QQuickView
    from .preview import QML, use_opengl

    use_opengl()
    app = QGuiApplication(sys.argv[:1])
    view = QQuickView()
    view.setResizeMode(QQuickView.SizeRootObjectToView)
    # frameless and kept off the taskbar: it lives inside the program's window, never on its own
    view.setFlags(Qt.FramelessWindowHint | Qt.Tool)
    view.setSource(QUrl.fromLocalFile(str(QML)))
    root = view.rootObject()
    if root is None:
        print("ERROR " + "; ".join(e.toString() for e in view.errors()), flush=True)
        return 1
    view.create()
    print(f"WINID {int(view.winId())}", flush=True)

    class Inbox(QObject):
        line = Signal(str)
        ended = Signal()

    inbox = Inbox()

    inbox.line.connect(lambda text: apply(root, view, text), Qt.QueuedConnection)
    inbox.ended.connect(app.quit, Qt.QueuedConnection)

    def read() -> None:
        # stdin is read on a thread of its own and handed to the window's thread: a read blocks, a window must not
        for text in sys.stdin:
            if text.strip():
                inbox.line.emit(text)
        inbox.ended.emit()

    threading.Thread(target=read, daemon=True, name="view-host-stdin").start()
    return app.exec()
