"""The 3D view in a process of its own (``--view-host``), its window shown inside the program's.

Qt Quick 3D's ``RuntimeLoader`` reads a model on the thread that renders the view, in one go and with no way to do
it elsewhere: a 21 MB model held one frame for 2.4 s (measured), and with the view in the window's own process the
whole window stopped answering for as long — lists, buttons, the log.  So the view runs here: ``preview.qml`` in a
frameless window whose id is printed (``WINID <id>``); the program embeds that window in its own
(:class:`preview.PreviewPanel`, ``QWindow.fromWinId``), so the mouse reaches the view directly, and a load that holds
this process holds nothing of the program's.

Commands come one JSON object a line on stdin: ``{"op": "set", "name": …, "value": …}`` sets a property of the view's
root (a value ``{"url": …}`` is a ``QUrl``, ``{"vec3": [x, y, z]}`` a ``QVector3D``), ``{"op": "look", "yaw": …,
"pitch": …}`` turns it, ``{"op": "pose", "index": …}`` plays one of the model's poses, ``{"op": "show"}`` shows
the window once it is embedded.  The process ends when stdin does: the program closing (or dying) takes its view
with it.

A model's poses are its glTF animations, each of which the loader makes a ``Timeline`` with a looping
``TimelineAnimation`` (named after the animation), the first one enabled.  Once a model is loaded the view says
which it has, ``POSES ["name", …]`` (the window lists them, :class:`preview.PreviewPanel`); a load with none says
``POSES []``.
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


#: the objects a loaded model's timelines are made of, passed over looking for them: a GF2 outfit with 31 poses
#: carries 570,000 keyframes, and a search of every object took 12.7 s where this takes milliseconds (measured)
KEYFRAMES = ("QQuickKeyframeGroup", "QQuickKeyframe")


#: a joint further than this many times the rest skeleton's size from the posed skeleton's middle is not framed: the
#: games hide a part by sending it away (Cheyanne SSR0101's shoes 217 m off in her bedroom)
OUTLIER = 3.0


def joint_box(positions, size: float | None = None):
    """The box ``(low, high)`` of *positions* ``(n, 3)``; with *size*, only those within :data:`OUTLIER` times it of
    their median.  None with none."""
    import numpy as np
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if size and len(points):
        points = points[np.linalg.norm(points - np.median(points, axis=0), axis=1) <= OUTLIER * size]
    return (points.min(0), points.max(0)) if len(points) else None


def skeleton(timelines) -> list:
    """The nodes the *timelines* move and the nodes below them (a model's mesh nodes left out): the joints.  The scene's
    own nodes and a skinned mesh's stay where they are whatever the pose, and framed with them the view reached back to
    the rest place."""
    seen, out = set(), []
    stack = []
    for timeline in timelines:
        for group in timeline.children():
            if group.metaObject().className() == "QQuickKeyframeGroup":
                target = group.property("target")
                if target is not None:
                    stack.append(target)
    while stack:
        node = stack.pop()
        if id(node) in seen or node.metaObject().className() == "QQuick3DModel" \
                or node.metaObject().indexOfProperty("scenePosition") < 0:
            continue
        seen.add(id(node))
        out.append(node)
        stack.extend(child for child in node.children() if child.metaObject().className() not in KEYFRAMES)
    return out


class Poses:
    """The poses of the model the view has loaded, as ``(name, timeline, animation)``, in the file's order.

    A pose may place the character elsewhere than its rest (GF2's ``CommandCenterBackGround`` seats Clukay SSR01 2.4 m
    to the side, 1.2 times her size), and the view framed the rest's box: she stood out of it.  So the view is framed
    again on each pose, by where its joints are: the rest's box (``frameLow``／``frameHigh``, from the file's own
    data) moved with them, as far past them as it reached at rest."""

    def __init__(self):
        self.found: list = []
        self.root = None
        #: the loaded model's nodes (its joints among them), the box they fill at rest and the view's box then
        self.nodes: list = []
        self.rest = None
        self.frame = None
        self.morphs = None

    def clear(self) -> None:
        # the loader drops the timelines with the model: none is touched once another file is asked for
        if self.morphs is not None:
            self.morphs.close()
        self.found, self.nodes, self.rest, self.frame = [], [], None, None
        self.morphs = None

    def read(self, root) -> list[str]:
        """The loaded model's poses found under *root*'s loader (``objectName`` "loader"), the first one playing
        and the rest stopped (the loader starts every animation, only the first timeline enabled); their names."""
        from PySide6.QtCore import QObject
        if self.morphs is not None:
            self.morphs.close()
        self.found = []
        loader = root.findChild(QObject, "loader")
        stack = list(reversed(loader.children())) if loader is not None else []
        while stack:
            node = stack.pop()
            kind = node.metaObject().className()
            if kind == "QQuickTimeline":
                animation = next((a for a in node.children()
                                  if a.metaObject().className() == "QQuickTimelineAnimation"), None)
                if animation is not None:
                    self.found.append((animation.objectName() or f"#{len(self.found) + 1}", node, animation))
            elif kind not in KEYFRAMES:
                stack.extend(reversed(node.children()))
        self.morphs = None
        if self.found and root.property("source") is not None:
            from .morph_playback import Playback
            try:
                self.morphs = Playback(root, self.found)
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                pass
        self.nodes = skeleton(timeline for _name, timeline, _animation in self.found)
        self.root = root
        if self.found:
            for _name, timeline, animation in self.found:      # every pose off: the joints where they rest
                animation.stop()
                timeline.setProperty("enabled", False)
            self.rest = self._joints()
            if root.property("framed"):
                low, high = root.property("frameLow"), root.property("frameHigh")
                self.frame = ((low.x(), low.y(), low.z()), (high.x(), high.y(), high.z()))
            self.play(0)
        return [name for name, _timeline, _animation in self.found]

    def _joints(self, size: float | None = None):
        positions = [node.property("scenePosition") for node in self.nodes]
        return joint_box([(p.x(), p.y(), p.z()) for p in positions if p is not None], size)

    def _frame(self) -> None:
        """The view framed on the pose now shown (its first frame): the rest's box moved with the joints."""
        import numpy as np
        if self.root is None or self.rest is None or self.frame is None:
            return
        rest_low, rest_high = (np.asarray(v) for v in self.rest)
        posed = self._joints(float(np.linalg.norm(rest_high - rest_low)) or None)
        if posed is None:
            return
        frame_low, frame_high = (np.asarray(v) for v in self.frame)
        # as far past the joints as the file's box reached at rest, on each side
        low = posed[0] - np.maximum(rest_low - frame_low, 0)
        high = posed[1] + np.maximum(frame_high - rest_high, 0)
        from PySide6.QtGui import QVector3D
        self.root.setProperty("frameLow", QVector3D(*map(float, low)))
        self.root.setProperty("frameHigh", QVector3D(*map(float, high)))
        self.root.refit()

    def play(self, index: int) -> bool:
        """Pose *index* looped from its start, every other one off; False when the model has no such pose.

        A pose's file leaves out the bones it holds at rest (``glb._animations``): every other timeline is
        disabled before this one is enabled, a disabled timeline putting what it moved back to where it was."""
        if not 0 <= index < len(self.found):
            return False
        for k, (_name, timeline, animation) in enumerate(self.found):
            if k != index:
                animation.stop()
                timeline.setProperty("enabled", False)
        _name, timeline, animation = self.found[index]
        if self.morphs is not None:
            self.morphs.reset()
        timeline.setProperty("enabled", True)
        timeline.setProperty("currentFrame", timeline.property("startFrame"))
        animation.setProperty("loops", -1)
        animation.restart()
        if self.morphs is not None:
            self.morphs.apply(index)
        self._frame()
        return True


def apply(root, view, text: str, poses: Poses | None = None) -> None:
    """One command line applied to the view (*root* its root object, *view* its window, *poses* the loaded
    model's); an unreadable line is passed over."""
    try:
        command = json.loads(text)
    except ValueError:
        return
    op = command.get("op") if isinstance(command, dict) else None
    if op == "set":
        if command.get("name") == "source":
            from pathlib import Path
            from .character_styles import descriptors
            source = decode(command.get("value"))
            root.setProperty("nativeMaterials", descriptors(Path(source.toLocalFile())) if source.toLocalFile() else [])
        if command.get("name") == "source" and poses is not None:
            poses.clear()
        root.setProperty(command["name"], decode(command.get("value")))
    elif op == "look":
        root.look(float(command.get("yaw", 0)), float(command.get("pitch", 0)))
    elif op == "pose" and poses is not None:
        try:
            poses.play(int(command.get("index", -1)))
        except (TypeError, ValueError):
            pass
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
    poses = Poses()

    def loaded() -> None:
        print("POSES " + json.dumps(poses.read(root), ensure_ascii=False), flush=True)

    root.loaded.connect(loaded)
    inbox.line.connect(lambda text: apply(root, view, text, poses), Qt.QueuedConnection)
    inbox.ended.connect(app.quit, Qt.QueuedConnection)

    def read() -> None:
        # stdin is read on a thread of its own and handed to the window's thread: a read blocks, a window must not
        for text in sys.stdin:
            if text.strip():
                inbox.line.emit(text)
        inbox.ended.emit()

    threading.Thread(target=read, daemon=True, name="view-host-stdin").start()
    return app.exec()
