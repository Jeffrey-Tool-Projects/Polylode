"""The model viewer page: any model file — the exports in the output folder, one opened or dropped — in the
preview's turnable view, with fixed views, wireframe, backdrop and light, and what the file holds (§7.1455).

Qt Quick 3D reads the file (``RuntimeLoader``, through the assimp importer the program carries), so glTF, FBX
and OBJ all open; what a glTF holds is read from its own JSON (:func:`model_info`), the other formats show
their size.  ``ModelExtractor.exe <file>`` opens straight into this page, for "Open with".
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

from PySide6.QtCore import QFile, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QMessageBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit,
    QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from .i18n import tr
from .preview import PreviewPanel

#: what the view opens (assimp reads these; .blend it does not)
SUFFIXES = (".glb", ".gltf", ".fbx", ".obj")
#: fixed views as (label, yaw, pitch)
VIEWS = (("正面", 0, 0), ("背面", 180, 0), ("左側", -90, 0), ("右側", 90, 0), ("上方", 0, -89))
BACKDROPS = {"深色背景": "#1e2129", "淺色背景": "#e9ecf2"}
#: what one export writes for a model beside its file of the same name (and the FBX's extracted images)
OUTPUTS = (*SUFFIXES, ".blend", ".json")


def outputs_of(path: Path) -> list[Path]:
    """*path* and what was exported with it: the files of its name in its folder, and its ``.fbm`` images."""
    path = Path(path)
    same = [p for p in path.parent.iterdir() if p.stem == path.stem and p.suffix.lower() in OUTPUTS and p.is_file()]
    images = path.with_suffix(".fbm")
    return sorted(same) + ([images] if images.is_dir() else [])


def trash(path: Path) -> bool:
    """*path* to the system's recycle bin (restorable), not deleted outright."""
    # PySide6 6.11's static overload gives a bool; earlier ones gave (bool, the path in the bin)
    moved = QFile.moveToTrash(str(path))
    return bool(moved[0] if isinstance(moved, tuple) else moved)


def _gltf(path: Path) -> dict:
    """The JSON of a ``.glb`` or ``.gltf``."""
    data = path.read_bytes()
    if path.suffix.lower() == ".glb":
        magic, _version, _length = struct.unpack_from("<III", data)
        if magic != 0x46546C67:
            raise ValueError("not a glTF binary")
        size, _kind = struct.unpack_from("<II", data, 12)
        return json.loads(data[20:20 + size])
    return json.loads(data.decode("utf-8"))


#: this program's glTF writer, by the names it has had (it was IslandLife's before this repository)
OUR_WRITERS = ("Polylode interchange", "IslandLife interchange")


def stale(gltf: dict) -> bool:
    """Whether a glTF this program wrote was written by older rules than it writes by now (``glb.RULES``)."""
    from ..interchange.glb import RULES
    asset = gltf.get("asset", {})
    return asset.get("generator") in OUR_WRITERS and (asset.get("extras") or {}).get("rules", 0) < RULES


STALE = "⚠ 這個檔案是用舊的規則寫出的（例如眼影、透明部件畫成不透明），顏色會不對：請到該遊戲頁重新匯出。"


def model_info(path: Path) -> str:
    """What *path* holds, as lines for the info panel: a glTF's meshes, vertices, materials… or a file's size."""
    path = Path(path)
    lines = [path.name, f"{path.stat().st_size / (1 << 20):.1f} MB"]
    if path.suffix.lower() not in (".glb", ".gltf"):
        return "\n".join(lines + ["", "（只有 glTF 會列出內容；這個格式只顯示）"])
    try:
        gltf = _gltf(path)
    except (OSError, ValueError, struct.error) as failure:
        return "\n".join(lines + [f"讀不出 glTF 內容：{failure}"])
    accessors = gltf.get("accessors", [])
    triangles = 0
    positions = set()               # the primitives of one mesh share its vertices: each array counted once
    shapes = set()
    for mesh in gltf.get("meshes", []):
        names = (mesh.get("extras") or {}).get("targetNames") or []
        shapes.update(names)
        for primitive in mesh.get("primitives", []):
            positions.add(primitive["attributes"]["POSITION"])
            if "indices" in primitive:
                triangles += accessors[primitive["indices"]]["count"] // 3
    vertices = sum(accessors[index]["count"] for index in positions)
    joints = {j for skin in gltf.get("skins", []) for j in skin.get("joints", [])}
    if stale(gltf):
        lines[0:0] = [STALE, ""]
    lines += ["", f"網格 {len(gltf.get('meshes', []))}　頂點 {vertices:,}　三角形 {triangles:,}",
              f"骨頭 {len(joints)}　形狀鍵 {len(shapes)}　動畫 {len(gltf.get('animations', []))}",
              f"材質 {len(gltf.get('materials', []))}　貼圖 {len(gltf.get('images', []))}", "", "材質："]
    for material in gltf.get("materials", []):
        mode = material.get("alphaMode", "OPAQUE")
        if mode == "MASK":
            mode += f" {material.get('alphaCutoff', 0.5):g}"
        lines.append(f"  {material.get('name', '?')}　{mode}")
    return "\n".join(lines)


class ViewerPage(QWidget):
    """The viewer: a file list, the view with its controls, and the file's contents."""

    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel("模型檢視器")
        title.setObjectName("title")
        top.addWidget(title)
        top.addStretch(1)
        open_button = QPushButton("開啟檔案…")
        open_button.clicked.connect(self._open_dialog)
        top.addWidget(open_button)
        layout.addLayout(top)
        hint = QLabel("輸出資料夾裡的模型列在左邊；也可以開啟或拖進任何 .glb／.gltf／.fbx／.obj。")
        hint.setObjectName("muted")
        layout.addWidget(hint)

        files = QWidget()
        files_layout = QVBoxLayout(files)
        files_layout.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜尋檔名…")
        self.search.textChanged.connect(self._filter)
        files_layout.addWidget(self.search)
        self.files = QListWidget()
        # a long relative path keeps its folder and its end in sight
        self.files.setTextElideMode(Qt.ElideMiddle)
        self.files.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.files.currentItemChanged.connect(lambda item, _before: item is not None and self.show_file(item.data(Qt.UserRole)))
        files_layout.addWidget(self.files, 1)
        buttons = QHBoxLayout()
        refresh = QPushButton("重新整理")
        refresh.clicked.connect(self.refresh)
        buttons.addWidget(refresh)
        remove = QPushButton("刪除…")
        remove.setToolTip("把選取的模型（同名的 .glb／.fbx／.blend… 與貼圖資料夾）移到資源回收筒")
        remove.clicked.connect(self.delete_selected)
        buttons.addWidget(remove)
        files_layout.addLayout(buttons)
        QShortcut(QKeySequence.Delete, self.files, self.delete_selected)

        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)
        self.panel = PreviewPanel()
        root = self.panel.view.rootObject()
        if root is not None:
            root.setProperty("hint", "從左邊選一個模型，或開啟、拖進一個檔案")
            root.setProperty("hintSource", "")
            self.panel.apply_texts()
        centre_layout.addWidget(self.panel, 1)
        # two rows, so the page narrows with the window instead of holding it wide
        views = QHBoxLayout()
        for label, yaw, pitch in VIEWS:
            button = QPushButton(label)
            button.setMinimumWidth(0)
            button.clicked.connect(lambda _c, y=yaw, p=pitch: self.look(y, p))
            views.addWidget(button)
        centre_layout.addLayout(views)
        controls = QHBoxLayout()
        self.wireframe = QCheckBox("線框")
        self.wireframe.toggled.connect(lambda on: self._set("wireframe", on))
        controls.addWidget(self.wireframe)
        self.backdrop = QPushButton(next(iter(BACKDROPS)))
        self.backdrop.clicked.connect(self._toggle_backdrop)
        controls.addWidget(self.backdrop)
        controls.addWidget(QLabel("光"))
        self.light = QSlider(Qt.Horizontal)
        self.light.setRange(20, 300)
        self.light.setValue(100)
        self.light.setMinimumWidth(60)
        self.light.valueChanged.connect(lambda value: self._set("light", value / 100))
        controls.addWidget(self.light, 1)
        centre_layout.addLayout(controls)

        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setPlaceholderText("檔案內容")

        split = QSplitter(Qt.Horizontal)
        split.addWidget(files)
        split.addWidget(centre)
        split.addWidget(self.info)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 6)
        split.setStretchFactor(2, 2)
        split.setSizes([260, 700, 260])
        layout.addWidget(split, 1)

    def _set(self, name: str, value) -> None:
        root = self.panel.view.rootObject()
        if root is not None:
            root.setProperty(name, value)

    def look(self, yaw: float, pitch: float) -> None:
        root = self.panel.view.rootObject()
        if root is not None:
            root.look(yaw, pitch)

    def _toggle_backdrop(self) -> None:
        names = list(BACKDROPS)
        now = names[(names.index(self.backdrop.text()) + 1) % len(names)]
        self.backdrop.setText(now)
        self._set("backdrop", BACKDROPS[now])

    def refresh(self) -> None:
        """The model files under the output folder, newest first."""
        self.files.clear()
        output = Path(self.window_.settings.output) if self.window_.settings.output else None
        if output is None or not output.is_dir():
            self.files.addItem("（在「設定」指定輸出資料夾後，匯出的模型會列在這裡）")
            return
        found = sorted((p for p in output.rglob("*") if p.suffix.lower() in SUFFIXES and p.is_file()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for path in found:
            item = QListWidgetItem(str(path.relative_to(output)))
            item.setData(Qt.UserRole, path)
            self.files.addItem(item)
        self._filter(self.search.text())

    def delete_selected(self) -> None:
        """The selected exported model's files to the recycle bin, after saying which; only a model listed from
        the output folder, never a file opened from elsewhere."""
        item = self.files.currentItem()
        path = item.data(Qt.UserRole) if item is not None else None
        output = Path(self.window_.settings.output) if self.window_.settings.output else None
        if not path or output is None or not Path(path).resolve().is_relative_to(output.resolve()):
            QMessageBox.information(self, "刪除模型", "先在左邊的清單選一個輸出資料夾裡的模型。")
            return
        doomed = outputs_of(Path(path))
        names = "\n".join(f"  {p.relative_to(output)}" for p in doomed)
        if QMessageBox.question(self, "刪除模型", "把這些移到資源回收筒？\n" + names) != QMessageBox.StandardButton.Yes:
            return
        if self.panel.current is not None and self.panel.current.resolve() in {p.resolve() for p in doomed}:
            self.panel.show_message("已刪除")
            self.info.clear()
        failed = [p for p in doomed if not trash(p)]
        folder = Path(path).parent
        while folder != output and folder.is_dir() and not any(folder.iterdir()):
            trash(folder)
            folder = folder.parent
        if failed:
            QMessageBox.warning(self, "刪除模型", "這些沒能移到資源回收筒：\n" + "\n".join(map(str, failed)))
        self.refresh()

    def _filter(self, text: str) -> None:
        for row in range(self.files.count()):
            item = self.files.item(row)
            item.setHidden(bool(text) and text.lower() not in item.text().lower())

    def _open_dialog(self) -> None:
        chosen, _kind = QFileDialog.getOpenFileName(self, tr("開啟模型"), "", tr("模型 (*.glb *.gltf *.fbx *.obj)"))
        if chosen:
            self.show_file(Path(chosen))

    def show_file(self, path) -> None:
        if not path:
            return
        path = Path(path)
        if not path.is_file():
            self.panel.show_message(f"找不到 {path}")
            return
        caption = path.name
        if path.suffix.lower() in (".glb", ".gltf"):
            try:
                if stale(_gltf(path)):
                    caption = STALE + "\n" + path.name
            except (OSError, ValueError, struct.error):
                pass
        self.panel.show_file(path, caption)
        self.info.setPlainText(tr(model_info(path)))

    def dragEnterEvent(self, event) -> None:
        if any(Path(url.toLocalFile()).suffix.lower() in SUFFIXES for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in SUFFIXES:
                self.show_file(path)
                return

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.files.count() == 0:
            self.refresh()

