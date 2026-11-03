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

from PySide6.QtCore import QFile, QProcess, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QMessageBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit,
    QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from .i18n import tr
from .preview import PreviewPanel

#: the settings key of the Discord application whose Activity a shared site's front page is (``share.SITE_SCRIPT``)
ACTIVITY = "discord_activity"

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
        self.share_button = QPushButton("分享到 GitHub Pages…")
        self.share_button.setToolTip("把顯示中的 .glb 寫進你的 GitHub Pages 網站並推送；網址貼到 Discord 會顯示成卡片，點開可以轉動模型")
        self.share_button.clicked.connect(self.share)
        files_layout.addWidget(self.share_button)
        self.share_process: QProcess | None = None
        self.sign_in_box: QMessageBox | None = None

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

    def share(self) -> None:
        """The model shown, on a GitHub Pages site, by a job of its own, its address copied: through the user's
        GitHub account (``github``: signed in in the browser the first time, the repository and uploads made by the
        program), or, when a site folder is set, committed and pushed in that clone (``share``).  The user is
        told first that the model becomes public."""
        path = self.panel.current
        if path is None or Path(path).suffix.lower() != ".glb":
            QMessageBox.information(self, tr("分享到 GitHub Pages"), tr("先開啟一個 .glb：網頁的 3D 檢視器讀的是 glTF。"))
            return
        from . import github, jobs, share
        site = self.window_.settings.tool("pages_site")
        activity = (self.window_.settings.tools.get(ACTIVITY) or "").strip()
        name = share.slug(Path(path))
        if site is not None:
            try:
                url = share.check_site(site)
            except share.ShareError as failure:
                QMessageBox.warning(self, tr("分享到 GitHub Pages"), tr(str(failure)))
                return
            where, command = str(site), jobs.share_command(Path(path), site, activity)
            ask = f"這個模型會推送到 GitHub，公開在 {url}{share.MODELS}/{name}/，任何拿到網址的人都看得到、也下載得到。"
        else:
            login = github.account()
            url = github.pages_url(login) if login else f"https://<帳號>.github.io/{github.REPOSITORY}/"
            where, command = f"GitHub {login or ''}".rstrip(), jobs.share_github_command(Path(path), activity)
            ask = (f"這個模型會上傳到你 GitHub 帳號的公開倉庫 {github.REPOSITORY}，公開在 {url}{share.MODELS}/{name}/，"
                   "任何拿到網址的人都看得到、也下載得到。")
            if not login:
                ask += "\n第一次會請你在瀏覽器登入 GitHub。"
        if QMessageBox.question(self, tr("分享到 GitHub Pages"), tr(
                ask + "\n模型來自遊戲，公開前請自行確認可以分享。要繼續嗎？")) != QMessageBox.StandardButton.Yes:
            return
        self.share_button.setEnabled(False)
        self.window_.log(f"分享 {Path(path).name} 到 {where}")
        self.share_url, self.share_error, self.share_settings, self.share_live = "", "", "", False
        self.sign_in_box = None
        self.share_process = QProcess(self)
        self.share_process.setProcessChannelMode(QProcess.MergedChannels)
        self.share_process.readyReadStandardOutput.connect(self._share_output)
        self.share_process.finished.connect(self._share_done)
        program, *arguments = command
        self.share_process.start(program, arguments)

    def _share_output(self) -> None:
        for raw in bytes(self.share_process.readAllStandardOutput()).decode("utf-8", "replace").splitlines():
            kind, _, text = raw.partition(" ")
            if kind == "URL":
                self.share_url = text
            elif kind == "LIVE":
                self.share_live = text == "yes"
            elif kind == "PAGES":
                self.share_settings = text
            elif kind == "DEVICE":
                code, _, address = text.partition(" ")
                self._sign_in(code, address)
            elif kind == "ERROR":
                self.share_error = text
                self.window_.log(f"  錯誤：{text}")
            elif kind == "LOG":
                self.window_.log(f"  {text}")
                self._signed_in()

    def _signed_in(self) -> None:
        """The sign-in box closed: the job has gone on (closed this way, it does not stop the share)."""
        if self.sign_in_box is not None:
            box, self.sign_in_box = self.sign_in_box, None
            box.done(0)

    def _sign_in(self, code: str, address: str) -> None:
        """GitHub's device sign-in: the code copied and the page opened; the box stays until the job goes on, and
        closing it stops the share."""
        QGuiApplication.clipboard().setText(code)
        QDesktopServices.openUrl(QUrl(address))
        self.sign_in_box = QMessageBox(QMessageBox.Information, tr("登入 GitHub"), tr(
            f"在瀏覽器開啟的 {address} 輸入這組代碼（已複製）：\n{code}\n\n授權後這裡會自動繼續。"),
            QMessageBox.Cancel, self)
        self.sign_in_box.setModal(False)
        self.sign_in_box.rejected.connect(self._stop_share)
        self.sign_in_box.show()

    def _stop_share(self) -> None:
        if self.share_process is not None and self.sign_in_box is not None:
            self.share_error = "登入被取消了"
            self.share_process.kill()

    def _share_done(self, code: int, _status) -> None:
        self._signed_in()
        self.share_button.setEnabled(True)
        self.share_process = None
        self.window_.settings_page.refresh_account()
        if code != 0 or not self.share_url:
            QMessageBox.warning(self, tr("分享到 GitHub Pages"),
                                tr(self.share_error) or tr("分享沒有完成，原因寫在下方的紀錄。"))
            return
        QGuiApplication.clipboard().setText(self.share_url)
        self.window_.log(f"  已分享：{self.share_url}（網址已複製）")
        # a link pasted before Pages serves it is a 404 and Discord shows no card, then keeps that answer a while
        if self.share_live:
            text = f"網頁已上線，網址已複製到剪貼簿：\n{self.share_url}\n\n貼到 Discord 會顯示成卡片。"
        else:
            text = (f"已推送，但 GitHub Pages 還沒發佈完成。網址已複製到剪貼簿：\n{self.share_url}\n\n"
                    "等網頁打得開再貼到 Discord（新倉庫第一次可能要幾分鐘）。\n"
                    "如果先貼了而沒有卡片，在網址後面加上 ?1 再貼一次：Discord 會記住沒有卡片的結果一陣子。")
        if self.share_settings:
            text += ("\n\n這個倉庫的 GitHub Pages 還沒開啟：按「開啟 Pages 設定」，"
                     "Source 選 Deploy from a branch、分支選 main、資料夾 / (root)，按 Save。只需要做一次。")
        box = QMessageBox(QMessageBox.Information, tr("分享到 GitHub Pages"), tr(text), QMessageBox.Ok, self)
        open_button = box.addButton(tr("在瀏覽器開啟"), QMessageBox.ActionRole)
        settings_button = box.addButton(tr("開啟 Pages 設定"), QMessageBox.ActionRole) if self.share_settings else None
        box.exec()
        if box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl(self.share_url))
        elif settings_button is not None and box.clickedButton() is settings_button:
            QDesktopServices.openUrl(QUrl(self.share_settings))

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

