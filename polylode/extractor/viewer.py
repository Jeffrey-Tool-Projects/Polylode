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
    QAbstractItemView, QCheckBox, QFileDialog, QMessageBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit,
    QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from .i18n import tr
from .preview import PreviewPanel, background

#: the settings key of the Discord application whose Activity a shared site's front page is (``share.SITE_SCRIPT``)
ACTIVITY = "discord_activity"

#: what the view opens (assimp reads these; .blend it does not)
SUFFIXES = (".glb", ".gltf", ".fbx", ".obj")
#: fixed views as (label, yaw, pitch)
VIEWS = (("正面", 0, 0), ("背面", 180, 0), ("左側", -90, 0), ("右側", 90, 0), ("上方", 0, -89))
BACKDROPS = {"深色背景": "#1e2129", "淺色背景": "#e9ecf2"}
#: how many of the files a delete takes are named in its question; the rest are counted
LISTED = 30
#: what one export writes for a model beside its file of the same name (and the FBX's extracted images)
OUTPUTS = (*SUFFIXES, ".blend", ".json")


class ShareRun:
    """What one share or site-list job has said: its addresses, whether its page is served, Pages' settings page
    when Pages is off, the models it took off the list, its error and its open sign-in box."""

    def __init__(self, kind: str):
        self.kind = kind
        self.urls: list[str] = []
        self.live = False
        self.settings = ""
        self.removed: list[str] = []
        self.error = ""
        self.sign_in_box = None


def trash_soon(path: Path, tries: int = 10, wait: float = 0.2) -> bool:
    """:func:`trash`, tried again for a moment while the file is held open: a model just selected is read on a
    thread (its summary, its display copy), and a delete pressed at once found it open and failed."""
    import time
    for attempt in range(tries):
        try:
            if trash(path):
                return True
        except OSError:
            pass
        if attempt + 1 < tries:
            QGuiApplication.processEvents()
            time.sleep(wait)
    return False


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
        # Ctrl- and Shift-click pick several models, which "刪除…" (and the Delete key) take together
        self.files.setSelectionMode(QAbstractItemView.ExtendedSelection)
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
        remove.setToolTip("把選取的模型（同名的 .glb／.fbx／.blend… 與貼圖資料夾）移到資源回收筒；按住 Ctrl 或 Shift 可以選好幾個")
        remove.clicked.connect(self.delete_selected)
        buttons.addWidget(remove)
        files_layout.addLayout(buttons)
        QShortcut(QKeySequence.Delete, self.files, self.delete_selected)
        self.share_button = QPushButton("分享到 GitHub Pages…")
        self.share_button.setToolTip("把選取的 .glb（沒選就是顯示中的那個；按住 Ctrl 或 Shift 可以選好幾個）一起寫進你的 GitHub Pages 網站並推送；"
                                     "網址貼到 Discord 會顯示成卡片，點開可以轉動模型")
        self.share_button.clicked.connect(self.share)
        files_layout.addWidget(self.share_button)
        self.tidy_button = QPushButton("更新網站清單")
        self.tidy_button.setToolTip("依 GitHub 上還在的模型重寫網站首頁與清單（不上傳模型）：在 GitHub 刪掉的模型就會從首頁拿掉")
        self.tidy_button.clicked.connect(self.tidy)
        files_layout.addWidget(self.tidy_button)
        #: the share and site-list jobs running: their process → what each has said (:class:`ShareRun`)
        self.share_runs: dict = {}
        #: what the running share job does: "share" a model, or "tidy" the site's list

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
        """The model files under the output folder, newest first, listed on a thread (a large output folder is
        thousands of files to walk) and shown when found."""
        self.files.clear()
        output = Path(self.window_.settings.output) if self.window_.settings.output else None
        if output is None or not output.is_dir():
            self.files.addItem("（在「設定」指定輸出資料夾後，匯出的模型會列在這裡）")
            return
        self._listing = getattr(self, "_listing", 0) + 1
        listing = self._listing
        self.files.addItem("讀取輸出資料夾…")

        def find():
            return sorted((p for p in output.rglob("*") if p.suffix.lower() in SUFFIXES and p.is_file()),
                          key=lambda p: p.stat().st_mtime, reverse=True)

        def found(paths, error) -> None:
            if listing != self._listing:
                return
            self.files.clear()
            if error is not None:
                self.files.addItem(f"讀不了輸出資料夾：{error}")
                return
            for path in paths:
                item = QListWidgetItem(str(path.relative_to(output)))
                item.setData(Qt.UserRole, path)
                self.files.addItem(item)
            self._filter(self.search.text())

        background(self, find, found)

    def delete_selected(self) -> None:
        """The selected exported models' files to the recycle bin, after saying which: every model selected (Ctrl- or
        Shift-click picks several), each with what was exported beside it; only models listed from the output folder,
        never a file opened from elsewhere."""
        output = Path(self.window_.settings.output) if self.window_.settings.output else None
        items = self.files.selectedItems() or ([self.files.currentItem()] if self.files.currentItem() else [])
        models = [Path(item.data(Qt.UserRole)) for item in items if item.data(Qt.UserRole)]
        if output is not None:
            models = [path for path in models if path.resolve().is_relative_to(output.resolve())]
        if not models or output is None:
            QMessageBox.information(self, "刪除模型", "先在左邊的清單選輸出資料夾裡的模型（按住 Ctrl 或 Shift 可以選好幾個）。")
            return
        doomed: list[Path] = []
        for path in models:
            doomed += [p for p in outputs_of(path) if p not in doomed]
        shown = [f"  {p.relative_to(output)}" for p in doomed]
        if len(shown) > LISTED:
            shown = shown[:LISTED] + [f"  …另外 {len(doomed) - LISTED} 個"]
        ask = (f"把這 {len(models)} 個模型的檔案移到資源回收筒？" if len(models) > 1 else "把這些移到資源回收筒？")
        if QMessageBox.question(self, "刪除模型", ask + "\n" + "\n".join(shown)) != QMessageBox.StandardButton.Yes:
            return
        if self.panel.current is not None and self.panel.current.resolve() in {p.resolve() for p in doomed}:
            self.panel.show_message("已刪除")
            self.info.clear()
        failed = [p for p in doomed if not trash_soon(p)]
        for path in models:                          # each model's folder, and the ones above it, left empty
            folder = path.parent
            while folder != output and folder.is_dir() and not any(folder.iterdir()):
                trash(folder)
                folder = folder.parent
        if failed:
            QMessageBox.warning(self, "刪除模型", "這些沒能移到資源回收筒：\n" + "\n".join(map(str, failed)))
        self.refresh()

    def _chosen_models(self) -> list[Path]:
        """What a share takes: every ``.glb`` selected in the list (Ctrl- or Shift-click picks several), else the one
        shown."""
        picked = [Path(item.data(Qt.UserRole)) for item in self.files.selectedItems() if item.data(Qt.UserRole)]
        picked = [path for path in picked if path.suffix.lower() == ".glb"]
        if picked:
            return picked
        shown = self.panel.current
        return [Path(shown)] if shown is not None and Path(shown).suffix.lower() == ".glb" else []

    def share(self) -> None:
        """The models chosen (:meth:`_chosen_models`) on a GitHub Pages site as one commit, by a job of its own, their
        addresses copied: through the user's GitHub account (``github``: signed in in the browser the first time, the
        repository and uploads made by the program), or, when a site folder is set, committed and pushed in that clone
        (``share``).  The user is told first that the models become public.  Another share may start while one runs:
        the jobs take turns with the site (``sources.shared_writes``)."""
        models = self._chosen_models()
        if not models:
            QMessageBox.information(self, tr("分享到 GitHub Pages"), tr("先開啟一個 .glb：網頁的 3D 檢視器讀的是 glTF。"))
            return
        from . import github, jobs, share
        site = self.window_.settings.tool("pages_site")
        activity = (self.window_.settings.tools.get(ACTIVITY) or "").strip()
        names = [share.slug(path) for path in models]
        if site is not None:
            try:
                url = share.check_site(site)
            except share.ShareError as failure:
                QMessageBox.warning(self, tr("分享到 GitHub Pages"), tr(str(failure)))
                return
            where, command = str(site), jobs.share_command(models, site, activity)
            first = "這個模型會推送到 GitHub" if len(models) == 1 else f"這 {len(models)} 個模型會一起推送到 GitHub"
        else:
            login = github.account()
            url = github.pages_url(login) if login else f"https://<帳號>.github.io/{github.REPOSITORY}/"
            where, command = f"GitHub {login or ''}".rstrip(), jobs.share_github_command(models, activity)
            first = (f"這個模型會上傳到你 GitHub 帳號的公開倉庫 {github.REPOSITORY}" if len(models) == 1 else
                     f"這 {len(models)} 個模型會一起上傳到你 GitHub 帳號的公開倉庫 {github.REPOSITORY}")
        places = "\n".join(f"  {url}{share.MODELS}/{name}/" for name in names[:LISTED])
        if len(names) > LISTED:
            places += "\n" + tr(f"  …另外 {len(names) - LISTED} 個")
        ask = f"{first}，公開在：\n{places}\n任何拿到網址的人都看得到、也下載得到。"
        if site is None and not github.account():
            ask += "\n第一次會請你在瀏覽器登入 GitHub。"
        if QMessageBox.question(self, tr("分享到 GitHub Pages"), tr(
                ask + "\n模型來自遊戲，公開前請自行確認可以分享。要繼續嗎？")) != QMessageBox.StandardButton.Yes:
            return
        self.window_.log(f"分享 {'、'.join(path.name for path in models[:3])}"
                         f"{f' 等 {len(models)} 個' if len(models) > 3 else ''} 到 {where}")
        self._start_share("share", command)

    def tidy(self) -> None:
        """The site's front page and list written again from the models still on it, by a job of its own: a model
        deleted on GitHub kept its card on the front page until the next share."""
        from . import jobs
        site = self.window_.settings.tool("pages_site")
        activity = (self.window_.settings.tools.get(ACTIVITY) or "").strip()
        command = jobs.tidy_command(site, activity) if site is not None else jobs.tidy_github_command(activity)
        if QMessageBox.question(self, tr("更新網站清單"), tr(
                "依網站上還在的模型重寫首頁與模型清單，已經刪掉的模型會從首頁拿掉。不會上傳任何模型。要繼續嗎？")) \
                != QMessageBox.StandardButton.Yes:
            return
        self.window_.log("更新網站清單")
        self._start_share("tidy", command)

    def _start_share(self, kind: str, command: list[str]) -> None:
        process = QProcess(self)
        run = ShareRun(kind)
        self.share_runs[process] = run
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self._share_output(process))
        process.finished.connect(lambda code, _status: self._share_done(process, code))
        program, *arguments = command
        process.start(program, arguments)

    def _share_output(self, process: QProcess) -> None:
        run = self.share_runs.get(process)
        if run is None:
            return
        for raw in bytes(process.readAllStandardOutput()).decode("utf-8", "replace").splitlines():
            kind, _, text = raw.partition(" ")
            if kind == "URL":
                run.urls.append(text)
            elif kind == "LIVE":
                run.live = text == "yes"
            elif kind == "PAGES":
                run.settings = text
            elif kind == "REMOVED":
                run.removed.append(text)
                self.window_.log(f"  從清單拿掉：{text}")
            elif kind == "DEVICE":
                code, _, address = text.partition(" ")
                self._sign_in(process, code, address)
            elif kind == "ERROR":
                run.error = text
                self.window_.log(f"  錯誤：{text}")
            elif kind == "LOG":
                self.window_.log(f"  {text}")
                self._signed_in(run)
            elif raw.strip() and kind not in ("PROGRESS",):
                # anything else the job printed (a traceback, a library's warning) goes to the log as it is: an
                # error the job did not word itself still has to be seen somewhere
                self.window_.log(f"  {raw}")

    def _signed_in(self, run: "ShareRun") -> None:
        """The sign-in box closed: the job has gone on (closed this way, it does not stop the share)."""
        if run.sign_in_box is not None:
            box, run.sign_in_box = run.sign_in_box, None
            box.done(0)

    def _sign_in(self, process: QProcess, code: str, address: str) -> None:
        """GitHub's device sign-in: the code copied and the page opened; the box stays until the job goes on, and
        closing it stops that share."""
        run = self.share_runs[process]
        QGuiApplication.clipboard().setText(code)
        QDesktopServices.openUrl(QUrl(address))
        run.sign_in_box = QMessageBox(QMessageBox.Information, tr("登入 GitHub"), tr(
            f"在瀏覽器開啟的 {address} 輸入這組代碼（已複製）：\n{code}\n\n授權後這裡會自動繼續。"),
            QMessageBox.Cancel, self)
        run.sign_in_box.setModal(False)
        run.sign_in_box.rejected.connect(lambda: self._stop_share(process))
        run.sign_in_box.show()

    def _stop_share(self, process: QProcess) -> None:
        run = self.share_runs.get(process)
        if run is not None and run.sign_in_box is not None:
            run.error = "登入被取消了"
            process.kill()

    def _share_done(self, process: QProcess, code: int) -> None:
        run = self.share_runs.pop(process, None)
        if run is None:
            return
        self._signed_in(run)
        process.deleteLater()
        self.window_.settings_page.refresh_account()
        if run.kind == "tidy":
            if code != 0 or not run.urls:
                QMessageBox.warning(self, tr("更新網站清單"), tr(run.error) or tr("沒有更新完成，原因寫在下方的紀錄。"))
                return
            url = run.urls[-1]
            removed = "\n".join(run.removed)
            text = (f"已從首頁拿掉 {len(run.removed)} 個已刪除的模型：\n{removed}" if run.removed
                    else "網站清單跟網站上的模型一致，沒有要拿掉的。")
            text += f"\n\nGitHub Pages 發佈要一點時間，之後 {url} 就會更新。"
            self.window_.log(f"  網站清單已更新：{url}")
            box = QMessageBox(QMessageBox.Information, tr("更新網站清單"), tr(text), QMessageBox.Ok, self)
            open_button = box.addButton(tr("在瀏覽器開啟"), QMessageBox.ActionRole)
            box.exec()
            if box.clickedButton() is open_button:
                QDesktopServices.openUrl(QUrl(url))
            return
        if code != 0 or not run.urls:
            QMessageBox.warning(self, tr("分享到 GitHub Pages"), tr(run.error) or tr("分享沒有完成，原因寫在下方的紀錄。"))
            return
        addresses = "\n".join(run.urls)
        QGuiApplication.clipboard().setText(addresses)
        for url in run.urls:
            self.window_.log(f"  已分享：{url}（網址已複製）")
        # a link pasted before Pages serves it is a 404 and Discord shows no card, then keeps that answer a while
        if run.live:
            text = f"網頁已上線，網址已複製到剪貼簿：\n{addresses}\n\n貼到 Discord 會顯示成卡片。"
        else:
            text = (f"已推送，但 GitHub Pages 還沒發佈完成。網址已複製到剪貼簿：\n{addresses}\n\n"
                    "等網頁打得開再貼到 Discord（新倉庫第一次可能要幾分鐘）。\n"
                    "如果先貼了而沒有卡片，在網址後面加上 ?1 再貼一次：Discord 會記住沒有卡片的結果一陣子。")
        if run.settings:
            text += ("\n\n這個倉庫的 GitHub Pages 還沒開啟：按「開啟 Pages 設定」，"
                     "Source 選 Deploy from a branch、分支選 main、資料夾 / (root)，按 Save。只需要做一次。")
        box = QMessageBox(QMessageBox.Information, tr("分享到 GitHub Pages"), tr(text), QMessageBox.Ok, self)
        open_button = box.addButton(tr("在瀏覽器開啟"), QMessageBox.ActionRole)
        settings_button = box.addButton(tr("開啟 Pages 設定"), QMessageBox.ActionRole) if run.settings else None
        box.exec()
        if box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl(run.urls[0]))
        elif settings_button is not None and box.clickedButton() is settings_button:
            QDesktopServices.openUrl(QUrl(run.settings))

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
        # what the file holds and whether older rules wrote it are read on a thread, then the view is given the
        # file: reading them here held the window as long as they took
        self._reading = getattr(self, "_reading", 0) + 1
        reading = self._reading
        self.info.setPlainText(tr("讀取檔案內容…"))

        def read():
            older = False
            if path.suffix.lower() in (".glb", ".gltf"):
                try:
                    older = stale(_gltf(path))
                except (OSError, ValueError, struct.error):
                    pass
            return older, model_info(path)

        def done(result, error) -> None:
            if reading != self._reading:
                return
            older, info = result if error is None else (False, f"{path.name}\n{error}")
            self.panel.show_file(path, STALE + "\n" + path.name if older else path.name)
            self.info.setPlainText(tr(info))

        background(self, read, done)

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

