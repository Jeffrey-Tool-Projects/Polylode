"""The window: name the paths, scan an install, tick models, export them; every scan and export in its own process."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PySide6.QtCore import QProcess, QSortFilterProxyModel, Qt
from PySide6.QtGui import QIcon, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QHeaderView, QInputDialog,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSplitter, QStackedWidget, QTableView, QVBoxLayout, QWidget,
)

from . import i18n, jobs, locate
from .i18n import tr
from . import settings as S
from .preview import PreviewPanel, cache_folder, cached, preview_settings
from .sources import Model, PathSpec, registry
from .stepbar import StepBar

TITLE = "Polylode"
#: drawn by ``tools/make_extractor_icon.py``; the packaged program's own icon is its ``icon.ico``
ICON = Path(__file__).with_name("icon.png")
#: Tools every source shares.
COMMON = (PathSpec("blender", "Blender 執行檔", "file", "blender.exe（4.x 或 5.x）；.blend 需要，.glb／FBX 有它時多做讀回核對"),)
FORMATS = (("glb", "glTF（.glb，貼圖內嵌）"), ("fbx", "FBX（貼圖內嵌）"),
           ("blend", "Blender（.blend，貼圖打包在檔內，需 Blender）"))

STYLE = """
QWidget { background: #1b1d23; color: #e6e8ee; font-size: 10pt;
          font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "Segoe UI", sans-serif; }
QFrame#side { background: #14161b; }
QListWidget#nav { border: none; background: transparent; }
QListWidget#nav::item { padding: 10px 14px; border-radius: 8px; margin: 2px 8px; }
QListWidget#nav::item:selected { background: #2f6feb; color: white; }
QLabel#title { font-size: 16pt; font-weight: 600; }
QLabel#muted { color: #9aa0ad; }
QLineEdit, QPlainTextEdit, QTableView { background: #23262e; border: 1px solid #333846; border-radius: 6px; padding: 4px; }
QTableView { gridline-color: #2c303a; selection-background-color: #2f6feb; }
QHeaderView::section { background: #23262e; border: none; padding: 6px; color: #9aa0ad; }
QPushButton { background: #2c303a; border: 1px solid #3a3f4c; border-radius: 6px; padding: 6px 14px; }
QPushButton:hover { background: #363b47; }
QPushButton#primary { background: #2f6feb; border: none; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #4682f0; }
QPushButton:disabled { color: #6b7180; }
QProgressBar { border: 1px solid #333846; border-radius: 6px; text-align: center; background: #23262e; }
QProgressBar::chunk { background: #2f6feb; border-radius: 6px; }
"""


class Lines:
    """A process's output as whole lines: a read can end inside a line (or a UTF-8 character), and that part
    waits for the rest instead of being parsed as a line of its own."""

    def __init__(self):
        self.pending = b""

    def feed(self, data: bytes) -> list[str]:
        self.pending += bytes(data)
        *complete, self.pending = self.pending.split(b"\n")
        return [line.rstrip(b"\r").decode("utf-8", "replace") for line in complete]

    def rest(self) -> list[str]:
        """What is left when the process has ended (a last line without a newline)."""
        left, self.pending = self.pending, b""
        return [left.rstrip(b"\r").decode("utf-8", "replace")] if left.strip() else []


def failed_start(process: QProcess) -> bool:
    return process.error() == QProcess.ProcessError.FailedToStart


class PathRow(QWidget):
    """A path field with a browse button and a mark saying whether it exists."""

    def __init__(self, spec: PathSpec, value: str, on_change):
        super().__init__()
        self.spec = spec
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(value)
        self.edit.setPlaceholderText(spec.hint)
        self.mark = QLabel()
        browse = QPushButton("瀏覽…")
        browse.clicked.connect(self._browse)
        self.edit.textChanged.connect(lambda _text: (self._check(), on_change()))
        layout.addWidget(self.edit, 1)
        layout.addWidget(browse)
        layout.addWidget(self.mark)
        self._check()

    def _browse(self) -> None:
        if self.spec.kind == "dir":
            chosen = QFileDialog.getExistingDirectory(self, tr(self.spec.label), self.edit.text())
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, tr(self.spec.label), self.edit.text())
        if chosen:
            self.edit.setText(chosen)

    def _check(self) -> None:
        text = self.edit.text().strip()
        path = Path(text) if text else None
        ok = path is not None and (path.is_dir() if self.spec.kind == "dir" else path.is_file())
        self.mark.setText("✓" if ok else ("✗" if text else "—"))
        self.mark.setStyleSheet(f"color: {'#3fb950' if ok else ('#f85149' if text else '#6b7180')}")

    def value(self) -> str:
        return self.edit.text().strip()


def _label(text: str) -> QLabel:
    """A form label that wraps instead of setting the form's width."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setMaximumWidth(260)
    return label


class SettingsPage(QWidget):
    def __init__(self, settings: S.Settings, sources: dict):
        super().__init__()
        self.settings = settings
        # the page scrolls instead of widening the window: its longest label and three format boxes set the
        # window's minimum width at 1,393 px, wider than a laptop screen at 150 % scaling (§7.1455)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        title = QLabel("路徑設定")
        title.setObjectName("title")
        layout.addWidget(title)
        hint = QLabel("指定各遊戲的安裝資料夾與需要的工具；設定存在你的使用者資料夾。")
        hint.setObjectName("muted")
        layout.addWidget(hint)
        form = QFormLayout()
        # a narrow page puts each label above its field rather than squeezing either
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.language = QComboBox()
        self.language.addItem("跟隨系統", "")
        for code, name in i18n.LANGUAGES.items():
            self.language.addItem(name, code)
        self.language.setCurrentIndex(max(0, self.language.findData(settings.language or "")))
        self.language.currentIndexChanged.connect(self._language_changed)
        form.addRow(_label("語言"), self.language)
        self.rows: list[PathRow] = []
        for spec in COMMON:
            self._add(form, spec)
        for source in sources.values():
            header = QLabel(source.label)
            header.setStyleSheet("font-weight: 600; margin-top: 12px;")
            form.addRow(header)
            for spec in source.paths:
                self._add(form, spec)
        self.output = PathRow(PathSpec("output", "輸出資料夾", "dir"), settings.output, self._save)
        form.addRow(_label("輸出資料夾"), self.output)
        self.format_boxes = {}
        formats = QVBoxLayout()
        for key, label in FORMATS:
            box = QCheckBox(label)
            box.setChecked(key in settings.formats)
            box.toggled.connect(lambda _on: self._save())
            self.format_boxes[key] = box
            formats.addWidget(box)
        form.addRow(_label("匯出格式"), formats)
        layout.addLayout(form)
        layout.addStretch(1)

    def _language_changed(self) -> None:
        """Another language: kept in the settings and shown at once in every open window."""
        self.settings.language = self.language.currentData() or ""
        self._save()
        i18n.set_language(self.settings.language)
        retranslate()

    def set_path(self, key: str, value: str) -> None:
        """Put *value* in the field of path *key* (it saves the settings as typing would)."""
        for row in self.rows:
            if row.spec.key == key:
                row.edit.setText(value)

    def _add(self, form, spec: PathSpec) -> None:
        value = (self.settings.installs if spec.install else self.settings.tools).get(spec.key, "")
        row = PathRow(spec, value, self._save)
        self.rows.append(row)
        form.addRow(_label(spec.label), row)

    def _save(self) -> None:
        for row in self.rows:
            (self.settings.installs if row.spec.install else self.settings.tools)[row.spec.key] = row.value()
        self.settings.output = self.output.value() if hasattr(self, "output") else self.settings.output
        if hasattr(self, "format_boxes"):
            self.settings.formats = [key for key, box in self.format_boxes.items() if box.isChecked()]
        S.save(self.settings)


class SourcePage(QWidget):
    """One source: scan its install, filter, tick models, queue them."""

    def __init__(self, window, source):
        super().__init__()
        self.window_ = window
        self.source = source
        self.models: list[Model] = []
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel(source.label)
        title.setObjectName("title")
        top.addWidget(title)
        top.addStretch(1)
        self.scan_button = QPushButton("掃描安裝")
        self.scan_button.clicked.connect(self.scan)
        top.addWidget(self.scan_button)
        layout.addLayout(top)
        self.status = QLabel("尚未掃描")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        self.scan_bar = StepBar()
        layout.addWidget(self.scan_bar)
        tools = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜尋角色、衣裝或編號…")
        tools.addWidget(self.search, 1)
        for text, state in (("全選", Qt.Checked), ("全不選", Qt.Unchecked)):
            button = QPushButton(text)
            button.clicked.connect(lambda _c, s=state: self._tick_visible(s))
            tools.addWidget(button)
        self.preview_button = QPushButton("預覽")
        self.preview_button.setToolTip("轉出選取那一列的 .glb 並在右邊顯示（雙擊一列也可以）")
        self.preview_button.clicked.connect(self.preview)
        tools.addWidget(self.preview_button)
        self.export_button = QPushButton("匯出勾選的模型")
        self.export_button.setObjectName("primary")
        self.export_button.clicked.connect(self.export)
        tools.addWidget(self.export_button)
        layout.addLayout(tools)
        self.table_model = QStandardItemModel(0, 3)
        self.table_model.setHorizontalHeaderLabels(["角色", "衣裝／版本", "模型編號"])
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(self.table_model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.doubleClicked.connect(lambda _index: self.preview())
        self.table.selectionModel().currentRowChanged.connect(lambda _now, _before: self._show_cached())
        self.preview_panel = PreviewPanel()
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.table)
        split.addWidget(self.preview_panel)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)
        self.process: QProcess | None = None
        self.preview_process: QProcess | None = None

    def ask_install(self) -> bool:
        """Before a scan: the source's install folders, asked for when unset, gone or without the game's data;
        the games Steam's libraries hold are offered first (``locate``). False when none was given."""
        settings = self.window_.settings
        for spec in (spec for spec in self.source.paths if spec.install):
            current = settings.installs.get(spec.key, "")
            if current and Path(current).is_dir() and (not spec.marker or locate.game_root(Path(current), spec.marker)):
                continue
            chosen, other = "", tr("選擇其他資料夾…")
            found = locate.steam_installs(spec.marker) if spec.marker else []
            if found:
                picked, ok = QInputDialog.getItem(self, tr(f"選擇{spec.label}"), tr("在 Steam 遊戲庫找到："),
                                                  [str(p) for p in found] + [other], 0, False)
                if not ok:
                    self.status.setText(tr(f"還沒有選{spec.label}，沒有掃描"))
                    return False
                chosen = "" if picked == other else picked
            if not chosen:
                chosen = QFileDialog.getExistingDirectory(self, tr(f"選擇{spec.label}（{spec.hint}）"), current)
            if not chosen:
                self.status.setText(tr(f"還沒有選{spec.label}，沒有掃描"))
                return False
            if spec.marker:
                root = locate.game_root(Path(chosen), spec.marker)
                if root is None:
                    self.status.setText(tr(f"{chosen} 裡找不到唯一一份 {spec.marker}，沒有掃描"))
                    return False
                chosen = str(root)
            self.window_.settings_page.set_path(spec.key, chosen)
        return True

    def ask_tools(self) -> bool:
        """Before a preview or an export: each tool the source cannot convert without (``PathSpec.required``),
        asked for when unset or gone.  False when one was not given."""
        settings = self.window_.settings
        for spec in (spec for spec in self.source.paths if spec.required and not spec.install):
            current = settings.tool(spec.key)
            if current is not None and (current.is_dir() if spec.kind == "dir" else current.is_file()):
                continue
            title = tr(f"選擇{spec.label}") + (f"（{tr(spec.hint)}）" if spec.hint else "")
            start = str(current.parent) if current is not None else ""
            chosen = (QFileDialog.getExistingDirectory(self, title, start) if spec.kind == "dir"
                      else QFileDialog.getOpenFileName(self, title, start)[0])
            if not chosen:
                self.status.setText(tr(f"還沒有選{spec.label}，沒有轉換"))
                return False
            self.window_.settings_page.set_path(spec.key, chosen)
        self.window_.settings_page._save()
        return True

    def scan(self) -> None:
        if self.process is not None:
            return
        self.window_.settings_page._save()
        if not self.ask_install():
            return
        settings_file = Path(tempfile.gettempdir()) / f"extractor_scan_{self.source.id}.json"
        S.save(self.window_.settings, settings_file)
        self.models = []
        self.table_model.removeRows(0, self.table_model.rowCount())
        self.status.setText(tr("掃描中…"))
        self.scan_bar.start("掃描中")
        self.scan_button.setEnabled(False)
        self.process = QProcess(self)
        self.lines = Lines()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._scan_output)
        self.process.finished.connect(self._scan_done)
        self.process.errorOccurred.connect(lambda _error: failed_start(self.process) and self._scan_done(-1, None))
        program, *arguments = jobs.scan_command(self.source.id, settings_file)
        self.process.start(program, arguments)

    def _scan_output(self) -> None:
        self._scan_lines(self.lines.feed(self.process.readAllStandardOutput()))

    def _scan_lines(self, lines: list[str]) -> None:
        for line in lines:
            if self.scan_bar.take(line):
                continue
            if line.startswith("MODEL "):
                model = Model(**json.loads(line[6:]))
                self.models.append(model)
                row = [QStandardItem(model.character), QStandardItem(model.outfit), QStandardItem(model.ident)]
                row[0].setCheckable(True)
                for item in row:
                    item.setEditable(False)
                row[0].setData(len(self.models) - 1, Qt.UserRole)
                self.table_model.appendRow(row)
            else:
                self.window_.log(f"[{self.source.label}] {line}")

    def _scan_done(self, code: int, status) -> None:
        if self.process is None:
            return
        self._scan_lines(self.lines.rest())
        crashed = status == QProcess.ExitStatus.CrashExit
        if code == -1:
            self.window_.log(f"[{self.source.label}] 無法啟動掃描：{self.process.errorString()}")
        ok = code == 0 and not crashed
        self.status.setText(tr(f"找到 {len(self.models)} 個可匯出的模型" if ok else "掃描失敗，見下方記錄"))
        self.scan_bar.stop()
        self.scan_button.setEnabled(True)
        self.process = None

    def _selected(self) -> Model | None:
        index = self.table.currentIndex()
        if not index.isValid():
            return None
        item = self.table_model.item(self.proxy.mapToSource(index).row(), 0)
        return self.models[item.data(Qt.UserRole)]

    def _show_cached(self) -> None:
        """A row whose preview was made before is shown at once; any other leaves the view as it is."""
        model = self._selected()
        path = cached(model) if model is not None else None
        if path is not None and self.preview_process is None:
            self.preview_panel.show_file(path, f"{model.character}　{model.outfit}（之前的預覽）")

    def preview(self) -> None:
        """The selected row as a turnable model: from the cache, or converted to .glb by a job of its own."""
        model = self._selected()
        if model is None:
            QMessageBox.information(self, TITLE, "先在表格選一列。")
            return
        path = cached(model)
        if path is not None:
            self.preview_panel.show_file(path, f"{model.character}　{model.outfit}（之前的預覽）")
            return
        if self.preview_process is not None:
            return
        self.window_.settings_page._save()
        if not self.ask_tools():
            return
        folder = cache_folder(model)
        folder.mkdir(parents=True, exist_ok=True)
        job = jobs.write_job(Path(tempfile.gettempdir()) / f"extractor_preview_{model.source}.json", model,
                             preview_settings(self.window_.settings), folder, ["glb"])
        self.preview_model, self.preview_written = model, None
        self.preview_panel.show_message(f"轉換 {model.character}　{model.outfit} 以預覽（第一次要幾十秒到幾分鐘）…")
        self.preview_panel.steps.start("轉換模型以預覽")
        self.preview_button.setEnabled(False)
        self.preview_process = QProcess(self)
        self.preview_lines = Lines()
        self.preview_process.setProcessChannelMode(QProcess.MergedChannels)
        self.preview_process.readyReadStandardOutput.connect(self._preview_output)
        self.preview_process.finished.connect(self._preview_done)
        self.preview_process.errorOccurred.connect(
            lambda _error: failed_start(self.preview_process) and self._preview_done(-1, None))
        program, *arguments = jobs.command(job)
        self.window_.log(f"預覽 {model.ident}")
        self.preview_process.start(program, arguments)

    def _preview_output(self) -> None:
        self._preview_lines(self.preview_lines.feed(self.preview_process.readAllStandardOutput()))

    def _preview_lines(self, lines: list[str]) -> None:
        for line in lines:
            if self.preview_panel.steps.take(line):
                continue
            kind, _, text = line.partition(" ")
            if kind == "OUTPUT" and text.endswith(".glb"):
                self.preview_written = Path(text)
            elif kind == "ERROR":
                self.window_.log(f"  預覽錯誤：{text}")
                self.preview_error = text

    def _preview_done(self, code: int, status) -> None:
        if self.preview_process is None:
            return
        self.preview_error = getattr(self, "preview_error", "")
        self._preview_lines(self.preview_lines.rest())
        model, written = self.preview_model, self.preview_written
        self.preview_panel.steps.stop()
        if code == -1:
            self.preview_panel.show_message(f"無法啟動預覽：{self.preview_process.errorString()}")
        elif written is not None and written.is_file():
            self.preview_panel.show_file(written, f"{model.character}　{model.outfit}")
        else:
            self.preview_panel.show_message(f"預覽失敗：{self.preview_error or '沒有寫出 .glb'}")
        self.preview_error = ""
        self.preview_process = None
        self.preview_button.setEnabled(True)

    def _tick_visible(self, state) -> None:
        for row in range(self.proxy.rowCount()):
            item = self.table_model.itemFromIndex(self.proxy.mapToSource(self.proxy.index(row, 0)))
            item.setCheckState(state)

    def export(self) -> None:
        chosen = [self.models[self.table_model.item(row, 0).data(Qt.UserRole)]
                  for row in range(self.table_model.rowCount())
                  if self.table_model.item(row, 0).checkState() == Qt.Checked]
        if not chosen:
            QMessageBox.information(self, TITLE, "先勾選要匯出的模型。")
            return
        if not self.ask_tools():
            return
        self.window_.queue(chosen)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.setWindowIcon(QIcon(str(ICON)))
        # fitted to the screen it opens on: 1180 × 760 at most, and never past 90 % of what is free
        screen = QApplication.primaryScreen()
        free = screen.availableGeometry() if screen is not None else None
        self.resize(min(1180, int(free.width() * 0.9)) if free else 1180,
                    min(760, int(free.height() * 0.9)) if free else 760)
        self.settings = S.load()
        self.sources = registry()
        self.pending: list[QListWidgetItem] = []
        self.process: QProcess | None = None
        self.queued = self.finished = 0
        self.current: QListWidgetItem | None = None

        side = QFrame()
        side.setObjectName("side")
        side.setFixedWidth(180)
        side_layout = QVBoxLayout(side)
        name = QLabel(TITLE)
        name.setStyleSheet("font-weight: 600; padding: 12px;")
        name.setWordWrap(True)
        side_layout.addWidget(name)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.nav.setWordWrap(True)
        side_layout.addWidget(self.nav, 1)

        self.pages = QStackedWidget()
        self.settings_page = SettingsPage(self.settings, self.sources)
        for source in self.sources.values():
            self.nav.addItem(source.label)
            self.pages.addWidget(SourcePage(self, source))
        from .viewer import ViewerPage
        self.viewer_page = ViewerPage(self)
        self.nav.addItem("模型檢視器")
        self.pages.addWidget(self.viewer_page)
        self.nav.addItem("設定")
        self.pages.addWidget(self.settings_page)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)

        self.queue_list = QListWidget()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        left = QVBoxLayout()
        left.addWidget(QLabel("匯出佇列"))
        left.addWidget(self.queue_list, 1)
        left.addWidget(self.progress)
        self.job_bar = StepBar()
        left.addWidget(self.job_bar)
        bottom_layout.addLayout(left, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("記錄"))
        right.addWidget(self.log_view, 1)
        bottom_layout.addLayout(right, 2)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.pages)
        split.addWidget(bottom)
        # the page takes what the window gains; the queue and log can be dragged shut on a small screen
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setCollapsible(0, False)
        split.setCollapsible(1, True)
        split.setSizes([480, 200])
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(side)
        body_layout.addWidget(split, 1)
        self.setCentralWidget(body)

    def log(self, line: str) -> None:
        self.log_view.appendPlainText(tr(line))

    def queue(self, models: list[Model]) -> None:
        self.settings_page._save()
        if not self.settings.output or not Path(self.settings.output).is_dir():
            QMessageBox.warning(self, TITLE, "先在「設定」指定存在的輸出資料夾。")
            return
        if not self.settings.formats:
            QMessageBox.warning(self, TITLE, "先在「設定」至少勾選一種匯出格式。")
            return
        for model in models:
            item = QListWidgetItem(f"⏳ {model.character}　{model.outfit}　({model.ident})")
            item.setData(Qt.UserRole, model)
            self.queue_list.addItem(item)
            self.pending.append(item)
        self.queued += len(models)
        self.progress.setRange(0, self.queued)
        self.progress.setValue(self.finished)
        self._next()

    def _next(self) -> None:
        if self.process is not None:
            return
        if not self.pending:
            self.job_bar.stop()
            return
        self.current = self.pending.pop(0)
        model: Model = self.current.data(Qt.UserRole)
        self.current.setText(f"▶ {model.character}　{model.outfit}　({model.ident})")
        output = Path(self.settings.output) / model.source / model.ident
        job = jobs.write_job(Path(tempfile.gettempdir()) / f"extractor_job_{model.source}_{model.ident}.json",
                             model, self.settings, output, self.settings.formats)
        self.process = QProcess(self)
        self.lines = Lines()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._job_output)
        self.process.finished.connect(self._job_done)
        self.process.errorOccurred.connect(lambda _error: failed_start(self.process) and self._job_done(-1, None))
        program, *arguments = jobs.command(job)
        self.log(f"開始匯出 {model.ident} → {output}")
        self.job_bar.start("開始匯出")
        self.process.start(program, arguments)

    def _job_output(self) -> None:
        self._job_lines(self.lines.feed(self.process.readAllStandardOutput()))

    def _job_lines(self, lines: list[str]) -> None:
        for line in lines:
            if self.job_bar.take(line):
                continue
            kind, _, text = line.partition(" ")
            self.log({"OUTPUT": f"  寫出 {text}", "ERROR": f"  錯誤：{text}"}.get(kind, f"  {text if kind == 'LOG' else line}"))

    def _job_done(self, code: int, status) -> None:
        if self.process is None:
            return
        self._job_lines(self.lines.rest())
        if code == -1:
            self.log(f"  無法啟動匯出：{self.process.errorString()}")
        ok = code == 0 and status != QProcess.ExitStatus.CrashExit
        model: Model = self.current.data(Qt.UserRole)
        self.current.setText(f"{'✓' if ok else '✗'} {model.character}　{model.outfit}　({model.ident})")
        self.finished += 1
        self.progress.setValue(self.finished)
        self.process = None
        self._next()

    def closeEvent(self, event) -> None:
        """Ask before closing over a running export or scan: closing stops it and leaves its file half written."""
        running = self.process is not None or any(getattr(self.pages.widget(i), "process", None) is not None
                                                  for i in range(self.pages.count()))
        if running and QMessageBox.question(self, TITLE, "還有匯出或掃描在進行，關閉會中止它。要關閉嗎？") \
                != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        # a preview only fills a cache: it is stopped without asking
        for i in range(self.pages.count()):
            process = getattr(self.pages.widget(i), "preview_process", None)
            if process is not None:
                process.kill()
                process.waitForFinished(3000)
        event.accept()


def retranslate() -> None:
    """Every open window in the language now chosen: widgets, the preview's own texts, the combo's first item."""
    app = QApplication.instance()
    i18n.retranslate(app)
    for window in app.topLevelWidgets():
        for panel in window.findChildren(PreviewPanel):
            panel.apply_texts()
        for combo in window.findChildren(QComboBox):
            if combo.count() and combo.itemData(0) == "":
                combo.setItemText(0, tr("跟隨系統"))


def run(open_file: Path | None = None) -> int:
    """The window; with *open_file* (``ModelExtractor.exe <model file>``), opened on the viewer showing it."""
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    i18n.set_language(S.load().language)
    i18n.install(app)
    window = Window()
    if open_file is not None:
        window.nav.setCurrentRow(window.pages.indexOf(window.viewer_page))
        window.viewer_page.show_file(open_file)
    window.show()
    retranslate()
    return app.exec()
