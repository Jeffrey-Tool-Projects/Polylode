"""The window's languages: Traditional Chinese (the source text), Simplified Chinese, English, Japanese, Korean.

Every text the program shows is written in Traditional Chinese; ``i18n.json`` holds each one in the other
languages, a message with values in it as a pattern (``"找到 {0} 個可匯出的模型"``).  :func:`tr` turns a text
into the chosen language — whole, else line by line, else by pattern, with the values inside a pattern turned
too (an error inside "  錯誤：{0}").  A text the catalogue does not hold is shown as it is, never blank.

The widgets are turned where they are shown: :class:`Translator` watches every widget and dialog of the
application and turns its texts when it appears, keeping the source text so another language can be chosen
while the window is open (:func:`retranslate`).  Texts the window sets later (the status line, the log, the
preview's caption, a file's contents) are turned where they are set.  The scan and export processes write in
the source language; their lines are turned when the window shows them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PySide6.QtCore import QEvent, QLocale, QObject, Qt
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QAbstractItemView, QApplication, QComboBox, QListWidget, QListWidgetItem, QWidget

CATALOG = Path(__file__).with_name("i18n.json")
SOURCE = "zh-TW"
#: code → the language's own name, in the order the settings list them
LANGUAGES = {"en": "English", "zh-TW": "繁體中文", "zh-CN": "简体中文", "ja": "日本語", "ko": "한국어"}
_state = {"language": SOURCE, "catalog": None, "patterns": None}
#: where a widget keeps the source text of each of its texts, and what was last shown from it
_SOURCE, _SHOWN = "i18nSource_", "i18nShown_"
_ROLE = Qt.UserRole + 70


def _catalog() -> tuple[dict, list]:
    if _state["catalog"] is None:
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        patterns = []
        for source in catalog:
            if re.search(r"\{\d+\}", source):
                parts = re.split(r"(\{\d+\})", source)
                regex = "".join("(.+?)" if re.fullmatch(r"\{\d+\}", p) else re.escape(p) for p in parts)
                literal = sum(len(p) for p in parts if not re.fullmatch(r"\{\d+\}", p))
                patterns.append((literal, re.compile(regex, re.DOTALL), source))
        patterns.sort(key=lambda row: -row[0])           # the most specific pattern first
        _state["catalog"], _state["patterns"] = catalog, patterns
    return _state["catalog"], _state["patterns"]


def system_language() -> str:
    """The system's language as one of :data:`LANGUAGES`: Traditional for Taiwan, Hong Kong and Macao,
    Simplified for other Chinese, English for any language not listed."""
    name = QLocale.system().name()                       # e.g. zh_TW, zh_CN, ja_JP
    if name.startswith("zh"):
        return "zh-TW" if name[3:] in ("TW", "HK", "MO") or "Hant" in QLocale.system().bcp47Name() else "zh-CN"
    return {"ja": "ja", "ko": "ko"}.get(name[:2], "en")


def set_language(code: str | None) -> str:
    """Show texts in *code* (empty or unknown: the system's language); the language now used."""
    _state["language"] = code if code in LANGUAGES else system_language()
    return _state["language"]


def language() -> str:
    return _state["language"]


def _one(text: str, depth: int) -> str | None:
    catalog, patterns = _catalog()
    lang = _state["language"]
    entry = catalog.get(text)
    if entry is not None:
        return entry.get(lang, text)
    if "\n" in text:
        # a pattern is one line's: across lines it took a block's first colon for its own (the viewer's
        # "材質：" stayed untranslated), so several lines go line by line (tr)
        return None
    for _literal, regex, source in patterns:
        match = regex.fullmatch(text)
        if match:
            values = [tr(value, depth + 1) if depth < 3 else value for value in match.groups()]
            return catalog[source].get(lang, source).format(*values)
    return None


def tr(text: str, depth: int = 0) -> str:
    """*text* in the chosen language, as much of it as the catalogue holds."""
    if not text or _state["language"] == SOURCE:
        return text
    whole = _one(text, depth)
    if whole is not None:
        return whole
    return "\n".join(_line(line, depth) for line in text.split("\n"))


def _line(line: str, depth: int) -> str:
    """One line, its leading spaces kept (the log indents a job's lines)."""
    turned = _one(line, depth)
    if turned is not None:
        return turned
    stripped = line.lstrip()
    if stripped != line:
        inner = _one(stripped, depth)
        if inner is not None:
            return line[:len(line) - len(stripped)] + inner
    return line


def _turn(widget: QWidget, attribute: str, setter: str) -> None:
    """One text of *widget*: from its source text, which is what the program set unless it is what was shown."""
    current = getattr(widget, attribute)()
    if not current:
        return
    shown = widget.property(_SHOWN + attribute)
    source = widget.property(_SOURCE + attribute) if current == shown else current
    widget.setProperty(_SOURCE + attribute, source)
    turned = tr(source)
    widget.setProperty(_SHOWN + attribute, turned)
    if turned != current:
        getattr(widget, setter)(turned)


def _set(item, role: int, value) -> None:
    """``QListWidgetItem.setData(role, value)``; ``QStandardItem.setData(value, role)``."""
    if isinstance(item, QListWidgetItem):
        item.setData(role, value)
    else:
        item.setData(value, role)


def _turn_items(items) -> None:
    """List and header items, as :func:`_turn`: from the source text unless the program set another."""
    for item in items:
        if item is None:
            continue
        current = item.text()
        source, shown = item.data(_ROLE), item.data(_ROLE + 1)
        if source is None or current != shown:
            source = current
        turned = tr(source)
        _set(item, _ROLE, source)
        _set(item, _ROLE + 1, turned)
        if turned != current:
            item.setText(turned)


def translate(widget: QWidget) -> None:
    """*widget* and every widget inside it in the chosen language."""
    for w in [widget, *widget.findChildren(QWidget)]:
        for attribute, setter in (("text", "setText"), ("placeholderText", "setPlaceholderText"),
                                  ("toolTip", "setToolTip"), ("labelText", "setLabelText"),
                                  ("informativeText", "setInformativeText")):
            if hasattr(w, attribute) and hasattr(w, setter) and not isinstance(w, QComboBox):
                try:
                    _turn(w, attribute, setter)
                except TypeError:
                    pass
        if w.isWindow():
            _turn(w, "windowTitle", "setWindowTitle")
        if isinstance(w, QListWidget):
            _turn_items(w.item(i) for i in range(w.count()))
        elif isinstance(w, QAbstractItemView) and isinstance(w.model(), QStandardItemModel):
            model = w.model()
            _turn_items(model.horizontalHeaderItem(c) for c in range(model.columnCount()))
        elif isinstance(w, QAbstractItemView) and hasattr(w.model(), "sourceModel") \
                and isinstance(w.model().sourceModel(), QStandardItemModel):
            model = w.model().sourceModel()
            _turn_items(model.horizontalHeaderItem(c) for c in range(model.columnCount()))


class Translator(QObject):
    """Turns each widget and dialog of the application into the chosen language as it is shown."""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget) and watched.isWindow():
            translate(watched)
        return False


def install(app: QApplication) -> Translator:
    translator = Translator(app)
    app.installEventFilter(translator)
    return translator


def retranslate(app: QApplication) -> None:
    """Every open window again, after the language changed."""
    for window in app.topLevelWidgets():
        translate(window)
