"""The window in each language (``extractor/i18n.py``)."""
import json

import pytest

from polylode.extractor import i18n
from polylode.extractor import settings as S


@pytest.fixture(autouse=True)
def _source_language():
    yield
    i18n.set_language("zh-TW")


def test_every_entry_has_every_language_and_its_values():
    catalog = json.loads(i18n.CATALOG.read_text(encoding="utf-8"))
    for source, targets in catalog.items():
        assert set(targets) == {"en", "zh-CN", "ja", "ko"}, source
        for target in targets.values():                      # a pattern's values all come through
            for n in range(source.count("{")):
                assert "{%d}" % n in target, (source, target)


def test_texts_patterns_values_and_indented_log_lines_are_turned():
    i18n.set_language("en")
    assert i18n.tr("掃描安裝") == "Scan install"
    assert i18n.tr("找到 223 個可匯出的模型") == "Found 223 models to export"
    # a value that is itself a message is turned too, and a log line keeps its indent
    assert i18n.tr("  錯誤：安裝裡沒有 chr_x") == "  Error: chr_x is not in the install"
    assert i18n.tr("  可玩角色預製物 42 個") == "  42 playable character prefabs"
    assert i18n.tr("沒有人寫過的句子") == "沒有人寫過的句子"                 # unknown: shown as it is
    i18n.set_language("zh-TW")
    assert i18n.tr("掃描安裝") == "掃描安裝"


def test_the_window_turns_into_a_language_and_back(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QPushButton
    from polylode.extractor import app as A
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    S.save(S.Settings())
    qt = QApplication.instance() or QApplication([])
    i18n.set_language("en")
    window = A.Window()
    A.retranslate()
    navigation = [window.nav.item(i).text() for i in range(window.nav.count())]
    assert "Settings" in navigation and "Model viewer" in navigation
    assert any(b.text() == "Scan install" for b in window.findChildren(QPushButton))
    i18n.set_language("ja")                                   # from the source text, not from the English
    A.retranslate()
    assert "設定" in [window.nav.item(i).text() for i in range(window.nav.count())]
    assert any(b.text() == "インストールをスキャン" for b in window.findChildren(QPushButton))
    i18n.set_language("zh-TW")
    A.retranslate()
    assert any(b.text() == "掃描安裝" for b in window.findChildren(QPushButton))
    window.close()
    del qt
