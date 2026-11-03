"""The README's screenshots: the window in one language, taken through its own flows.

    python tools/screenshots.py [--language en] [--source gf2] [--pick c_LennaSSR01_slg_skin] [--out docs/images]

The window is made with ``WA_DontShowOnScreen``: Qt draws it into its own buffer and ``grab()`` saves that, so
nothing appears on the desktop and nothing takes the focus.  It runs a real scan of the source's install (the
scan job, with its progress bar caught part way), previews the row whose model name contains ``--pick`` (the
preview job and the 3D view), opens that preview in the model viewer, and shows the settings page.

It works from a copy of the saved settings (the language set to ``--language``, the output folder a scratch
one), so the user's own settings are left alone.  The scan and preview use the caches of wherever the data
folder is: set ``ISLANDLIFE_DERIVED_ROOT`` to the packaged program's (``%LOCALAPPDATA%\\Polylode``) to reuse
its bundle index and stores instead of building them again in the checkout.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", default="en")
    parser.add_argument("--source", default="gf2")
    parser.add_argument("--pick", default="c_LennaSSR01_slg_skin", help="part of the model name of the row to preview")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    from PySide6.QtCore import QItemSelectionModel, Qt
    from PySide6.QtWidgets import QApplication
    from polylode.extractor import app as A
    from polylode.extractor import i18n
    from polylode.extractor import settings as S

    work = Path(tempfile.mkdtemp(prefix="polylode_shots_"))
    output = work / "output"
    output.mkdir()
    saved = S.load()
    S.SETTINGS = work / "settings.json"                      # the copy: the user's own file is not written
    S.save(replace(saved, language=args.language, output=str(output), formats=["glb", "fbx"]))

    qt = QApplication.instance() or QApplication([])
    qt.setStyle("Fusion")
    qt.setStyleSheet(A.STYLE)
    i18n.set_language(args.language)
    i18n.install(qt)
    window = A.Window()
    window.setAttribute(Qt.WA_DontShowOnScreen)
    window.resize(args.width, args.height)
    window.show()
    A.retranslate()
    args.out.mkdir(parents=True, exist_ok=True)

    def pump(seconds: float, until=lambda: False) -> bool:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            qt.processEvents()
            if until():
                return True
            time.sleep(0.02)
        return until()

    def shoot(name: str) -> None:
        pump(0.5)
        path = args.out / f"{name}.png"
        window.grab().save(str(path))
        print(f"wrote {path}")

    page = next(window.pages.widget(i) for i in range(window.pages.count())
                if getattr(getattr(window.pages.widget(i), "source", None), "id", "") == args.source)
    window.nav.setCurrentRow(window.pages.indexOf(page))

    # the scan, caught while its bar counts
    page.scan()
    if page.process is None:
        print(f"the scan did not start: {page.status.text()}")
        return 1
    if pump(1800, lambda: page.scan_bar.isVisible() and page.scan_bar.bar.maximum() > 0
            and page.scan_bar.bar.value() >= page.scan_bar.bar.maximum() // 4):
        shoot("scan")
    else:
        print("no counted step showed during the scan: no scan picture")
    if not pump(3600, lambda: page.process is None):
        print("the scan did not finish")
        return 1
    print(page.status.text())

    # a preview: the row, the preview job, the view once the file is in
    page.search.setText(args.pick)
    pump(0.3)
    if page.proxy.rowCount() == 0:
        print(f"no row has {args.pick!r}")
        return 1
    index = page.proxy.index(0, 0)
    page.table.selectionModel().setCurrentIndex(index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
    page.preview()
    pump(600, lambda: page.preview_process is None)
    root = page.preview_panel.view.rootObject()
    if not pump(120, lambda: root.property("state_") == "ready"):
        print(f"the preview did not load: {root.property('state_')} / {page.preview_panel.caption.text()}")
        return 1
    page.search.setText("")
    page.table.scrollTo(page.table.currentIndex())
    shoot("preview")

    # the model viewer, on that preview copied into the scratch output folder
    shown = page.preview_panel.current
    target = output / args.source / shown.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shown, target)
    viewer = window.viewer_page
    window.nav.setCurrentRow(window.pages.indexOf(viewer))
    viewer.refresh()
    pump(0.3)
    viewer.files.setCurrentRow(0)
    viewer_root = viewer.panel.view.rootObject()
    pump(120, lambda: viewer_root.property("state_") == "ready")
    viewer_root.look(35, -10)                       # a three-quarter view: the viewer's fixed views turn it
    shoot("viewer")

    window.nav.setCurrentRow(window.pages.indexOf(window.settings_page))
    shoot("settings")
    window.close()
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
