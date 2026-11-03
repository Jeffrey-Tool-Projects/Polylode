"""``python -m polylode.extractor`` (or the packaged program): the window, ``--run-job <job.json>``
``--scan <source> <settings.json>``, ``--share <model.glb> <site folder>``, ``--share-github <model.glb>``,
``--selftest <report.json>``, or a model file to open in the viewer."""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from . import bundled
    bundled.use_runtime(bundled.dotnet())      # the carried runtime for AnimeStudio and the WuWa reader
    # The window reads a scan's and a job's lines as UTF-8; the console code page (cp950 here) cannot
    # even hold every character name the games use, and a scan stopped at the first one it could not.
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="strict")
    from .blender_files import load_source_modules
    load_source_modules()
    if not argv or argv[0] not in ("--run-job", "--scan", "--selftest", "--discover-wuwa-keys", "--download-wuwa-keys",
                                       "--share", "--share-github", "--view-host"):
        from .migrate import from_islandlife
        from_islandlife(lambda _line: None)      # the window's first start: the old folder's data, once
    if len(argv) == 2 and argv[0] == "--run-job":
        from .jobs import run_job
        return run_job(argv[1])
    if len(argv) == 3 and argv[0] == "--scan":
        from .jobs import run_scan
        return run_scan(argv[1], argv[2])
    if len(argv) == 2 and argv[0] in ("--discover-wuwa-keys", "--download-wuwa-keys"):
        from .jobs import run_keys
        return run_keys(argv[1], public=argv[0] == "--download-wuwa-keys")
    if len(argv) in (3, 4) and argv[0] == "--share":
        from .share import run_share
        return run_share(*argv[1:])
    if len(argv) in (2, 3) and argv[0] == "--share-github":
        from .github import run_share as run_github_share
        return run_github_share(*argv[1:])
    if argv == ["--view-host"]:
        from .view_host import run as run_view_host
        return run_view_host()
    if len(argv) == 2 and argv[0] == "--selftest":
        return selftest(argv[1])
    from .app import run
    if len(argv) == 1 and Path(argv[0]).suffix.lower() in (".glb", ".gltf", ".fbx", ".obj"):
        return run(Path(argv[0]))                 # "Open with": straight into the viewer
    return run()


def selftest(path: str) -> int:
    """Write what a packaged program can import and find to *path*; 0 when nothing is missing.

    The window has no console, so a packaged build is checked by this file
    (``build.py`` runs it): the converters' imports, every source,
    and the source files Blender is handed beside the program.
    """
    import json
    import traceback
    from pathlib import Path

    from .blender_files import missing
    report: dict = {"imports": {}, "sources": [], "blender_files_missing": []}
    for name in ("numpy", "psutil", "PySide6.QtWidgets", "PySide6.QtQuickWidgets", "PySide6.QtQuick3D",
                 # UnityPy's texture codecs, imported only when a texture is read; they load data files of
                 # their own (archspec's CPU tables) that a package must carry (§7.1455)
                 "etcpak", "astc_encoder", "texture2ddecoder", "UnityPy", "polylode.gf2.export", "polylode.gf2.batch",
                 "polylode.endfield.pipeline", "polylode.endfield.fbx_export",
                 "polylode.endfield.hscope", "polylode.endfield.body_scope", "polylode.gf2.convert",
                 "polylode.gf2.bundles", "polylode.gf2.catalog", "polylode.gf2.selection",
                 "polylode.wuthering_waves.assembly", "polylode.wuthering_waves.source_scene",
                 "polylode.wuthering_waves.extract", "polylode.wuthering_waves.keys", "polylode.converted.names", "polylode.converted.scope",
                 "polylode.interchange.glb", "polylode.interchange.fbx", "polylode.extractor.share", "polylode.extractor.github", "polylode.extractor.view_host",
                 "polylode.extractor.app"):
        try:
            __import__(name)
            report["imports"][name] = "ok"
        except Exception as failure:  # the report says what a packaged program cannot import
            report["imports"][name] = "".join(traceback.format_exception(failure, limit=2))
    try:
        from .sources import registry
        report["sources"] = sorted(registry())
    except Exception as failure:
        report["sources"] = "".join(traceback.format_exception(failure, limit=2))
    report["blender_files_missing"] = missing()
    # what a shared site serves itself (share.VENDORED): a Discord Activity can load nothing from outside it
    from .share import VENDORED, WEB
    report["web_missing"] = [name for name in VENDORED if not (WEB / name).is_file()]
    from . import bundled
    from ..core.paths import frozen
    # the packaged program carries AnimeStudio, the Wuthering Waves reader and a .NET runtime (``build.py``)
    report["bundled_missing"] = [name for name, there in (("tools/AnimeStudio/AnimeStudio.CLI.exe", bundled.animestudio()),
                                                          ("tools/WuwaReader/wuwa.exe", bundled.wuwa_reader()),
                                                          ("dotnet/dotnet.exe", bundled.dotnet())) if frozen() and there is None]
    # what the window starts its scans and jobs with must be there (Nuitka's sys.executable is not, §7.1452)
    from .jobs import program
    report["program"] = str(program())
    report["program_exists"] = program().is_file()
    # the preview needs Qt Quick 3D's QML modules beside the program, which only a load finds missing
    # GF2's bundles hold AudioClips, which UnityPy reads through fmod_toolkit's own fmod.dll (§7.1455)
    try:
        from fmod_toolkit.importer import import_pyfmodex
        import_pyfmodex()
        report["fmod"] = "ok"
    except Exception as failure:                   # noqa: BLE001 - any failure to load it is the finding
        report["fmod"] = f"{type(failure).__name__}: {failure}"
    # the platform's own plugin, not "offscreen" (which has no OpenGL): nothing here makes a window, only an
    # off-screen surface (``preview.load_check``)
    from PySide6.QtGui import QGuiApplication
    from .preview import use_opengl
    use_opengl()                     # before the application: the check draws with OpenGL, not Qt Quick's default
    QGuiApplication.instance() or QGuiApplication([])
    from .preview import load_check, load_errors, triangle_glb
    report["preview_qml_errors"] = load_errors()
    # the importer that reads .glb is a plugin of its own (``plugins/assetimporters``): only loading a file finds it
    # missing, which the window then reported as "Unknown error" (§7.1455)
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        report["preview_load"] = load_check(triangle_glb(Path(folder) / "triangle.glb"))
    if report["preview_load"]["state"] != "ready":
        report["preview_qml_errors"].append(f"a .glb does not load: {report['preview_load']['state']}")
    ok = (all(v == "ok" for v in report["imports"].values()) and isinstance(report["sources"], list)
          and not report["blender_files_missing"] and not report["bundled_missing"] and report["program_exists"]
          and not report["preview_qml_errors"] and report["fmod"] == "ok" and not report["web_missing"])
    report["ok"] = ok
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
