"""Build the character model extractor (``polylode.extractor``) as a Windows program folder.

    python build.py              # compile, copy Blender's source files, run --selftest
    python build.py --check-only # only copy the source files and run --selftest again
    python build.py --jobs 1     # one C compiler: the compile cache keeps what earlier runs finished

Carried beside the program: AnimeStudio (``tools/AnimeStudio``, MIT, with its licence) and the .NET 10
runtime (``dotnet``, from ``--dotnet`` or ``DOTNET_ROOT``, with Microsoft's licence and notices), which
AnimeStudio and the WuWa reader run on (``polylode/extractor/bundled.py``).

1. **compile** — Nuitka standalone (a folder, as ``build.py`` does for the
   game) from ``main.py``: ``dist/Polylode/Polylode.exe``.
   The same program is the window, a scan (``--scan``) and an export job
   (``--run-job``): the window starts itself for each.
2. **Blender's files** — Blender runs the converters' workers as ``.py`` files
   and the GF2 export signs its result with their bytes
   (``extractor/blender_files.py``).  The package's source and
   ``tools/ef_fbx_blender.py`` are copied beside the program at the same
   relative paths.
3. **run it** — ``Polylode.exe --selftest <json>`` imports the
   converters, lists the sources and checks those files.  Exit 1 if any step
   fails.

The folder holds the converters only: nothing converted from a game is in it
(``CLAUDE.md`` 四之一).  Each user points it at their own installs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "Polylode"
OUT = ROOT / "build"
DIST = OUT / f"{Path('main.py').stem}.dist"
FINAL = ROOT / "dist" / NAME
EXE = FINAL / f"{NAME}.exe"


#: the FMOD library fmod_toolkit loads from its own folder
FMOD_DLL = Path(__import__("fmod_toolkit").__file__).with_name("libfmod") / "Windows" / "x64" / "fmod.dll"

def copy_source_modules(folder: Path) -> None:
    """The third-party modules left as source (``blender_files.SOURCE_MODULES``), where the program loads them."""
    import importlib.util
    from polylode.extractor.blender_files import SOURCE_MODULES
    for name in SOURCE_MODULES:
        source = Path(importlib.util.find_spec(name).origin)
        target = folder.joinpath(*name.split(".")).with_suffix(".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def nuitka_command(jobs: int = 8) -> list[str]:
    from polylode.extractor.blender_files import SOURCE_MODULES
    return [
        sys.executable, "-m", "nuitka", "--standalone", "--assume-yes-for-downloads",
        f"--output-dir={OUT}", f"--output-filename={NAME}.exe",
        # the window shows no console; a scan or job started with pipes still writes to them
        "--windows-console-mode=attach",
        "--enable-plugin=pyside6",
        # the preview's Qt Quick 3D (RuntimeLoader, OrbitCameraController) is QML: its modules and plugins
        # RuntimeLoader reads .glb through the assimp importer, a plugin of its own (``plugins/assetimporters``):
        # without it every preview said only "Unknown error" (§7.1455)
        "--include-qt-plugins=qml,assetimporters", "--include-module=PySide6.QtQuick3D",
        # UnityPy reads an AudioClip through fmod_toolkit, which loads its own fmod.dll by path and imports
        # pyfmodex inside a function: Nuitka followed neither, and a GF2 conversion stopped on the DLL (§7.1455)
        "--include-package=fmod_toolkit", "--include-package=pyfmodex",
        # etcpak and astc_encoder (UnityPy's texture codecs) read archspec's CPU tables when imported
        "--include-package-data=archspec",
        f"--include-data-files={FMOD_DLL}=fmod_toolkit/libfmod/Windows/x64/fmod.dll",
        f"--windows-icon-from-ico={ROOT / 'polylode/extractor/icon.ico'}",
        "--include-package-data=polylode.extractor",
        "--company-name=Jeffrey Chen", "--product-name=Polylode",
        # Every C compiler runs inside one memory cap (tools/guard.ps1): at the default parallelism they ran
        # out of heap (C1060), and Scons, unable to decode the localized message, reported a UnicodeDecodeError.
        f"--jobs={jobs}",
        "--file-version=0.1.0.0", "--product-version=0.1.0.0",
        # Only the extractor's own package whole: the converters come in by the imports it (and they)
        # make, lazy ones included.  Including polylode.gf2 / polylode.endfield whole compiled ~716
        # modules — reconstruction, H and the game's figure code with them — against 140 the tool loads.
        "--include-package=polylode.extractor",
        "--include-package=UnityPy", "--include-package-data=UnityPy",
        f"--include-data-files={ROOT / 'polylode/gf2/source_dependencies.json'}=polylode/gf2/source_dependencies.json",
        # endfield.rigscene imports its shading module, which imports PyOpenGL at the top
        "--nofollow-import-to=pytest",
        # Reached only through lazy imports in game and body-building code the tool never runs (measured:
        # none is in sys.modules after the scans and exports); scipy.stats alone was a 24 MB C file, and eight
        # compilers on files that size overran a 10 GB cap.  A path that did reach one stops with
        # ModuleNotFoundError rather than doing something else.
        *(f"--nofollow-import-to={name}" for name in (
            "scipy.stats", "scipy.integrate", "scipy.optimize", "scipy.ndimage", "scipy.interpolate",
            "scipy.signal", "polylode.game", "polylode.ui", "polylode.endfield.seams")),
        # UnityPy's generated class table is a 31 MB C file that cl cannot optimise within 9 GB even alone;
        # it ships as its .py and the program loads it from there (blender_files.SOURCE_MODULES).
        *(f"--nofollow-import-to={name}" for name in SOURCE_MODULES),
        "--no-deployment-flag=excluded-module-usage",
        # PyOpenGL's Cython accelerator fails to initialise once compiled (KeyError '__reduce_cython__');
        # without it PyOpenGL uses its pure-Python paths, which is all endfield.shading's constants need.
        "--nofollow-import-to=OpenGL_accelerate",
        # scipy.spatial.transform imports its array-API backends by name at run time
        "--include-package=scipy._external.array_api_compat",
        "--remove-output",
        str(ROOT / "main.py"),
    ]


def copy_sources(folder: Path) -> int:
    """The package's ``.py`` and data files and the Blender scripts in ``tools/``, beside the program."""
    from polylode.extractor.blender_files import REQUIRED
    count = 0
    for path in (ROOT / "polylode").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc",):
            target = folder / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            count += 1
    writers = [rel for rel in REQUIRED if rel.startswith("tools/")]
    for rel in writers:
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, folder / rel)
    return count + len(writers)


def selftest() -> bool:
    report = FINAL / "selftest.json"
    report.unlink(missing_ok=True)
    code = subprocess.run([str(EXE), "--selftest", str(report)], check=False).returncode
    if not report.is_file():
        print(f"  selftest wrote nothing (exit {code})")
        return False
    data = json.loads(report.read_text(encoding="utf-8"))
    print(json.dumps(data, ensure_ascii=False, indent=1))
    return code == 0 and data.get("ok") is True


#: The AnimeStudio build the Endfield source runs (MIT; its licence is in the research kit's source tree).
ANIMESTUDIO = ROOT / "tools/AnimeStudio"
ANIMESTUDIO_LICENCE = ANIMESTUDIO / "LICENSE"


def copy_animestudio(folder: Path) -> int:
    """AnimeStudio beside the program (``tools/AnimeStudio``), without debug symbols, with its licence."""
    target = folder / "tools" / "AnimeStudio"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ANIMESTUDIO, target, ignore=shutil.ignore_patterns("*.pdb"))
    shutil.copy2(ANIMESTUDIO_LICENCE, target / "LICENSE")
    return sum(1 for p in target.rglob("*") if p.is_file())


def copy_wuwa_reader(folder: Path, source: Path) -> int:
    """The Wuthering Waves reader (``tools/wuwa_cli/build.ps1``'s output, with CUE4Parse's licence) beside the
    program as ``tools/WuwaReader``; how many files."""
    source = Path(source)
    if not (source / "wuwa.exe").is_file() or not (source / "CUE4Parse-LICENSE").is_file():
        raise SystemExit(f"no Wuthering Waves reader in {source} (build it with tools/wuwa_cli/build.ps1, "
                         "or give --wuwa-reader <its folder>)")
    target = folder / "tools" / "WuwaReader"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("*.pdb"))
    return sum(1 for p in target.rglob("*") if p.is_file())


def copy_dotnet(folder: Path, source: Path) -> str:
    """The .NET runtime (``dotnet.exe``, ``host``, ``shared/Microsoft.NETCore.App/<version>``, licence and
    notices) beside the program as ``dotnet``; its version.  The runtime only, no SDK."""
    versions = sorted((source / "shared" / "Microsoft.NETCore.App").glob("10.*"))
    if not (source / "dotnet.exe").is_file() or not versions:
        raise SystemExit(f"no .NET 10 runtime in {source} (give --dotnet <a dotnet folder with shared/Microsoft.NETCore.App/10.x>)")
    target = folder / "dotnet"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in ("dotnet.exe", "LICENSE.txt", "ThirdPartyNotices.txt"):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    shutil.copytree(source / "host", target / "host")
    shutil.copytree(versions[-1], target / "shared" / "Microsoft.NETCore.App" / versions[-1].name)
    return versions[-1].name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dotnet", type=Path, default=os.environ.get("DOTNET_ROOT"),
                        help="a dotnet folder whose .NET 10 runtime the program carries (default: DOTNET_ROOT)")
    parser.add_argument("--wuwa-reader", type=Path, default=ROOT / "tools" / "WuwaReader",
                        help="the Wuthering Waves reader the program carries (tools/wuwa_cli/build.ps1's -Output)")
    parser.add_argument("--jobs", type=int, default=8,
                        help="C compilers at once (1 lets the largest generated file have the whole memory cap)")
    args = parser.parse_args()
    if not args.check_only:
        started = time.perf_counter()
        code = subprocess.run(nuitka_command(args.jobs), cwd=ROOT, check=False).returncode
        print(f"Nuitka exit {code} in {(time.perf_counter() - started) / 60:.1f} min")
        if code != 0:
            return 1
        if FINAL.exists():
            shutil.rmtree(FINAL)
        FINAL.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(DIST), str(FINAL))
    if not EXE.is_file():
        print(f"no program at {EXE}")
        return 1
    print(f"copied {copy_sources(FINAL)} source files for Blender")
    copy_source_modules(FINAL)
    print(f"AnimeStudio: {copy_animestudio(FINAL)} files")
    print(f"Wuthering Waves reader: {copy_wuwa_reader(FINAL, args.wuwa_reader)} files")
    # Qt's software OpenGL beside the program: the preview draws on a machine without a GPU driver (and the
    # selftest's off-screen load on a CI runner) with QT_OPENGL=software
    import PySide6
    shutil.copy2(Path(PySide6.__file__).with_name("opengl32sw.dll"), FINAL / "opengl32sw.dll")
    if args.dotnet is None:
        print("no --dotnet and no DOTNET_ROOT: the program would carry no .NET runtime")
        return 1
    print(f".NET runtime {copy_dotnet(FINAL, Path(args.dotnet))}")
    if not selftest():
        return 1
    print(f"OK: {EXE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
