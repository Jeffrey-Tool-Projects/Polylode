"""Run the game-file reader (``tools/wuwa_cli``, a CUE4Parse program) on the user's own install.

The reader is a .NET 10 program the user builds from ``tools/wuwa_cli``; it
mounts ``<install>/Client/Content`` with the user's AES keys file
(``mainKey``, ``dynamicKeys``) and writes what a mode asks for under an output
folder: ``raw-list <list>`` copies those archive files out as they are,
``mesh <package>`` dumps one skeletal mesh with its materials and textures,
``metadata <scope>`` writes each texture's colour space, ``json``/``blueprints``
read assets as JSON.  A non-zero exit stops the caller with the reader's own
last lines.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


class ReaderError(RuntimeError):
    """The reader failed; its message ends with what it printed."""


def command(reader: Path, dotnet: Path, install: Path, keys: Path, output: Path, mode: str,
            argument: str | None = None) -> list[str]:
    """The reader's command line: an apphost ``wuwa.exe`` directly, a ``wuwa.dll`` through ``dotnet``."""
    program = [str(reader)] if reader.suffix.lower() == ".exe" else [str(dotnet / "dotnet.exe"), str(reader)]
    return [*program, str(install), str(keys), str(output), mode, *([argument] if argument is not None else [])]


def run(reader: Path, dotnet: Path, install: Path, keys: Path, output: Path, mode: str,
        argument: str | None = None, log=print) -> str:
    """Run one reader mode; its output, or :class:`ReaderError`."""
    environment = dict(os.environ, DOTNET_ROOT=str(dotnet), DOTNET_CLI_TELEMETRY_OPTOUT="1",
                       DOTNET_PROCESSOR_COUNT=os.environ.get("DOTNET_PROCESSOR_COUNT", "6"))
    output.mkdir(parents=True, exist_ok=True)
    # CUE4Parse fetches its Oodle and zlib-ng libraries into the folder it runs in on first use: a folder the
    # user can write (the packaged program's own may not be), kept so they are fetched once
    from ..core.paths import derived_root
    workdir = derived_root() / "assets_cache" / "wuthering_waves" / "reader_runtime"
    workdir.mkdir(parents=True, exist_ok=True)
    log(f"讀取遊戲檔案（{mode}）…")
    done = subprocess.run(command(reader, dotnet, install, keys, output, mode, argument), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=False, env=environment, cwd=workdir,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if done.returncode != 0:
        raise ReaderError(f"讀取器 {mode} exit {done.returncode}：{(done.stdout + done.stderr)[-1500:]}")
    return done.stdout


def install_stamp(install: Path) -> list:
    """The archives' count, size and newest change: another stamp means the game was updated."""
    files = [p for p in (install / "Client" / "Content" / "Paks").rglob("*") if p.is_file()]
    return [len(files), sum(p.stat().st_size for p in files), max((p.stat().st_mtime_ns for p in files), default=0)]
