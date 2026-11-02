"""The tools the packaged extractor carries: AnimeStudio (MIT), the Wuthering Waves reader (built from
``tools/wuwa_cli`` and CUE4Parse, Apache 2.0) and a .NET runtime for both.

``build.py`` copies them beside the program: ``tools/AnimeStudio``
(the Endfield research kit's ``endfield-vfs5-1070`` build, with its
licence) and ``dotnet`` (the .NET runtime, with Microsoft's licence and
third-party notices).  In a checkout the AnimeStudio build is the one under
``tools/endfield_extract`` and there is no bundled runtime: the system's is
used.

A path the user set always wins, and a set path that is not there is an
error (``sources.require``); the bundled tool is only what an empty field
means, and the window says so.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..core.paths import frozen
from .blender_files import root

ANIMESTUDIO = "AnimeStudio.CLI.exe"


def animestudio() -> Path | None:
    """The AnimeStudio CLI this program carries (or the checkout's research-kit build), when it is there."""
    folder = root() / "tools" / "AnimeStudio"
    path = folder / ANIMESTUDIO
    return path if path.is_file() else None


def wuwa_reader() -> Path | None:
    """The Wuthering Waves reader this program carries (``tools/WuwaReader``, which ``build.py`` copies from
    ``tools/wuwa_cli/build.ps1``'s output; a checkout has one only when it was built there)."""
    path = root() / "tools" / "WuwaReader" / "wuwa.exe"
    return path if path.is_file() else None


def dotnet() -> Path | None:
    """The .NET runtime folder this program carries, when it is there (never in a checkout)."""
    folder = root() / "dotnet"
    return folder if frozen() and (folder / "dotnet.exe").is_file() else None


def use_runtime(folder: Path | None) -> None:
    """Let the .NET programs this process starts run on *folder*'s runtime, a newer major one included
    (AnimeStudio targets .NET 9; the bundled runtime is .NET 10, which the WuWa reader needs)."""
    if folder is None:
        return
    os.environ["DOTNET_ROOT"] = str(folder)
    os.environ["DOTNET_ROLL_FORWARD"] = "Major"
    os.environ["DOTNET_MULTILEVEL_LOOKUP"] = "0"
