"""Where the user's installs, Blender and the extractors are: chosen in the window, kept per user."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.paths import _platform_user_root, write_json_atomic

#: The settings file, beside the user's other data (``core.paths``): never in the program's folder.
SETTINGS = _platform_user_root() / "extractor" / "settings.json"


@dataclass
class Settings:
    """Every path the user names: game installs by source id, tools by name, the output folder."""
    installs: dict = field(default_factory=dict)
    tools: dict = field(default_factory=dict)
    output: str = ""
    #: the window's language (``i18n.LANGUAGES``); empty follows the system
    language: str = ""
    #: ``.glb`` and FBX need no Blender; ``.blend`` and the shaded companion do
    formats: list = field(default_factory=lambda: ["glb"])
    #: a character's display poses written into its ``.glb`` as animations, where the game has them
    #: (``endfield/poses``); a preview carries them too, to switch between (``preview.preview_settings``)
    poses: bool = True
    #: the character's shader data (``<model>.shaders``) and, with Blender set, a shaded ``.blend`` beside each export:
    #: only when ticked — written every time, a Blender build took minutes a model and the folder was not asked for
    export_shaders: bool = False
    #: how many models the export queue converts at once (``app.Window``); each is a process of its own, and what they
    #: share is written one at a time (``sources.shared_writes``)
    parallel_exports: int = 2

    def install(self, source: str) -> Path | None:
        value = self.installs.get(source)
        return Path(value) if value else None

    def tool(self, name: str) -> Path | None:
        value = self.tools.get(name)
        return Path(value) if value else None


def load(path: Path | None = None) -> Settings:
    """The saved settings; a missing or unreadable file gives empty ones (the window asks for each path)."""
    try:
        data = json.loads(Path(path or SETTINGS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    # every field read back, the language too: left out, the window's choice was forgotten at every start
    known = {key: data[key] for key in ("installs", "tools", "output", "language", "formats", "poses", "export_shaders",
                                       "parallel_exports") if key in data}
    return Settings(**known)


def save(settings: Settings, path: Path | None = None) -> None:
    path = Path(path or SETTINGS)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, asdict(settings), indent=1)
