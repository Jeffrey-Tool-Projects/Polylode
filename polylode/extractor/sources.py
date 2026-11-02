"""What every source answers: which paths it needs, which models an install has, how one is exported."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .settings import Settings

#: ``log(message)``: a line for the window's log (and the job's own).
Log = Callable[[str], None]


class ExtractorError(RuntimeError):
    """A path is missing, a tool refused, or a model could not be exported: said, never worked round."""


@dataclass(frozen=True)
class PathSpec:
    """One path the user names for a source: an install folder, a program or a data folder."""
    key: str
    label: str
    kind: str          # "dir" | "file"
    hint: str = ""
    install: bool = False   # the game's install folder itself (``Settings.installs``), else a tool (``Settings.tools``)
    #: an install's data, relative to the folder it is found from (``locate.game_root``): the folder named may be above or inside it
    marker: str = ""
    #: a tool an export cannot run without and this program does not carry: the window asks for it
    #: before a preview or an export rather than letting the job fail on it (``SourcePage.ask_tools``)
    required: bool = False


@dataclass(frozen=True)
class Model:
    """A model an install has that this project may take, as the window lists it."""
    source: str
    ident: str
    character: str
    outfit: str = ""
    detail: dict = field(default_factory=dict)


class Source(Protocol):
    id: str
    label: str
    paths: tuple[PathSpec, ...]

    def scan(self, settings: Settings, log: Log) -> list[Model]: ...

    def export(self, model: Model, settings: Settings, output: Path, formats: list[str], log: Log) -> list[Path]: ...


def require(settings: Settings, spec: PathSpec, default: Path | None = None) -> Path:
    """The path the user named for *spec*, which must exist; otherwise say which and where to set it.

    *default* is what an empty field means (a tool this program carries, ``bundled``); a field that names
    a path that is not there is still an error, never answered with the default.
    """
    path = settings.install(spec.key) if spec.install else settings.tool(spec.key)
    if path is None and default is not None:
        return default
    if path is None:
        raise ExtractorError(f"「{spec.label}」還沒有設定")
    if (spec.kind == "dir" and not path.is_dir()) or (spec.kind == "file" and not path.is_file()):
        raise ExtractorError(f"「{spec.label}」不存在：{path}")
    if spec.marker:
        from .locate import candidates
        found = candidates(path, spec.marker)
        if len(found) != 1:
            several = "；有好幾份：" + "、".join(map(str, found)) if found else ""
            raise ExtractorError(f"「{spec.label}」裡找不到唯一一份 {spec.marker}：{path}{several}")
        return found[0]
    return path


#: Formats only Blender writes; ``.glb`` and FBX are this program's own (``interchange``).
NEEDS_BLENDER = ("blend",)
OWN_FORMATS = ("glb", "fbx")


def blender_for(settings, formats: list[str], spec: PathSpec) -> Path | None:
    """Blender when a requested format needs it (then it must be set), else the one set or ``None``.

    ``.glb`` and FBX need no Blender; with one set, each file is read back there too.
    """
    if any(kind in NEEDS_BLENDER for kind in formats):
        return require(settings, spec)
    path = settings.tool(spec.key)
    return path if path is not None and path.is_file() else None


def write_own(formats: list[str], document: Path, folder: Path, stem: str, blender: Path | None,
              log: Log) -> list[Path]:
    """The requested ``.glb``／FBX of the interchange scene *document*, written as ``folder/stem.<kind>`` and
    read back; a failed read-back stops the job, and a file Blender did not read says so.

    FBX is always read back by the project's own node reader (``endfield/fbxfile``) as well.
    """
    from ..core import progress
    from ..interchange import fbx, glb
    written = []
    kinds = [kind for kind in OWN_FORMATS if kind in formats]
    for number, kind in enumerate(kinds):
        progress.report(number, len(kinds), f"寫出並讀回 .{kind}")
        module = glb if kind == "glb" else fbx
        path = Path(folder) / f"{stem}.{kind}"
        try:
            _report, problems = module.write_checked(document, path, blender)
        except (glb.GlbError, fbx.FbxWriteError) as refusal:
            raise ExtractorError(f"{stem}.{kind}：{refusal}") from None
        if problems:
            raise ExtractorError(f"{stem}.{kind} 讀回沒有通過：{'; '.join(problems)}")
        checked = "Blender 讀回通過" if blender else "沒有指定 Blender，未經 Blender 讀回"
        log(f"{stem}.{kind}：" + (f"檔內骨頭與綁定矩陣核對通過，{checked}" if kind == "fbx" else checked))
        written.append(path)
    return written


def registry() -> dict[str, Source]:
    """Every source, by id; imported here so the window can list them without loading their converters."""
    from .converted_source import ConvertedSource
    from .endfield_source import EndfieldSource
    from .gf2_source import GF2Source
    from .wuwa_source import WuwaSource
    return {source.id: source for source in (ConvertedSource(), EndfieldSource(), GF2Source(), WuwaSource())}
