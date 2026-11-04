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
    #: a tool an export cannot run without: the window asks for it before a preview or an export rather
    #: than letting the job fail on it (``SourcePage.ask_tools``) — unless this program carries one
    required: bool = False
    #: the ``bundled`` function naming the tool this program carries, what an empty field means
    bundled: str = ""
    #: what the tool is and where it comes from, shown before the user is asked for it
    explain: str = ""
    #: the source's method that turns pasted text into the tool's file (``paste_<key>``): the user may paste
    #: instead of choosing a file
    pasted: bool = False


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


#: how often a job waiting for another's shared writes looks again, seconds
LOCK_EVERY = 0.5


class shared_writes:
    """``with shared_writes("gf2", log):`` — the job's writes to what every model of a game shares (its converted store,
    indexes, the files read from the install), one process at a time: exports run side by side (the window's
    "同時匯出"), and a preview converts while an export does.  A lock file of this program's own, held by the
    operating system (released when the process ends, however it ends); a job that has to wait says so once."""

    def __init__(self, name: str, log: Log | None = None):
        self.name, self.log, self.handle = name, log, None

    def __enter__(self) -> "shared_writes":
        import time
        from ..core.paths import derived_root
        folder = derived_root() / "assets_cache" / "extractor" / "locks"
        folder.mkdir(parents=True, exist_ok=True)
        self.handle = open(folder / f"{self.name}.lock", "a+b")
        told = False
        while not self._try():
            if not told and self.log is not None:
                self.log("等待另一個匯出或預覽寫完共用的轉換資料…")
                told = True
            time.sleep(LOCK_EVERY)
        return self

    def _try(self) -> bool:
        import sys
        if sys.platform == "win32":
            import msvcrt
            self.handle.seek(0)
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                return False
            return True
        import fcntl
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def __exit__(self, *_exc) -> None:
        import sys
        try:
            if sys.platform == "win32":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def blender_for(settings, formats: list[str], spec: PathSpec) -> Path | None:
    """Blender when a requested format needs it (then it must be set), else the one set or ``None``.

    ``.glb`` and FBX need no Blender; with one set, each file is read back there too.
    """
    if any(kind in NEEDS_BLENDER for kind in formats):
        return require(settings, spec)
    path = settings.tool(spec.key)
    return path if path is not None and path.is_file() else None


def write_own(formats: list[str], document: Path, folder: Path, stem: str, blender: Path | None,
              log: Log, *, export_shaders: bool = False) -> list[Path]:
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
    if "glb" in kinds:
        from ..interchange.expressions import write_sidecar
        sidecar = write_sidecar(document, Path(folder) / f"{stem}.expressions.json")
        if sidecar is not None:
            written.append(sidecar)
            log(f"{sidecar.name}：保留原始表情控制與未對應的曲線資料")
    if export_shaders:
        from .shader_export import write_bundle, write_blend
        import tempfile
        bundle = Path(folder) / f"{stem}.shaders"
        try:
            if "glb" in kinds:
                write_bundle(Path(folder) / f"{stem}.glb", bundle)
            else:
                with tempfile.TemporaryDirectory(prefix="polylode-shader-") as temporary:
                    support = Path(temporary) / "model.glb"
                    _report, problems = glb.write_checked(document, support, None)
                    if problems:
                        raise ValueError("shader model read-back failed: " + "; ".join(problems))
                    write_bundle(support, bundle)
            written.extend(sorted(path for path in bundle.rglob("*") if path.is_file()))
            if blender is not None:
                shaded = write_blend(bundle, blender, Path(folder) / f"{stem}.shaded.blend")
                written.append(shaded)
                log(f"{shaded.name}：角色著色節點與貼圖已封裝，Blender 讀回通過")
            else:
                log(f"{bundle.name}：已匯出完整著色器資料；未設定 Blender，尚未建立 .shaded.blend")
        except (OSError, ValueError) as refusal:
            raise ExtractorError(f"{stem} 角色著色匯出失敗：{refusal}") from None
    return written


def registry() -> dict[str, Source]:
    """Every source, by id; imported here so the window can list them without loading their converters."""
    from .converted_source import ConvertedSource
    from .endfield_source import EndfieldSource
    from .gf2_source import GF2Source
    from .wuwa_source import WuwaSource
    return {source.id: source for source in (ConvertedSource(), EndfieldSource(), GF2Source(), WuwaSource())}
