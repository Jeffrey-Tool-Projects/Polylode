"""Wuthering Waves: the install's playable characters' display models, read through the user's own game-file reader.

Scanning copies the role, model and language tables out of the install
(``wuthering_waves.scope.FILES``, with ``tools/wuwa_cli``'s ``raw-list``) and
reads the model table as JSON (``assembly.MODEL_TABLE``) into
``assets_cache/wuthering_waves/config`` — again only when the archives
changed.  It lists one row per display model the character menu puts
together (``assembly.display_models``: a primary mesh with its UI parts),
named from the game's own traditional Chinese text.  Every playable
character is listed, women and men, with no exclusion: these are the game's
own models for the user's own use (owner 2026-11-02, §7.1455), written only to
the output folder and ``assets_cache/wuthering_waves``, which the game does
not read.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..core import progress
from .sources import ExtractorError, Log, Model, PathSpec, require

INSTALL = PathSpec("wuwa", "鳴潮安裝資料夾", "dir", "遊戲資料夾或它的上一層（官方啟動器或 Steam）", install=True,
                   marker="Client/Content/Paks")
READER = PathSpec("wuwa_reader", "鳴潮讀取器（wuwa.exe 或 wuwa.dll）", "file", "留空：用內建的讀取器",
                  required=True, bundled="wuwa_reader")
DOTNET = PathSpec("dotnet", ".NET 10 執行環境資料夾", "dir", "含 dotnet.exe 的資料夾；打包版留空用內建的",
                  required=True, bundled="dotnet")
KEYS = PathSpec("wuwa_keys", "鳴潮 AES 金鑰檔", "file", "含 mainKey 與 dynamicKeys 的 JSON", required=True,
                pasted=True, explain='鳴潮的遊戲檔是加密的，讀取需要你這個用戶端版本的 AES 金鑰（mainKey：0x 加 64 位十六進位數字）。程式不附帶金鑰。可以在鳴潮頁面按「自動取得 AES 金鑰」，從本機遊戲程式尋找並驗證；部分遊戲版本可能不支援。也可按「下載公開 AES 金鑰…」，同意後從第三方下載並驗證。\n\n你可以選擇你已有的金鑰檔（含 mainKey 與 dynamicKeys 的 JSON），或直接貼上金鑰字串，程式會替你存成金鑰檔。\n\n之後把金鑰檔放在 %LOCALAPPDATA%\\Polylode、程式旁、安裝資料夾、下載、文件或桌面，程式就會自動找到。遊戲更新後金鑰可能會變，到時換上新版本的金鑰即可。')
BLENDER = PathSpec("blender", "Blender 執行檔", "file")
MODEL_TABLE_JSON = "model_config/package.json"


#: a keys file this large is not one (a JSON of two keys and a list is a few kilobytes)
KEYS_MOST = 256 * 1024


def is_keys_file(path: Path) -> bool:
    """Whether *path* is a keys file by its content: a JSON object with a hexadecimal ``mainKey`` and a
    ``dynamicKeys`` list (its name may be anything)."""
    try:
        if not path.is_file() or path.stat().st_size > KEYS_MOST:
            return False
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict) or not isinstance(data.get("dynamicKeys"), list):
        return False
    key = data.get("mainKey")
    if not isinstance(key, str):
        return False
    digits = key[2:] if key.lower().startswith("0x") else key
    return len(digits) == 64 and all(c in "0123456789abcdefABCDEF" for c in digits)


def keys_places(settings) -> list[Path]:
    """Where a keys file the user already has is looked for: beside the program, in its data folder (the
    packaged program's, which a checkout run from an IDE shares), the install folder named and the one above
    it, and the user's Downloads, Documents and Desktop."""
    from ..core.paths import _platform_user_root, derived_root
    from .blender_files import root
    places = [root(), derived_root(), _platform_user_root()]
    install = settings.install(INSTALL.key)
    if install is not None:
        places += [install, install.parent]
    home = Path.home()
    places += [home / "Downloads", home / "Documents", home / "Desktop"]
    return list(dict.fromkeys(p for p in places if p.is_dir()))


#: the .NET runtime the reader is built for
DOTNET_MAJOR = "10."


def has_runtime(folder: Path) -> bool:
    """Whether *folder* is a .NET install with a .NET 10 runtime (``shared/Microsoft.NETCore.App/10.*``)."""
    shared = folder / "shared" / "Microsoft.NETCore.App"
    return (folder / "dotnet.exe").is_file() and shared.is_dir() and any(
        p.name.startswith(DOTNET_MAJOR) for p in shared.iterdir() if p.is_dir())


def find_dotnet() -> Path | None:
    """A .NET 10 runtime installed on this computer — ``DOTNET_ROOT``, the ``dotnet`` on ``PATH``, or the
    usual ``Program Files\\dotnet`` — for a checkout, which carries none."""
    import os
    import shutil
    places = [os.environ.get("DOTNET_ROOT")]
    on_path = shutil.which("dotnet")
    places.append(str(Path(on_path).resolve().parent) if on_path else None)
    places.append(str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "dotnet"))
    return next((Path(p) for p in places if p and has_runtime(Path(p))), None)


def find_reader() -> Path | None:
    """A reader built in this checkout: ``tools/wuwa_cli/build.ps1``'s usual output, or ``dotnet build``'s own."""
    from .blender_files import root
    for path in (root() / "tools" / "WuwaReader" / "wuwa.exe",
                 root() / "tools" / "wuwa_cli" / "bin" / "Release" / "net10.0" / "wuwa.exe"):
        if path.is_file():
            return path
    return None


def keys_from_text(text: str) -> dict:
    """What the user pasted as a keys file's content: the whole JSON, or the main key alone (``0x`` and 64
    hexadecimal digits, the ``0x`` optional); ``ValueError`` saying what is wrong otherwise."""
    text = text.strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            raise ValueError("貼上的 JSON 讀不出來") from None
    else:
        data = {"mainKey": text, "dynamicKeys": []}
    if not isinstance(data, dict) or not isinstance(data.get("dynamicKeys", []), list):
        raise ValueError("金鑰檔要有 mainKey 與 dynamicKeys（清單）")
    key = str(data.get("mainKey", ""))
    digits = key[2:] if key.lower().startswith("0x") else key
    if len(digits) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digits):
        raise ValueError("mainKey 要是 0x 加 64 位十六進位數字")
    return {"mainKey": "0x" + digits.upper(), "dynamicKeys": data.get("dynamicKeys", [])}


def paste_keys(text: str) -> Path:
    """Write what the user pasted (``keys_from_text``) as ``wuwa_keys.json`` in the program's data folder,
    one of the places ``find_keys`` looks; the file."""
    from ..core.paths import _platform_user_root
    data = keys_from_text(text)
    path = _platform_user_root() / "wuwa_keys.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def find_keys(settings) -> Path | None:
    """A keys file the user already has (``is_keys_file``) in one of ``keys_places``, the files directly in
    each; the newest when there are several.  This search only finds existing files; automatic discovery is a separate action."""
    found = []
    for place in keys_places(settings):
        try:
            found += [p for p in place.glob("*.json") if is_keys_file(p)]
        except OSError:
            continue
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def config_root() -> Path:
    from ..core.paths import derived_root
    return derived_root() / "assets_cache" / "wuthering_waves" / "config"


#: where a .NET runtime is downloaded, for the message that says one is needed
DOTNET_DOWNLOAD = "https://dotnet.microsoft.com/download/dotnet/10.0"


def runtime_for(reader: Path, dotnet: Path) -> None:
    """Say plainly when the reader cannot be started for want of a runtime: a ``wuwa.dll`` runs through *dotnet*, which
    must hold .NET 10 (a folder named for it that was not one — the game's own — failed as a bare WinError 2)."""
    if reader.suffix.lower() == ".exe" or has_runtime(dotnet):
        return
    found = find_dotnet()
    hint = (f"這台電腦的 {found} 有 .NET 10：把這個欄位清空就會用它。" if found else
            f"這台電腦沒有 .NET 10：到 {DOTNET_DOWNLOAD} 安裝 .NET 10 Runtime（x64），或改用內建執行環境的打包版。")
    raise ExtractorError(f"「{DOTNET.label}」不是 .NET 10 執行環境（要有 dotnet.exe 與 shared\\Microsoft.NETCore.App\\10.*）："
                         f"{dotnet}。{hint}")


def _reader(settings, mode: str, output: Path, argument: str, log: Log) -> None:
    from ..wuthering_waves import extract
    try:
        from . import bundled
        reader = require(settings, READER, bundled.wuwa_reader() or find_reader())
        dotnet = require(settings, DOTNET, bundled.dotnet() or find_dotnet())
        runtime_for(reader, dotnet)
        extract.run(reader, dotnet, require(settings, INSTALL), require(settings, KEYS), output, mode, argument, log)
    except extract.ReaderError as failure:
        raise ExtractorError(str(failure)) from None


def extract_config(settings, log: Log) -> Path:
    """The tables a scan reads, copied out of this install; copied again when its archives changed."""
    from ..wuthering_waves import assembly, extract, scope
    install = require(settings, INSTALL)
    root = config_root()
    stamp_path = root / "install.json"
    stamp = {"install": str(install), "archives": extract.install_stamp(install)}
    wanted = [*scope.FILES, MODEL_TABLE_JSON]
    if (stamp_path.is_file() and json.loads(stamp_path.read_text(encoding="utf-8")) == stamp
            and all((root / name).is_file() for name in wanted)):
        return root
    root.mkdir(parents=True, exist_ok=True)
    stamp_path.unlink(missing_ok=True)
    listing = root / "files.txt"
    listing.write_text("\n".join(scope.FILES) + "\n", encoding="utf-8")
    progress.report(0, 2, "從安裝讀角色表")
    _reader(settings, "raw-list", root, str(listing), log)
    progress.report(1, 2, "從安裝讀角色表")
    _reader(settings, "json", (root / MODEL_TABLE_JSON).parent, assembly.MODEL_TABLE, log)
    missing = [name for name in wanted if not (root / name).is_file()]
    if missing:
        raise ExtractorError(f"讀取器沒有寫出：{', '.join(missing)}")
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
    return root


def display_models(root: Path, log: Log) -> list[dict]:
    """The menu's display models of this extraction (``assembly.display_models``)."""
    from ..wuthering_waves import assembly, scope
    found = scope.read_scope(root)
    if found["unresolved"]:
        log(f"{len(found['unresolved'])} 個衣裝的模型編號在預載表裡沒有：{found['unresolved'][:5]}")
    table = json.loads((root / MODEL_TABLE_JSON).read_text(encoding="utf-8-sig"))[0]["Rows"]
    return assembly.display_models(found, assembly.ui_assemblies(found, table))


def _label(keys: list[str], root: Path) -> str:
    """The game's own text for *keys*; a key the table does not hold is shown as itself, never translated."""
    from ..converted import names, store
    shown = []
    for key in dict.fromkeys(keys):
        try:
            shown.append(names.wuthering_name(root, [key], extraction=root)[0])
        except store.WarehouseError:
            shown.append(key)
    return "、".join(shown)


class WuwaSource:
    id = "wuwa"
    label = "鳴潮"
    paths = (INSTALL, READER, DOTNET, KEYS)

    def find_tool(self, key: str, settings) -> Path | None:
        """A tool the user already has on this computer, found before they are asked for it."""
        if key == KEYS.key:
            return find_keys(settings)
        if key == DOTNET.key:
            return find_dotnet()
        if key == READER.key:
            return find_reader()
        return None

    def tool_problem(self, key: str, path: Path) -> str | None:
        """Why *path* cannot be the tool *key*, or None."""
        if key == KEYS.key and not is_keys_file(path):
            return "這不是金鑰檔：需要含 mainKey（0x 加 64 位十六進位）與 dynamicKeys 的 JSON"
        return None

    def paste_wuwa_keys(self, text: str) -> Path:
        return paste_keys(text)

    def scan(self, settings, log: Log) -> list[Model]:
        root = extract_config(settings, log)
        models = []
        for row in display_models(root, log):
            owners = row["owners"]
            models.append(Model("wuwa", row["ident"], _label([o["role_name_key"] for o in owners], root),
                                _label([o["outfit_name_key"] for o in owners], root),
                                {"parts": row["parts"], "model_ids": row["model_ids"]}))
        log(f"可玩角色的顯示模型 {len(models)} 個")
        return models

    def _poses(self, settings, package: str, case: Path, folder: Path, log: Log) -> int:
        return _poses_of(settings, package, case, folder, log)

    def export(self, model: Model, settings, output: Path, formats: list[str], log: Log) -> list[Path]:
        """Dump each part with the reader, assemble the display model, have Blender build and read it back."""
        from .sources import shared_writes
        # the tables, the parts (shared between models) and their materials go into folders every model reads: one
        # job at a time; assembling the model and what follows run beside the others
        with shared_writes("wuwa", log):
            root, row, sources = self._dumped(model, settings, log)
        return self._assembled(model, settings, output, formats, log, root, row, sources)

    def _dumped(self, model: Model, settings, log: Log):
        root = extract_config(settings, log)
        row = next((r for r in display_models(root, log) if r["ident"] == model.ident), None)
        if row is None:
            raise ExtractorError(f"這個安裝的角色選單沒有 {model.ident}")
        sources = root.parent / "source"
        stamp = (root / "install.json").read_text(encoding="utf-8")       # the archives the tables were read from
        dumped = []
        for number, part in enumerate(row["parts"]):
            progress.report(number, len(row["parts"]), "讀取部件網格")
            ident = part["package"].rsplit("/", 1)[-1]
            folder = sources / ident
            receipt, dumped_from = folder / "export.json", folder / "install.json"
            # a dump is reused only from the same archives: after a game update the mesh is read again
            if not (receipt.is_file() and dumped_from.is_file() and dumped_from.read_text(encoding="utf-8") == stamp
                    and all(item.get("Success") for item in json.loads(receipt.read_text(encoding="utf-8")))):
                shutil.rmtree(folder, ignore_errors=True)
                _reader(settings, "mesh", folder, part["package"] + ".uasset", log)
                dumped_from.write_text(stamp, encoding="utf-8")
            dumped.append({"ident": ident, "package": part["package"] + ".uasset"})
        listing = sources / f"{model.ident}.metadata.json"
        listing.write_text(json.dumps({"models": dumped}), encoding="utf-8")
        progress.step("讀取材質")
        _reader(settings, "metadata", sources, str(listing), log)
        return root, row, sources

    def _assembled(self, model: Model, settings, output: Path, formats: list[str], log: Log, root, row, sources):
        from ..wuthering_waves.source_scene import assemble
        from .sources import blender_for, write_own
        blender = blender_for(settings, formats, BLENDER)
        log("組合顯示模型（主網格、子網格、插槽裝飾）…")
        progress.step("組合顯示模型")
        try:
            case = assemble(row, sources, root.parent / "cases")
        except ValueError as refusal:
            raise ExtractorError(f"{model.ident} 組合失敗：{refusal}") from None
        # a masked material's coverage as the game's own shader gives it (shader_bake), for every format
        from ..wuthering_waves.shader_bake import bake_case
        from .sources import shared_writes
        with shared_writes("wuwa", log):                 # GameFiles fills the shader cache every model reads
            bake_case(case, GameFiles(settings, log), log, progress.step)
        # the fur shader's parallax depth as layers of strands (fur_shells), after the coverage it stands on
        from ..wuthering_waves.fur_shells import apply as fur_layers
        progress.step("把毛髮視差做成毛絲層")
        fur_layers(case, log)
        if settings.poses and ("glb" in formats or settings.export_shaders):
            progress.step("寫入姿勢")
            self._poses(settings, row["parts"][0]["package"], case, sources / f"{model.ident}.anim", log)
        written = []
        if "blend" in formats:
            log(f"以 Blender 建 .blend 並讀回 {model.ident}（每個模型幾分鐘）…")
            for number, mode in enumerate(("build", "check")):
                progress.report(number, 2, "以 Blender 建 .blend 並讀回")
                blender_run(blender, mode, case, model.ident)
            written.append(Path(shutil.copy2(case / f"{model.ident}.blend", output / f"{model.ident}.blend")))
        return written + write_own(formats, case / "source_document.json", output, model.ident, blender, log,
                                   export_shaders=settings.export_shaders)


def _poses_of(settings, package: str, case: Path, folder: Path, log: Log) -> int:
    """Decode the model's display poses and add them to *case*."""
    from ..wuthering_waves import poses
    ui = poses.ui_folder(package)
    if not ui:
        return poses.add(case, [], package, log)
    from .sources import shared_writes
    with shared_writes("wuwa", log):                     # the install's file list, written once a game version
        index = archive_index(settings, log)
    listed = [line[:-len(".uasset")] for line in index.read_text(encoding="utf-8").splitlines()
              if line.startswith(ui) and line.endswith(".uasset") and "/" not in line[len(ui):]]
    if not listed:
        return poses.add(case, [], package, log)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "list.txt").write_text("\n".join(listed) + "\n", encoding="utf-8")
    try:
        _reader(settings, "anim", folder, str(folder / "list.txt"), log)
    except ExtractorError as failure:
        if "Unknown mode" in str(failure):
            raise ExtractorError("讀取器版本太舊，無法解碼動作。請更新或選擇新版讀取器；若只要靜態模型，請關閉姿勢。") from None
        raise
    decoder = None
    if any(folder.glob("*.acl")):
        # the game's ACL buffers, decoded with the library the bundled AnimeStudio carries (endfield.acl)
        from ..endfield import acl
        from . import bundled
        from .endfield_source import EXPORTER
        try:
            decoder = acl.Decoder(require(settings, EXPORTER, bundled.animestudio()))
        except (acl.AclError, ExtractorError) as missing:
            raise ExtractorError(f"找不到動作解碼需要的 ACL 程式庫（{missing}）。請設定含解碼器的 AnimeStudio；若只要靜態模型，請關閉姿勢。") from None
    indices = [k for k in range(len(listed)) if (folder / f"{k}.json").is_file()]
    if any("curves" not in json.loads((folder / f"{k}.json").read_text(encoding="utf-8")) for k in indices):
        raise ExtractorError("讀取器版本太舊，沒有導出表情曲線。請更新或選擇新版讀取器。")
    records = [poses.read(folder, k, decoder) for k in indices]
    return poses.add(case, records, package, log)


def archive_index(settings, log: Log) -> Path:
    """Every file in the install's archives (the reader's ``index``), listed once per version of the archives."""
    from ..wuthering_waves import extract
    root = config_root().parent / "index"
    stamp = json.dumps(extract.install_stamp(require(settings, INSTALL)))
    marker, listing = root / "install.json", root / "files.txt"
    if not (listing.is_file() and marker.is_file() and marker.read_text(encoding="utf-8") == stamp):
        shutil.rmtree(root, ignore_errors=True)
        _reader(settings, "index", root, None, log)
        marker.write_text(stamp, encoding="utf-8")
    return listing


class GameFiles:
    """The game's own files a shader bake reads (``shader_bake.bake_case``), copied out of the install once
    and kept by the archives' stamp: the SM5 shader archive, material packages and their JSON."""

    ARCHIVE = "Client/Content/ShaderArchive-Client_Chunk-PCD3D_SM5.ushaderbytecode"

    def __init__(self, settings, log: Log):
        from ..wuthering_waves import extract
        self.settings, self.log = settings, log
        self.root = config_root().parent / "shaders"
        stamp = json.dumps(extract.install_stamp(require(settings, INSTALL)))
        marker = self.root / "install.json"
        if marker.is_file() and marker.read_text(encoding="utf-8") != stamp:
            shutil.rmtree(self.root, ignore_errors=True)    # the game changed: its shaders may have
        self.root.mkdir(parents=True, exist_ok=True)
        marker.write_text(stamp, encoding="utf-8")
        self.cache = self.root / "coverage"

    def _raw(self, paths: list[str]) -> None:
        listing = self.root / "raw_list.txt"
        listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
        _reader(self.settings, "raw-list", self.root, str(listing), self.log)

    def archive(self) -> Path:
        path = self.root / self.ARCHIVE
        if not path.is_file():
            self.log("從安裝取出 shader 封存檔（第一次，約 1.2 GB）…")
            self._raw([self.ARCHIVE])
        return path

    def package(self, package: str) -> bytes:
        path = self.root / (package + ".uexp")
        if not path.is_file():
            self._raw([package + ".uasset", package + ".uexp"])
        return path.read_bytes()

    def json(self, package: str) -> dict:
        path = self.root / "json" / (package + ".json")
        if not path.is_file():
            folder = self.root / "json_out"
            shutil.rmtree(folder, ignore_errors=True)
            _reader(self.settings, "json", folder, package + ".uasset", self.log)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(folder / "package.json"), path)
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        return next(e for e in items if isinstance(e, dict) and e.get("Type", "").startswith(("Material", "MaterialParameterCollection")))

    def root_of(self, package: str):
        """The root material's ``CachedExpressionData`` above *package*, and its parameter collections."""
        for _ in range(16):
            item = self.json(package)
            if item.get("Type") == "Material":
                data = item.get("CachedExpressionData") or item.get("Properties", {}).get("CachedExpressionData") or {}
                collections = []
                for info in data.get("ParameterCollectionInfos") or []:
                    path = (info.get("ParameterCollection") or {}).get("ObjectPath", "").rsplit(".", 1)[0]
                    collections.append(self.json(path).get("Properties", {}) if path else {})
                return data, collections
            parent = (item.get("Properties", {}).get("Parent") or {}).get("ObjectPath")
            if not parent:
                break
            package = parent.rsplit(".", 1)[0]
        raise ExtractorError(f"{package}: no root material above it")


def blender_run(blender: Path, mode: str, case: Path, ident: str) -> None:
    """``tools/wuwa_source_blender.py`` in *mode*; its last lines when it fails."""
    from .blender_files import root
    done = subprocess.run([str(blender), "--background", "--factory-startup", "--python-exit-code", "1", "--python",
                           str(root() / "tools" / "wuwa_source_blender.py"), "--", mode, str(case), ident],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if done.returncode != 0:
        raise ExtractorError(f"Blender {mode} exit {done.returncode}：{(done.stdout + done.stderr)[-1500:]}")
