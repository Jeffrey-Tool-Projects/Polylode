"""The converted models already delivered: each game's ``*_Converted/covered_models.json``, copied out as they are.

These are the ``_covered`` deliveries (original clothes with a completed body,
textures packed in the ``.blend`` and embedded in the FBX), listed by the
manifest each conversion wrote.  A copy is checked against the SHA-256 the
manifest recorded; a file that does not match is not handed out.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PureWindowsPath

from ..converted.scope import EXCLUDED_WUWA
from .sources import ExtractorError, Log, Model, PathSpec, require

FOLDER = PathSpec("converted", "既有轉出物資料夾", "dir",
                  "含 Endfield_Converted／GF2_Converted／WutheringWaves_Converted 的資料夾", install=True)
#: (folder, label, where its rows are)
GAMES = (("Endfield_Converted", "終末地", "rows"), ("GF2_Converted", "GF2", "rows"),
         ("WutheringWaves_Converted", "鳴潮", "models"))


def _local(recorded: str, root: Path, folder: str) -> Path:
    """The manifest's file under *root*: manifests record absolute paths, which move with the folder."""
    parts = PureWindowsPath(recorded).parts
    if folder not in parts:
        raise ExtractorError(f"清單記的路徑不在 {folder} 底下：{recorded}")
    return root.joinpath(folder, *parts[parts.index(folder) + 1:])


def _name(folder: str, row: dict, relative: Path) -> str:
    if folder == "WutheringWaves_Converted":
        return row.get("display_name") or row["prefab"]
    if folder == "GF2_Converted":
        owners = row.get("characters") or []
        return "、".join(sorted({o["name"] for o in owners if isinstance(o, dict)})) or row["prefab"]
    return relative.parts[1] if len(relative.parts) > 2 else row["prefab"]   # <body type>/<name>/…


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ConvertedSource:
    id = "converted"
    label = "既有轉出物"
    paths = (FOLDER,)

    def scan(self, settings, log: Log) -> list[Model]:
        root = require(settings, FOLDER)
        models = []
        for folder, label, key in GAMES:
            manifest = root / folder / "covered_models.json"
            if not manifest.is_file():
                log(f"{label}：沒有 {folder}/covered_models.json，略過")
                continue
            rows = json.loads(manifest.read_text(encoding="utf-8"))[key]
            for row in rows:
                if folder == "WutheringWaves_Converted" and row["prefab"] in EXCLUDED_WUWA:
                    continue
                files = {kind: _local(row[kind], root, folder) for kind in ("blend", "fbx") if row.get(kind)}
                relative = next(iter(files.values())).relative_to(root / folder)
                models.append(Model("converted", row["prefab"], _name(folder, row, relative), label,
                                    {"folder": folder,
                                     "files": {kind: path.relative_to(root).as_posix() for kind, path in files.items()},
                                     "sha256": {kind: row.get(f"{kind}_sha256") for kind in files}}))
            log(f"{label}：{sum(1 for m in models if m.detail['folder'] == folder)} 個")
        if not models:
            raise ExtractorError(f"{root} 底下沒有任何轉出物清單")
        return models

    def export(self, model: Model, settings, output: Path, formats: list[str], log: Log) -> list[Path]:
        root = require(settings, FOLDER)
        written = []
        for kind in formats:
            relative = model.detail["files"].get(kind)
            if relative is None:
                log(f"{model.ident} 沒有 .{kind}（既有轉出物只有 {'、'.join('.' + k for k in model.detail['files'])}），略過")
                continue
            source = root / relative
            if not source.is_file():
                raise ExtractorError(f"清單記的檔案不在：{source}")
            target = Path(shutil.copy2(source, output / source.name))
            expected = model.detail["sha256"].get(kind)
            if expected and sha256(target) != expected:
                target.unlink()
                raise ExtractorError(f"{source.name} 和清單記的 SHA-256 不同，沒有交出")
            log(f"{source.name}：SHA-256 與清單相同" if expected else f"{source.name}：清單沒有記 SHA-256，未核對")
            written.append(target)
            textures = source.parent / "textures"
            if kind == "fbx" and textures.is_dir():
                shutil.copytree(textures, output / "textures", dirs_exist_ok=True)
                written.append(output / "textures")
        if not written:
            raise ExtractorError(f"{model.ident} 沒有任何要求的格式（{'、'.join(formats)}）")
        return written
