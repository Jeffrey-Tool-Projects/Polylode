"""Converted Endfield characters in the consolidated folder: ``Charas/Endfield/<character>/<model>/``.

Every converted character keeps its runtime resources under ``Charas``, one
folder per character and one per model under it (owner, §7.1247／§7.1278).
For Endfield a model folder is a complete store of its own, in the same
format and with the same relative paths as the shared store
(``polylode/endfield/store.py``): its own ``manifest.json``, the model's
items (``chara`` / ``body`` / ``body_source`` / ``reconstruction``), their
tables, every texture they name and every file their tables point at.  The
paths a certification binds are relative to the store, so a model copied
whole is certified exactly as it was.

``character.json`` names the folder's model and how its character's name was
read (:mod:`.names`).  Folders are discovered by that descriptor alone; no
character list is kept anywhere.  What is not one model's — clips, morph
tables, props, the age table, ``links.json`` — stays in the shared store.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.paths import write_json_atomic
from ..origin import bake
from . import store
from .body_archive import file_digest
from .names import Identity

GAME = "Endfield"
DESCRIPTOR = "character.json"
VERSION = 1
#: The categories whose item is one model's (its ident is the model's ident).
MODEL_CATEGORIES = store.MODEL_CATEGORIES
#: The kinds of file a table or an archive's metadata points at inside the store.
_FILE_KINDS = frozenset(("npz", "json", "png", "jpg", "md", "txt", "bin"))


class CharasError(ValueError):
    """A model folder is incomplete, inconsistent, or a copy did not arrive intact."""


@dataclass
class Scan:
    """The model folders found by their descriptors, and the folders that could not be read."""
    models: dict[str, Path] = field(default_factory=dict)
    problems: list[dict] = field(default_factory=list)

    def broken(self, ident: str) -> dict | None:
        """The problem recorded for *ident*'s folder, if there is one."""
        return next((row for row in self.problems if row["ident"] == ident), None)


def _descriptor_problem(folder: Path, base: Path, data: dict, source: str = store.SOURCE) -> str | None:
    if not folder.resolve().is_relative_to(base.resolve()):
        return "model folder leaves the Charas folder"
    if data.get("version") != VERSION or data.get("source") != source:
        return "unsupported descriptor"
    if data.get("ident") != folder.name:
        return "descriptor names another model"
    if data.get("name") != folder.parent.name:
        return "descriptor names another character folder"
    if not (folder / "manifest.json").is_file():
        return "model folder has no manifest"
    return None


def scan(base: Path, source: str = store.SOURCE) -> Scan:
    """Every ``<character>/<model>/character.json`` under *base* of game *source*; a model found twice is a problem."""
    result = Scan()
    if not base.is_dir():
        return result
    seen: dict[str, list[Path]] = {}
    for descriptor in sorted(base.glob(f"*/*/{DESCRIPTOR}")):
        folder = descriptor.parent
        try:
            data = json.loads(descriptor.read_text(encoding="utf-8"))
            problem = _descriptor_problem(folder, base, data if isinstance(data, dict) else {}, source)
        except (OSError, ValueError) as exc:
            problem = f"descriptor unreadable: {exc}"
        if problem:
            result.problems.append({"ident": folder.name, "path": str(folder), "error": problem})
            continue
        seen.setdefault(folder.name, []).append(folder)
    for ident, folders in seen.items():
        if len(folders) == 1:
            result.models[ident] = folders[0]
        else:
            result.problems.append({"ident": ident, "path": [str(f) for f in folders],
                                    "error": "the same model is in more than one character folder"})
    return result


# ---------------------------------------------------------------------------
# what one model is made of
# ---------------------------------------------------------------------------
def _index(root: Path) -> dict:
    with bake.store(root):
        return bake.manifest(refresh=True)["items"]


def _walk(value, base: Path, found: dict) -> None:
    """Collect named textures (``{"cab", "name"}``) and store files a JSON value points at (*base* resolved)."""
    if isinstance(value, dict):
        if isinstance(value.get("cab"), str) and isinstance(value.get("name"), str):
            found["named"].add(bake.named_key(store.SOURCE, value["cab"], value["name"]))
        for key, inner in value.items():
            _walk(key, base, found)
            _walk(inner, base, found)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            _walk(inner, base, found)
    elif isinstance(value, str) and "/" in value and len(value) < 512 and value.rpartition(".")[2] in _FILE_KINDS:
        path = base / value
        try:
            if path.resolve().is_relative_to(base) and path.is_file():
                found["files"].add(value)
        except (OSError, ValueError):
            return


def _content(root: Path, relative: str):
    path = root / relative
    if relative.endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            return json.loads(bytes(archive["meta"]).decode("utf-8")) if "meta" in archive else {}
    if relative.endswith(".json"):
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def own_keys(ident: str, index: dict) -> list[str]:
    """The manifest keys that are model *ident*'s own: its items and the tables named after it."""
    tables = f"table/{store.SOURCE}/"
    return sorted(key for key, entry in index.items()
                  if (key.split("/", 1)[-1] == ident and key.split("/", 1)[0] in MODEL_CATEGORIES)
                  or (key.startswith(tables) and key.endswith("/" + ident)))


def references(keys, index: dict, root: Path) -> dict:
    """``{"named": texture keys, "files": store files}`` the entries under *keys* point at."""
    found = {"named": set(), "files": set()}
    base = root.resolve()
    for key in keys:
        relative = (index.get(key) or {}).get("file")
        if relative and (root / relative).is_file():
            _walk(_content(root, relative), base, found)
    by_file = {entry["file"]: key for key, entry in index.items()
               if entry.get("kind") == "named" and entry.get("file")}
    found["named"].update(by_file[f] for f in found["files"] if f in by_file)
    return found


def closure(ident: str, root: Path, index: dict | None = None) -> tuple[dict, set]:
    """``(manifest entries, files)`` a store at *root* needs to hold model *ident* by itself."""
    index = _index(root) if index is None else index
    keys = own_keys(ident, index)
    found = references(keys, index, root)
    entries = {key: index[key] for key in keys}
    entries.update({key: index[key] for key in found["named"] if key in index})
    files = {entry["file"] for entry in entries.values() if entry.get("file")} | found["files"]
    return entries, {f for f in files if (root / f).is_file()}


# ---------------------------------------------------------------------------
# making a model folder
# ---------------------------------------------------------------------------
def folder_for(identity: Identity, base: Path) -> Path:
    """Where *identity*'s model lives under the Charas folder *base*."""
    return base / identity.folder / identity.ident


def _write_descriptor(folder: Path, identity: Identity, origin: str) -> None:
    write_json_atomic(folder / DESCRIPTOR, {"version": VERSION, "source": store.SOURCE, "ident": identity.ident,
                                            **identity.describe(), "made_from": origin}, indent=1)


def _copy_verified(source: Path, target: Path) -> dict:
    """Copy one file; ``{"source", "target"}`` SHA-256, which must agree."""
    before = file_digest(source)
    if target.is_file() and file_digest(target) == before:
        return {"source": before, "target": before, "copied": False}
    if target.exists():
        raise CharasError(f"{target} already holds other bytes; refusing to overwrite it")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    after = file_digest(target)
    if after != before or file_digest(source) != before:
        raise CharasError(f"copy of {source} did not arrive intact")
    return {"source": before, "target": after, "copied": True}


def create(identity: Identity, base: Path, shared: Path) -> tuple[Path, dict]:
    """Make *identity*'s model folder, copying the model whole out of *shared* when it is there.

    The copy is verified file by file (SHA-256 before and after) and the descriptor is written
    last, so a folder that stopped half way is never discovered.  Nothing in *shared* changes.
    """
    folder = folder_for(identity, base)
    entries, files = closure(identity.ident, shared) if shared.is_dir() else ({}, set())
    copies = {relative: _copy_verified(shared / relative, folder / relative) for relative in sorted(files)}
    folder.mkdir(parents=True, exist_ok=True)
    with bake.store(folder):
        current = bake.manifest(refresh=True)
        for key, entry in entries.items():
            if key in current["items"] and current["items"][key] != entry:
                raise CharasError(f"{folder}: {key} is already recorded differently")
        current["items"].update(entries)
        bake._save_manifest(current)
    _write_descriptor(folder, identity, "copied from the shared store" if entries else "new conversion")
    return folder, {"folder": str(folder), "entries": len(entries), "files": copies}


# ---------------------------------------------------------------------------
# leaving the shared store
# ---------------------------------------------------------------------------
def _through_link(path: Path, root: Path) -> bool:
    current = path
    while current != root and current.is_relative_to(root):
        if current.is_symlink() or current.is_junction():
            return True
        current = current.parent
    return False


def verify_copy(ident: str, shared: Path, folder: Path) -> dict:
    """Every file and entry *ident* has in *shared* is in *folder* with the same bytes; ``{file: sha}``."""
    entries, files = closure(ident, shared)
    theirs = _index(folder)
    for key, entry in entries.items():
        if theirs.get(key) != entry:
            raise CharasError(f"{folder}: {key} is missing or recorded differently")
    digests = {}
    for relative in sorted(files):
        digest = file_digest(shared / relative)
        if not (folder / relative).is_file() or file_digest(folder / relative) != digest:
            raise CharasError(f"{folder}: {relative} is missing or differs from the shared store")
        digests[relative] = digest
    return digests


def remove_shared(ident: str, shared: Path, folder: Path) -> dict:
    """Take model *ident* out of the shared store once *folder* holds it whole.

    Removes the model's own entries and files, and the textures and files it points at that
    no entry left in the shared store points at.  A path reached through a junction or link
    is not deleted (the link's target is someone's source data): the removal stops instead.
    """
    verified = verify_copy(ident, shared, folder)
    index = _index(shared)
    entries, files = closure(ident, shared, index)
    drop_keys, drop_files = _unshared(ident, index, shared, entries, files)
    linked = sorted(f for f in drop_files if _through_link(shared / f, shared))
    if linked:
        raise CharasError(f"refusing to delete through a junction or link: {linked[:5]}")
    with bake.store(shared):
        current = bake.manifest(refresh=True)
        for key in drop_keys:
            current["items"].pop(key, None)
        bake._save_manifest(current)
    for relative in sorted(drop_files):
        (shared / relative).unlink()
    _prune_empty(shared, drop_files)
    return {"verified_files": len(verified), "removed_entries": sorted(drop_keys),
            "removed_files": sorted(drop_files), "kept_shared_textures": sorted(set(entries) - drop_keys)}


def _unshared(ident: str, index: dict, root: Path, entries: dict, files: set) -> tuple[set, set]:
    """The keys and files of *ident*'s closure that nothing else left in the store points at."""
    own = set(own_keys(ident, index))
    rest = [key for key in index if key not in own and index[key].get("kind") != "named"]
    kept = references(rest, index, root)
    kept_files = kept["files"] | {index[key]["file"] for key in rest if index[key].get("file")}
    named = set(entries) - own
    drop_named = {key for key in named if key not in kept["named"]}
    kept_files |= {index[key]["file"] for key in named - drop_named if index[key].get("file")}
    return own | drop_named, {f for f in files if f not in kept_files}


def _prune_empty(root: Path, removed: set) -> None:
    for relative in sorted(removed, key=lambda f: -f.count("/")):
        parent = (root / relative).parent
        while parent != root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent


def descriptor(folder: Path) -> dict:
    """A model folder's descriptor."""
    return json.loads((Path(folder) / DESCRIPTOR).read_text(encoding="utf-8"))
