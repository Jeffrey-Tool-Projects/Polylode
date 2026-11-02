"""Where converted Endfield content lives: one store per model under ``Charas``, and a shared store.

The archives are ``origin.bake``'s format, written and read by its
functions, but in roots of their own with manifests of their own
(``bake.store``).  Koikatsu's re-bake selectors and ``--prune`` walk
``assets_baked`` and judge what they find by Koikatsu's lists; none of that
applies here, and folders of their own are also what the release audit and
``.gitignore`` name.

* **A model's store** — ``Charas/Endfield/<character>/<model>/`` (:mod:`.charas`):
  the model's items (:data:`MODEL_CATEGORIES`), their tables and every texture
  and file they point at.  Every converted character lives here (owner,
  §7.1247／§7.1278); the folders are found by their descriptors, never listed.
* **The shared store** — :func:`root`, ``derived_root()/Endfield``: what is
  not one model's (clips, morph tables, props, the age table, ``links.json``),
  and the models converted before the folder existed until they are moved.

Reading a model asks its ``Charas`` folder first and the shared store only
when it has none (:func:`home`); texture keys name the game's own files, so a
texture is read from the first store that has it.  Writing a model's item
always goes to its ``Charas`` folder, which is made — the model copied over
whole when the shared store has it — the first time (:func:`ensure_home`).

:data:`DIR_ENV` pins **one** folder as the whole store (tests, the candidate
stores of the rebuild tools), as it did before ``Charas``; with
:data:`CHARAS_ENV` also set, it is the shared store and that one is ``Charas/Endfield``.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path

from ..core.paths import derived_root
from ..origin import bake
from ..origin import colorspace as CS

DIR_ENV = "ISLANDLIFE_ENDFIELD_DIR"
#: Points ``Charas/Endfield`` somewhere else; with :data:`DIR_ENV` alone there is no ``Charas`` lookup.
CHARAS_ENV = "ISLANDLIFE_ENDFIELD_CHARAS_DIR"
#: ``source`` of an item and of a texture in the store's named textures.
SOURCE = "endfield"
#: The categories whose item is one model's own (its ident is the model's).
MODEL_CATEGORIES = ("chara", "body", "body_source", "reconstruction")

#: ``only``: one folder pinned as the whole store; ``target``: where a model being converted writes.
_local = threading.local()
_made = threading.Lock()
_scanned: dict = {}


def _pinned() -> Path | None:
    pinned = getattr(_local, "only", None)
    if pinned is not None:
        return pinned
    env = os.environ.get(DIR_ENV)
    return Path(env) if env and not os.environ.get(CHARAS_ENV) else None


def root() -> Path:
    """The shared store's folder (never inside the release payload)."""
    pinned = _pinned()
    if pinned is not None:
        return pinned
    env = os.environ.get(DIR_ENV)
    return Path(env) if env else derived_root() / "Endfield"


def charas_root() -> Path | None:
    """``Charas/Endfield``, or ``None`` while one folder is pinned as the whole store."""
    if _pinned() is not None:
        return None
    env = os.environ.get(CHARAS_ENV)
    if env:
        return Path(env)
    from ..converted.store import root as consolidated
    return consolidated() / "Endfield"


@contextmanager
def only(path: Path):
    """Treat *path* as the whole store on this thread (a candidate store, a temporary one)."""
    previous = getattr(_local, "only", None)
    _local.only = Path(path)
    try:
        yield
    finally:
        _local.only = previous


def _signature(base: Path) -> tuple | None:
    """What changes when a character or model folder is added, renamed or rewritten."""
    try:
        rows = []
        with os.scandir(base) as characters:
            for character in characters:
                if not character.is_dir():
                    continue
                with os.scandir(character.path) as models:
                    rows.append((character.name, character.stat().st_mtime_ns,
                                 tuple(sorted((m.name, m.stat().st_mtime_ns) for m in models if m.is_dir()))))
        return tuple(sorted(rows))
    except FileNotFoundError:
        return None


def rescan() -> None:
    """Forget the scanned ``Charas`` folders (after making or moving one in this process)."""
    _scanned.clear()


def charas_roots() -> list[tuple[str, Path]]:
    """``(source, folder)`` of every game this runtime runs (:data:`.sources.GAMES`), Endfield first.

    With :data:`CHARAS_ENV` naming a folder it is Endfield's alone (tests, candidate stores);
    with one folder pinned as the whole store there are none.
    """
    base = charas_root()
    if base is None:
        return []
    if os.environ.get(CHARAS_ENV):
        return [(SOURCE, base)]
    from .sources import GAMES
    return [(source, base.parent / game) for source, game in GAMES.items()]


def models():
    """The model folders under ``Charas/<game>`` (:class:`charas.Scan`); rescanned when the folders change.

    A model found under two games' folders is a problem, as one found twice under one.
    """
    from . import charas
    roots = charas_roots()
    if not roots:
        return charas.Scan()
    signature = tuple((str(base), _signature(base)) for _source, base in roots)
    cached = _scanned.get("scan")
    if cached is None or cached[0] != signature:
        merged = charas.Scan()
        for source, base in roots:
            found = charas.scan(base, source)
            merged.problems.extend(found.problems)
            for ident, folder in found.models.items():
                if ident in merged.models:
                    merged.problems.append({"ident": ident, "path": [str(merged.models.pop(ident)), str(folder)],
                                            "error": "the same model is in more than one game's folder"})
                else:
                    merged.models[ident] = folder
        cached = (signature, merged)
        _scanned["scan"] = cached
    return cached[1]


def model_root(ident: str) -> Path | None:
    """Model *ident*'s ``Charas`` folder, ``None`` when it has none; a broken one is an error."""
    found = models()
    problem = found.broken(ident)
    if problem is not None:
        from .actor import ActorError
        raise ActorError(f"Charas folder of {ident} cannot be read: {problem['error']} ({problem['path']})")
    return found.models.get(ident)


def home(ident: str) -> Path:
    """The store holding model *ident*: its ``Charas`` folder, else the shared store."""
    found = model_root(ident)
    return found if found is not None else root()


def ensure_home(ident: str) -> Path:
    """Model *ident*'s ``Charas`` folder, made now when it has none (the shared store's copy moved over whole).

    The folder is named from the local name tables (:mod:`.names`); with one folder pinned as
    the whole store it is that folder.
    """
    found = model_root(ident)
    base = charas_root()
    if found is not None or base is None:
        return found if found is not None else root()
    from . import charas, links, names
    with _made:
        found = model_root(ident)
        if found is None:
            identity = names.identify(ident, links.read())
            found, _report = charas.create(identity, base, root())
            rescan()
    return found


def stores() -> list[Path]:
    """Every store, in the order a texture is looked up: the ``Charas`` folders, then the shared store."""
    return [*(folder for _ident, folder in sorted(models().models.items())), root()]


@contextmanager
def opened(ident: str | None = None, *, write: bool = False):
    """Run ``bake`` functions against model *ident*'s store (:func:`home`), or the shared store.

    *write*: the model's own ``Charas`` folder, made if it has none (:func:`ensure_home`).
    """
    if ident is None:
        folder = root()
    else:
        folder = ensure_home(ident) if write else home(ident)
    with bake.store(folder):
        yield


@contextmanager
def writing(ident: str):
    """Write what a model's conversion writes — its item, table and textures — into its own store."""
    previous = getattr(_local, "target", None)
    _local.target = ensure_home(ident)
    try:
        yield _local.target
    finally:
        _local.target = previous


def _target() -> Path | None:
    return getattr(_local, "target", None)


def write_target() -> Path:
    """Where a texture written now lands: the store :func:`writing` opened, else the shared store."""
    target = _target()
    return target if target is not None else root()


def has(category: str, ident: str) -> bool:
    """Is ``category/ident`` converted and on disk?"""
    with opened(ident):
        return bake.has(category, ident)


def _listed(folder: Path, prefix: str = "") -> set[str]:
    with bake.store(folder):
        return {key for key, entry in bake.manifest()["items"].items()
                if key.startswith(prefix) and "kind" not in entry and "/" in key}


def idents(category: str) -> list[str]:
    """Every converted item of *category* in every store, sorted."""
    prefix = f"{category}/"
    keys = set().union(*(_listed(folder, prefix) for folder in stores()))
    return sorted(key[len(prefix):] for key in keys)


def items() -> list[tuple[str, str]]:
    """Every converted item in every store as ``(category, ident)``, sorted."""
    keys = set().union(*(_listed(folder) for folder in stores()))
    return sorted(tuple(key.split("/", 1)) for key in keys)


def write_item(category: str, ident: str, pieces: list, nodes: dict,
               extra: dict | None = None):
    """Store one converted prefab; *extra* is JSON kept beside the pieces.

    A model's item goes to its ``Charas`` folder (:func:`ensure_home`); any other item to the
    store a conversion is writing (:func:`writing`), else the shared store.
    """
    target = _target()
    if target is not None:
        folder = target
    elif category in MODEL_CATEGORIES:
        folder = ensure_home(ident)
    else:
        folder = root()
    with bake.store(folder):
        path = bake.write_item(category, ident, pieces, label=ident,
                               source=SOURCE, nodes=nodes)
        if extra is not None:
            bake.write_named_table(SOURCE, category, ident, extra)
    return path


#: The prefix of the lent body's joints kept under the character's own (``body.GRAFT``), and the one
#: an earlier version wrote: an item or a table stored with it is read with this one
GRAFT = "Origin_body_"
LEGACY_GRAFT = "KK_body_"
#: tables whose content is hashed by the certification as stored (``certification.capture``): not renamed
CERTIFIED = ("composite",)


def _grafted(name):
    return GRAFT + name[len(LEGACY_GRAFT):] if isinstance(name, str) and name.startswith(LEGACY_GRAFT) else name


def _regraft(value):
    if isinstance(value, dict):
        return {_grafted(k): (v if k in CERTIFIED else _regraft(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_regraft(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_regraft(v) for v in value)
    return _grafted(value)


def read_item(category: str, ident: str):
    """The stored item, or ``None`` when it was never converted.

    Joints named with :data:`LEGACY_GRAFT` read as :data:`GRAFT`.
    """
    with opened(ident):
        item = bake.read_item(category, ident)
    if item is not None:
        if item.nodes and any(str(name).startswith(LEGACY_GRAFT) for name in item.nodes):
            item.nodes = {_grafted(name): (_grafted(node[0]), *node[1:]) for name, node in item.nodes.items()}
        item.pieces = [piece._replace(bones=[_grafted(b) for b in piece.bones])
                       if piece.bones and any(str(b).startswith(LEGACY_GRAFT) for b in piece.bones) else piece
                       for piece in item.pieces]
    return item


def read_extra(category: str, ident: str) -> dict | None:
    """The JSON stored with an item (``None`` when there is none).

    Legacy joint names read as :func:`read_item` reads them.
    """
    with opened(ident):
        table = bake.named_table(SOURCE, category, ident)[1]
    return table if table is None else _regraft(table)


def _texture_store(cab: str, name: str) -> Path | None:
    """The first store with an answer for texture ``(cab, name)``: the one being written, else any."""
    target = _target()
    for folder in ([target] if target is not None else stores()):
        with bake.store(folder):
            if bake.named_entry(SOURCE, cab, name) is not None:
                return folder
    return None


def has_texture(cab: str, name: str) -> bool:
    """Is texture *name* of serialized file *cab* stored (in the store being written, while one is)?"""
    return _texture_store(cab, name) is not None


def write_texture(cab: str, name: str, texture: CS.Map) -> None:
    """Store one texture, keyed by the serialized file it came from, in the store being written."""
    with bake.store(write_target()):
        bake.write_named_texture(SOURCE, cab, name, texture)


def read_texture(cab: str, name: str) -> CS.Map | None:
    """The stored texture, or ``None``."""
    folder = _texture_store(cab, name)
    if folder is None:
        return None
    with bake.store(folder):
        return bake.read_named_texture(SOURCE, cab, name)


def texture_path(cab: str, name: str) -> Path | None:
    """The stored texture's own file (a PNG), or ``None`` when it is not stored."""
    folder = _texture_store(cab, name)
    if folder is None:
        return None
    with bake.store(folder):
        entry = bake.named_entry(SOURCE, cab, name)
        if entry is None or entry.get("absent"):
            return None
        return bake.bake_dir() / entry["file"]
