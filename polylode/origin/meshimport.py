"""Load meshes and textures out of Illusion asset bundles.

Bundles are opened lazily and cached; a mesh comes back as an ordinary project
``Mesh`` in metres, with Unity's left-handed Z flipped so it matches this
project's right-handed convention.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..mesh.mesh import Mesh
from ..core import swallowed

_bundle_cache: "OrderedDict[str, object]" = OrderedDict()
_mesh_cache: "OrderedDict[tuple, Mesh]" = OrderedDict()

# A single Illusion bundle can decompress to hundreds of megabytes, so these
# caches are deliberately tiny — holding several at once is what makes the
# importer eat memory.
MAX_OPEN_BUNDLES = 6
MAX_CACHED_MESHES = 128


class BundleTooLarge(RuntimeError):
    pass


# Bundles that could not be opened, by reason.  **A load that fails for want of
# memory is not a load that found nothing**, and until §7.363 the two were the
# same thing: the texture came back ``None``, the layer was silently not drawn,
# and the card's report said ``missing 0``.  One card
# (``Community_Damien_6``) loses its cheek and eyeshadow sheets that way as
# soon as the job's committed memory is near its ceiling — the same answer
# ``bake.PROVIDER_ERRORS`` and ``TextureCache.evicted`` give: countable, so a
# run with a non-zero tally is not evidence about what a card is missing.
LOAD_FAILURES: dict[str, int] = {}


def note_load_failure(what: str, exc: BaseException) -> None:
    """Record a bundle that would not open (called where the error is swallowed)."""
    key = f"{type(exc).__name__}: {what}"
    LOAD_FAILURES[key] = LOAD_FAILURES.get(key, 0) + 1


def load_failures() -> dict:
    """A copy of the tally — empty means every bundle asked for opened."""
    return dict(LOAD_FAILURES)


def clear_load_failures() -> None:
    LOAD_FAILURES.clear()


def _evict_bundles(keep: int) -> None:
    import gc
    while len(_bundle_cache) > keep:
        _, env = _bundle_cache.popitem(last=False)
        try:
            for f in getattr(env, "files", {}).values():
                if hasattr(f, "close"):
                    f.close()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.meshimport._evict_bundles: getattr.values", exc)
        del env
    gc.collect()


def _load_bundle(path: Path):
    key = str(path)
    env = _bundle_cache.get(key)
    if env is not None:
        _bundle_cache.move_to_end(key)
        return env
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    from .memguard import max_bundle_bytes, over_budget
    if size > max_bundle_bytes():
        raise BundleTooLarge(f"{p.name} is {size / 1e6:.0f} MB")
    import UnityPy
    _evict_bundles(MAX_OPEN_BUNDLES - 1)             # make room *before* loading
    # the count cap is a poor proxy for bytes; if loading this one would take us
    # over the real budget, drop everything rather than risk the machine
    if over_budget(margin=size / (1024 * 1024)):
        _evict_bundles(0)
        _mesh_cache.clear()
    env = UnityPy.load(str(p))
    _bundle_cache[key] = env
    return env


def load_bytes_bundle(key: str, data: bytes):
    """A bundle that only exists in memory (a ``.zipmod`` member), cached.

    Same cache, same eviction and the same memory budget as a bundle on disk
    (:func:`_load_bundle`) — one owner, because the reason to cache these is the
    same and having two would mean two budgets.  **The point is the repeat**:
    ``sideloader.load_mod_bundle`` decompressed and re-parsed the whole bundle
    for *every* texture read out of it, and one card that takes four sheets from
    ``[Kero]CharamakerSupportMOD`` (33.6 MB compressed) walked the head stage
    from 0.42 GB to 1.20 GB and printed UnityPy's "Error parsing file" —
    a ``MemoryError`` — on the way (§7.363).

    *key* identifies the member (``<zipmod>!<inner path>``); *data* is the
    member's bytes, which the caller has already read.  Raises
    :class:`BundleTooLarge` past the per-machine ceiling, like the disk path.
    """
    env = _bundle_cache.get(key)
    if env is not None:
        _bundle_cache.move_to_end(key)
        return env
    from .memguard import max_bundle_bytes, over_budget
    size = len(data or b"")
    if size > max_bundle_bytes():
        raise BundleTooLarge(f"{key.rsplit('!', 1)[-1]} is {size / 1e6:.0f} MB")
    _evict_bundles(MAX_OPEN_BUNDLES - 1)
    if over_budget(margin=size / (1024 * 1024)):
        _evict_bundles(0)
        _mesh_cache.clear()
    env = memory_bundle(data)
    _bundle_cache[key] = env
    return env


def memory_bundle(data: bytes):
    """A bundle read from bytes, which has no folder to find other files in.

    The reader gives such a bundle the **working directory** as its folder and
    walks all of it for a file a reference names (``unity_builtin_extra``, a
    neighbouring cab).  Whatever happens to lie under there is then loaded as
    part of the bundle — another game's ``unity_builtin_extra`` left in a
    scratch folder did, in the middle of the walk that indexes the bundle,
    and every card with a modded head lost its materials to "dictionary
    changed size during iteration" (§7.989).  A member of an archive has no
    neighbours: the reference stays unresolved, as it was before that file
    was there.
    """
    import UnityPy
    env = UnityPy.load(data)
    env.path = ""
    return env


def _cache_mesh(key, mesh: Mesh) -> None:
    _mesh_cache[key] = mesh
    while len(_mesh_cache) > MAX_CACHED_MESHES:
        _mesh_cache.popitem(last=False)


def clear_cache() -> None:
    import gc
    _evict_bundles(0)
    _mesh_cache.clear()
    gc.collect()


def trim_cache() -> None:
    """Release the bundles but keep the (much smaller) decoded meshes."""
    _evict_bundles(0)


# --------------------------------------------------------------------------
def _parse_obj(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse UnityPy's OBJ export into (positions, normals, uvs, indices)."""
    v, vn, vt, faces = [], [], [], []
    for line in text.splitlines():
        if not line:
            continue
        c = line[0]
        if c == "v":
            if line[1] == " ":
                v.append(line[2:])
            elif line[1] == "n":
                vn.append(line[3:])
            elif line[1] == "t":
                vt.append(line[3:])
        elif c == "f":
            faces.append(line[2:])

    def nums(rows, n):
        if not rows:
            return np.zeros((0, n), dtype=np.float32)
        return np.fromstring(" ".join(rows), sep=" ",
                             dtype=np.float32).reshape(-1, n)

    pos = nums(v, 3)
    nrm = nums(vn, 3)
    uv = nums(vt, 2)
    idx = []
    for f in faces:
        parts = f.split()
        if len(parts) < 3:
            continue
        ids = [int(p.split("/", 1)[0]) - 1 for p in parts]
        for k in range(1, len(ids) - 1):
            idx.extend((ids[0], ids[k], ids[k + 1]))
    return pos, nrm, uv, np.asarray(idx, dtype=np.uint32)


def load_mesh(bundle_path: Path, asset_name: str, scale: float = 1.0,
              name: str | None = None) -> Mesh | None:
    """Read one named Mesh from a bundle into project space."""
    key = (str(bundle_path), asset_name, round(scale, 6))
    cached = _mesh_cache.get(key)
    if cached is not None:
        return cached.copy()

    try:
        env = _load_bundle(Path(bundle_path))
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.meshimport.load_mesh: env = _load_bundle(Path(bundle_path))", exc)
        return None

    obj_text = None
    for obj in env.objects:
        if obj.type.name != "Mesh":
            continue
        try:
            d = obj.read()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.meshimport.load_mesh: obj.read", exc)
            continue
        if d.m_Name != asset_name:
            continue
        try:
            obj_text = d.export()
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.meshimport.load_mesh: obj_text = d.export()", exc)
            return None
        break
    if not obj_text:
        return None

    pos, nrm, uv, idx = _parse_obj(obj_text)
    if len(pos) == 0 or len(idx) == 0:
        return None

    # Unity is left-handed and both engines face +Z, so the conversion mirrors
    # X -- but UnityPy's OBJ writer has already done it (positions, normals and
    # face winding alike), so there is nothing left to flip here.
    pos = pos * float(scale)
    if len(nrm) != len(pos):
        nrm = None

    m = Mesh.from_arrays(pos, idx, normals=nrm,
                         uvs=uv if len(uv) == len(pos) else None,
                         name=name or asset_name)
    _cache_mesh(key, m)
    return m.copy()


def list_meshes(bundle_path: Path) -> list[str]:
    try:
        env = _load_bundle(Path(bundle_path))
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.meshimport.list_meshes: env = _load_bundle(Path(bundle_path))", exc)
        return []
    out = []
    for obj in env.objects:
        if obj.type.name == "Mesh":
            try:
                out.append(obj.read().m_Name)
            except Exception as exc:                 # noqa: BLE001
                swallowed.note("origin.meshimport.list_meshes: obj.read", exc)
    return out


def find_meshes(bundle_path: Path, prefix: str) -> list[str]:
    return [n for n in list_meshes(bundle_path) if n.startswith(prefix)]


# --------------------------------------------------------------------------
def load_texture(bundle_path: Path, asset_name: str) -> np.ndarray | None:
    """Read a Texture2D as float RGBA in [0,1]."""
    try:
        env = _load_bundle(Path(bundle_path))
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.meshimport.load_texture: env = _load_bundle(Path(bundle_path))", exc)
        return None
    for obj in env.objects:
        if obj.type.name != "Texture2D":
            continue
        try:
            d = obj.read()
            if d.m_Name != asset_name:
                continue
            img = d.image
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.meshimport.load_texture: obj.read", exc)
            continue
        if img is None:
            return None
        # row 0 = v = 1, same as every other texture in the project;
        # ``TextureCache.upload`` does the flip for GL (see prefab._decode)
        return np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    return None
