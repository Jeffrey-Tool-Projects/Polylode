"""Bake Illusion ``.unity3d`` items into this project's own asset format.

Reading a bundle means decompressing it, walking a Unity serialized file and
re-parsing an OBJ export per renderer.  For one card that is ~10 s and a
gigabyte of transient memory, every time, because nothing about the result
depends on the card — the same prefab yields the same geometry.

A bake writes what we actually keep::

    assets_baked/
        manifest.json                 category → id → {file, label, source}
        <category>/<id>.npz           every renderer's arrays, one archive
        <category>/<id>/<n>_tex.png   the maps that survived resolution
        tex/<category>/<id>.png       a texture-only list's entry (mt_eye …)
        tex/named/<source>/<bundle>/<texture>.png
                                      a map a list row names by bundle and
                                      texture name (garment masks, lid shadow)

``.npz`` because the payload *is* numpy arrays — positions, indices, skin
weights, bind matrices — and it round-trips them exactly with no schema to
keep in sync.  Textures go out as PNG rather than into the archive so they
stay inspectable and the loader can skip ones it does not need.

Loading a baked item needs numpy and Pillow only.  UnityPy is still a **runtime**
dependency of the release, though: since §7.145 what ships is the converter
rather than its output, so opening a bundle is something the shipped program
does on the user's machine (:mod:`convert`).
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.paths import derived_root, write_json_atomic
from ..mesh.mesh import Mesh
from . import colorspace as CS
from . import progress as PROG
from .prefab import Piece
from ..core import swallowed

BAKE_DIR_ENV = "ISLANDLIFE_ASSET_BAKE"
# 2: piece.shader carries ``_shader`` (the material's shader name).  Whether a
# pass clips is a property of the shader, not of the material's ``_alphaTest``
# float — ``cf_m_hitomi_00`` ships 0.5 while ``toon_eye_lod0`` never discards —
# so a bake without it cannot set the cut-out up correctly.
# 3: rigid pieces carry their MeshRenderer's material.  A ``MeshFilter`` holds
# the mesh but no materials, so every rigid accessory used to bake with no
# texture, no colours and no shader — glasses drew as a flat slab.
# 4: pieces carry their blend shapes.  Illusion's facial expressions are
# blend shapes on the head meshes, so a bake without them leaves an imported
# character unable to blink or change expression at all.
# 5: rigid pieces no longer name-guess their albedo (green beret bug).
# 6: pieces carry their second UV set and their ``_LineMask``.  Hair lays each
# strand out on UV1 and ``main_hair`` samples ``_HairGloss`` along it — without
# it there is no highlight band at all — and the line masks feed the shading
# term the forward passes multiply into the diffuse.
# 7: pieces carry ``bone_rest`` — the sway-bone track after Illusion's own
# DynamicBone solver has been run to steady state, plus the accessory
# component's colour switches inside ``shader``.
# 8: the item carries ``sway`` — the per-bone local rotation the same solver
# produces for chains the *body* armature owns (``cf_j_sk_*``).  ``bone_rest``
# above only covers a piece's own prefab bones, so a baked skirt kept its mesh
# but lost the drape: ``dress`` never saw an ``item.sway`` to write onto the
# skeleton and every baked bottom hung in its authored bell shape.
# 9: an item's own metadata (piece names, bone lists, shader, map paths) moved
# out of ``manifest.json`` and into its ``.npz``.  The manifest was rewritten in
# full after every single item, and a mod item carries ~16 KB of it: at the
# 25,736 sideloader items the index reached 19 MB after 1,155 of them, heading
# for ~300 MB rewritten 25,736 times.  The manifest is now a thin key → file
# index, so a bake is O(1) per item and the game parses kilobytes at startup
# instead of hundreds of megabytes.
# 10: pieces carry ``_clothRole`` inside ``shader`` — which of
# ``ChaClothesComponent``'s renderer lists a garment's piece belongs to.  That
# is what says which of the list row's three texture pairs it takes and which
# pieces are the emblem quads; the previous bake resolved the maps with a
# name-matching guess, so re-reading it would hand the old answer back.
# 11: pieces carry UV2, UV3 and their COLOR stream (``Mesh.vcolor``, which is
# *not* ``Mesh.colors`` — see the note on ``Mesh.OPTIONAL_STREAMS``).
# ``main_skin`` places
# its three decal layers on coordinate sets of their own and gates two of them
# on COLOR — the areola on UV1 × COLOR.r, the under-hair on UV2 × COLOR.b, the
# eyeshadow on UV3 (§7.50 六) — so a bake with only UV0 and UV1 cannot
# composite any of them where the game does.  This is not additive: an archive
# written at 10 has no way to say "this mesh genuinely has no UV2" as against
# "the format did not store one", and reading it either way would put the
# areola somewhere else on the map.
# 12: the item carries its ``N_move`` / ``N_move2`` chain and every piece the
# mask saying which of the two it hangs under (``prefab.move_chain``).  The
# card's ``addMove`` **replaces** those two nodes' local TRS in the middle of
# the accessory hierarchy, so a reader needs the parent and node world matrices
# to turn the card's numbers into a placement; the bake used to keep only the
# finished chain, which forced the offset to be multiplied onto the outside of
# the item and dropped whatever scale or rotation sits above the node — 1.5× on
# ``p_acs_valletta_heart1``, 5× on ``p_acs_ear_weddingshizuku``, a full
# rotation on ``p_acs_catear_*``, and 131 of the 357 base accessories carry
# something (§7.67).
# 13: every map carries its ``Texture2D.m_ColorSpace`` — per piece in
# ``pieces[i].map_cs``, and per texture-only list entry in the manifest row's
# ``cs``.  Koikatsu is a Linear project (``m_ActiveColorSpace = 1``), so the
# ``*_create`` shaders sample an sRGB-flagged map through the hardware's decode
# and a linear-flagged one raw, and Illusion's import settings are not
# consistent: locally the ``*_mc`` colour masks split 62 linear / 40 sRGB while
# 96 of 107 ``*_t`` albedos are sRGB.  A mask is a lerp weight, so the flag
# moves the colour and not just its precision — and it is only knowable at the
# bundle, because a baked PNG is bytes with no ``m_ColorSpace`` left on it.
# Not additive: an archive written at 12 cannot say "this map is linear" as
# against "the format had no field for it", and reading it either way picks a
# colour for every garment (:mod:`colorspace`).
# 14: ``piece.shader`` carries the material's ``m_ShaderKeywords``.  These
# shaders ship two forward bases — one ending in ``discard_nz`` and one not —
# and the keywords are what select between them; ``_ALPHA_A_ON`` /
# ``_ALPHA_B_ON`` switch on the cut-out whose test is
# ``min(alpha-mask term, _MainTex.a) < 0.5``.  Without the field the loader was
# guessing from the composited pixels, and had that guess switched **off** for
# tops and bottoms, so a garment whose albedo carries compressed-block garbage
# in its transparent texels drew the garbage (§7.88).  Not additive: an archive
# written at 13 cannot say "this material has no keywords" as against "the
# format had nowhere to put them", and the two answers are opposite.
# 15: ``bone_world`` / ``bone_rest`` are the **whole** prefab hierarchy.  The
# walk that built them stopped at a depth ceiling of 12 (``_collect_worlds``),
# which left 35.1% of the skinning bones in the sampled assets with no matrix at
# all — and the archive stores that as a hole, indistinguishable from a renderer
# naming a bone the prefab genuinely does not carry.  Reading a format-14
# archive therefore hands the truncated track back however the walk is fixed, so
# this is not additive; every item has to be baked again (§7.96).
# 16: ``piece.shader`` carries ``_rendName`` — the renderer's own GameObject
# name.  MaterialEditor addresses renderers by exactly that string
# (``MaterialAPI.SetRendererEnabled`` matches ``NameFormatted(r) ==
# RendererName``), and 101 of this install's cards switch **720** of them off;
# without the field there is nothing on this side for those rows to name, so
# every one of them was drawn (§7.138).  Technically additive — a format-15
# archive just has no ``_rendName`` — but deliberately not treated as such: an
# item missing the name matches nothing, which is indistinguishable from "the
# card's rows are orphans", and the two want opposite responses.  A version
# bump is what turns that into "re-bake" instead of a card that quietly keeps
# the geometry its author hid.
# Format 17 (§7.379): four things the old archives cannot say, all riding in
# what a piece already stores, and each silent when absent — so a bump, not an
# "additive" field (the ``_rendName`` lesson above):
#   * ``_clothOpt`` — the ``objOpt01/02`` group a garment renderer belongs to,
#     which the card's ``hideOpt`` switches off; absent, every option draws;
#   * ``_stateNode`` — the ``n_top_a``/``n_top_b``/… node a garment renderer
#     hangs under, which ``UpdateVisible`` switches per clothes state;
#   * ``_matName`` — the material's own name, which MaterialEditor's
#     float / colour / texture / shader / queue rows address;
#   * ``_submeshes`` — how many submeshes the renderer's mesh has, which
#     decides what a MaterialEditor material copy draws (§7.381);
#   * ``_mapWrap`` — each map's ``m_TextureSettings.m_WrapMode``, which a
#     tiled ``_NormalMap_ST`` needs (§7.383);
#   * ``_queue`` / ``_shaderQueue`` — the render queue the blended pass sorts
#     by; absent, a lens draws before the eyes behind it;
#   * ``{i}_shp{j}_n`` — a blend-shape frame's normal delta; absent, a morph
#     keeps the rest pose's shading;
#   * one piece per **submesh** of a multi-material renderer, not the whole
#     mesh once per material (5.7% of sampled zipmod renderers).
# §7.509 adds three, and they are additive on purpose: absent reads as "drawn",
# which is what every archive without them did, and the prefabs that carry any
# are a scanned, short list (``dev/_inactive_scan.py``) re-baked by name rather
# than twenty-thousand items by a version bump:
#   * ``_offNode`` — the inactive GameObject on the renderer's chain that none of
#     Origin's run-time switches turns on; Unity never draws the renderer;
#   * ``_rendOff`` — ``m_Enabled`` 0; only a MaterialEditor ``Enabled 1`` row
#     turns it on (``matedit.unity_drawn``);
#   * ``_clothSleeve`` — the ``objSleeves01/02/03`` group, of which the top's
#     ``DrawSode`` draws the one the card's ``sleevesType`` names.
FORMAT_VERSION = 17
#: Revision of an item's sway data (``sway``, ``sway_spec``), kept in its meta
#: so a change there re-bakes the items it touches and nothing else
#: (``convert.items_stale_sway``).  2: the body colliders in the game's order
#: and ``sway_spec`` with its ``collide`` flag; 3: the particles
#: ``AppendParticles`` makes (``m_notRolls``, end particles, the distribution
#: along the path) and one solver for the settle and the run time (§7.540).
#: Absent reads as 1.
SWAY_REV = 3
_MAP_NAMES = ("texture", "mask", "detail", "normal", "alpha_mask", "ramp",
              "line_mask", "glass_ramp")


def bake_dir() -> Path:
    """Where the archives live — **never inside the release payload**.

    A bake is a conversion of the user's own game data, and converting does not
    change who owns it (著作權.md 一), so it is written on their machine and
    stays there.  ``derived_root()`` is the checkout in a source tree and a
    per-user directory once frozen; the env var still wins so ``bake_verify``
    can point a run at an empty directory to force the bundle path.
    """
    other = getattr(_other, "root", None)
    if other is not None:
        return other
    env = os.environ.get(BAKE_DIR_ENV)
    return Path(env) if env else derived_root() / "assets_baked"


def manifest_path() -> Path:
    return bake_dir() / "manifest.json"


# ---------------------------------------------------------------------------
# another store
# ---------------------------------------------------------------------------
#: The store this *thread* is working in when it is not the default one, and
#: the manifest state of every such store by root: ``[manifest, stale version,
#: loaded mtime]``, the same three things the globals below hold for the
#: default store.
_other = threading.local()
_other_state: dict[Path, list] = {}


@contextmanager
def store(root: Path):
    """Read and write the store at *root* instead of :func:`bake_dir`'s.

    Same format, same functions, its own ``manifest.json`` — for content that
    is not Koikatsu's and must not sit among its items, where the re-bake
    selectors and ``--prune`` would judge it by Koikatsu's lists
    (``polylode/endfield``, §7.563).

    Per thread, because dressing runs on a worker: a character loading in the
    background keeps reading the default store while this thread is in here.
    Manifest writes are immediate in another store (``defer_saves`` is the
    default store's bulk mode).
    """
    previous = getattr(_other, "root", None)
    _other.root = Path(root)
    try:
        yield
    finally:
        _other.root = previous


def _read_manifest(path: Path) -> list:
    """``[manifest, stale version, mtime]`` of the manifest file at *path*."""
    data = {"version": FORMAT_VERSION, "items": {}}
    stale = mtime = None
    if path.exists():
        try:
            mtime = path.stat().st_mtime
            loaded = json.loads(path.read_text(encoding="utf-8"))
            version = int(loaded.get("version", 0))
            if version == FORMAT_VERSION:
                data = loaded
            elif loaded.get("items"):
                stale = version
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    return [data, stale, mtime]


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
_manifest: dict | None = None
# The version the archive on disk was written at, when that is not the version
# this build reads.  A mismatch means every item has to be baked again — the
# archives themselves are still there, they just cannot be trusted — and the
# manifest is rebuilt from scratch, which silently orphans them.  Bumping
# ``FORMAT_VERSION`` to 7 left ~1 GB of format-6 ``.npz`` on disk that nothing
# referenced and nothing would ever clean up, so the number is kept and
# ``stale_files()`` can find them.
_stale_version: int | None = None
# When the manifest this process holds was last written to disk, read as it was
# loaded — what :func:`written_since` needs to tell a row that was flushed from
# a file whose row died with a killed run.
_loaded_mtime: float | None = None


def manifest(refresh: bool = False) -> dict:
    global _manifest, _stale_version, _loaded_mtime
    other = getattr(_other, "root", None)
    if other is not None:
        if refresh or other not in _other_state:
            _other_state[other] = _read_manifest(manifest_path())
        return _other_state[other][0]
    if _manifest is not None and not refresh:
        return _manifest
    _manifest, _stale_version, _loaded_mtime = _read_manifest(manifest_path())
    return _manifest


def stale_version() -> int | None:
    """The format the on-disk manifest was written at, if it is not ours."""
    manifest()
    other = getattr(_other, "root", None)
    return _stale_version if other is None else _other_state[other][1]


def _manifest_mtime() -> float | None:
    """When the manifest in use was last written, as it was loaded."""
    other = getattr(_other, "root", None)
    return _loaded_mtime if other is None else _other_state[other][2]


#: Stores that live under :func:`bake_dir` with **their own** index, so the
#: item manifest has no entry for them and must not judge them.  One owner for
#: each name: the modules that write there build their paths from these.
#:
#: This exists because it went wrong: ``stale_files`` walked the whole tree,
#: and once the H clips moved in beside the items, ``tools/bake_assets.py
#: --prune`` listed **9,022 of them (4.46 GB) plus ``face_ptn.json``** as
#: "nothing points at these" — with ``--force`` one flag away, and the status
#: file saying that flag was the next step (§7.352).
H_ANIM_DIR = "h_anim"            # origin.hanim — its own manifest.json
FACE_PTN_FILE = "face_ptn.json"  # figure.face_morph.bake_face_patterns
RESUME_FILE = "resume.json"      # resume_stamp — forced passes still under way
FOREIGN = (H_ANIM_DIR, FACE_PTN_FILE, RESUME_FILE)


def stale_files() -> list[Path]:
    """Baked files no manifest entry points at — dead weight from an old format.

    Compares the tree against the manifest rather than trusting a version
    number, so it also catches an item whose list id disappeared from
    ``assets_cache`` between bakes.  Anything under :data:`FOREIGN` belongs to
    a store with its own index and is never listed.
    """
    root = bake_dir()
    if not root.is_dir():
        return []
    foreign = {root / name for name in FOREIGN}
    # An item is one ``<category>/<ident>.npz`` plus the map PNGs beside it in
    # ``<category>/<ident>/``.  Since format 9 the map paths live inside the
    # archive, so the directory is what marks them live — opening 25,000
    # archives to list their PNGs would make a prune cost as much as a bake.
    live: set[Path] = set()
    live_dirs: set[Path] = set()
    for entry in manifest()["items"].values():
        rel = entry.get("file")
        if not rel:
            continue
        path = root / rel
        live.add(path)
        if path.suffix == ".npz":
            live_dirs.add(path.with_suffix(""))
    out = []
    for path in root.rglob("*"):
        if not path.is_file() or path == manifest_path() or path in live:
            continue
        if path.parent in live_dirs:
            continue
        if path in foreign or any(f in path.parents for f in foreign):
            continue
        out.append(path)
    return sorted(out)


# A long bake writes thousands of items; flushing the index after each one is
# quadratic.  Callers that write in bulk turn on ``defer_saves()`` and call
# ``flush()`` periodically and at the end — an interrupted run then loses the
# index entries since the last flush, not the archives, and ``--prune`` sweeps
# the files those entries would have pointed at.
_deferred = [False]
_dirty = [False]


def defer_saves(on: bool = True) -> None:
    _deferred[0] = bool(on)


def flush() -> None:
    if _dirty[0] and _manifest is not None:
        _write_manifest(_manifest)
        _dirty[0] = False


def _write_manifest(data: dict) -> None:
    write_json_atomic(manifest_path(), data, indent=1)


def _save_manifest(data: dict) -> None:
    if _deferred[0] and getattr(_other, "root", None) is None:
        _dirty[0] = True
        return
    _write_manifest(data)


def key_of(category: str, ident) -> str:
    return f"{category}/{ident}"


def has(category: str, ident) -> bool:
    entry = manifest()["items"].get(key_of(category, ident))
    return bool(entry) and (bake_dir() / entry["file"]).exists()


def written_since(key: str, since: float | None) -> bool:
    """Was the file behind manifest row *key* written at or after *since*?

    A forced pass that the guard kills starts over from nothing (every entry
    already exists, so ``force`` is what makes it re-convert).  Every writer
    here replaces a file whose name is its key, so a row whose file is newer
    than the pass's first start was re-converted by that pass and a retry can
    keep it (:func:`resume_stamp`).

    **And older than the manifest this process loaded**: a bulk writer defers
    the manifest (``defer_saves``), so a run killed between a file and the next
    flush leaves the new file under the old row — the old colour space, the old
    wrap mode.  A file written after the last flush the loaded manifest saw is
    redone.  ``None``, a row with no file (an ``absent`` answer) or a missing
    file: False, so the entry is redone.
    """
    if since is None:
        return False
    entry = manifest()["items"].get(key)
    rel = entry.get("file") if entry else None
    loaded = _manifest_mtime()
    if not rel or loaded is None:
        return False
    try:
        return since <= (bake_dir() / rel).stat().st_mtime <= loaded
    except OSError:
        return False


def override_sources() -> dict:
    """``{stock bundle path: sources fingerprint}`` the converter last converted against.

    A stock asset is Sideloader's to hand out (§7.486): a mod registered at its
    path, or a loose PNG in its folder, answers before the install.  What
    answered is not in the archive, so a mod added or removed after a
    conversion left every item on that path converted from the old source —
    and nothing asked again, because ``force`` was the only thing that did
    (§7.491).  Kept in the manifest, so a format bump forgets it with the items
    it describes.  A path with no entry was converted from the install alone.
    """
    return dict(manifest().get("override_sources") or {})


def record_override_sources(fingerprints: dict) -> None:
    """Merge ``{path: fingerprint}``; an empty fingerprint drops the path (the install answers)."""
    data = manifest()
    table = dict(data.get("override_sources") or {})
    for path, print_ in fingerprints.items():
        if print_:
            table[path] = print_
        else:
            table.pop(path, None)
    data["override_sources"] = table
    _save_manifest(data)


def _resume_path() -> Path:
    return bake_dir() / RESUME_FILE


def resume_stamp(job: str) -> float:
    """When forced pass *job* first started, recorded on the first call.

    Stays on disk until :func:`resume_done`, so a pass killed part-way and run
    again keeps what it already wrote (:func:`written_since`) instead of
    re-converting all of it — the lesson of the H clip re-index (§7.371).
    """
    import time
    path = _resume_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    if job not in data:
        data[job] = time.time()
        write_json_atomic(path, data)
    return float(data[job])


def resume_done(job: str) -> None:
    """Forget :func:`resume_stamp`'s record for *job*: the pass ran to the end."""
    path = _resume_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return
    if not isinstance(data, dict) or data.pop(job, None) is None:
        return
    if data:
        write_json_atomic(path, data)
    else:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
def _png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    if a.shape[2] == 3:
        a = np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])
    # arrays already run top-down (row 0 = v 1), so the file is written as-is.
    # Beside the target and moved into place, as the archive is: the path is
    # the item's own (``{category}/{ident}/{i}_{name}.png``), so a re-bake the
    # guard kills mid-save used to truncate the very file the old, still valid
    # archive points at — and a truncated map reads back as no map (§7.514).
    tmp = path.with_name(path.name + ".part")
    Image.fromarray(a.astype(np.uint8), "RGBA").save(tmp, format="PNG")
    os.replace(tmp, path)


def _read_png(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGBA"), dtype=np.uint8)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.bake._read_png: from PIL import Image", exc)
        return None


def write_item(category: str, ident, pieces: list, *, label: str = "",
               source: str = "base", extra_maps: dict | None = None,
               sway: dict | None = None,
               sway_spec: list | None = None,
               implant: list | None = None,
               moves: list | None = None,
               nodes: dict | None = None,
               missing_maps: list | None = None) -> Path | None:
    """Store one resolved item.  Returns the archive path.

    *missing_maps* is what ``load_mesh.apply_list_textures`` reported while the item was resolved:
    maps its list row names that did not resolve.  Stored so the run time reports them from the
    converted item as the bundle path does (#138).  Always written, an empty list included: its
    absence is how a reader tells an archive from before it, without a format bump — the version
    is shared with the Endfield store, whose archives a bump would void too.
    """
    if not pieces:
        return None
    root = bake_dir()
    rel_npz = f"{category}/{ident}.npz"
    out = root / rel_npz
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    meta: list[dict] = []
    for i, p in enumerate(pieces):
        m = p.mesh
        arrays[f"{i}_pos"] = m.positions.astype(np.float32)
        arrays[f"{i}_idx"] = m.indices.astype(np.uint32)
        arrays[f"{i}_nrm"] = m.normals.astype(np.float32)
        arrays[f"{i}_uv"] = m.uvs.astype(np.float32)
        # Whatever streams the mesh has, verbatim.  The importer already drops
        # an all-white COLOR (see ``prefab._unorm``), so there is no filtering
        # to repeat here — and doing it in two places is what would let the
        # bake and the bundle disagree about which meshes have one.
        for s, _ in Mesh.OPTIONAL_STREAMS:
            extra = getattr(m, s, None)
            if extra is None or not m.vertex_count or len(extra) != m.vertex_count:
                continue
            arrays[f"{i}_{'col' if s == 'vcolor' else s}"] = np.asarray(
                extra, dtype=np.float32)
        if p.skin is not None:
            arrays[f"{i}_skin"] = np.asarray(p.skin, dtype=np.float32)
        if p.binds is not None:
            arrays[f"{i}_binds"] = np.asarray(p.binds, dtype=np.float32)
        if p.bone_world:
            # a prefab bone list has holes where a renderer names a bone the
            # prefab does not carry; keep them as holes rather than dropping
            # the whole track, or the indices stop lining up with ``bones``
            mats = np.zeros((len(p.bone_world), 4, 4), dtype=np.float32)
            present = np.zeros(len(p.bone_world), dtype=bool)
            for bi, bw in enumerate(p.bone_world):
                if bw is None:
                    continue
                mats[bi] = np.asarray(bw, dtype=np.float32).reshape(4, 4)
                present[bi] = True
            if present.any():
                arrays[f"{i}_bworld"] = mats
                arrays[f"{i}_bworld_ok"] = present
        rest = getattr(p, "bone_rest", None)
        if rest:
            # the same track after the prefab's DynamicBone chains were solved
            rm = np.zeros((len(rest), 4, 4), dtype=np.float32)
            rok = np.zeros(len(rest), dtype=bool)
            for bi, bw in enumerate(rest):
                if bw is None:
                    continue
                rm[bi] = np.asarray(bw, dtype=np.float32).reshape(4, 4)
                rok[bi] = True
            if rok.any():
                arrays[f"{i}_brest"] = rm
                arrays[f"{i}_brest_ok"] = rok
        # Blend shapes: one sparse (index, delta) pair per shape.  Stored as
        # two arrays per shape rather than one ragged object array so the npz
        # stays a plain numeric archive.
        shp = list(p.shapes or [])
        if shp:
            for j, (sname, sidx, sdelta, *rest) in enumerate(shp):
                arrays[f"{i}_shp{j}_i"] = np.asarray(sidx, dtype=np.int32)
                arrays[f"{i}_shp{j}_d"] = np.asarray(sdelta, dtype=np.float32)
                snrm = rest[0] if rest else None
                if snrm is not None:
                    # the frame's normal delta (§7.379); absent = none authored
                    arrays[f"{i}_shp{j}_n"] = np.asarray(snrm, dtype=np.float32)
        bmove = getattr(p, "move_bones", None)
        if bmove is not None and len(bmove) and np.any(bmove):
            # all-zero means "under neither move node", which is also what a
            # missing array reads as — no point storing it for the 96% of
            # renderers that are not skinned to anything the card can move
            arrays[f"{i}_mvbone"] = np.asarray(bmove, dtype=np.uint8)
        maps = {}
        for name in _MAP_NAMES:
            arr = getattr(p, name, None)
            if arr is None:
                continue
            rel = f"{category}/{ident}/{i}_{name}.png"
            _png(root / rel, arr)
            maps[name] = rel
        # The colour space of each map above.  Written beside ``maps`` rather
        # than inside the PNG because a PNG has nowhere to put it, and read back
        # into ``Piece.map_cs`` — a map whose flag is missing has none, which is
        # a resolution failure downstream and not a default (:mod:`colorspace`).
        cs = {k: int(v) for k, v in (getattr(p, "map_cs", None) or {}).items()
              if k in maps}
        meta.append({
            "name": getattr(m, "name", "") or "",
            "map_cs": cs,
            "bones": list(p.bones or []),
            "shader": dict(p.shader or {}),
            "tex_set": int(getattr(p, "tex_set", 0)),
            # which of the renderer's materials this pass is —
            # dropping it merged the eyelid shadow back into the lash
            "mat_slot": int(getattr(p, "mat_slot", 0)),
            "shapes": [str(s[0]) for s in shp],
            # which ``N_move`` nodes this renderer's own transform sits under
            "move": int(getattr(p, "move_mask", 0) or 0),
            "maps": maps,
        })
    extras, extras_cs = {}, {}
    for name, m in (extra_maps or {}).items():
        if m is None or m.px is None:
            continue
        rel = f"{category}/{ident}/extra_{name}.png"
        _png(root / rel, m.px)
        extras[name] = rel
        extras_cs[name] = int(m.space)
    # The chains that live on the *body's* bones, not the item's own: a skirt
    # hangs on ``cf_j_sk_*``, which the armature carries, so the solve has to
    # be written onto the skeleton at dress time rather than into the mesh.
    sway_names = sorted(sway or {})
    if sway_names:
        arrays["sway"] = np.stack(
            [np.asarray(sway[n], dtype=np.float32).reshape(3, 3)
             for n in sway_names])
    # The two nodes ``ChangeAccessory`` overwrites from the card, as (parent
    # world, node world) inside the prefab.  One pair per node so a reader can
    # form ``P · M · W⁻¹`` without re-opening the bundle.
    mv = list(moves or [])
    mv_ok = np.zeros(2, dtype=bool)
    mv_mat = np.zeros((2, 2, 4, 4), dtype=np.float32)
    for k in range(min(2, len(mv))):
        if mv[k] is None:
            continue
        mv_mat[k, 0] = np.asarray(mv[k][0], dtype=np.float32).reshape(4, 4)
        mv_mat[k, 1] = np.asarray(mv[k][1], dtype=np.float32).reshape(4, 4)
        mv_ok[k] = True
    if mv_ok.any():
        arrays["moves"] = mv_mat
        arrays["moves_ok"] = mv_ok
    # The prefab's whole node hierarchy, for the nodes nothing is skinned to.
    # A head's ``a_n_*`` attach frames live only here — they carry no vertices,
    # so they are in no ``Piece.bones`` list, and taking them off a generic rig
    # instead puts every hat 2 cm low and every hairpin 9 cm out (§7.102).
    #
    # Written as ``(parent name, local 4×4)`` rather than as prefab-space world
    # matrices: the graft rebuilds a subtree, and a local is what survives its
    # parent being shaped by the card's sliders.  The list of names is in the
    # meta so its **absence** means "baked before this existed" rather than
    # "this prefab has no nodes" — a distinction the reader acts on.
    node_names = sorted(nodes or {})
    if node_names:
        arrays["nodes"] = np.stack(
            [np.asarray(nodes[n][1], dtype=np.float32).reshape(4, 4)
             for n in node_names])
    # Everything that describes the item travels with it.  Keeping it in the
    # manifest meant rewriting the whole index once per item — fine for the 849
    # base ids, hopeless for 25,736 mod ones.
    info = {"label": label, "source": source, "category": category,
            "pieces": meta, "extras": extras, "extras_cs": extras_cs,
            "sway": sway_names}
    if sway_names or sway_spec:
        info["sway_rev"] = SWAY_REV
    # ``sway`` is the solve's *result* in the prefab's rest pose; ``sway_spec``
    # is what the solver needs to run again against the character's current
    # pose (``prefab.sway_chain_specs``).  Both are needed and they are not
    # the same thing: the first is a fixed drape, the second is why a skirt
    # gets out of the way of a raised thigh and why BetterPenetration's dan
    # capsule can open the crotch (§7.128).
    #
    # It rides in the JSON rather than as arrays, and there is **no format
    # bump**: a few hundred floats is nothing beside the meshes, and an
    # additive JSON key is invisible to a reader that predates it — every
    # existing archive keeps reading, and only the items re-baked since gain
    # the field.  Bumping would have invalidated the whole library for one
    # optional list.
    # Bones this item needs that Illusion's reference armature has not got.
    # An unknown bone is not skipped by ``skinmap`` — it is redirected to
    # ``cf_j_hips`` — so a pack that adds its own has to bring them along or
    # its geometry follows the pelvis (§7.129).
    if implant:
        info["implant"] = [dict(e) for e in implant]
    info["missing_maps"] = [str(m) for m in (missing_maps or [])]
    if sway_spec:
        info["sway_spec"] = [
            {k: (v.tolist() if hasattr(v, "tolist") else v)
             for k, v in chain.items()}
            for chain in sway_spec]
    if nodes is not None:
        info["nodes"] = [[n, nodes[n][0]] for n in node_names]
    arrays["meta"] = np.frombuffer(
        json.dumps(info, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    # Written beside the archive and moved into place: a bake killed mid-write
    # (the guard's memory floor does that) used to leave a truncated ``.npz``
    # that every later reader, the rebake selectors included, skipped as
    # unreadable (§7.446).
    tmp = out.with_name(out.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, out)
    data = manifest()
    data["items"][key_of(category, ident)] = {"file": rel_npz}
    _save_manifest(data)
    return out


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
@dataclass
class BakedItem:
    label: str
    source: str
    pieces: list
    category: str = ""
    maps: dict = field(default_factory=dict)
    sway: dict = field(default_factory=dict)   # bone name → 3×3 local rotation
    #: Per-chain solver parameters, for running ``dynbone`` at run time against
    #: the *current* pose.  Empty for an archive baked before the field existed,
    #: which the callers treat as "no per-frame chains here" — the same state
    #: they were already in.
    sway_spec: list = field(default_factory=list)
    #: ``{name, parent, local}`` for bones the reference rig lacks; empty for
    #: every stock body, which is why an archive without it behaves as before.
    implant: list = field(default_factory=list)
    # ``N_move`` / ``N_move2`` as (parent world, node world), or None per node
    moves: list = field(default_factory=lambda: [None, None])
    # name → (parent name, local 4×4) for the prefab's whole hierarchy.
    # ``None`` means this archive predates the table, which is not the same as
    # an empty one and is why the caller can tell "not baked" from "no nodes".
    nodes: dict | None = None
    #: maps the item's list row names that did not resolve when it was converted (#138); ``None``
    #: for an archive written before the field, which cannot say (not the same as an empty list)
    missing_maps: list | None = None


def _bone_world(arch, i: int, stem: str = "bworld"):
    if f"{i}_{stem}" not in arch:
        return None
    mats = arch[f"{i}_{stem}"]
    ok = (arch[f"{i}_{stem}_ok"] if f"{i}_{stem}_ok" in arch
          else np.ones(len(mats), dtype=bool))
    return [mats[k] if ok[k] else None for k in range(len(mats))]


def _nodes_from(arch, entry) -> dict | None:
    """``meta['nodes']`` + the matching array, or None for an older archive."""
    rows = entry.get("nodes")
    if rows is None:
        return None
    mats = arch["nodes"] if "nodes" in arch else []
    return {str(n): (str(p), np.asarray(mats[k], dtype=np.float64))
            for k, (n, p) in enumerate(rows) if k < len(mats)}


def read_nodes(category: str, ident) -> dict | None:
    """Just the node hierarchy of a baked item — no meshes, no maps.

    The graft needs a head's attach frames *before* the head's meshes are
    loaded, and reading the whole archive for a hundred 4×4s would pull tens of
    megabytes of vertex data through for nothing.  ``np.load`` on an ``.npz``
    is lazy per array, so this touches two of them.
    """
    entry = manifest()["items"].get(key_of(category, ident))
    if not entry:
        return None
    path = bake_dir() / entry["file"]
    if not path.exists():
        return None
    try:
        arch = np.load(path)
    except (OSError, ValueError):
        return None
    with arch:
        if "meta" not in arch:
            return None
        try:
            meta = json.loads(bytes(arch["meta"]).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return _nodes_from(arch, meta)


def read_item(category: str, ident) -> BakedItem | None:
    entry = manifest()["items"].get(key_of(category, ident))
    if not entry:
        return None
    path = bake_dir() / entry["file"]
    if not path.exists():
        return None
    try:
        arch = np.load(path)
    except (OSError, ValueError):
        return None
    root = bake_dir()
    pieces = []
    sway: dict = {}
    moves: list = [None, None]
    nodes: dict | None = None
    with arch:
        if "meta" not in arch:
            return None                  # written by a format we do not read
        try:
            entry = json.loads(bytes(arch["meta"]).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        nodes = _nodes_from(arch, entry)
        sway_spec = list(entry.get("sway_spec") or [])
        implant = list(entry.get("implant") or [])
        names = list(entry.get("sway") or [])
        if names and "sway" in arch:
            rots = arch["sway"]
            sway = {n: np.asarray(rots[k], dtype=np.float64)
                    for k, n in enumerate(names) if k < len(rots)}
        if "moves" in arch and "moves_ok" in arch:
            mv, ok = arch["moves"], arch["moves_ok"]
            moves = [(np.asarray(mv[k, 0], dtype=np.float64),
                      np.asarray(mv[k, 1], dtype=np.float64))
                     if k < len(ok) and ok[k] else None for k in range(2)]
        for i, meta in enumerate(entry.get("pieces") or []):
            # Every dress stage comes through here and each piece decodes
            # several PNGs, so this is where a stage with many pieces would
            # otherwise sit silent (see .progress).  Not where the body's 7 s
            # went, though — that was the skin composite, and adding a tick
            # here first changed nothing at all.
            PROG.tick()
            if f"{i}_pos" not in arch:
                continue
            mesh = Mesh.from_arrays(
                arch[f"{i}_pos"], arch[f"{i}_idx"],
                normals=arch.get(f"{i}_nrm"), uvs=arch.get(f"{i}_uv"),
                name=meta.get("name") or f"{category}_{ident}_{i}")
            for s, _ in Mesh.OPTIONAL_STREAMS:
                key = f"{i}_{'col' if s == 'vcolor' else s}"
                if key in arch:
                    setattr(mesh, s, np.asarray(arch[key], dtype=np.float32))
            maps = meta.get("maps") or {}
            pieces.append(Piece(
                mesh=mesh,
                bones=list(meta.get("bones") or []),
                skin=arch[f"{i}_skin"] if f"{i}_skin" in arch else None,
                binds=arch[f"{i}_binds"] if f"{i}_binds" in arch else None,
                texture=_read_png(root / maps["texture"]) if "texture" in maps else None,
                mask=_read_png(root / maps["mask"]) if "mask" in maps else None,
                bone_world=_bone_world(arch, i),
                detail=_read_png(root / maps["detail"]) if "detail" in maps else None,
                shader=dict(meta.get("shader") or {}),
                tex_set=int(meta.get("tex_set", 0)),
                mat_slot=int(meta.get("mat_slot", 0)),
                normal=_read_png(root / maps["normal"]) if "normal" in maps else None,
                alpha_mask=(_read_png(root / maps["alpha_mask"])
                            if "alpha_mask" in maps else None),
                ramp=_read_png(root / maps["ramp"]) if "ramp" in maps else None,
                line_mask=(_read_png(root / maps["line_mask"])
                           if "line_mask" in maps else None),
                glass_ramp=(_read_png(root / maps["glass_ramp"])
                            if "glass_ramp" in maps else None),
                bone_rest=_bone_world(arch, i, "brest"),
                shapes=[(n, arch[f"{i}_shp{j}_i"], arch[f"{i}_shp{j}_d"],
                         arch[f"{i}_shp{j}_n"] if f"{i}_shp{j}_n" in arch else None)
                        for j, n in enumerate(meta.get("shapes") or [])
                        if f"{i}_shp{j}_i" in arch] or None,
                move_mask=int(meta.get("move", 0) or 0),
                move_bones=(np.asarray(arch[f"{i}_mvbone"], dtype=np.uint8)
                            if f"{i}_mvbone" in arch else None),
                map_cs={k: int(v) for k, v in
                        (meta.get("map_cs") or {}).items()},
            ))
    if not pieces:
        return None
    ecs = entry.get("extras_cs") or {}
    maps = {k: CS.pair(_read_png(root / rel), int(ecs[k]) == CS.SRGB)
            for k, rel in (entry.get("extras") or {}).items() if k in ecs}
    return BakedItem(label=entry.get("label") or str(ident),
                     source=entry.get("source") or "baked",
                     pieces=pieces, category=category,
                     maps={k: v for k, v in maps.items() if v is not None},
                     sway=sway, sway_spec=sway_spec, implant=implant,
                     moves=moves, nodes=nodes,
                     missing_maps=(None if entry.get("missing_maps") is None
                                   else list(entry["missing_maps"])))


def write_texture(category: str, ident, tex) -> Path | None:
    """Bake one entry of a texture-only list (mt_eye, mt_cheek, mt_nip …).

    These have no mesh, so they skip the ``.npz`` entirely — the PNG *is* the
    item.  Worth doing: a card touches eight or nine of these lists, each in a
    different bundle, and opening a bundle costs about 1.2 s the first time.

    *tex* is a :class:`colorspace.Map`; its ``m_ColorSpace`` goes in the
    manifest row because the PNG cannot carry it, and these maps are not
    incidental to the flag — ``mt_eye`` is linear while ``mt_eye_gradation`` /
    ``mt_eye_white`` / ``mt_eye_hi_*`` beside it are sRGB, and both halves meet
    inside ``create_eye``.
    """
    if tex is None or tex.px is None:
        return None
    key = key_of(category, ident)
    rel = f"tex/{category}/{int(ident)}.png"
    _png(bake_dir() / rel, np.asarray(tex.px))
    data = manifest()
    data["items"][key] = {"file": rel, "kind": "texture",
                          "cs": int(tex.space),
                          "label": f"{category}#{ident}"}
    _save_manifest(data)
    return bake_dir() / rel


#: The ``source`` of a named map that lives in the install's ``abdata``.
INSTALL = "abdata"


def named_key(source: str, bundle: str, name: str) -> str:
    """Manifest key of a map a list row names by ``(bundle, texture)``.

    *source* is where the bundle lives — :data:`INSTALL` or a zipmod's GUID —
    because the same ``(bundle, name)`` can be two different images: a mod may
    ship its own ``chara/mt_mask_body_00.unity3d`` inside its archive.  The
    bundle is lower-cased (the install's file names are, and a row may not
    spell it that way); the texture name is not — ``texture_by_name`` matches it
    exactly.
    """
    return f"named/{source}/{_bundle_key(bundle)}/{name}"


def _bundle_key(bundle: str) -> str:
    return str(bundle).replace("\\", "/").strip().lower()


def _path_safe(text: str) -> str:
    return "".join("_" if c in '<>:"/\\|?*' else c for c in str(text))


def _named_file(source: str, bundle: str, name: str) -> str:
    stem = _bundle_key(bundle)
    if stem.endswith(".unity3d"):
        stem = stem[:-len(".unity3d")]
    return f"tex/named/{_path_safe(source)}/{stem}/{_path_safe(name)}.png"


def write_named_texture(source: str, bundle: str, name: str, tex) -> Path | None:
    """Bake one map a list row names by ``(bundle, texture)`` rather than by id.

    The texture-only lists are baked by id (:func:`write_texture`) and a
    garment's own albedo and colour mask go into the item.  What neither covers
    is a **column the run time reads off the row** — a top's
    ``OverBodyMask`` / ``OverBraMask``, a jacket's ``OverInnerMask``, a lash
    row's ``EyelineShadowTex`` — and those opened their bundle on every card
    that wore one (§7.473).  The wrap mode is kept beside the colour space: the
    bra mask is sampled off the unit square (§7.367), and past the decode the
    mode is otherwise gone.
    """
    if tex is None or tex.px is None:
        return None
    rel = _named_file(source, bundle, name)
    _png(bake_dir() / rel, np.asarray(tex.px))
    data = manifest()
    data["items"][named_key(source, bundle, name)] = {
        "file": rel, "kind": "named", "cs": int(tex.space),
        "wrap": None if tex.wrap is None else int(tex.wrap),
        "label": f"{source}:{bundle}/{name}"}
    _save_manifest(data)
    return bake_dir() / rel


def write_named_absent(source: str, bundle: str, name: str) -> None:
    """Record that *source* does **not** have ``(bundle, name)``.

    A zipmod row may name a map its archive lacks, and the run time then takes
    the install's copy (``_ref_texture``'s fallback).  Without this record the
    only way to learn "not in the archive" is to open the archive, which is
    the read the bake exists to remove (§7.474).
    """
    data = manifest()
    data["items"][named_key(source, bundle, name)] = {
        "kind": "named", "absent": True,
        "label": f"{source}:{bundle}/{name} (not there)"}
    _save_manifest(data)


def named_entry(source: str, bundle: str, name: str) -> dict | None:
    """The manifest row of a named map, or ``None`` when nothing was recorded.

    A row with ``absent`` is an answer (:func:`write_named_absent`); any other
    row counts only when its file is on disk.
    """
    entry = manifest()["items"].get(named_key(source, bundle, name))
    if not entry or entry.get("kind") != "named":
        return None
    if entry.get("absent"):
        return entry
    if "cs" not in entry:
        return None
    return entry if (bake_dir() / entry["file"]).exists() else None


def read_named_texture(source: str, bundle: str, name: str):
    """:func:`write_named_texture`'s map back, as a ``colorspace.Map``.

    ``None`` both when nothing was converted and when the source is recorded
    as not having the map — :func:`named_entry` tells the two apart.
    """
    entry = named_entry(source, bundle, name)
    if entry is None or entry.get("absent"):
        return None
    px = _read_png(bake_dir() / entry["file"])
    if px is None:
        return None
    wrap = entry.get("wrap")
    return CS.Map(px, int(entry["cs"]) == CS.SRGB,
                  None if wrap is None else int(wrap))


def table_key(source: str, bundle: str, name: str) -> str:
    """Manifest key of a parsed table (a TextAsset, a material's saved properties).

    Keyed like :func:`named_key` — the same ``(bundle, name)`` can come from two
    sources — but in its own namespace: a bundle may hold a texture and a
    TextAsset of the same name.
    """
    return f"table/{source}/{_bundle_key(bundle)}/{name}"


def write_named_table(source: str, bundle: str, name: str, payload: dict | None) -> None:
    """Bake one table the run time reads out of a bundle, **already parsed**.

    The shape-curve tables, the slider → curve list, the bone-type offsets and
    the create materials' saved properties were each read out of their bundle
    once per process (``runtime_reads``, §7.482).  *payload* is JSON — what the
    loader's own ``to_payload`` makes of its parsed value — so the run time
    reads our format and not the TextAsset's; ``None`` records that *source*
    has no such asset, which is an answer too.
    """
    data = manifest()
    key = table_key(source, bundle, name)
    if payload is None:
        data["items"][key] = {"kind": "table", "absent": True,
                              "label": f"{source}:{bundle}/{name} (not there)"}
    else:
        stem = _bundle_key(bundle)
        if stem.endswith(".unity3d"):
            stem = stem[:-len(".unity3d")]
        rel = f"table/{_path_safe(source)}/{stem}/{_path_safe(name)}.json"
        write_json_atomic(bake_dir() / rel, payload)
        data["items"][key] = {"kind": "table", "file": rel,
                              "label": f"{source}:{bundle}/{name}"}
    _save_manifest(data)


def drop_tables(live: set) -> list:
    """Forget every table whose key is not in *live*; returns the keys dropped.

    A table is keyed by where it was read from, so a reader that starts asking
    a different source (a mod head's material, §7.484) leaves the old entry
    behind — never read again, and looking like a live answer to anyone who
    lists the bake.  Its file goes with it.
    """
    data = manifest()
    gone = [k for k, v in data["items"].items()
            if v.get("kind") == "table" and k not in live]
    for key in gone:
        rel = data["items"].pop(key).get("file")
        if rel:
            try:
                (bake_dir() / rel).unlink()
            except FileNotFoundError:
                pass
    if gone:
        _save_manifest(data)
    return gone


def named_table(source: str, bundle: str, name: str) -> tuple[bool, dict | None]:
    """``(recorded, payload)`` — :func:`write_named_table`'s answer back.

    ``recorded`` is False when the converter never wrote this table (the
    caller then reads the bundle as before); True with ``None`` is "the source
    does not have it".
    """
    entry = manifest()["items"].get(table_key(source, bundle, name))
    if not entry or entry.get("kind") != "table":
        return False, None
    if entry.get("absent"):
        return True, None
    path = bake_dir() / entry["file"]
    try:
        return True, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        swallowed.note("origin.bake.named_table", exc)
        return False, None


_TEXTURE_PROVIDER = None
_PROVIDER_SETTLED = False

#: ``category -> (count, last message)`` for providers that **raised**.
#:
#: Read it in any tool whose conclusion is about the generated sheets.  Empty is
#: the only reading under which such a run says anything: the fallback is the
#: converted sheet, so a generator that throws still renders a correct-looking
#: character (§7.181).
PROVIDER_ERRORS: dict = {}


def provider_errors() -> dict:
    """A copy of :data:`PROVIDER_ERRORS`."""
    return dict(PROVIDER_ERRORS)


def clear_provider_errors() -> None:
    PROVIDER_ERRORS.clear()


def _install_default_provider() -> None:
    """Pick a texture provider from the setting, once, on first use.

    Without this the generated sheets were something only a tool could turn on,
    so running the actual game showed no change at all — which is what the
    first report of "the character's textures did not change" was.

    ``startup.own_face_tex``:

    * ``auto`` (default) — use the generated sheets only when the bake has no
      face sheets to offer, i.e. on a build that ships none of Illusion's.
      That is the case ``PROGRESS.md`` F0 exists for.
    * ``always`` — use them even when the bake could answer.  This is how to
      look at them in the running game.
    * ``never`` — off.

    ``auto`` rather than ``always`` because which art a build ships is a
    product decision, not one this module gets to make (``CLAUDE.md`` 七).

    **Two halves, one seam.**  The face sheets come from
    :mod:`polylode.mesh.facetex` under ``own_face_tex``; the garment pattern
    tiles, the hair gloss band, the nose line and the under-hair patch come
    from :mod:`polylode.mesh.parttex` under ``own_part_tex``, which is the
    same switch its piece maps use.  They are chained here rather than given a
    provider slot each, because the caller can only install one and two slots
    would drift (§7.143 五).
    """
    global _TEXTURE_PROVIDER, _PROVIDER_SETTLED
    if _PROVIDER_SETTLED:
        return
    _PROVIDER_SETTLED = True
    chain = []
    if _own_face_mode() != "off":
        # **``Config`` is not a module-level name here.**  It was used inside
        # this block without being imported, so the line raised ``NameError``
        # every single time — and the ``except Exception: pass`` below swallowed
        # it, so ``facetex.provide`` was **never appended to the chain**.  The
        # generated face sheets have therefore not reached a render since
        # §7.143 十三 wired them up: every `own_face_tex=always` run, including
        # this section's own first regression, was quietly drawing Origin's.
        #
        # It is the shape ``CLAUDE.md`` 七 keeps recording — "before measuring
        # that A and B are identical under two settings, check the setting
        # reaches A" (§7.83 三).  Two 30-card audits came out condition-for-
        # condition identical and the honest reading was that nothing was
        # switched on, not that the change was neutral.
        #
        # The import is inside the block rather than at module level to match
        # ``_own_face_mode``; what it must not be is absent.
        try:
            from ..core.paths import Config
            from ..mesh import facetex
            # Which sheets we decline to draw is a per-machine preference:
            # a machine that has converted its own copy can use the real lash
            # line, and one that has not still gets ours (facetex._DEFER_TO_BAKE).
            got = Config("startup").get("own_face_tex_defer", None)
            if got is not None:
                facetex.defer_to_bake(got)
            chain.append(facetex.provide)
        except Exception as exc:                             # noqa: BLE001
            swallowed.note("origin.bake._install_default_provider: Config", exc)
    try:
        from ..mesh import parttex
        if parttex.enabled():
            chain.append(parttex.provide_sheet)
    except Exception as exc:                                 # noqa: BLE001
        swallowed.note("origin.bake._install_default_provider: parttex.enabled", exc)
    if not chain:
        return

    def provide(category, ident):
        for fn in chain:
            got = fn(category, ident)
            if got is not None:
                return got
        return None

    _TEXTURE_PROVIDER = provide


def _own_face_mode() -> str:
    """``"on"`` / ``"off"`` for the face half of the generated sheets.

    ``ISLANDLIFE_OWN_FACE_TEX`` overrides the setting, matching the
    ``ISLANDLIFE_OWN_PART_TEX`` the garment half has had since §7.144.  Without
    it the only way to run a regression against the generated sheets was to
    edit ``startup.own_face_tex`` in the user's own config — which then has to
    be put back, and the run that forgets leaves the product on a different
    default.  A check that can only be run by changing the thing under test is
    a check that will not be run.
    """
    import os
    mode = os.environ.get("ISLANDLIFE_OWN_FACE_TEX") or ""
    if not mode:
        try:
            from ..core.paths import Config
            mode = str(Config("startup").get("own_face_tex", "auto") or "auto")
        except Exception as exc:                             # noqa: BLE001
            swallowed.note("origin.bake._own_face_mode: from ..core.paths import Config", exc)
            mode = "auto"
    if mode == "never":
        return "off"
    if mode == "always":
        return "on"
    try:                         # auto: the bake can answer -> leave it to it
        items = manifest()["items"]
        if any(k.split("/", 1)[0] in ("eyebrow", "eye") for k in items):
            return "off"
    except Exception as exc:                                 # noqa: BLE001
        swallowed.note("origin.bake._own_face_mode: manifest", exc)
    return "on"


def set_texture_provider(fn) -> None:
    """Install a source of **generated** sheets, consulted before the bake.

    This is the seam ``PROGRESS.md`` F0 needs: a build that ships none of
    Illusion's textures still has to put a face on a character, and
    :func:`polylode.mesh.facetex.provide` draws one.  It sits here rather
    than in each of ``read_texture``'s three callers so there is one switch,
    not three that can drift apart.

    Pass ``None`` to remove it.  A provider that returns ``None`` for a
    category falls through to the bake, so installing one is additive.
    """
    global _TEXTURE_PROVIDER, _PROVIDER_SETTLED
    _TEXTURE_PROVIDER = fn
    _PROVIDER_SETTLED = True     # an explicit choice, including ``None``


def read_texture(category: str, ident):
    """The map for a texture-only list entry as a ``colorspace.Map``.

    The texture provider (:func:`provided_texture`) answers first, then the
    converted sheet (:func:`baked_texture`).
    """
    got = provided_texture(category, ident)
    return got if got is not None else baked_texture(category, ident)


def provided_texture(category: str, ident):
    """The texture provider's answer alone, or ``None`` when it declines.

    Split from :func:`read_texture` for the one caller that has something to
    ask **between** the two: a head's own eyeline variant is the game's answer
    for that row, so it beats the converted stock sheet — and loses to a
    provider the setting chose, which replaces the game's answers (§7.517).
    """
    if not _PROVIDER_SETTLED:
        _install_default_provider()
    got = None
    if _TEXTURE_PROVIDER is not None:
        try:
            got = _TEXTURE_PROVIDER(category, ident)
        except Exception as exc:                             # noqa: BLE001
            # **A provider that raised is not a provider that declined.**  The
            # fallback below is the converted sheet, which looks *right*, so a
            # broken generator produced a perfect render and every ruler said
            # fine: with ``facetex.iris`` and ``facetex.eyeline_shadow`` made to
            # raise, ``origin_audit --count 12 ISLANDLIFE_OWN_FACE_TEX=always``
            # still reported 12/12, 0 crashes and the same three missing
            # reasons — and four sections had cited that run as evidence that
            # the generated path "is really walked" (§7.181).
            # Swallowing stays: one bad row must not brick a character.  What
            # changes is that it is now countable, the same answer
            # ``TextureCache.evicted`` got — a run with a non-zero count is not
            # evidence about the generators.
            PROVIDER_ERRORS[str(category)] = (
                PROVIDER_ERRORS.get(str(category), (0, ""))[0] + 1,
                f"{type(exc).__name__}: {exc}")
            got = None
    return got


def baked_texture(category: str, ident):
    """The converted sheet for a texture-only list entry, or ``None``.

    A row with no ``cs`` is **not** taken: it was written by a format that had
    no colour space, and the caller re-resolving from the bundle is correct
    where inventing one is not.  (In practice the version gate already discards
    those rows; this is the belt to its braces.)
    """
    try:
        entry = manifest()["items"].get(key_of(category, int(ident)))
    except (TypeError, ValueError):
        return None
    if not entry or entry.get("kind") != "texture" or "cs" not in entry:
        return None
    px = _read_png(bake_dir() / entry["file"])
    return None if px is None else CS.Map(px, int(entry["cs"]) == CS.SRGB)


def stats() -> dict:
    items = manifest()["items"]
    by_cat: dict[str, int] = {}
    for key in items:
        by_cat[key.split("/", 1)[0]] = by_cat.get(key.split("/", 1)[0], 0) + 1
    size = 0
    root = bake_dir()
    if root.is_dir():
        size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    return {"items": len(items), "categories": by_cat,
            "mb": round(size / (1024 * 1024), 1), "dir": str(root)}
