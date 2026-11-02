"""Load every mesh piece from a Unity prefab under assets_cache or a zipmod."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..install import configured
from ..prefab import PrefabLoader, Piece
from .. import colorspace as CS
from .. import sideloader as SL
from ..apply import is_explicit_overlay
from ...core import swallowed


@dataclass
class LoadedItem:
    label: str
    source: str                 # base | zipmod
    pieces: list                # list[Piece]
    category: str = ""
    # bone name -> the rotation the prefab's DynamicBone solve added, for
    # chains whose bones the body armature shares (a skirt's ``cf_j_sk_*``).
    # Those pieces retarget onto our skeleton, so the settled shape has to be
    # pushed onto the bones rather than baked into the piece.
    sway: dict = field(default_factory=dict)
    # Those chains' solver parameters (``prefab.sway_chain_specs``): what the
    # run time needs to solve them again against the current pose, so a raised
    # thigh moves the skirt the way the game's ``DynamicBone`` does (§7.540).
    sway_spec: list = field(default_factory=list)
    # ``[(parent_world, node_world) | None, …]`` for ``N_move`` and ``N_move2``
    # (:meth:`prefab.PrefabLoader.move_chain`).  The card's ``addMove``
    # overwrites those two nodes' local TRS, so this is what turns the card's
    # numbers into a placement — it belongs to the item, not to the card.
    moves: list = field(default_factory=lambda: [None, None])
    # what ``apply_list_textures`` reported when the item was converted: a map its list row names
    # that did not resolve.  The converted item has none of those maps, and a reader that did not
    # hand these on would draw it as if nothing were missing (#138)
    missing_maps: list | None = None


def _hand_on_missing(hit: "LoadedItem", missing: list | None) -> None:
    """Report a converted item's unresolved maps (#138) — or that its archive predates the record."""
    if missing is None:
        return
    if hit.missing_maps is None:
        missing.append(f"{hit.label}: 轉出物比解析失敗的記錄早，不知道有沒有缺貼圖（重轉這一件）")
    else:
        missing.extend(hit.missing_maps)


def _release(loader: PrefabLoader | None) -> None:
    if loader is None:
        return
    loader.env = None
    loader.by_id = {}


def _open_bundle(bundle_rel: str, asset: str | None = None) -> PrefabLoader | None:
    """The bundle a stock list row's request is answered from.

    Without *asset* it is the install's file.  With it, a path something else
    can answer at (``SL.overrides_at``: a mod registered it, or ships loose
    PNGs in its folder) goes through Sideloader's rule — the first source
    that contains the asset — because the game's request for a stock row's
    asset is ``RedirectHook``'s like any other (§7.486).  Every other path
    keeps its one install read.
    """
    rel = bundle_rel.replace("\\", "/")
    if asset and SL.overrides_at(rel):
        return _open_resolved(rel, asset)
    inst = configured()
    if not inst.valid:
        return None
    path = inst.bundle(rel)
    if not path.exists():
        return None
    try:
        return PrefabLoader(path)
    except Exception as exc:  # noqa: BLE001
        from ..meshimport import note_load_failure
        note_load_failure(bundle_rel, exc)
        return None


def _open_resolved(bundle_rel: str, asset: str) -> PrefabLoader | None:
    """A loader over the source ``SL.open_asset_bundle`` picks for *asset*, or ``None``."""
    env = SL.open_asset_bundle(bundle_rel, asset)
    if env is None:
        return None
    try:
        return PrefabLoader(Path(bundle_rel), env=env)
    except Exception as exc:  # noqa: BLE001
        from ..meshimport import note_load_failure
        note_load_failure(bundle_rel, exc)
        return None


def _stock_answer(bundle_rel: str, name: str, *, read, loose, swallow: str | None = None):
    """What the game hands out for stock map *name* of *bundle_rel*, read off by *read*.

    Where only the bundle can answer, its install file (one open for the name
    and its ``_low`` twin).  Where something else can (``SL.overrides_at``),
    each name in turn: a loose PNG first (``GetPng``, read by *loose*), then the
    first source that contains it — the Texture Fixes zipmod's copy of a stock
    bra map wins over the install's (§7.486).  *read(loader, *names)* is the
    loader call; *swallow* names it when an exception out of the install read
    is to be recorded and answered with ``None`` rather than raised.
    """
    rel = bundle_rel.replace("\\", "/")
    if not SL.overrides_at(rel):
        loader = _open_bundle(rel)
        if loader is None:
            return None
        try:
            return read(loader, name, f"{name}_low")
        except Exception as exc:  # noqa: BLE001
            if swallow is None:
                raise
            swallowed.note(f"origin.figure.load_mesh._stock_answer: {swallow}", exc)
            return None
        finally:
            _release(loader)
    for want in (name, f"{name}_low"):
        raw = SL.png_bytes(rel, want)
        if raw:
            return loose(rel, want, raw)
        loader = _open_resolved(rel, want)
        if loader is None:
            continue
        try:
            got = read(loader, want)
        finally:
            _release(loader)
        if got is not None:
            return got
    return None


def _stock_texture(bundle_rel: str, name: str):
    """A map a stock row names, resolved the way the game resolves it (:func:`_stock_answer`)."""
    return _stock_answer(bundle_rel, name,
                         read=lambda loader, *names: loader.texture_by_name(*names),
                         loose=lambda rel, want, raw: decode_image_bytes(raw),
                         swallow="texture_by_name")


def _stock_texture_wrap(bundle_rel: str, name: str) -> int | None:
    """``m_WrapMode`` of :func:`_stock_texture`'s answer (a loose PNG's is ``GetPng``'s)."""
    return _stock_answer(bundle_rel, name,
                         read=lambda loader, *names: loader.texture_wrap_by_name(*names),
                         loose=lambda rel, want, raw: _loose_png_wrap(rel, want))


def _open_zipmod(mref) -> PrefabLoader | None:
    z = SL.resolve_zipmod_path(mref.zipmod)
    if z is None:
        return None
    mref.zipmod = str(z)
    env = SL.load_mod_bundle(mref)
    if env is None:
        return None
    return PrefabLoader(z, env=env)


# ``LoadCharaFbxDataAsync``'s ``copyDynamicBone`` argument, read off its four
# call sites: ``<ChangeClothes*Async>`` pass **true**, ``<ChangeHairAsync>``,
# ``<ChangeHeadAsync>`` and ``<ChangeAccessoryAsync>`` pass **false**.  It is
# the only thing in the game that fills a ``DynamicBone``'s ``m_Colliders``
# (:meth:`prefab.PrefabLoader._colliders_for`), so it decides whether a chain
# is solved against the body's colliders or against nothing.  Keyed off the
# slot rather than the prefab, because that is what picks the loader.
# A combo top's three sub-parts are loaded by ``<ChangeClothesTopPartsAsync>``,
# whose ``LoadCharaFbxDataAsync`` call pushes ``1`` for ``copyDynamicBone``
# (argument 6) and ``1`` for ``copyWeights`` — the garment loaders' values —
# so the six ``cpo_*`` lists are garments here too (§7.477).
_COLLIDER_CATEGORIES = frozenset((
    "top", "bottom", "bra", "panty", "gloves",
    "legwear", "socks", "shoes", "shoes_inner", "shoes_outer",
    "cpo_sailor_a", "cpo_sailor_b", "cpo_sailor_c",
    "cpo_jacket_a", "cpo_jacket_b", "cpo_jacket_c",
))

#: ``MainData`` values that name no mesh: Illusion ships 「なし」 as a real list
#: row pointing at ``p_dummy`` (an empty GameObject), and mods do the same.
EMPTY_ASSETS = frozenset({"p_dummy", "", "0", "-", "none"})


def names_nothing(ref) -> bool:
    """Does this list row's asset draw nothing (a 「なし」 row)?"""
    return str(getattr(ref, "asset", "") or "").strip().lower() in EMPTY_ASSETS


def copies_colliders(category: str) -> bool:
    """Does the game hand this slot the body's colliders?"""
    return str(category or "") in _COLLIDER_CATEGORIES


def _pieces_from_loader(loader: PrefabLoader, asset: str, *,
                        copy_dynamic_bone: bool = True) -> list[Piece]:
    """The pieces of the prefab the row names, as ``AssetBundle.LoadAsset`` finds it.

    Only *asset* is asked (:meth:`PrefabLoader.find_prefab`, the container):
    the ``p_`` variants and the GameObject-name match that followed it found
    nothing the game would (§7.498).  A prefab the container does not list is
    an empty result, which the caller reports.  Every node is read, as Origin
    does: the node-name list this used to apply dropped nothing but renderers
    Unity draws (``_low`` in ``lowson``, ``_col`` in ``collar``, §7.500).
    """
    if not asset or loader is None:
        return []
    loader.copy_dynamic_bone = bool(copy_dynamic_bone)
    try:
        got = loader.build_skinned(asset)
    except Exception as exc:  # noqa: BLE001
        # counted: an item that builds nothing is baked as "no pieces",
        # and that used to be all anyone saw of a decode error (§7.494)
        swallowed.note("origin.figure.load_mesh._pieces_from_loader: build_skinned", exc)
        got = []
    return _filter_pieces(got)


def _filter_pieces(pieces: list) -> list:
    """Drop explicit-overlay / empty renderers that cards sometimes embed."""
    out = []
    for p in pieces:
        if p.mesh is None or p.mesh.vertex_count == 0:
            continue
        if is_explicit_overlay(getattr(p.mesh, "name", None) or ""):
            continue
        out.append(p)
    return out


_TEX_CACHE: dict[tuple, "CS.Map | None"] = {}


def _baked_named(bundle_rel: str, name: str):
    """The converted copy of a map in an install bundle, or ``None``."""
    from .. import bake
    try:
        return bake.read_named_texture(bake.INSTALL, bundle_rel, name)
    except (OSError, ValueError, KeyError) as exc:
        swallowed.note("origin.figure.load_mesh._baked_named", exc)
        return None


def _mod_named(mref, bundle_rel: str, name: str) -> tuple[bool, "CS.Map | None"]:
    """``(known, map)`` — what the bake recorded for a map a zipmod row names.

    ``known`` is False when the converter never looked (the caller then opens
    the archive as before); True with ``None`` is "the archive does not have
    it" (``bake.write_named_absent``), which the caller answers without opening
    anything (§7.474).
    """
    from .. import bake
    guid = str(getattr(mref, "guid", "") or "")
    if not guid:
        return False, None
    entry = bake.named_entry(guid, bundle_rel, name)
    if entry is None:
        return False, None
    if entry.get("absent"):
        return True, None
    got = bake.read_named_texture(guid, bundle_rel, name)
    return got is not None, got


def _named_texture(bundle_rel: str, name: str, *, from_bundle: bool = False):
    """A Texture2D the list row names outright, from the installed bundles.

    Returns a :class:`colorspace.Map`: the bytes and the asset's own
    ``m_ColorSpace``.  The flag has to come back with the pixels because this is
    the last point that has it — see :mod:`colorspace`.

    The converted copy answers first (``bake.read_named_texture``, 四之一): the
    columns a list row is read for at run time — the garment masks, the lid
    shadow — are baked by ``convert.bake_row_maps`` (§7.473).  *from_bundle*
    skips it, for the converter itself: a re-conversion has to read the bundle,
    not the copy it is about to replace.
    """
    key = (bundle_rel, name)
    if not from_bundle and key in _TEX_CACHE:
        return _TEX_CACHE[key]
    got = None if from_bundle else _baked_named(bundle_rel, name)
    if got is not None:
        _TEX_CACHE[key] = got
        return got
    got = _stock_texture(bundle_rel, name)
    if not from_bundle:          # the converter reads each map once (§7.474)
        _TEX_CACHE[key] = got
    return got


def _loose_png(mref):
    """``abdata/<MainAB>/<asset>.png`` inside the mod's archive, or ``None``.

    A loose PNG carries no ``m_ColorSpace`` — it is not a Unity asset.
    Sideloader decides for it: ``Sideloader.GetPng`` builds
    ``new Texture2D(2, 2, format, mipmaps)``, the **four**-argument overload, so
    the ``linear`` parameter of the five-argument one is never supplied and
    defaults to false.  These are sRGB.
    """
    raw = SL.load_mod_file(mref)
    if not raw:
        return None
    return decode_image_bytes(raw)


def decode_image_bytes(raw: bytes):
    """An image file's bytes as an sRGB :class:`colorspace.Map`, or ``None``.

    For images a plugin builds with ``new Texture2D(2, 2, format, mipmaps)``
    and ``LoadImage`` — the four-argument overload, so sRGB (Sideloader's loose
    PNGs above, MaterialEditor's ``TextureDictionary``).  PNG and WebP both
    decode; a PNG whose only fault is a damaged ancillary chunk is read the way
    libpng reads it (:func:`_drop_bad_ancillary`).
    """
    if not raw:
        return None
    import io
    from PIL import Image
    last = None
    for data in (raw, _drop_bad_ancillary(raw)):
        if data is None:
            continue
        try:
            with Image.open(io.BytesIO(data)) as im:
                return CS.Map(np.asarray(im.convert("RGBA"), dtype=np.uint8),
                              True)
        except Exception as exc:  # noqa: BLE001  (PIL raises OSError, SyntaxError, ValueError…)
            last = exc                           # the repaired copy may still read
    if last is not None:
        swallowed.note("origin.figure.load_mesh.decode_image_bytes: Image.open", last)
    return None


def _drop_bad_ancillary(raw: bytes) -> bytes | None:
    """The PNG with every **ancillary** chunk whose CRC fails left out.

    PIL rejects the whole file on any chunk CRC error.  The PNG specification
    (§13.2) lets a decoder ignore a damaged ancillary chunk — it cannot change
    the pixels — and libpng's default does exactly that: warn and discard for
    ancillary chunks, error for critical ones.  A local lash mod
    (``[Augh] Starou Eyes``) ships its lid shadow with a corrupt ``iCCP`` and
    nothing else wrong.  A bad **critical** chunk (``IHDR``/``PLTE``/``IDAT``/
    ``IEND``) still fails: ``None`` is returned and the caller reports it.
    ``None`` also when there was nothing to drop, so the retry is not a second
    identical attempt.
    """
    import struct
    import zlib
    sig = b"\x89PNG\r\n\x1a\n"
    if not raw.startswith(sig):
        return None
    out = [sig]
    dropped = 0
    i = len(sig)
    while i + 12 <= len(raw):
        (n,) = struct.unpack(">I", raw[i:i + 4])
        end = i + 12 + n
        if end > len(raw):
            return None
        kind, body = raw[i + 4:i + 8], raw[i + 8:i + 8 + n]
        (crc,) = struct.unpack(">I", raw[i + 8 + n:end])
        ok = zlib.crc32(kind + body) & 0xFFFFFFFF == crc
        ancillary = bool(kind[0] & 0x20)
        if ok:
            out.append(raw[i:end])
        elif ancillary:
            dropped += 1
        else:
            return None
        i = end
        if kind == b"IEND":
            break
    return b"".join(out) if dropped else None


def _zipmod_named_texture(mref, bundle_rel: str, name: str, *,
                          from_bundle: bool = False):
    """The same lookup, but inside the .zipmod the item came from.

    A mod's ``MainTexAB`` is usually its own bundle, and that bundle only
    exists inside the archive — ``_open_bundle`` looks under the *install*, so
    it can only ever answer for a row that points at a base-game bundle.  The
    row may name either, which is why both are tried (caller's order).

    **Loose files first**, the same order as :func:`mod_texture` and for the
    same reason: a texture-only mod may register ``MainAB`` as a folder, and
    then the bundle the row names does not exist.  Only the row's *main* map
    used to get that; a second column off the same row (``EyelineShadowTex``)
    went to the bundle alone and 15 local lash rows reported their lid shadow
    unloadable while it sat in the archive as a PNG (§7.356).

    The converted copy answers first, including "not in this archive"
    (:func:`_mod_named`); *from_bundle* skips it for the converter (§7.474).
    """
    from dataclasses import replace as _replace
    key = (str(getattr(mref, "zipmod", "")), bundle_rel, name)
    if not from_bundle and key in _TEX_CACHE:
        return _TEX_CACHE[key]
    if not from_bundle:
        known, got = _mod_named(mref, bundle_rel, name)
        if known:
            _TEX_CACHE[key] = got
            return got
    try:
        alt = _replace(mref, bundle=bundle_rel, asset=name)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.load_mesh._zipmod_named_texture: alt = _replace(mref, bundle=bundle_re…", exc)
        _TEX_CACHE[key] = None
        return None
    got = _loose_png(alt)
    loader = None if got is not None else _open_zipmod(alt)
    if loader is not None:
        try:
            got = loader.texture_by_name(name, f"{name}_low")
        except Exception as exc:  # noqa: BLE001
            swallowed.note("origin.figure.load_mesh._zipmod_named_texture: got = loader.texture_by_name(name…", exc)
            got = None
        finally:
            _release(loader)
    if not from_bundle:          # the converter reads each map once (§7.474)
        _TEX_CACHE[key] = got
    return got


def _loose_png_wrap(bundle_rel: str, name: str) -> int:
    """Wrap mode Sideloader gives a loose PNG: ``Sideloader.GetPng`` verbatim.

    ``path.Contains("clamp")`` → Clamp (1), else ``Contains("repeat")`` → Repeat
    (0), else nothing is set and the ``new Texture2D(2, 2, …)`` default stands,
    which is Repeat.  The test is on the path string, case-sensitive, as the IL
    does it.
    """
    path = f"{bundle_rel}/{name}"
    if "clamp" in path:
        return 1
    return 0


_WRAP_CACHE: dict[tuple, "int | None"] = {}


def ref_texture_wrap(ref, ab_key: str, tex_key: str, *,
                     from_bundle: bool = False) -> int | None:
    """``m_WrapMode`` (0 Repeat, 1 Clamp) of the map a list row names.

    Resolved along the same two paths as :func:`_ref_texture` and in the same
    order — a zipmod's loose PNG, then its bundle, then the install — so the
    mode always belongs to the pixels that were actually loaded.  ``None`` when
    the row names nothing or the map could not be opened; the caller reports
    that instead of assuming a mode (一之一).

    The converted copies answer first — the mode is stored beside the pixels
    (§7.473, §7.474); *from_bundle* skips them for the converter.
    """
    from .. import bake
    if ref is None or not hasattr(ref, "texture_ref"):
        return None
    spec = ref.texture_ref(ab_key, tex_key)
    if spec is None:
        return None
    bundle, name = spec
    key = (str(getattr(ref, "zipmod", "")), bundle, name)
    if not from_bundle and key in _WRAP_CACHE:
        return _WRAP_CACHE[key]
    mode = None
    zipped = bool(getattr(ref, "zipmod", ""))
    guid = str(getattr(ref, "guid", "") or "")
    entry = (None if from_bundle or not zipped or not guid
             else bake.named_entry(guid, bundle, name))
    if entry is not None and not entry.get("absent"):
        _WRAP_CACHE[key] = entry.get("wrap")
        return entry.get("wrap")
    if zipped and entry is None:
        from dataclasses import replace as _replace
        try:
            alt = _replace(ref, bundle=bundle, asset=name)
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.figure.load_mesh.ref_texture_wrap: alt = _replace(ref, bundle=bundle, ass…", exc)
            alt = None
        if alt is not None and SL.load_mod_file(alt):
            mode = _loose_png_wrap(bundle, name)
        elif alt is not None:
            loader = _open_zipmod(alt)
            if loader is not None:
                try:
                    mode = loader.texture_wrap_by_name(name, f"{name}_low")
                finally:
                    _release(loader)
    if mode is None:
        entry = None if from_bundle else bake.named_entry(bake.INSTALL, bundle, name)
        if entry is not None and not entry.get("absent"):
            # the converted copy carries the mode it was decoded with (§7.473)
            mode = entry.get("wrap")
            _WRAP_CACHE[key] = mode
            return mode
        mode = _stock_texture_wrap(bundle, name)
    _WRAP_CACHE[key] = mode
    return mode


def _ref_texture(ref, ab_key: str, tex_key: str):
    """``(spec, array)`` for one texture column of a list row.

    ``spec`` is what the row asked for (``None`` when it names nothing) and
    ``array`` is what came back.  The two are returned together so the caller
    can tell **"this item has no such map"** — normal, the shader samples its
    declared default — from **"the row named one and we could not load it"**,
    which is a failure and has to be reported rather than papered over with a
    colour (§一之一).
    """
    if ref is None or not hasattr(ref, "texture_ref"):
        return None, None
    spec = ref.texture_ref(ab_key, tex_key)
    if spec is None:
        return None, None
    bundle, name = spec
    if getattr(ref, "zipmod", ""):
        got = _zipmod_named_texture(ref, bundle, name)
        if got is None:                              # a row may point at abdata
            got = _named_texture(bundle, name)
    else:
        got = _named_texture(bundle, name)
    return spec, got


def mod_texture(mref, *, from_bundle: bool = False):
    """A map a zipmod supplies for one of the texture-only lists.

    The texture-only lists (mt_eye, mt_eyeline_up, mt_nip …) point straight at
    an image rather than a prefab, and a mod may ship it either way: as a
    ``Texture2D`` inside a bundle, or — about half the time locally — as a
    loose ``abdata/<MainAB>/<MainData>.png``.  The loose form is tried first
    because when it is present there is no bundle of that name to open at all.

    The converted copy answers first (``convert.bake_mod_textures`` bakes what
    the cards name, under the mod's GUID, §7.474); *from_bundle* skips it.
    """
    key = (str(getattr(mref, "zipmod", "")), str(getattr(mref, "asset", "")))
    if not from_bundle and key in _TEX_CACHE:
        return _TEX_CACHE[key]
    if not from_bundle:
        _known, got = _mod_named(mref, mref.bundle, mref.asset)
        if got is not None:
            _TEX_CACHE[key] = got
            return got
    got = _loose_png(mref)
    if got is None:
        loader = _open_zipmod(mref)
        if loader is not None:
            try:
                got = loader.texture_by_name(mref.asset, f"{mref.asset}_low")
            except Exception as exc:  # noqa: BLE001
                swallowed.note("origin.figure.load_mesh.mod_texture: got = loader.texture_by_name(mref.asset…", exc)
                got = None
            finally:
                _release(loader)
    if got is None:
        # A mod may register a *base-game* asset under a new slot id rather
        # than ship its own — charamaker support MOD re-lists a dozen stock
        # highlights that way, and its rows point at ``chara/mt_eye_hi_up_00``.
        got = _named_texture(mref.bundle, mref.asset, from_bundle=from_bundle)
    if not from_bundle:          # the converter reads each map once (§7.474)
        _TEX_CACHE[key] = got
    return got


# ``ChaClothesComponent`` role -> which of the row's three texture pairs the
# renderer is given.  ``ChangeCustomClothes`` hands ct[0] to ``rendNormal01``,
# ``rendAlpha01`` and ``rendAccessory``, ct[1] to ``rendNormal02`` and ct[2] to
# ``rendNormal03``; anything the component does not list keeps set 0, which is
# what a single-renderer garment is.
_ROLE_SET = {"normal02": 1, "normal03": 2}


def apply_list_textures(pieces: list, ref, *, missing: list | None = None) -> list:
    """Fill in albedo / colour mask from the item's own list row.

    **This is the primary path, not a fallback.**  A shipped clothing material
    binds neither ``_MainTex`` nor ``_ColorMask`` — both are ``None`` in the
    prefab, and ``ChaControl.ChangeCustomClothes`` assigns them at runtime from
    the list row (``MainTex`` / ``ColorMaskTex``, plus the bundle when it is not
    the item's own).  Guessing them from the material name is what the loader
    used to do, and it silently picked up ``_md`` detail maps as if they were
    albedos.  A material that really does bind its own ``_MainTex`` (some
    zipmods) keeps it; otherwise the list wins.

    **This runs for zipmod items too.**  It used to bail out on anything
    without an :class:`lists.AssetRef`, on the grounds that "zipmod refs carry
    no list row" — they do, the whole row is in ``ModRef.extra``, and the game
    reads it through the same ``GetInfo`` call for both.  Every modded garment
    that leaves ``_MainTex`` unbound therefore arrived with no albedo and was
    drawn as one flat colour.

    There are **three** pairs, not two — ``InitBaseCustomTextureClothes`` reads
    ``MainTex`` / ``MainTex02`` / ``MainTex03`` with their colour masks and
    builds one composite each — and which renderer gets which is stated by
    ``ChaClothesComponent`` (see :meth:`prefab.PrefabLoader.clothes_roles`),
    not inferred from the mesh's name.  The ``*AB`` column beside each names
    the bundle; a blank one means the item's own, and the game falls back to
    ``MainAB`` for that too.

    The emblem renderers take none of these: their map is the card's, applied
    by :func:`apply_emblem` after the outfit is known.
    """
    if ref is None or not pieces or not hasattr(ref, "texture_ref"):
        return pieces
    specs = (("MainTexAB", "MainTex", "ColorMaskAB", "ColorMaskTex"),
             ("MainTex02AB", "MainTex02", "ColorMask02AB", "ColorMask02Tex"),
             ("MainTex03AB", "MainTex03", "ColorMask03AB", "ColorMask03Tex"))
    sets: list[tuple] = []
    for ab, tex, mab, mtex in specs:
        main_spec, main = _ref_texture(ref, ab, tex)
        mask_spec, mask = _ref_texture(ref, mab, mtex)
        for key, spec, got in ((tex, main_spec, main), (mtex, mask_spec, mask)):
            if spec is not None and got is None and missing is not None:
                missing.append(
                    f"{getattr(ref, 'name', '?')}: {key} 解析不到"
                    f"（{spec[0]} / {spec[1]}）")
        sets.append((main, mask))
    if not any(t is not None or m is not None for t, m in sets):
        return pieces
    out = []
    for p in pieces:
        role = str((p.shader or {}).get("_clothRole") or "")
        si = _ROLE_SET.get(role, 0)
        if sets[si][0] is None and sets[si][1] is None:
            si = 0                      # component names a set the row has not
        tex_map, mask_map = sets[si]
        # The map and its colour space are replaced **together**.  Leaving the
        # flag behind would give the list row's pixels the prefab material's
        # ``m_ColorSpace`` — the same two-paths-for-one-row split that let a
        # combo top's mesh come from the card's UAR while its mask came from a
        # slot scan (§7.74), except the failure here is a wrong lerp weight and
        # nothing reports it.
        cs = dict(p.map_cs or {})
        tex, msk = p.texture, p.mask
        if tex_map is not None:
            tex, cs["texture"] = tex_map.px, tex_map.space
        if mask_map is not None:
            msk, cs["mask"] = mask_map.px, mask_map.space
        out.append(p._replace(texture=tex, mask=msk, tex_set=si, map_cs=cs))
    return out


def apply_emblem(pieces: list, emblem_ids, res=None, props=None) -> list:
    """``ChangeCustomEmblem`` — the chest badge, from the card, not the garment.

    ``ChaControl.ChangeCustomEmblem(clothesKind, no)`` looks
    ``parts[0].emblemeId`` / ``emblemeId2`` up in CategoryNo **431**
    (``mt_emblem``), takes ``MainTexAB`` / ``MainTex`` off the row and assigns
    the texture to the emblem renderers' ``_MainTex``.  Nothing else about them
    is set: ``main_emblem`` declares only ``_MainTex``, ``_shadowcolor`` and
    ``_Cutoff``, samples no colour mask and reads no ``_Color``, so the badge
    is its own map straight through.

    Row 0 is 「なし」 and names no texture at all, in which case the game leaves
    ``_MainTex`` null and the shader falls back to the property's declared
    default, ``"black"`` — RGBA(0,0,0,**0**) — which its own alpha test
    (``pow(a, 0.3) < 0.5``, i.e. a < 0.0992) then discards outright.  So an
    emblem-less card draws no badge, and the piece is dropped here.
    """
    if not pieces:
        return pieces
    ids = list(emblem_ids or ())
    maps: dict[str, object] = {}
    for slot in (0, 1):
        ident = ids[slot] if slot < len(ids) else 0
        prop = props[slot] if props and slot < len(props) else None
        maps[f"emblem{slot}"] = _emblem_texture(ident, res=res, prop=prop)
    out = []
    for p in pieces:
        role = str((p.shader or {}).get("_clothRole") or "")
        if not role.startswith("emblem"):
            out.append(p)
            continue
        tex = maps.get(role)
        if tex is None:
            continue                     # 「なし」: the shader discards it
        # The badge replaces the albedo outright, so its colour space replaces
        # the piece's too — and the mask goes with it (``main_emblem`` samples
        # none), which means dropping that entry rather than leaving a flag
        # pointing at a map that is no longer there.
        cs = {k: v for k, v in (p.map_cs or {}).items() if k != "mask"}
        cs["texture"] = tex.space
        out.append(p._replace(texture=tex.px, mask=None, tex_set=0, map_cs=cs))
    return out


def _emblem_texture(ident, res=None, prop: str | None = None):
    """One row of ``mt_emblem``; ``None`` when the row names no map.

    A modded badge is a sideloader slot, and ``prop`` is the UAR property that
    names it — ``ClothesTopEmblem`` / ``…Emblem2``, off
    ``StructReference.ChaFileClothesGenerator`` (CategoryNo 431).  Asked first
    for the same reason every other id is (:func:`resolve.uar_named`).  Without
    it this fell through to ``find_any`` (``CompatibilityResolve``), which takes
    the claimant whose GUID sorts first whenever two installed mods number a
    badge alike — the game's answer for a card with no UAR row, and the wrong
    one for a card that has it (§7.478).  Coming back empty
    means 「なし」 and the caller drops the quad.
    """
    from .. import bake, lists as L
    from . import resolve as R
    try:
        ident = int(ident)
    except (TypeError, ValueError):
        return None
    if ident < 0:
        return None
    gs = R.uar_named(prop, ident, res) if prop else None
    if gs:
        mref = SL.index().resolve(gs[0], "emblem", gs[1])
        if mref is not None:
            return mod_texture(mref)
    hit = bake.read_texture("emblem", ident)
    if hit is not None:
        return hit
    ref = L.index().get("emblem", ident)
    if ref is not None:
        return _named_texture(ref.bundle, ref.asset)
    if L.index().is_none("emblem", ident):
        return None                      # the list says 「なし」, not "missing"
    if gs:
        return None                      # named, not installed: nothing owns it
    mref = SL.index().find_any("emblem", ident)
    return None if mref is None else mod_texture(mref)


def _baked(category: str, ident) -> LoadedItem | None:
    """A previously converted item, if one is on disk.

    Nothing about a resolved item depends on the card, so re-decompressing the
    bundle and re-parsing every renderer on each load is pure repetition —
    ``tools/bake_assets.py`` writes the result out once (see
    :mod:`polylode.origin.bake`).
    """
    if ident is None:
        return None
    try:
        from .. import bake
        item = bake.read_item(category, ident)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("origin.figure.load_mesh._baked: from .. import bake", exc)
        return None
    if item is None:
        return None
    # ``sway`` has to come back out with the mesh: it is the solve for the
    # chains the *body* owns (a skirt's ``cf_j_sk_*``), which ``dress`` writes
    # onto the skeleton.  Dropping it left every baked skirt in its authored
    # bell shape while the same garment loaded from the bundle draped.
    return LoadedItem(label=item.label, source=item.source,
                      pieces=item.pieces, category=category,
                      sway=dict(item.sway or {}),
                      sway_spec=list(getattr(item, "sway_spec", None) or []),
                      moves=list(item.moves or [None, None]),
                      missing_maps=item.missing_maps)


# Every sideloader item bakes under one directory rather than under the list
# category it happens to fill, because the category ``resolve`` looks an item up
# under is not always the one the mod files it as, and writer and reader have to
# agree.  The identity inside that directory is the index's own key —
# ``(guid, category, slot)``, all three of them:
#
#   * the slot alone collides wholesale.  440 zipmods ship a ``bottom`` slot 1,
#     and every base ``bra`` id is also somebody's mod slot, so a bare id hands
#     a card the wrong garment as soon as anything is baked under that number.
#   * GUID plus slot still collides, because one zipmod reuses a slot number
#     across categories — ``Clannad`` has both a ``hair_back`` 220 and a
#     ``hair_front`` 220, ``DarkMagicianGirl`` has gloves / legwear / panty all
#     at 55.  1,945 keys and 6,000 items collide that way, and the A/B check
#     caught it as a card wearing 189-vertex shoes instead of 1,326-vertex ones.
MOD_CATEGORY = "mod"


def mod_pieces(head):
    """A head prefab that lives in a zipmod, plus its own face maps.

    Same shape as the base path in ``body_head.load_head_prefab``.  The maps
    are looked up inside the mod's own bundle first, because a head mod that
    ships a face texture names it in its own row; when it names none, the
    stock pair is what the row fell back to and that is a base-bundle name.
    """
    mref = head.mod
    loader = _open_zipmod(mref) if mref is not None else None
    if loader is None:
        raise FileNotFoundError(f"讀不到 zipmod 的頭：{head.prefab}")
    try:
        # <ChangeHeadAsync> passes copyDynamicBone = false.
        pieces = _pieces_from_loader(loader, head.prefab, copy_dynamic_bone=False)
        if not pieces:
            raise FileNotFoundError(f"zipmod 裡沒有 {head.prefab}")
        face_t = face_mc = None
        try:
            face_t = loader.texture_by_name(head.tex, head.tex + "_low")
            face_mc = loader.texture_by_name(head.mask, head.mask + "_low")
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.figure.load_mesh.mod_pieces: loader.texture_by_name", exc)
    finally:
        _release(loader)
    if face_t is None:
        face_t = _named_texture(HEAD_FALLBACK_BUNDLE, head.tex)
    if face_mc is None:
        face_mc = _named_texture(HEAD_FALLBACK_BUNDLE, head.mask)
    return pieces, face_t, face_mc


HEAD_FALLBACK_BUNDLE = "chara/bo_head_00.unity3d"


def mod_bake_key(mref) -> str | None:
    """The bake identity of a sideloader item — the index's own three-part key."""
    guid = str(getattr(mref, "guid", "") or "")
    ident = getattr(mref, "id", None)
    if not guid or ident is None:
        return None
    parts = f"{guid}_{getattr(mref, 'category', '') or ''}_{int(ident)}"
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in parts)


def load_base(bundle: str, asset: str, *, label: str = "",
              category: str = "", ref=None,
              missing: list | None = None) -> LoadedItem | None:
    ident = getattr(ref, "id", None)
    hit = _baked(category, ident)
    if hit is not None:
        _hand_on_missing(hit, missing)
        return hit
    loader = _open_bundle(bundle, asset)
    if loader is None:
        return None
    try:
        pieces = _pieces_from_loader(
            loader, asset, copy_dynamic_bone=copies_colliders(category))
        sway = dict(getattr(loader, "last_sway", None) or {})
        sway_spec = list(getattr(loader, "last_sway_spec", None) or [])
        moves = list(getattr(loader, "last_moves", None) or [None, None])
    finally:
        _release(loader)
    if not pieces:
        return None
    return LoadedItem(label=label or asset, source="base",
                      pieces=apply_list_textures(pieces, ref, missing=missing),
                      category=category, sway=sway, sway_spec=sway_spec, moves=moves)


def load_mod(mref, *, category: str = "", ref=None,
             missing: list | None = None) -> LoadedItem | None:
    key = mod_bake_key(mref)
    hit = _baked(MOD_CATEGORY, key) if key else None
    if hit is not None:
        hit.category = category          # the slot it fills, for the caller
        _hand_on_missing(hit, missing)
        return hit
    loader = _open_zipmod(mref)
    if loader is None:
        return None
    try:
        pieces = _pieces_from_loader(
            loader, mref.asset,
            copy_dynamic_bone=copies_colliders(category))
        sway = dict(getattr(loader, "last_sway", None) or {})
        sway_spec = list(getattr(loader, "last_sway_spec", None) or [])
        moves = list(getattr(loader, "last_moves", None) or [None, None])
    finally:
        _release(loader)
    if not pieces:
        return None
    return LoadedItem(
        label=f"{mref.name} ({Path(mref.zipmod).name})",
        source="zipmod",
        pieces=apply_list_textures(pieces, ref or mref, missing=missing),
        category=category, sway=sway, sway_spec=sway_spec, moves=moves,
    )


# ChaControl.ChangeClothesTop(id, subId01, subId02, subId03):
#   co_top id 1 → CategoryNo cpo_sailor_{a,b,c} + subPartsId
#   co_top id 2 → CategoryNo cpo_jacket_{a,b,c} + subPartsId
# Each sub-part is loaded via LoadCharaFbxData(GetListInfo(cat, subId)).
COMBO_TOP_CATS = {
    1: ("sailor", ("cpo_sailor_a", "cpo_sailor_b", "cpo_sailor_c")),
    2: ("jacket", ("cpo_jacket_a", "cpo_jacket_b", "cpo_jacket_c")),
}
# Sideloader UAR property names for the three sub-parts, by top kind.  A modded
# sub-part's id is a sideloader slot, not a base-list id, and the card records
# which mod it came from under these keys.
COMBO_UAR = {
    "sailor": ("ClothesSailorSubA", "ClothesSailorSubB", "ClothesSailorSubC"),
    "jacket": ("ClothesJacketSubA", "ClothesJacketSubB", "ClothesJacketSubC"),
}
# ``ChaListDefine.CategoryNo`` for the same three sub-parts.
COMBO_CAT_NO = {"sailor": (200, 201, 202), "jacket": (210, 211, 212)}
# ``<ChangeClothesTopPartsAsync>``'s ``defaultId``: the array literal
# ``new int[2, 3] {{0, 0, 1}, {0, 1, 1}}`` (``origin_il_dump --data
# '$field-567D1C9D…'``), row ``topType == 1 ? 0 : 1``, handed to
# ``LoadCharaFbxDataAsync`` — which tries the card's id, then this default,
# then id 0, and only then gives up (§7.477).
COMBO_DEFAULT_ID = {"sailor": (0, 0, 1), "jacket": (0, 1, 1)}


def _row_or_default(base, category: str, ident: int, default: int):
    """``LoadCharaFbxDataAsync``'s lookup: the id, else the default, else id 0."""
    row = base.get(category, ident)
    if row is None and ident != default:
        row = base.get(category, default)
    if row is None:
        row = base.get(category, 0)
    return row


class ComboSub(NamedTuple):
    """One sub-part of a combo top, resolved both ways it can come."""
    slot: int
    category: str
    sid: int
    mod: object | None       # the zipmod row the card's UAR names, if any
    base: object | None      # the base-list row of that number


def combo_sub_refs(item_id: int, sub_parts, res: dict | None = None,
                   base=None) -> list[ComboSub]:
    """The three sub-parts of combo top *item_id*, each resolved once.

    The card's UAR first (a modded sub-part's id is a sideloader slot), then the
    base list the way ``LoadCharaFbxDataAsync`` reads it: the id, else
    :data:`COMBO_DEFAULT_ID`, else id 0.  (This used to fall back to ``sid + 1``
    on the theory that some tables are numbered from 1 — a guess the IL does
    not make.)  The loader, the bra / body mask rules and the converter all read
    sub-parts, and
    they used to each resolve them; two lookups that disagree is how the mask
    and the mesh came from different rows (§7.74), and the converter's missing
    copy is why modded sub-parts were never baked (§7.477).  *base* is a
    ``lists.AssetIndex``; without it only the mod column is filled.
    """
    entry = COMBO_TOP_CATS.get(int(item_id))
    if entry is None:
        return []
    kind, cats = entry
    subs = [int(s) for s in (sub_parts or [])][:3]
    subs += [0] * (3 - len(subs))
    uar, cat_nos = COMBO_UAR[kind], COMBO_CAT_NO[kind]
    out = []
    for slot, cat in enumerate(cats):
        sid = subs[slot]
        row = (None if base is None
               else _row_or_default(base, cat, sid, COMBO_DEFAULT_ID[kind][slot]))
        out.append(ComboSub(slot, cat, sid,
                            _sub_part_mod(res, uar[slot], sid, cat_nos[slot]), row))
    return out


def _sub_part_mod(res: dict | None, field: str, sid: int, cat_no: int = 0):
    """The zipmod a card names for one combo sub-part, if any.

    ``cat_no`` pins the lookup to the sub-part's own list: an outfit pack numbers
    its blouse, collar and tie from the same run as its skirt, so ``(guid, slot)``
    alone can answer with the wrong piece (§7.66 五).
    """
    if not res or not field:
        return None
    for key, guid_slot in res.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        prop, ident = key
        # a card can carry resolution rows with no id (whole-mod entries)
        if ident is None or not str(prop).endswith(field):
            continue
        try:
            if int(ident) != int(sid):
                continue
        except (TypeError, ValueError):
            continue
        try:
            return SL.index().find(guid_slot[0], guid_slot[1], cat_no or None)
        except Exception as exc:                     # noqa: BLE001
            swallowed.note("origin.figure.load_mesh._sub_part_mod: return SL.index().find(guid_slot[0], guid…", exc)
            return None
    return None


def load_combo_top(item_id: int, sub_parts: list[int] | None,
                   res: dict | None = None,
                   missing: list | None = None) -> LoadedItem | None:
    """Assemble a p_dummy school top the way ChaControl.ChangeClothesTop does.

    Looks up each sub-part in ``list/characustom`` (cpo_sailor_a/b/c …) by the
    card's ``subPartsId`` — not by inventing ``p_o_top_sailorNN`` names.

    A sub-part can also come from a zipmod, in which case its id is a
    sideloader slot and the base list has nothing at that number (a jacket
    asking for ``subPartsId = [3696, 127, 0]`` against a ten-row table).  The
    card records the owning GUID under ``ClothesJacketSubA`` and friends, so
    that is consulted first; without it the whole garment silently failed to
    load rather than one piece of it.
    """
    from .. import lists as L

    entry = COMBO_TOP_CATS.get(int(item_id))
    if entry is None:
        return None
    kind, _cats = entry

    inst = configured()
    if not inst.valid:
        return None
    base = L.index(inst)
    pieces: list[Piece] = []
    labels: list[str] = []
    inner_mask = None
    def tagged(got: list, slot: int) -> list:
        """Remember which sub-part a piece came from.

        ``ChangeCustomClothes`` colours sub-part **kind 2** from
        ``colorInfo[3]`` instead of ``colorInfo[0]`` (see
        :func:`attach.clothes_slots`), and once the three sub-parts are merged
        into one piece list there is nothing left to tell them apart.  It rides
        in ``shader``, which is a free-form dict that already survives the bake.
        """
        return [p._replace(shader={**(p.shader or {}), "_comboSub": slot})
                for p in got]

    for slot, cat, sid, mref, ref in combo_sub_refs(item_id, sub_parts, res, base):
        if mref is not None:
            item = load_mod(mref, category=cat, ref=mref, missing=missing)
            if item is not None:
                # The mask column belongs to the *row*, so a modded kind-1
                # sub-part carries it exactly like a stock one does.  Reading it
                # only in the base-list branch below meant a modded blazer never
                # cut the shirt underneath it (§7.74).
                if slot == 1:
                    spec, tex = _ref_texture(mref, "OverInnerMaskAB",
                                             "OverInnerMask")
                    if spec is not None and tex is None and missing is not None:
                        missing.append(
                            f"inner mask {spec[0]}/{spec[1]} 載不到（{mref.name}）")
                    elif tex is not None:
                        inner_mask = tex
                pieces.extend(tagged(item.pieces, slot))
                labels.append(f"{mref.name}#{sid}(mod)")
                continue
        if ref is None:
            # The card names a sub-part that neither the base table nor the
            # mod it points at can answer for — a modded blouse whose zipmod
            # is not installed.  ``ChangeClothesTop`` loads the three
            # independently, so the rest of the garment still arrives and the
            # only sign is a missing layer; say so rather than return a
            # silently incomplete top (§一之一).
            if missing is not None and sid > 0:
                missing.append(f"top: {cat} #{sid} 找不到（清單與 zipmod 都沒有）")
            continue
        if slot == 1:                    # only kind 1 supplies the inner mask
            spec, tex = _ref_texture(ref, "OverInnerMaskAB", "OverInnerMask")
            if spec is not None and tex is None and missing is not None:
                missing.append(
                    f"inner mask {spec[0]}/{spec[1]} 載不到（{ref.name}）")
            elif tex is not None:
                inner_mask = tex
        if names_nothing(ref):
            # 「なし」: the game instantiates ``p_dummy``, an empty GameObject.
            # Opening ``etc.unity3d`` to learn that was a run-time bundle read
            # on every card wearing a combo top without that part (§7.477).
            continue
        hit = _baked(cat, ref.id)
        if hit is not None:
            _hand_on_missing(hit, missing)
            pieces.extend(tagged(hit.pieces, slot))
            labels.append(f"{ref.name}#{sid}")
            continue
        loader = _open_bundle(ref.bundle, ref.asset)
        if loader is None:
            continue
        try:
            got = _pieces_from_loader(loader, ref.asset,
                                      copy_dynamic_bone=copies_colliders(cat))
        finally:
            _release(loader)
        if got:
            pieces.extend(tagged(
                apply_list_textures(got, ref, missing=missing), slot))
            labels.append(f"{ref.name}#{sid}")
    if not pieces:
        return None
    pieces = _mask_inner(pieces, inner_mask)
    lab = f"{kind} [{', '.join(labels)}] (list)"
    return LoadedItem(label=lab, source="base", pieces=pieces, category="top")


def _mask_inner(pieces: list, mask) -> list:
    """Let the outer layer punch a hole in the inner shirt.

    ``ChangeClothesTopPartsAsync`` calls ``LoadAlphaMaskTexture(ab, tex, 2)``
    — kind 2 is ``texInnerAlphaMask`` — **only for sub-part kind 1**, and
    assigns the result to ``rendInner``'s ``_AlphaMask``.  The columns are
    ``OverInnerMask`` / ``OverInnerMaskAB`` (ChaListDefine.KeyType 51 / 52),
    and only ``cpo_jacket_b`` carries them: a blazer cutting away the shirt
    underneath it.

    ``rendInner`` is the object Illusion names ``o_inner_a`` / ``o_inner_b``
    (ChaReference), which is how the piece is picked out here — the inner
    shirt arrives from sub-part slot 0 (``cpo_jacket_a`` =
    ``p_o_top_inner01_*``) while the mask comes from slot 1.
    """
    if mask is None:
        return pieces
    out = []
    for p in pieces:
        name = (getattr(p.mesh, "name", None) or "").lower()
        if name.startswith("o_inner") or name.startswith("n_inner"):
            # The flag travels with the map even though nothing reads it for
            # this one — ``_AlphaMask`` is a cut-out the forward pass
            # thresholds, not a create-pass colour.  Leaving the piece's old
            # entry in place would have it describe a map that is gone.
            p = p._replace(alpha_mask=mask.px,
                           map_cs={**(p.map_cs or {}),
                                   "alpha_mask": mask.space})
        out.append(p)
    return out