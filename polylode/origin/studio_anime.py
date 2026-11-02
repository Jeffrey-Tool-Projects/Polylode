"""Studio's own animations for scene characters: ``Anime_*`` rows, converted (§7.520).

A scene character plays ``dicAnimeLoadInfo[group][category][no]``
(:class:`studio_catalog.StudioAnime`): ``OCIChar.LoadAnime`` loads the row's
controller and ``CharAnimeCtrl.Play`` hands ``Animator.Play`` the row's クリップ名,
which is a **state** — so the clips are identified the way the H packs are,
``(controller, state)`` through :func:`hanim.extract_clips_by_controller`.  A
body-size tree bakes as its ``S_`` / ``M_`` / ``L_`` leaves and plays blended by
the card's height (``height`` = ``shapeValueBody[0]``); any other tree stays
unconverted and is reported.

The converter writes ``assets_baked/h_anim/studio/`` (``manifest.json`` plus one
JSON per clip); the run time reads only that (``CLAUDE.md`` 四之一).

An ``HAnime_*`` row (same table, §7.521) plays a state of an H pack's
controller, which the H bake converts; the manifest records what
``h_runtime.studio_pose`` needs of it (:func:`h_spec`, :func:`h_row`).
"""
from __future__ import annotations

import json
from pathlib import Path

from .hanim import BAKE_H_DIR, CLIP_PREFIX, SIZE_PREFIXES, migrate_baked_names

STUDIO_DIR = BAKE_H_DIR / "studio"
MANIFEST = STUDIO_DIR / "manifest.json"


def clip_name(pack: str, controller: str, state: str) -> str:
    """The converter's key, the same shape as :func:`hanim.extract_clips_by_controller`'s."""
    return f"{CLIP_PREFIX}x_{pack}_{controller}_{state}"[:96]


def bake(force: bool = False, log=print) -> dict:
    """Convert every controller state an ``Anime_*`` or ``HandAnime_*`` row names.

    Bundle by bundle out of the project cache (``assets_cache/<バンドルパス>``);
    the manifest is saved after each so a run the guard kills keeps what it
    wrote, and a clip already on disk is not converted again unless *force*.
    A hand controller's clips are converted under the transform its
    ``Animator`` sits on (:data:`HAND_ROOTS`).
    """
    from ..core.paths import write_json_atomic
    from .install import cache_dir
    from .studio_catalog import catalog

    rows = catalog().anime
    by_bundle: dict[str, set] = {}
    for r in rows.values():
        if r.kind == "anime":
            by_bundle.setdefault(r.bundle, set()).add(r.controller)
    hands = catalog().hands
    hand_bundles: dict[str, set] = {}
    for h in hands.values():
        hand_bundles.setdefault(h.bundle, set()).add(h.controller)
    index = {"clips": {}}
    migrate_baked_names()
    if MANIFEST.is_file() and not force:
        index = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tally = {"bundles": 0, "written": 0, "blend_tree": 0, "unresolved": 0, "missing_bundle": 0,
             "unrooted": 0}
    (STUDIO_DIR / "x").mkdir(parents=True, exist_ok=True)
    _convert(by_bundle, "cf_j_root", index, tally, force, log)
    _convert(hand_bundles, HAND_ROOTS, index, tally, force, log)
    index["hands"] = {f"{h.hand}/{h.no}": hand_name(h, index["clips"]) for h in hands.values()}
    tally["hand_rows"] = len(hands)
    tally["hand_rows_playable"] = sum(1 for v in index["hands"].values() if v)
    params = _base_params({(r.base_bundle, r.base_controller) for r in rows.values()
                           if r.kind == "h"}, cache_dir(), log)
    rows_out = {f"{r.group}/{r.category}/{r.no}": (lookup_names(r, index["clips"]) if r.kind == "anime"
                                                  else {"h": h_spec(r, params)})
                for r in rows.values()}
    index["rows"] = rows_out
    write_json_atomic(MANIFEST, index, indent=1)
    tally["rows"] = len(rows_out)
    tally["rows_playable"] = sum(1 for v in rows_out.values() if v and "h" not in v)
    tally["h_rows"] = sum(1 for v in rows_out.values() if "h" in v)
    tally["h_rows_no_params"] = sum(1 for v in rows_out.values()
                                    if "h" in v and v["h"]["params"] is None)
    log(f"studio anime: {tally}")
    return tally


#: Where a hand's ``Animator`` can sit: the transforms above the fingers.  The
#: bake keeps the one every binding path of a controller resolves under
#: (``hanim.animator_root``) — this lists the candidates, it does not pick.
HAND_ROOTS = ("cf_s_hand_L", "cf_j_hand_L", "cf_s_hand_R", "cf_j_hand_R", "cf_j_root")


def _convert(by_bundle: dict, root, index: dict, tally: dict, force: bool, log) -> None:
    """Convert the named controllers bundle by bundle into ``STUDIO_DIR``, saving the manifest after each."""
    from ..core.paths import write_json_atomic
    from .hanim import ON_DISK, _clip_to_json, extract_clips_by_controller
    from .install import cache_dir
    for bundle, ctrls in sorted(by_bundle.items()):
        path = cache_dir() / bundle
        if not path.is_file():
            log(f"  ! {bundle} 不在 assets_cache（sync 過了嗎）")
            tally["missing_bundle"] += 1
            continue
        clips, rep = extract_clips_by_controller(
            path, max_clips=1_000_000, controllers=ctrls, all_clips=True, root=root,
            have=None if force else (lambda rel: (STUDIO_DIR / rel).is_file()))
        tally["bundles"] += 1
        for k in ("blend_tree", "unresolved", "unrooted"):
            tally[k] += int(rep.get(k, 0) or 0)
        for clip, name, meta in clips:
            rel = f"x/{name}.json"
            if clip is not ON_DISK:
                write_json_atomic(STUDIO_DIR / rel, _clip_to_json(clip))
                tally["written"] += 1
            index["clips"][name] = {"file": rel, "controller": meta["controller"],
                                    "stage": meta["stage"], "pack": meta["pack"],
                                    **({"root": meta["root"]} if "root" in meta else {})}
        write_json_atomic(MANIFEST, index, indent=1)
        log(f"  {bundle}: {len(clips)} clips {rep}")


def hand_name(row, clips: dict) -> str | None:
    """The converted clip a ``HandAnime_`` row's state is, ``None`` when it did not convert."""
    name = clip_name(Path(row.bundle).stem, row.controller, row.state)
    return name if name in clips else None


def hand_rotations(hand_patterns) -> tuple[dict, list]:
    """``({bone: {"rot": absolute local}}, notes)`` the two hand animations hold (E1d).

    ``OCIChar.ChangeHandAnime(kind, ptn)`` hands ``handPtn[kind]`` to the
    ``HandAnimeCtrl`` of that hand (0 Left, 1 Right); ``LoadAnime`` looks it up
    in ``dicHandAnime[hand]`` and, found, plays the row's state on the hand's
    own ``Animator`` — not found disables it and the body animation keeps the
    fingers.  The clip is read at its start (``Animator.Play(clip)``; the scene
    saves no time for it), rotations only.  A row whose clip did not convert
    is a note, not a silent pass.
    """
    hands = _manifest().get("hands")
    out: dict = {}
    notes: list = []
    for kind, ptn in enumerate(list(hand_patterns or ())[:2]):
        key = f"{kind}/{int(ptn)}"
        if hands is None:
            notes.append("Studio 手型還沒轉檔（tools/bake_hanim.py --studio）")
            break
        if key not in hands:
            continue
        name = hands[key]
        clip = _clip(name) if name else None
        if clip is None:
            notes.append(f"手型 {key} 沒有轉出")
            continue
        for bone, ch in clip.sample(0.0).items():
            if ch.get("rot") is not None:
                out[bone] = {"rot": ch["rot"]}
    return out, notes


def h_spec(row, params: dict) -> dict:
    """What :func:`h_runtime.studio_pose` needs of an ``HAnime_*`` row.

    The pack is the override bundle's stem under ``h/anim/<sex>/`` — the key the
    H bake already files that controller's states under — and ``params`` the
    base controller's parameter names (``None`` when its bundle was not there
    to read).
    """
    bundle, controller = played_controller(row)
    sex, pack = _sex_pack(bundle)
    base_sex, base_pack = _sex_pack(row.base_bundle) if row.base_bundle else ("", "")
    return {"sex": sex, "pack": pack, "controller": controller,
            "state": row.state, "params": params.get((row.base_bundle, row.base_controller)),
            "breast_layer": row.breast_layer, "is_motion": row.is_motion, "name": row.name,
            "base_sex": base_sex, "base_pack": base_pack, "base_controller": row.base_controller}


def base_of(row: dict) -> tuple:
    """``(sex, pack, controller)`` of the base controller an ``HAnime_*`` spec's override lies over.

    ``OCIChar.LoadAnime`` makes the override with ``SetupAnimatorOverrideController``: an
    ``AnimatorOverrideController`` **on the base**, each clip of the override file put over the
    base's clip of the same name.  The states are the base's, so ``Play(state)`` on a state the
    override file does not have plays the base's own (38 rows on this install: every one of their
    states is a base state, §7.1153).  The override files' clips are their states one for one
    (§7.112), so an override state is found under its own name first.
    """
    return (str(row.get("base_sex") or row.get("sex") or ""), str(row.get("base_pack") or ""),
            str(row.get("base_controller") or ""))


def played_controller(row) -> tuple[str, str]:
    """``(bundle, controller)`` whose state machine an ``HAnime_*`` row plays.

    ``OCIChar.LoadAnime`` loads the base controller and lays the override over
    it only when ``overrideFile.check`` — a row with no override plays the base.
    """
    if row.bundle and row.controller:
        return row.bundle, row.controller
    return row.base_bundle, row.base_controller


def _sex_pack(bundle: str) -> tuple[str, str]:
    """``("female" | "male" | "", pack)`` of an ``h/anim/<sex>/<pack>.unity3d`` path."""
    parts = Path(bundle).as_posix().split("/")
    sex = "female" if "female" in parts else "male" if "male" in parts else ""
    return sex, Path(bundle).stem


def studio_h_states() -> dict:
    """``{(sex letter, pack): {(controller, state)}}`` every ``HAnime_*`` row plays.

    The H bake keeps these states even where its own stage filter would not
    (``hanim.bake_h_clips``): Studio plays ``Pull``, ``Drink``, ``Orgasm_A``… that
    no phase of the H scene does.
    """
    from .studio_catalog import catalog
    out: dict = {}
    for r in catalog().anime.values():
        if r.kind != "h":
            continue
        bundle, controller = played_controller(r)
        sex, pack = _sex_pack(bundle)
        out.setdefault((sex[:1] or "x", pack), set()).add((controller, r.state))
        # the base controller's own state too: the one ``Play`` reaches when the override has none
        # by that name (:func:`base_of`, E1f)
        if r.base_bundle and r.base_controller:
            bsex, bpack = _sex_pack(r.base_bundle)
            out.setdefault((bsex[:1] or "x", bpack), set()).add((r.base_controller, r.state))
    return out


def _base_params(wanted: set, root: Path, log) -> dict:
    """``{(bundle, controller): [parameter names]}`` for the H rows' base controllers."""
    import UnityPy

    from .hlist import controller_params
    out: dict = {}
    by_bundle: dict[str, set] = {}
    for bundle, ctrl in wanted:
        by_bundle.setdefault(bundle, set()).add(ctrl)
    for bundle, ctrls in sorted(by_bundle.items()):
        path = root / bundle
        if not path.is_file():
            log(f"  ! {bundle} 不在 assets_cache：它的 {len(ctrls)} 個 base controller 的參數讀不到")
            continue
        for o in UnityPy.load(str(path)).objects:
            if o.type.name != "AnimatorController" or o.peek_name() not in ctrls:
                continue
            out[(bundle, o.peek_name())] = controller_params(o.read_typetree())
    return out


def h_row(anime: dict | None) -> dict | None:
    """The recorded ``HAnime_*`` spec for a scene's ``anime_info``, ``None`` for anything else."""
    if not anime:
        return None
    key = f"{anime.get('group')}/{anime.get('category')}/{anime.get('no')}"
    got = (_manifest().get("rows") or {}).get(key)
    return got.get("h") if isinstance(got, dict) else None


def lookup_names(row, clips: dict) -> dict:
    """``{"": name}`` for a plain state, ``{"S_": …, "M_": …, "L_": …}`` for a size tree, ``{}`` for neither."""
    pack = Path(row.bundle).stem
    plain = clip_name(pack, row.controller, row.state)
    if plain in clips:
        return {"": plain}
    sized = {p: clip_name(pack, row.controller, p + row.state) for p in SIZE_PREFIXES}
    return sized if all(n in clips for n in sized.values()) else {}


_CACHE: dict = {}


def _manifest() -> dict:
    got = _CACHE.get("m")
    if got is None:
        migrate_baked_names()
        got = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
        _CACHE["m"] = got
    return got


def _clip(name: str):
    from .hanim import _clip_from_json
    got = _CACHE.get(name)
    if got is None:
        entry = (_manifest().get("clips") or {}).get(name)
        if entry is None:
            return None
        got = _clip_from_json(json.loads((STUDIO_DIR / entry["file"]).read_text(encoding="utf-8")))
        _CACHE[name] = got
    return got


def pose_at(skeleton, anime: dict | None, normalized_time: float, height: float) -> tuple[dict | None, str]:
    """``(additive pose, "")`` of a scene character's animation at its saved time, or ``(None, why)``.

    *anime* is the scene's ``anime_info`` (``group`` / ``category`` / ``no``);
    *height* is the card's ``shapeValueBody[0]``, which ``LoadAnime`` writes
    into ``height`` for a tree that reads it.
    """
    from ..anim.clip import blend_poses
    from .hanim import absolute_pose_to_additive, size_pair
    if not anime:
        return None, "場景沒有指定動作"
    key = f"{anime.get('group')}/{anime.get('category')}/{anime.get('no')}"
    rows = _manifest().get("rows")
    if rows is None:
        return None, "Studio 動作還沒轉檔（tools/bake_hanim.py --studio）"
    names = rows.get(key)
    if names is None:
        return None, f"動作 {key} 不在 Studio 的動作清單（mod 的？）"
    if "h" in names:
        return None, f"動作 {key} 是 H 動作（h_runtime.studio_pose）"
    if not names:
        return None, f"動作 {key} 是轉不了的 blend tree 或解不到 clip"
    if "" in names:
        clip = _clip(names[""])
        return absolute_pose_to_additive(skeleton, clip.sample(_at(clip, normalized_time))), ""
    lo, hi, w = size_pair(height)
    a, b = _clip(names[lo]), _clip(names[hi])
    raw = blend_poses(a.sample(_at(a, normalized_time)), b.sample(_at(b, normalized_time)), w)
    return absolute_pose_to_additive(skeleton, raw), ""


def duration(anime: dict | None, height: float) -> float | None:
    """How long one pass of a scene character's ``Anime_`` row lasts (seconds), ``None`` where it does not play.

    A size pair is a 1D blend tree on ``height``; Unity runs such a state at the
    weighted mean of its clips' lengths, so its normalized time advances by that.
    """
    from .hanim import size_pair
    if not anime:
        return None
    names = (_manifest().get("rows") or {}).get(f"{anime.get('group')}/{anime.get('category')}/{anime.get('no')}")
    if not names or "h" in names:
        return None
    if "" in names:
        clip = _clip(names[""])
        return None if clip is None else float(clip.duration)
    lo, hi, w = size_pair(height)
    a, b = _clip(names[lo]), _clip(names[hi])
    if a is None or b is None:
        return None
    return float(a.duration) * (1.0 - w) + float(b.duration) * w


def _at(clip, normalized_time: float) -> float:
    """The clip time of a normalized time; a looping state wraps, as ``Play(name, 0, t)`` does."""
    u = float(normalized_time or 0.0)
    return (u % 1.0) * clip.duration if u >= 1.0 else max(0.0, u) * clip.duration
