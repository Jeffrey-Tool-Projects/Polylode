"""What GF2's own data says about an exported model's renderers beyond the prefab's flags: which are weapons, and
what the prefab's timelines do with the parts it ships switched off.

**Weapons.**  ``GunWeaponData`` (the install's ``LocalCache/Data/Table``, protobuf: ``parts.rows``) lists every
weapon (field 1 its id) with its prefab (field 9, ``Player/GrozaR01/GrozaR01_WL``); the prefab's root
GameObject carries the prefab's name.  A character renderer drawn with one of those prefabs' materials — the same
material asset, by bundle and path id — is the weapon: Groza SR0105's ``c_GrozaSR0105_slg_Gun_lod0`` draws with
``cw_KseniaSR01_WL_uber``, the material of weapon 10362's prefab; Faye SSR01's pistol with ``cw_FayeSR01_WL_uber``
(weapon 10432).  A renderer drawn with a material of the outfit's own (Faye's ``c_FayeSSR01_slg_weapon_uber``) is
not named a weapon by any table and stays undecided.  Textures are no evidence: the weapon prefabs share
the game's default maps with everything else.  The table of materials is read once per ``GunWeaponData`` and bundle
index and kept in the store (``weapon_materials.json``).

**Timelines.**  The ``*_Class``, ``*_Upgrade`` and ``*_Gacha`` prefabs are cut scenes: a ``PlayableDirector`` on
the root (and sometimes on a child) plays a ``TimelineAsset``.  The character in them is shipped switched off —
its node or its renderers' nodes inactive — and what turns things on while the scene plays is the timeline: an
``ActivationTrack`` bound to a GameObject, a ``ControlTrack`` clip whose source GameObject it activates
(``active``), an ``AnimationTrack`` bound to an Animator (whose clip may set ``m_IsActive``).  :func:`timeline`
lists, for each node a part's path runs through, the tracks of every director bound to it, so a reader sees both
the prefab's own state (what the export shows) and what the scene does with it.  The character track's clips are
named ``c_{0}_Upgrade_Exclusive``: the game fills in the character when it plays the scene.

Every model's sidecar carries them (:func:`apply_timeline`), an empty ``directors`` saying the prefab has no
``PlayableDirector``: of the 223 GF2 models the scan lists, 12 have one — the 11 ``_Class``/``_Upgrade``/``_Gacha``
prefabs and a dormitory idle scene — and no combat (``slg``) prefab does (IslandLife §7.1503).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import parts as table_rows

#: bump when what the weapon table holds changes
WEAPON_VERSION = 1


def _ref(owner, pointer) -> tuple[str, int] | None:
    """A pointer's target as (CAB, path id), without loading it."""
    from .catalog import cab_key
    if pointer is None or pointer.m_PathID == 0:
        return None
    source = owner.assets_file
    name = source.name if pointer.m_FileID == 0 else source.externals[pointer.m_FileID - 1].path
    return cab_key(name), int(pointer.m_PathID)


def weapons(install: Path) -> dict[int, str]:
    """``GunWeaponData``: weapon id → its prefab's path."""
    path = Path(install) / table_rows.TABLES / "GunWeaponData.bytes"
    out = {}
    for entries in table_rows.rows(path).values():
        for entry in entries:
            if isinstance(entry.get(9), bytes) and entry.get(1):
                out[int(entry[1])] = entry[9].decode("utf-8")
    return out


def weapon_materials(catalog, root: Path, log=print) -> dict[str, list[dict]]:
    """``"<cab>/<path id>"`` of every material a weapon prefab's renderers draw with → the weapons
    (``{"weapon": id, "prefab": path}``); cached in *root* by the table's and the bundle index's hashes."""
    table = Path(catalog.install) / table_rows.TABLES / "GunWeaponData.bytes"
    index = Path(root) / "bundle_index.jsonl"
    key = hashlib.sha256(f"{WEAPON_VERSION}|".encode() + table.read_bytes()
                         + str(index.stat().st_size).encode() + str(index.stat().st_mtime_ns).encode()).hexdigest()
    cache = Path(root) / "weapon_materials.json"
    if cache.is_file():
        held = json.loads(cache.read_text(encoding="utf-8"))
        if held.get("key") == key:
            return held["materials"]
    by_name: dict[str, list[dict]] = {}
    for entry in catalog.entries:
        by_name.setdefault(entry["name"], []).append(entry)
    log("讀武器表的預製物材質（第一次）…")
    env = catalog.environment()
    found: dict[str, list[dict]] = {}
    for weapon, prefab in sorted(weapons(catalog.install).items()):
        for entry in by_name.get(prefab.rsplit("/", 1)[-1], []):
            stack = [env.find_file(entry["cab"]).objects[entry["id"]]]
            while stack:
                game_object = stack.pop().read()
                for item in game_object.m_Component:
                    component = item.component.deref()
                    if component.type.name in ("SkinnedMeshRenderer", "MeshRenderer"):
                        for pointer in component.read().m_Materials:
                            target = _ref(component, pointer)
                            if target is not None:
                                row = {"weapon": weapon, "prefab": prefab}
                                listed = found.setdefault(f"{target[0]}/{target[1]}", [])
                                if row not in listed:
                                    listed.append(row)
                    elif component.type.name in ("Transform", "RectTransform"):
                        stack.extend(child.read().m_GameObject.deref() for child in component.read().m_Children)
    from ..core.paths import write_json_atomic
    write_json_atomic(cache, {"key": key, "materials": found})
    return found


def _script(component) -> str:
    return component.read().m_Script.deref().read().m_ClassName


def timeline(catalog, entry: dict) -> dict:
    """The prefab *entry*'s directors, its switched-off nodes, and for each node a director binds or a control
    clip sources what the track does there: ``{"directors": [...], "inactive": [path, ...],
    "nodes": {path: [track, ...]}}`` (paths below the prefab's root, the root itself "")."""
    env = catalog.environment()

    def target(owner, pointer):
        if not pointer or pointer["m_PathID"] == 0:
            return None
        source = owner.assets_file
        if pointer["m_FileID"] != 0:
            source = env.find_file(source.externals[pointer["m_FileID"] - 1].path)
        return source.objects.get(pointer["m_PathID"])

    # a node's path is the prefab's own path below its root (``fx/FayeSSR01``), as a part's ``path`` is; the root's
    # is ""
    paths, inactive, directors = {}, [], []
    stack = [(env.find_file(entry["cab"]).objects[entry["id"]], None)]
    while stack:
        reader, parent = stack.pop()
        game_object = reader.read()
        path = "" if parent is None else f"{parent}/{game_object.m_Name}" if parent else game_object.m_Name
        paths[reader.path_id] = path
        if not game_object.m_IsActive:
            inactive.append(path)
        for item in game_object.m_Component:
            component = item.component.deref()
            if component.type.name == "PlayableDirector":
                directors.append((path, component))
            elif component.type.name in ("Transform", "RectTransform"):
                stack.extend((child.read().m_GameObject.deref(), path) for child in component.read().m_Children)
    out = {"directors": [], "inactive": sorted(inactive), "nodes": {}}

    def note(game_object_id, row):
        path = paths.get(game_object_id)
        if path is not None:
            out["nodes"].setdefault(path, []).append(row)

    for path, director in directors:
        tree = director.read_typetree()
        asset = target(director, tree["m_PlayableAsset"])
        asset_tree = asset.read_typetree() if asset is not None else {}
        out["directors"].append({"node": path, "timeline": asset_tree.get("m_Name"),
                                 "duration": asset_tree.get("m_FixedDuration"),
                                 "plays_on_awake": bool(tree.get("m_InitialState"))})
        exposed = dict(tree.get("m_ExposedReferences", {}).get("m_References", []))
        for binding in tree.get("m_SceneBindings", []):
            bound, track = target(director, binding["value"]), target(director, binding["key"])
            if bound is None:
                continue
            # a binding whose track is not an object of the files read (a null key) still binds the object
            track_tree = track.read_typetree() if track is not None else {}
            holder = bound.path_id if bound.type.name == "GameObject" else bound.read().m_GameObject.path_id
            note(holder, {"director": path, "track": _script(track) if track is not None else None,
                          "name": track_tree.get("m_Name"), "bound": bound.type.name,
                          "clips": [{"start": round(float(c["m_Start"]), 4), "duration": round(float(c["m_Duration"]), 4),
                                     "name": c.get("m_DisplayName")} for c in track_tree.get("m_Clips", [])]})
        tracks = [target(asset, pointer) for pointer in asset_tree.get("m_Tracks", [])]
        while tracks:
            track = tracks.pop()
            if track is None:
                continue
            track_tree = track.read_typetree()
            tracks.extend(target(track, pointer) for pointer in track_tree.get("m_Children", []))
            if _script(track) != "ControlTrack":
                continue
            for clip in track_tree.get("m_Clips", []):
                control = target(track, clip["m_Asset"])
                if control is None:
                    continue
                control_tree = control.read_typetree()
                name = (control_tree.get("sourceGameObject") or {}).get("exposedName")
                source = target(director, exposed.get(str(name))) if name is not None else None
                if source is not None:
                    note(source.path_id, {"director": path, "track": "ControlTrack", "name": track_tree.get("m_Name"),
                                          "bound": "GameObject (control clip source)",
                                          "activates": bool(control_tree.get("active")),
                                          "post_playback": control_tree.get("postPlayback"),
                                          "clips": [{"start": round(float(clip["m_Start"]), 4),
                                                     "duration": round(float(clip["m_Duration"]), 4),
                                                     "name": clip.get("m_DisplayName")}]})
    return out


def apply_timeline(document: dict, facts: dict) -> None:
    """Write *facts* (:func:`timeline`) into a scene *document*: its ``timeline`` (the prefab's directors, ``[]`` when
    it has none, and its switched-off nodes) and every part's ``timeline`` (:func:`path_facts`)."""
    document['timeline'] = {'directors': facts['directors'], 'inactive': facts['inactive']}
    for part in document['parts']:
        part['timeline'] = path_facts(part.get('path', ''), facts)


def path_facts(part_path: str, facts: dict) -> dict:
    """For a part at *part_path* (the prefab's own path below its root): the switched-off nodes it runs through
    and the timeline tracks bound to the root or any node on the way."""
    steps = part_path.split("/")
    on_path = ["/".join(steps[:k]) for k in range(1, len(steps) + 1)]
    off = [path for path in on_path if path in set(facts["inactive"])]
    tracks = [dict(row, node=path) for path in [""] + on_path for row in facts["nodes"].get(path, [])]
    activated = [path for path in off if any(row["track"] == "ActivationTrack" or row.get("activates")
                                             for row in facts["nodes"].get(path, []))]
    return {"inactive_nodes": off, "tracks": tracks,
            "activated_by_timeline": bool(off) and len(activated) == len(off)}
