"""Each source's statement for an FBX's sidecar (``interchange.sidecar``): which model of which install it is, the
hashes of the files it was read from, and where its human-bone table comes from — or why it has none.

Only what the install and the conversion say is written; a value they do not give is ``None`` with the reason.
(The character's sex and age are not stated here: the program that reads the sidecar copies them from its own
reviewed lists.)
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _sha256(path: Path, start: int = 0, length: int | None = None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        stream.seek(start)
        left = length
        while left is None or left > 0:
            chunk = stream.read(1 << 20 if left is None else min(1 << 20, left))
            if not chunk:
                break
            digest.update(chunk)
            if left is not None:
                left -= len(chunk)
    return digest.hexdigest()


def steam_build(install: Path) -> dict | None:
    """The Steam build the install is, from the library's ``appmanifest_<id>.acf`` whose ``installdir`` is the
    install's folder under ``steamapps/common``; ``None`` for an install Steam does not manage."""
    install = Path(install).resolve()
    for folder in (install, *install.parents):
        if folder.parent.name.lower() == "common" and folder.parent.parent.name.lower() == "steamapps":
            library = folder.parent.parent
            for manifest in sorted(library.glob("appmanifest_*.acf")):
                text = manifest.read_text(encoding="utf-8", errors="replace")
                fields = dict(re.findall(r'"(\w+)"\s+"([^"]*)"', text))
                if fields.get("installdir", "").lower() == folder.name.lower():
                    return {"source": manifest.name, "appid": fields.get("appid"), "buildid": fields.get("buildid"),
                            "last_updated": fields.get("LastUpdated")}
            return None
    return None


# -- Endfield -------------------------------------------------------------------------------------------------

def bundle_slices(cabs, names: list[str], install: Path) -> list[dict]:
    """Where each of the serialized files *names* is in the install's VFS chunks, and the SHA-256 of its bundle's
    bytes: from its offset to the next bundle the CAB maps list in the same chunk (or the chunk's end).  A whole
    chunk is up to 1.6 GB, so the chunk itself is named and not hashed."""
    install = Path(install).resolve()
    starts: dict[str, list[int]] = {}
    for entry in cabs._entries.values():
        starts.setdefault(str(entry.source).lower(), []).append(int(entry.offset))
    for key in starts:
        starts[key] = sorted(set(starts[key]))
    out, seen = [], set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        try:
            entry = cabs.entry(name)
        except LookupError as missing:
            out.append({"cab": name, "file": None, "reason": str(missing)})
            continue
        source = Path(entry.source)
        later = [o for o in starts[str(source).lower()] if o > entry.offset]
        end = later[0] if later else source.stat().st_size
        try:
            relative = source.resolve().relative_to(install).as_posix()
        except ValueError:
            relative = str(source)
        out.append({"cab": name, "file": relative, "offset": int(entry.offset), "length": int(end - entry.offset),
                    "sha256": _sha256(source, entry.offset, end - entry.offset)})
    return out


def endfield_statement(model, install: Path, pipeline, extra: dict, archive: Path) -> dict:
    """An Endfield prefab's identity (its container, model asset and Avatar; the bundles of its prefab, Avatar and
    maps, hashed) and its human bones: the Avatar's own table (``endfield.components.humanoid``)."""
    from ..endfield import pipeline as PL
    from ..endfield import store
    job = next((j for j in PL.select(pipeline.assets, ["chara"]) if j.ident == model.ident), None)
    prefab = [cab for cab in (pipeline.cabs.at(*slot) for slot in (job.slots if job else [])) if cab]
    avatar = (extra.get("avatar") or {}).get("cab")
    maps = []
    item = store.read_item("chara", model.ident)
    for piece in item.pieces:
        for value in ((piece.shader or {}).get("maps") or {}).values():
            if isinstance(value, dict) and isinstance(value.get("cab"), str):
                maps.append(value["cab"])
    identity = {"game": "endfield", "polylode_ident": model.ident, "character": model.character,
                "outfit": model.outfit or None,
                "model_id": {"prefab": (model.detail or {}).get("container") or (job.container if job else None),
                             "model_asset": extra.get("model"), "avatar": extra.get("avatar")},
                "install": {"path": str(install), "steam": steam_build(install)},
                "source_files": bundle_slices(pipeline.cabs, [*prefab, *([avatar] if avatar else []),
                                                             *sorted(set(maps))], install),
                "source_files_basis": "the bundle bytes of the prefab, its Avatar and its maps in the VFS chunks",
                "converted_archive": {"file": Path(archive).name,
                                      "sha256": _sha256(archive) if Path(archive).is_file() else None}}
    definition = (extra.get("human") or {}).get("definition") or {}
    statement = {"basis": "unity_avatar",
                 "source": {"avatar": extra.get("avatar"), "table": "m_Avatar.m_Human.m_HumanBoneIndex through "
                            "m_TOS to the prefab's transforms (endfield.components.humanoid)"},
                 "reason": "the prefab's Animator has no Avatar, or its Avatar maps no bone of this prefab"}
    if definition and not definition.get("m_HasLeftHand") and not definition.get("m_HasRightHand"):
        statement["fingers"] = {"bones": {}, "reason": "the Avatar declares no hands (m_HasLeftHand and "
                                                       "m_HasRightHand are false)"}
    elif definition:
        statement["fingers"] = {"bones": {}, "reason": "the Avatar's hand bones (m_HandBoneIndex) are not read"}
    return {"identity": identity, "humanoid": statement}


# -- GF2 ------------------------------------------------------------------------------------------------------

def gf2_version(install: Path) -> dict:
    """The resource and code versions the GF2 client wrote into ``LocalCache/Data/GameConfig.cfg``."""
    path = Path(install) / "GF2_Exilium_Data" / "LocalCache" / "Data" / "GameConfig.cfg"
    if not path.is_file():
        return {"source": None, "reason": f"{path} is not there"}
    fields = dict(line.split(":", 1) for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                  if ":" in line)
    return {"source": "GF2_Exilium_Data/LocalCache/Data/GameConfig.cfg",
            **{key: fields.get(key, "").strip() or None for key in ("CacheResVersion", "CodeResVersion")}}


def gf2_statement(model, install: Path, ident: str, extra: dict, archive: Path, unity: dict | None) -> dict:
    """A GF2 prefab's identity (its bundle entry, the roster rows it belongs to, the bundles it was read from with
    their hashes) and its human bones: Unity Avatars only — GF2's character prefabs carry Generic ones."""
    identity = {"game": "gf2", "polylode_ident": model.ident, "converted_ident": ident,
                "character": model.character, "outfit": model.outfit or None,
                "model_id": {"prefab": (extra.get("entry") or {}).get("name"), "entry": extra.get("entry"),
                             "roster": extra.get("characters")},
                "install": {"path": str(install), "unity": unity, "version": gf2_version(install)},
                "source_files": extra.get("source_files"),
                "source_files_basis": "every bundle the conversion read, with the member offset (gf2.convert)",
                "converted_archive": {"file": Path(archive).name,
                                      "sha256": _sha256(archive) if Path(archive).is_file() else None}}
    from ..gf2.convert import unwrap_data
    avatars = []
    for avatar in extra.get("avatars") or []:
        flat = unwrap_data(avatar.get("data") or {})
        human = (flat.get("m_Avatar") or {}).get("m_Human") or {}
        indices = [int(v) for v in human.get("m_HumanBoneIndex") or []]
        described = len(((flat.get("m_HumanDescription") or {}).get("m_Human")) or [])
        avatars.append({"name": flat.get("m_Name"), "identity": avatar.get("identity"),
                        "human_description_bones": described,
                        "mapped_human_bones": sum(1 for v in indices if v >= 0)})
    generic = avatars and all(not a["human_description_bones"] and not a["mapped_human_bones"] for a in avatars)
    statement = {"basis": "unity_avatar",
                 "source": {"avatars": avatars, "table": "m_Avatar.m_Human.m_HumanBoneIndex through m_TOS"},
                 "reason": ("the prefab's Avatar is Generic: its HumanDescription lists no bones and every "
                            "m_HumanBoneIndex is -1; the prefab has no other component naming its bones"
                            if generic else "the prefab's Animator has no Avatar" if not avatars
                            else "the Avatar maps no bone of this prefab")}
    return {"identity": identity, "humanoid": statement}


# -- Wuthering Waves ------------------------------------------------------------------------------------------

def wuwa_version(install: Path) -> dict:
    """The version the launcher wrote into ``launcherDownloadConfig.json`` (per resource bundle)."""
    path = Path(install) / "launcherDownloadConfig.json"
    if not path.is_file():
        return {"source": None, "reason": f"{path} is not there"}
    data = json.loads(path.read_text(encoding="utf-8"))
    bundles = {name: value.get("version") for name, value in (data.get("bundles") or {}).items()}
    return {"source": "launcherDownloadConfig.json", "version": data.get("version") or None, "bundles": bundles}


def wuwa_statement(model, install: Path, sources: Path, case_document: dict, skeleton: dict) -> dict:
    """A display model's identity (its menu ids, the parts it is assembled from with the hashes of their packages)
    and its human bones: the biped's fixed names checked against the hierarchy (``source_scene.humanoid``), the
    game's skeleton having no humanoid table of its own (*skeleton*: what its ``USkeleton`` was found to hold)."""
    from ..wuthering_waves.extract import install_stamp
    parts = []
    for part in (model.detail or {}).get("parts") or []:
        ident = part["package"].rsplit("/", 1)[-1]
        provenance = sources / ident / "source_provenance.json"
        files = json.loads(provenance.read_text(encoding="utf-8")) if provenance.is_file() else None
        parts.append({**part, "files": [{"path": f.get("Path"), "archive": Path(f.get("archive") or "").name or None,
                                         "bytes": f.get("Size"), "sha256": f.get("sha256")} for f in files or []],
                      **({} if files is not None else {"reason": f"{provenance} is not there"})})
    identity = {"game": "wuwa", "polylode_ident": model.ident, "character": model.character,
                "outfit": model.outfit or None,
                "model_id": {"display_model_ids": (model.detail or {}).get("model_ids"),
                             "default_display_model_id": case_document.get("default_display_model_id"),
                             "packages": [p["package"] for p in parts]},
                "install": {"path": str(install), "version": wuwa_version(install),
                            "archives_stamp": install_stamp(install)},
                "source_files": parts,
                "source_files_basis": "every package file of every part, hashed as the reader took it out"}
    statement = {"basis": "biped_naming_with_parent_chain",
                 "source": {"rule": "wuthering_waves.source_scene.humanoid: the 3ds Max Biped's fixed bone names "
                                    "(Bip001Pelvis …), each limb checked to be the child of the one before it",
                            "game_skeleton": skeleton},
                 "reason": "the rig is not the native biped"}
    return {"identity": identity, "humanoid": statement}


def wuwa_skeleton(mesh_json: list | dict, skeleton_json: list | dict | None) -> dict:
    """What the mesh's ``USkeleton`` says about human bones: Unreal keeps a humanoid rig (``RigConfig``) and
    retarget sources on a skeleton; a cooked one may keep neither."""
    items = mesh_json if isinstance(mesh_json, list) else [mesh_json]
    mesh = next((e for e in items if isinstance(e, dict) and e.get("Type") == "SkeletalMesh"), {})
    path = ((mesh.get("Properties") or {}).get("Skeleton") or {}).get("ObjectPath")
    if skeleton_json is None:
        return {"skeleton": path, "read": False}
    items = skeleton_json if isinstance(skeleton_json, list) else [skeleton_json]
    skeleton = next((e for e in items if isinstance(e, dict) and e.get("Type") == "Skeleton"), {})
    properties = skeleton.get("Properties") or {}
    return {"skeleton": path, "read": True,
            "rig_config": properties.get("RigConfig") or skeleton.get("RigConfig"),
            "anim_retarget_sources": len(skeleton.get("AnimRetargetSources") or {}),
            "sockets": len(properties.get("Sockets") or [])}
