"""Which models are the playable characters', read from the game's own role tables (§7.1121, §7.1455).

The role, skin and morph tables (``ConfigDB/db_role.db``) and the model
preload table (``db_model_config_preload.db``) keep each row as a FlatBuffers
blob.  The field offsets are not guessed: they are read from the getters of
the game's own JavaScript schema for that table (``Core/Define/Config/*.js``,
``__offset(this.z7,<n>)``), so a game update that moves a field moves it here
too, and a getter of an unsupported kind stops the read.

A playable character is a role row with ``RoleType`` 1, a ``RoleBody`` and
not ``IsTrial`` — women and men alike: the model extractor exports the game's
own models of every one of them (owner 2026-11-02, §7.1455); who the game
itself converts is decided elsewhere (``converted.scope``).  Each outfit (the
role's own row, its skins, its morphs) names model ids for the world
(``MeshId``) and the menus (``UiMeshId``), and a model id's preload row names
the skeletal meshes, which must be under ``/Game/Aki/Character/Role``.  A
model is one mesh (its object name is the ident, e.g. ``R2T1YangyangMd10011``)
with every outfit that uses it.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import struct
from pathlib import Path

CONFIG = "Client/Content/Aki/ConfigDB"
SCHEMA = "Client/Content/Aki/JavaScript/Core/Define/Config"
#: The game files a scan reads, as the archive names them.
FILES = (f"{CONFIG}/db_role.db", f"{CONFIG}/db_model_config_preload.db",
         f"{CONFIG}/zh-Hant/lang_multi_text.db", f"{CONFIG}/zh-Hant/lang_multi_text_1sthalf.db",
         *(f"{SCHEMA}/{name}.js" for name in ("RoleInfo", "RoleSkin", "RoleMorph", "ModelConfigPreload")))
ROLE_MESHES = "/Game/Aki/Character/Role/"


class ScopeError(ValueError):
    """A table or schema that does not read the way the scope needs."""


class Table:
    """One FlatBuffers table, its fields read through the game's own getter code."""

    def __init__(self, data: bytes):
        self.data = data
        self.root = struct.unpack_from("<I", data)[0]
        self.vtable = self.root - struct.unpack_from("<i", data, self.root)[0]

    def field(self, offset: int) -> int | None:
        length = struct.unpack_from("<H", self.data, self.vtable)[0]
        value = struct.unpack_from("<H", self.data, self.vtable + offset)[0] if offset < length else 0
        return self.root + value if value else None

    def string_at(self, position: int) -> str:
        start = position + struct.unpack_from("<I", self.data, position)[0]
        length = struct.unpack_from("<I", self.data, start)[0]
        if start + 4 + length >= len(self.data):
            raise ScopeError("a string runs past its row")
        return self.data[start + 4:start + 4 + length].decode("utf-8")

    def value(self, getter: str, schema: str):
        match = re.search(r"\b" + getter + r"\([^{}]*\)\{([^{}]*)\}", schema)
        if not match:
            raise ScopeError(f"the schema has no getter {getter}")
        code = match[1]
        offset = int(re.search(r"__offset\(this\.z7,(\d+)\)", code)[1])
        position = self.field(offset)
        if "__vector(" in code:
            if position is None:
                return []
            if "__string" not in code:
                raise ScopeError(f"{getter}: a vector of something other than strings")
            start = position + struct.unpack_from("<I", self.data, position)[0]
            count = struct.unpack_from("<I", self.data, start)[0]
            return [self.string_at(start + 4 + index * 4) for index in range(count)]
        if "__string" in code:
            return self.string_at(position) if position is not None else None
        if "readInt32" in code:
            return struct.unpack_from("<i", self.data, position)[0] if position is not None else 0
        if "readInt8" in code:
            return bool(self.data[position]) if position is not None else "return!t||" in code
        raise ScopeError(f"unsupported getter {getter}: {code}")


def _rows(database: Path, query: str):
    with sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True) as connection:
        return list(connection.execute(query))


def read_table(root: Path, table: str, schema_name: str, fields: list[str]) -> list[dict]:
    schema = (root / SCHEMA / f"{schema_name}.js").read_text(encoding="utf-8-sig")
    return [{**{name: Table(blob).value(name.lower(), schema) for name in fields},
             "row_sha256": hashlib.sha256(blob).hexdigest()}
            for blob, in _rows(root / CONFIG / "db_role.db", f'SELECT BinData FROM "{table}"')]


def read_scope(root: Path) -> dict:
    """The playable characters' role rows (``roles``), all skin rows (``skins``), and ``models``:
    ``{"ident", "package", "owners": [...]}`` for every mesh a playable outfit uses, by package;
    ``unresolved``: the outfits whose model id has no preload row (said, not dropped)."""
    roles = read_table(root, "roleinfo", "RoleInfo", ["Id", "RoleType", "IsTrial", "Name", "ParentId", "SkinId",
                                                       "MeshId", "UiMeshId", "RoleBody", "IsShow"])
    skins = read_table(root, "roleskin", "RoleSkin", ["Id", "RoleId", "Name", "MeshId", "UiMeshId", "RoleBody"])
    morphs = read_table(root, "rolemorph", "RoleMorph", ["Id", "RoleId", "SkinId", "MeshId", "UiMeshId", "RoleBody"])
    schema = (root / SCHEMA / "ModelConfigPreload.js").read_text(encoding="utf-8-sig")
    preloads = {id_: {"meshes": Table(blob).value("meshes", schema), "row_sha256": hashlib.sha256(blob).hexdigest()}
                for id_, blob in _rows(root / CONFIG / "db_model_config_preload.db",
                                       "SELECT Id,BinData FROM modelconfigpreload")}
    playable = [r for r in roles if r["RoleType"] == 1 and r["RoleBody"] and not r["IsTrial"]]
    models: dict[str, dict] = {}
    unresolved = []
    for role in playable:
        outfits = [{"Id": role["SkinId"], "Name": role["Name"], "MeshId": role["MeshId"],
                    "UiMeshId": role["UiMeshId"], "RoleBody": role["RoleBody"]}]
        outfits += [skin for skin in skins if skin["RoleId"] == role["Id"]]
        outfits += [{**morph, "Name": role["Name"], "morph_id": morph["Id"]}
                    for morph in morphs if morph["RoleId"] == role["Id"]]
        for outfit in outfits:
            for usage in ("MeshId", "UiMeshId"):
                model_id = outfit[usage]
                if not model_id:
                    continue
                if model_id not in preloads:
                    unresolved.append({"role": role["Id"], "outfit": outfit["Id"], "usage": usage, "model_id": model_id})
                    continue
                for mesh in preloads[model_id]["meshes"]:
                    if not mesh.startswith(ROLE_MESHES):
                        raise ScopeError(f"a playable role names a mesh outside {ROLE_MESHES}: {mesh}")
                    package = "Client/Content/" + mesh.removeprefix("/Game/").split(".")[0] + ".uasset"
                    item = models.setdefault(package, {"ident": mesh.rsplit(".", 1)[-1], "package": package,
                                                       "owners": []})
                    owner = {"role_id": role["Id"], "role_name_key": role["Name"], "outfit_id": outfit["Id"],
                             "outfit_name_key": outfit["Name"], "body_type": outfit["RoleBody"] or role["RoleBody"],
                             "model_id": model_id, "morph_id": outfit.get("morph_id"), "usage": usage}
                    if owner not in item["owners"]:
                        item["owners"].append(owner)
    return {"roles": playable, "skins": skins, "models": list(models.values()), "unresolved": unresolved}
