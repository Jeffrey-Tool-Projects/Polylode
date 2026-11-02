"""One display model per primary mesh, put together the way the game's character menu does (§7.1121).

The model table (:data:`MODEL_TABLE`, read as JSON by the game-file reader)
gives each model id a primary mesh, its UI child meshes (each with
``DefaultVisible``) and its UI decorations (a mesh on a socket, with a
transform).  Every outfit model id of one primary is one display model: the
default outfit's recipe decides which parts show, and the parts only other
recipes use are kept hidden.  A role's own UI model is dropped when that
role's default skin picks a different complete display model — the menu
never shows it (:func:`drop_superseded_carriers`).

The column names are the table's own (its struct's field names); a missing
column stops the read rather than being skipped.
"""
from __future__ import annotations

import collections

MODEL_TABLE = "Client/Content/Aki/Data/Entity/Sub_DT_ModelConfig/DT_ModelConfig_Player"
MESH = "网格体_168_BEB7464046E518BA05D4C799C3CC4633"
UI_CHILD = "UI用子网格体_239_BA84BC02424D649817028F908F3098BC"
UI_CHILD_MESH = "SkeletalMesh_29_7018A6E146FFB04877B240A839BA6715"
CHILD_NAME = "ComponentName_18_7B4E2D19480D94C3B7A2D08B0BEF03CC"
CHILD_VISIBLE = "DefaultVisible_23_1AE797964BA1929C2D63C3A41010B915"
DECORATION = "UiModelDecorationArray_219_69CDBE4941E99643335E1B894A99EA66"
DECORATION_MESH = "SkeletalMesh_8_F58B41EB40ED76B40BB3CA920B0B58DB"
DECORATION_SOCKET = "SocketName_9_732CC00348A5405A1B5B2193F186F032"
DECORATION_TRANSFORM = "Transform_5_D5FB56E94DB8B07C32C25FB5FFBE50AE"


def package_path(value) -> str | None:
    """``/Game/X/Y.Y`` (or a soft pointer to it) → ``Client/Content/X/Y``; ``None`` for no asset."""
    if isinstance(value, dict):
        value = value.get("AssetPathName") or value.get("ObjectPath")
    if not value or value == "None":
        return None
    value = value.split(".")[0]
    return "Client/Content/" + value.removeprefix("/Game/") if value.startswith("/Game/") else value


def ui_assemblies(scope: dict, model_rows: dict) -> list[dict]:
    """For every UI model id a woman's outfit uses: its primary mesh and its UI parts, with who uses it."""
    owners: dict[int, list] = {}
    for model in scope["models"]:
        for owner in model["owners"]:
            if owner["usage"] == "UiMeshId" and owner not in owners.setdefault(owner["model_id"], []):
                owners[owner["model_id"]].append(owner)
    assemblies = []
    for model_id, provenance in sorted(owners.items()):
        config = model_rows[str(model_id)]
        primary = package_path(config[MESH])
        parts = [dict(package=primary, component="CharacterMesh0", hidden=False, source="model_table_primary")]
        for child in config[UI_CHILD]:
            parts.append(dict(package=package_path(child[UI_CHILD_MESH]),
                              component=child.get(CHILD_NAME, "model_table_child"),
                              hidden=not child.get(CHILD_VISIBLE, True), source="model_table_child"))
        for child in config[DECORATION]:
            parts.append(dict(package=package_path(child[DECORATION_MESH]), component="ui_decoration", hidden=False,
                              source="ui_model_decoration", attachment_socket=child[DECORATION_SOCKET],
                              attachment_transform=child[DECORATION_TRANSFORM]))
        seen, unique = set(), []
        for part in parts:
            if part["package"] and part["package"] not in seen:
                seen.add(part["package"])
                unique.append(part)
        assemblies.append(dict(model_id=model_id, primary=primary, owners=provenance, parts=unique))
    return assemblies


def display_models(scope: dict, assemblies: list[dict]) -> list[dict]:
    """One row per primary: ``ident``, its ``parts`` (primary first) each ``hidden`` or not, ``model_ids``, ``owners``."""
    grouped = collections.defaultdict(list)
    for row in assemblies:
        grouped[row["primary"]].append(row)
    role_defaults = {row["UiMeshId"] for row in scope["roles"]}
    default_skin_ids = {row["SkinId"] for row in scope["roles"]}
    skin_defaults = {row["UiMeshId"] for row in scope["skins"] if row["Id"] in default_skin_ids}
    rows = []
    for primary, recipes in sorted(grouped.items()):
        recipes.sort(key=lambda row: (row["model_id"] not in skin_defaults, row["model_id"] not in role_defaults,
                                      row["model_id"]))
        default = recipes[0]
        chosen = {part["package"]: part for part in default["parts"]}
        parts = {primary: default["parts"][0]}
        for recipe in recipes:
            for part in recipe["parts"]:
                previous = parts.get(part["package"])
                if previous is None:
                    parts[part["package"]] = part
                elif any(previous.get(key) != part.get(key) for key in ("attachment_socket", "attachment_transform")):
                    raise ValueError(f"{primary}: {part['package']} sits differently in two recipes")
        ordered = [parts.pop(primary), *sorted(parts.values(), key=lambda part: part["package"])]
        records = [dict(part, hidden=chosen[part["package"]]["hidden"] if part["package"] in chosen else True)
                   for part in ordered]
        provenance = []
        for recipe in recipes:
            for owner in recipe["owners"]:
                if owner not in provenance:
                    provenance.append(owner)
        rows.append(dict(ident=primary.rsplit("/", 1)[-1], parts=records, model_ids=[r["model_id"] for r in recipes],
                         default_model_id=default["model_id"], owners=provenance))
    return drop_superseded_carriers(rows, scope)


def drop_superseded_carriers(rows: list[dict], scope: dict) -> list[dict]:
    """Drop a model only a role's own row names when that role's default skin shows another complete model."""
    roles = {r["Id"]: r for r in scope["roles"]}
    skins = {r["Id"]: r for r in scope["skins"]}
    by_model = {model: row["ident"] for row in rows for model in row["model_ids"]}
    shown = {o["role_id"] for r in rows for o in r["owners"]}     # roles with a model before any is dropped
    kept = []
    for row in rows:
        for owner in row["owners"]:
            role = roles[owner["role_id"]]
            skin = skins.get(role["SkinId"])
            superseded = (skin is not None and owner.get("morph_id") is None and owner["outfit_id"] == role["SkinId"]
                          and owner["model_id"] == role["UiMeshId"] and skin["UiMeshId"] != role["UiMeshId"]
                          and skin["UiMeshId"] in by_model and by_model[skin["UiMeshId"]] != row["ident"])
            if not superseded:
                kept.append(row)
                break
    if {o["role_id"] for r in kept for o in r["owners"]} != shown:
        raise ValueError("dropping the superseded carriers left a role with no model")
    return kept
