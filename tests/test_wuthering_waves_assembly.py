"""The character menu's display models, put together from the model table (``wuthering_waves/assembly.py``)."""
import pytest

from polylode.wuthering_waves import assembly as A


def _config(primary, children=(), decorations=()):
    return {A.MESH: {"AssetPathName": f"/Game/Aki/Character/Role/Female/{primary}.{primary}"},
            A.UI_CHILD: [{A.UI_CHILD_MESH: f"/Game/Aki/Character/Role/Female/{name}.{name}", A.CHILD_VISIBLE: shown}
                         for name, shown in children],
            A.DECORATION: [{A.DECORATION_MESH: f"/Game/Aki/Deco/{name}.{name}", A.DECORATION_SOCKET: socket,
                            A.DECORATION_TRANSFORM: {"t": 1}} for name, socket in decorations]}


def _owner(role, outfit, model_id):
    return {"role_id": role, "role_name_key": f"r{role}", "outfit_id": outfit, "outfit_name_key": f"o{outfit}",
            "model_id": model_id, "morph_id": None, "usage": "UiMeshId"}


def test_package_paths_drop_the_object_name_and_keep_no_asset_as_none():
    assert A.package_path("/Game/Aki/X/Y.Y") == "Client/Content/Aki/X/Y"
    assert A.package_path({"AssetPathName": "/Game/A.A"}) == "Client/Content/A"
    assert A.package_path("None") is None and A.package_path({}) is None


def test_the_default_skin_decides_what_shows_and_other_recipes_parts_stay_hidden():
    scope = {"roles": [{"Id": 1, "SkinId": 10, "UiMeshId": 100}],
             "skins": [{"Id": 10, "UiMeshId": 100}, {"Id": 11, "UiMeshId": 101}],
             "models": [{"owners": [_owner(1, 10, 100), _owner(1, 11, 101)]}]}
    table = {"100": _config("Body", children=[("Hair", True), ("Cape", False)], decorations=[("Glasses", "head")]),
             "101": _config("Body", children=[("Hat", True)])}
    rows = A.display_models(scope, A.ui_assemblies(scope, table))
    assert [row["ident"] for row in rows] == ["Body"] and rows[0]["model_ids"] == [100, 101]
    shown = {part["package"].rsplit("/", 1)[-1]: part["hidden"] for part in rows[0]["parts"]}
    assert shown == {"Body": False, "Cape": True, "Hair": False, "Hat": True, "Glasses": False}
    assert rows[0]["parts"][0]["package"].endswith("/Body")             # the primary first


def test_a_roles_own_model_is_dropped_when_her_default_skin_shows_another():
    scope = {"roles": [{"Id": 1, "SkinId": 10, "UiMeshId": 100}],
             "skins": [{"Id": 10, "UiMeshId": 200}],
             "models": [{"owners": [_owner(1, 10, 100), _owner(1, 10, 200)]}]}
    table = {"100": _config("Carrier"), "200": _config("Shown")}
    assert [row["ident"] for row in A.display_models(scope, A.ui_assemblies(scope, table))] == ["Shown"]


def test_one_part_on_two_sockets_stops_the_read():
    scope = {"roles": [{"Id": 1, "SkinId": 10, "UiMeshId": 100}], "skins": [{"Id": 10, "UiMeshId": 100}],
             "models": [{"owners": [_owner(1, 10, 100), _owner(1, 11, 101)]}]}
    table = {"100": _config("Body", decorations=[("Glasses", "head")]),
             "101": _config("Body", decorations=[("Glasses", "hand")])}
    with pytest.raises(ValueError, match="differently"):
        A.display_models(scope, A.ui_assemblies(scope, table))


def test_a_woman_with_no_resolved_ui_model_does_not_stop_the_scan():
    scope = {"roles": [{"Id": 1, "SkinId": 10, "UiMeshId": 100}, {"Id": 2, "SkinId": 20, "UiMeshId": 0}],
             "skins": [{"Id": 10, "UiMeshId": 100}],
             "models": [{"owners": [_owner(1, 10, 100)]}]}
    rows = A.display_models(scope, A.ui_assemblies(scope, {"100": _config("Body")}))
    assert [row["ident"] for row in rows] == ["Body"]
