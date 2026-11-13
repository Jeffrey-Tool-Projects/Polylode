"""What GF2's data says beyond a prefab's flags (``gf2/scene_facts``): the weapon table and a cut scene's timeline."""
from polylode.gf2 import parts, scene_facts
from test_gf2_parts import entry, table


def test_the_weapon_table_names_each_weapons_prefab(tmp_path):
    folder = tmp_path / parts.TABLES
    folder.mkdir(parents=True)
    table(folder / "GunWeaponData.bytes", {0: [entry(f1=11010, f8="Weapon_OTS14_3", f9="Player/GrozaR01/GrozaR01_WL"),
                                               entry(f1=10362, f9="Player/KseniaSR01/KseniaSR01_WL")]})
    assert scene_facts.weapons(tmp_path) == {11010: "Player/GrozaR01/GrozaR01_WL",
                                             10362: "Player/KseniaSR01/KseniaSR01_WL"}


def test_a_parts_path_gathers_its_switched_off_nodes_and_the_tracks_on_its_way():
    control = {"director": "", "track": "ControlTrack", "name": "c", "bound": "GameObject (control clip source)",
               "activates": True, "clips": [{"start": 0.0, "duration": 7.6, "name": "x"}]}
    animation = {"director": "", "track": "AnimationTrack", "name": "a", "bound": "Animator", "clips": []}
    facts = {"inactive": ["VFX/Mirage", "VFX/Other"], "nodes": {"VFX/Mirage": [control], "": [animation]}}
    found = scene_facts.path_facts("VFX/Mirage/body_lod0", facts)
    assert found["inactive_nodes"] == ["VFX/Mirage"] and found["activated_by_timeline"] is True
    assert [(t["node"], t["track"]) for t in found["tracks"]] == [("", "AnimationTrack"), ("VFX/Mirage", "ControlTrack")]
    # a node no track switches on stays off while the scene plays; a part under no switched-off node has none
    facts["nodes"]["VFX/Mirage"] = [animation]
    assert scene_facts.path_facts("VFX/Mirage/body_lod0", facts)["activated_by_timeline"] is False
    assert scene_facts.path_facts("Body/body_lod0", facts) == {"inactive_nodes": [], "tracks": [
        dict(animation, node="")], "activated_by_timeline": False}
