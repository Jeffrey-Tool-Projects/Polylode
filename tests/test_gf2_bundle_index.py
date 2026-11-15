import json

from polylode.gf2 import bundles


def test_a_bundle_the_install_no_longer_has_leaves_the_index(tmp_path, monkeypatch):
    install = tmp_path / "install"
    folder = install / "GF2_Exilium_Data" / "LocalCache" / "Data" / "AssetBundles_Windows"
    folder.mkdir(parents=True)
    kept = folder / "kept.bundle"
    kept.write_bytes(b"x")
    stat = kept.stat()
    relative = "GF2_Exilium_Data/LocalCache/Data/AssetBundles_Windows/"
    index = tmp_path / "bundle_index.jsonl"
    rows = [{"path": relative + "kept.bundle", "stamp": [stat.st_size, stat.st_mtime_ns], "members": []},
            {"path": relative + "renamed_away.bundle", "stamp": [1, 1], "members": [{"offset": 0, "cabs": [], "containers": []}]}]
    index.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    monkeypatch.setattr(bundles, "configure_unity", lambda _install: {"version": "2019.4.29f1", "source": "test"})
    monkeypatch.setattr(bundles, "index_row", lambda *a, **k: (_ for _ in ()).throw(AssertionError("nothing to read")))
    bundles.scan(install, index)
    left = [json.loads(line)["path"] for line in index.read_text(encoding="utf8").splitlines()]
    assert left == [relative + "kept.bundle"]
