"""A model shared on the user's GitHub Pages site (``extractor/share.py``)."""
import json
import subprocess

import pytest

from polylode.extractor import preview, share


def test_a_github_remote_gives_its_pages_address():
    assert share.site_url("https://github.com/Me/models.git") == "https://me.github.io/models/"
    assert share.site_url("git@github.com:Me/models.git") == "https://me.github.io/models/"
    assert share.site_url("https://token@github.com/Me/models") == "https://me.github.io/models/"
    assert share.site_url("https://github.com/Me/me.github.io.git") == "https://me.github.io/"   # the user site
    with pytest.raises(share.ShareError):
        share.site_url("https://gitlab.com/me/models.git")


def test_a_models_folder_name_drops_the_exports_hash():
    assert share.slug(preview.Path("c_LennaSSR01_slg_skin_f4bb31ee2e.glb")) == "c_lennassr01_slg_skin"
    assert share.slug(preview.Path("心 (2).glb")) == "2"
    assert share.slug(preview.Path("心.glb")) == "model"


def test_the_programs_own_folder_is_never_a_site(tmp_path):
    with pytest.raises(share.ShareError, match="本程式自己的資料夾"):
        share.check_site(share._own_tree())
    with pytest.raises(share.ShareError, match="不存在"):
        share.check_site(tmp_path / "missing")


def _git(folder, *arguments):
    subprocess.run(["git", "-C", str(folder), *arguments], check=True, capture_output=True)


def test_a_shared_model_is_written_committed_alone_and_pushed(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "derived_root", lambda: tmp_path / "derived")
    # the card is the preview's drawing, which needs OpenGL: here a stand-in image
    monkeypatch.setattr(preview, "snapshot", lambda path, out, w, h: out.write_bytes(b"\x89PNG card") and out)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    site = tmp_path / "site"
    site.mkdir()
    _git(site, "init")
    _git(site, "config", "user.email", "me@example.com")
    _git(site, "config", "user.name", "Me")
    _git(site, "remote", "add", "origin", "https://github.com/Me/models.git")
    _git(site, "remote", "set-url", "--push", "origin", str(remote))      # pushes go to the local stand-in
    (site / "notes.txt").write_text("the user's own, not shared")
    _git(site, "add", "notes.txt")                                     # staged by the user: stays out
    model = preview.triangle_glb(tmp_path / "c_X_slg_skin_0123456789.glb")
    lines = []
    waited = []
    monkeypatch.setattr(share, "wait_live", lambda url, stamp, log: waited.append((url, stamp)) or True)
    url, live = share.publish(model, site, log=lines.append)
    assert live and waited == [(url, json.loads((site / "models" / "c_x_slg_skin" / "model.json").read_text(
        encoding="utf-8"))["shared"])]                                 # waited for this share's own stamp
    assert url == "https://me.github.io/models/models/c_x_slg_skin/"
    folder = site / "models" / "c_x_slg_skin"
    assert (folder / "model.glb").read_bytes() == model.read_bytes()
    page = (folder / "index.html").read_text(encoding="utf-8")
    assert f'content="{url}card.png"' in page and 'src="model.glb"' in page and share.VIEWER in page
    assert json.loads((folder / "model.json").read_text(encoding="utf-8"))["title"] == "c_X_slg_skin"
    assert 'href="models/c_x_slg_skin/"' in (site / "index.html").read_text(encoding="utf-8")
    # the site serves its own viewer and Discord's SDK (an Activity reaches nothing else), licences beside them
    for file in share.VENDORED:
        assert (site / "vendor" / file).read_bytes() == (share.WEB / file).read_bytes()
    assert (site / "site.js").read_text(encoding="utf-8") == share.SITE_SCRIPT
    pushed = subprocess.run(["git", "-C", str(remote), "show", "--name-only", "--format=%s", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.split()
    assert pushed[:2] == ["Share", "c_x_slg_skin"]
    assert "notes.txt" not in pushed and "models/c_x_slg_skin/model.glb" in pushed   # only the model's files
    assert "vendor/model-viewer-4.0.0.min.js" in pushed and "site.js" in pushed
    with pytest.raises(share.ShareError, match="只能分享 .glb"):
        share.publish(tmp_path / "x.fbx", site, push=False)


def test_the_card_is_drawn_in_the_first_pose_and_the_page_lists_the_poses(tmp_path):
    import numpy as np
    from polylode.interchange import glb as G
    from tests.test_interchange_glb import _read, _scene, _turn
    document, arrays = _scene(tmp_path)
    rest = arrays["bone_rest"]
    arrays["anim0"] = np.stack([np.stack([rest[0], rest[1] @ _turn(a)]) for a in (0.5, 1.0)]).astype(np.float32)
    document["animations"] = [{"name": "relax", "rate": 30.0, "frames": 2}]
    G.write(document, arrays, tmp_path, tmp_path / "posed.glb")
    held = share.posed(tmp_path / "posed.glb", tmp_path / "held.glb")
    gltf, accessor, _view = _read(held)
    (channel,) = gltf["animations"][0]["channels"]
    first = accessor(gltf["animations"][0]["samplers"][channel["sampler"]]["output"])[0]
    assert np.allclose(gltf["nodes"][channel["target"]["node"]][channel["target"]["path"]], first)
    page = share._page("x", "https://me.github.io/models/models/x/")
    assert "availableAnimations" in page and 'id="pose"' in page and "autoplay" in page


def test_a_share_is_live_once_the_page_serves_its_own_stamp_not_an_earlier_one():
    answers = iter([None, b'{"shared": "earlier"}', b'{"shared": "now"}'])
    asked, slept = [], []

    def fetch(url):
        asked.append(url)
        return next(answers)

    assert share.wait_live("https://me.github.io/m/models/x/", "now", log=lambda _l: None, fetch=fetch,
                           sleep=slept.append, every=10)
    assert len(asked) == 3 and all(u.startswith("https://me.github.io/m/models/x/model.json?t=") for u in asked)
    assert len(set(asked)) == 3 and slept == [10, 10]                    # each ask past the cache, a pause between
    assert not share.wait_live("https://me.github.io/m/models/x/", "now", log=lambda _l: None,
                               fetch=lambda _u: b'{"shared": "earlier"}', sleep=lambda _s: None, wait=0)


def test_the_front_page_is_the_named_discord_activity_and_a_plain_list_without_one():
    entries = [{"name": "a", "title": "A", "shared": "2026-11-03T00:00:00+00:00"}]
    plain = share._index(entries, "https://me.github.io/m/")
    activity = share._index(entries, "https://me.github.io/m/", "636443597507395605")
    assert 'data-activity=""' in plain and 'data-activity="636443597507395605"' in activity
    for page in (plain, activity):
        assert 'src="vendor/model-viewer-4.0.0.min.js"' in page and 'src="site.js"' in page
        assert 'href="models/a/" data-name="a"' in page and "cdn." not in page        # nothing from outside the site
    assert "frame_id" in share.SITE_SCRIPT and share.SDK in share.SITE_SCRIPT
    # a card whose model was deleted since the page was written comes off, once the model's file answers 404
    assert "answer.status === 404" in share.SITE_SCRIPT and "card.remove()" in share.SITE_SCRIPT


def test_a_model_folder_without_its_model_is_not_listed(tmp_path):
    """A model deleted from the site (its model.glb gone) was listed again by the next share."""
    for name, kept in (("a", True), ("b", False)):
        folder = tmp_path / share.MODELS / name
        folder.mkdir(parents=True)
        (folder / "model.json").write_text(json.dumps({"title": name.upper(), "shared": "2026-11-04"}), encoding="utf-8")
        if kept:
            (folder / "model.glb").write_bytes(b"glTF")
    assert [e["name"] for e in share.entries_in(tmp_path)] == ["a"]
    assert "cdn." not in share._page("x", "https://me.github.io/m/models/x/")


def test_a_clones_list_is_written_again_from_the_models_left_in_it(tmp_path, monkeypatch):
    """The --tidy job: a model folder deleted from the clone comes off the front page and list, committed alone."""
    monkeypatch.setattr(preview, "derived_root", lambda: tmp_path / "derived")
    monkeypatch.setattr(preview, "snapshot", lambda path, out, w, h: out.write_bytes(b"\x89PNG card") and out)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    site = tmp_path / "site"
    site.mkdir()
    _git(site, "init")
    _git(site, "config", "user.email", "me@example.com")
    _git(site, "config", "user.name", "Me")
    _git(site, "remote", "add", "origin", "https://github.com/Me/models.git")
    _git(site, "remote", "set-url", "--push", "origin", str(remote))
    for stem in ("c_A_slg_skin_0123456789", "c_B_slg_skin_0123456789"):
        share.publish(preview.triangle_glb(tmp_path / f"{stem}.glb"), site, log=lambda _l: None, push=False)
    _git(site, "rm", "-r", "-q", "models/c_a_slg_skin")
    _git(site, "commit", "-q", "-m", "Remove a model")
    url, removed = share.tidy(site, log=lambda _l: None)
    assert url == "https://me.github.io/models/" and removed == ["c_a_slg_skin"]
    assert [e["name"] for e in json.loads((site / share.LIST).read_text(encoding="utf-8"))] == ["c_b_slg_skin"]
    assert "c_a_slg_skin" not in (site / "index.html").read_text(encoding="utf-8")
    pushed = subprocess.run(["git", "-C", str(remote), "log", "-1", "--format=%s"], capture_output=True, text=True,
                            check=True).stdout.strip()
    assert pushed == share.TIDY_MESSAGE
    assert share.tidy(site, log=lambda _l: None)[1] == []                    # nothing more to take off
