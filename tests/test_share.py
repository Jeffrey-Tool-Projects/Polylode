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
    url = share.publish(model, site, log=lines.append)
    assert url == "https://me.github.io/models/models/c_x_slg_skin/"
    folder = site / "models" / "c_x_slg_skin"
    assert (folder / "model.glb").read_bytes() == model.read_bytes()
    page = (folder / "index.html").read_text(encoding="utf-8")
    assert f'content="{url}card.png"' in page and 'src="model.glb"' in page and share.VIEWER in page
    assert json.loads((folder / "model.json").read_text(encoding="utf-8"))["title"] == "c_X_slg_skin"
    assert 'href="models/c_x_slg_skin/"' in (site / "index.html").read_text(encoding="utf-8")
    pushed = subprocess.run(["git", "-C", str(remote), "show", "--name-only", "--format=%s", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.split()
    assert pushed[:2] == ["Share", "c_x_slg_skin"]
    assert "notes.txt" not in pushed and "models/c_x_slg_skin/model.glb" in pushed   # only the model's files
    with pytest.raises(share.ShareError, match="只能分享 .glb"):
        share.publish(tmp_path / "x.fbx", site, push=False)
