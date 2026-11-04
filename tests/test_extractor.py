"""The model extractor's own rules: settings, paths it requires, job commands, GF2 names (``polylode/extractor``)."""
import json
import sys

import pytest

from polylode.extractor import jobs, settings as S
from polylode.extractor.gf2_source import _outfit, copy_out
from polylode.extractor.sources import ExtractorError, Model, PathSpec, require


def test_settings_survive_a_save_and_an_unreadable_file_gives_empty_ones(tmp_path):
    path = tmp_path / "settings.json"
    S.save(S.Settings(installs={"gf2": "D:/GF2"}, tools={"blender": "B"}, output="O", formats=["fbx"]), path)
    loaded = S.load(path)
    assert loaded.install("gf2").as_posix() == "D:/GF2" and loaded.tool("blender").name == "B"
    assert loaded.formats == ["fbx"] and loaded.output == "O"
    path.write_text("{not json", encoding="utf-8")
    assert S.load(path) == S.Settings()


def test_a_required_path_says_which_one_is_unset_or_missing(tmp_path):
    spec = PathSpec("gf2", "GF2 安裝資料夾", "dir", install=True)
    with pytest.raises(ExtractorError, match="還沒有設定"):
        require(S.Settings(), spec)
    with pytest.raises(ExtractorError, match="不存在"):
        require(S.Settings(installs={"gf2": str(tmp_path / "nowhere")}), spec)
    assert require(S.Settings(installs={"gf2": str(tmp_path)}), spec) == tmp_path


def test_a_job_is_the_same_program_started_again(tmp_path):
    job = jobs.write_job(tmp_path / "job.json", Model("gf2", "c_X@1", "X"), S.Settings(), tmp_path, ["fbx"])
    assert json.loads(job.read_text(encoding="utf-8"))["model"]["ident"] == "c_X@1"
    command = jobs.command(job)
    assert command[0] == sys.executable and command[-2:] == ["--run-job", str(job)]
    assert jobs.scan_command("gf2", job)[-3:] == ["--scan", "gf2", str(job)]


def test_a_job_for_an_unknown_source_says_so(tmp_path, capsys):
    job = jobs.write_job(tmp_path / "job.json", Model("nowhere", "x", "x"), S.Settings(), tmp_path / "out", ["fbx"])
    assert jobs.run_job(job) == 1
    assert "ERROR" in capsys.readouterr().out


def test_gf2_outfits_are_what_follows_the_characters_name():
    assert _outfit("c_GrozaDorm_slg_skin", "Groza") == "Dorm slg_skin"
    assert _outfit("c_GrozaSR01_slg_skin", "Groza") == "SR01 slg_skin"
    assert _outfit("c_Groza_Firecracker_slg", "Groza") == "Firecracker_slg"


def test_copying_out_takes_the_fbx_with_its_textures_and_refuses_a_missing_blend(tmp_path):
    source = tmp_path / "src"
    (source / "textures").mkdir(parents=True)
    (source / "textures" / "a.png").write_bytes(b"png")
    fbx = source / "m.fbx"
    fbx.write_bytes(b"fbx")
    out = tmp_path / "out"
    out.mkdir()
    written = copy_out(fbx, None, out, ["fbx"])
    assert (out / "m.fbx").is_file() and (out / "textures" / "a.png").is_file() and len(written) == 2
    with pytest.raises(ExtractorError, match=r"\.blend"):
        copy_out(fbx, None, out, ["blend"])


def test_the_window_scans_and_exports_converted_models_in_processes_of_their_own(tmp_path, monkeypatch):
    """The whole path through the window: a scan process, ticking a row, an export job; files checked by SHA."""
    import hashlib
    import os
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEventLoop, QTimer, Qt
    from PySide6.QtWidgets import QApplication
    from polylode.extractor import app as A
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    root = tmp_path / "project"
    model = root / "WutheringWaves_Converted" / "R2T1XMd10011"
    model.mkdir(parents=True)
    (model / "R2T1XMd10011_covered.fbx").write_bytes(b"fbx")
    (model / "R2T1XMd10011_covered.blend").write_bytes(b"blend")
    excluded = root / "WutheringWaves_Converted" / "R2T1AnkeMd10011"
    excluded.mkdir()
    rows = [{"prefab": name, "display_name": shown,
             "fbx": f"D:\\elsewhere\\WutheringWaves_Converted\\{name}\\{name}_covered.fbx",
             "blend": f"D:\\elsewhere\\WutheringWaves_Converted\\{name}\\{name}_covered.blend",
             "fbx_sha256": hashlib.sha256(b"fbx").hexdigest(), "blend_sha256": hashlib.sha256(b"blend").hexdigest()}
            for name, shown in (("R2T1XMd10011", "某角色"), ("R2T1AnkeMd10011", "排除"))]
    (root / "WutheringWaves_Converted" / "covered_models.json").write_text(json.dumps({"models": rows}), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    S.save(S.Settings(installs={"converted": str(root)}, output=str(out), formats=["blend", "fbx"]))
    application = QApplication.instance() or QApplication([])
    window = A.Window()
    page = next(window.pages.widget(i) for i in range(window.pages.count())
                if getattr(window.pages.widget(i), "source", None) and window.pages.widget(i).source.id == "converted")

    def wait(done, seconds=120):
        loop = QEventLoop()
        timer = QTimer()
        timer.timeout.connect(lambda: done() and loop.quit())
        timer.start(50)
        QTimer.singleShot(seconds * 1000, loop.quit)
        loop.exec()
        timer.stop()
        return done()

    page.scan()
    assert wait(lambda: page.process is None)
    assert [m.ident for m in page.models] == ["R2T1XMd10011"]          # Encore is never listed
    page.table_model.item(0, 0).setCheckState(Qt.Checked)
    page.export()
    assert wait(lambda: window.process is None and not window.pending)
    folder = out / "converted" / "R2T1XMd10011"
    assert (folder / "R2T1XMd10011_covered.fbx").read_bytes() == b"fbx"
    assert (folder / "R2T1XMd10011_covered.blend").read_bytes() == b"blend"
    assert window.queue_list.item(0).text().startswith("✓")
    assert (window.progress.value(), window.progress.maximum()) == (1, 1)
    window.close()
    assert os.environ["QT_QPA_PLATFORM"] == "offscreen" and application is not None


def test_a_converted_file_that_differs_from_its_manifest_is_not_handed_out(tmp_path):
    from polylode.extractor.converted_source import ConvertedSource
    folder = tmp_path / "GF2_Converted" / "c_X"
    folder.mkdir(parents=True)
    (folder / "c_X_covered.fbx").write_bytes(b"changed")
    model = Model("converted", "c_X", "X", "GF2", {"folder": "GF2_Converted",
                                                    "files": {"fbx": "GF2_Converted/c_X/c_X_covered.fbx"},
                                                    "sha256": {"fbx": "0" * 64}})
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ExtractorError, match="SHA-256"):
        ConvertedSource().export(model, S.Settings(installs={"converted": str(tmp_path)}), out, ["fbx"], print)
    assert not (out / "c_X_covered.fbx").exists()


def test_endfield_names_and_a_folder_that_is_not_an_install(tmp_path):
    from polylode.extractor.endfield_source import EndfieldSource, character_name, data_root
    assert character_name("chr_0036_jsspsi_postmodel") == "jsspsi"
    assert character_name("chr_0002_endminm_postmodel") == "endminm"
    assert character_name("odd") == "odd"
    (tmp_path / "cli.exe").write_bytes(b"")
    settings = S.Settings(installs={"endfield": str(tmp_path)}, tools={"animestudio": str(tmp_path / "cli.exe")})
    with pytest.raises(ExtractorError, match="找不到唯一一份 Endfield_Data"):
        EndfieldSource().scan(settings, print)
    (tmp_path / "Endfield_Data" / "StreamingAssets" / "VFS").mkdir(parents=True)
    assert data_root(tmp_path) == tmp_path / "Endfield_Data"


def test_a_blend_without_its_images_or_meshes_fails_its_read_back():
    from types import SimpleNamespace as NS
    from polylode.endfield.fbx_export import blend_problems
    scene = NS(parts=[NS(name="body", positions=[0] * 4)])
    good = {"armatures": 1, "meshes": {"body": {"vertices": 4, "armature": True}}, "not_packed": []}
    assert blend_problems(scene, good) == []
    assert "not inside" in blend_problems(scene, dict(good, not_packed=["skin.png"]))[0]
    assert "not in it" in blend_problems(scene, dict(good, meshes={}))[0]
    assert "no armature" in blend_problems(scene, dict(good, meshes={"body": {"vertices": 4, "armature": False}}))[0]


def test_a_scan_prints_names_outside_the_console_code_page(tmp_path):
    """The window reads UTF-8; a name the console code page cannot hold (咲) must not stop the scan."""
    import os
    import subprocess
    folder = tmp_path / "GF2_Converted" / "c_X"
    folder.mkdir(parents=True)
    (folder / "c_X_covered.fbx").write_bytes(b"fbx")
    row = {"prefab": "c_X", "fbx": str(folder / "c_X_covered.fbx"), "characters": [{"name": "咲"}]}
    (tmp_path / "GF2_Converted" / "covered_models.json").write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    settings = tmp_path / "settings.json"
    S.save(S.Settings(installs={"converted": str(tmp_path)}), settings)
    environment = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    done = subprocess.run(jobs.scan_command("converted", settings), capture_output=True, env=environment, check=False)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert "咲" in done.stdout.decode("utf-8")


def test_output_lines_wait_for_their_end_even_inside_a_character():
    from polylode.extractor.app import Lines
    lines = Lines()
    data = "MODEL {\"character\": \"咲\"}\nLOG done\n".encode("utf-8")
    cut = data.index("咲".encode("utf-8")) + 1                  # inside the character's three bytes
    assert lines.feed(data[:cut]) == []
    assert lines.feed(data[cut:cut + 5]) == ['MODEL {"character": "咲"}']   # its rest and the newline arrived
    assert lines.feed(data[cut + 5:]) == ["LOG done"]
    assert lines.feed(b"ERROR last line, no newline") == [] and lines.rest() == ["ERROR last line, no newline"]


def test_a_scan_asks_for_the_install_first_and_keeps_the_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QFileDialog
    from polylode.extractor import app as A
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    S.save(S.Settings())
    QApplication.instance() or QApplication([])
    window = A.Window()
    page = next(window.pages.widget(i) for i in range(window.pages.count())
                if getattr(getattr(window.pages.widget(i), "source", None), "id", "") == "converted")
    asked = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: asked.append(a[1]) or "")
    assert page.ask_install() is False and asked and "還沒有選" in page.status.text()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: str(tmp_path))
    assert page.ask_install() is True
    assert S.load(tmp_path / "settings.json").installs["converted"] == str(tmp_path)
    asked.clear()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: asked.append(a) or "")
    assert page.ask_install() is True and not asked                  # set and there: not asked again
    window.close()


def test_an_empty_tool_field_means_the_carried_tool_but_a_wrong_path_is_still_an_error(tmp_path):
    spec = PathSpec("animestudio", "AnimeStudio", "file")
    carried = tmp_path / "carried.exe"
    carried.write_bytes(b"")
    assert require(S.Settings(), spec, carried) == carried
    with pytest.raises(ExtractorError, match="不存在"):
        require(S.Settings(tools={"animestudio": str(tmp_path / "typo.exe")}), spec, carried)


def test_an_install_is_found_from_above_inside_or_in_a_steam_library(tmp_path):
    """The folder named may be the game's, one above it (a launcher or Steam folder) or inside its data (§7.1454)."""
    from polylode.extractor import locate
    marker = "Client/Content/Paks"
    game = tmp_path / "Wuthering Waves" / "Wuthering Waves Game"
    (game / marker).mkdir(parents=True)
    assert locate.game_root(game, marker) == game
    assert locate.game_root(tmp_path / "Wuthering Waves", marker) == game
    assert locate.game_root(game / "Client" / "Content", marker) == game
    other = tmp_path / "Wuthering Waves" / "Beta"
    (other / marker).mkdir(parents=True)
    assert locate.game_root(tmp_path / "Wuthering Waves", marker) is None          # two games: not picked
    steam = tmp_path / "Steam"
    library = tmp_path / "Library D"
    (steam / "steamapps").mkdir(parents=True)
    (steam / "steamapps" / "libraryfolders.vdf").write_text(
        '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n\t"1"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n}\n'
        % (str(steam).replace("\\", "\\\\"), str(library).replace("\\", "\\\\")), encoding="utf-8")
    endfield = library / "steamapps" / "common" / "Arknights Endfield"
    (endfield / "Endfield_Data" / "StreamingAssets" / "VFS").mkdir(parents=True)
    (library / "steamapps" / "common" / "Other Game" / "Data").mkdir(parents=True)
    assert locate.libraries_from_vdf((steam / "steamapps" / "libraryfolders.vdf").read_text(encoding="utf-8")) == [steam, library]
    assert locate.steam_installs("Endfield_Data/StreamingAssets/VFS", steam) == [endfield]
    assert locate.steam_installs(marker, steam) == []
    from polylode.extractor.sources import PathSpec
    spec = PathSpec("wuwa", "鳴潮", "dir", install=True, marker=marker)
    assert require_(S.Settings(installs={"wuwa": str(game / "Client")}), spec) == game
    with pytest.raises(ExtractorError, match="唯一一份"):
        require_(S.Settings(installs={"wuwa": str(tmp_path / "Wuthering Waves")}), spec)


def require_(settings, spec):
    from polylode.extractor.sources import require
    return require(settings, spec)


def test_a_scan_offers_the_steam_install_and_refuses_a_folder_without_the_game(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog
    from polylode.extractor import app as A
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    S.save(S.Settings())
    game = tmp_path / "steamapps" / "common" / "Arknights Endfield"
    (game / "Endfield_Data" / "StreamingAssets" / "VFS").mkdir(parents=True)
    monkeypatch.setattr(A.locate, "steam_installs", lambda marker: [game])
    QApplication.instance() or QApplication([])
    window = A.Window()
    page = next(window.pages.widget(i) for i in range(window.pages.count())
                if getattr(getattr(window.pages.widget(i), "source", None), "id", "") == "endfield")
    offered = []
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a: offered.append(a[3]) or (str(game), True))
    assert page.ask_install() is True and offered[0][0] == str(game)
    assert S.load(tmp_path / "settings.json").installs["endfield"] == str(game)
    S.save(S.Settings())
    window.settings = S.load()
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a: (a[3][-1], True))      # 選擇其他資料夾…
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: str(tmp_path / "steamapps"))
    (tmp_path / "steamapps" / "Other").mkdir()
    assert page.ask_install() is True                                               # found below: the one game
    (tmp_path / "elsewhere").mkdir()
    window.settings = S.Settings()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: str(tmp_path / "elsewhere"))
    assert page.ask_install() is False and "找不到唯一一份" in page.status.text()
    window.close()


def test_outfits_of_one_endfield_character_are_written_apart(tmp_path, monkeypatch):
    """endminf, endminfink and endminfunmasked share the model's asset folder; each is written as itself (§7.1454)."""
    from polylode.endfield import fbx_export as FX
    from polylode.endfield import store
    from polylode.extractor import endfield_source as E
    folders = []
    monkeypatch.setattr(store, "read_extra", lambda kind, ident: {"model": "assets/art/actor/girl/endminf/models/x.prefab"})
    monkeypatch.setattr(FX, "scene_document", lambda target, output, *_: folders.append((target.folder, target.stem)) or (None, None))
    from polylode.extractor import sources
    monkeypatch.setattr(sources, "write_own", lambda *a, **kw: [])
    for ident in ("chr_0003_endminf_postmodel", "chr_0003_endminfink_postmodel"):
        E.write_files(ident, None, tmp_path, ["glb"], print)
    assert folders[0][1] != folders[1][1] and folders[0][0] != folders[1][0]
    assert folders[1] == (folders[1][0].rpartition("/")[0] + "/endminfink", "endminfink")


def test_the_extractor_converts_into_its_own_endfield_store_never_the_games(monkeypatch):
    """Every prefab is exported (§7.1455), so none of it may land where the game lists Endfield items."""
    from polylode.endfield import store
    from polylode.extractor.endfield_source import own_store
    monkeypatch.delenv(store.DIR_ENV, raising=False)
    game_store = store.root()
    with own_store() as folder:
        assert store.root() == folder != game_store and "extractor" in folder.parts
        assert store.charas_root() is None and store.stores() == [folder]
    assert store.root() == game_store and store.DIR_ENV not in __import__("os").environ


def test_gf2_every_playable_character_only_inside_the_extractors_own_scope():
    """The extractor exports men and unreviewed identities too (§7.1455); the game's processes still do not."""
    from polylode.gf2 import gender
    from polylode.extractor.gf2_source import own_scope
    from polylode.gf2 import store
    someone = {"id": 9999, "gun_id": 9999, "name": "Somebody", "model": "Somebody"}
    assert not gender.verified(someone) and gender.policy() == gender.POLICY
    game_store = store.root()
    with own_scope():
        assert gender.verified(someone) and gender.policy() == gender.POLICY_ALL
        assert store.root() != game_store and "extractor" in store.root().parts
        selection = {"eligible": True, "gender_scope": gender.policy(), "characters": [someone]}
        assert gender.selection_verified(selection)
    assert not gender.verified(someone) and store.root() == game_store
    assert not gender.selection_verified(selection)          # an extractor's selection is refused outside it


def test_a_preview_is_a_glb_only_job_shown_in_the_panel_and_kept_for_next_time(tmp_path, monkeypatch):
    """§7.1455: preview before export; the job writes .glb only, without Blender, into the preview cache."""
    import sys
    from pathlib import Path
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QApplication
    from polylode.extractor import app as A
    from polylode.extractor import preview as PV
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    S.save(S.Settings(tools={"blender": "C:/blender.exe", "dotnet": "D:/x"}, formats=["fbx", "blend"]))
    monkeypatch.setattr(PV, "derived_root", lambda: tmp_path)
    monkeypatch.setattr(A, "cache_folder", PV.cache_folder)
    asked = {}

    def command(job):
        asked.update(json.loads(Path(job).read_text(encoding="utf-8")))
        glb = Path(asked["output"]) / "girl" / "x" / "x.glb"
        return [sys.executable, "-c", f"import pathlib; p = pathlib.Path(r'{glb}'); p.parent.mkdir(parents=True);"
                                      f" p.write_bytes(b'glTF'); print('OUTPUT ' + str(p))"]
    monkeypatch.setattr(A.jobs, "command", command)
    QApplication.instance() or QApplication([])
    window = A.Window()
    page = next(window.pages.widget(i) for i in range(window.pages.count())
                if getattr(getattr(window.pages.widget(i), "source", None), "id", "") == "endfield")
    page.models = [Model("endfield", "chr_0001_x_postmodel", "x", "")]
    item = A.QStandardItem("x")
    item.setData(0, A.Qt.UserRole)
    page.table_model.appendRow([item, A.QStandardItem(""), A.QStandardItem("chr_0001_x_postmodel")])
    page.table.setCurrentIndex(page.proxy.index(0, 0))
    page.preview()
    assert page.preview_process.waitForFinished(30000)
    root = page.preview_panel.view.rootObject()
    _until(lambda: root.property("source"))
    assert asked["formats"] == ["glb"] and "blender" not in asked["settings"]["tools"]       # no Blender read-back
    shown = Path(root.property("source").toLocalFile())
    assert shown.name == "x.glb" and shown.is_relative_to(tmp_path)
    assert PV.cached(page.models[0]) == shown
    monkeypatch.setattr(A.jobs, "command", lambda job: (_ for _ in ()).throw(AssertionError("converted again")))
    root.setProperty("source", QUrl())
    page.preview()                                                    # from the cache: no second job
    _until(lambda: root.property("source").toLocalFile())
    assert Path(root.property("source").toLocalFile()) == shown
    window.close()


def test_a_gf2_scan_lists_only_roots_with_their_own_skinned_mesh(monkeypatch):
    """§7.1455: dorm scenes and props are roots too; convert_entry's gate is asked at the scan."""
    from types import SimpleNamespace as NS
    from polylode.gf2 import batch, playable
    roots = [{"cab": "A", "id": i, "name": name} for i, name in enumerate(("c_X", "c_XDorm_Bedroom_0101", "c_Y"))]
    roots.append({"cab": "B", "id": 9, "name": "c_Z"})
    roots.append({"cab": "A", "id": 3, "name": "c_X_Eagle_2_skin"})
    roots.append({"cab": "A", "id": 4, "name": "c_X_slg_skin_model"})
    opened = []

    class Objects(dict):
        def __getitem__(self, key):
            if key == 2:
                raise KeyError("unreadable")
            return NS(read=lambda key=key: NS(key=key))

    def environment():
        return NS(find_file=lambda cab: opened.append(cab) or NS(objects=Objects()))
    monkeypatch.setattr(playable, "families", lambda name, declared: [{"name": name}])
    monkeypatch.setattr(playable, "has_owned_skin", lambda go, matches: go.key in (0, 3))
    monkeypatch.setattr(playable, "has_owned_skin", lambda go, matches: go.key in (0, 3, 4))
    monkeypatch.setattr(playable, "has_human_rig", lambda go: go.key in (0, 4))     # the eagle has none
    monkeypatch.setattr(playable, "has_owned_mesh", lambda go, matches: go.key == 0)  # _model: no mesh
    results = {"B/9": {"status": "not_model"}}
    kept = batch.skinned_roots(NS(environment=environment), roots, {}, results, log=lambda _line: None)
    assert [e["name"] for e in kept] == ["c_X", "c_Y"]                  # c_Y unread: listed, not dropped
    assert results["A/1"]["status"] == "not_model" and "skinned" in results["A/1"]["reason"]
    assert results["A/3"]["status"] == "not_model" and "skeleton" in results["A/3"]["reason"]
    assert results["A/4"]["status"] == "not_model" and "no mesh" in results["A/4"]["reason"]
    assert sorted(opened) == ["A", "B"]                                 # one opening per bundle file


def test_the_viewer_lists_the_outputs_shows_a_file_and_says_what_it_holds(tmp_path, monkeypatch):
    """§7.1455: the program as a model viewer — any model file, fixed views, wireframe, backdrop, light."""
    from pathlib import Path
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from polylode.extractor import app as A
    from polylode.extractor.preview import triangle_glb
    from polylode.extractor.viewer import model_info
    output = tmp_path / "out"
    (output / "gf2" / "x").mkdir(parents=True)
    glb = triangle_glb(output / "gf2" / "x" / "x.glb")
    (output / "gf2" / "x" / "notes.txt").write_text("not a model", encoding="utf-8")
    info = model_info(glb)
    assert "頂點 3" in info and "網格 1" in info and "材質 0" in info
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    S.save(S.Settings(output=str(output)))
    QApplication.instance() or QApplication([])
    window = A.Window()
    page = window.viewer_page
    assert window.nav.item(window.pages.indexOf(page)).text() == "模型檢視器"
    page.refresh()
    _until(lambda: page.files.count() and page.files.item(0).data(A.Qt.UserRole))           # listed on a thread
    assert [page.files.item(i).text() for i in range(page.files.count())] == [str(Path("gf2/x/x.glb"))]
    page.show_file(glb)
    root = page.panel.view.rootObject()
    _until(lambda: root.property("source"))                                                 # read on a thread
    assert Path(root.property("source").toLocalFile()) == glb and "頂點 3" in page.info.toPlainText()
    page.wireframe.setChecked(True)
    page.light.setValue(250)
    page._toggle_backdrop()
    assert root.property("wireframe") is True and abs(root.property("light") - 2.5) < 1e-9
    assert root.property("backdrop") == "#e9ecf2"
    page.look(180, 0)
    assert root.sent[-1] == {"op": "look", "yaw": 180.0, "pitch": 0.0}                    # sent to the view's process
    assert {"op": "set", "name": "backdrop", "value": "#e9ecf2"} in root.sent
    window.close()


def _until(ready, seconds: float = 30.0):
    """Events handled until *ready*() holds: what a background step hands back arrives through them."""
    import time
    from PySide6.QtWidgets import QApplication
    end = time.monotonic() + seconds
    while not ready():
        assert time.monotonic() < end, "never ready"
        QApplication.processEvents()
        time.sleep(0.01)


def test_the_view_process_applies_what_the_window_sends(tmp_path, monkeypatch):
    """The 3D view runs in a process of its own: a model's load there holds nothing of the window's."""
    import json
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QProcess, QUrl
    from PySide6.QtGui import QColor, QGuiApplication, QVector3D
    from polylode.extractor import jobs, preview as PV, view_host as H
    QGuiApplication.instance() or QGuiApplication([])
    glb = PV.triangle_glb(tmp_path / "x.glb")

    class Root:
        def __init__(self):
            self.set, self.looked = {}, []

        def setProperty(self, name, value):
            self.set[name] = value

        def look(self, yaw, pitch):
            self.looked.append((yaw, pitch))
    root = Root()
    shown = []
    for value in (QUrl.fromLocalFile(str(glb)), QVector3D(1, 2, 3), True, 2.5):
        line = json.dumps({"op": "set", "name": "x", "value": H.encode(value)})
        shown.append(H.decode(json.loads(line)["value"]))
    assert shown == [QUrl.fromLocalFile(str(glb)), QVector3D(1, 2, 3), True, 2.5]
    assert H.encode(QColor("#e9ecf2")) == "#ffe9ecf2"
    H.apply(root, None, json.dumps({"op": "set", "name": "source", "value": H.encode(QUrl.fromLocalFile(str(glb)))}))
    H.apply(root, None, json.dumps({"op": "look", "yaw": 180, "pitch": 0}))
    for junk in ("not a command", "[1]", "{}"):
        H.apply(root, None, junk)
    assert root.set["source"] == QUrl.fromLocalFile(str(glb)) and root.looked == [(180.0, 0.0)]
    assert root.set["nativeMaterials"] == [[{}]]
    # the process itself: it gives its window and ends with its input
    host = QProcess()
    program, *arguments = jobs.view_host_command()
    host.start(program, arguments)
    assert host.waitForStarted(30000)
    said = b""
    while b"\n" not in said and host.waitForReadyRead(60000):
        said += bytes(host.readAllStandardOutput())
    assert said.startswith(b"WINID "), said + bytes(host.readAllStandardError())
    host.closeWriteChannel()
    assert host.waitForFinished(30000) and host.exitCode() == 0


class _Loaded:
    """An object of a loaded model as the view's process walks it: its class, name, children and properties."""

    def __init__(self, kind, name="", children=(), log=None, **values):
        self.kind, self.name, self.kids, self.log, self.values = kind, name, list(children), log, values

    def metaObject(self):
        from types import SimpleNamespace
        return SimpleNamespace(className=lambda: self.kind,
                               indexOfProperty=lambda name: 0 if name in self.values else -1)

    def objectName(self):
        return self.name

    def children(self):
        if self.kind == "QQuickKeyframe":
            raise AssertionError("a keyframe is never walked into")
        return self.kids

    def property(self, name):
        return self.values.get(name)

    def setProperty(self, name, value):
        self.values[name] = value
        self.log.append((self.name, name, value))

    def findChild(self, _kind, name):
        return next((c for c in self.kids if c.name == name), None)

    def stop(self):
        self.values["running"] = False

    def restart(self):
        self.values["running"] = True


def test_a_models_poses_are_found_listed_and_switched():
    """The preview switches poses: the loader makes each glTF animation a Timeline, the first one enabled."""
    import json
    from polylode.extractor import view_host as H
    log = []

    def pose(name, enabled):
        keys = _Loaded("QQuickKeyframeGroup", children=[_Loaded("QQuickKeyframe")])
        return _Loaded("QQuickTimeline", f"{name}-timeline", [keys, _Loaded("QQuickTimelineAnimation", name, log=log,
                                                                             running=True, loops=-1)],
                       log=log, enabled=enabled, startFrame=0.0, currentFrame=0.0)
    scene = _Loaded("QQuick3DNode", children=[_Loaded("QQuick3DModel"), pose("rise", True), pose("slide", False)])
    root = _Loaded("Root", children=[_Loaded("QQuick3DRuntimeLoader", "loader", [scene])], log=log)
    poses = H.Poses()
    assert poses.read(root) == ["rise", "slide"]
    timelines = [timeline for _name, timeline, _animation in poses.found]
    animations = [animation for _name, _timeline, animation in poses.found]
    assert [t.values["enabled"] for t in timelines] == [True, False]
    assert [a.values["running"] for a in animations] == [True, False]     # every animation started, one shown
    log.clear()
    timelines[1].values["currentFrame"] = 500.0
    assert poses.play(1)
    # the shown one off before the next is on: a disabled timeline puts back what it moved
    assert log.index(("rise-timeline", "enabled", False)) < log.index(("slide-timeline", "enabled", True))
    assert [t.values["enabled"] for t in timelines] == [False, True] and timelines[1].values["currentFrame"] == 0.0
    assert [a.values["running"] for a in animations] == [False, True]
    assert not poses.play(2) and not poses.play(-1)
    # the window's commands: a pose by its place in the list, a new file forgetting the last one's
    H.apply(root, None, json.dumps({"op": "pose", "index": 0}), poses)
    assert [t.values["enabled"] for t in timelines] == [True, False]
    H.apply(root, None, json.dumps({"op": "pose", "index": "x"}), poses)
    H.apply(root, None, json.dumps({"op": "set", "name": "source", "value": {"url": ""}}), poses)
    assert poses.found == [] and root.values["source"].isEmpty()
    assert H.Poses().read(_Loaded("Root", log=log)) == []                  # nothing loaded: no poses


def test_a_pose_that_places_the_character_elsewhere_is_framed_where_it_is():
    """GF2's CommandCenterBackGround seats Clukay SSR01 2.4 m to the side: the view framed the rest's box, empty."""
    from PySide6.QtGui import QVector3D
    from polylode.extractor import view_host as H
    log = []
    at = {"hips": (0, 0.9, 0), "head": (0, 1.5, 0), "shoe": (0, 0.05, 0)}

    def joint(name, children=()):
        return _Loaded("QQuick3DNode", name, children, log=log, scenePosition=QVector3D(*at[name]))
    head, shoe = joint("head"), joint("shoe")
    hips = joint("hips", [head, shoe])
    mesh = _Loaded("QQuick3DModel", "body", log=log, scenePosition=QVector3D(0, 0, 0))   # stays at the origin

    def timeline(name, enabled):
        group = _Loaded("QQuickKeyframeGroup", log=log, target=hips)
        return _Loaded("QQuickTimeline", name, [group, _Loaded("QQuickTimelineAnimation", name, log=log, running=True)],
                       log=log, enabled=enabled, startFrame=0.0, currentFrame=0.0)
    scene = _Loaded("QQuick3DNode", "scene", [hips, mesh, timeline("idle", True), timeline("seated", False)], log=log,
                    scenePosition=QVector3D(0, 0, 0))

    class Root(_Loaded):
        def __init__(self):
            super().__init__("Root", children=[_Loaded("QQuick3DRuntimeLoader", "loader", [scene], log=log)], log=log,
                             framed=True, frameLow=QVector3D(-0.3, 0, -0.2), frameHigh=QVector3D(0.3, 1.7, 0.2))
            self.refits = 0

        def refit(self):
            self.refits += 1
    root = Root()
    poses = H.Poses()
    assert poses.read(root) == ["idle", "seated"] and {n.name for n in poses.nodes} == {"hips", "head", "shoe"}
    assert root.refits == 1                                       # framed on the first pose once loaded
    # the second pose seats the character 2.4 m to the side, a shoe hidden 217 m away
    at.update(hips=(-2.4, 0.9, 1.0), head=(-2.4, 1.5, 1.0), shoe=(217.0, 0.05, 0))
    for node in (hips, head, shoe):
        node.values["scenePosition"] = QVector3D(*at[node.name])
    poses.play(1)
    low, high = root.values["frameLow"], root.values["frameHigh"]
    assert root.refits == 2
    assert (round(low.x(), 3), round(high.x(), 3)) == (-2.7, -2.1)      # the rest's margins round where she is now
    assert (round(low.z(), 3), round(high.z(), 3)) == (0.8, 1.2) and round(high.y(), 3) == 1.7
    assert H.joint_box([], 1.0) is None


def test_the_preview_panel_lists_the_poses_the_view_found(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from polylode.extractor.preview import PreviewPanel
    QApplication.instance() or QApplication([])
    panel = PreviewPanel()
    assert not panel.pose_row.isVisibleTo(panel)
    panel._host_said('POSES ["BarrackIdle", "姿勢 2"]\n')
    assert panel.pose_row.isVisibleTo(panel)
    assert [panel.pose_list.itemText(i) for i in range(panel.pose_list.count())] == ["BarrackIdle", "姿勢 2"]
    assert panel.pose_list.currentIndex() == 0 and not [c for c in panel.root.sent if c["op"] == "pose"]
    panel.pose_list.setCurrentIndex(1)
    assert panel.root.sent[-1] == {"op": "pose", "index": 1}
    panel.show_message("x")                                                 # another file: its own poses, later
    assert not panel.pose_row.isVisibleTo(panel) and panel.pose_list.count() == 0
    panel._host_said("POSES not json")
    assert not panel.pose_row.isVisibleTo(panel)


def test_the_window_fits_a_small_screen(tmp_path, monkeypatch):
    """§7.1455: the settings page set the window's minimum width at 1,393 px, wider than a laptop at 150 %."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from polylode.extractor import app as A
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    S.save(S.Settings())
    QApplication.instance() or QApplication([])
    window = A.Window()
    hint = window.minimumSizeHint()
    assert hint.width() <= 1000 and hint.height() <= 600, hint.toTuple()   # 1280 × 720 at 150 % fits
    window.close()


def test_the_viewer_says_a_file_of_older_writing_rules_should_be_exported_again():
    """§7.1455: the viewer drew files exported before the coverage fixes, and nothing said they were old."""
    from polylode.extractor.viewer import stale
    from polylode.interchange.glb import RULES
    ours = {"generator": "Polylode interchange"}
    assert stale({"asset": ours})                                                   # written before the rules
    assert stale({"asset": {**ours, "extras": {"rules": RULES - 1}}})
    assert not stale({"asset": {**ours, "extras": {"rules": RULES}}})
    assert not stale({"asset": {"generator": "Blender glTF exporter"}})            # someone else's file



def test_a_human_rig_is_every_required_node_under_the_root():
    """§7.1455: the scan asks the hierarchy for the skeleton conversion requires; a pet has none of it."""
    from types import SimpleNamespace as NS
    from polylode.gf2 import playable

    def node(name, *children):
        transform = NS(m_GameObject=NS(read=lambda: NS(m_Name=name)),
                       m_Children=[NS(read=lambda c=c: c) for c in children])
        return transform

    def root(transform):
        return NS(m_Component=[NS(component=NS(type=NS(name="Transform"), read=lambda: transform))])
    body = node("c_X", node("Root_M", *(node(n) for n in sorted(playable.HUMAN_RIG - {"Root_M"}))))
    assert playable.has_human_rig(root(body))
    bird = node("c_X_Eagle", node("Root_M", node("Head_M"), node("Wing_L")))
    assert not playable.has_human_rig(root(bird))


def test_the_viewer_deletes_a_listed_model_and_what_was_exported_with_it(tmp_path, monkeypatch):
    """§7.1455: delete from the viewer — the model's files of its name and its .fbm, to the recycle bin."""
    import shutil
    from pathlib import Path
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from polylode.extractor import app as A
    from polylode.extractor import viewer as V
    from polylode.extractor.preview import triangle_glb
    output = tmp_path / "out"
    folder = output / "gf2" / "x"
    folder.mkdir(parents=True)
    triangle_glb(folder / "x.glb")
    for name in ("x.fbx", "x.blend", "x.json"):
        (folder / name).write_bytes(b"x")
    (folder / "x.fbm").mkdir()
    keep = triangle_glb(output / "gf2" / "y.glb")
    thrown = []
    monkeypatch.setattr(V, "trash", lambda path: thrown.append(Path(path).name) or (
        shutil.rmtree(path) if Path(path).is_dir() else Path(path).unlink()) or True)
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    S.save(S.Settings(output=str(output)))
    QApplication.instance() or QApplication([])
    window = A.Window()
    page = window.viewer_page
    page.refresh()
    _until(lambda: page.files.count() and page.files.item(0).data(A.Qt.UserRole))           # listed on a thread
    row = next(i for i in range(page.files.count()) if page.files.item(i).text().endswith("x.glb"))
    page.files.setCurrentRow(row)
    page.delete_selected()
    assert sorted(thrown) == ["x", "x.blend", "x.fbm", "x.fbx", "x.glb", "x.json"]   # the emptied folder too
    assert keep.is_file() and not folder.exists()
    thrown.clear()
    outside = triangle_glb(tmp_path / "elsewhere.glb")                       # opened, not listed: not deleted
    page.files.clear()
    page.show_file(outside)
    _until(lambda: page.panel.current == outside)
    page.delete_selected()
    assert not thrown and outside.is_file()
    window.close()


def test_the_viewer_deletes_every_model_selected_at_once(tmp_path, monkeypatch):
    """Ctrl/Shift-click several listed models: one question names them all, each goes with its own files."""
    import shutil
    from pathlib import Path
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from polylode.extractor import app as A
    from polylode.extractor import viewer as V
    from polylode.extractor.preview import triangle_glb
    output = tmp_path / "out"
    made = {}
    for name in ("a", "b", "c"):
        folder = output / "gf2" / name
        folder.mkdir(parents=True)
        made[name] = triangle_glb(folder / f"{name}.glb")
        (folder / f"{name}.json").write_bytes(b"x")
    thrown, asked = [], []
    monkeypatch.setattr(V, "trash", lambda path: thrown.append(Path(path).name) or (
        shutil.rmtree(path) if Path(path).is_dir() else Path(path).unlink()) or True)
    monkeypatch.setattr(QMessageBox, "question", lambda _w, _t, text: asked.append(text) or QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "information", lambda *a: asked.append("told"))
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    S.save(S.Settings(output=str(output)))
    QApplication.instance() or QApplication([])
    window = A.Window()
    page = window.viewer_page
    page.refresh()
    _until(lambda: page.files.count() == 3 and page.files.item(0).data(A.Qt.UserRole))
    for row in range(page.files.count()):
        item = page.files.item(row)
        item.setSelected(not item.text().endswith("b.glb"))
    page.delete_selected()
    assert len(asked) == 1 and "2 個模型" in asked[0] and "a.glb" in asked[0] and "c.glb" in asked[0]
    assert "b.glb" not in asked[0]
    assert sorted(thrown) == ["a", "a.glb", "a.json", "c", "c.glb", "c.json"]   # each emptied folder too
    assert made["b"].is_file() and not made["a"].exists() and not made["c"].exists()
    # nothing of the output folder selected: told, nothing thrown
    thrown.clear()
    asked.clear()
    page.files.clearSelection()
    page.files.setCurrentItem(None)
    page.delete_selected()
    assert asked == ["told"] and not thrown
    window.close()


def test_the_first_start_moves_the_extractors_data_from_islandlife_once(tmp_path):
    """The tool was IslandLife's: its settings, maps and stores move over instead of being made again."""
    from polylode.extractor.migrate import from_islandlife
    old, new = tmp_path / "IslandLife", tmp_path / "Polylode"
    (old / "extractor").mkdir(parents=True)
    (old / "extractor" / "settings.json").write_text("{}", encoding="utf-8")
    (old / "assets_cache" / "endfield" / "meta").mkdir(parents=True)
    (old / "Charas").mkdir()                                              # the game's own: not touched
    assert sorted(from_islandlife(lambda _l: None, old, new)) == ["assets_cache/endfield/meta", "extractor"]
    assert (new / "extractor" / "settings.json").is_file() and (old / "Charas").is_dir()
    (old / "extractor").mkdir()
    assert from_islandlife(lambda _l: None, old, new) == []               # Polylode has its own now


def test_every_setting_is_read_back_as_saved(tmp_path):
    from dataclasses import asdict
    from polylode.extractor import settings as S
    saved = S.Settings(installs={"gf2": "D:/GF2"}, tools={"discord_activity": "1"}, output="D:/out", language="ja",
                       formats=["glb", "fbx"], poses=False)
    S.save(saved, tmp_path / "settings.json")
    assert asdict(S.load(tmp_path / "settings.json")) == asdict(saved)


def test_one_box_ticks_every_listed_model_or_none_and_shows_how_many_are(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from polylode.extractor import app as A
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    S.save(S.Settings())
    QApplication.instance() or QApplication([])
    window = A.Window()
    page = next(window.pages.widget(i) for i in range(window.pages.count())
                if getattr(getattr(window.pages.widget(i), "source", None), "id", "") == "gf2")
    box = page.tick_all
    assert not box.isEnabled()                                          # nothing listed yet
    for k, name in enumerate(("Clukay", "Cheyanne", "Lenna")):
        item = A.QStandardItem(name)
        item.setCheckable(True)
        item.setData(k, A.Qt.UserRole)
        page.table_model.appendRow([item, A.QStandardItem(""), A.QStandardItem(f"c_{name}")])
    assert box.isEnabled() and box.checkState() == A.Qt.Unchecked
    page.table_model.item(0, 0).setCheckState(A.Qt.Checked)
    assert box.checkState() == A.Qt.PartiallyChecked                    # some ticked
    box.click()
    ticked = [page.table_model.item(r, 0).checkState() for r in range(3)]
    assert ticked == [A.Qt.Checked] * 3 and box.checkState() == A.Qt.Checked
    box.click()                                                          # all ticked: a click clears them
    assert [page.table_model.item(r, 0).checkState() for r in range(3)] == [A.Qt.Unchecked] * 3
    assert box.checkState() == A.Qt.Unchecked
    # with a search, only the rows it leaves
    page.search.setText("Ch")
    assert page.proxy.rowCount() == 1 and box.checkState() == A.Qt.Unchecked
    box.click()
    assert [page.table_model.item(r, 0).checkState() for r in range(3)] == [A.Qt.Unchecked, A.Qt.Checked, A.Qt.Unchecked]
    assert box.checkState() == A.Qt.Checked
    page.search.setText("")
    assert box.checkState() == A.Qt.PartiallyChecked
    window.close()


def test_the_export_queue_runs_as_many_models_at_once_as_the_setting_allows(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    from polylode.extractor import app as A
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    out = tmp_path / "out"
    out.mkdir()
    S.save(S.Settings(output=str(out), formats=["glb"], parallel_exports=2))
    marks = tmp_path / "marks"
    marks.mkdir()

    def command(job):
        ident = json.loads(Path(job).read_text(encoding="utf-8"))["model"]["ident"]
        script = (f"import pathlib, time; m = pathlib.Path(r'{marks}'); (m / '{ident}.start').write_text(str(time.time()));"
                  f" time.sleep(1.5); (m / '{ident}.end').write_text(str(time.time())); print('LOG done {ident}')")
        return [sys.executable, "-c", script]
    monkeypatch.setattr(A.jobs, "command", command)
    QApplication.instance() or QApplication([])
    window = A.Window()
    window.queue([Model("gf2", f"c_{name}", name, "") for name in ("A", "B", "C")])
    assert len(window.running) == 2 and len(window.pending) == 1 and window.process is not None
    loop, timer = QEventLoop(), QTimer()
    timer.timeout.connect(lambda: (not window.running and not window.pending) and loop.quit())
    timer.start(50)
    QTimer.singleShot(60000, loop.quit)
    loop.exec()
    timer.stop()
    assert window.process is None and window.finished == 3
    spans = {name: (float((marks / f"c_{name}.start").read_text()), float((marks / f"c_{name}.end").read_text()))
             for name in "ABC"}
    assert spans["A"][0] < spans["B"][1] and spans["B"][0] < spans["A"][1]       # A and B ran side by side
    assert spans["C"][0] >= min(spans["A"][1], spans["B"][1]) - 0.05             # C waited for one of them
    texts = [window.queue_list.item(i).text() for i in range(3)]
    assert all(text.startswith("✓") for text in texts)
    window.close()


def test_shared_writes_are_one_process_at_a_time(tmp_path, monkeypatch):
    """Exports side by side (and a preview beside an export) write what a game's models share one at a time."""
    from pathlib import Path
    import subprocess
    import time
    from polylode.extractor import sources
    monkeypatch.setattr("polylode.core.paths.derived_root", lambda: tmp_path)
    told = []
    holder = subprocess.Popen([sys.executable, "-c",
                               "import sys, time; sys.path.insert(0, r'" + str(Path(__file__).resolve().parents[1]) + "');"
                               "import polylode.core.paths as P; P.derived_root = lambda: __import__('pathlib').Path(r'"
                               + str(tmp_path) + "');"
                               "from polylode.extractor.sources import shared_writes\n"
                               "with shared_writes('gf2'):\n    print('held', flush=True); time.sleep(1.5)"],
                              stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    start = time.monotonic()
    with sources.shared_writes("gf2", told.append):
        waited = time.monotonic() - start
    holder.wait(30)
    assert waited > 0.5 and told == ["等待另一個匯出或預覽寫完共用的轉換資料…"]
    with sources.shared_writes("gf2", told.append), sources.shared_writes("wuwa", told.append):
        pass                                                   # another game's lock is another lock
    assert len(told) == 1
