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
    monkeypatch.setattr(FX, "scene_document", lambda target, output, _: folders.append((target.folder, target.stem)) or (None, None))
    from polylode.extractor import sources
    monkeypatch.setattr(sources, "write_own", lambda *a: [])
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
    for _ in range(20):
        QApplication.processEvents()
    assert asked["formats"] == ["glb"] and "blender" not in asked["settings"]["tools"]       # no Blender read-back
    root = page.preview_panel.view.rootObject()
    shown = Path(root.property("source").toLocalFile())
    assert shown.name == "x.glb" and shown.is_relative_to(tmp_path)
    assert PV.cached(page.models[0]) == shown
    monkeypatch.setattr(A.jobs, "command", lambda job: (_ for _ in ()).throw(AssertionError("converted again")))
    root.setProperty("source", QUrl())
    page.preview()                                                    # from the cache: no second job
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
    assert [page.files.item(i).text() for i in range(page.files.count())] == [str(Path("gf2/x/x.glb"))]
    page.show_file(glb)
    root = page.panel.view.rootObject()
    assert Path(root.property("source").toLocalFile()) == glb and "頂點 3" in page.info.toPlainText()
    page.wireframe.setChecked(True)
    page.light.setValue(250)
    page._toggle_backdrop()
    assert root.property("wireframe") is True and abs(root.property("light") - 2.5) < 1e-9
    assert root.property("backdrop").name() == "#e9ecf2"
    page.look(180, 0)
    assert root.findChild(object, "origin").property("eulerRotation").y() == 180
    window.close()


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
    row = next(i for i in range(page.files.count()) if page.files.item(i).text().endswith("x.glb"))
    page.files.setCurrentRow(row)
    page.delete_selected()
    assert sorted(thrown) == ["x", "x.blend", "x.fbm", "x.fbx", "x.glb", "x.json"]   # the emptied folder too
    assert keep.is_file() and not folder.exists()
    thrown.clear()
    outside = triangle_glb(tmp_path / "elsewhere.glb")                       # opened, not listed: not deleted
    page.files.clear()
    page.show_file(outside)
    page.delete_selected()
    assert not thrown and outside.is_file()
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
