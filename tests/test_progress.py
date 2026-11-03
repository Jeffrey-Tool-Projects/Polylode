"""Progress of a long step: the converters' reports, a job's PROGRESS lines and the window's bar."""
import json

from polylode.core import progress
from polylode.extractor import jobs
from polylode.extractor import settings as S
from polylode.extractor.sources import Model


def test_reports_reach_only_the_sinks_installed_and_nothing_without_one():
    heard = []
    progress.report(1, 2, "nobody listens")                 # no sink: nothing happens
    with progress.reporting(lambda *a: heard.append(a)):
        progress.report(1, 4, "a")
        progress.step("b")
    progress.report(3, 4, "after")
    assert heard == [(1, 4, "a"), (0, 0, "b")]


def test_progress_lines_keep_new_steps_and_ends_and_thin_out_the_rest():
    now, written = [0.0], []
    lines = jobs.ProgressLines(clock=lambda: now[0], write=lambda text, flush: written.append(text))
    lines(1, 100, "read")
    for done in range(2, 100):                              # within the same tenth of a second: dropped
        lines(done, 100, "read")
    lines(100, 100, "read")                                 # the last of a count always
    lines(0, 0, "next   step\nsplit")                       # a new step always, on one line
    now[0] = 0.5
    lines(3, 10, "")                                        # a count after the pause, the step's text kept
    assert written == ["PROGRESS 1 100 read", "PROGRESS 100 100 read", "PROGRESS 0 0 next step split",
                       "PROGRESS 3 10 "]
    assert [jobs.read_progress(line.partition(" ")[2]) for line in written] == [
        (1, 100, "read"), (100, 100, "read"), (0, 0, "next step split"), (3, 10, "")]
    assert jobs.read_progress("many of them") is None


def test_a_job_prints_its_sources_reports_as_progress_lines(tmp_path, monkeypatch, capsys):
    class Counting:
        id, label, paths = "counting", "Counting", ()

        def scan(self, settings, log):
            for done in range(1, 4):
                progress.report(done, 3, "counting")
            return []

        def export(self, model, settings, output, formats, log):
            progress.step("writing")
            (output / "x.glb").write_bytes(b"")
            return [output / "x.glb"]

    monkeypatch.setattr(jobs, "registry", lambda: {"counting": Counting()})
    settings_file = tmp_path / "settings.json"
    S.save(S.Settings(), settings_file)
    assert jobs.run_scan("counting", settings_file) == 0
    job = jobs.write_job(tmp_path / "job.json", Model("counting", "x", "X", "", {}), S.Settings(), tmp_path / "out", ["glb"])
    assert jobs.run_job(job) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "PROGRESS 1 3 counting" and "PROGRESS 3 3 counting" in printed
    assert "PROGRESS 0 0 writing" in printed and printed[-1].startswith("OUTPUT ")
    assert not progress._sinks                              # the job's sink is gone after it


def test_the_window_bar_counts_moves_and_hides(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from polylode.extractor.stepbar import StepBar
    qt = QApplication.instance() or QApplication([])
    bar = StepBar()
    bar.start("掃描中")
    assert not bar.isHidden() and (bar.bar.minimum(), bar.bar.maximum()) == (0, 0)
    assert bar.take("PROGRESS 5 20 建立 bundle 索引")
    assert (bar.bar.maximum(), bar.bar.value(), bar.label.text()) == (20, 5, "建立 bundle 索引")
    assert bar.take("PROGRESS 6 20 ")                       # no text: the step's name stays
    assert bar.label.text() == "建立 bundle 索引" and bar.bar.value() == 6
    assert bar.take("PROGRESS 0 0 組合場景") and bar.bar.maximum() == 0
    assert not bar.take("LOG 寫出 x.glb")                    # other lines are not the bar's
    bar.stop()
    assert bar.isHidden()
    del qt


def test_every_step_the_converters_name_has_every_language():
    """The step names the jobs report are shown through tr(): each has a catalog entry."""
    import ast
    import re
    from pathlib import Path
    catalog = json.loads((Path(jobs.__file__).with_name("i18n.json")).read_text(encoding="utf-8"))
    root = Path(jobs.__file__).parents[1]
    named = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "progress"):
                text = node.args[-1] if node.func.attr == "step" else (node.args[2] if len(node.args) > 2 else None)
                if isinstance(text, ast.Constant):
                    named.add(text.value)
                elif isinstance(text, ast.JoinedStr):
                    named.add(re.sub(r"\{[^}]*\}", "{0}", ast.unparse(text)[2:-1]))
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "start"
                    and isinstance(node.func.value, ast.Attribute) and node.func.value.attr.endswith(("bar", "steps"))
                    and node.args and isinstance(node.args[0], ast.Constant)):
                named.add(node.args[0].value)
    assert len(named) >= 20
    assert not sorted(named - set(catalog))


def test_a_gltf_box_is_its_meshes_placed_by_their_nodes_and_a_skinned_mesh_by_its_joints(tmp_path):
    import struct
    from polylode.extractor.preview import gltf_bounds
    positions = {"min": [-1, 0, -1], "max": [1, 2, 1]}
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 2]}],
                "nodes": [{"translation": [10, 0, 0], "children": [1]},
                          {"mesh": 0, "scale": [2, 2, 2]},            # under a moved node: 8..12 in x
                          {"mesh": 0, "skin": 0, "translation": [0, 50, 0]}],   # skinned: its node does not move it
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "accessors": [{"count": 2, "type": "VEC3", "componentType": 5126, **positions}]}
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    path = tmp_path / "box.glb"
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(text)) + struct.pack("<II", len(text), 0x4E4F534A) + text)
    low, high = gltf_bounds(path)
    assert low == (-1.0, 0.0, -2.0) and high == (12.0, 4.0, 2.0)
    (tmp_path / "x.fbx").write_bytes(b"Kaydara FBX Binary")
    assert gltf_bounds(tmp_path / "x.fbx") is None             # not glTF: the view measures it itself


def test_a_glb_is_shown_without_its_hidden_parts_and_the_file_itself_is_left_alone(tmp_path, monkeypatch):
    import struct
    from polylode.extractor import preview
    monkeypatch.setattr(preview, "derived_root", lambda: tmp_path / "derived")
    hidden = {"KHR_node_visibility": {"visible": False}}
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}],
                "nodes": [{"name": "lod0", "mesh": 0}, {"name": "lod1", "mesh": 0, "skin": 0, "extensions": hidden}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "accessors": [{"count": 2, "type": "VEC3", "componentType": 5126, "min": [0, 0, 0], "max": [1, 1, 1]}]}

    def glb(doc, path):
        text = json.dumps(doc).encode()
        text += b" " * (-len(text) % 4)
        blob = b"\0" * 8
        path.write_bytes(struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(blob)) + struct.pack("<II", len(text), 0x4E4F534A)
                         + text + struct.pack("<II", len(blob), 0x004E4942) + blob)
        return path

    original = glb(document, tmp_path / "model.glb")
    before = original.read_bytes()
    shown = preview.shown_copy(original)
    assert shown != original and original.read_bytes() == before
    data = shown.read_bytes()
    nodes = json.loads(data[20:20 + struct.unpack_from("<I", data, 12)[0]])["nodes"]
    assert "mesh" in nodes[0] and "mesh" not in nodes[1] and "skin" not in nodes[1]
    assert data.endswith(before[-16:]) and struct.unpack_from("<I", data, 8)[0] == len(data)
    assert preview.shown_copy(original) == shown                    # kept: not written again
    plain = glb({**document, "nodes": document["nodes"][:1]}, tmp_path / "plain.glb")
    assert preview.shown_copy(plain) == plain                       # nothing hidden: the file itself


def test_an_obj_box_is_its_vertices_and_a_file_not_measured_says_so(tmp_path):
    from polylode.extractor.preview import model_bounds
    (tmp_path / "m.obj").write_text("# x\nv 0 0 0\nv 1 2 -3\nvt 0 0\nf 1 2 1\n", encoding="utf-8")
    assert model_bounds(tmp_path / "m.obj") == ((0.0, 0.0, -3.0), (1.0, 2.0, 0.0))
    (tmp_path / "broken.fbx").write_bytes(b"not an fbx")
    assert model_bounds(tmp_path / "broken.fbx") is None
    assert model_bounds(tmp_path / "x.stl") is None


def test_a_preview_or_export_asks_for_each_required_tool_it_lacks_once(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QFileDialog
    from polylode.extractor import app as A
    from polylode.extractor import bundled
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    monkeypatch.setattr(bundled, "wuwa_reader", lambda: None)   # a checkout without the built reader
    from polylode.extractor import wuwa_source
    monkeypatch.setattr(wuwa_source, "find_keys", lambda settings: None)   # no keys file on this computer
    S.save(S.Settings())
    qt = QApplication.instance() or QApplication([])
    window = A.Window()
    page = next(window.pages.widget(i) for i in range(window.pages.count())
                if getattr(getattr(window.pages.widget(i), "source", None), "id", "") == "wuwa")
    reader, keys = tmp_path / "wuwa.exe", tmp_path / "keys.json"
    reader.write_bytes(b""), keys.write_text("{}")
    asked = []
    answers = iter([str(reader), str(keys)])
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (asked.append(a[1]) or next(answers), ""))
    assert page.ask_tools()
    assert len(asked) == 2 and S.load().tools["wuwa_reader"] == str(reader) and S.load().tools["wuwa_keys"] == str(keys)
    assert page.ask_tools() and len(asked) == 2                 # set and there: not asked again
    keys.unlink()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: ("", ""))
    assert not page.ask_tools()                                 # gone and not given: nothing converted
    window.close()
    del qt


def test_a_tool_the_program_carries_is_not_asked_for(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QFileDialog
    from polylode.extractor import app as A
    from polylode.extractor import bundled
    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    S.save(S.Settings())
    reader, keys = tmp_path / "wuwa.exe", tmp_path / "keys.json"
    reader.write_bytes(b""), keys.write_text("{}")
    monkeypatch.setattr(bundled, "wuwa_reader", lambda: reader)
    from polylode.extractor import wuwa_source
    monkeypatch.setattr(wuwa_source, "find_keys", lambda settings: None)
    qt = QApplication.instance() or QApplication([])
    window = A.Window()
    page = next(window.pages.widget(i) for i in range(window.pages.count())
                if getattr(getattr(window.pages.widget(i), "source", None), "id", "") == "wuwa")
    asked = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (asked.append(a[1]) or str(keys), ""))
    assert page.ask_tools() and len(asked) == 1                  # the keys only: the reader is carried
    assert S.load().tools.get("wuwa_reader", "") == ""            # its field stays empty: the carried one
    window.close()
    del qt


def test_a_keys_file_the_user_has_is_found_by_its_content_and_used_without_asking(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from pathlib import Path
    from PySide6.QtWidgets import QApplication, QFileDialog
    from polylode.extractor import app as A
    from polylode.extractor import bundled
    from polylode.extractor import wuwa_source as W
    home = tmp_path / "home"
    (home / "Downloads").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    keys = home / "Downloads" / "anything.json"
    keys.write_text(json.dumps({"mainKey": "0x" + "ab" * 32, "dynamicKeys": []}), encoding="utf-8")
    (home / "Downloads" / "other.json").write_text('{"mainKey": "nope", "dynamicKeys": []}', encoding="utf-8")
    (home / "Downloads" / "list.json").write_text("[1, 2]", encoding="utf-8")
    assert W.is_keys_file(keys) and not W.is_keys_file(home / "Downloads" / "other.json")
    assert W.find_keys(S.Settings()) == keys

    monkeypatch.setattr(S, "SETTINGS", tmp_path / "settings.json")
    monkeypatch.setattr(bundled, "wuwa_reader", lambda: tmp_path / "wuwa.exe")
    S.save(S.Settings())
    qt = QApplication.instance() or QApplication([])
    window = A.Window()
    page = next(window.pages.widget(i) for i in range(window.pages.count())
                if getattr(getattr(window.pages.widget(i), "source", None), "id", "") == "wuwa")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (_ for _ in ()).throw(AssertionError("asked")))
    assert page.ask_tools()                                    # found: no file picker
    assert S.load().tools["wuwa_keys"] == str(keys)
    window.close()
    del qt
