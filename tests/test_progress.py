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
