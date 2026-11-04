"""One export as a job of its own: a second process of the same program, so a failing converter leaves the window alone.

The window writes the job (:func:`write_job`) and starts :func:`command`; the
job process runs :func:`run_job`, which prints one line per event:

* ``LOG <text>`` — a line for the log;
* ``OUTPUT <path>`` — a file written;
* ``PROGRESS <done> <total> <text>`` — how far the current step has got (``core.progress``; *total* 0
  when it cannot be counted, *text* empty to keep the step's last one);
* ``ERROR <text>`` — the reason it stopped (exit status 1).

A scan is the same kind of process (:func:`scan_command`, :func:`run_scan`):
it prints ``MODEL <json>`` per model the install has.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from ..core import progress
from ..core.paths import frozen
from . import settings as S
from .sources import ExtractorError, Model, registry


def write_job(path: Path, model: Model, settings: S.Settings, output: Path, formats: list[str]) -> Path:
    Path(path).write_text(json.dumps({"model": asdict(model), "settings": asdict(settings),
                                      "output": str(output), "formats": list(formats)}, ensure_ascii=False),
                          encoding="utf-8")
    return Path(path)


def program() -> Path:
    """The file this process runs from: the packaged program, or the Python interpreter of a checkout.

    Not ``sys.executable`` when packaged: Nuitka's standalone build sets it to
    ``<program folder>/python.exe`` (the build interpreter's file name), which
    is not there, and the window's scans and jobs failed to start (§7.1452).
    """
    if not frozen():
        return Path(sys.executable)
    if sys.platform == "win32":
        import ctypes
        buffer = ctypes.create_unicode_buffer(32768)
        ctypes.windll.kernel32.GetModuleFileNameW(None, buffer, len(buffer))
        return Path(buffer.value)
    return Path(sys.argv[0]).resolve()


def _start(*arguments: str) -> list[str]:
    if frozen():
        return [str(program()), *arguments]
    return [str(program()), "-u", "-m", "polylode.extractor", *arguments]


def command(job: Path) -> list[str]:
    """How to start a job: the packaged program itself, or this package's entry point from a checkout."""
    return _start("--run-job", str(job))


def scan_command(source: str, settings_file: Path) -> list[str]:
    return _start("--scan", source, str(settings_file))


def share_command(model: Path, site: Path, activity: str = "") -> list[str]:
    """The model shared on the GitHub Pages site cloned at *site* (``share.run_share``); with *activity*, a Discord
    application whose Activity the site's front page is."""
    return _start("--share", str(model), str(site), *([activity] if activity else []))


def share_github_command(model: Path, activity: str = "") -> list[str]:
    """The model shared through the user's own GitHub account (``github.run_share``)."""
    return _start("--share-github", str(model), *([activity] if activity else []))


def tidy_command(site: Path, activity: str = "") -> list[str]:
    """The front page and list of the site cloned at *site* written again from the models in it (``share.run_tidy``)."""
    return _start("--tidy", str(site), *([activity] if activity else []))


def tidy_github_command(activity: str = "") -> list[str]:
    """The front page and list of the account's site written again from the models in it (``github.run_tidy``)."""
    return _start("--tidy-github", *([activity] if activity else []))


#: what a ``--worker <kind>`` process runs: ``module:function`` taking the request's task and giving its JSON result
WORKERS = {"gf2-index": "polylode.extractor.gf2_source:index_rows",
           "gf2-roots": "polylode.extractor.gf2_source:root_checks",
           "echo": "polylode.extractor.jobs:echo"}


def echo(task: dict):
    """``--worker echo``: the task's value back — what ``--selftest`` starts, to see a worker run."""
    return task.get("value")
from ..core.parallel import WORKERS_MOST, slices, worker_count  # noqa: E402,F401 - the workers' split


def in_workers(kind: str, tasks: list[dict], log=print) -> list:
    """Each of *tasks* worked out by a worker process of this program (``--worker <kind>``, :data:`WORKERS`), side by
    side: their results in the tasks' order, None for one whose worker failed (its caller works it out itself)."""
    import shutil
    import subprocess
    import tempfile
    scratch = Path(tempfile.mkdtemp(prefix=f"polylode-{kind}-"))
    try:
        started = []
        for number, task in enumerate(tasks):
            request, result = scratch / f"{number}.json", scratch / f"{number}.result.json"
            request.write_text(json.dumps({**task, "result": str(result)}, ensure_ascii=False), encoding="utf-8")
            started.append((subprocess.Popen(_start("--worker", kind, str(request)), stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)), result))
        out = []
        for process, result in started:
            said, _ = process.communicate()
            if process.returncode == 0 and result.is_file():
                out.append(json.loads(result.read_text(encoding="utf-8")))
                continue
            tail = said.decode("utf-8", "replace").strip().splitlines()[-1:] if said else []
            log(f"  一個工作行程失敗，改在本行程處理：{tail[0] if tail else process.returncode}")
            out.append(None)
        return out
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_worker(kind: str, request: str) -> int:
    """``--worker <kind> <request.json>``: the task's result written where the request says (:func:`in_workers`)."""
    import importlib
    if kind not in WORKERS:
        print(f"no worker {kind}", flush=True)
        return 2
    module, _, name = WORKERS[kind].partition(":")
    task = json.loads(Path(request).read_text(encoding="utf-8"))
    result = getattr(importlib.import_module(module), name)(task)
    Path(task["result"]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


def view_host_command() -> list[str]:
    """The 3D view in a process of its own (``view_host.run``), its window shown in the program's."""
    return _start("--view-host")


def keys_command(settings_file: Path, public: bool = False) -> list[str]:
    return _start("--download-wuwa-keys" if public else "--discover-wuwa-keys", str(settings_file))


def run_keys(settings_file: Path, public: bool = False) -> int:
    from ..core.paths import _platform_user_root
    from ..wuthering_waves.keys import discover_keys, download_keys
    from . import bundled, wuwa_source as W
    from .sources import require
    try:
        settings = S.load(Path(settings_file))
        output = (download_keys if public else discover_keys)(require(settings, W.INSTALL),
                               require(settings, W.READER, bundled.wuwa_reader() or W.find_reader()),
                               require(settings, W.DOTNET, bundled.dotnet() or W.find_dotnet()),
                               _platform_user_root() / "wuwa_aes_key.json",
                               lambda line: _emit("LOG", line))
        _emit("OUTPUT", str(output))
        return 0
    except Exception as failure:
        _emit("ERROR", str(failure))
        return 1


def run_scan(source_id: str, settings_file: Path) -> int:
    """Print the models *source_id*'s install has, one ``MODEL`` line each; 1 with ``ERROR`` when it cannot."""
    settings = S.load(Path(settings_file))
    source = registry().get(source_id)
    if source is None:
        _emit("ERROR", f"沒有這個來源：{source_id}")
        return 1
    from .sources import shared_writes
    log = lambda line: _emit("LOG", line)                   # noqa: E731 - the job's one way to say a line
    try:
        # a scan writes what the game's exports and previews read and write (its indexes, its store): one at a time
        with progress.reporting(ProgressLines()), shared_writes(source_id, log):
            models = source.scan(settings, log)
    except ExtractorError as refusal:
        _emit("ERROR", str(refusal))
        return 1
    except Exception as failure:  # a converter's own failure: its traceback is the report
        _emit("ERROR", "".join(traceback.format_exception(failure)))
        return 1
    for model in models:
        print("MODEL " + json.dumps(asdict(model), ensure_ascii=False), flush=True)
    return 0


class ProgressLines:
    """The job's ``PROGRESS`` lines: a new step's text and a count's last report always, the rest at most
    ten a second (a loop over thousands of bundle files would otherwise flood the window)."""

    def __init__(self, clock=time.monotonic, write=print):
        self.clock, self.write = clock, write
        self.last, self.text = float("-inf"), None

    def __call__(self, done: int, total: int, text: str) -> None:
        now = self.clock()
        fresh = bool(text) and text != self.text
        if not fresh and not (total and done >= total) and now - self.last < 0.1:
            return
        self.last = now
        if text:
            self.text = text
        self.write(f"PROGRESS {done} {total} {' '.join(text.split())}", flush=True)


def read_progress(text: str) -> tuple[int, int, str] | None:
    """A ``PROGRESS`` line's ``(done, total, text)`` (the part after the word), or None when it is not one."""
    done, _, rest = text.partition(" ")
    total, _, label = rest.partition(" ")
    try:
        return int(done), int(total), label
    except ValueError:
        return None


def _emit(kind: str, text: str) -> None:
    for line in str(text).splitlines() or [""]:
        print(f"{kind} {line}", flush=True)


def run_job(path: Path) -> int:
    """Export the job's model; 0 when every requested file was written and read back by its source."""
    job = json.loads(Path(path).read_text(encoding="utf-8"))
    model = Model(**job["model"])
    settings = S.Settings(**job["settings"])
    source = registry().get(model.source)
    if source is None:
        _emit("ERROR", f"沒有這個來源：{model.source}")
        return 1
    output = Path(job["output"])
    try:
        output.mkdir(parents=True, exist_ok=True)
        with progress.reporting(ProgressLines()):
            written = source.export(model, settings, output, list(job["formats"]), lambda line: _emit("LOG", line))
    except ExtractorError as refusal:
        _emit("ERROR", str(refusal))
        return 1
    except Exception as failure:  # a converter's own failure: its traceback is the report
        _emit("ERROR", "".join(traceback.format_exception(failure)))
        return 1
    for file in written:
        _emit("OUTPUT", str(file))
    return 0
