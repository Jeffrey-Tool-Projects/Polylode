"""A model on the user's own GitHub Pages site: a page that turns it in 3D, and a card a chat shows for the link.

Discord, like most chats, shows no 3D model: a ``.glb`` posted there is a download.  A link is shown as a card from
the page's Open Graph tags (title, description, image), so a model is shared as a page of its own on a site the
user publishes: ``models/<name>/index.html`` with ``model-viewer`` (Google's web component, from a pinned CDN
version) turning ``model.glb``, and ``card.png`` drawn by this program's preview (:func:`preview.snapshot`) for the
card.  ``index.html`` at the site's root lists every model shared there.

The site is a folder the user names: a local clone of a GitHub repository whose Pages publish it.  Sharing writes
the model's files, commits only them and pushes; the page is at ``https://<owner>.github.io/<repository>/``.  The
program's own source tree is never a site: what the games hold is not published with the program.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

#: what the site serves itself (``web/``, carried by the program): Google's ``<model-viewer>`` and Discord's Embedded
#: App SDK, pinned and with their licences.  A Discord Activity reaches nothing but the site it is mapped to (its
#: proxy, ``<app id>.discordsays.com``), so neither can come from a CDN; and a version carried here does not change
#: under a page already shared.
WEB = Path(__file__).with_name("web")
VENDOR = "vendor"
VIEWER = "model-viewer-4.0.0.min.js"
SDK = "discord-embedded-app-sdk-2.5.0.js"
VENDORED = (VIEWER, SDK, "LICENSE-model-viewer.txt", "LICENSE-discord-embedded-app-sdk.txt")
#: the front page's script: in a Discord Activity (launched with ``frame_id``) the handshake and the models opened
#: on the page itself — a page loaded in its place would lose the ``frame_id`` the SDK reads
SCRIPT = "site.js"
#: the card a chat shows a link with: Open Graph's large image, 1.91:1
CARD = (1200, 630)
#: GitHub refuses a pushed file over 100 MB; Pages serves at most 1 GB a site
FILE_LIMIT = 100 * 1024 * 1024
MODELS = "models"


class ShareError(ValueError):
    """The site cannot take the model: not a GitHub clone, the model too large, Git missing or failing."""


def _git(site: Path, *arguments: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise ShareError("沒有找到 Git：分享到 GitHub Pages 需要安裝 Git")
    done = subprocess.run([git, "-C", str(site), *arguments], capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    if done.returncode != 0:
        raise ShareError(f"git {arguments[0]} 失敗：{(done.stderr or done.stdout).strip()}")
    return done.stdout.strip()


def site_url(remote: str) -> str:
    """The Pages address of the GitHub repository *remote* (an HTTPS or SSH URL)."""
    match = re.fullmatch(r"(?:https://(?:[^@/]+@)?github\.com/|git@github\.com:|ssh://git@github\.com/)"
                         r"([^/]+)/([^/]+?)(?:\.git)?/?", remote.strip())
    if not match:
        raise ShareError(f"網站資料夾的 origin 不是 GitHub 倉庫：{remote}")
    owner, repository = match[1], match[2]
    if repository.lower() == f"{owner.lower()}.github.io":
        return f"https://{owner.lower()}.github.io/"
    return f"https://{owner.lower()}.github.io/{repository}/"


def slug(path: Path) -> str:
    """A model's folder name on the site: its file name, the export's trailing hash dropped, in URL-safe letters."""
    stem = re.sub(r"_[0-9a-f]{10}$", "", Path(path).stem)
    return re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-").lower() or "model"


def _own_tree() -> Path:
    return Path(__file__).resolve().parents[2]


def check_site(site: Path) -> str:
    """*site*'s Pages address, after making sure it can take a model: a Git clone with a GitHub origin, and not
    this program's own source tree."""
    site = Path(site)
    if not site.is_dir():
        raise ShareError(f"網站資料夾不存在：{site}")
    own = _own_tree()
    resolved = site.resolve()
    if resolved == own or resolved.is_relative_to(own) or own.is_relative_to(resolved):
        raise ShareError("網站資料夾不能是本程式自己的資料夾：遊戲的模型不跟程式一起發布")
    return site_url(_git(site, "remote", "get-url", "origin"))


def _page(title: str, url: str) -> str:
    title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="Interactive 3D model: drag to turn, scroll or pinch to zoom.">
<meta property="og:type" content="website">
<meta property="og:title" content="{title}">
<meta property="og:description" content="Interactive 3D model: drag to turn, scroll or pinch to zoom.">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{url}card.png">
<meta property="og:image:width" content="{CARD[0]}">
<meta property="og:image:height" content="{CARD[1]}">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#2f6feb">
<script type="module" src="../../{VENDOR}/{VIEWER}"></script>
<style>
  html, body {{ margin: 0; height: 100%; background: #1e2129; color: #e6e8ee;
               font: 15px/1.5 system-ui, "Segoe UI", "Microsoft JhengHei", sans-serif; }}
  model-viewer {{ width: 100%; height: 100%; --poster-color: transparent; }}
  header {{ position: fixed; top: 0; left: 0; right: 0; padding: 12px 16px; display: flex; gap: 12px;
            align-items: baseline; background: linear-gradient(#1e2129ee, #1e212900); pointer-events: none; }}
  header h1 {{ font-size: 17px; margin: 0; font-weight: 600; overflow-wrap: anywhere; }}
  header a {{ color: #8ab4ff; pointer-events: auto; font-size: 13px; }}
  .hint {{ position: fixed; bottom: 10px; left: 0; right: 0; text-align: center; color: #9aa0ad; font-size: 13px; }}
  [hidden] {{ display: none !important; }}   /* a picker laid out with display: flex shows through hidden otherwise */
  .poses {{ position: fixed; top: 52px; left: 16px; display: flex; gap: 8px; align-items: center; font-size: 14px; }}
  .poses select {{ background: #262a33; color: inherit; border: 1px solid #3a3f4b; border-radius: 6px;
                  padding: 4px 8px; font: inherit; max-width: 70vw; }}
</style>
</head>
<body>
<model-viewer src="model.glb" poster="card.png" alt="{title}" camera-controls
              touch-action="pan-y" interaction-prompt="none"
              environment-image="neutral" exposure="1" shadow-intensity="0.6" autoplay></model-viewer>
<header><h1>{title}</h1><a href="../../">All models</a></header>
<div class="poses" hidden><label id="poses">Pose <select id="pose"></select></label>
  <label id="faces" hidden>Expression <select id="face"></select></label></div>
<div class="hint">Drag to turn · right-drag to pan · scroll or pinch to zoom</div>
<script type="module">
  // the model's poses and expressions are its animations (written by the exporter, an expression's named
  // "expression/…"), listed apart; the viewer plays one at a time: an expression shown, "—" or a pose goes back to the
  // pose; the first pose from the start, a model with none shows no picker
  const viewer = document.querySelector("model-viewer");
  const box = document.querySelector(".poses");
  const pick = document.getElementById("pose");
  const face = document.getElementById("face");
  const EXPRESSION = "expression/";
  const show = (name) => {{ viewer.animationName = name; viewer.currentTime = 0; viewer.play(); }};
  viewer.addEventListener("load", () => {{
    const names = viewer.availableAnimations;
    const poses = names.filter((name) => !name.startsWith(EXPRESSION));
    const looks = names.filter((name) => name.startsWith(EXPRESSION));
    if (!names.length) return;
    for (const name of poses) pick.add(new Option(name, name));
    face.add(new Option("—", ""));
    for (const name of looks) face.add(new Option(name.slice(EXPRESSION.length), name));
    document.getElementById("poses").hidden = !poses.length;
    document.getElementById("faces").hidden = !looks.length;
    if (poses.length) {{ pick.value = poses[0]; show(poses[0]); }}
    box.hidden = false;
  }});
  pick.addEventListener("change", () => {{ face.value = ""; show(pick.value); }});
  face.addEventListener("change", () => show(face.value || pick.value));
</script>
</body>
</html>
"""


#: the site's list of what is shared on it, ``[{"name", "title", "shared"}]``: what the front page is made from,
#: and all a share through the API needs to read back from the repository
LIST = "models.json"


def entries_in(site: Path) -> list[dict]:
    """The models shared on the site cloned at *site*, from each one's ``model.json``; a folder whose ``model.glb``
    is gone (deleted from the site) is not one."""
    found = []
    for meta in (site / MODELS).glob("*/model.json"):
        if not (meta.parent / "model.glb").is_file():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found.append({"name": meta.parent.name, "title": data.get("title", meta.parent.name),
                      "shared": data.get("shared", "")})
    return found


def _index(entries: list[dict], url: str, activity: str = "") -> str:
    """The site's front page: every model shared there, newest first.  With *activity* (a Discord application id)
    it is also that application's Activity (:data:`SCRIPT`)."""
    shared = sorted(((e.get("shared", ""), e.get("title", e["name"]), e["name"]) for e in entries), reverse=True)
    cards = "\n".join(
        f'  <a class="card" href="{MODELS}/{name}/" data-name="{name}"><img src="{MODELS}/{name}/card.png" alt="" '
        f'loading="lazy"><span>{html.escape(title)}</span></a>' for _when, title, name in shared)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D models</title>
<meta property="og:type" content="website">
<meta property="og:title" content="3D models">
<meta property="og:url" content="{url}">
<style>
  body {{ margin: 0; padding: 24px 16px; background: #1e2129; color: #e6e8ee;
         font: 15px/1.5 system-ui, "Segoe UI", "Microsoft JhengHei", sans-serif; }}
  h1 {{ font-size: 22px; margin: 0 0 16px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }}
  .card {{ display: block; background: #262a33; border-radius: 10px; overflow: hidden; color: inherit;
          text-decoration: none; }}
  .card img {{ display: block; width: 100%; aspect-ratio: {CARD[0]} / {CARD[1]}; object-fit: cover; }}
  .card span {{ display: block; padding: 8px 12px; overflow-wrap: anywhere; }}
  #view {{ position: fixed; inset: 0; background: #1e2129; }}
  #view model-viewer {{ width: 100%; height: 100%; }}
  #view .bar {{ position: fixed; top: 10px; left: 12px; display: flex; gap: 10px; align-items: center; z-index: 1; }}
  #view button, #view select {{ background: #262a33; color: inherit; border: 1px solid #3a3f4b; border-radius: 6px;
                               padding: 4px 10px; font: inherit; }}
</style>
<script type="module" src="{VENDOR}/{VIEWER}"></script>
<script type="module" src="{SCRIPT}"></script>
</head>
<body data-activity="{html.escape(activity)}">
<h1>3D models</h1>
<div class="grid">
{cards}
</div>
<section id="view" hidden>
  <div class="bar"><button id="back">← All models</button><label class="poses" hidden>Pose <select id="pose"></select></label>
    <label class="faces" hidden>Expression <select id="face"></select></label></div>
  <model-viewer camera-controls touch-action="none"
                interaction-prompt="none" environment-image="neutral" exposure="1" shadow-intensity="0.6"
                autoplay></model-viewer>
</section>
</body>
</html>
"""


#: the front page's script (:data:`SCRIPT`): it takes off the cards of models deleted since the page was written, and
#: inside a Discord Activity opens the models on the page itself (elsewhere the cards are links)
SITE_SCRIPT = """// A model deleted from the site keeps its card until the next share writes this page again: a card whose image
// does not load is taken off once the model's own file answers 404 (not for a network that failed for a moment).
for (const image of document.querySelectorAll("a.card img")) {
  const card = image.closest("a.card");
  const check = () => fetch(`models/${card.dataset.name}/model.json`, { method: "HEAD", cache: "no-store" })
    .then((answer) => { if (answer.status === 404) card.remove(); })
    .catch(() => {});
  if (image.complete && !image.naturalWidth) check();
  else image.addEventListener("error", check);
}

// Inside a Discord Activity (Discord loads this page with ?frame_id=…): Discord's handshake, then the models open
// here, turned and posed on this page; navigating to a model's own page would lose the frame_id the SDK reads.
// Opened anywhere else the cards stay plain links to each model's page.
const app = document.body.dataset.activity;
const params = new URLSearchParams(location.search);
if (app && params.has("frame_id")) {
  // the handshake is not waited for: the page works the same while (or if) it does not finish
  import("./vendor/discord-embedded-app-sdk-2.5.0.js")
    .then(({ DiscordSDK }) => new DiscordSDK(app).ready())
    .catch((failure) => console.warn("Discord SDK", failure));
  const view = document.getElementById("view");
  const viewer = view.querySelector("model-viewer");
  const box = view.querySelector(".poses");
  const pick = document.getElementById("pose");
  // poses and expressions (animations named "expression/…") listed apart; one plays at a time
  const faces = view.querySelector(".faces");
  const face = document.getElementById("face");
  const show = (name) => { viewer.animationName = name; viewer.currentTime = 0; viewer.play(); };
  viewer.addEventListener("load", () => {
    pick.replaceChildren();
    face.replaceChildren(new Option("—", ""));
    const names = viewer.availableAnimations;
    const poses = names.filter((name) => !name.startsWith("expression/"));
    const looks = names.filter((name) => name.startsWith("expression/"));
    box.hidden = !poses.length;
    faces.hidden = !looks.length;
    for (const name of poses) pick.add(new Option(name, name));
    for (const name of looks) face.add(new Option(name.slice("expression/".length), name));
    if (poses.length) show(poses[0]);
  });
  pick.addEventListener("change", () => { face.value = ""; show(pick.value); });
  face.addEventListener("change", () => show(face.value || pick.value));
  for (const card of document.querySelectorAll("a.card")) {
    card.addEventListener("click", (event) => {
      event.preventDefault();
      viewer.src = `models/${card.dataset.name}/model.glb`;
      view.hidden = false;
    });
  }
  document.getElementById("back").addEventListener("click", () => { view.hidden = true; viewer.removeAttribute("src"); });
}
"""


def posed(glb: Path, out: Path, animation: int = 0, still: bool = False) -> Path:
    """*glb* held in the first frame of its *animation*: each animated node's translation, rotation or scale set to
    that frame's value, the rest of the file as it is (only its JSON chunk changes).  A file with no animation is
    copied.  The card is drawn from it, so a character with poses is shown in one rather than at its bind pose:
    the preview's loader plays no animation."""
    import struct
    data = Path(glb).read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + length])
    animations = document.get("animations") or []
    if animation >= len(animations):
        shutil.copyfile(glb, out)
        return out
    binary = data[20 + length + 8:]
    widths = {"VEC3": 3, "VEC4": 4, "SCALAR": 1}
    for channel in animations[animation]["channels"]:
        accessor = document["accessors"][animations[animation]["samplers"][channel["sampler"]]["output"]]
        view = document["bufferViews"][accessor["bufferView"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        node = document["nodes"][channel["target"]["node"]]
        width = (len(document["meshes"][node["mesh"]]["primitives"][0].get("targets", []))
                 if channel["target"]["path"] == "weights" else widths[accessor["type"]])
        first = list(struct.unpack_from(f"<{width}f", binary, start))
        node.pop("matrix", None)
        node[channel["target"]["path"]] = first
    if still:
        # the pose is in the nodes now: a card is drawn from them, and Qt's loader makes an object of every key of
        # every animation (a character's 400 poses and expressions held its load for seconds)
        document.pop("animations", None)
    text = json.dumps(document, ensure_ascii=False).encode("utf-8")
    text += b" " * (-len(text) % 4)
    rest = data[20 + length:]
    Path(out).write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(text) + len(rest))
                          + struct.pack("<II", len(text), 0x4E4F534A) + text + rest)
    return Path(out)


def plain_card(out: Path, title: str) -> Path:
    """A card of *title* alone, the site's colours, :data:`CARD` big: for a model whose own picture cannot be drawn."""
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter
    image = QImage(*CARD, QImage.Format_RGB32)
    image.fill(QColor("#1e2129"))
    painter = QPainter(image)
    painter.setPen(QColor("#e6e8ee"))
    font = QFont()
    font.setPixelSize(64)
    painter.setFont(font)
    painter.drawText(QRect(60, 0, CARD[0] - 120, CARD[1]), Qt.AlignCenter | Qt.TextWordWrap, title)
    painter.end()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    image.save(str(out))
    return Path(out)


def write_model(model: Path, site: Path, url: str, title: str = "",
                others: list[dict] | None = None, activity: str = "", log=print) -> tuple[str, list[Path]]:
    """*model* (a ``.glb``) written into *site* as its page, with the site's front page and list; the page's
    address and the files written.  *others* are the models already on the site (read from *site* when not
    given: a clone has them all, a share through the API only what it writes); *activity* a Discord application
    whose Activity the front page also is.  The card is drawn by the preview, so this needs a
    ``QGuiApplication`` with OpenGL."""
    from . import preview
    model = Path(model)
    if model.suffix.lower() != ".glb":
        raise ShareError(f"只能分享 .glb（網頁的 3D 檢視器讀它）：{model.name}")
    shown = preview.shown_copy(model)             # hidden parts off: a web viewer draws them all
    if shown.stat().st_size > FILE_LIMIT:
        raise ShareError(f"{model.name} 超過 GitHub 的單檔上限 100 MB，推送不上去")
    name = slug(model)
    folder = site / MODELS / name
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(shown, folder / "model.glb")
    import tempfile
    title = title or re.sub(r"_[0-9a-f]{10}$", "", model.stem)
    with tempfile.TemporaryDirectory() as scratch:
        try:
            preview.snapshot(posed(shown, Path(scratch) / "posed.glb", still=True), folder / "card.png", *CARD,
                             seconds=CARD_WAIT)
        except ValueError as failure:
            # one model the preview cannot draw (pelica.glb's load never ended) stopped a share of twenty
            log(f"{model.name}：卡片圖畫不出來，改用只有名稱的卡片（{failure}）")
            plain_card(folder / "card.png", title)
    page_url = f"{url}{MODELS}/{name}/"
    (folder / "index.html").write_text(_page(title, page_url), encoding="utf-8")
    entry = {"name": name, "title": title, "shared": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (folder / "model.json").write_text(json.dumps({"title": title, "source": model.name, "shared": entry["shared"]},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    entries = [e for e in (entries_in(site) if others is None else others) if e.get("name") != name] + [entry]
    return page_url, [folder, *write_front(site, entries, url, activity)]


def write_front(site: Path, entries: list[dict], url: str, activity: str = "") -> list[Path]:
    """The site's front page, list (:data:`LIST`), script and the viewer it serves, written into *site* for
    *entries*; the files written."""
    site = Path(site)
    (site / ".nojekyll").touch()
    (site / LIST).write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    (site / "index.html").write_text(_index(entries, url, activity), encoding="utf-8")
    (site / SCRIPT).write_text(SITE_SCRIPT, encoding="utf-8")
    (site / VENDOR).mkdir(exist_ok=True)
    for file in VENDORED:
        shutil.copyfile(WEB / file, site / VENDOR / file)
    return [site / "index.html", site / LIST, site / ".nojekyll", site / SCRIPT, site / VENDOR]


def listed(site: Path) -> list[str]:
    """The names the site's list (:data:`LIST`) holds now, whatever is left of them."""
    try:
        entries = json.loads((Path(site) / LIST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e["name"] for e in entries if isinstance(e, dict) and e.get("name")] if isinstance(entries, list) else []


#: the commit a site's front page and list are written again by, with no model shared (:func:`tidy`)
TIDY_MESSAGE = "Update the model list"


def tidy(site: Path, log=print, push: bool = True, activity: str = "") -> tuple[str, list[str]]:
    """The front page and list of the site cloned at *site* written again from the models still in it, committed
    (only those files) and pushed: deleting a model's folder rewrites neither, and its card stayed on the front page.
    The site's address and the models taken off the list."""
    site = Path(site)
    url = check_site(site)
    before = listed(site)
    entries = entries_in(site)
    kept = {e["name"] for e in entries}
    log(f"寫入 {site}")
    paths = [str(p.relative_to(site)) for p in write_front(site, entries, url, activity)]
    _git(site, "add", "--", *paths)
    if _git(site, "status", "--porcelain", "--", *paths):
        _git(site, "commit", "-m", TIDY_MESSAGE, "--", *paths)
        if push:
            log("推送到 GitHub…")
            _git(site, "push", "origin", "HEAD")
    return url, [name for name in before if name not in kept]


def report_tidy(work) -> int:
    """A tidy job's lines: ``LOG``, ``REMOVED <name>`` per model taken off the list, ``URL <site>``, or ``ERROR``."""
    try:
        url, removed = work(lambda line: print(f"LOG {line}", flush=True))
    except Exception as failure:  # noqa: BLE001 - every failure is the window's to show, never a bare traceback
        print(f"ERROR {failure if isinstance(failure, ShareError) else f'{type(failure).__name__}: {failure}'}",
              flush=True)
        return 1
    for name in removed:
        print(f"REMOVED {name}", flush=True)
    print(f"URL {url}", flush=True)
    return 0


def run_tidy(site: str, activity: str = "") -> int:
    """``--tidy <site>``: :func:`tidy` as the window's job (:func:`report_tidy`)."""
    return report_tidy(lambda log: tidy(Path(site), log, activity=activity))


#: how long a share waits for its page to be served, and how often it looks: seconds.  Pages builds a push in about
#: half a minute; a repository's first site took longer, and a link pasted before then is a 404 with no card
LIVE_WAIT, LIVE_EVERY = 300, 10


def shared_stamp(folder: Path) -> str:
    """When the model in *folder* (``models/<name>/``) was shared: its ``model.json``'s ``shared``."""
    return json.loads((Path(folder) / "model.json").read_text(encoding="utf-8"))["shared"]


def wait_live(page_url: str, stamp: str, log=print, wait: float = LIVE_WAIT, every: float = LIVE_EVERY,
              fetch=None, sleep=None) -> bool:
    """Is *page_url* serving this share — its ``model.json`` saying it was shared at *stamp* — within *wait*
    seconds?  Asked every *every* seconds, past the CDN's cache (a query of the time); an earlier share of the same
    model answers 200 with the old file, so a status alone does not say."""
    import time
    import urllib.request

    def get(url: str) -> bytes | None:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Polylode"}), timeout=30) as r:
                return r.read()
        except OSError:
            return None

    fetch, sleep = fetch or get, sleep or time.sleep
    log("等 GitHub Pages 發佈這個網頁…")
    deadline = time.monotonic() + wait
    while True:
        body = fetch(f"{page_url}model.json?t={time.time_ns()}")
        try:
            if body is not None and json.loads(body).get("shared") == stamp:
                log("網頁已上線")
                return True
        except ValueError:
            pass
        if time.monotonic() >= deadline:
            log("GitHub Pages 還沒發佈完成")
            return False
        sleep(every)


#: how long a card's model may take to load, seconds (a character loads in 6-8)
CARD_WAIT = 60
#: what a share waiting for another to finish with the same site says
WAITING = "等待另一個分享寫完網站…"


def commit_message(models: list[Path]) -> str:
    """``Share a`` or ``Share a, b and 3 more``: what a share's one commit is called."""
    names = [slug(Path(m)) for m in models]
    shown = ", ".join(names[:3])
    return f"Share {shown}" + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def publish_many(models: list[Path], site: Path, log=print, push: bool = True,
                 activity: str = "") -> tuple[list[str], bool]:
    """*models* shared on the Pages site cloned at *site*: each written, all committed as one (only their files) and
    pushed; their pages' addresses, and whether the last was being served before the wait ran out (``wait_live``:
    Pages publishes the whole site at once).  Another share of the same clone waits for this one
    (``sources.shared_writes``): two at once wrote the list over each other."""
    from .sources import shared_writes
    site = Path(site)
    url = check_site(site)
    with shared_writes("pages-site", log, WAITING):
        log(f"寫入 {site}")
        pages, written = [], []
        for model in models:
            page_url, files = write_model(Path(model), site, url, activity=activity, log=log)
            pages.append(page_url)
            written += [f for f in files if f not in written]
        paths = [str(p.relative_to(site)) for p in written]
        _git(site, "add", "--", *paths)
        if _git(site, "status", "--porcelain", "--", *paths):
            _git(site, "commit", "-m", commit_message(models), "--", *paths)
        if not push:
            return pages, False
        log("推送到 GitHub…")
        _git(site, "push", "origin", "HEAD")
    last = site / MODELS / slug(Path(models[-1]))
    return pages, wait_live(pages[-1], shared_stamp(last), log)


def publish(model: Path, site: Path, log=print, push: bool = True, activity: str = "") -> tuple[str, bool]:
    """*model* shared on the Pages site cloned at *site* (:func:`publish_many` of one): its page's address and
    whether it was served before the wait ran out."""
    pages, live = publish_many([Path(model)], site, log, push, activity)
    return pages[0], live


def report_share(work) -> int:
    """A share job's lines: ``LOG``, ``ERROR``, ``PAGES <settings page>`` when Pages is off, ``LIVE yes|no`` and one
    ``URL <address>`` per model shared.  *work* ``(log) → (addresses, settings page, live)``."""
    try:
        pages, settings_page, live = work(lambda line: print(f"LOG {line}", flush=True))
    except Exception as failure:  # noqa: BLE001 - every failure is the window's to show, never a bare traceback
        print(f"ERROR {failure if isinstance(failure, ShareError) else f'{type(failure).__name__}: {failure}'}",
              flush=True)
        return 1
    if settings_page:
        print(f"PAGES {settings_page}", flush=True)
    print(f"LIVE {'yes' if live else 'no'}", flush=True)
    for page in pages:
        print(f"URL {page}", flush=True)
    return 0


def run_share(site: str, activity: str, *models: str) -> int:
    """``--share <site> <activity> <model>…``: the window's share job (:func:`report_share`)."""
    from PySide6.QtGui import QGuiApplication
    from .preview import use_opengl
    use_opengl()
    QGuiApplication.instance() or QGuiApplication([])

    def work(log):
        pages, live = publish_many([Path(m) for m in models], Path(site), log, activity=activity)
        return pages, "", live
    return report_share(work)
