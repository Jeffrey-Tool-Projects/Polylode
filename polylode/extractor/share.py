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

#: Google's ``<model-viewer>``, pinned: a new release must not change a page already shared
VIEWER = "https://cdn.jsdelivr.net/npm/@google/model-viewer@4.0.0/dist/model-viewer.min.js"
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
<script type="module" src="{VIEWER}"></script>
<style>
  html, body {{ margin: 0; height: 100%; background: #1e2129; color: #e6e8ee;
               font: 15px/1.5 system-ui, "Segoe UI", "Microsoft JhengHei", sans-serif; }}
  model-viewer {{ width: 100%; height: 100%; --poster-color: transparent; }}
  header {{ position: fixed; top: 0; left: 0; right: 0; padding: 12px 16px; display: flex; gap: 12px;
            align-items: baseline; background: linear-gradient(#1e2129ee, #1e212900); pointer-events: none; }}
  header h1 {{ font-size: 17px; margin: 0; font-weight: 600; overflow-wrap: anywhere; }}
  header a {{ color: #8ab4ff; pointer-events: auto; font-size: 13px; }}
  .hint {{ position: fixed; bottom: 10px; left: 0; right: 0; text-align: center; color: #9aa0ad; font-size: 13px; }}
</style>
</head>
<body>
<model-viewer src="model.glb" poster="card.png" alt="{title}" camera-controls auto-rotate auto-rotate-delay="0"
              rotation-per-second="20deg" touch-action="pan-y" interaction-prompt="none"
              environment-image="neutral" exposure="1" shadow-intensity="0.6"></model-viewer>
<header><h1>{title}</h1><a href="../../">All models</a></header>
<div class="hint">Drag to turn · right-drag to pan · scroll or pinch to zoom</div>
</body>
</html>
"""


#: the site's list of what is shared on it, ``[{"name", "title", "shared"}]``: what the front page is made from,
#: and all a share through the API needs to read back from the repository
LIST = "models.json"


def entries_in(site: Path) -> list[dict]:
    """The models shared on the site cloned at *site*, from each one's ``model.json``."""
    found = []
    for meta in (site / MODELS).glob("*/model.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found.append({"name": meta.parent.name, "title": data.get("title", meta.parent.name),
                      "shared": data.get("shared", "")})
    return found


def _index(entries: list[dict], url: str) -> str:
    """The site's front page: every model shared there, newest first."""
    shared = sorted(((e.get("shared", ""), e.get("title", e["name"]), e["name"]) for e in entries), reverse=True)
    cards = "\n".join(
        f'  <a class="card" href="{MODELS}/{name}/"><img src="{MODELS}/{name}/card.png" alt="" loading="lazy">'
        f'<span>{html.escape(title)}</span></a>' for _when, title, name in shared)
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
</style>
</head>
<body>
<h1>3D models</h1>
<div class="grid">
{cards}
</div>
</body>
</html>
"""


def write_model(model: Path, site: Path, url: str, title: str = "",
                others: list[dict] | None = None) -> tuple[str, list[Path]]:
    """*model* (a ``.glb``) written into *site* as its page, with the site's front page and list; the page's
    address and the files written.  *others* are the models already on the site (read from *site* when not
    given: a clone has them all, a share through the API only what it writes).  The card is drawn by the
    preview, so this needs a ``QGuiApplication`` with OpenGL."""
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
    preview.snapshot(shown, folder / "card.png", *CARD)
    title = title or re.sub(r"_[0-9a-f]{10}$", "", model.stem)
    page_url = f"{url}{MODELS}/{name}/"
    (folder / "index.html").write_text(_page(title, page_url), encoding="utf-8")
    entry = {"name": name, "title": title, "shared": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (folder / "model.json").write_text(json.dumps({"title": title, "source": model.name, "shared": entry["shared"]},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    entries = [e for e in (entries_in(site) if others is None else others) if e.get("name") != name] + [entry]
    (site / ".nojekyll").touch()
    (site / LIST).write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    (site / "index.html").write_text(_index(entries, url), encoding="utf-8")
    return page_url, [folder, site / "index.html", site / LIST, site / ".nojekyll"]


def publish(model: Path, site: Path, log=print, push: bool = True) -> str:
    """*model* shared on the Pages site cloned at *site*: written, committed (only its files) and pushed.
    Its page's address."""
    site = Path(site)
    url = check_site(site)
    log(f"寫入 {site}")
    page_url, written = write_model(Path(model), site, url)
    paths = [str(p.relative_to(site)) for p in written]
    _git(site, "add", "--", *paths)
    if _git(site, "status", "--porcelain", "--", *paths):
        _git(site, "commit", "-m", f"Share {slug(Path(model))}", "--", *paths)
    if push:
        log("推送到 GitHub…")
        _git(site, "push", "origin", "HEAD")
    return page_url


def run_share(model: str, site: str) -> int:
    """``--share <model> <site>``: the window's share job, its lines as a job's (``LOG``, ``ERROR``) and the
    page's address as ``URL <address>``."""
    from PySide6.QtGui import QGuiApplication
    from .preview import use_opengl
    use_opengl()
    QGuiApplication.instance() or QGuiApplication([])
    try:
        url = publish(Path(model), Path(site), log=lambda line: print(f"LOG {line}", flush=True))
    except (ShareError, ValueError, OSError, RuntimeError) as failure:
        print(f"ERROR {failure}", flush=True)
        return 1
    print(f"URL {url}", flush=True)
    return 0
