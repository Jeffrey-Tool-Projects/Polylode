"""Sharing through the user's own GitHub account, with no Git and no clone: sign in, a repository, Pages, uploads.

The user signs in with GitHub's device flow: the program shows a short code, the user enters it at
``github.com/login/device`` in their browser and approves; the program never sees a password.  The app is
registered as Polylode's OAuth app (:data:`CLIENT_ID`, public by design: the device flow takes no secret) and asks
for ``public_repo`` only.  The token is kept on this computer alone, encrypted for this Windows user (DPAPI), and
never printed.

With it the program makes a public repository in the user's account (:data:`REPOSITORY`) and commits the files
:func:`share.write_model` writes, through the Git Data API (blobs, a tree, a commit, the branch moved).  Turning
Pages on through the API wants the ``repo`` scope, which also reaches every private repository; the program does not
ask for it.  It looks whether the site already answers, tries the API once, and otherwise names the one click left:
the repository's Settings → Pages.
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..core.paths import _platform_user_root
from .share import LIST, ShareError, shared_stamp, slug, wait_live, write_model

#: Polylode's OAuth app (device flow enabled)
CLIENT_ID = "Ov23li8eFLr9Qb5BbBll"
SCOPE = "public_repo"
#: the repository a share makes in the user's account
REPOSITORY = "polylode-models"
API = "https://api.github.com"
TOKEN_FILE = _platform_user_root() / "extractor" / "github_token.bin"
ACCOUNT_FILE = _platform_user_root() / "extractor" / "github_account.json"


class GitHubError(ShareError):
    """GitHub refused or could not be reached."""


def _request(method: str, url: str, token: str | None = None, body: dict | None = None,
             accept: str = "application/vnd.github+json"):
    """``(status, payload)``: the JSON answer, or the bytes of a raw one."""
    headers = {"Accept": accept, "User-Agent": "Polylode", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=300) as answer:
            status, raw = answer.status, answer.read()
    except urllib.error.HTTPError as failure:
        status, raw = failure.code, failure.read()
    except urllib.error.URLError as failure:
        raise GitHubError(f"連不上 GitHub：{failure.reason}") from None
    # a raw file asked for (``…raw+json``) is its bytes, not JSON to parse: parsed here, a JSON file came back
    # as the list it holds and its reader parsed it again — every share after the first stopped there
    if "json" in accept and "raw" not in accept:
        try:
            return status, json.loads(raw) if raw else {}
        except ValueError:
            return status, {}
    return status, raw


def _api(method: str, path: str, token: str, body: dict | None = None, ok=(200, 201)) -> dict:
    status, payload = _request(method, API + path, token, body)
    if status not in ok:
        message = payload.get("message", "") if isinstance(payload, dict) else ""
        if status == 401:
            forget()
            raise GitHubError("GitHub 的授權已失效，下次分享時會重新登入")
        raise GitHubError(f"GitHub 回應 {status}：{method} {path} {message}".rstrip())
    return payload


# --- the token, on this computer only -------------------------------------------------------------------------

def _dpapi(data: bytes, protect: bool) -> bytes:
    """*data* encrypted (or decrypted) for this Windows user: ``CryptProtectData``/``CryptUnprotectData``."""
    if sys.platform != "win32":
        raise GitHubError("帳號登入只支援 Windows")
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt = ctypes.windll.crypt32
    source = Blob(len(data), ctypes.cast(ctypes.create_string_buffer(data, len(data)), ctypes.POINTER(ctypes.c_char)))
    out = Blob()
    call = crypt.CryptProtectData if protect else crypt.CryptUnprotectData
    if not call(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(out)):
        raise GitHubError("沒辦法讀寫保存的 GitHub 授權")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(out.pbData)


def saved_token() -> str | None:
    try:
        return _dpapi(TOKEN_FILE.read_bytes(), protect=False).decode("utf-8")
    except (OSError, GitHubError, UnicodeDecodeError):
        return None


def save(token: str, login: str) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_bytes(_dpapi(token.encode("utf-8"), protect=True))
    ACCOUNT_FILE.write_text(json.dumps({"login": login}), encoding="utf-8")


def account() -> str:
    """The account signed in on this computer, as last seen; empty when none."""
    try:
        return json.loads(ACCOUNT_FILE.read_text(encoding="utf-8")).get("login", "") if TOKEN_FILE.is_file() else ""
    except (OSError, ValueError):
        return ""


def forget() -> None:
    """Signed out here (the user can also revoke the app at GitHub's Settings → Applications)."""
    TOKEN_FILE.unlink(missing_ok=True)
    ACCOUNT_FILE.unlink(missing_ok=True)


def device_login(announce, wait=None) -> str:
    """A token from GitHub's device flow: *announce* ``(code, address)`` is told what the user enters where."""
    status, start = _request("POST", f"https://github.com/login/device/code?client_id={CLIENT_ID}&scope={SCOPE}",
                             accept="application/json")
    if status != 200 or "device_code" not in start:
        raise GitHubError(f"GitHub 沒有開始登入：{start.get('error_description') or status}")
    announce(start["user_code"], start["verification_uri"])
    wait = wait or time.sleep
    interval, deadline = int(start.get("interval", 5)), time.monotonic() + int(start.get("expires_in", 900))
    while time.monotonic() < deadline:
        wait(interval)
        _status, answer = _request(
            "POST", "https://github.com/login/oauth/access_token"
                    f"?client_id={CLIENT_ID}&device_code={start['device_code']}"
                    "&grant_type=urn:ietf:params:oauth:grant-type:device_code", accept="application/json")
        if answer.get("access_token"):
            return answer["access_token"]
        error = answer.get("error")
        if error == "slow_down":
            interval = int(answer.get("interval", interval + 5))
        elif error != "authorization_pending":
            raise GitHubError({"access_denied": "登入被取消了", "expired_token": "登入代碼過期了，請再按一次分享"}
                              .get(error, f"GitHub 登入失敗：{answer.get('error_description') or error}"))
    raise GitHubError("登入代碼過期了，請再按一次分享")


def signed_in(announce, log=print) -> tuple[str, str]:
    """``(token, login)``: the saved token when GitHub still takes it, else a new sign-in."""
    token = saved_token()
    if token:
        status, user = _request("GET", API + "/user", token)
        if status == 200:
            return token, user["login"]
        forget()
    log("登入 GitHub…")
    token = device_login(announce)
    login = _api("GET", "/user", token)["login"]
    save(token, login)
    log(f"已登入 GitHub：{login}")
    return token, login


# --- the repository and its site ------------------------------------------------------------------------------

def pages_url(login: str, repository: str = REPOSITORY) -> str:
    return f"https://{login.lower()}.github.io/{repository}/"


def repository(token: str, login: str, log=print, name: str = REPOSITORY) -> str:
    """The share repository's default branch, the repository made (public, with a first commit) when missing."""
    status, found = _request("GET", f"{API}/repos/{login}/{name}", token)
    if status == 200:
        if found.get("private"):
            raise GitHubError(f"{login}/{name} 是私人倉庫：免費方案的 Pages 只發布公開倉庫")
        return found["default_branch"]
    log(f"在 GitHub 建立公開倉庫 {login}/{name}")
    made = _api("POST", "/user/repos", token, {"name": name, "description": "3D models shared from Polylode",
                                               "private": False, "auto_init": True, "has_issues": False,
                                               "has_wiki": False, "has_projects": False})
    return made["default_branch"]


def site_answers(url: str) -> bool:
    status, _raw = _request("HEAD", url, accept="text/html")
    return status == 200


def pages(token: str, login: str, branch: str, log=print, name: str = REPOSITORY) -> str:
    """Pages for the repository: ``""`` when it is on (or was just turned on), else the settings page where one
    click turns it on (the API wants a scope this program does not ask for)."""
    if site_answers(pages_url(login, name)):
        return ""
    status, _payload = _request("POST", f"{API}/repos/{login}/{name}/pages", token,
                                {"source": {"branch": branch, "path": "/"}})
    if status in (201, 409):                       # made, or on already
        return ""
    log("GitHub Pages 還沒開啟：到倉庫的 Settings → Pages 選 Deploy from a branch")
    return f"https://github.com/{login}/{name}/settings/pages"


def read_list(token: str, login: str, name: str = REPOSITORY) -> list[dict]:
    """The site's list of shared models (``share.LIST``); empty when there is none yet."""
    status, raw = _request("GET", f"{API}/repos/{login}/{name}/contents/{LIST}", token,
                           accept="application/vnd.github.raw+json")
    if status != 200:
        return []
    try:
        entries = json.loads(raw)
    except ValueError:
        return []
    return [e for e in entries if isinstance(e, dict) and e.get("name")] if isinstance(entries, list) else []


def commit(token: str, login: str, branch: str, files: dict[str, Path], message: str, log=print,
           name: str = REPOSITORY) -> str:
    """*files* (repository path → local file) committed on *branch* as one commit; its id."""
    base = f"/repos/{login}/{name}"
    head = None
    for _try in range(10):                         # a repository just made may not show its branch yet
        status, ref = _request("GET", f"{API}{base}/git/ref/heads/{branch}", token)
        if status == 200:
            head = ref["object"]["sha"]
            break
        time.sleep(2)
    if head is None:
        raise GitHubError(f"{login}/{name} 沒有 {branch} 分支")
    tree = _api("GET", f"{base}/git/commits/{head}", token)["tree"]["sha"]
    entries = []
    for path, local in files.items():
        log(f"上傳 {path}")
        blob = _api("POST", f"{base}/git/blobs", token,
                    {"content": base64.b64encode(local.read_bytes()).decode("ascii"), "encoding": "base64"})
        entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    new_tree = _api("POST", f"{base}/git/trees", token, {"base_tree": tree, "tree": entries})["sha"]
    made = _api("POST", f"{base}/git/commits", token, {"message": message, "tree": new_tree, "parents": [head]})["sha"]
    _api("PATCH", f"{base}/git/refs/heads/{branch}", token, {"sha": made})
    return made


def publish(model: Path, announce, log=print, activity: str = "") -> tuple[str, str, bool]:
    """*model* shared through the user's account: ``(page address, settings page left to open or "", served yet)``;
    with Pages off there is nothing to wait for."""
    token, login = signed_in(announce, log)
    branch = repository(token, login, log)
    url = pages_url(login)
    with tempfile.TemporaryDirectory() as folder:
        staging = Path(folder)
        page_url, written = write_model(Path(model), staging, url, others=read_list(token, login), activity=activity)
        files = {}
        for item in written:
            for local in ([p for p in item.rglob("*") if p.is_file()] if item.is_dir() else [item]):
                files[local.relative_to(staging).as_posix()] = local
        log("推送到 GitHub…")
        commit(token, login, branch, files, f"Share {slug(Path(model))}", log)
        stamp = shared_stamp(written[0])
    settings_page = pages(token, login, branch, log)
    return page_url, settings_page, False if settings_page else wait_live(page_url, stamp, log)


def run_share(model: str, activity: str = "") -> int:
    """``--share-github <model>``: the window's share job through the account; lines as :func:`share.run_share`
    plus ``DEVICE <code> <address>`` while the user signs in and ``PAGES <settings page>`` when Pages is off."""
    from PySide6.QtGui import QGuiApplication
    from .preview import use_opengl
    use_opengl()
    QGuiApplication.instance() or QGuiApplication([])

    def log(line: str) -> None:
        print(f"LOG {line}", flush=True)

    try:
        url, settings_page, live = publish(
            Path(model), lambda code, address: print(f"DEVICE {code} {address}", flush=True), log, activity)
    except Exception as failure:  # noqa: BLE001 - every failure is the window's to show, never a bare traceback
        print(f"ERROR {failure if isinstance(failure, ShareError) else f'{type(failure).__name__}: {failure}'}",
              flush=True)
        return 1
    if settings_page:
        print(f"PAGES {settings_page}", flush=True)
    print(f"LIVE {'yes' if live else 'no'}", flush=True)
    print(f"URL {url}", flush=True)
    return 0

