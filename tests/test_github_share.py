"""Sharing through the user's own GitHub account (``extractor/github.py``), against a stand-in for GitHub."""
import base64
import json
import re
import sys

import pytest

from polylode.extractor import github, preview

TOKEN = "gho_stand-in-token"


class FakeGitHub:
    """GitHub's endpoints the share calls, in memory: one user, their repositories, blobs, trees and commits."""

    def __init__(self, pages_api=403):
        self.repos, self.objects, self.calls, self.pages_api = {}, {}, [], pages_api
        self.polls = iter([{"error": "authorization_pending"}, {"error": "slow_down", "interval": 10},
                           {"access_token": TOKEN}])

    def _store(self, value) -> str:
        sha = f"{len(self.objects):040x}"
        self.objects[sha] = value
        return sha

    def files(self, name="polylode-models") -> dict:
        """The repository's files on its branch now: path → bytes."""
        repo = self.repos[name]
        tree = self.objects[self.objects[repo["head"]]["tree"]]
        return {path: self.objects[sha] for path, sha in tree.items()}

    def __call__(self, method, url, token=None, body=None, accept="application/vnd.github+json"):
        self.calls.append((method, url))
        if url.startswith("https://github.com/login/device/code"):
            return 200, {"device_code": "dc", "user_code": "WDJB-MJHT", "verification_uri": "https://github.com/login/device",
                         "interval": 5, "expires_in": 900}
        if url.startswith("https://github.com/login/oauth/access_token"):
            return 200, next(self.polls)
        if method == "HEAD":                                    # the public site: not up yet
            return 404, b""
        assert token == TOKEN, url
        path = url.removeprefix(github.API)
        if path == "/user":
            return 200, {"login": "Someone"}
        if path == "/user/repos" and method == "POST":
            readme = self._store(b"# models")
            head = self._store({"tree": self._store({"README.md": readme}), "parents": []})
            self.repos[body["name"]] = {"head": head, "private": body["private"]}
            return 201, {"default_branch": "main"}
        match = re.fullmatch(r"/repos/Someone/([^/]+)(/.*)?", path)
        name, rest = match[1], match[2] or ""
        if name not in self.repos:
            return 404, {"message": "Not Found"}
        repo = self.repos[name]
        if rest == "":
            return 200, {"default_branch": "main", "private": repo["private"]}
        if rest == "/pages":
            return self.pages_api, {"message": "Resource not accessible by integration"}
        if rest == "/contents/models.json":
            listed = self.files(name).get("models.json")
            return (200, listed) if listed is not None else (404, b"")
        if rest == "/git/ref/heads/main":
            return 200, {"object": {"sha": repo["head"]}}
        if rest.startswith("/git/commits/") and method == "GET":
            return 200, {"tree": {"sha": self.objects[rest.rsplit("/", 1)[1]]["tree"]}}
        if rest == "/git/blobs":
            return 201, {"sha": self._store(base64.b64decode(body["content"]))}
        if rest == "/git/trees":
            tree = dict(self.objects[body["base_tree"]])
            tree.update({entry["path"]: entry["sha"] for entry in body["tree"]})
            return 201, {"sha": self._store(tree)}
        if rest == "/git/commits":
            return 201, {"sha": self._store({"tree": body["tree"], "parents": body["parents"], "message": body["message"]})}
        if rest == "/git/refs/heads/main" and method == "PATCH":
            assert self.objects[body["sha"]]["parents"] == [repo["head"]]   # a fast-forward of the branch
            repo["head"] = body["sha"]
            return 200, {}
        raise AssertionError(f"unexpected {method} {url}")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    stand_in = FakeGitHub()
    monkeypatch.setattr(github, "_request", stand_in)
    monkeypatch.setattr(github, "TOKEN_FILE", tmp_path / "token.bin")
    monkeypatch.setattr(github, "ACCOUNT_FILE", tmp_path / "account.json")
    monkeypatch.setattr(github.time, "sleep", lambda _s: None)
    monkeypatch.setattr(preview, "derived_root", lambda: tmp_path / "derived")
    monkeypatch.setattr(preview, "snapshot", lambda path, out, w, h: out.write_bytes(b"\x89PNG card") and out)
    return stand_in


def test_the_device_flow_waits_as_github_asks_and_gives_the_token(fake):
    told, waits = [], []
    token = github.device_login(lambda code, address: told.append((code, address)), wait=waits.append)
    assert token == TOKEN and told == [("WDJB-MJHT", "https://github.com/login/device")]
    assert waits == [5, 5, 10]                             # pending, then slow_down's new interval
    fake.polls = iter([{"error": "access_denied"}])
    with pytest.raises(github.GitHubError, match="取消"):
        github.device_login(lambda *_a: None, wait=lambda _s: None)


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is Windows'")
def test_the_token_is_kept_encrypted_for_this_user_and_forgotten(fake, tmp_path):
    github.save(TOKEN, "Someone")
    assert TOKEN.encode() not in (tmp_path / "token.bin").read_bytes()
    assert github.saved_token() == TOKEN and github.account() == "Someone"
    github.forget()
    assert github.saved_token() is None and github.account() == ""


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is Windows'")
def test_a_share_signs_in_makes_the_repository_and_commits_the_page(fake, tmp_path):
    told = []
    first = preview.triangle_glb(tmp_path / "c_A_slg_skin_0123456789.glb")
    url, settings, live = github.publish(first, lambda code, address: told.append(code), log=lambda _l: None)
    assert not live                                  # Pages is off: nothing to wait for
    assert told == ["WDJB-MJHT"] and github.account() == "Someone"
    assert url == "https://someone.github.io/polylode-models/models/c_a_slg_skin/"
    assert settings == "https://github.com/Someone/polylode-models/settings/pages"   # the one click left
    files = fake.files()
    assert files["models/c_a_slg_skin/model.glb"] == first.read_bytes() and "README.md" in files
    assert f'content="{url}card.png"' in files["models/c_a_slg_skin/index.html"].decode()
    assert not fake.repos["polylode-models"]["private"]
    # a second model, signed in already: the list keeps the first, nothing is made again
    second = preview.triangle_glb(tmp_path / "c_B_slg_skin_0123456789.glb")
    calls = len(fake.calls)
    github.publish(second, lambda *_a: pytest.fail("signed in again"), log=lambda _l: None)
    assert not any(url.endswith("/user/repos") for _m, url in fake.calls[calls:])
    listed = {e["name"] for e in json.loads(fake.files()["models.json"])}
    assert listed == {"c_a_slg_skin", "c_b_slg_skin"}
    front = fake.files()["index.html"].decode()
    assert 'href="models/c_a_slg_skin/"' in front and 'href="models/c_b_slg_skin/"' in front


def test_pages_turned_on_by_the_api_leaves_nothing_to_click(fake):
    fake.pages_api = 201
    fake.repos["polylode-models"] = {"head": None, "private": False}
    assert github.pages(TOKEN, "Someone", "main") == ""


def test_a_raw_file_comes_back_as_its_bytes_and_an_api_answer_as_json(monkeypatch):
    # the share's stand-in replaces _request whole, so what the real one makes of an answer is tested here: a raw
    # models.json parsed as JSON came back as a list, its reader parsed it again, every share after the first failed
    import io

    class Answer(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    body = b'[{"name": "a"}]'
    monkeypatch.setattr(github.urllib.request, "urlopen", lambda request, timeout: Answer(body))
    assert github._request("GET", "https://api.github.com/x", accept="application/vnd.github.raw+json") == (200, body)
    assert github._request("GET", "https://api.github.com/x") == (200, [{"name": "a"}])


def test_a_401_for_a_large_upload_keeps_the_sign_in_and_a_refused_sign_in_lets_it_go(fake, monkeypatch):
    # GitHub answers a too-large upload with 401 Bad credentials (measured): the token is let go only when /user
    # refuses it as well — letting it go at the upload signed the user in again at every share, to fail again
    github.save(TOKEN, "Someone")
    answers = {"/git/blobs": (401, {"message": "Bad credentials"})}
    real = fake.__call__

    def stand_in(method, url, token=None, body=None, accept="application/vnd.github+json"):
        path = url.removeprefix(github.API)
        return answers.get(path) or real(method, url, token, body, accept)
    monkeypatch.setattr(github, "_request", stand_in)
    with pytest.raises(github.GitHubError, match="401") as refused:
        github._api("POST", "/git/blobs", TOKEN, {})
    assert "授權已失效" not in str(refused.value) and github.TOKEN_FILE.is_file()
    answers["/user"] = (401, {"message": "Bad credentials"})
    with pytest.raises(github.GitHubError, match="授權已失效"):
        github._api("POST", "/git/blobs", TOKEN, {})
    assert not github.TOKEN_FILE.is_file()


def test_a_file_over_the_upload_limit_is_pushed_with_git(fake, tmp_path, monkeypatch):
    import shutil
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("no git")

    def git(*arguments, cwd=tmp_path):
        return subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
    origin = tmp_path / "origin.git"
    git("init", "-q", "--bare", "-b", "main", str(origin))
    git("-C", str(origin), "config", "uploadpack.allowFilter", "true")
    seed = tmp_path / "seed"
    git("init", "-q", "-b", "main", str(seed))
    (seed / "models.json").write_text("[]", encoding="utf-8")
    git("-C", str(seed), "add", ".")
    git("-C", str(seed), "-c", "user.name=a", "-c", "user.email=a@b", "commit", "-q", "-m", "first")
    git("-C", str(seed), "push", "-q", str(origin), "main")
    big = tmp_path / "model.glb"
    big.write_bytes(b"x" * (github.BLOB_LIMIT + 1))
    small = tmp_path / "index.html"
    small.write_text("<html>", encoding="utf-8")
    real_push = github.push_with_git

    def pushed(token, login, branch, files, message, log=print, name=github.REPOSITORY):
        return real_push(token, login, branch, files, message, log, name, remote=origin.as_uri())   # a local origin
    monkeypatch.setattr(github, "push_with_git", pushed)
    made = github.commit(TOKEN, "Someone", "main", {"models/m/model.glb": big, "index.html": small}, "Share m",
                         log=lambda _line: None)
    assert git("-C", str(origin), "rev-parse", "main") == made
    listed = git("-C", str(origin), "ls-tree", "-r", "--name-only", "main").splitlines()
    assert sorted(listed) == ["index.html", "models.json", "models/m/model.glb"]          # the earlier file kept
    assert git("-C", str(origin), "log", "-1", "--format=%an %s", "main") == "Someone Share m"
    assert not any("/git/blobs" in url for _method, url in fake.calls)                  # nothing through the API
    huge = tmp_path / "huge.glb"
    with open(huge, "wb") as out:
        out.truncate(github.FILE_LIMIT + 1)
    with pytest.raises(github.GitHubError, match="100 MB"):
        github.commit(TOKEN, "Someone", "main", {"huge.glb": huge}, "Share huge")

