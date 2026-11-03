# Polylode

**English** · [繁體中文](docs/i18n/README.zh-TW.md) · [简体中文](docs/i18n/README.zh-CN.md) · [日本語](docs/i18n/README.ja.md) · [한국어](docs/i18n/README.ko.md) · [Español](docs/i18n/README.es.md) · [Français](docs/i18n/README.fr.md) · [Deutsch](docs/i18n/README.de.md)

**Step-by-step user guide:** [https://je-chen.github.io/Polylode/](https://je-chen.github.io/Polylode/)

Polylode mines the character models out of the games you have installed — *poly*gons from the *lode* — and
writes them as textured, rigged files you can open anywhere: glTF (`.glb`), FBX and Blender (`.blend`). It is a
Windows desktop program with a window, a preview you can turn with the mouse, and a model viewer.

| Game | What it reads | Notes |
|---|---|---|
| Arknights: Endfield | Every playable character prefab | The asset maps are built from your install on the first scan (in checked, resumable batches) and again after a game update |
| Girls' Frontline 2: Exilium | Every playable character's outfits | Only roots with their own skinned mesh and a whole human skeleton are listed (no pets, props or empty variants). An outfit with swappable parts is shown as the game dresses it by default; the other versions and lower detail levels are in the FBX and .blend, hidden (the .glb holds only what is shown: Blender and most viewers ignore glTF's hidden flag) |
| Wuthering Waves | Every playable character's display models | Needs your AES keys file; the game-file reader (built from `tools/wuwa_cli`) comes with the program; each material section (hair, face, upper body…) is its own part |
| Existing conversions | Finished models in `*_Converted` folders | Copied out after checking each file's SHA-256 |

## Screenshots

The English window, taken from a real scan and preview with `tools/screenshots.py` (Girls' Frontline 2 shown).

| | |
|---|---|
| ![Scanning an install](docs/images/scan.png) | ![A model and its preview](docs/images/preview.png) |
| *Scanning: each step that can be counted has its bar* | *The model list and its preview, framed to the model* |
| ![The model viewer](docs/images/viewer.png) | ![Settings](docs/images/settings.png) |
| *The model viewer and what the file holds* | *Settings: install folders, tools, output and language* |

## Features

- **Scan, tick, export.** Each game has a page: scan the install, search, tick the models and export them in a
  queue. Every scan and export runs in a process of its own, so a failing converter never takes the window down.
- **Finds your installs.** Before a scan it offers the copies it finds in your Steam libraries; any folder you pick
  may be the game's, the one above it or one inside its data.
- **Three formats, no Blender needed for two.** `.glb` and FBX are written by Polylode itself (skeleton, skinned
  meshes, shape keys, embedded textures, transparency); `.blend` needs Blender. With Blender set, every file is
  read back in it as a check.
- **Materials as the games draw them.** Cut-outs, transparency, eye-shadow overlays and eye highlight layers are
  written from the rules read out of each game's shaders. Where glTF/FBX cannot blend a layer, its effect is baked
  into the texture beneath it; the `.blend` keeps real layers. Wuthering Waves' cut-out coverage comes from running the game's own compiled shader, and its parallax fur is built as layers of strands. Endfield's effect materials (wings, glows) are run through the game's own compiled shader and baked, and emission maps are written as emission; a material that cannot be drawn is made invisible and logged, never drawn grey.
- **Preview before exporting.** Select a row and press Preview (or double-click): the model is converted to `.glb`
  into a cache and shown beside the list — left-drag turns, right-drag pans, the wheel zooms. **Convert again** makes it anew, after this program or the game files are updated.
- **Model viewer.** Lists every model in your output folder and opens any `.glb`, `.gltf`, `.fbx` or `.obj`
  (dropped, chosen, or passed as `Polylode.exe <file>` so it can be your "Open with" program): front, back, side
  and top views, wireframe, light or dark background, light strength, and what a glTF holds. A listed model can be
  deleted to the recycle bin.
- **Share to Discord through GitHub Pages.** Discord shows no 3D model, so the viewer's **Share on GitHub Pages…**
  writes the `.glb` shown into your own GitHub Pages site as a page that turns it in any browser, pushes it and
  copies its link; pasted in Discord, the link shows as a card with the model's picture.
- **Five languages.** English, 繁體中文, 简体中文, 日本語 and 한국어; it follows your system until you choose one in
  Settings, and switching applies at once.

## Download

Take the latest `Polylode-<version>-windows-x64.zip` from the
[Releases](https://github.com/JE-Chen/Polylode/releases) page, unzip it anywhere and run `Polylode.exe`. It carries everything it needs:
AnimeStudio (for Endfield), the Wuthering Waves reader, a .NET 10 runtime, the glTF importer and a software OpenGL for computers without a GPU
driver.

Requirements: Windows 10 or 11, 64-bit. Blender 4.x or 5.x only for `.blend`.

## User guide

### 1. First start

Unzip the release anywhere and run `Polylode.exe`. The window follows your Windows language; choose another under
**Settings → Language** and it switches at once. Settings are saved as you type them and live, with every cache,
in `%LOCALAPPDATA%\Polylode`. If you used the extractor when it was part of IslandLife, its settings, Endfield
asset maps, GF2 index and stores are moved over on the first start, so nothing is built again.

### 2. Settings

- **Output folder** — where exports go, one folder per model: `<output>\<game>\<model id>\`.
- **Export formats** — `.glb`, FBX and `.blend`; tick any of them.
- **Blender executable** — optional. Needed only for `.blend`; when it is set, every `.glb` and FBX is also read
  back in Blender as a check.
- **Install folders** — may stay empty: a scan asks for each one and offers the copies it finds in your Steam
  libraries. The folder may be the game's, the one above it or one inside its data. A ✓ marks a path that is there.
- **Tools** — leave them empty to use what the program carries: AnimeStudio (Endfield), the Wuthering Waves reader
  and the .NET runtime. Wuthering Waves also needs your **AES keys file** (a JSON with `mainKey` and `dynamicKeys`
  for your client version): its files are encrypted, and the key is yours, not something the program can carry.
  Before the first preview or export it looks for a keys file you already have — beside the program, in its data folder, in the install folder and the one above it, in Downloads, Documents and Desktop — and asks only when it finds none.

### 3. Scanning

Open a game's page and press **Scan install**. The bar under the status line names the step and counts it where it
can. What a first scan costs:

- **Arknights: Endfield** builds its asset maps from the install: tens of minutes, in batches. Closing the program
  part way loses nothing; the next scan carries on.
- **Girls' Frontline 2** indexes the bundles (minutes) and checks which roots are character models.
- **Wuthering Waves** reads its character tables out of the archives.

Later scans reuse all of it until the game updates. Scan again after an update; a GF2 preview or export says so
when the install changed since its scan.

### 4. Previewing

Pick a row — the search box filters by character, outfit or id — and press **Preview** or double-click it. The
first preview of a model converts it (seconds to a minute, with its own bar); after that, selecting the row shows
it from the cache at once. In the view: **left-drag** turns, **right-drag** pans, the **wheel** zooms. Each model
is framed whole and centred when it loads.

A preview stays as it was first made. **Convert again** converts the selected row anew — after updating Polylode
or the game — and replaces the cached one once the new one is done; until then the earlier one is kept.

### 5. Exporting

Tick the models (**Select all** and **Select none** act on the rows the search shows) and press
**Export ticked models**. The queue at the bottom runs them one at a time: ⏳ waiting, ▶ running, ✓ written,
✗ failed — the log beside it says why. The bar under the queue counts models; the one below it shows the step of
the model being written. Closing the window while an export runs asks first.

### 6. Model viewer

The viewer lists the models in your output folder, newest first. **Open file…**, drop a file on the window, or
start `Polylode.exe <file>` (so it can be your "Open with" program) to look at any `.glb`, `.gltf`, `.fbx` or
`.obj`. **Front**, **Back**, **Left**, **Right** and **Top** turn the model; **Wireframe**, the background button
and **Light** change how it is drawn. The panel on the right lists what a glTF holds — meshes, vertices, bones,
shape keys and each material's blending — and marks a file written by older rules, to be exported again.
**Delete…** moves the model, its sibling files and its texture folder to the recycle bin.

### 7. Sharing on GitHub Pages (for Discord)

Discord cannot show a 3D model: a `.glb` posted there is only a download. Polylode shares a model as a page of its own
on a GitHub Pages site in your own GitHub account; the page turns the model in any browser, and its link shows in Discord
as a card with the model's picture. Nothing else needs installing: no Git, no clone.

1. Open a `.glb` in the model viewer and press **Share on GitHub Pages…**. It names the address the model will be public
   at and asks before going on.
2. The first time, it opens GitHub's sign-in page in your browser and shows a short code (already copied): enter it and
   authorize Polylode. It asks only for `public_repo` (making and writing public repositories) and keeps the
   authorization on this computer, encrypted for your Windows user. **Settings → GitHub account → Sign out** forgets it;
   GitHub's Settings → Applications revokes it.
3. It makes the public repository `polylode-models` in your account (once), and uploads `models/<name>/` (the model, a
   1200×630 card picture drawn as the preview shows it, and the page) with the site's list of models, as one commit.
4. The link is copied: `https://<account>.github.io/polylode-models/models/<name>/`. The first time, if Pages is not on
   for the repository yet, the message offers **Open Pages settings**: choose **Deploy from a branch**, `main`,
   `/ (root)` and Save — once. (Turning it on from the program would need access to all your private repositories,
   which it does not ask for.)
5. Pages takes a minute or two to update; then paste the link in Discord.

Already have a Pages site? Set **Settings → GitHub Pages site folder** to its local clone: sharing then commits there
and pushes with Git instead.

Only `.glb` can be shared, up to 100 MB (GitHub's file limit); sharing a model again replaces its page. Everything
shared is public: the models come from games, so make sure you may share them. The program's own folder is refused
as a site.

### Troubleshooting

- **A dialog or a file picker opens before a scan, preview or export** — a tool the conversion cannot run without is not set, and none was found where the program looks. For Wuthering Waves it is the AES keys file: the dialog says what it is; choose your keys file, or paste the key (`0x` and 64 hexadecimal digits) and it is saved as a keys file. Either way it is remembered.
- **"“…” is not set" or "“…” does not exist"** in the log — fix that path under **Settings**.
- **"This file's size could not be measured"** — the viewer could not read the file's extent (an FBX 7.5 file, for
  one); the view is at the origin, zoom with the wheel.
- **No `.blend` written** — set the Blender executable under **Settings**.
- **“git push failed” when sharing** — Git cannot push to the site's repository: sign in to GitHub in Git (for
  example with Git Credential Manager) and check that you can push from that folder yourself. **“The site folder's
  origin is not a GitHub repository”** — the folder must be a clone of a GitHub repository.
- **Something else failed** — the log at the bottom keeps every line the scan or export printed.

## Building from source

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m polylode.extractor          # run the window from the checkout
.venv\Scripts\python -m pytest tests                # tests
.venv\Scripts\python build.py --dotnet <dotnet folder>   # dist\Polylode\Polylode.exe, then its selftest
```

The build compiles with Nuitka (a C compiler is needed: Visual Studio's), copies AnimeStudio, a .NET 10 runtime
(`--dotnet`, default `DOTNET_ROOT`) and the software OpenGL beside the program, and runs
`Polylode.exe --selftest`, which loads every converter, the preview's QML and a real `.glb`; the build fails if any
of it is missing.

The Wuthering Waves reader the program carries is built by `tools/wuwa_cli/build.ps1` (see its `README.md`) and
given to the build with `--wuwa-reader`; the workflows do both. `tools/screenshots.py` takes the screenshots above
again, through the window's own scan and preview.

## CI/CD

- **CI** (`.github/workflows/ci.yml`) — on every push and pull request to `main` and `dev`: lint, tests and the
  checkout's selftest on Windows.
- **Release** (`.github/workflows/release.yml`) — on every push to `main`: tests, the Nuitka build with the packaged
  selftest, and a GitHub Release with the zipped program.

Work goes to `dev`; `main` is what is released.

## Layout

```
polylode/extractor/      the window, scans, jobs, preview, viewer, languages
polylode/endfield/       Arknights: Endfield — asset maps, conversion, FBX scene
polylode/gf2/            Girls' Frontline 2 — bundles, playable roster, conversion, eye-layer bake
polylode/wuthering_waves/  Wuthering Waves — role tables, display-model assembly
polylode/interchange/    the glTF and FBX writers (no Blender)
tools/                   Blender scripts, the icon, the FBX SDK check, the WuWa reader's source, AnimeStudio
tests/                   pytest
docs/                    screenshots, tool notes, architecture, open items, the README's translations
build.py, main.py        the packaged program
```

## License and content

Polylode's code is under the GNU LGPL 2.1 (`LICENSE`); AnimeStudio, which it carries, is MIT
(`tools/AnimeStudio/LICENSE`). Polylode contains no game content: it converts files from the installs on your own
computer, and what it writes stays there. The screenshots show a model converted from the author's own install.
Use it only with games you own, and respect their terms.
