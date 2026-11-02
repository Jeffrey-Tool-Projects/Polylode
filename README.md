# Polylode

**English** · [繁體中文](docs/i18n/README.zh-TW.md) · [简体中文](docs/i18n/README.zh-CN.md) · [日本語](docs/i18n/README.ja.md) · [한국어](docs/i18n/README.ko.md) · [Español](docs/i18n/README.es.md) · [Français](docs/i18n/README.fr.md) · [Deutsch](docs/i18n/README.de.md)

Polylode mines the character models out of the games you have installed — *poly*gons from the *lode* — and
writes them as textured, rigged files you can open anywhere: glTF (`.glb`), FBX and Blender (`.blend`). It is a
Windows desktop program with a window, a preview you can turn with the mouse, and a model viewer.

| Game | What it reads | Notes |
|---|---|---|
| Arknights: Endfield | Every playable character prefab | The asset maps are built from your install on the first scan (in checked, resumable batches) and again after a game update |
| Girls' Frontline 2: Exilium | Every playable character's outfits | Only roots with their own skinned mesh and a whole human skeleton are listed (no pets, props or empty variants) |
| Wuthering Waves | Every playable character's display models | Needs your AES keys file; the game-file reader (built from `tools/wuwa_cli`) comes with the program |
| Existing conversions | Finished models in `*_Converted` folders | Copied out after checking each file's SHA-256 |

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
  into the texture beneath it; the `.blend` keeps real layers.
- **Preview before exporting.** Select a row and press Preview (or double-click): the model is converted to `.glb`
  into a cache and shown beside the list — left-drag turns, right-drag pans, the wheel zooms.
- **Model viewer.** Lists every model in your output folder and opens any `.glb`, `.gltf`, `.fbx` or `.obj`
  (dropped, chosen, or passed as `Polylode.exe <file>` so it can be your "Open with" program): front, back, side
  and top views, wireframe, light or dark background, light strength, and what a glTF holds. A listed model can be
  deleted to the recycle bin.
- **Five languages.** English, 繁體中文, 简体中文, 日本語 and 한국어; it follows your system until you choose one in
  Settings, and switching applies at once.

## Download

Take the latest `Polylode-<version>-windows-x64.zip` from the
[Releases](https://github.com/JE-Chen/Polylode/releases) page, unzip it anywhere and run `Polylode.exe`. It carries everything it needs:
AnimeStudio (for Endfield), the Wuthering Waves reader, a .NET 10 runtime, the glTF importer and a software OpenGL for computers without a GPU
driver.

Requirements: Windows 10 or 11, 64-bit. Blender 4.x or 5.x only for `.blend`.

## Using it

1. **Settings** — name the output folder and the export formats. Install folders can be left empty: a scan asks
   for them (and offers the ones it finds in Steam).
2. **A game's page** — *Scan install*. The first Endfield scan builds its asset maps (tens of minutes, resumable);
   the first GF2 scan indexes the bundles. Later scans reuse both until the game updates.
3. Tick the models, *Export ticked models*, and watch the queue and log at the bottom.
4. **Model viewer** — look at what you exported, compare, delete.

Your settings and caches live in `%LOCALAPPDATA%\Polylode`. If you used the extractor when it was part of
IslandLife, its settings, Endfield asset maps, GF2 index and stores are moved over on the first start, so nothing
is built again.

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
build.py, main.py        the packaged program
```

## License and content

Polylode's code is under the GNU LGPL 2.1 (`LICENSE`); AnimeStudio, which it carries, is MIT
(`tools/AnimeStudio/LICENSE`). Polylode contains no game content: it converts files from the installs on your own
computer, and what it writes stays there. Use it only with games you own, and respect their terms.
