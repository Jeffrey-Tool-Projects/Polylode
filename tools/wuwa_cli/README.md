# wuwa_cli — the Wuthering Waves game-file reader

Polylode reads a Wuthering Waves install through this small .NET 10 program (`polylode/wuthering_waves/extract.py` runs it), which the packaged program carries as `tools/WuwaReader`. It mounts
`<install>/Client/Content` with your AES keys file and writes what a mode asks for:

    wuwa.exe <install> <keys.json> <output> index
    wuwa.exe <install> <keys.json> <output> raw-list <list.txt>   # copy those archive files out as they are
    wuwa.exe <install> <keys.json> <output> json <package>        # one package's exports as JSON
    wuwa.exe <install> <keys.json> <output> blueprints <scope.json>
    wuwa.exe <install> <keys.json> <output> mesh <package>        # one skeletal mesh, its materials and textures
    wuwa.exe <install> <keys.json> <output> metadata <scope.json> # each exported texture's colour space

`<keys.json>` holds `mainKey` and `dynamicKeys` (`[{guid, key}]`) for your client version.

## Building

The CI and Release workflows build it on every push (`.github/workflows`); to build it yourself:

1. Clone [CUE4Parse](https://github.com/FabianFG/CUE4Parse) at commit
   `2a159ad0702120db30fdd85c1219dff78e5204b8` into `tools/wuwa_cli/CUE4Parse`, with its submodules.
2. `tools/wuwa_cli/build.ps1 -CUE4Parse tools/wuwa_cli/CUE4Parse -Output tools/WuwaReader` (the .NET 10 SDK
   on `PATH`). It applies `cue4parse_wuthering_waves.patch` (this game's material parameter records carry a
   4-byte flags field and do not pad scalar parameters to 8 bytes), builds, copies `Detex.dll` out of CUE4Parse and puts its licence
   beside `wuwa.exe`; `build.py` copies that folder into the program.

CUE4Parse fetches its Oodle and zlib-ng libraries on first use, into the folder the reader runs in
(Polylode runs it in `%LOCALAPPDATA%\Polylode\assets_cache\wuthering_waves\reader_runtime`).

Nothing it writes is tracked or shipped: it is your own game data, on your machine.
