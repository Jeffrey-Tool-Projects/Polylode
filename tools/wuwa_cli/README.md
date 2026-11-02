# wuwa_cli — the Wuthering Waves game-file reader

The model extractor (`python -m islandlife.extractor`, §7.1446) reads a Wuthering Waves install through
this small .NET 10 program (`islandlife/wuthering_waves/extract.py` runs it). It mounts
`<install>/Client/Content` with your AES keys file and writes what a mode asks for:

    wuwa.exe <install> <keys.json> <output> index
    wuwa.exe <install> <keys.json> <output> raw-list <list.txt>   # copy those archive files out as they are
    wuwa.exe <install> <keys.json> <output> json <package>        # one package's exports as JSON
    wuwa.exe <install> <keys.json> <output> blueprints <scope.json>
    wuwa.exe <install> <keys.json> <output> mesh <package>        # one skeletal mesh, its materials and textures
    wuwa.exe <install> <keys.json> <output> metadata <scope.json> # each exported texture's colour space

`<keys.json>` holds `mainKey` and `dynamicKeys` (`[{guid, key}]`) for your client version.

## Building

1. Clone [CUE4Parse](https://github.com/FabianFG/CUE4Parse) at commit
   `2a159ad0702120db30fdd85c1219dff78e5204b8` (§7.1121) into `tools/wuwa_cli/CUE4Parse`
   (or anywhere, and pass `-p:CUE4ParseRoot=<that folder>`), with its submodules.
2. Apply `cue4parse_wuthering_waves.patch` there (`git apply`): this game's material parameter
   records carry a 4-byte flags field and do not pad scalar parameters to 8 bytes.
3. `dotnet build -c Release tools/wuwa_cli/wuwa.csproj` (the .NET 10 SDK). `Detex.dll`, the Oodle
   and zlib libraries must sit beside the built `wuwa.exe`.

Nothing it writes is tracked or shipped: it is your own game data, on your machine.
