# Polylode 的工具與內部說明

（從 IslandLife 的 `docs/tools.md` 搬來，2026-11-02；節號 §7.x 指的是 IslandLife 的逐節紀錄 `docs/updates/`，證據留在那裡。）

## 匯出工具

`python -m polylode.extractor` 開視窗；同一支程式也是 `--scan <來源> <settings.json>`（每個模型印一行 `MODEL <json>`）、
`--run-job <job.json>`（印 `LOG`／`OUTPUT`／`ERROR`，失敗 exit 1）與 `--selftest <報告.json>`（匯入、來源清單與 Blender 要讀的原始檔）。
視窗把每一次掃描與匯出都用 `QProcess` 另起一個自己，轉檔器出錯不會帶走視窗。設定在 `%LOCALAPPDATA%/Polylode/extractor/settings.json`。

- GF2：掃描先 `bundles.scan_directories`＋`bundles.scan`（續跑：大小與 mtime 不變的 bundle 不再讀），再 `gf2.batch.declared_entries` → `selection.roots_only`；
  `conversion.json` 記為 `not_model`／`excluded_scope` 的不列。匯出 `gf2.batch.convert_entry` → `gf2.export.export_one`（Blender 寫出與讀回），
  `ok` 才複製 FBX（連同 `textures/`）與 `.blend` 到 `<輸出>/gf2/<編號>/`。轉換暫存仍在 `gf2.store.root()`。
- 鳴潮、GF2 全部可玩角色（§7.1455）：`wuthering_waves.scope.read_scope` 讀每個 `RoleType` 1、非試用的角色（網格在 `/Game/Aki/Character/Role/` 下）；GF2 的匯出在 `gf2_source.own_scope()` 裡跑：`gf2.gender.all_playable()`（角色表的每個身分都通過，選取記 `all_playable`，只有這個模式裡算數）＋工具自己的 GF2 倉庫（`ISLANDLIFE_GF2_DIR` → `assets_cache/extractor/gf2_store`）。遊戲自己的轉換仍是已核實女性。
- 模型檢視器（§7.1455，`extractor/viewer.py`）：同一個 `preview.qml`（多了 `wireframe`、`backdrop`、`light` 與 `look(yaw, pitch)`），列出輸出資料夾的 `.glb`／`.gltf`／`.fbx`／`.obj`，可開啟或拖進檔案；`ModelExtractor.exe <檔案>` 直接開在這頁。glTF 的內容由 `model_info` 讀它自己的 JSON（頂點按 POSITION 陣列只算一次）。
- 預覽（§7.1455，`extractor/preview.py`、`preview.qml`）：選一列按「預覽」，用匯出的同一個子行程只寫 `.glb`、不經 Blender，到 `assets_cache/extractor/preview/<來源>/<模型>/`；Qt Quick 3D 的 `RuntimeLoader` 讀它、照邊界取景，`OrbitCameraController` 轉動。打包帶 `--include-qt-plugins=qml`，`--selftest` 的 `preview_qml_errors` 實際載入一次。
- 找安裝（§7.1454，`extractor/locate.py`）：每個來源的安裝欄位有 `PathSpec.marker`（鳴潮 `Client/Content/Paks`、GF2 `GF2_Exilium_Data`、終末地 `Endfield_Data/StreamingAssets/VFS`）；`sources.require` 從填的資料夾往上（到特徵的層數）或往下三層找唯一一份，有好幾份就報錯不挑。掃描前的詢問先列 `locate.steam_installs`（登錄檔的 Steam 位置＋`steamapps/libraryfolders.vdf` 的每個遊戲庫，`steamapps/common/<遊戲>` 往下兩層）。
- 終末地的資產對照表（§7.1452）：`endfield.mapbuild.build(exe, Endfield_Data, work, meta)` 從安裝自動建，`endfield_source.ensure_maps` 在 VFS 檔案戳記不變且已建成時沿用；步驟與檢查見模組說明，可續跑。打包版內建 AnimeStudio（`tools/AnimeStudio`，MIT）與 .NET 10（`dotnet`），`bundled.py`；`build.py --dotnet <資料夾>`（預設 `DOTNET_ROOT`）。
- 終末地（`endfield_source`，§7.1446 時的版本）：要安裝、AnimeStudio CLI 與研究套件從這個安裝建的 `meta`；工具不自己建表（現行遊戲單趟建表會 OOM，
  不帶 `--filter_data` 會讀進被取代的資料槽，§7.1070 是分 255 批建的），讀表後用 `CabIndex.outside` 核對區塊路徑都在所指的安裝底下。
  掃描 `PL.select(…, ["chara"])` 全部列出；匯出不判範圍（§7.1455，擁有者 2026-11-02：遊戲原本的模型），轉進工具自己的倉庫
  （`endfield_source.own_store`：`store.DIR_ENV` 釘在 `assets_cache/extractor/endfield_store`，不查 `Charas`），遊戲的倉庫與範圍規則不受影響。
  寫檔是 `endfield.fbx_export.write_one(…, blend=True)`：FBX 寫出與讀回之外，同一個場景另存 `.blend`（貼圖打包），
  再開一次核對網格、骨架與「沒有不在檔內的圖」（`blend_problems`）。
- 鳴潮（`wuwa_source`）：要遊戲資料夾、`tools/wuwa_cli` 建出的讀取器（`.exe` 直接跑、`.dll` 經 `dotnet`）、.NET 10 資料夾與 AES 金鑰檔。
  掃描用 `raw-list` 抄出 `wuthering_waves.scope.FILES`（角色、預載、繁中語言表與四份 JS 讀取器），再用 `json` 讀模型表
  （`assembly.MODEL_TABLE`），都放在 `assets_cache/wuthering_waves/config`，封存檔的數量／大小／時間不變就不再讀。
  `scope.read_scope` 照遊戲自己的 getter 讀 FlatBuffers 列，`assembly.display_models` 照角色選單的組法每個主網格一列，
  再去掉 `converted.scope.EXCLUDED_WUWA`。
  匯出：每個部件 `mesh` 傾印到 `assets_cache/wuthering_waves/source/<部件>`（已有成功的 `export.json` 就不重讀），再對這些部件跑 `metadata`；
  `wuthering_waves.source_scene.assemble` 組成 `cases/<編號>`，`tools/wuwa_source_blender.py` 的 `build` 建 `.blend`／FBX、`check` 讀回
  （FBX 與開回的 `.blend` 都要在 0.05 mm、骨頭 0.01° 內，`.blend` 裡不能有不在檔內的圖），通過才複製出去。
- `.glb`（§7.1447）：`polylode.interchange.glb.write` 讀三個來源共用的交換文件與陣列寫出，`write_checked` 在有 Blender 時以
  `tools/ef_fbx_blender.py` 的 `glb` 模式讀回（門檻同 FBX，`readback_problems`）；`.glb` 不需要 Blender，`.blend`／FBX 需要（`sources.blender_for`）。
  獨立的規格檢查用 Khronos 的 glTF-Validator（npm `gltf-validator`，在自己的資料夾裝，不進專案）。
- FBX（§7.1448）：`polylode.interchange.fbx.write` 從同一份交換場景寫 FBX 7.4 二進位，`write_checked` 一律用 `endfield/fbxfile.bone_errors`
  量檔內骨頭與綁定矩陣（0.01 mm／0.01°），有 Blender 時再以 `check` 模式匯入量（`glb.readback_problems`）。只有 `.blend` 需要 Blender（`sources.NEEDS_BLENDER`）。
  第三方讀取器的檢查用 three.js 的 FBXLoader（npm `three`，Node 裡把貼圖載入換成空殼），裝在自己的資料夾，不進專案。
- Autodesk FBX SDK 讀回（§7.1449）：`<裝了 fbx wheel 的 Python 3.11> tools/fbx_sdk_check.py <檔.fbx> <場景.json>`。SDK 是 Autodesk 的，
  從它的 FBX SDK 頁下載 `fbx2020311_fbxpythonsdk_win.exe`，用 7-Zip 解出 `fbx-2020.3.11-cp311-none-win_amd64.whl` 裝進自己的 venv（加 numpy），
  不進版控也不進發行物。印一行 JSON：`imported`／`status`、軸與單位、骨頭全域靜止偏差與父骨、`parts_bad`（控制點、多邊形、蒙皮、權重頂點、
  形態鍵、UV 組、法線任一不同的部件）、`bind_mm`、貼圖數、SDK 解出的內嵌圖（`<檔>.fbm`）與底色圖的縮放／偏移（`placements`）。
- 既有轉出物（`converted_source`）：讀所選資料夾底下三個 `*_Converted/covered_models.json`，清單記的是絕對路徑，
  照資料夾名之後的那一段對回所選資料夾；鳴潮套 `converted.scope.EXCLUDED_WUWA`（`converted/build.py` 也從那裡讀）。
  複製後逐檔比對清單的 `*_sha256`，不同就刪掉複本並報錯。
- 打包版在 `%LOCALAPPDATA%/Polylode` 下有自己的倉庫與快取；要讓它讀 checkout 的終末地倉庫，設 `ISLANDLIFE_ENDFIELD_DIR` 與
  `ISLANDLIFE_ENDFIELD_CHARAS_DIR`（後者指 `Charas/Endfield`，只設前者會關掉 `Charas` 查找）。子行程把標準輸出固定成 UTF-8（主控台的 cp950 放不下所有角色名）。
- 建置：`python build.py [--jobs N]`（預設 8 個 C 編譯器；每個都在 guard 的同一個上限裡，失敗時已編好的留在 Nuitka 的 clcache，再跑會接續）。
  只收 `polylode.extractor`，轉換器靠 import 跟進；執行時不載入的 `scipy.stats` 等與 `polylode.game`／`ui`／`endfield.seams` 不跟進；
  `UnityPy.classes.generated`（31 MB 的 C 檔，單獨也編不進 9 GB）以 `.py` 放在程式旁、由 `blender_files.load_source_modules` 載入；
  PyOpenGL 的 Cython 加速模組不收（編進去會 `KeyError: '__reduce_cython__'`）。結果 `dist/ModelExtractor/`：exe 80 MB、整個資料夾 245 MB、664 個檔。
- Blender 用的原始檔：建置把 `polylode/` 與 `tools/` 裡的 Blender 腳本（`blender_files.REQUIRED`）原樣複製到程式旁邊，
  因為 Blender 以 `.py` 執行 worker、GF2 匯出也用它們的位元組簽章；之後跑 `--selftest`。`--check-only` 只做複製與自檢。
