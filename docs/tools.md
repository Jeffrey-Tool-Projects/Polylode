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

## 鳴潮的裁切材質：執行遊戲自己的 shader 烘焙覆蓋率（`wuthering_waves/shader_bake.py`、`shaders/`）

鳴潮的裁切材質（`BlendMode` 1）透明度不一定是底色圖的 alpha。例如「心」的狐狸尾巴用毛髮 shader，透明度來自 `TypeMask`.R、
`Fur_Tex`.R（第二組 UV）、頂點色、螢幕空間抖動，有的組合甚至用受光顏色的亮度。每種靜態開關組合的編譯結果都不同，所以不寫成規則，
而是直接執行遊戲編譯好的 shader：

1. **shader map**：材質實例的 `.uexp` 嵌有自己的 shader map（凍結記憶體映像）。`shaders/inline.py` 從名稱表、修補表找出 uniform expression
   set（向量／純量 preshader、參數、2D 貼圖參數、preshader 位元組碼，在凍結內容裡首尾相接）。CUE4Parse 的傾印裡這些是空的。
2. **uniform buffer**：`shaders/preshader.py` 依 UE 4.26 的 `EMaterialPreshaderOpcode` 執行 preshader，用實例自己的參數值填出 Material
   uniform buffer（前面是向量，後面每 4 個純量一格）。遇到沒讀過的指令就報錯。
3. **shader 程式碼**：shader map 的雜湊對到 `ShaderArchive-Client_Chunk-PCD3D_SM5.ushaderbytecode`（`shaders/archive.py`；LZ4 壓縮）。
   選陰影深度 pass 裡最小、有 `discard` 的像素 shader。`shaders/dxbc.py` 讀 UE 的資源表（暫存器 → uniform buffer 資源；材質第 i 張
   2D 貼圖是資源 2i）、uniform buffer 名稱、輸入簽章，用 `d3dcompiler_47` 反組譯。
4. **執行**：`shaders/vm.py` 直譯 `ps_5_0` 組合語言，所有點同時執行，暫存器存原始位元，分支與迴圈用遮罩。對底色圖每個被三角形
   蓋到的像素，代入內插的 UV（UE 的 v 朝下）、頂點色、切線框架，從正面看（相機相對位置 −N、每點各自的視圖矩陣），在 10 個螢幕位置
   （涵蓋 5 種抖動圖樣）上執行，沒被丟棄的比例就是 alpha。大多 0／1 的材質用 0.5 裁切，有部分覆蓋的用半透明。
   **UV 重疊**：一張圖每個像素只能放一個覆蓋率，但三角形可能疊在同樣的像素上、覆蓋率卻不同（頂點色或其他 UV 不同；
   「心」耳內的不透明視差毛髮卡片疊在毛絲卡片上，只取第一個三角形時卡片被切成灰色碎片）。所以每個（像素, 三角形）都算；
   每張圖由還沒放好的三角形先到先得填滿，自己的取樣和圖平均差超過 0.25 的三角形移到下一張（`_layers`，最多 4 張）。
   第 k 張圖給材質的複本 `<材質>_Overlap<k>`，那些三角形搬到自己的部件 `<部件>_Overlap<k>`（一個部件一個材質，和 `split.by_material` 一致，
   形態鍵跟著搬）。左右鏡像的島覆蓋率相同，會留在同一張。為了不讓重疊的地方重跑 shader：每個像素先跑第一個三角形，每個三角形再抽 6 個
   自己的取樣比對，只有不一致的三角形整個重跑（`_covered`），其餘直接用那個像素的值。
5. **執行期欄位**：給的是遊戲預設畫面在 UE 4.26 View／Primitive uniform buffer 的值，每一格都從 shader 的用法認出來（`shader_bake.RUNTIME`／`FRAME`）：
   View [12..15] 視圖矩陣（每點各自朝 −N 看）、[36] ClipToView 第一列 1（90° 視野）、[67]／[70] 相機位置 0（座標以相機為準）、
   [134] 視窗原點、[148] 遊戲時間、[149] mip 偏移、[158] 影格都是 0、[144] `DiffuseOverrideParameter` (0,0,0,1)、[147].w 1；
   Primitive [0..3] LocalToWorld 單位矩陣、[4].w 行列式符號 1、[27] 自訂資料與 [33] 濕度 0；材質參數集合用資產的預設值。
   讀到這以外欄位的 shader 不烘（之前 LocalToWorld 留 0 時，「心」下身的距離算成 NaN、腿整段不見）。
   shader 讀了網格沒有的輸入（頂點 shader 算出的自訂 UV，輸入簽章的 Used 欄）時，先用 2 萬個點各以 0 與 1 試跑，結果變動超過 1% 就不烘。
6. **快取與失敗**：遊戲檔（約 1.2 GB 的封存檔、材質套件、上層材質與參數集合的 JSON）第一次由讀取器取出，放在
   `assets_cache/wuthering_waves/shaders`，依安裝狀態區分。烘焙結果依材質與網格快取。執行不了的材質照原本的裁切輸出，並在記錄說明原因。

### 視差毛髮做成毛絲層（`wuthering_waves/fur_shells.py`）

毛髮 shader 的 `UseFurryParallax`（從「心」尾巴的 base pass 讀出）只作用在頂點色 alpha 0.4～0.6 的毛髮卡片上。**卡片本身是底面**，毛從底面往外長，
最高 `Height` × 0.01 UV；高度場是 `HeightMap`.R，在卡片 UV 上鋪 `HeightMapSize`（T）次。shader 從最高處沿視線往下走 n 步
（n = lerp(`MaxStep`, `MinStep`, |N·V|)，正面看是 `MinStep`）；在高度 t 時視線位於 uv + P·t，P = (V·T, V·B)/(V·N) × 深度。
第一個「毛比 t 高」的層決定像素的樣子；一層都沒有就落到卡片，照原樣顯示。往上長的時候有兩件事會移動毛：

- **邊緣淡出**：所有高度乘上 smoothstep(`ParallaxRimFadeMin`, `ParallaxRimFadeMax`, N·V)，越斜看毛越貼近卡片。
- **流向圖**：第 j 步（高度 t = 1 − j/n，j ≥ 1）查高度與顏色的位置，都偏移 m_j × (`FlowmapMap`.RG × 2 − 1) × `FlowMapInt` / n 個高度圖格，
  其中 m_j = `FurRootInv` × (n + 2 − 2(j − 1)) + (j − 1)。流向圖在 `FlowmapScale` × uv 取樣，所以毛會順著流向傾斜。

glTF／FBX 不能走視線，所以把同一個高度場做成幾何：卡片原地不動，上面在每個高度 t = k/n（k = 1…n−1）各放一層。
- 每層只留比 t 高的毛：`<底色圖>_strands.png` 的 alpha 是毛的高度（只在卡片有覆蓋的地方），第 k 層的裁切值是 k/n。
- 流向的傾斜做法是把各層的 UV 在每個頂點按該層的偏移量移動；流向圖在一個三角形內平滑的地方，結果是精確的。
- 邊緣淡出取決於視角：每層記在 `fur_rim_fade`，`.blend` 在 shader 裡用 Geometry 節點的 N·V 乘到高度上再裁切；glTF／FBX 沒有這種節點，毛從各個方向看都是全高。
- 世界深度是 UV 深度乘上每個頂點的「世界長度／UV 長度」。各層放在另一個部件（`<部件>_FurLayers`）。
- `Plane` 不是 0 的材質走法不同，沒有讀過，不做毛絲層（記錄裡會寫）。

「心」兩套都只有尾巴部件裡的白毛（袖口、耳內毛球）落在這個頂點色範圍，深度約 6 mm，7 層。尾巴本體是另外的毛絲卡片（頂點 alpha 0 或 1），由上面的覆蓋率處理。
