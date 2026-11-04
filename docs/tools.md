# Polylode 的工具與內部說明

（從 IslandLife 的 `docs/tools.md` 搬來，2026-11-02；節號 §7.x 指的是 IslandLife 的逐節紀錄 `docs/updates/`，證據留在那裡。）

## 匯出工具

`python -m polylode.extractor` 開視窗；同一支程式也是 `--scan <來源> <settings.json>`（每個模型印一行 `MODEL <json>`）、
`--run-job <job.json>`（印 `LOG`／`OUTPUT`／`ERROR`，失敗 exit 1）與 `--selftest <報告.json>`（匯入、來源清單與 Blender 要讀的原始檔）。
視窗把每一次掃描與匯出都用 `QProcess` 另起一個自己，轉檔器出錯不會帶走視窗。設定在 `%LOCALAPPDATA%/Polylode/extractor/settings.json`。

- GF2：掃描先 `bundles.scan_directories`＋`bundles.scan`（續跑：大小與 mtime 不變的 bundle 不再讀），再 `gf2.batch.declared_entries` → `selection.roots_only`；
  `conversion.json` 記為 `not_model`／`excluded_scope` 的不列。匯出 `gf2.batch.convert_entry` → `gf2.export.export_one`（Blender 寫出與讀回），
  `ok` 才複製 FBX（連同 `textures/`）與 `.blend` 到 `<輸出>/gf2/<編號>/`。轉換暫存仍在 `gf2.store.root()`。
- GF2 的可見性（沒有審過的 `display_profiles` 時，`gf2.export.scene_document`）：三條規則的聯集，被藏的照樣寫進 FBX 與 `.blend`、只是隱藏（Blender 匯入 FBX 時會照 `Visibility` 0 隱藏）。`.glb` 只寫顯示的部件（`interchange.glb.shown`，三個來源都一樣）：glTF 唯一的隱藏方式 `KHR_node_visibility` 在 Blender 5.2 與多數檢視器被忽略，隱藏的 lod1、替換造型會全部疊在 lod0 上；放進第二個場景或不放進任何場景也實測過，Blender 一樣會匯入。
  - LOD：每個 LOD 家族只顯示最細的一層（`presentation.display_lods`）。
  - 可替換部件（`gf2/parts.py`）：有些衣裝在同一個 prefab 裡放了一個部位的好幾種版本（妮基塔 SSR0102 的上衣、裙子、腿各有
    `P1`／`P2`／`P3`，renderer 全都 active），遊戲執行時每個部位挑一種。挑法是安裝裡 `LocalCache/Data/Table` 的資料表（protobuf，
    `rows` 不靠 schema 解）：`ModelConfigData` 名稱 → 模型編號；`PartsGroupData` 的部位（欄位 5 是衣裝 = 模型編號 + 1 000 000，表上 36 套都成立）；
    `PartsListData` 的選項（欄位 7 部位、欄位 2 是 P 幾；欄位 4 = 1 是不用解鎖、預設穿的那一個，其他帶欄位 5 的解鎖條件，111 個部位都是 index 1）；
    `PartsResourceData` 是選項的節點名（`P1_hip2` 對到 `…_P1_hip2_trans_lod0`）。預設以外的選項的節點藏起來；記在文件的 `parts_choice`。
    少數選項（欄位 8／9，例：庫蕾 SSR0104）的資源不在 `PartsResourceData`，不是換節點，這裡不動它們。
  - 宿舍版本：不是宿舍模型的模型，`_Dorm` 字尾（不分大小寫）的部件藏起來（`presentation.combat_variants`）。`slg` prefab 是戰鬥模型；
    依據是遊戲程式的字串常值：戰鬥合併旁的 `(\w*_lod[0-1](?:_Fight)?)`、可替換部件程式（`ReplaceableModelRes`）裡的 `Fight`，全程式沒有 `_Dorm`。
    戰鬥版和宿舍版不一定在同一個 renderer（春田 SSR0101 的戰鬥高跟鞋在 `cloth2_lod0_Fight`、宿舍涼鞋在 `body_lod0_Dorm`），戰鬥字尾也有
    `_fight`（昭輝、Vepley、Cheeta）與 `_Flight`（Nagant SR0101）的拼法，所以只認宿舍字尾與模型名：63 個宿舍模型都叫 `…Dorm_…`，
    其中唯一帶 `_Dorm` 部件的（Phaetusa）只有宿舍版。
    遊戲程式（HybridCLR 的 `Assembly-CSharp.dll.bytes`）的方法本體是加密的，只讀了沒加密的字串常值表 `#US`，沒有解密程式。
- 鳴潮、GF2 全部可玩角色（§7.1455）：`wuthering_waves.scope.read_scope` 讀每個 `RoleType` 1、非試用的角色（網格在 `/Game/Aki/Character/Role/` 下）；GF2 的匯出在 `gf2_source.own_scope()` 裡跑：`gf2.gender.all_playable()`（角色表的每個身分都通過，選取記 `all_playable`，只有這個模式裡算數）＋工具自己的 GF2 倉庫（`ISLANDLIFE_GF2_DIR` → `assets_cache/extractor/gf2_store`）。遊戲自己的轉換仍是已核實女性。
- 模型檢視器（§7.1455，`extractor/viewer.py`）：同一個 `preview.qml`（多了 `wireframe`、`backdrop`、`light` 與 `look(yaw, pitch)`），列出輸出資料夾的 `.glb`／`.gltf`／`.fbx`／`.obj`，可開啟或拖進檔案；`ModelExtractor.exe <檔案>` 直接開在這頁。glTF 的內容由 `model_info` 讀它自己的 JSON（頂點按 POSITION 陣列只算一次）。
- 分享到 GitHub Pages（`extractor/share.py`，`--share <模型.glb> <網站資料夾>`）：Discord 不顯示 3D 模型，連結只顯示 Open Graph 卡片，所以模型寫成網站上的 `models/<名稱>/`（`model.glb` 先經 `shown_copy` 拿掉隱藏部件、`card.png` 1200×630 由 `preview.snapshot` 照預覽離屏畫出、`index.html` 用固定版本的 `model-viewer`），首頁列出所有分享的模型。網址由 origin 推得（`https://<帳號>.github.io/<倉庫>/`）；只 `git commit -- <那些路徑>`，使用者自己暫存的東西不會被帶進去；單檔超過 GitHub 的 100 MB 上限就拒絕。倉庫裡被刪掉的模型（`models/<名稱>/model.glb` 不在了）不再列入：API 分享用 `git/trees/<分支>?recursive=1` 對過 `models.json`（GitHub 回應被截斷時不刪），本機複本只算有 `model.glb` 的資料夾；首頁的 `site.js` 在卡片圖載不到、而且模型檔回 404 時把那張卡片拿掉，下次分享前也不會留著壞掉的項目。
- 用 GitHub 帳號分享（`extractor/github.py`，`--share-github <模型.glb>`）：網站資料夾留空時走這條。輸出行多了 `DEVICE <代碼> <網址>`（視窗複製代碼、開瀏覽器、顯示可取消的對話框）與 `PAGES <設定頁>`（Pages 還沒開時）。授權檔 `extractor/github_token.bin`（DPAPI）與 `github_account.json`（只有帳號名）在使用者資料夾；token 不會印出。測試用 `tests/test_github_share.py` 的假 GitHub。
- Discord Activity（`share.SITE_SCRIPT`，設定 `discord_activity`）：Discord 載入 Activity 是透過自己的代理（`<app id>.discordsays.com`，URL 對應的根 `/` 指到分享網站），而且只能連到那個網站，所以 3D 檢視器與 Discord Embedded App SDK（`extractor/web/`，固定版本、附授權）都由網站自己提供，模型頁也不再用 CDN。首頁帶 `?frame_id=` 開啟時就是 Activity：非同步做 SDK 的 `ready()`，點模型在同一頁開啟（換頁會丟掉 SDK 讀的 `frame_id`）；在一般瀏覽器裡卡片仍是連到各模型頁的連結。SDK 檔名用 `.js`：本機測試伺服器把 `.mjs` 回成 `text/plain`，模組腳本就被拒絕。`--selftest` 檢查這些檔案有沒有帶進打包版。
- 姿勢（`endfield/poses.py`，設定 `poses`，預覽照同一個設定）：角色自己 `animations/ui/` 底下、clip 自己 `m_LoopTime` 為真的 clip（endminm 24 個裡 10 個）；第一次用 `Pipeline.convert_clips` 從安裝解碼進倉庫（約 15 秒）。每格在角色自己的骨架上評估，經 `rigscene.skinned` 成為匯出骨頭的座標框：runtime 蒙皮用的變換（評估出的世界矩陣乘上靜止世界矩陣的反矩陣，接到 `bone_rest`，靜止時差 0）。只取 `rest_matrix` 會丟掉 clip 的縮放：jsspsi 的嘴唇縮到 0.35、瞳孔 0.16、眼睛高光 0.27，endminf 的牙齒 0.32、舌頭 0.41，臉差 8–12 mm；改用 `skinned` 後逐格比對 0.00 mm。存成 `arrays["anim<k>"]`；`interchange/glb` 把骨頭改寫成 translation／rotation 節點並寫成 glTF animation，只寫離開靜止的頻道。分享的網頁用 `model-viewer` 的 `availableAnimations` 列出姿勢，卡片圖由 `share.posed`（第一個姿勢的第一格寫進關節）畫出。
- 少女前線 2 的姿勢（`gf2/clips.py`、`gf2/poses.py`）：服裝自己名字的、不含 `_slg_`（戰鬥用）的 clip，`m_LoopTime` 為真、transform 曲線至少一半落在這套服裝的節點上（Nikketa SSR0102：8 個待機）。bundle 索引本來就列出每個 `AnimationClip`（沒有名字），`clip_index.jsonl` 只打開有 clip 的 2133 個 bundle 補上名字，之後只重讀有變動的。曲線照 Unity 的三個區塊（streamed 每個 key 存它那一段的三次式、dense、constant）依序編號，binding 依序取用：Transform 的位置／縮放／尤拉角 3 條、四元數 4 條，**不論 customType**（尤拉角的 customType 是旋轉順序；當成 1 條會讓後面每個 binding 都讀錯，全身縮成一點），最後沒有 binding 的曲線（展示動作有 54 條）不用。姿勢幀照存檔節點的同一條路做出（`to_project(local_matrix)`、根在原點、`rest_matrix`），不動的節點重建誤差 2e-8。`rest_matrix` 拿掉縮放，所以姿勢幀不是 clip 世界矩陣的 `rest_matrix`，而是 Unity 蒙皮用的變換（`clips.skinned`：世界矩陣乘上靜止世界矩陣的反矩陣，再接到場景的剛體靜止幀），縮放、剪切都在；glTF 寫成 `scale` 頻道。少了它，縮放過的骨頭底下每根骨頭都離父骨頭遠了那個倍數、皮卻是靜止的大小：Clukay SSR01 的 CommandCenterBackGround 把 `root` 放大 1.202，眼睛（自己的骨頭）跑到臉上方；`hand_*_Ctrl2_Jnt_skin`／`foot_*_Ctrl4_Jnt_skin` 被不等比縮放到約 2 倍，手腕、腳踝差到 75 mm；Cheyanne SSR0101 的裙擺三層不等比縮放，末端差 106 mm。寫檔時 `glb._local_trs` 對著 glTF 實際組出的父骨頭求每根骨頭的局部值，並用 `glb._fitted`（3×3 的極分解：最近的旋轉、沿它的軸的縮放）取最接近的 TRS；Unity 的變換階層本來就是一層層 TRS，所以這樣組回來跟 Unity 一樣。照 Unity 蒙皮（`world_frame @ inverse(world_rest)`）逐骨比對，Clukay、Cheyanne 的身體骨頭都是 0.0 mm。過場用的 clip 會在鏡頭看不到的地方把身體拉開（`Upgrade_*` 在臉部特寫時把 Clukay 兩邊肩胛骨推開 1 m、Cheyanne 的頸 0.46 m；Clukay 的 `Class_*` 0.54 m），自由轉動看就像身體撕裂：`clips.strays` 量骨架骨頭（名字 `_L`／`_M`／`_R` 結尾、父骨頭也是，根底下的臀部不算）離開父骨頭下原位的最大距離，超過 `poses.TORN`（0.4 m）就不寫；一般姿勢最多 0.13 m。離原位超過 10 m（`clips.HIDDEN`）的那一格是遊戲把部件送走藏起來（Cheyanne 臥室裡的鞋 217 m、Clukay 上課時的斧頭 100 km），不算撕裂。有些 clip 用 stepped key 在鏡頭用的那幾秒把骨頭瞬間移走再移回（Love_4 在 2.4–6.07 秒把 Clukay 的左肩移出肩胛骨 11 cm，Upgrade_* 的肩胛骨 1 m，Class_* 分兩步 0.33／0.54 m），自由轉動看就是手脫臼：`clips.unsnapped` 把骨架下骨頭的位置曲線在一格跳超過 5 cm（`clips.SNAP`）的地方切開，中位數離原位超過 5 cm 的段落放回原位（轉向保留），臀部（根骨頭底下）與超過 10 m 的隱藏不動。Clukay 一般動作一格最多跳 12 mm。
- 鳴潮的姿勢（`wuthering_waves/poses.py`）：角色資料夾的 `CommonAnim/UiAnim`，只取在模型自己骨架上、非 additive、**結尾回到起點**的 clip（UE 的 AnimSequence 沒有循環旗標；Xin 的 `_Loop` 與 `Team_Stand` 首尾差 ≤ 0.02 cm／1.7°，`_Start`／`_End` 差 86～309 cm，門檻 0.5 cm／2°）。遊戲用 ACL 壓縮，CUE4Parse 要 CMake 建的原生庫才解得開，所以讀取器的 `anim` 模式把序列化的骨頭資料原樣寫出（`<k>.acl`，patch 讓 `UAnimSequence` 保留它），連同 track→骨頭對照與參考姿勢，用 AnimeStudio 帶的 `acldb_zzz.dll`（終末地同一個）解碼：標頭 `0xAC11AC11`，每個 track 是 UE 空間的旋轉／位移／縮放。幀照 case 的靜止姿勢同一條路做出（鏡像、公分→公尺、組合），再把 UE 蒙皮用的變換（世界矩陣乘上原生靜止世界矩陣的反矩陣）接到 `bone_rest`（`rigscene.carried`），縮放都在，不動時差 6e-8。之前照 `bake_uniform` 逐幀去縮放：Xin 的武器在 `Ui_Chip_End`、`Ui_Resonant_Start` 被縮到 0 藏起來、`Ui_Weapon_*` 縮到 0.2，匯出卻是原大小（差 30 mm）；縮到 0 的軸畫成 `rigscene.LEAST_SCALE` 長，轉向照 clip（`rigscene.least`）。Xin 的 15 個 clip 逐骨比對 0.0 mm。
- 表情導出沿用 `poses` 設定，寫入 GLB。少前 2 按原始 renderer 路徑與形狀鍵 hash 綁定，Unity 百分比權重除以 100；鳴潮按原始曲線名稱綁定，保留 Unreal 權重與播放速率。表情權重和骨骼共用動作時間軸。
- 原始形狀鍵另列為 `expression/<名稱>` 動畫。終末地以角色自己的臉部與耳朵配置評估控制項、情緒及姿態預設。材質控制與未對應曲線保留在旁邊的 `<模型>.expressions.json`，不當成已經套用到 GLB。FBX 與 `.blend` 沿用原有靜態形狀鍵功能。
- 終末地資產索引包含 `MonoBehaviour`。索引完成記錄和每批收據都比對物件類型清單；舊索引會重建，避免表情配置永久查不到。
- 實際匯出已驗證終末地 jsspsi 的 339 個表情／臉部控制動畫、少前 2 Nikketa SSR0102 的 24 個表情控制、鳴潮 Xin 的 176 個控制。少前 2 與鳴潮的 GLB 經 Blender 5.2.2 匯入後閉眼控制權重為 1。內建預覽直接讀取 GLB 的完整權重時間軸，補齊 Qt 匯入器省略的後段形狀鍵，切換動畫時恢復預設權重。
- 匯出佇列與速度（`extractor/app.py`、`sources.shared_writes`）：佇列照設定「同時匯出」（1–4，預設 2）同時跑幾個匯出行程。各遊戲共用的轉換資料（GF2 的倉庫與 results 檔、clip 索引；終末地的倉庫轉換、clip 解碼、shader；鳴潮的設定表、部件網格、材質、shader 快取與檔案索引）由 `shared_writes(<遊戲>)` 跨行程鎖住，一次一個寫（預覽和匯出同時跑時也一樣），之後的組合、姿勢、寫檔各自平行。姿勢原本逐骨逐格算（Clukay SSR01 一個模型 210 萬次小 SVD，約 4 分鐘）：改成一根骨頭的所有格一次算（`rigscene.skinned`／`carried` 收整段、靜止幀只算一次；GF2 `clips._locals`、鳴潮 `_native_locals`），終末地另把只有一個 key 的曲線放到同一個時間格（`poses.one_grid`）走批次取樣，並把各 clip 分給最多 6 個工作行程（`--pose-frames`，`endfield_source.frames_in_workers`，失敗的那組改在本行程算）。一個 clip 只解碼一次（`clips.DECODED`）。結果與舊算法逐格相同（差 ≤ 7e-6）：Clukay 整個匯出約 5 分鐘 → 55 秒，jsspsi 25 個姿勢 167 秒 → 20 秒。`--selftest` 會啟動一個空的姿勢工作行程（`pose_worker`）。掃描也拿同一把遊戲鎖（`jobs.run_scan`），跟同遊戲的匯出、預覽不會同時寫共用索引。GF2 的掃描每次都把 8,059 個 bundle 的目錄重讀一遍、逐位元組解遮罩：`bundles.scan_directories` 改成大小與時間沒變的檔案沿用舊列、解遮罩用 numpy（`_unmasked`），熱掃描 87 秒 → 8 秒，8,059 列逐列相同。第一次掃描或遊戲更新後的冷掃描，bundle 索引（`bundles.index_row`）與根節點檢查（`batch.checked_groups`）分給工作行程（`--worker gf2-index`／`gf2-roots`，`jobs.in_workers`，切成連續的段、照順序接回，失敗的段改在本行程做）；工作行程啟動約 3–4 秒，所以超過 100 個 bundle（索引）／150 個（根節點）才分。200 個 bundle 的索引：本行程 10.6 秒、6 個行程 6.0 秒，結果相同。`--selftest` 也會啟動一個 `--worker echo`（`worker`）。
- 預覽（§7.1455，`extractor/preview.py`、`preview.qml`）：選一列按「預覽」，用匯出的同一個子行程只寫 `.glb`、不經 Blender，到 `assets_cache/extractor/preview/<來源>/<模型>/`；Qt Quick 3D 的 `RuntimeLoader` 讀它、照邊界取景，`OrbitCameraController` 轉動。打包帶 `--include-qt-plugins=qml`，`--selftest` 的 `preview_qml_errors` 實際載入一次。
  - 姿勢：`RuntimeLoader` 把每個 glTF animation 做成一個 `Timeline`＋循環的 `TimelineAnimation`（以動畫命名，只有第一個啟用）。載入完成後 `view_host.Poses` 找出它們（略過關鍵格：GF2 Clukay 31 個姿勢有 57 萬個 `Keyframe`，全部找一遍要 12.7 秒），回報 `POSES [...]`，預覽面板與檢視器列成下拉選單；換姿勢時先停用其他 timeline（停用會把它動過的骨頭放回原處，檔案裡沒寫的靜止頻道才回得去）再啟用選的那個。`--selftest` 載入帶兩個動畫的三角形，確認找得到兩個姿勢。姿勢可能把角色放到別處（GF2 的 CommandCenterBackGround 把 Clukay SSR01 放在旁邊 2.4 m、1.2 倍大），所以每換一個姿勢，`Poses` 量骨架（姿勢會動的節點與其下的節點，網格節點不算）的位置，把檔案的靜止範圍跟著移過去、保留它超出骨架的邊距，再從目前的角度重新取景（`preview.qml` 的 `refit`）；離骨架中央超過靜止骨架 3 倍遠的節點（遊戲把部件送走藏起來，Cheyanne 的鞋 217 m）不算。
- 角色著色（`extractor/character_styles.py`、`CharacterMaterial.qml`、`character.frag`）：預覽與模型檢視器可選攝影棚、終末地、少女前線 2 或鳴潮。三款遊戲分別使用來源程式的漫反射／色階／高光公式；原始材質參數與查表貼圖隨 GLB 保存，切換模型後保留選定模式。這是角色著色的預覽改編，不包含完整遊戲場景後製、投影與眼睛／頭髮專用效果。
- Blender 著色輸出（`extractor/shader_export.py`、`blender_shader.py`）：預設勾選「一併輸出 Blender 角色著色」。每個模型另有 `<名稱>.shaders/`，包含完整 GLB、相對路徑貼圖、材質對應表、著色器原始碼與獨立 Blender 建置腳本；設定 Blender 後會在背景建立並讀回 `<名稱>.shaded.blend`，貼圖和可編輯節點都封裝於檔內。原本 `.blend` 保留原有眼睛分層等結構。未設定 Blender 時保留完整資料包並明示尚未建立著色版；資料夾中的 README 提供重建命令。預覽不執行這個額外匯出。
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

## 終末地沒有底色圖的材質：執行遊戲自己的 shader 烘焙（`endfield/vfx_bake.py`、`endfield/shader_code.py`）

寫出器靠底色圖畫材質。沒有底色圖的材質（特效 shader `HGRP/CharacterNPR_VFX`：Si 的翅膀、尾端的光環、潛能特效）以前寫成沒有貼圖的灰白平面。
現在在 `fbx_export.scene_document` 寫出交換文件後（`.blend`、glb、FBX 三條路都經過這裡）由 `vfx_bake.complete` 補齊：

1. **shader 程式碼**（`shader_code.py`）：匯出器把 `Shader` 物件寫成 JSON；每個 sub-shader 的 D3D11 blob 是 LZ4 分段、開頭一張
   `(offset, length, segment)` 項目表，項目是程式（各階段的 DXBC）或參數區塊（每個常數緩衝的變數與 byte offset、每個資源在像素／頂點階段的 slot）。
   依材質的 keyword 挑 pass（`ForwardOnly`／`UniversalForward`／`SRPDefaultUnlit`）裡「包含全部 keyword、多餘的最少」的變體。快取裡沒有的 shader
   第一次由 `endfield_source._fetch_shaders` 從安裝取出（存在倉庫的 `shaders/`）。
2. **只烘不打光的 pass**：pass 的 `RenderType` 含 `Unlit` 才烘，它的輸出就是畫出來的顏色；打光的 pass 烘下去會帶進這個程式自己選的燈，拒絕。
3. **執行**：先在每個三角形角上跑頂點階段（網格的位置、法線、切線、頂點色、各組 UV 照語意餵入；蒙皮在靜止時由 `unity_LODFade` 跳過），
   輸出照重心座標內插到底色格的每個像素，再跑像素階段（鳴潮的直譯器 `wuthering_waves/shaders/vm.py`，現在也保留 `o#` 輸出）。
   網格沒有頂點色時讀白色（Unity 對缺少的 COLOR 給白，也是 Si 的翅膀在遊戲裡看得見的唯一值：顏色的 alpha 就是它的覆蓋率）。
   其他網格沒有的輸入（切線、蒙皮權重、UV 的 zw）各以 0 和 1 試跑，顏色差超過 0.02 就不烘。
4. **常數**：材質的常數緩衝（`UnityPerMaterial`、`_Globals`）依變數名從材質讀，沒有就用 shader 宣告的預設值，再沒有就是 0（Unity 的材質緩衝初值）。
   其他緩衝只給 `vfx_bake.FRAME` 列的靜止畫面值：單位矩陣、時間 0、沒有鏡頭淡出與 LOD 交錯、曝光 1、反向深度的 `_ZBufferParams`、頂點動畫關閉
   （`_DisableAnimateVert` 1：匯出的是靜止姿勢）、沒有霧（指數霧、大氣霧、體積霧的參數全 0；體積霧由 `_VolumetricFogParams0.z > 0` 開關）。
   霧的值用實驗確認：同一個 `_USE_FOG` 材質拿掉這個 keyword 再烘，結果逐像素相同。畫面貼圖給一個像素：`_CameraDepthTexture` 遠平面 0、
   `_IntegratedLightScattering` (0,0,0,1)。讀到 `FRAME` 以外、沒有值的變數就不烘。
5. **混合**：照 pass 宣告的混合因子讀輸出：`One, OneMinusSrcAlpha` 是預乘、`SrcAlpha, OneMinusSrcAlpha` 是一般 alpha；其他組合不烘。
   預乘輸出裡 alpha 是 0 卻有顏色的像素（`_BlendMode` 1：只把光加到後面，21 個角色的 32 個特效裡有 16 個是這樣）在 glTF 沒有加亮混合可用，
   改成「覆蓋率 ＝ 亮度、顏色 ＝ 加上去的光 ÷ 亮度」（`straight`）：加上去的光照樣在，但後面的東西會暗 (1 − 亮度)，遊戲不會。寫法是 glTF 畫不打光表面的方式：底色係數黑、底色圖的 alpha 是覆蓋率、顏色當發光（超過 1 的部分縮進貼圖，倍數寫成
   `KHR_materials_emissive_strength`／FBX 的 `EmissiveFactor`／Blender 的 Emission Strength）。
6. **角色 shader 的發光**（`npr_emission`）：`HGRP/CharacterNPR`、`_Hair`、`_Skin`、`_Eye`、`_LiquidAg` 的 ForwardLit 在 `_EMISSION` 時都加上
   同一項 `_EmissionMap`.rgb × `_EmissionColor` × `_EmissionBrightness`（逐支讀過）；視差水晶那一項只讀了主體 shader。主體 shader 的 ForwardLit 在打光後的顏色上加上 `_EmissionMap`.rgb × `_EmissionColor` ×
   `_EmissionBrightness`，以及（`_PARALLAX_MAP`）`_ParallaxTex`.r × `_ParallaxColor` × 底色 alpha × `_BaseColor`.a；視差圖照視線步進的位置取樣，
   正面看就是它自己的平鋪位置。Si 的角就是這樣：一層淡的發光圖，加上底色 alpha 標出的水晶。烘成發光圖寫出。
7. **不需要烘的情況**：疊加陰影層（`HGRP/CharacterNPR_OverlayShadow`）沒有貼圖時，寫出器用它的色調乘上去（§7.1455），算可畫；
   每個 pass 的顏色遮罩都是 0（`Effect/VFXTransparentDepthOnly`：只寫深度；遮罩常是材質屬性 `_ColorMask`，要讀材質的值，render state 裡的 0 只是佔位）、或混合是 (Zero, One)（不改變畫面）就照遊戲不畫顏色、設為不可見；打光的 shader 沒有綁底色屬性時
   （`shading.base_property`，例：Purrche 的 `_BaseMap`），遊戲取 shader 宣告的預設貼圖（`white`／`black`／`gray`…）乘 `_BaseColor`，
   寫成定色，`_BaseColor` 的 alpha 只在 `_ALPHABLEND_ON` 時算（Purrche 那個是 0：透明）。
8. **烘不了的材質不畫灰白**：標成 `native_surface_unmapped` 並寫下原因（`unmapped_reason`），三種格式都畫成透明，記錄列出原因。
9. **快取與速度**：結果依規則版本、shader、材質與網格存在倉庫的 `baked/`，同一次匯出的 `.blend` 與 glb／FBX、下一次匯出都重用。
   缺的輸入與螢幕位置的影響都先在 4096 個抽樣點上試（`PROBE`）；結果和螢幕位置無關就只跑一個位置。直譯器的逐分量指令只算目的遮罩的分量
   （`vm.ELEMENTWISE`）。Si 的翅膀 21.7 秒 → 4.0 秒，結果逐位元相同；鳴潮的覆蓋率也照同樣的抽樣決定螢幕位置（有抖動的仍跑十個），
   「心」重烘的兩張圖與舊結果逐位元相同。

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
4. **執行**：`shaders/vm.py` 直譯 `ps_5_0` 組合語言，所有點同時執行，暫存器存原始位元，分支與迴圈用遮罩。數值規則照 Direct3D：`min`／`max` 遇到 NaN 回傳另一個運算元、saturate 把 NaN 變 0、`ftoi`／`ftou` 把 NaN 變 0 並夾在型別範圍內（「心」下身的閃光高光在零光向量下算出 NaN，用 NumPy 的 `maximum` 會一路傳到不透明度、把整雙腿丟掉；遊戲的 GPU 在 `max(…, 0)` 就把它變成 0）。對底色圖每個被三角形
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
