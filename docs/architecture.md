# Polylode 的架構

（從 IslandLife 的 `docs/architecture.md` 搬來，2026-11-02；§7.x 是 IslandLife 的逐節紀錄。）

`polylode/extractor`（§7.1446）：獨立的 PySide6 匯出工具，不屬於遊戲執行期，也不進遊戲的發行物（自己的 `build.py`）。
`sources.Source` 是擴充點：每個來源給 `paths`（使用者要指定的安裝／工具）、`scan`、`export`；`app` 只認這個介面，掃描與匯出由 `jobs`
在另一個行程跑同一支程式。GF2 的「宣告的根節點」與「轉一個根節點」從 `tools/import_gf2.py` 搬進 `gf2.batch`，命令列與工具共用。
終末地的 FBX 寫出與讀回核對從 `tools/ef_export_fbx.py` 搬進 `endfield.fbx_export`（工具只剩命令列）；匯出工具轉遊戲原本的模型不判範圍，
轉進自己的倉庫（`endfield_source.own_store`，§7.1455），遊戲讀的倉庫看不到。`Pipeline.run` 的 `admit` 仍是在 Avatar 讀到後、材質讀取前
丟掉一個預製物的擴充點；`body_scope.MEN`（`hrole.MEN` 指向它）是 Free H 的男角。
`polylode/extractor/viewer.py`（§7.1455）：匯出工具的模型檢視器頁（任何模型檔、固定視角、線框、背景、燈光、glTF 內容）。
`polylode/extractor/share.py`：模型檢視器的「分享到 GitHub Pages」——把 `.glb` 寫成使用者自己 GitHub Pages 網站上的一頁（`model-viewer` 轉動模型、Open Graph 卡片圖由 `preview.snapshot` 離屏畫出），只提交那些檔案並推送；由 `--share <模型> <網站資料夾>` 這個獨立行程執行。網站是使用者指定的另一個倉庫，本程式自己的資料夾會被拒絕：遊戲的模型不隨程式發布。
`polylode/extractor/github.py`：沒有指定網站資料夾時的分享——GitHub 的裝置登入（Polylode 的 OAuth app，只要 `public_repo`；授權以 DPAPI 加密存在使用者資料夾），在使用者自己的帳號建立公開倉庫 `polylode-models`，用 Git Data API 把 `share.write_model` 寫出的檔案提交成一個 commit（有檔案超過 `BLOB_LIMIT` 時改用 `push_with_git`：不含檔案的淺層 fetch、裸索引、推送；GitHub 對太大的上傳回 401，所以 401 先問 `/user` 才放掉授權）；由 `--share-github <模型>` 執行。開啟 Pages 的 API 要 `repo` 權限，程式不要求，改成先看網站有沒有回應、試一次 API，不行就給使用者 Settings → Pages 的連結。
`polylode/endfield/poses.py`：終末地角色的介面姿勢（全部 `ui` clip，遊戲循環播放的排前面；少女前線 2 與鳴潮的 `poses.py` 同樣全部寫、停留的排前面）在匯出時寫進場景文件，`interchange/glb` 寫成 glTF animation，只留線性內插需要的關鍵格（`glb.keys`）；分享的網頁可以切換。
`polylode/extractor/preview.py`（§7.1455）：匯出前的 3D 預覽（只寫 `.glb` 的匯出工作＋Qt Quick 3D）。3D 檢視跑在自己的行程（`view_host.py`，`--view-host`）：Qt Quick 3D 在算圖的執行緒上一次讀完整個模型（21 MB 的模型一格卡 2.4 秒），放在視窗的行程裡會卡住整個視窗；`PreviewPanel` 用 `QWindow.fromWinId` 把它的視窗嵌進來，屬性與視角以每行一個 JSON 指令傳過去。顯示用的複本與模型範圍、檢視器的清單與檔案摘要都在背景執行緒讀（`preview.background`），讀完才交回視窗的執行緒。匯出工具三個來源都不判範圍、寫進自己的倉庫（終末地 `own_store`、GF2 `own_scope`，鳴潮本來就只寫 `assets_cache/wuthering_waves`）。
`polylode/extractor/locate.py`（§7.1454）：從使用者選的資料夾或 Steam 遊戲庫找遊戲資料所在的那一層。
`polylode/endfield/mapbuild.py`（§7.1452）：從遊戲安裝分批建終末地的 CAB／資產對照表（§7.1070 的 dev 腳本搬進套件）。

`polylode/endfield/shader_code.py` 與 `vfx_bake.py`：終末地沒有底色圖的材質（特效 shader）在寫出交換文件後，用遊戲編譯好的 pass 變體（頂點＋像素階段，鳴潮的直譯器）烘成貼圖；主體 shader 的發光照讀出的式子烘成發光圖；烘不了的標成不可見並寫下原因（`docs/tools.md`）。
`polylode/interchange`（§7.1447／§7.1448）：不靠 Blender 的交換檔寫出器，讀所有轉換器交給 Blender 的同一份場景（文件＋陣列）：`glb` 與 `fbx`。
每個 FBX 旁邊另有 `sidecar` 寫的 `<名>.islandlife.json`（IslandLife 讀 FBX 時要的、FBX 裝不下的資料；跨倉庫的介面，有 schema 版本）：
`sources.write_own` 寫完 FBX 就寫它，來源的身分與人形骨表出處由各來源交給它（`extractor/identity.py`），材質的角色著色參數與預覽共用 `character_styles.native_style`。
鳴潮的讀表與組合從未進版控的 dev 腳本搬進 `polylode/wuthering_waves/`：`scope`（角色表 → 女性可玩角色的網格）、
`assembly`（模型表 → 角色選單的顯示模型）、`extract`（跑 `tools/wuwa_cli` 讀取器）、`source_scene`（傾印 → 交換格式，Blender 端是 `tools/wuwa_source_blender.py`）；`converted.names.wuthering_name`
可以指定讀哪一份抽取。讀取器的 C# 原始碼、上游 CUE4Parse 版本與本機修補在 `tools/wuwa_cli/`。

三個遊戲的眼睛程式（IslandLife §7.1491）：`endfield/eye_bake.py`（虹膜的高光貼片與 matcap，烘給 glTF／FBX）、`gf2/eye_bake.py`（疊層與高光貼圖烘進底色，並把加法／乘法層照眼睛 UV 另寫成貼圖）、`wuthering_waves/eye_bake.py`（執行眼睛的 base pass，寫成受光的底色與不受光的發光）；材質帶 `eye`，旁檔照抄，預覽由 `character_styles.eye_style` 讀。

`polylode/wuthering_waves/skin_band.py`：鳴潮 `MSM_ToonCommon` 材質的皮膚帶（實例沒設 `UseSkinMask` 的也讀，旁檔的角色照讀到的定）——在 base pass 的編譯碼上掃 `TypeMask` 與頂點色，量出 `SkinColor` lerp 的係數跟哪個輸入、在哪個門檻跳；係數跟著別的圖走（ID 圖和 `SkinMin`／`SkinMax` 比，IslandLife §7.1502）時讀成那些圖的 8 位元值查表（`docs/tools.md`），寫進交換場景的材質（`skin_band`），旁檔照抄。
`polylode/gf2/scene_facts.py`：GF2 資料在預製物旗標之外說的事——`GunWeaponData` 武器預製物用的材質（旁檔的 `weapon` 角色）、過場預製物的 Timeline 對關著的節點做什麼（旁檔的 `timeline`），由 `gf2.export.source_facts` 在建交換場景時讀。
`polylode/wuthering_waves/shaders/` 與 `shader_bake.py`：鳴潮裁切材質的覆蓋率由遊戲自己的 shader 決定——讀材質實例嵌入的 shader map、執行 preshader、從 shader 封存檔取出陰影深度 pass 的 DXBC，用直譯器逐像素執行後烘進底色圖的 alpha（`docs/tools.md` 有細節）。遊戲檔由 `extractor/wuwa_source.GameFiles` 取出並快取。`wuthering_waves/fur_shells.py` 接在後面，把視差毛髮的高度場做成一層層毛絲幾何。
