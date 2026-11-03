# Polylode 的架構

（從 IslandLife 的 `docs/architecture.md` 搬來，2026-11-02；§7.x 是 IslandLife 的逐節紀錄。）

`polylode/extractor`（§7.1446）：獨立的 PySide6 匯出工具，不屬於遊戲執行期，也不進遊戲的發行物（自己的 `build.py`）。
`sources.Source` 是擴充點：每個來源給 `paths`（使用者要指定的安裝／工具）、`scan`、`export`；`app` 只認這個介面，掃描與匯出由 `jobs`
在另一個行程跑同一支程式。GF2 的「宣告的根節點」與「轉一個根節點」從 `tools/import_gf2.py` 搬進 `gf2.batch`，命令列與工具共用。
終末地的 FBX 寫出與讀回核對從 `tools/ef_export_fbx.py` 搬進 `endfield.fbx_export`（工具只剩命令列）；匯出工具轉遊戲原本的模型不判範圍，
轉進自己的倉庫（`endfield_source.own_store`，§7.1455），遊戲讀的倉庫看不到。`Pipeline.run` 的 `admit` 仍是在 Avatar 讀到後、材質讀取前
丟掉一個預製物的擴充點；`body_scope.MEN`（`hrole.MEN` 指向它）是 Free H 的男角。
`polylode/extractor/viewer.py`（§7.1455）：匯出工具的模型檢視器頁（任何模型檔、固定視角、線框、背景、燈光、glTF 內容）。
`polylode/extractor/preview.py`（§7.1455）：匯出前的 3D 預覽（只寫 `.glb` 的匯出工作＋Qt Quick 3D）。匯出工具三個來源都不判範圍、寫進自己的倉庫（終末地 `own_store`、GF2 `own_scope`，鳴潮本來就只寫 `assets_cache/wuthering_waves`）。
`polylode/extractor/locate.py`（§7.1454）：從使用者選的資料夾或 Steam 遊戲庫找遊戲資料所在的那一層。
`polylode/endfield/mapbuild.py`（§7.1452）：從遊戲安裝分批建終末地的 CAB／資產對照表（§7.1070 的 dev 腳本搬進套件）。
`polylode/interchange`（§7.1447／§7.1448）：不靠 Blender 的交換檔寫出器，讀所有轉換器交給 Blender 的同一份場景（文件＋陣列）：`glb` 與 `fbx`。
鳴潮的讀表與組合從未進版控的 dev 腳本搬進 `polylode/wuthering_waves/`：`scope`（角色表 → 女性可玩角色的網格）、
`assembly`（模型表 → 角色選單的顯示模型）、`extract`（跑 `tools/wuwa_cli` 讀取器）、`source_scene`（傾印 → 交換格式，Blender 端是 `tools/wuwa_source_blender.py`）；`converted.names.wuthering_name`
可以指定讀哪一份抽取。讀取器的 C# 原始碼、上游 CUE4Parse 版本與本機修補在 `tools/wuwa_cli/`。

`polylode/wuthering_waves/shaders/` 與 `shader_bake.py`：鳴潮裁切材質的覆蓋率由遊戲自己的 shader 決定——讀材質實例嵌入的 shader map、執行 preshader、從 shader 封存檔取出陰影深度 pass 的 DXBC，用直譯器逐像素執行後烘進底色圖的 alpha（`docs/tools.md` 有細節）。遊戲檔由 `extractor/wuwa_source.GameFiles` 取出並快取。`wuthering_waves/fur_shells.py` 接在後面，把視差毛髮的高度場做成一層層毛絲幾何。
