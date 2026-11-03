# 待辦

只記還沒做完的事，做完的整段刪掉。§7.x 是 IslandLife 的逐節紀錄（`D:\Codes\IslandLife\docs\updates\`）。

- **FBX 的貼圖平移方向**（§7.1449）：自寫的 FBX 已經用 Autodesk FBX SDK 2020.3.11 讀回過，終末地 jsspsi／ardelia、GF2 Groza、
  鳴潮 Rebecca 逐項相符，但還沒有在 Unity 或 Maya 的畫面上打開過。
  - 底色圖的縮放與偏移照 Unity 的 tiling／offset，寫進 `Texture` 的 `Scaling`／`Translation`。
  - Autodesk 文件沒寫平移的正負方向：three.js 讀成 `uv×縮放+偏移`（跟 Unity 一樣），Blender 的 FBX 匯入器讀成 `(uv−偏移)×縮放`。
  - 確認方法：在 Unity 開 ardelia 的毛皮材質（`M_actor_ardelia_fur_02`，縮放 5.99×5.22、偏移 1.97／0.12），看貼圖位置對不對。
- **預覽與模型檢視器只在離屏驗證過**（§7.1455）：`--selftest` 用 `QQuickRenderControl` 在離屏載入 `.glb`。真正的視窗裡，
  旋轉、平移、縮放、視角按鈕和刪除都還沒有實際操作過。
  - 自動取景（`preview.model_bounds`）用離屏算圖量過：GF2 Lenna、終末地 Zhuangfy、鳴潮 R2T1ShouanrenMd10011
    各佔畫面 70–80%，置中。
- **別的工具寫的 FBX 的取景**：`preview.fbx_bounds` 只讀 `Vertices` 與 `Lcl`。單位已經量過：Qt 不套用
  `UnitScaleFactor`，100 和 1 都畫在原始外框。
  - 還沒量過的：Z 軸朝上的 FBX（`UpAxis` 2），以及帶樞軸（pivot）的節點。
  - FBX 7.5 的 64 位元紀錄讀不了，那種檔案會顯示「量不到這個檔案的大小」。
  - 確認方法：拿 Blender、Maya 各匯出一個 Z 軸朝上的 FBX，開在檢視器裡，看是否置中。
- **鳴潮讀取器的打包版還沒有實際讀過遊戲檔**：讀取器在 GitHub 的 runner 上從原始碼建出（CI 的 `wuwa-reader`）；
  本機用同一份原始碼建的 v8 版已經跑通預覽（R2T1ShouanrenMd10011）。
  - 確認方法：Release 的 exe 留空讀取器欄位，預覽一個鳴潮模型。

- **鳴潮本機 AES 自動取得的版本支援**：目前安裝的遊戲程式未找到可驗證的候選金鑰；需研究該版本的程式封裝後再擴充。
- **鳴潮裁切材質的覆蓋率，只烘了不依賴執行期光照的那些**（`wuthering_waves/shader_bake.py`）：
  - 像「心」`R2T1ShenXinMd10011` 的尾巴與衣物、`R2T1XinMd10011` 的下身，它們的深度 shader 讀了 View 矩陣、`DiffuseOverrideParameter`、
    材質參數集合，覆蓋率取決於受光顏色的亮度。這些照材質宣告的裁切輸出。
  - 要烘它們，得先決定一組「標準光照」（並且要能說出依據），再讓直譯器給這些欄位真實的值。
  - 視差毛髮（`HeightMap` 逐步取樣）只取正面、沒有位移的那一層，轉到側面不會有毛絲的立體感，這是 glTF／FBX 的限制。
- **烘焙速度**：一個 1024² 的毛髮材質約 1～2 分鐘（10 個抖動位置），結果有快取。直譯器可以再加速（暫存器改成分量優先的排列、減少複製）。
