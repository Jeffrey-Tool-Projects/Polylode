# 待辦

只記還沒做完的事，做完的整段刪掉。§7.x 是 IslandLife 的逐節紀錄（`D:\Codes\IslandLife\docs\updates\`）。

- **FBX 的貼圖平移方向**（§7.1449）：自寫的 FBX 已經用 Autodesk FBX SDK 2020.3.11 讀回過，終末地 jsspsi／ardelia、GF2 Groza、
  鳴潮 Rebecca 逐項相符，但還沒有在 Unity 或 Maya 的畫面上打開過。
  - 底色圖的縮放與偏移照 Unity 的 tiling／offset，寫進 `Texture` 的 `Scaling`／`Translation`。
  - Autodesk 文件沒寫平移的正負方向：three.js 讀成 `uv×縮放+偏移`（跟 Unity 一樣），Blender 的 FBX 匯入器讀成 `(uv−偏移)×縮放`。
  - 確認方法：在 Unity 開 ardelia 的毛皮材質（`M_actor_ardelia_fur_02`，縮放 5.99×5.22、偏移 1.97／0.12），看貼圖位置對不對。
- **預覽與模型檢視器只在離屏驗證過**（§7.1455）：`--selftest` 用 `QQuickRenderControl` 在離屏載入 `.glb`。真正的視窗裡，
  旋轉、平移、縮放、視角按鈕和刪除都還沒有實際操作過。
- **CI 的 selftest 在 GitHub 的 Windows runner 上靠軟體 OpenGL 跑**（`QT_OPENGL=software`，用 windows 平台）：第一次跑通後刪掉這一條；
  如果跑不通，就改成只在 Release 的打包版自檢裡載入 `.glb`。
