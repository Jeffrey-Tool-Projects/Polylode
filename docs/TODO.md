# 待辦

只記還沒做完的事，做完的整段刪掉。§7.x 是 IslandLife 的逐節紀錄（`D:\Codes\IslandLife\docs\updates\`）。

- **鳴潮用 ID 圖選皮膚的材質讀不出皮膚帶**（IslandLife §7.1492）：約 110 個 `MSM_ToonCommon` 材質（shader map 沒有 `UseSkinMask_DYN`、
  材質有 `SkinMin`／`SkinMax`：PlayerFemale 10011／20011、Lucy、Aimisi、Suoming、Luosela、Mone、Linnaea、Qingxiao…）的 lerp 係數跟著一張貼圖
  解碼出的整數 ID 與 `SkinMin`／`SkinMax` 比；`skin_band` 掃到它跟著別的圖走就寫 `null`，旁檔角色 `unknown`，這些角色在 IslandLife 轉不了。
  - 要讀出那張 ID 圖（哪張、哪個通道、怎麼解碼成整數、比哪個範圍都在 base pass 裡），寫出「哪些 ID 是皮膚」，旁檔像 `TypeMask` 一樣帶上那張圖。
  - 另有 4 個 lerp 寫法讀不懂（Fuludelisi 10011 `Cloth`／`Alpha`、Shouanren 10011 `Up02`、Younuo 10011 `Alpha`），
    Xuanling 10011 `Down` 讀材質時索引越界。
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
- **眼睛烘焙是正面看的一個畫面**（§7.1491）：glTF／FBX 帶的是正面、燈 1 的結果。
  - GF2 `eye` 的視差偏移（頂點程式的輸出 × `_CorneaParallax`／`_SpecularParallax`）沒有讀，`_Specularmap` 照原 UV 取樣。
  - 鳴潮眼睛 base pass 的 `EM` 視差在正面看是 0；側看時虹膜的位移只有預覽與 IslandLife 以外的讀者會缺。
  - 終末地 matcap 用「相機的右、上沿著 u、v」的約定；左右眼 UV 若是鏡像，烘出的 matcap 會左右相反，還沒有對遊戲畫面比過。
- **鳴潮覆蓋率烘焙的貼圖一律 wrap**（§7.1491 發現）：`shader_bake` 的直譯器不看貼圖資產的 `AddressX`／`AddressY`；
  眼睛烘焙已改照資產讀（`TA_Clamp` 的 `HeightLightMap` 用 wrap 會鋪滿整顆眼睛），裁切材質的覆蓋率還沒有改、也還沒有量過有沒有 clamp 的貼圖受影響。

