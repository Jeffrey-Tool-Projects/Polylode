# FBX 旁檔的 `rbf` 段（姿勢修正：RBF 驅動與 PoseDriver）

（IslandLife §7.1514，2026-11-16。IslandLife §7.1511 的建議 3：鳴潮的肩部修正形狀鍵與頭轉髮束。）

`<名>.islandlife.json` 的頂層多一段 `rbf`。它是**加上去的欄位**，沒有移動或改義任何既有欄位，旁檔仍是
`polylode.islandlife-sidecar` 第 1 版；沒有這一段的舊旁檔，讀者當成 `status: "unread"`（`interchange.sidecar.UNREAD_RBF`）。
現在只有鳴潮寫它（`wuthering_waves/rbf.py`，匯出時在 `wuwa_source._rbf` 讀）；終末地與 GF2 寫 `unread`。

## 頂層

| 欄位 | 內容 |
|---|---|
| `format` | 這一段自己的版本，現在 `1` |
| `status` | `read` 或 `unread`（`reason` 說為什麼） |
| `space` | 座標說明（下面「空間」） |
| `basis` | 讀了哪些資產（骨架、預算表、動畫藍圖） |
| `drivers` | 骨架的 `RBFDriver`，每個一列 |
| `pose_drivers` | 動畫藍圖的 `AnimNode_PoseDriver`，每個一列 |

## 空間

- `*_fbx_m`、`direction_fbx`：在**基準骨的 FBX 靜止框架**（不含縮放）裡的向量，公尺。
- `quat_fbx`：骨頭自己的 FBX 靜止框架裡的旋轉（x, y, z, w）。
- `game_*`、`quat_game`、`rotation_game`：Unreal 的原值（左手系、公分、`FRotator` 度），原樣。
- 換算：向量 `FBX = diag(1, −1, 1) · game / 100`；四元數 `(x, y, z, w) → (−x, y, −z, w)`。根據：交換場景的骨頭框架是遊戲的鏡像
  `C·R·C`（`source_scene.native_world`），每列的 `rest_translation.base_frame_gap_deg` 是實際量到的差（鳴潮 12 個驅動都是 0）。

## `drivers[]`（`RBFDriver`，系統名 `kuro_rbf_driver`）

開發商自己的類別，掛在 `<模型>_Skeleton` 的 `RBFDrivers`。類別本身**沒有**核函數、半徑或正規化的欄位；
骨架的 `Precomputations` 指向每個體型一份的 `RBFPrecomputations`（`<體型>/RBFConfig/<體型>_Skeleton_RBFPrecomputation`），
裡面照驅動名存著解好的權重表。

| 欄位 | 內容 |
|---|---|
| `id`、`object` | `DriverName`、物件名 |
| `driver_bone`、`base_bone` | 驅動骨與它掛的基準骨（必須是父子） |
| `measure` | 讀的量：`TranslationX/Y/Z`。**驅動骨在基準骨靜止框架裡的位移，隨基準骨的轉動而轉**：`v = turn(基準骨) · 驅動骨的靜止位移`，只有方向會變 |
| `rest_translation` | 第一個 target（遊戲的值）、換到 FBX 的值、交換場景量到的驅動骨靜止位移、兩者差（`gap_cm`）、基準骨框架差（`base_frame_gap_deg`） |
| `outputs` | 驅動的形狀鍵名與帶它們的部件（`parts`；FBX 的 BlendShape 已有這些鍵） |
| `targets` | 每個 target 的輸入（`game_cm`、`fbx_m`、單位方向 `direction_fbx`）與每個形狀鍵的權重 |
| `weight_modifier` | `WeightModifier` 原樣（`source_weight`、`target_driver_index`／`target_driver`、`target_weight`、`factor_curve` 的鍵）；`semantics: null`——類別怎麼用它在遊戲程式碼裡，不是資料 |
| `solver` | 量到的解法（下面）與它的參數 |
| `table` | 遊戲的預算表（下面）；沒有時 `status: null` 加原因 |
| `problems` | 靜止位移或框架對不上時的說明（有就不該相信這一列的 FBX 換算） |

### 預算表 `table`

`weights[i][j][k]`：第 `k` 個形狀鍵在格點 `(i, j)` 的權重。格點 `(i, j)` 是長度 `radius_cm` 的輸入，方位角
`atan2(y, x) = −limit_xy_deg + i·interval_deg`、仰角 `asin(z / r) = −limit_z_deg + j·interval_deg`，**在遊戲的基準骨框架**
（`y_game = −y_fbx`）。遊戲存的 `LookupAngleXY`／`LookupAngleZ` 是這些角度量回來的值（差 0.03° 以內），`WeightLookupTable`
是 `(i, j) → i·nz + j`（讀的時候檢查，不是就拒絕），`RotationLimitX` 是 0（沒有扭轉軸，不是 0 就拒絕）。

`max_error` 是 `solver` 在每個格點算出的值和表的最大差；`matches_targets` 是它小於 1e-5。

### 解法 `solver`（量出來的，不是讀到的）

在 3 個體型（FemaleM、FemaleMS、FemaleXL）的 12 個驅動上，下面的式子在全部格點重現遊戲的表到 6e-7：

1. 每個屬性除以它在 targets 上那一欄的 L2 範數（範數小於 `zero_column_below_cm` = 1e-3 cm 的欄除以 1：FemaleM 鎖骨驅動的 Y 欄是
   9.7e-6／4.7e-6 cm 的浮點雜訊，除以它自己就對不上）；
2. 距離是除過以後的歐氏距離，`width` = 完整 N×N 距離矩陣（含對角線的 0）的平均；
3. 核函數 `exp(−d / width²)`，插值型（係數解 targets 上的線性方程，targets 上權重等於表值）；
4. 輸出夾到 [0, 1]（不夾差 2.6e-2；Gaussian、逆多二次或正規化的加法型差 0.1–0.9）。

例外：FemaleMS 的 `weightClavicle_R`（Katixiya）表的峰在方位角 −8°，和 Katixiya 自己的 targets 差 0.79——那一列 `matches_targets: false`。
遊戲執行期用表還是自己解，資料沒說；讀者要選一個並說明。

## `pose_drivers[]`（`AnimNode_PoseDriver`，系統名 `unreal_pose_driver`）

模型的動畫藍圖 `ABP_<模型>_PC`（模型表的動畫藍圖欄，`kawaii.BLUEPRINT_COLUMN`）類別預設物件上的節點。

| 欄位 | 內容 |
|---|---|
| `source_bones` | `SourceBones`（鳴潮都是 `Bip001Head`） |
| `input` | 輸入的定義：**每根來源骨相對參考姿勢、在它自己框架裡的轉動**（`ref⁻¹ · posed`）。根據：每個 target 的旋轉都等於姿勢資產裡那根骨的加法旋轉 `A` 換到骨頭自己框架的 `ref⁻¹·A·ref`（椿 6 個動到的 target 全在 1e-5° 內；`A` 本身在轉軸不是參考姿勢自己的軸時差 3.3°）。`offset_type` 是開發商的 `KuroCachedBone` |
| `rbf_params` | `FRBFParams` 原樣（`SolverType`、`Radius`、`Function`、`DistanceMethod`、`NormalizeMethod`、`WeightThreshold`…） |
| `targets` | 每個 target：驅動的姿勢名 `pose`、`scale`（`TargetScale`）、`rotation_game`（`FRotator`）、`quat_game`（`FRotator::Quaternion`）、`quat_fbx`、各自的距離／函數設定、自訂曲線 |
| `only_drive_selected`、`driven_bones` | `bOnlyDriveSelectedBones` 與 `OnlyDriveBones` |
| `pose_asset` | `PoseAsset` 的 `additive` 與每個姿勢裡被驅動骨（與來源骨）的加法變換：`quat_game`／`quat_fbx`、`translation_game_cm`／`translation_fbx_m`、`scale_game`；加法照 Unreal 的 `posed = A · reference`（父側） |

RBF 的核函數式子（`FRBFSolver`）在引擎程式碼裡，旁檔只寫參數；讀者照 Unreal 的定義實作並說明根據。
