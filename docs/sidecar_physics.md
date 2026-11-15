# FBX 旁檔的 `physics` 段（擺動鏈與碰撞體）

（IslandLife §7.1512，2026-11-15。IslandLife §7.1511 的建議 1：轉換角色的物理鏈與碰撞清單從來源遊戲讀。）

`<名>.islandlife.json` 的頂層多一段 `physics`。它是**加上去的欄位**，沒有移動或改義任何既有欄位，所以旁檔仍是
`polylode.islandlife-sidecar` 第 1 版；沒有這一段的舊旁檔，讀者當成 `status: "unread"`。
這一段只寫遊戲資料說的事：哪幾根骨頭是一條擺動鏈、它的參數（遊戲自己的欄位名與單位，原樣）、它和哪些碰撞體碰、
碰撞體掛在哪根骨頭上、什麼形狀。解算器怎麼用這些參數是讀者的事（IslandLife 的 `endfield/cloth.py`、`endfield/kawaii.py`）。

## 頂層

| 欄位 | 內容 |
|---|---|
| `format` | 這一段自己的版本，現在是 `1`；欄位移動或改義就跳號（旁檔的 `version` 不動） |
| `status` | `read`（讀到了，`chains` 可能是空的：遊戲這個模型沒有擺動元件）、`none`（遊戲資料說這個模型沒有執行期物理，`reason` 說根據）、`unread`（這個來源沒有讀，`reason` 說為什麼） |
| `system` | 鏈的解算器家族：`magica_cloth2.bone_cloth`（終末地 `BeyondBoneCloth`）、`kawaii_physics`（鳴潮 `AnimNode_KawaiiPhysicsGroup`），或其他來源自己定的名字（例：GF2 從 clip 反推的參數，由寫它的人在本檔補一節）；`status` 不是 `read` 時是 `null` |
| `basis` | 讀了什麼（類別名、節點型別、資產路徑） |
| `reason` | `status` 不是 `read` 時的原因 |
| `space` | 固定字串，說明下面的座標：見「空間」 |
| `chains` | 擺動鏈（元件／節點），每個一列 |
| `colliders` | 碰撞體，每個一列；鏈用 `id` 引用 |
| `unread` | 讀到但沒有解讀的欄位：`{欄位名: 說明}`（意義不明、開發商改過、讀者用不上的），不猜它的意思 |

## 空間

- **骨頭上的幾何**（碰撞體的端點、法線、鏈的框架）寫在那根骨頭**在 FBX 裡的靜止框架**（FBX 綁定的那個框架，不含縮放）的
  local 座標，**公尺**。和 `expressions.bone_poses` 同一個空間；讀者把它乘上那根骨頭的世界矩陣就到世界。
- **世界方向**（重力方向）寫在 FBX 的世界（Y 朝上、公尺，`model.space`）。
- `params` 裡的值是**遊戲自己的單位與座標慣例**（Unity：左手系、公尺；Unreal：左手系、Z 朝上、公分），原樣；
  換算到上面兩個空間的版本另外寫在 `resolved`。讀者要嘛讀 `resolved`，要嘛自己照遊戲的慣例換算，不能混用。

## `chains[]`

| 欄位 | 內容 |
|---|---|
| `id` | 遊戲裡的身分（終末地：元件的 path id；鳴潮：動畫藍圖節點名） |
| `system` | 同頂層 |
| `frame` | 元件本身的框架：`{"bone": FBX 骨名或 null, "node": 遊戲節點名, "local": 4×4（列優先）}`——元件掛在不是骨頭的節點上時（終末地的 `MC_*` 是根節點的子節點），`bone` 是往上最近的 FBX 骨頭，`null` 表示模型的根（骨架物件本身），`local` 是節點相對它的變換（公尺） |
| `roots` | 根骨（FBX 骨名）；一個節點可以有多個根（鳴潮的 Group 節點） |
| `exclude` | 不模擬的骨頭與它底下的整支（終末地 `ignoreFromRootBones`、鳴潮 `ExcludeBones`） |
| `bones` | 照遊戲的規則展開的鏈骨（根與它底下全部、扣掉 `exclude`），父在子前 |
| `states` | `[{"node": 骨名, "state": "fixed"/"move"/"off"}]` 或 `null`：遊戲有逐骨的設定時（終末地 `selectionData` 的塗刷點，Polylode 在遊戲的座標裡照 `maxConnectionDistance` 配到最近的骨頭）才有；`null` 是系統自己的規則（根固定、其餘移動） |
| `colliders` | 這條鏈測試的碰撞體 `id`，照遊戲的清單順序；遊戲清單裡空的槽（`null`）不列，個數記在 `empty_slots` |
| `params` | 系統自己的參數，原樣（終末地：`serializeData` 去掉 `rootBones`／`ignoreFromRootBones`／`colliderList`／`selectionData` 這些已經解到上面欄位的；節點引用一律寫成 `{"node": FBX 骨名}`。鳴潮：節點的 `PhysicsSettings`、曲線、`DummyBoneLength`、`BoneForwardAxis`、`PlanarConstraint`、`Gravity`、`TeleportDistanceThreshold`／`TeleportRotationThreshold`、`TargetFramerate`、`Alpha` 與它的輸入方式…） |
| `resolved` | `params` 換到上面空間的值（見各系統） |

## `colliders[]`

| 欄位 | 內容 |
|---|---|
| `id` | 遊戲裡的身分（終末地：元件 path id；鳴潮：`<節點>/<Spherical|Capsule|Planar>/<序號>`） |
| `kind` | `sphere`、`capsule`、`plane` |
| `bone` | 掛在哪根 FBX 骨頭（遊戲的宿主節點；不是骨頭時往上最近的骨頭，幾何照樣換到它的框架） |
| `start`、`end` | 骨頭 local 的兩端（公尺）；球兩端相同；平面的 `start` 是平面上一點、`end` 是單位法線（碰撞的東西留在法線那一側） |
| `radius` | `[起端, 末端]`（公尺，已乘上遊戲節點的縮放）；平面是 `[0, 0]` |
| `inside` | `true`：把質點留在球**裡面**（KawaiiPhysics 的 `Inner`）；其餘 `false` |
| `host` | `{"humanoid": Unity HumanBodyBones 名或 null, "on_chain": 布林}`：宿主骨往上（含自己）最近的人形骨；`on_chain` 是宿主骨本身就是某條鏈模擬的骨頭（外層衣服掛的碰撞體，§7.1511「衣服壓衣服」） |
| `source` | `{"class": 遊戲的類別或結構名, "node": 遊戲的宿主節點名, "fields": 原樣的欄位}` |

`host` 讓讀者知道一個碰撞體代表的是身體的哪一段：IslandLife 的轉換角色穿的是借來的身體，遊戲的身體碰撞體大小只合遊戲的身體，
所以它用 `host.humanoid` 決定「這條鏈要和哪幾段身體碰」，半徑量自己的身體（§7.1512）。

## 各系統

### `magica_cloth2.bone_cloth`（終末地）

- 讀：預製物上 `BeyondBoneCloth` 與 `BeyondBoneCapsuleCollider`／`SphereCollider`／`PlaneCollider`、舊式 `MagicaCapsuleCollider`／`SphereCollider`／`PlaneCollider`（`endfield.components.KEPT`，IslandLife §7.564）。
- `params` 是 `serializeData`（Unity 慣例：向量是 Unity 世界、左手系；讀者鏡像 x 到右手系）。
- `resolved`：`gravity_world`（`gravity × gravityDirection`，FBX 世界，m/s²）。
- 碰撞體的形狀照 IslandLife `endfield.cloth.collider` 的讀法（膠囊 `size` 是 x 起端半徑、y 末端半徑（`radiusSeparation`）、z 全長含兩端半球，
  `direction` 軸、`reverseDirection`、`alignedOnCenter`；平面是宿主的 +Y），換到宿主骨的 FBX 框架。

### `kawaii_physics`（鳴潮）

- 讀：模型表的動畫藍圖欄（`assembly.MODEL_TABLE` 的 `动画蓝图_*`）指的 `ABP_<模型>_PC` 的 CDO 上每個 `AnimGraphNode_KawaiiPhysicsGroup*`
  （外掛 KawaiiPhysics 1.6.1，開發商改過：一個節點多個 `RootBones`、`PhysicsSettings.Shape`／`HalfWidth`、`MultFix*`、`BindInfos`、`KuroLODMask`、`IgnoreMechanicsLevel`…）。
- `params` 原樣（公分、Unreal 慣例）。
- `resolved`：`radius_m`（`PhysicsSettings.Radius`／100）、`dummy`：`{"length_m", "forward": [x,y,z]}`（`BoneForwardAxis` 換到 FBX 骨頭 local 的單位向量）、
  `gravity_world`（`Gravity`／100，FBX 世界，m/s²）、`teleport_distance_m`、`alpha`（`AlphaInputType` 是 Float 時就是 `Alpha × AlphaScaleBias`；
  是曲線時 `null`，曲線名在 `params`）、曲線（`DampingCurve`… 引用的 `CurveFloat` 資產讀到的鍵，讀不到的寫 `null` 加原因）。
- 碰撞體：`SphericalLimits`、`CapsuleLimits`、`PlanarLimits`（與 `*LimitsData`、`LimitsDataAsset` 的）照 1.6.1 的 `UpdateCapsuleLimits` 組：
  宿主骨的 component-space 變換換到 `BCS_BoneSpace`（相對宿主骨自己，所以是單位變換），旋轉前乘 `OffsetRotation`、平移加 `OffsetLocation`，再換回 component space——
  碰撞體跟著宿主骨走，偏移在宿主骨的框架裡。膠囊兩端是中心 ± 旋轉的 Z 軸 × `Length`/2、半徑 `Radius`；平面法線是旋轉的上方向（Z）。
  Polylode 在參考姿勢算出來，換到宿主骨的 FBX 框架寫進 `start`／`end`；`source.fields` 保留原值。
  （核對：椿的裙節點兩個大腿膠囊的中心在髖關節下方 11 公分、沿大腿；若把偏移當成父骨框架，中心會跑到髖關節上方 11 公分。）

### 沒有讀的來源

- GF2：`status: "unread"`，`reason` 說明（遊戲的戰鬥預製物沒有物理元件，擺動在每套衣裝的 clip 裡，IslandLife §7.1511）。
  從 clip 反推的參數由寫它的人定 `system` 名稱並在本檔加一節；格式的其他欄位照上面。
