# Polylode

[English](../../README.md) · **繁體中文** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md)

Polylode 從你安裝的遊戲裡把角色模型挖出來——從礦脈（*lode*）裡挖出多邊形（*poly*gon）——寫成帶貼圖、帶骨架、到處都能開的檔案：
glTF（`.glb`）、FBX 與 Blender（`.blend`）。它是 Windows 桌面程式，有視窗、能用滑鼠轉動的預覽，以及模型檢視器。

| 遊戲 | 讀什麼 | 說明 |
|---|---|---|
| 明日方舟：終末地 | 每個可玩角色的預製物 | 第一次掃描會從你的安裝建資產對照表（分批核對、可中斷續跑），遊戲更新後重建 |
| 少女前線 2：追放 | 每個可玩角色的各套衣裝 | 只列出有自己蒙皮網格與完整人形骨架的根節點（不含寵物、道具與空的變體） |
| 鳴潮 | 每個可玩角色的顯示模型 | 需要由 `tools/wuwa_cli` 建出的遊戲檔讀取器與你的 AES 金鑰檔 |
| 既有轉出物 | `*_Converted` 資料夾裡做好的模型 | 逐檔核對 SHA-256 後複製出來 |

## 功能

- **掃描、勾選、匯出。** 每個遊戲一頁：掃描安裝、搜尋、勾選模型，排進佇列匯出。每次掃描與匯出都在自己的行程裡跑，轉換器出錯不會拖垮視窗。
- **自動找安裝。** 掃描前會列出在 Steam 遊戲庫找到的安裝；選遊戲資料夾、它的上一層或資料夾裡面都可以。
- **三種格式，其中兩種不需要 Blender。** `.glb` 與 FBX 由 Polylode 自己寫（骨架、蒙皮網格、形狀鍵、內嵌貼圖、透明）；`.blend` 需要 Blender。
  設定了 Blender 時，每個檔案都會在 Blender 讀回核對。
- **材質照遊戲的畫法。** 裁切、半透明、眼影疊層與眼睛高光層都照從各遊戲 shader 讀出的規則寫出；glTF／FBX 畫不出的疊層，效果會烘進底下的貼圖，
  `.blend` 保留真正的分層。
- **匯出前預覽。** 選一列按「預覽」（或雙擊）：模型轉成 `.glb` 放進快取、顯示在清單旁——左鍵拖曳旋轉、右鍵平移、滾輪縮放。
- **模型檢視器。** 列出輸出資料夾裡的所有模型，也能開任何 `.glb`、`.gltf`、`.fbx`、`.obj`（拖進來、選檔，或用 `Polylode.exe <檔案>`，
  可以設成「開啟檔案」的程式）：正面、背面、側面、上方視角、線框、深淺背景、燈光強度，以及 glTF 的內容。清單裡的模型可以刪到資源回收筒。
- **五種語言。** English、繁體中文、简体中文、日本語、한국어；預設跟隨系統，可在「設定」切換，立即生效。

## 下載

到 [Releases](https://github.com/JE-Chen/Polylode/releases) 下載最新的 `Polylode-<版本>-windows-x64.zip`，解壓到任何地方，執行 `Polylode.exe`。
需要的東西都已經附上：AnimeStudio（終末地用）、.NET 10 執行環境、glTF 匯入器，以及給沒有顯示卡驅動的電腦用的軟體 OpenGL。

需求：Windows 10 或 11，64 位元。只有 `.blend` 需要 Blender 4.x 或 5.x。

## 使用方式

1. **設定**——指定輸出資料夾與匯出格式。安裝資料夾可以先留空：掃描時會詢問（並列出在 Steam 找到的）。
2. **遊戲頁**——「掃描安裝」。終末地第一次掃描會建資產對照表（幾十分鐘、可續跑）；GF2 第一次掃描會建 bundle 索引。之後的掃描在遊戲更新前都沿用。
3. 勾選模型，按「匯出勾選的模型」，在下方看佇列與記錄。
4. **模型檢視器**——看匯出的成果、比對、刪除。

設定與快取放在 `%LOCALAPPDATA%\Polylode`。如果你在它還是 IslandLife 一部分時用過匯出工具，第一次啟動會把它的設定、終末地資產對照表、
GF2 索引與倉庫搬過來，不用重建。

## 從原始碼建置

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m polylode.extractor          # 從原始碼開視窗
.venv\Scripts\python -m pytest tests                # 測試
.venv\Scripts\python build.py --dotnet <dotnet 資料夾>   # dist\Polylode\Polylode.exe，接著跑它的自檢
```

建置用 Nuitka 編譯（需要 C 編譯器：Visual Studio 的），把 AnimeStudio、.NET 10 執行環境（`--dotnet`，預設 `DOTNET_ROOT`）與軟體 OpenGL
放到程式旁，再執行 `Polylode.exe --selftest`：它會載入每個轉換器、預覽的 QML 與一個真的 `.glb`，少了任何一樣建置就失敗。

## CI/CD

- **CI**（`.github/workflows/ci.yml`）——每次推送與 PR 到 `main`、`dev`：在 Windows 上跑程式碼檢查、測試與自檢。
- **Release**（`.github/workflows/release.yml`）——每次推送到 `main`：測試、Nuitka 建置與打包版自檢，並把壓縮好的程式發佈成 GitHub Release。

開發在 `dev`，`main` 是發佈的版本。

## 結構

```
polylode/extractor/      視窗、掃描、工作、預覽、檢視器、語言
polylode/endfield/       明日方舟：終末地——資產對照表、轉換、FBX 場景
polylode/gf2/            少女前線 2——bundle、可玩名冊、轉換、眼睛疊層烘焙
polylode/wuthering_waves/  鳴潮——角色表、顯示模型組合
polylode/interchange/    glTF 與 FBX 寫出器（不需要 Blender）
tools/                   Blender 腳本、圖示、FBX SDK 檢查、鳴潮讀取器原始碼、AnimeStudio
tests/                   pytest
build.py, main.py        打包的程式
```

## 授權與內容

Polylode 的程式碼採 GNU LGPL 2.1（`LICENSE`）；附帶的 AnimeStudio 是 MIT（`tools/AnimeStudio/LICENSE`）。Polylode 不含任何遊戲內容：
它轉換的是你自己電腦上安裝的檔案，寫出的東西也留在你的電腦上。只用在你擁有的遊戲上，並遵守它們的條款。
