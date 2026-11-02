# Polylode

[English](../../README.md) · [繁體中文](README.zh-TW.md) · **简体中文** · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md)

Polylode 从你安装的游戏里把角色模型挖出来——从矿脉（*lode*）里挖出多边形（*poly*gon）——写成带贴图、带骨架、到处都能打开的文件：
glTF（`.glb`）、FBX 与 Blender（`.blend`）。它是 Windows 桌面程序，有窗口、能用鼠标转动的预览，以及模型查看器。

| 游戏 | 读取什么 | 说明 |
|---|---|---|
| 明日方舟：终末地 | 每个可玩角色的预制体 | 第一次扫描会从你的安装建立资产对照表（分批核对、可中断续跑），游戏更新后重建 |
| 少女前线 2：追放 | 每个可玩角色的各套服装 | 只列出有自己蒙皮网格与完整人形骨架的根节点（不含宠物、道具与空的变体） |
| 鸣潮 | 每个可玩角色的显示模型 | 需要由 `tools/wuwa_cli` 构建的游戏文件读取器与你的 AES 密钥文件 |
| 既有转出物 | `*_Converted` 文件夹里做好的模型 | 逐个核对 SHA-256 后复制出来 |

## 功能

- **扫描、勾选、导出。** 每个游戏一页：扫描安装、搜索、勾选模型，排进队列导出。每次扫描与导出都在自己的进程里运行，转换器出错不会拖垮窗口。
- **自动找安装。** 扫描前会列出在 Steam 游戏库找到的安装；选游戏文件夹、它的上一层或文件夹里面都可以。
- **三种格式，其中两种不需要 Blender。** `.glb` 与 FBX 由 Polylode 自己写（骨架、蒙皮网格、形状键、内嵌贴图、透明）；`.blend` 需要 Blender。
  设置了 Blender 时，每个文件都会在 Blender 读回核对。
- **材质按游戏的画法。** 裁切、半透明、眼影叠层与眼睛高光层都按从各游戏 shader 读出的规则写出；glTF／FBX 画不出的叠层，效果会烘进底下的贴图，
  `.blend` 保留真正的分层。
- **导出前预览。** 选一行按「预览」（或双击）：模型转成 `.glb` 放进缓存、显示在列表旁——左键拖动旋转、右键平移、滚轮缩放。
- **模型查看器。** 列出输出文件夹里的所有模型，也能打开任何 `.glb`、`.gltf`、`.fbx`、`.obj`（拖进来、选文件，或用 `Polylode.exe <文件>`，
  可以设成「打开方式」的程序）：正面、背面、侧面、上方视角、线框、深浅背景、灯光强度，以及 glTF 的内容。列表里的模型可以删到回收站。
- **五种语言。** English、繁體中文、简体中文、日本語、한국어；默认跟随系统，可在「设置」切换，立即生效。

## 下载

到 [Releases](https://github.com/JE-Chen/Polylode/releases) 下载最新的 `Polylode-<版本>-windows-x64.zip`，解压到任何地方，运行 `Polylode.exe`。
需要的东西都已附上：AnimeStudio（终末地用）、.NET 10 运行环境、glTF 导入器，以及给没有显卡驱动的电脑用的软件 OpenGL。

需求：Windows 10 或 11，64 位。只有 `.blend` 需要 Blender 4.x 或 5.x。

## 使用方式

1. **设置**——指定输出文件夹与导出格式。安装文件夹可以先留空：扫描时会询问（并列出在 Steam 找到的）。
2. **游戏页**——「扫描安装」。终末地第一次扫描会建立资产对照表（几十分钟、可续跑）；GF2 第一次扫描会建立 bundle 索引。之后的扫描在游戏更新前都沿用。
3. 勾选模型，点「导出勾选的模型」，在下方看队列与记录。
4. **模型查看器**——看导出的成果、比对、删除。

设置与缓存放在 `%LOCALAPPDATA%\Polylode`。如果你在它还是 IslandLife 一部分时用过导出工具，第一次启动会把它的设置、终末地资产对照表、
GF2 索引与仓库搬过来，不用重建。

## 从源码构建

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m polylode.extractor          # 从源码打开窗口
.venv\Scripts\python -m pytest tests                # 测试
.venv\Scripts\python build.py --dotnet <dotnet 文件夹>   # dist\Polylode\Polylode.exe，接着运行它的自检
```

构建用 Nuitka 编译（需要 C 编译器：Visual Studio 的），把 AnimeStudio、.NET 10 运行环境（`--dotnet`，默认 `DOTNET_ROOT`）与软件 OpenGL
放到程序旁，再运行 `Polylode.exe --selftest`：它会加载每个转换器、预览的 QML 与一个真的 `.glb`，缺任何一样构建就失败。

## CI/CD

- **CI**（`.github/workflows/ci.yml`）——每次推送与 PR 到 `main`、`dev`：在 Windows 上运行代码检查、测试与自检。
- **Release**（`.github/workflows/release.yml`）——每次推送到 `main`：测试、Nuitka 构建与打包版自检，并把压缩好的程序发布成 GitHub Release。

开发在 `dev`，`main` 是发布的版本。

## 结构

```
polylode/extractor/      窗口、扫描、任务、预览、查看器、语言
polylode/endfield/       明日方舟：终末地——资产对照表、转换、FBX 场景
polylode/gf2/            少女前线 2——bundle、可玩名册、转换、眼睛叠层烘焙
polylode/wuthering_waves/  鸣潮——角色表、显示模型组合
polylode/interchange/    glTF 与 FBX 写出器（不需要 Blender）
tools/                   Blender 脚本、图标、FBX SDK 检查、鸣潮读取器源码、AnimeStudio
tests/                   pytest
build.py, main.py        打包的程序
```

## 许可与内容

Polylode 的代码采用 GNU LGPL 2.1（`LICENSE`）；附带的 AnimeStudio 是 MIT（`tools/AnimeStudio/LICENSE`）。Polylode 不含任何游戏内容：
它转换的是你自己电脑上安装的文件，写出的东西也留在你的电脑上。只用在你拥有的游戏上，并遵守它们的条款。
