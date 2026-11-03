# Polylode

[English](../../README.md) · [繁體中文](README.zh-TW.md) · **简体中文** · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md)

**分步使用教程（英文／繁体中文）：** [https://je-chen.github.io/Polylode/](https://je-chen.github.io/Polylode/)

Polylode 从你安装的游戏里把角色模型挖出来——从矿脉（*lode*）里挖出多边形（*poly*gon）——写成带贴图、带骨架、到处都能打开的文件：
glTF（`.glb`）、FBX 与 Blender（`.blend`）。它是 Windows 桌面程序，有窗口、能用鼠标转动的预览，以及模型查看器。

| 游戏 | 读取什么 | 说明 |
|---|---|---|
| 明日方舟：终末地 | 每个可玩角色的预制体 | 第一次扫描会从你的安装建立资产对照表（分批核对、可中断续跑），游戏更新后重建 |
| 少女前线 2：追放 | 每个可玩角色的各套服装 | 只列出有自己蒙皮网格与完整人形骨架的根节点（不含宠物、道具与空的变体）。有可替换部件的服装按游戏默认的穿法显示，其他版本与较低细节层级留在 FBX 与 .blend 里、设为隐藏（.glb 只放显示的部分：Blender 与多数查看器不理会 glTF 的隐藏标记） |
| 鸣潮 | 每个可玩角色的显示模型 | 需要你的 AES 密钥文件；游戏文件读取器（由 `tools/wuwa_cli` 构建）已随程序附上；每个材质段落（头发、脸、上身…）各自是一个部件 |
| 既有转出物 | `*_Converted` 文件夹里做好的模型 | 逐个核对 SHA-256 后复制出来 |

## 截图

英文界面，用 `tools/screenshots.py` 实际扫描与预览时拍下的（画面是少女前线 2）。

| | |
|---|---|
| ![扫描安装](../images/scan.png) | ![模型与预览](../images/preview.png) |
| *扫描：能计数的步骤都有自己的进度条* | *模型列表与预览，视角自动对准整个模型* |
| ![模型查看器](../images/viewer.png) | ![设置](../images/settings.png) |
| *模型查看器与文件内容* | *设置：安装文件夹、工具、输出与语言* |

## 功能

- **扫描、勾选、导出。** 每个游戏一页：扫描安装、搜索、勾选模型，排进队列导出。每次扫描与导出都在自己的进程里运行，转换器出错不会拖垮窗口。
- **自动找安装。** 扫描前会列出在 Steam 游戏库找到的安装；选游戏文件夹、它的上一层或文件夹里面都可以。
- **三种格式，其中两种不需要 Blender。** `.glb` 与 FBX 由 Polylode 自己写（骨架、蒙皮网格、形状键、内嵌贴图、透明）；`.blend` 需要 Blender。
  设置了 Blender 时，每个文件都会在 Blender 读回核对。
- **材质按游戏的画法。** 裁切、半透明、眼影叠层与眼睛高光层都按从各游戏 shader 读出的规则写出；glTF／FBX 画不出的叠层，效果会烘进底下的贴图，
  `.blend` 保留真正的分层。鸣潮的裁切覆盖率由执行游戏自己编译好的 shader 得出，视差毛发做成一层层毛丝。终末地的特效材质（翅膀、光效）用游戏编译好的 shader 执行后烘焙，发光贴图写成发光；画不出来的材质不会画成灰白，而是设为不可见并记录原因。
- **导出前预览。** 选一行按「预览」（或双击）：模型转成 `.glb` 放进缓存、显示在列表旁——左键拖动旋转、右键平移、滚轮缩放。程序或游戏文件更新后，按「**重新转换**」重做。
- **模型查看器。** 列出输出文件夹里的所有模型，也能打开任何 `.glb`、`.gltf`、`.fbx`、`.obj`（拖进来、选文件，或用 `Polylode.exe <文件>`，
  可以设成「打开方式」的程序）：正面、背面、侧面、上方视角、线框、深浅背景、灯光强度，以及 glTF 的内容。列表里的模型可以删到回收站。
- **通过 GitHub Pages 分享到 Discord。** Discord 不能显示 3D 模型，所以模型查看器的 **分享到 GitHub Pages…** 会把显示中的
  `.glb` 写进你自己的 GitHub Pages 网站，做成一个在浏览器里就能转动模型的网页，推送后复制网址；网址贴到 Discord 会显示成带模型图片的卡片。
- **五种语言。** English、繁體中文、简体中文、日本語、한국어；默认跟随系统，可在「设置」切换，立即生效。

## 下载

到 [Releases](https://github.com/JE-Chen/Polylode/releases) 下载最新的 `Polylode-<版本>-windows-x64.zip`，解压到任何地方，运行 `Polylode.exe`。
需要的东西都已附上：AnimeStudio（终末地用）、鸣潮读取器、.NET 10 运行环境、glTF 导入器，以及给没有显卡驱动的电脑用的软件 OpenGL。

需求：Windows 10 或 11，64 位。只有 `.blend` 需要 Blender 4.x 或 5.x。

## 使用说明

### 1. 第一次启动

把 Release 解压到任何地方，运行 `Polylode.exe`。界面跟随 Windows 的语言；到「**设置 → 语言**」可以换，立即生效。设置边输入边保存，
和所有缓存一起放在 `%LOCALAPPDATA%\Polylode`。如果你在它还是 IslandLife 一部分时用过导出工具，第一次启动会把它的设置、
终末地资产对照表、GF2 索引与仓库搬过来，不用重建。

### 2. 设置

- **输出文件夹**——导出的位置，每个模型一个文件夹：`<输出>\<游戏>\<模型编号>\`。
- **导出格式**——`.glb`、FBX、`.blend`，可以多选。
- **Blender 可执行文件**——选填。只有 `.blend` 需要；设置之后，每个 `.glb` 与 FBX 也会在 Blender 读回核对。
- **安装文件夹**——可以留空：扫描时会逐一询问，并列出在 Steam 游戏库找到的。选游戏文件夹、它的上一层或文件夹里面都可以。
  存在的路径旁边会打 ✓。
- **工具**——留空就用程序内置的：AnimeStudio（终末地）、鸣潮读取器与 .NET 运行环境。鸣潮另外需要你的 **AES 密钥文件**
  （含你那个版本 `mainKey` 与 `dynamicKeys` 的 JSON）：它的文件是加密的，密钥属于你自己，程序不能附带。第一次预览或导出前，会先在程序旁、它的数据文件夹、安装文件夹与上一层、下载、文档、桌面找你已经有的密钥文件，找不到才问。

### 3. 扫描

打开游戏页，点「**扫描安装**」。状态栏下方的进度条会写出当前的步骤，能计数的就显示进度。第一次扫描要花的时间：

- **明日方舟：终末地**从安装建立资产对照表：几十分钟，分批进行。中途关掉程序不会白做，下次扫描接着做。
- **少女前线 2**建立 bundle 索引（几分钟），并检查哪些根节点是角色模型。
- **鸣潮**从封存文件读出角色表。

之后的扫描在游戏更新前都沿用。游戏更新后请重新扫描；GF2 的安装在扫描后有变动时，预览或导出会提醒你。

### 4. 预览

选一行——搜索框可以按角色、服装或编号过滤——点「**预览**」或双击。一个模型第一次预览时会先转换（几秒到一分钟，有自己的
进度条）；之后选那一行就直接从缓存显示。在预览里：**左键拖动**旋转、**右键拖动**平移、**滚轮**缩放。每个模型加载时都会
自动框住整个模型并居中。

预览会保持第一次做出来的样子。「**重新转换**」会重新转换选中的那一行——更新 Polylode 或游戏之后用——新的做好才替换缓存，
在那之前保留原来的。

### 5. 导出

勾选模型（「**全选**」「**全不选**」只作用在搜索后显示的行），点「**导出勾选的模型**」。下方的队列一次做一个：⏳ 等待、
▶ 进行中、✓ 完成、✗ 失败——旁边的记录写出原因。队列下的进度条计算模型数，再下面一条是正在写出的模型当前的步骤。导出进行中
关闭窗口会先询问。

### 6. 模型查看器

列出输出文件夹里的模型，最新的在前。「**打开文件…**」、把文件拖进窗口，或用 `Polylode.exe <文件>` 启动（可以设成
「打开方式」的程序），就能看任何 `.glb`、`.gltf`、`.fbx`、`.obj`。「**正面**」「**背面**」「**左侧**」「**右侧**」「**上方**」
转动模型；「**线框**」、背景按钮与「**光**」改变画法。右侧面板列出 glTF 的内容——网格、顶点、骨头、形状键与每个材质的混合
方式——并标出按旧规则写出、应重新导出的文件。「**删除…**」把模型、同名的文件与贴图文件夹移到回收站。

### 7. 分享到 GitHub Pages（给 Discord 用）

Discord 不能显示 3D 模型，贴上去的 `.glb` 只是一个下载文件。Polylode 把模型分享成你自己 GitHub 账号里 GitHub Pages 网站上的一个网页：
网页在任何浏览器里都能转动模型，网址贴到 Discord 会显示成带模型图片的卡片。不用 clone；只有超过 12 MB 的文件要装 Git 来推送（GitHub 的上传 API 不收更大的请求）。

1. 在模型查看器打开一个 `.glb`，按 **分享到 GitHub Pages…**。它会先说模型会公开在哪个网址，问过你才继续。
2. 第一次会在浏览器打开 GitHub 的登录页，并显示一组短代码（已复制）：输入代码并授权 Polylode。它只要求 `public_repo`
   （创建与写入公开仓库），授权只存在这台电脑，并以你的 Windows 用户加密保存。**设置 → GitHub 账号 → 退出登录** 会把它删掉；
   也可以在 GitHub 的 Settings → Applications 撤销。
3. 它会在你的账号创建公开仓库 `polylode-models`（只有第一次），然后把 `models/<名称>/`（模型、照预览画出的 1200×630 卡片图、网页）
   与网站的模型列表，以一个 commit 上传。
4. 网址已复制：`https://<账号>.github.io/polylode-models/models/<名称>/`。第一次如果这个仓库的 Pages 还没开启，消息会有
   **打开 Pages 设置**：选 **Deploy from a branch**、`main`、`/ (root)`，按 Save，只需要做一次。（从程序直接开启需要访问你所有私有仓库的权限，程序不会要这个权限。）
5. GitHub Pages 通常一两分钟后更新，之后把网址贴到 Discord 即可。

**姿势。** 导出的 `.glb` 会带着角色界面播放的每一个姿势：在那里停留的姿势排在前面，姿势之间的过渡动作接在后面
（终末地角色的 `ui` 动作，第一次从安装解码，大约半分钟；少女前线 2 服装在战斗以外的动作——宿舍、衣柜、卧室、结局——
第一次导出时建一次名称索引，几分钟；鸣潮角色在自己骨架上的 `CommonAnim/UiAnim` 动作，用这个版本构建的读取器解码，旧版读取器会略过）。
分享的网页有 **Pose** 菜单可以切换，卡片图显示第一个姿势。**设置 → 姿势** 可以关掉；预览一律不含姿势。
只写姿势需要的关键帧（前后两帧之间的动作差在 0.1 mm 与 0.01° 以内的帧就省略），三十几个姿势的模型也远低于 GitHub 单文件 100 MB 的上限。

已经有自己的 Pages 网站？把 **设置 → GitHub Pages 网站文件夹** 指到它在本机的 clone，分享就会改成提交到那里、用 Git 推送。

只能分享 `.glb`，最大 100 MB（GitHub 的单文件上限）；同一个模型再分享一次会替换它的网页。分享出去的东西都是公开的：
模型来自游戏，请自行确认可以分享。本程序自己的文件夹不能当网站。

#### 在 Discord 里转动模型（Discord Activity）

Discord 的链接卡片是一张图：Discord 只让少数网站在聊天里放可以操作的东西。Discord Activity 是能操作的做法：在
**设置 → Discord Activity** 填 Discord 应用程序的 **Application ID**，分享网站的首页同时就是那个应用程序的 Activity——
在语音或文字频道从 Discord 的 Activity 菜单打开，点一个模型，就能在 Discord 里转动、缩放、切换姿势。在 Discord
开发者平台设置一次：**Activities → URL Mappings** 的根 `/` 指到 `<账号>.github.io/polylode-models`（不要加 `https://`），
再到 **Activities → Settings** 打开 **Enable Activities**，**Installation** 勾 User Install 与 Guild Install。填好 ID 后再分享
一次模型，首页才会照它写出。网站自己带 3D 查看器与 Discord Embedded App SDK 的副本（附授权）：Activity 连不到网站以外的地方。

### 疑难排解

- **扫描、预览或导出前弹出说明或选文件窗口**——转换少了一个必需的工具，程序会找的地方也没有。鸣潮的话就是 AES 密钥文件：窗口会说明它是什么；选你的密钥文件，或粘贴密钥（`0x` 加 64 位十六进制），程序会存成密钥文件。之后都会记住。
- 记录出现 **「…还没有设置」或「…不存在」**——到「**设置**」修正那个路径。
- **「量不到这个文件的大小」**——查看器读不出文件的范围（例如 FBX 7.5 的文件）；视角放在原点，用滚轮缩放。
- **没有写出 `.blend`**——到「**设置**」指定 Blender 可执行文件。
- **分享时出现「git push 失败」** —— Git 推送不到网站的仓库：在 Git 登录 GitHub（例如 Git Credential Manager），并确认自己能从那个文件夹推送。
  **「网站文件夹的 origin 不是 GitHub 仓库」** —— 那个文件夹要是某个 GitHub 仓库的 clone。
- **其他失败**——下方的记录保留了扫描或导出打印的每一行。

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

程序附带的鸣潮读取器由 `tools/wuwa_cli/build.ps1` 构建（见它的 `README.md`），再用 `--wuwa-reader` 交给构建；两个工作流都会做这两步。`tools/screenshots.py` 会通过窗口自己的扫描与预览重拍上面的截图。

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
docs/                    截图、工具说明、架构、待办、README 的翻译
build.py, main.py        打包的程序
```

## 许可与内容

Polylode 的代码采用 GNU LGPL 2.1（`LICENSE`）；附带的 AnimeStudio 是 MIT（`tools/AnimeStudio/LICENSE`）。Polylode 不含任何游戏内容：
它转换的是你自己电脑上安装的文件，写出的东西也留在你的电脑上。截图里的模型是从作者自己的安装转出的。只用在你拥有的游戏上，并遵守它们的条款。
