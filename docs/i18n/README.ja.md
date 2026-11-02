# Polylode

[English](../../README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · **日本語** · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md)

Polylode はインストール済みのゲームからキャラクターモデルを掘り出し——鉱脈（*lode*）からポリゴン（*poly*gon）を——テクスチャとリグ付きの、
どこでも開けるファイルに書き出します：glTF（`.glb`）、FBX、Blender（`.blend`）。ウィンドウ、マウスで回せるプレビュー、モデルビューアを備えた
Windows デスクトッププログラムです。

| ゲーム | 読み取るもの | 備考 |
|---|---|---|
| アークナイツ：エンドフィールド | すべてのプレイアブルキャラクターのプレハブ | 初回スキャンでインストールから資産対応表を作成（分割検証・中断再開可）、ゲーム更新後に作り直し |
| ドールズフロントライン 2：エクシリウム | すべてのプレイアブルキャラクターの衣装 | 自身のスキンメッシュと完全な人型骨格を持つルートのみ（ペット・小物・空のバリアントは除外） |
| 鳴潮 | すべてのプレイアブルキャラクターの表示モデル | `tools/wuwa_cli` からビルドした読み取りツールと AES キーファイルが必要 |
| 既存の変換物 | `*_Converted` フォルダの完成モデル | 各ファイルの SHA-256 を確認してからコピー |

## 機能

- **スキャン、チェック、書き出し。** ゲームごとにページがあり、インストールをスキャンし、検索し、モデルをチェックしてキューで書き出します。
  スキャンと書き出しはそれぞれ別プロセスで動くため、変換の失敗がウィンドウを巻き込みません。
- **インストールを自動で探す。** スキャン前に Steam ライブラリで見つかったインストールを候補に出します。ゲームのフォルダ、その一つ上、データの中のどれを選んでも構いません。
- **三つの形式、うち二つは Blender 不要。** `.glb` と FBX は Polylode 自身が書き出します（骨格、スキンメッシュ、シェイプキー、埋め込みテクスチャ、透明）。
  `.blend` には Blender が必要です。Blender を指定すると、すべてのファイルを Blender で読み戻して確認します。
- **ゲームどおりのマテリアル。** カットアウト、半透明、アイシャドウのオーバーレイ、目のハイライト層を各ゲームのシェーダーから読んだ規則で書き出します。
  glTF／FBX で描けない層はその効果を下のテクスチャに焼き込み、`.blend` は本当の層のまま残します。
- **書き出し前のプレビュー。** 行を選んで「プレビュー」（またはダブルクリック）：モデルを `.glb` に変換してキャッシュに置き、一覧の横に表示します——
  左ドラッグで回転、右ドラッグで移動、ホイールで拡大縮小。
- **モデルビューア。** 出力フォルダのすべてのモデルを一覧し、任意の `.glb`・`.gltf`・`.fbx`・`.obj` を開けます（ドロップ、選択、または
  `Polylode.exe <ファイル>`。「プログラムから開く」に設定可）。正面・背面・側面・上からの視点、ワイヤーフレーム、明暗の背景、ライトの強さ、glTF の内容。
  一覧のモデルはごみ箱へ削除できます。
- **五つの言語。** English、繁體中文、简体中文、日本語、한국어。既定はシステムに従い、「設定」で選ぶとすぐ切り替わります。

## ダウンロード

[Releases](https://github.com/JE-Chen/Polylode/releases) から最新の `Polylode-<バージョン>-windows-x64.zip` を取得し、任意の場所に展開して
`Polylode.exe` を実行します。必要なものはすべて同梱しています：AnimeStudio（エンドフィールド用）、.NET 10 ランタイム、glTF インポーター、
GPU ドライバのないコンピュータ向けのソフトウェア OpenGL。

動作環境：Windows 10 または 11（64 ビット）。`.blend` にのみ Blender 4.x または 5.x が必要です。

## 使い方

1. **設定**——出力フォルダと書き出し形式を指定します。インストールフォルダは空のままで構いません：スキャン時に尋ねます（Steam で見つかったものも候補に出します）。
2. **ゲームのページ**——「インストールをスキャン」。エンドフィールドの初回スキャンは資産対応表を作ります（数十分、再開可）。GF2 の初回スキャンは bundle の索引を作ります。
   以降はゲームが更新されるまで再利用します。
3. モデルをチェックして「チェックしたモデルを書き出す」を押し、下のキューとログを見ます。
4. **モデルビューア**——書き出した結果を見る、比べる、削除する。

設定とキャッシュは `%LOCALAPPDATA%\Polylode` にあります。IslandLife の一部だった頃の書き出しツールを使っていた場合、初回起動時にその設定、
エンドフィールドの資産対応表、GF2 の索引とストアを移すので、作り直しは不要です。

## ソースからビルド

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m polylode.extractor          # ソースからウィンドウを起動
.venv\Scripts\python -m pytest tests                # テスト
.venv\Scripts\python build.py --dotnet <dotnet フォルダ>   # dist\Polylode\Polylode.exe、続けてセルフテスト
```

ビルドは Nuitka でコンパイルし（C コンパイラが必要：Visual Studio のもの）、AnimeStudio、.NET 10 ランタイム（`--dotnet`、既定は `DOTNET_ROOT`）、
ソフトウェア OpenGL をプログラムの横に置き、`Polylode.exe --selftest` を実行します。すべての変換モジュール、プレビューの QML、実際の `.glb` を読み込み、
どれかが欠けていればビルドは失敗します。

## CI/CD

- **CI**（`.github/workflows/ci.yml`）——`main` と `dev` へのプッシュと PR のたびに、Windows でリント、テスト、セルフテストを実行。
- **Release**（`.github/workflows/release.yml`）——`main` へのプッシュのたびに、テスト、Nuitka ビルドとパッケージ版のセルフテストを行い、
  圧縮したプログラムを GitHub Release として公開。

開発は `dev`、`main` がリリース版です。

## 構成

```
polylode/extractor/      ウィンドウ、スキャン、ジョブ、プレビュー、ビューア、言語
polylode/endfield/       アークナイツ：エンドフィールド——資産対応表、変換、FBX シーン
polylode/gf2/            ドールズフロントライン 2——bundle、プレイアブル名簿、変換、目の層の焼き込み
polylode/wuthering_waves/  鳴潮——ロール表、表示モデルの組み立て
polylode/interchange/    glTF と FBX のライター（Blender 不要）
tools/                   Blender スクリプト、アイコン、FBX SDK 確認、鳴潮読み取りツールのソース、AnimeStudio
tests/                   pytest
build.py, main.py        パッケージ版のプログラム
```

## ライセンスとコンテンツ

Polylode のコードは GNU LGPL 2.1（`LICENSE`）、同梱の AnimeStudio は MIT（`tools/AnimeStudio/LICENSE`）です。Polylode はゲームのコンテンツを含みません：
あなた自身のコンピュータにインストールされたファイルを変換し、書き出したものもそこに残ります。所有しているゲームにのみ使い、その規約を守ってください。
