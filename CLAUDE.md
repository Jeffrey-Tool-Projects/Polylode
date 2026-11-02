# CLAUDE.md — Polylode 的工作規則

## commit 與 PR 不得提到由 AI 實作

commit 訊息、PR 標題與內文、PR／issue 留言、Release 說明、倉庫描述等 GitHub 上的文字，**一律不寫**由 AI／Claude／Copilot／助理實作或產生，
也不加 `Co-Authored-By`、「Generated with」之類的署名或產生器標記。訊息只寫**做了什麼、為什麼**，用一般開發者的口吻。
倉庫裡的檔案（包括本檔，以及程式註解裡引用 IslandLife `CLAUDE.md` 的地方）可以保留。

## 分支與發行

- 開發在 `dev`；`main` 是發行版。推到 `main` 會由 `.github/workflows/release.yml` 建置 `Polylode.exe` 並發佈 Release，所以只把通過 CI 的 `dev` 快轉進 `main`。
- 待辦只記在 `docs/TODO.md`，做完的整段刪掉。工具說明在 `docs/tools.md`，架構在 `docs/architecture.md`。
- 程式從 IslandLife 搬來（`D:\Codes\IslandLife`，索引在它的 `docs/related_repositories.md`）。兩邊不互相 import；同一個轉換器的錯如果兩邊都有，兩邊都要修。
  註解裡的 §7.x 指的是 IslandLife 的 `docs/updates/` 逐節紀錄。

## 驗證

- `python -m pytest tests`、`python -m ruff check --select F,E9 polylode tests tools build.py main.py`，以及 `python -m polylode.extractor --selftest <json>`。
- 改到打包內容（新的延遲匯入、資料檔、外掛）時，跑一次 `python build.py`，並且把那一項加進 selftest 的檢查清單。
