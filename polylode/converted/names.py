"""Resolve Wuthering Waves folder names from extracted native Chinese localization tables."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

from . import store


#: Where the project's extraction keeps the game's own tables (``Client/Content/…`` below it).
EXTRACTION = "WutheringWaves/extraction_1121"


def wuthering_name(project: Path, keys: list[str], extraction: Path | None = None) -> tuple[str, dict]:
    """Read each role's actual localized name; absent or conflicting names stop conversion.

    *extraction* is the folder the game's files were extracted into (the
    project's own, :data:`EXTRACTION`, unless named).
    """
    base = (extraction or project / EXTRACTION) / "Client/Content/Aki/ConfigDB/zh-Hant"
    databases = [base / "lang_multi_text.db", base / "lang_multi_text_1sthalf.db"]
    translations = {}
    for key in sorted(set(keys)):
        matches = []
        for path in databases:
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as database:
                columns = {row[1] for row in database.execute("PRAGMA table_info(MultiText)")}
                if columns == {"Id", "Content", "RedirectDbIndex"}:
                    query = "SELECT Content,RedirectDbIndex FROM MultiText WHERE Id=?"
                elif columns == {"Id", "Content"}:  # native first-half database has no redirect field
                    query = "SELECT Content,0 FROM MultiText WHERE Id=?"
                else:
                    raise store.WarehouseError(f"unsupported native localization schema: {path.name}")
                for content, redirect in database.execute(query, (key,)):
                    if content and not redirect:
                        matches.append({"name": content, "database": path.name, "key": key})
        values = {row["name"] for row in matches}
        if len(values) != 1:
            raise store.WarehouseError(f"unresolved or conflicting native Chinese role name: {key}")
        translations[key] = matches[0]
    if not translations:
        raise store.WarehouseError("no verified role localization keys")
    name = "、".join(sorted({row["name"] for row in translations.values()}))
    return name, {"language": "zh-Hant", "translations": translations}
