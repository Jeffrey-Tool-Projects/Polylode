"""The declared playable roots of an install, and converting one of them: what ``tools/import_gf2.py`` and the extractor share.

``conversion.json`` (in the store) keeps every root's outcome by ``cab/id``:
``converted`` (with its ident), ``not_model`` (a child node, no owned skin,
no geometry), ``excluded_scope`` or ``failed`` with the reason.
"""
from __future__ import annotations

import gc
import json
import traceback
from pathlib import Path

from ..core import progress
from ..core.paths import write_json_atomic
from . import playable, store

REPORT = "conversion.json"


def declared_entries(catalog, declared, only: str = "", names=()) -> list[dict]:
    """The catalog's GameObjects of the locally confirmed playable families, once each, by name."""
    entries = [e for e in catalog.entries if playable.families(e["name"], declared)
               and only.lower() in e["name"].lower() and (not names or e["name"] in names)]
    unique = {(e["cab"], e["id"]): e for e in entries}
    return sorted(unique.values(), key=lambda e: (e["name"], e["cab"], e["id"]))


def load_results() -> dict:
    """The stored outcomes, by ``cab/id``."""
    path = store.root() / REPORT
    previous = json.loads(path.read_text(encoding="utf8")) if path.exists() else {"results": []}
    return {f'{r["entry"]["cab"]}/{r["entry"]["id"]}': r for r in previous["results"]}


def save_results(install: Path, results: dict) -> None:
    write_json_atomic(store.root() / REPORT, {"version": 1, "install": str(install),
                                             "results": list(results.values())}, indent=1)


def skinned_roots(catalog, entries: list[dict], declared, results: dict, log=print) -> list[dict]:
    """*entries* (roots) that have their own skinned mesh of a declared family and the whole skeleton a playable
    character's conversion requires (``playable.has_human_rig``) — asked before anything is converted, so a scan
    lists characters and not dorm scenes, pets, weapons or props.

    A root without one is kept in *results* as ``not_model`` (as a conversion would have); a root
    already known as not a model is left out; a root that cannot be read stays listed, for its
    export to say why.  Each bundle file is opened once.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["cab"]].append(entry)
    kept = []
    for done, (cab, members) in enumerate(sorted(groups.items()), 1):
        progress.report(done, len(groups), '檢查哪些根節點是角色模型')
        env = catalog.environment()
        try:
            file = env.find_file(cab)
            for entry in members:
                key = f'{entry["cab"]}/{entry["id"]}'
                if results.get(key, {}).get("status") in ("not_model", "excluded_scope"):
                    continue
                try:
                    go = file.objects[entry["id"]].read()
                    owned = playable.has_owned_skin(go, playable.families(entry["name"], declared))
                    rigged = owned and playable.has_human_rig(go)
                    meshed = rigged and playable.has_owned_mesh(go, playable.families(entry["name"], declared))
                except (ValueError, TypeError, KeyError, IndexError, OSError, EOFError, StopIteration,
                        NotImplementedError, RuntimeError, AttributeError):
                    owned = rigged = meshed = True     # unread: listed, its export reports the reason
                if owned and rigged and meshed:
                    kept.append(entry)
                else:
                    results[key] = {"entry": entry, "status": "not_model",
                                    "reason": "no owned skinned mesh renderer" if not owned
                                    else "no complete human skeleton (a pet, weapon or prop)" if not rigged
                                    else "its skinned renderers point at no mesh (a _model variant)"}
        except (ValueError, TypeError, KeyError, IndexError, OSError, EOFError, NotImplementedError,
                RuntimeError, AttributeError):
            kept.extend(members)
        finally:
            del env
        if done % 100 == 0 or done == len(groups):
            log(f"檢查蒙皮網格：{done} / {len(groups)} 個檔案，{len(kept)} 個是模型")
    return kept


def convert_entry(catalog, entry: dict, declared, results: dict, *, force: bool = False, log=print) -> str:
    """Convert one root into the store unless it already is; its outcome, also kept in *results*.

    Only a root with its own skinned mesh of a declared family is a model: a
    child node or a root whose skin all belongs to props is ``not_model``.
    """
    from .convert import NoGeometryError, convert
    key = f'{entry["cab"]}/{entry["id"]}'
    cached = results.get(key, {})
    if (cached.get("status") in ("converted", "not_model", "excluded_scope") and not force
            and (cached["status"] in ("not_model", "excluded_scope") or store.read_item(cached["ident"]) is not None)):
        return cached["status"]
    env = catalog.environment()
    go = transform = None
    try:
        go = env.find_file(entry["cab"]).objects[entry["id"]].read()
        transform = next(c.component.read() for c in go.m_Component if c.component.type.name == "Transform")
        if transform.m_Father:
            results[key] = {"entry": entry, "status": "not_model", "reason": "child of another GameObject"}
            return "not_model"
        # Foreign props can reference undecoded native resources. They cannot
        # make a root without owned skin into a playable character.
        if not playable.has_owned_skin(go, playable.families(entry["name"], declared)):
            results[key] = {"entry": entry, "status": "not_model", "reason": "no owned skinned mesh renderer"}
            return "not_model"
        log(f'CONVERT {entry["name"]}')
        ident, extra = convert(catalog, entry)
        results[key] = {"entry": entry, "status": "converted", "ident": ident,
                        "renderers": extra["renderers"], "source_files": extra["source_files"],
                        "archive_readback": True}
        log(f"OK {ident}")
        return "converted"
    except NoGeometryError as exc:
        results[key] = {"entry": entry, "status": "not_model", "reason": str(exc)}
        log(f'NO_GEOMETRY {entry["name"]} {exc}')
        return "not_model"
    except (ValueError, TypeError, KeyError, IndexError, OSError, EOFError,
            NotImplementedError, RuntimeError, AttributeError, AssertionError) as exc:
        results[key] = {"entry": entry, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        log(f'FAILED {entry["name"]} {results[key]["reason"]}')
        log(traceback.format_exc())
        return "failed"
    finally:
        del env
        go = transform = None
        gc.collect()
