"""Verify independently reopened material-only descendants of covered model deliveries."""
from __future__ import annotations

import json
from pathlib import Path

from . import store


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def covered_file(project: Path, source: str, ident: str, delivery: dict, kind: str) -> tuple[Path, dict]:
    """Accept the recorded model or its hash-linked, geometry-identical native material descendant.

    Only Blender files have a documented descendant workflow. A changed FBX,
    changed geometry fingerprint, stale source document or unverified material
    check is refused instead of loosening the original delivery's hash check.
    """
    path = Path(delivery[kind]).resolve()
    allowed = (project / f"{store.SOURCES[source]}_Converted").resolve()
    if not path.is_relative_to(allowed):
        raise store.WarehouseError("covered delivery leaves the source output folder")
    actual, parent = store.digest(path), delivery[f"{kind}_sha256"]
    if actual == parent:
        return path, {"sha256": actual, "parent_sha256": parent, "material_overlay": False}
    if source != "wuthering_waves" or kind != "blend":
        raise store.WarehouseError(f"covered {kind} changed without a verified descendant: {ident}")
    base = project / "WutheringWaves/shaders_1132"
    installed_path = base / "installed" / f"{ident}.json"
    installed = _read(installed_path)
    if (installed.get("ident") != ident or installed.get("formal_sha256") != actual
            or installed.get("parent_sha256") != parent or Path(installed["formal"]).resolve() != path
            or installed.get("independent_formal_reopen_passed") is not True
            or installed.get("protected_materials_unchanged") is not True):
        raise store.WarehouseError("native material descendant has no matching independent formal readback")
    receipt_path = Path(installed["candidate_receipt"]).resolve()
    if not receipt_path.is_relative_to(base.resolve()):
        raise store.WarehouseError("material descendant receipt leaves its evidence folder")
    if store.digest(receipt_path) != installed["candidate_receipt_sha256"]:
        raise store.WarehouseError("native material descendant receipt changed")
    receipt = _read(receipt_path)
    case = project / "WutheringWaves/covered_1121" / ident
    checks = (receipt.get("candidate_sha256") == actual, receipt.get("parent_sha256") == parent,
              receipt.get("independent_reopen_passed") is True, bool(receipt.get("replaced")),
              receipt.get("geometry_sha256") == installed.get("geometry_sha256"),
              bool(installed.get("geometry_sha256")),
              store.digest(case / "source_document.json") == receipt.get("source_document_sha256"))
    if not all(checks):
        raise store.WarehouseError("material descendant does not preserve the certified source geometry")
    for replacement in receipt["replaced"]:
        check = replacement["numerical_material_check"]
        if (check["samples"] <= 0 or check["alpha_mismatches"] != 0 or check["maximum_color_error"] > 5e-5
                or check.get("native_sampler_addressing_verified") is not True
                or check.get("source_color_spaces_verified") is not True):
            raise store.WarehouseError("material descendant numerical or sampler check failed")
    return path, {"sha256": actual, "parent_sha256": parent, "material_overlay": True,
                  "geometry_sha256": installed["geometry_sha256"],
                  "installed_receipt_sha256": store.digest(installed_path),
                  "candidate_receipt_sha256": store.digest(receipt_path)}
