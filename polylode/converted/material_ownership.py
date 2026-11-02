"""Recover exact GF2 material owners per renderer slot, without matching material names."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import store


def slot_owners(document: dict, pieces: list[dict], materials: dict[tuple, dict]) -> list[dict]:
    """Retain CAB/path-id owners even when identical exported properties were deduplicated.

    Renderer labels follow the original rig-scene grouping rule. Each slot's
    properties must agree with both its baked source and the original material
    catalog. A name alone never establishes material or shader ownership.
    """
    groups = []
    for piece in pieces:
        if piece["mat_slot"] == 0 or not groups:
            groups.append([])
        groups[-1].append(piece["shader"])
    renderers, seen = {}, {}
    for group in groups:
        name = group[0]["_rendName"]
        seen[name] = seen.get(name, 0) + 1
        label = name if seen[name] == 1 else f"{name}.{seen[name]}"
        renderers[label] = group
    rows = []
    for part in document["parts"]:
        group = renderers[part["name"]]
        if len(group) != len(part["materials"]):
            raise store.WarehouseError("source renderer material slot count changed")
        slots = []
        for name, shader in zip(part["materials"], group):
            identity = tuple(shader["source_material"])
            original = materials[identity]
            record = document["materials"][name]
            checks = [(record["shader"], shader["_shader"]), (record["queue"], shader["_queue"]),
                      (record["floats"], shader["floats"]), (record["colors"], shader["colors"]),
                      (record["keywords"], shader["keywords"]),
                      (original["name"], shader["_material"]), (original["floats"], shader["floats"]),
                      (original["colors"], shader["colors"]), (original["render_queue"], shader["_queue"]),
                      (original["disabled_passes"], shader["disabled_passes"]), (original["tags"], shader["tags"])]
            if any(left != right for left, right in checks):
                raise store.WarehouseError(f"source material properties changed: {part['name']}/{name}")
            slots.append({"material": name, "identity": list(identity),
                          "shader_identity": original["shader_identity"],
                          "serialized_material_sha256": original["serialized_material_sha256"],
                          "serialized_shader_sha256": original["serialized_shader_sha256"],
                          "disabled_passes": original["disabled_passes"], "tags": original["tags"]})
        rows.append({"renderer": part["name"], "slots": slots})
    return rows


class Gf2Catalog:
    """Read one exact source catalog and only the metadata member of each source archive."""

    def __init__(self, root: Path):
        self.root = Path(root)
        catalog_path = self.root / "runtime/native_material_textures.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog["schema"] != "gf2_native_material_textures_v1":
            raise store.WarehouseError("unsupported native material catalog")
        self.catalog_sha256 = store.digest(catalog_path)
        material_path = store.child(self.root, catalog["materials"]["file"])
        if store.digest(material_path) != catalog["materials"]["sha256"]:
            raise store.WarehouseError("native material catalog changed")
        rows = json.loads(material_path.read_text(encoding="utf-8"))["materials"]
        self.materials = {tuple(row["identity"]): row for row in rows}
        if len(self.materials) != len(rows):
            raise store.WarehouseError("ambiguous native material catalog")

    def prepare(self, ident: str, document: dict) -> dict:
        """Recover this character's slot owners; no vertex arrays or source images are decoded."""
        archive = store.child(self.root, f"chara/{ident}.npz")
        with np.load(archive, allow_pickle=False) as data:
            metadata = json.loads(bytes(data["meta"]).decode("utf-8"))
        if metadata["source"] != "gf2" or metadata["category"] != "chara":
            raise store.WarehouseError("source metadata belongs to another game or category")
        return {"schema": "gf2_native_material_owners_v1", "source": "gf2", "ident": ident,
                "catalog_sha256": self.catalog_sha256, "source_archive_sha256": store.digest(archive),
                "renderers": slot_owners(document, metadata["pieces"], self.materials)}
