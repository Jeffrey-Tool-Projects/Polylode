"""Named source morph targets as selectable expression animations."""
from __future__ import annotations

import numpy as np


def write_sidecar(document_path, target):
    """Keep source scalar tracks and authored controls beside the exported model."""
    import json
    from pathlib import Path
    document = json.loads(Path(document_path).read_text(encoding="utf-8"))
    records = [{key: record[key] for key in ("name", "clip", "rate", "frames", "source_curves", "controls",
                                            "unbound_controls", "material_values") if key in record}
               for record in document.get("animations", []) if "source_curves" in record or "controls" in record]
    if not records and not document.get("expression_controls"):
        return None
    target = Path(target)
    target.write_text(json.dumps({"source": document.get("source", ""), "animations": records,
                                  "expression_controls": document.get("expression_controls", []),
                                  "expression_presets": document.get("expression_presets", [])},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def add_shapes(document: dict, arrays: dict) -> int:
    """Append one held expression per source shape name, shared across its renderers."""
    names = dict.fromkeys(name for part in document["parts"] if not part.get("hidden")
                          for name in part["shapes"])
    animations = document.setdefault("animations", [])
    for name in names:
        index = len(animations)
        arrays[f"anim{index}"] = np.repeat(np.asarray(arrays["bone_rest"])[None], 2, axis=0)
        for part_index, part in enumerate(document["parts"]):
            if name in part["shapes"]:
                weights = np.zeros((2, len(part["shapes"])), dtype=np.float32)
                weights[:, part["shapes"].index(name)] = 1.0
                arrays[f"p{part_index}_anim{index}_weights"] = weights
        animations.append({"name": f"expression/{name}", "clip": name, "kind": "expression",
                           "rate": 1.0, "frames": 2})
    return len(names)
