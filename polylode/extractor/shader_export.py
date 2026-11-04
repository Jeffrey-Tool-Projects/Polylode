"""Portable character shader sources and a directly usable Blender companion."""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from .character_styles import LIMITS, read_materials

ADAPTER = Path(__file__).with_name("blender_shader.py")
SHADER = Path(__file__).with_name("character.frag")


def write_bundle(model: Path, target: Path) -> Path:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    texture_dir = target / "textures"
    document, blob, materials = read_materials(Path(model), texture_dir)
    used = {p["material"] for mesh in document.get("meshes", []) for p in mesh["primitives"] if "material" in p}
    for index, material in enumerate(document.get("materials", [])):
        material.setdefault("extras", {})["polylode_material_index"] = index
    for data in materials:
        for key in ("baseTexture", "ramp", "surface", "shadowLut"):
            uri = data[key]
            if not uri:
                continue
            if uri.startswith("data:"):
                header, encoded = uri.split(",", 1)
                pixels = base64.b64decode(encoded) if ";base64" in header else unquote(encoded).encode()
                suffix = ".png" if "image/png" in header else ".jpg"
            else:
                parsed = urlparse(uri)
                if parsed.scheme != "file":
                    raise ValueError(f"shader texture must be local: {uri}")
                source = Path(url2pathname(unquote(parsed.path)))
                pixels, suffix = source.read_bytes(), source.suffix
            texture_dir.mkdir(exist_ok=True)
            name = hashlib.sha256(pixels).hexdigest() + suffix
            (texture_dir / name).write_bytes(pixels)
            data[key] = "textures/" + name
    text = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    blob += b"\0" * (-len(blob) % 4)
    model_path = target / "model.glb"
    with model_path.open("wb") as out:
        out.write(struct.pack("<III", 0x46546C67, 2, 28+len(text)+len(blob)))
        out.write(struct.pack("<II", len(text), 0x4E4F534A)); out.write(text)
        out.write(struct.pack("<II", len(blob), 0x004E4942)); out.write(blob)
    bindings = [{"node": i, "mesh": node["mesh"], "primitive": j, "material": p.get("material")}
                for i, node in enumerate(document.get("nodes", [])) if "mesh" in node
                for j, p in enumerate(document["meshes"][node["mesh"]]["primitives"])]
    manifest = {"schema": 1, "model": "model.glb", "limitations": LIMITS,
                "materials": [row for row in materials if row["id"] in used], "bindings": bindings}
    (target / "materials.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy2(ADAPTER, target / "import_blender.py")
    shutil.copy2(SHADER, target / "character.frag")
    (target / "README.txt").write_text(
        "Blender character shading\n\n"
        "Open the adjacent .shaded.blend directly. Its node materials and images are packed.\n"
        "To rebuild with Blender 4.x/5.x, run from any directory:\n"
        'blender --background --factory-startup --python-exit-code 1 --python "<this folder>/import_blender.py" '
        '-- "<output>.shaded.blend"\n\n'
        "Keep this entire folder together. No Polylode installation is required.\n"
        "The adapter imports model.glb into a new scene; it does not modify an open project.\n"
        "Materials translate character.frag into native Blender nodes, with explicit preview illumination.\n"
        "These are adapted character shaders, not full game scene lighting, shadows, postprocessing, or dedicated eye/hair effects.\n"
        "Unlit/VFX materials retain their imported material. The original .blend remains separate.\n",
        encoding="utf-8")
    return target


def write_blend(bundle: Path, blender: Path, output: Path) -> Path:
    done = subprocess.run([str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
                           "--python", str(Path(bundle).resolve() / "import_blender.py"), "--", str(Path(output).resolve())],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if done.returncode != 0 or not Path(output).is_file():
        raise ValueError(f"Blender character shading failed ({done.returncode}): {(done.stdout+done.stderr)[-3000:]}")
    return Path(output)
