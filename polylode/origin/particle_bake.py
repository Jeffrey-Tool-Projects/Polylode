"""The H scene's particle systems, converted (``HParticleCtrl``, D13).

``HParticleCtrl.Init`` reads the ``h_particle`` list -- id, bundle, prefab, the body it hangs on
(``numParent``), the bone (``nameParent``), a local position and an Euler rotation -- and ``Load`` hangs
each prefab under its bone; ``Play`` plays its ``ParticleSystem``.  ``MetaballCtrl.Proc`` plays id 0 for
emitters 1 and 4 when ``SemenType`` is 2 (IL_0080–IL_0097).

This module turns each prefab's systems (the root's and the sub-emitters it names) into the numbers a
runtime needs, and writes their textures to PNG with their colour space.  Only the modules this project
simulates (``game.intimate.fluid_particles``) are converted; any other module a system turns on is listed
under ``unsupported`` for the runtime to report, not approximated (CLAUDE.md 一之一).
"""
from __future__ import annotations

from pathlib import Path

#: the ``ParticleSystem`` modules the runtime simulates
SUPPORTED = ("InitialModule", "EmissionModule", "ForceModule", "CollisionModule", "SizeModule",
             "ColorModule", "SubModule", "ShapeModule", "InheritVelocityModule")
#: modules a system may carry; one of these turned on is reported (``ShapeModule`` only when on)
OTHERS = ("RotationModule", "UVModule", "VelocityModule", "ExternalForcesModule",
          "ClampVelocityModule", "NoiseModule", "SizeBySpeedModule", "RotationBySpeedModule",
          "ColorBySpeedModule", "TriggerModule", "LightsModule", "TrailModule", "CustomDataModule")


def particle_rows(rows: list) -> list:
    """``HParticleCtrl.Init``'s rows: ``{id, bundle, prefab, body, bone, pos, rot}`` (IL_0049–IL_0166)."""
    out = []
    for cols in rows:
        if len(cols) < 11 or not cols[0].strip().lstrip("-").isdigit():
            continue
        try:
            nums = [float(c) for c in cols[5:11]]
            out.append({"id": int(cols[0]), "bundle": cols[1].strip(), "prefab": cols[2].strip(),
                        "body": int(cols[3]), "bone": cols[4].strip(), "pos": nums[:3], "rot": nums[3:6]})
        except ValueError:
            continue
    return out


def _curve(c: dict) -> dict:
    """A ``MinMaxCurve``: its mode (0 constant, 1 curve, 2 two curves, 3 two constants) and values."""
    def keys(k):
        return [[float(x["time"]), float(x["value"]), float(x.get("inSlope", 0.0)), float(x.get("outSlope", 0.0))]
                for x in (k or {}).get("m_Curve", [])]
    return {"mode": int(c.get("minMaxState", 0)), "scalar": float(c.get("scalar", 0.0)),
            "min": float(c.get("minScalar", 0.0)), "keys": keys(c.get("maxCurve")), "min_keys": keys(c.get("minCurve"))}


def _gradient(g: dict) -> dict:
    """A ``Gradient``: its colour and alpha keys as ``[time, …]`` (the times are 16-bit fractions)."""
    nc, na = int(g.get("m_NumColorKeys", 2)), int(g.get("m_NumAlphaKeys", 2))
    colors = [[g[f"ctime{i}"] / 65535.0, g[f"key{i}"]["r"], g[f"key{i}"]["g"], g[f"key{i}"]["b"]] for i in range(nc)]
    alphas = [[g[f"atime{i}"] / 65535.0, g[f"key{i}"]["a"]] for i in range(na)]
    return {"colors": colors, "alphas": alphas, "mode": int(g.get("m_Mode", 0))}


def _color(c: dict) -> dict:
    """A ``MinMaxGradient``: mode (0 colour, 1 gradient, 2 two colours, 3 two gradients) and values."""
    def rgba(x):
        return [float(x["r"]), float(x["g"]), float(x["b"]), float(x["a"])]
    return {"mode": int(c.get("minMaxState", 0)), "max": rgba(c["maxColor"]), "min": rgba(c["minColor"]),
            "gradient": _gradient(c["maxGradient"]), "min_gradient": _gradient(c["minGradient"])}


def read_system(L, ps, textures: dict, out_dir: Path, seen: set | None = None) -> dict:
    """One ``ParticleSystem`` (and the sub-emitters it names) as plain numbers; its texture written to *out_dir*."""
    seen = set() if seen is None else seen
    seen.add(ps.path_id)
    t = ps.read_typetree()
    im, em = t["InitialModule"], t["EmissionModule"]
    out = {
        "duration": float(t["lengthInSec"]), "looping": bool(t["looping"]), "speed": float(t["simulationSpeed"]),
        "world": int(t["moveWithTransform"]) == 1, "start_delay": _curve(t["startDelay"]),
        "lifetime": _curve(im["startLifetime"]), "start_speed": _curve(im["startSpeed"]),
        "size": _curve(im["startSize"]), "rotation": _curve(im["startRotation"]),
        "color": _color(im["startColor"]), "gravity": _curve(im["gravityModifier"]),
        "max_particles": int(im["maxNumParticles"]),
        "rate": _curve(em["rateOverTime"]) if em.get("enabled") else None,
        "bursts": [{"time": float(b["time"]), "min": int(b["minCount"]), "max": int(b["maxCount"]),
                    "cycles": int(b.get("cycleCount", 1)), "interval": float(b.get("repeatInterval", 0.0))}
                   for b in (em.get("m_Bursts") or [])[:int(em.get("m_BurstCount", 0))]] if em.get("enabled") else [],
        "unsupported": [m for m in OTHERS if (t.get(m) or {}).get("enabled")],
    }
    if (t.get("ShapeModule") or {}).get("enabled"):
        out["unsupported"].append("ShapeModule")
    fm = t.get("ForceModule") or {}
    if fm.get("enabled"):
        out["force"] = {"x": _curve(fm["x"]), "y": _curve(fm["y"]), "z": _curve(fm["z"]),
                        "world": bool(fm.get("inWorldSpace")), "per_frame": bool(fm.get("randomizePerFrame"))}
    cm = t.get("CollisionModule") or {}
    if cm.get("enabled"):
        if int(cm.get("type", 0)) != 1:
            out["unsupported"].append("CollisionModule(planes)")
        out["collision"] = {"dampen": _curve(cm["m_Dampen"]), "bounce": _curve(cm["m_Bounce"]),
                            "lifetime_loss": _curve(cm["m_EnergyLossOnCollision"]),
                            "min_kill_speed": float(cm.get("minKillSpeed", 0.0)),
                            "radius_scale": float(cm.get("radiusScale", 1.0))}
    iv = t.get("InheritVelocityModule") or {}
    if iv.get("enabled"):
        # 0 Initial: the emitter's velocity at birth, times the curve; 1 Current: every frame
        out["inherit"] = {"mode": int(iv.get("m_Mode", 0)), "curve": _curve(iv["m_Curve"])}
        if out["inherit"]["mode"] != 0:
            out["unsupported"].append("InheritVelocityModule(current)")
    sm = t.get("SizeModule") or {}
    if sm.get("enabled"):
        out["size_over_life"] = _curve(sm["curve"])
        if sm.get("separateAxes"):
            out["unsupported"].append("SizeModule(separateAxes)")
    co = t.get("ColorModule") or {}
    if co.get("enabled"):
        out["color_over_life"] = _color(co["gradient"])
    subs = []
    for s in ((t.get("SubModule") or {}).get("subEmitters") or []) if (t.get("SubModule") or {}).get("enabled") else ():
        child = L.by_id.get(s["emitter"]["m_PathID"])
        if child is None or child.path_id in seen:
            continue
        # 0 birth, 1 collision, 2 death (``ParticleSystemSubEmitterType``)
        subs.append({"type": int(s["type"]), "system": read_system(L, child, textures, out_dir, seen)})
    out["subs"] = subs
    out["renderer"] = _renderer(L, t["m_GameObject"]["m_PathID"], textures, out_dir)
    return out


def _renderer(L, go_id: int, textures: dict, out_dir: Path) -> dict | None:
    """The system's ``ParticleSystemRenderer``: on or off, its mode and its first material's map and colour."""
    go = L.by_id.get(go_id)
    for comp in (go.read_typetree().get("m_Component") or []) if go is not None else ():
        r = L.by_id.get(comp["component"]["m_PathID"])
        if r is None or r.type.name != "ParticleSystemRenderer":
            continue
        rt = r.read_typetree()
        out = {"enabled": bool(rt.get("m_Enabled")), "mode": int(rt.get("m_RenderMode", 0)),
               "length_scale": float(rt.get("m_LengthScale", 2.0)), "velocity_scale": float(rt.get("m_VelocityScale", 0.0)),
               "max_size": float(rt.get("m_MaxParticleSize", 0.5)), "texture": None, "srgb": True,
               "tint": [1.0, 1.0, 1.0, 1.0], "shader": ""}
        mats = [L.by_id.get(m["m_PathID"]) for m in rt.get("m_Materials") or []]
        mat = next((m for m in mats if m is not None), None)
        if mat is None:
            return out
        mt = mat.read_typetree()
        sh = L.by_id.get(mt["m_Shader"]["m_PathID"])
        out["shader"] = (sh.read_typetree().get("m_ParsedForm") or {}).get("m_Name", "") if sh is not None else ""
        props = mt.get("m_SavedProperties") or {}
        for name, c in props.get("m_Colors") or []:
            if name in ("_Color", "_TintColor"):
                out["tint"] = [float(c["r"]), float(c["g"]), float(c["b"]), float(c["a"])]
        for name, env in props.get("m_TexEnvs") or []:
            tex = L.by_id.get(env["m_Texture"]["m_PathID"]) if name == "_MainTex" else None
            if tex is None:
                continue
            key = str(tex.read_typetree().get("m_Name") or tex.path_id)
            if key not in textures:
                m = L._decode(tex)
                if m is None:
                    textures[key] = None
                else:
                    from PIL import Image
                    out_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(m.px, "RGBA").save(out_dir / f"{key}.png")
                    textures[key] = {"srgb": bool(m.srgb)}
            if textures[key] is not None:
                out["texture"], out["srgb"] = key, textures[key]["srgb"]
        return out
    return None


def bake_particles(rows: list, abdata: Path, out_dir: Path) -> tuple:
    """``({id: {row…, "system": …}}, failed)``: each ``h_particle`` row's prefab, its textures in *out_dir*."""
    from .prefab import loader
    out: dict = {}
    failed: list = []
    textures: dict = {}
    for row in rows:
        path = abdata / row["bundle"]
        if not path.is_file():
            failed.append(f"particle {row['id']}: no {row['bundle']}")
            continue
        L = loader(path)
        go = L.find_prefab(row["prefab"])
        ps = None
        for comp in (go.read_typetree().get("m_Component") or []) if go is not None else ():
            obj = L.by_id.get(comp["component"]["m_PathID"])
            if obj is not None and obj.type.name == "ParticleSystem":
                ps = obj
        if ps is None:
            failed.append(f"particle {row['id']}: {row['prefab']} has no ParticleSystem")
            continue
        out[str(row["id"])] = dict(row, system=read_system(L, ps, textures, out_dir))
    return out, failed
