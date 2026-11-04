"""Standalone Blender node translation of Polylode's character.frag preview equations."""
from __future__ import annotations

import json
import math
from pathlib import Path


class Value:
    def __init__(self, graph, socket, vector=False):
        self.g, self.socket, self.vector = graph, socket, vector
        self.components = None

    def __add__(self, other): return self.g.op("ADD", self, other)
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self.g.op("SUBTRACT", self, other)
    def __rsub__(self, other): return self.g.op("SUBTRACT", other, self)
    def __mul__(self, other): return self.g.op("MULTIPLY", self, other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self.g.op("DIVIDE", self, other)
    def __rtruediv__(self, other): return self.g.op("DIVIDE", other, self)
    def __neg__(self): return self * -1
    def __pow__(self, other): return self.g.op("POWER", self, other)

    def __getitem__(self, index):
        if self.components is None:
            node = self.g.node("ShaderNodeSeparateXYZ")
            self.g.link(self, node.inputs[0])
            self.components = [Value(self.g, socket) for socket in node.outputs]
        return self.components[index]


class Graph:
    def __init__(self, tree):
        self.tree = tree
        self.images = {}

    def node(self, kind, label=""):
        node = self.tree.nodes.new(kind)
        node.label = label
        node.location = (len(self.tree.nodes) % 8 * 220, -(len(self.tree.nodes) // 8) * 160)
        return node

    def link(self, value, socket):
        if isinstance(value, Value):
            self.tree.links.new(value.socket, socket)
        elif socket.type in ("VECTOR", "RGBA"):
            size = 4 if socket.type == "RGBA" else 3
            values = list(value) if isinstance(value, (tuple, list)) else [float(value)] * size
            socket.default_value = (*values[:3], 1.) if size == 4 and len(values) == 3 else values[:size]
        else:
            socket.default_value = float(value)

    def value(self, value, label=""):
        node = self.node("ShaderNodeValue", label)
        node.outputs[0].default_value = float(value)
        return Value(self, node.outputs[0])

    def vec(self, x, y=None, z=None):
        if y is None:
            if isinstance(x, Value) and x.vector:
                return x
            if isinstance(x, (tuple, list)):
                x, y, z = x[:3]
            else:
                y = z = x
        node = self.node("ShaderNodeCombineXYZ")
        for socket, value in zip(node.inputs, (x, y, z)):
            self.link(value, socket)
        return Value(self, node.outputs[0], True)

    def op(self, operation, a, b=None):
        vector = any(isinstance(v, (tuple, list)) or isinstance(v, Value) and v.vector for v in (a, b))
        if vector and operation in ("POWER", "ABSOLUTE", "FLOOR"):
            a = self.vec(a)
            return self.vec(*(self.op(operation, a[i], b) for i in range(3)))
        node = self.node("ShaderNodeVectorMath" if vector else "ShaderNodeMath")
        node.operation = operation
        self.link(a, node.inputs[0])
        if b is not None:
            self.link(b, node.inputs[1])
        return Value(self, node.outputs[0], vector)

    def dot(self, a, b):
        node = self.node("ShaderNodeVectorMath")
        node.operation = "DOT_PRODUCT"
        self.link(a, node.inputs[0]); self.link(b, node.inputs[1])
        return Value(self, node.outputs["Value"])

    def normalize(self, value): return self.op("NORMALIZE", self.vec(value))
    def min(self, a, b): return self.op("MINIMUM", a, b)
    def max(self, a, b): return self.op("MAXIMUM", a, b)
    def clamp(self, value, low=0., high=1.): return self.min(self.max(value, low), high)
    def mix(self, a, b, t): return self.op("ADD", self.op("MULTIPLY", a, 1-t), self.op("MULTIPLY", b, t))
    def luma(self, value): return self.dot(value, (.212673, .715152, .072175))
    def sat(self, value, amount): return self.mix(self.vec(self.luma(value)), value, amount)
    def cubic(self, value): return value * value * (3 - 2 * value)

    def sample(self, path, uv, srgb, clamp=False):
        import bpy
        key = (str(path), srgb)
        if key not in self.images:
            image = bpy.data.images.load(str(path), check_existing=False)
            image.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
            image.pack()
            self.images[key] = image
        node = self.node("ShaderNodeTexImage", Path(path).name)
        node.image = self.images[key]
        node.extension = "EXTEND" if clamp else "REPEAT"
        self.link(uv, node.inputs["Vector"])
        return Value(self, node.outputs["Color"], True), Value(self, node.outputs["Alpha"])


def apply_material(material, data, folder):
    material.use_nodes = True
    tree = material.node_tree
    g = Graph(tree)
    original = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if original is None:
        raise ValueError(f"no imported base material for {material.name}")

    def original_input(name, vector=False):
        socket = original.inputs[name]
        if socket.is_linked:
            return Value(g, socket.links[0].from_socket, vector)
        return g.vec(socket.default_value[:3]) if vector else g.value(socket.default_value)

    base, alpha = original_input("Base Color", True), original_input("Alpha")
    p = {key: g.value(value, key) for key, value in data.items() if isinstance(value, (int, float))}
    geom = g.node("ShaderNodeNewGeometry")
    normal = Value(g, geom.outputs["Normal"], True)
    incoming = Value(g, geom.outputs["Incoming"], True)
    normal = g.normalize(g.vec(normal[0], normal[2], -normal[1]))
    view = g.normalize(g.vec(incoming[0], incoming[2], -incoming[1]))
    light = g.normalize((.147, .530, .835))
    nl, nv = g.dot(normal, light), g.max(g.dot(normal, view), 0)
    uv_node = g.node("ShaderNodeTexCoord")
    uv = Value(g, uv_node.outputs["UV"], True)

    def sample(key, coords, clamp=False):
        return g.sample(folder / data[key], coords, data.get(key + "Srgb", key == "baseTexture"), clamp)

    def surface(default):
        if data.get("surface"):
            rgb, a = sample("surface", uv)
            return rgb[0], rgb[1], rgb[2], a
        return default

    def ramp(x, y): return sample("ramp", g.vec(x, y, 0), True)
    def step(x): return g.cubic(g.clamp((x-.25)*1.333333))

    if data["family"] == "endfield":
        metal, specular, ao, smooth = surface((p["metallic"], p["specularLevel"], 1., p["smoothness"]))
        transform = g.node("ShaderNodeVectorTransform")
        transform.vector_type, transform.convert_from, transform.convert_to = "VECTOR", "CAMERA", "WORLD"
        transform.inputs[0].default_value = (0, 0, 1)
        forward = Value(g, transform.outputs[0], True)
        forward = g.normalize(g.vec(forward[0], forward[2], -forward[1]))
        ndl = g.clamp(nl-.4, -1, 1)
        view_dot = g.dot(normal, forward)
        if data.get("ramp"):
            ramp_rgb, ramp_a = ramp(.5+.5*ndl, .5)
            _, view_a = ramp(.5+.5*view_dot, .5)
        else:
            ramp_rgb, ramp_a, view_a = g.vec(1), step(ndl), step(view_dot)
        kd = .96*(1-metal)
        albedo = base*kd
        if data.get("shadowLut"):
            encoded = []
            for i in range(3):
                c = base[i]
                high = 1.055*(g.max(c, 0)**.416667)-.055
                encoded.append(g.clamp(g.mix(c*12.92, high, 1-g.op("LESS_THAN", c, .0031308))))
            r, green, blue = encoded
            index = g.op("FLOOR", blue*31)
            lut_uv = g.vec(index/32+r*31/1024+.5/1024, green*31/32+.5/32, 0)
            lo, _ = sample("shadowLut", lut_uv, True)
            hi, _ = sample("shadowLut", lut_uv+g.vec(1/32, 0, 0), True)
            shadow = g.mix(lo, hi, blue*31-index)*kd*.8
        else:
            shadow = g.sat(base*p["shadowBrightness"], p["shadowSaturation"])*kd*.8
        lit = g.min(ramp_a, ao)
        deep = g.sat(shadow*.65, 1.2)
        c1 = g.mix(g.mix(deep, shadow, g.clamp(view_a*ao+ramp_a)), albedo, lit)
        chroma = g.max(ramp_rgb[0], g.max(ramp_rgb[1], ramp_rgb[2]))-g.min(ramp_rgb[0], g.min(ramp_rgb[1], ramp_rgb[2]))
        c2 = c1*g.mix(g.vec(1), ramp_rgb, chroma)
        diffuse = c2*g.clamp(g.luma(c1)/g.max(g.luma(c2), .001), 0, 1.5)
        grad = g.clamp(normal[1]+.15)*1.5+.5
        lighting = (g.mix(g.vec(.7830188, .8293082, 1), g.vec(1), lit)*grad+g.vec(.9))*.45
        half = g.normalize(light+2*g.normalize(g.vec(forward[0], light[1], forward[2]))+3*view)
        rough = g.max((1-smooth)*(1-smooth), .0078125)
        a2, nh = rough*rough, g.dot(normal, half)
        d = (nh*a2-nh)*nh+1
        ggx = g.clamp(a2/g.max(d*d, 1e-12)*.5/(2*nv+rough+.0001)-.000061, 0, 20)
        f0 = g.mix(g.vec(.04*specular), base, metal)
        color = diffuse*lighting+f0*ggx*lighting*(.5+.5*lit)*g.mix(.8, 1., lit)
        k = g.clamp(g.luma(color)-.5, 0, .5)
        color = g.sat(color, 1+k*k)
    elif data["family"] == "gf2":
        rough, metal, _ao, emission_mask = surface((1-p["smoothness"], p["metallic"], 1., 0.))
        rough = g.max(rough, .001)
        f0 = g.mix(g.vec(.04), base, metal)
        lambert = g.max(nl, 0)
        x = g.max(lambert*(3*.333299994468689), .00006103515625)
        diffuse_ramp = g.vec(lambert)
        if data.get("ramp"):
            diffuse_ramp, _ = ramp(x, .125)
            diffuse_ramp *= g.mix(1., g.min(lambert/x, 1), g.op("GREATER_THAN", x, .00006103515625))
        half = g.normalize(view+light)
        nh, vh, lh = (g.max(g.dot(a, half), 0) for a in (normal, view, light))
        nv = g.min(nv+.00001, 1)
        a = rough*rough
        d = g.min((a/(nh*nh*a*a+1-nh*nh))**2, 2048)
        vis = g.min(.5/g.max(lambert*(nv*(1-a)+a)+nv*(lambert*(1-a)+a), .0001), 1)
        fres = 1+(g.clamp(50*f0[1])-1)*(g.max(1-vh, .001)**5)
        spec = g.vec(d*vis*fres)
        if data.get("ramp"):
            peak_d = g.min((a/(a*a))**2, 2048)
            peak_v = g.min(.5/g.max(vh*(lh*(1-a)+a)+lh*(vh*(1-a)+a), .0001), 1)
            spec_ramp, _ = ramp(g.clamp(d*vis/(peak_d*peak_v)), .375)
            spec = peak_d*peak_v*fres*spec_ramp
        color = (base*(1-metal)+f0*g.clamp(spec, 0, 10))*diffuse_ramp+base*p["emission"]*emission_mask
    else:
        view_light = .5*(g.dot(light, view)*(1-g.op("ABSOLUTE", view[1]))+1)
        threshold = (1-g.clamp(g.dot(normal, view)))*g.mix(p["backShadowOffset"], p["frontShadowOffset"], view_light)+p["shadowProcess"]
        width = g.max(p["shadowWidth"], 1e-8)
        q = g.cubic(g.clamp((.5*nl+.5-threshold+width)/(2*width)))
        upper = g.cubic(q)
        u = g.clamp(1-p["smoothness"])**.1
        exponent, scale = 1000-999*u, .1+19.9*(1-u)
        metal = 1-g.op("LESS_THAN", p["metallic"], .85)
        power = g.mix(p["specularPower"], p["metalSpecularPower"], metal)
        flip = g.op("LESS_THAN", q, .4)*(1-metal)
        mirrored = g.vec(light[0]*(1-2*flip), light[1], light[2]*(1-2*flip))
        ndh = g.clamp(g.dot(normal, g.normalize(view+mirrored)))
        lobe = g.clamp(((ndh**exponent)*power*(power+2)/(2*math.pi)-.6)*(4/3))
        lobe = g.cubic(lobe)*scale
        spec_color = g.mix(g.vec(.04+.04*p["metallic"]), base, p["metallic"])
        material_color = base*(1-p["metallic"]*p["pbrRate"])+spec_color*p["pbrRate"]*lobe
        color = material_color*g.mix(g.vec(.18), g.vec(1), upper)

    emission = g.node("ShaderNodeEmission", "Source character lighting")
    g.link(color, emission.inputs["Color"])
    output = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    output = output if output is not None else g.node("ShaderNodeOutputMaterial")
    shader = emission.outputs[0]
    if data["alphaMode"] != "OPAQUE":
        if data["alphaMode"] == "MASK":
            alpha = 1-g.op("LESS_THAN", alpha, data["alphaCutoff"])
        transparent = g.node("ShaderNodeBsdfTransparent")
        mix = g.node("ShaderNodeMixShader")
        g.link(alpha, mix.inputs[0])
        tree.links.new(transparent.outputs[0], mix.inputs[1])
        tree.links.new(shader, mix.inputs[2])
        shader = mix.outputs[0]
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
    tree.links.new(shader, output.inputs["Surface"])
    material.use_backface_culling = not data.get("doubleSided", False)
    material["polylode_character_style"] = data["family"]


def build(bundle, output):
    import bpy
    from mathutils import Quaternion, Vector
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = json.loads((bundle / "materials.json").read_text(encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(bundle / manifest["model"]))
    descriptors = {row["id"]: row for row in manifest["materials"]}
    styled = []
    for material in bpy.data.materials:
        index = material.get("polylode_material_index")
        if index not in descriptors:
            continue
        data = descriptors[index]
        if data["family"] and not data.get("preserveOriginal"):
            apply_material(material, data, bundle)
            styled.append(index)
    expected = {row["id"] for row in manifest["materials"] if row["family"] and not row.get("preserveOriginal")}
    if expected - set(styled):
        raise ValueError(f"Blender did not import shader material IDs {sorted(expected-set(styled))}")
    bpy.ops.file.pack_all()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.world = bpy.data.worlds.new("Character preview world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.015, .018, .025, 1)
    scene.render.resolution_x, scene.render.resolution_y = 512, 768
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_objects = [o for o in scene.objects if o.type == "MESH"]
    evaluated = [o.evaluated_get(depsgraph) for o in mesh_objects]
    points = [o.matrix_world @ Vector(corner) for o in evaluated for corner in o.bound_box]
    if points:
        low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
        high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
        center, half = (low+high)/2, (high-low)/2
        camera = bpy.data.objects.new("Character preview camera", bpy.data.cameras.new("Character preview camera"))
        camera.data.sensor_fit = "HORIZONTAL"
        scene.collection.objects.link(camera)
        tangent = math.tan(camera.data.angle_x/2)
        aspect = scene.render.resolution_x/scene.render.resolution_y
        distance = max(half.x/tangent, half.z*aspect/tangent, .01)*1.12+half.y
        camera.location = center+Vector((0, -distance, 0))
        camera.rotation_euler = (center-camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.spaces.active.shading.type = "MATERIAL"
                    region = area.spaces.active.region_3d
                    region.view_location = center
                    region.view_distance = distance
                    region.view_rotation = Quaternion((1, 0, 0), math.pi/2)
    scene["polylode_shader_materials"] = sorted(set(styled))
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    bpy.ops.wm.open_mainfile(filepath=str(output))
    if set(bpy.context.scene["polylode_shader_materials"]) != expected:
        raise ValueError("saved Blender material IDs changed")
    for image in bpy.data.images:
        if image.source == "FILE" and image.size[0] and not image.packed_file:
            raise ValueError(f"unpacked image {image.name}")
    return {"materials": len(expected), "images": len(bpy.data.images), "output": str(output)}


if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    folder = Path(__file__).resolve().parent
    destination = Path(args[0]) if args else folder.with_suffix(".shaded.blend")
    print(json.dumps(build(folder, destination), ensure_ascii=False))
