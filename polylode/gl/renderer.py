"""Forward cel-shading renderer: sky, outline pass, opaque, transparent, post."""
from __future__ import annotations

import math
import time
import weakref
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from OpenGL import GL

from ..converted.clock import FrameClock
from ..converted.scene import (
    NativeItem,
    NativeStage,
    PreparedScene,
    SceneTarget,
    SceneView,
)
from ..core import math3d as m3
from ..core import swallowed
from ..mesh import skinning
from ..mesh.mesh import Material, Model, Part
from ..origin import colorspace as CS
from ..origin.figure import toon as TOON
from . import shaders
from . import shaders_endfield as EF
from .camera import Camera
from .framebuffer import Framebuffer
from .mesh_gpu import GPUMesh, LineBatch, make_grid
from .program import Program
from .textures import TEXTURE_FACTORY, TextureCache, resolve_texture


def _lin3(c) -> tuple:
    """A colour uniform as the GPU sees it in a Linear project.

    Every colour input of ``main_skin`` / ``main_opaque`` / ``main_hair`` is
    declared ``Color`` in the shader's property table and **none** carries
    ``[Gamma]`` (checked per shader with ``origin_shader_dump --props``), so Unity
    converts it on the way in — the same rule §7.85 established for the six
    create passes, now that the forward half runs in linear too (§7.126).

    It covers the **material's own** colours (``_ShadowColor``,
    ``_SpecularColor``, ``_Color``…) and ``_LightColor0``, and deliberately
    **not** the four ``…G`` globals: those are not declared properties, so the
    rule has no declaration to read — see ``AMBIENT_SHADOW_CONVERTED``.

    Our own procedural colours — sky, grid, fog, rim — go through it as well,
    for a different reason: the scene target encodes on write, so a value handed
    over unconverted comes out brighter than it was authored.
    """
    return _linear(_colour_key(c))[:3]


def _lin4(c) -> tuple:
    return _linear(_colour_key(c))


def _colour_key(c) -> tuple:
    if not hasattr(c, "__len__"):
        return (float(c),)
    return tuple(float(x) for x in c)


@lru_cache(maxsize=4096)
def _linear(key: tuple) -> tuple:
    """``colour_to_linear`` of one colour, remembered.

    The renderer converts every material colour on every draw, and they are
    constants: 269 numpy conversions a frame for one character, the largest
    share of ``_material_uniforms`` (§7.552).  A tuple of floats in, the same
    four floats out every time.
    """
    v = CS.colour_to_linear(key)
    return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))


@dataclass
class LightSettings:
    direction: tuple = (-0.35, -0.48, -0.80)
    color: tuple = (1.0, 0.98, 0.95)
    intensity: float = 0.84
    ambient: tuple = (0.17, 0.17, 0.21)
    rim_direction: tuple = (0.5, 0.25, 0.8)
    fog: bool = False
    fog_color: tuple = (0.75, 0.80, 0.88)
    fog_range: tuple = (6.0, 30.0)
    # Origin lights characters with a *separate* light, and every scene it ships
    # authors that light's angle itself -- see ORIGIN_CHARA_LIGHT.  Only the
    # direction comes from there; colour and intensity stay the scene's, so
    # dusk still tints people.
    chara_light: bool = True
    chara_angle: tuple = (0.0, 0.0)
    # True: ``chara_angle`` is (X, Y) degrees off the camera axis, the light
    # being a child of the camera.  False: it is a world-space Unity
    # ``Euler(X, Y, 0)`` and the camera does not move it (CharaStudio).
    chara_from_camera: bool = True
    # ``CorrectLightAngle``: when set, the light is re-aimed every frame off
    # the character's *head*, and ``chara_angle`` is only the fallback.  The
    # pair is the component's (correctRX, correctRY).  See ORIGIN_CHARA_LIGHT.
    chara_track_head: tuple = ()
    # ``_LightColor0`` for the character shaders -- this light's own colour and
    # intensity, not the scene light's.  Origin ships white at 1.0 in every scene;
    # a screen that wants people to read as dusk dims *this*, so it is doing so
    # on purpose rather than as a side effect of lighting the ground.
    chara_color: tuple = (1.0, 1.0, 1.0)
    chara_intensity: float = 1.0
    # The lamp Endfield's own programs are lit by, whatever lamp the screen's
    # card characters stand under: a name in ``EF_CHARA_LIGHT``, which also
    # picks that game scene's volume (``shaders_endfield.VOLUMES``).  The two
    # games light their characters apart, each by its own.
    ef_lamp: str = "ef_charinfo"
    # ``shadows=soft`` with a 1024 cull is what Origin's life-scene character light
    # ships, and its ``main_*`` forward bases carry the variant that reads the
    # resulting map (§7.341/§7.343).
    #
    # **Off, and that is a measurement, not a shrug** (§7.343).  The mechanism
    # is transcribed and correct -- forced 170 degrees off the camera the
    # figure goes into shadow exactly where it should -- but *at the angles Origin
    # actually ships* the character light sits within 40 degrees of the camera
    # (``ORIGIN_CHARA_LIGHT``: maker 0, life 20, config 20/25, live 12), and a
    # light on the view axis casts nothing the view can see.  Measured on the
    # test card at the ``life`` angle: **88 pixels** move by more than 8 levels
    # and all nine of ``reference/patches.json``'s samples are bit-for-bit
    # unchanged, while the pass costs **20% of the frame rate** at one
    # character and at three (A/B alternating twice each, 六之六).
    #
    # 一之二 governs the *mechanism*; what to draw and what the default is, is
    # a product call (§7.135) -- and this one has a number on both sides.
    chara_shadow: bool = False
    chara_shadow_size: int = 1024

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    def from_dict(self, d):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, tuple(v) if isinstance(v, list) else v)
        return self


@dataclass
class PostSettings:
    # ``enabled`` switches off **this project's** grade — bloom, vignette,
    # saturation, contrast, DOF, the lot.  It does **not** switch off ``lut``:
    # that one is Illusion's own ``AmplifyColorEffect``, transcribed in §7.40,
    # and it is part of what the game's picture *is*, not of what we added on
    # top.  Turning the two off together is what made every saturated garment
    # read 40-50 steps bright against ``reference/`` and sent §7.52 looking for
    # a shading bug that was not there (the sailor collar: +49.9 with the LUT
    # off, -6.7 with it on).
    # Off unless the player turns it on (設定 → 啟用後製效果, owner 2026-10-16): the game's own
    # picture is the LUT alone, and these four are a look of ours on top of the whole game.
    enabled: bool = False
    bloom: float = 0.22
    bloom_threshold: float = 0.72
    bloom_lod: float = 3.0          # mip level the bloom blur samples from
    vignette: float = 0.30
    saturation: float = 1.06
    contrast: float = 1.03
    brightness: float = 0.0
    tint: tuple = (1.0, 1.0, 1.0)
    dof_strength: float = 0.0
    dof_focus: float = 2.0
    chromatic: float = 0.0
    sepia: bool = False
    # Illusion's own colour grade (``AmplifyColorEffect`` -> ``Lut_Amount``).
    # Origin runs it at 1.0 on every camera that has it; 0 disables, and it is
    # forced off anyway when Koikatsu is not installed -- the LUT is read from
    # the player's install, never shipped.  See origin/grading.py.
    lut: float = 1.0

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    def from_dict(self, d):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, tuple(v) if isinstance(v, list) else v)
        return self


@dataclass
class Background:
    mode: str = "gradient"      # gradient | solid | transparent | image
    top: tuple = (0.62, 0.72, 0.88)
    bottom: tuple = (0.93, 0.90, 0.92)
    power: float = 0.85
    grid: bool = True
    grid_y: float = 0.0
    # Full-screen image backdrop (Origin studio thumb / UserData/bg).  Path only —
    # the renderer loads it lazily into the texture cache.
    image_path: str = ""
    image_dim: float = 0.08     # darken slightly so characters stay readable

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    def from_dict(self, d):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, tuple(v) if isinstance(v, list) else v)
        return self


@dataclass
class DrawItem:
    model: Model
    transform: np.ndarray = field(default_factory=m3.identity)
    skin: np.ndarray | None = None      # (nbones, 4, 4) skinning matrices
    visible: bool = True
    outline_scale: float = 1.0
    # World rotation of ``cf_s_head``, for items that are characters.  This is
    # what Origin's CorrectLightAngle aims the character light off; ``None`` marks
    # everything that is not a person (maps, props), which is also how the
    # "exactly one character" test is counted.
    head_rot: np.ndarray | None = None


@dataclass
class RenderScene:
    camera: Camera = field(default_factory=Camera)
    items: list = field(default_factory=list)
    light: LightSettings = field(default_factory=LightSettings)
    post: PostSettings = field(default_factory=PostSettings)
    background: Background = field(default_factory=Background)
    overlay_lines: tuple | None = None    # (verts, colors)
    show_outline: bool = True
    # the player's 顯示地面格線, over whatever grid a screen's backdrop asks for
    show_grid: bool = True
    # The Origin camera this screen stands for carries ``CameraEffectorConfig``
    # (``ActionCamera`` in level16, the H camera in level18), so the player's
    # グラフィック switches reach it; CustomScene and CharaStudio have none and
    # ignore them (§7.554).
    camera_effector: bool = False
    native_items: list[NativeItem] = field(default_factory=list)
    clock: FrameClock = field(default_factory=FrameClock)


#: Every camera effect on: what a camera without ``CameraEffectorConfig`` runs.
_EFFECTS_ON = {k: True for k in TOON.CAMERA_EFFECTS}


def scene_effects(scene: "RenderScene") -> dict:
    """``{effect: on}`` for *scene*: the player's switches where Origin applies them."""
    return TOON.camera_effects() if scene.camera_effector else _EFFECTS_ON


# Clip-space nudge for the outline hull; see Renderer._draw_outlines.
OUTLINE_Z_BIAS = 0.0022


def _group_by_item(pairs) -> list:
    """``[(item, part), …]`` -> ``[(item, [part, …]), …]``, order preserved."""
    out: list = []
    for it, part in pairs:
        if out and out[-1][0] is it:
            out[-1][1].append(part)
        else:
            out.append((it, [part]))
    return out

# Texture units, in one place because they are shared between things that are
# bound at different rates.  A **material** rebinds 0..6 and 8..10 on every part:
#
#     0 base map   1 pattern (main) / shade mask (outline)   2 ramp
#     3 normal     4 shade mask     5 view ramp              6 gloss
#     8 detail mask   9 highlight   10 glass ramp
#
# The **skin** is bound once per item and has to survive all of that, so it
# gets a unit no material touches.  It used to share unit 1 with the pattern,
# which meant every part after the first read its bone matrices out of a white
# texture — see Renderer._bind_skin.  GL 3.3 guarantees at least 16 combined
# units, so this is not tight.
BONE_TEX_UNIT = 7
#: The character light's depth map, bound once per scene (``_setup_toon``).  It
#: sat on 9 until the iris highlight took 9 per material (§7.403): with the
#: shadow on, its sampler read the highlight sheet or white (§7.553).
SHADOW_TEX_UNIT = 11
#: The inverse of the frame's colour grade, for the programs it is not meant
#: for (``shaders_endfield.efUngrade``); per scene, on a unit no material binds.
UNGRADE_TEX_UNIT = 12
#: ``(unit, material field, use flag, sampler)`` a material binds past the four
#: fixed ones (0 base, 1 pattern, 2 ramp, 3 normal).
MATERIAL_TEX_UNITS = (
    (4, "shade_mask", "uUseShadeMask", "uShadeMask"),
    (5, "view_ramp", "uUseViewRamp", "uViewRamp"),
    (6, "gloss", "uUseGloss", "uGloss"),
    (8, "detail_mask", "uUseDetail", "uDetailMask"),
    (9, "highlight", "uUseHighlight", "uHighlight"),
    (10, "glass_ramp", "uUseGlassRamp", "uGlassRamp"),
    # ``main_skin``'s liquid layer (``origin.figure.siru``): the three maps of the material's own
    (13, "liquid_mask", "uUseLiquidMask", "uLiquidMask"),
    (14, "liquid_drop", "uUseLiquidDrop", "uLiquidDrop"),
    (15, "liquid_normal", "uUseLiquidNormal", "uLiquidNormal"),
)


#: the liquid layer's units (bound only on a material with ``liquid_mask``)
_LIQUID_UNITS = frozenset(unit for unit, attr, _f, _n in MATERIAL_TEX_UNITS if attr.startswith("liquid_"))
#: the table's units, each read behind its own ``uUse*`` switch by the card game's programs
_FLAGGED_UNITS = frozenset(unit for unit, _attr, _f, _n in MATERIAL_TEX_UNITS)
#: Every map a material binds, by unit: the four fixed ones, then the table.
_MATERIAL_MAPS = ((0, "texture"), (1, "pattern"), (2, "ramp"), (3, "normal_map"),
                  *((unit, attr) for unit, attr, _flag, _name in MATERIAL_TEX_UNITS))


_IDENTITY = np.eye(4, dtype=np.float32)
#: ``uSkinDual`` for each ``Part.skinning`` (``gl/shaders.py``)
_SKIN_DUAL = {skinning.LINEAR: 0, skinning.DUAL_QUATERNION: 1, skinning.MIXED: 2}
#: ``uEfScene``: this renderer's stand-ins for the game's per-frame settings.
_EF_SCENE = (EF.SCENE["direct"], EF.SCENE["shadow_intensity"],
             EF.SCENE["ambient"], EF.SCENE["specular"])
_STENCIL_FUNCS = {"always": GL.GL_ALWAYS, "equal": GL.GL_EQUAL, "gequal": GL.GL_GEQUAL}


def _stencil_begin(state: tuple) -> None:
    """Test and stamp as ``Material.stencil_state`` says, for one draw."""
    ref, compare, read, write = state
    GL.glEnable(GL.GL_STENCIL_TEST)
    GL.glStencilFunc(_STENCIL_FUNCS[compare], ref, read)
    GL.glStencilOp(GL.GL_KEEP, GL.GL_KEEP, GL.GL_REPLACE if write else GL.GL_KEEP)
    GL.glStencilMask(write)


def _stencil_end() -> None:
    GL.glStencilMask(0xFF)
    GL.glStencilFunc(GL.GL_ALWAYS, 0, 0xFF)
    GL.glDisable(GL.GL_STENCIL_TEST)


def ef_lamp(light: LightSettings) -> tuple:
    """``(lamp entry, volume)`` Endfield's programs are lit by under *light*.

    The entry is ``EF_CHARA_LIGHT``'s (angle, from the camera, head tracking,
    colour, intensity); a lamp without a volume of its own draws by the
    stand-ins alone.
    """
    return EF_CHARA_LIGHT[light.ef_lamp], EF.VOLUMES.get(light.ef_lamp, EF.NO_VOLUME)


def ef_light_dir(scene: "RenderScene") -> tuple:
    """The direction Endfield's lamp travels in *scene*.

    A scene whose character light is switched off for a fixed one (a tool
    looking at where a shadow falls) lights everything by that.
    """
    li = scene.light
    if not li.chara_light:
        return li.direction
    angle, from_camera, _track, _colour, _intensity = EF_CHARA_LIGHT[li.ef_lamp]
    return chara_light_dir(scene.camera, angle, from_camera)


def ef_lit_light(light: LightSettings) -> tuple:
    """What a fully lit, unshadowed ``HGRP/CharacterNPR*`` surface is lit by.

    ``efLighting`` at ``lit = 1``, ``S = 1``: the outline pass uses it, since
    it has no surface of its own to light.
    """
    (_angle, _camera, _track, colour, intensity), volume = ef_lamp(light)
    return tuple((1.0 + c * float(intensity)) * EF.SCENE["direct"] * volume.main_light
                 for c in _lin3(colour))


def _material_values(mat: Material) -> list:
    """``[(kind, uniform, args)]`` for one material: what the toon pass reads.

    *kind* picks the ``glUniform*`` (``i``, ``f``, ``v2``, ``v3``, ``v4``) and
    *args* is already in the form :class:`Program` records as sent, so a
    value set here and one set through ``Program.set_*`` compare equal.  A
    function of the material and two module constants only -- which is what
    lets ``Renderer._material_entries`` keep it on the material (§7.555).
    """
    if mat.normal_encoding not in (0, 1):
        raise ValueError(f'unsupported normal encoding: {mat.normal_encoding}')
    if mat.normal_encoding == 1 and mat.toon_family != 0:
        raise ValueError('GF2 normal component requires the procedural lighting preview')

    def f(name, v):
        return ("f", name, (float(v),))

    def i(name, v):
        return ("i", name, (int(v),))

    def v2(name, v):
        return ("v2", name, (float(v[0]), float(v[1])))

    def v3(name, v):
        return ("v3", name, (float(v[0]), float(v[1]), float(v[2])))

    def v4(name, v):
        return ("v4", name, (float(v[0]), float(v[1]), float(v[2]),
                             float(v[3]) if len(v) > 3 else 1.0))

    g = lambda attr, default: getattr(mat, attr, default)   # noqa: E731
    out = [
        v4("uBaseColor", _lin4(mat.base_color)),
        v4("uShadowColor", _lin4(mat.shadow_color)),
        v4("uSpecColor", _lin4(mat.specular_color)),
        f("uSpecPower", mat.specular_power),
        f("uSpecStrength", mat.specular_strength),
        v4("uRimColor", _lin4(mat.rim_color)),
        f("uRimStrength", mat.rim_strength),
        f("uRimPower", mat.rim_power),
        # Illusion's specular block; zero on every procedural material, which
        # is what keeps its lobe out of this project's own model.
        f("uOriginSpecular", g("origin_specular", 0.0)),
        f("uSpecNail", g("spec_nail", 0.0)),
        f("uNoTexSpec", g("no_tex_spec", 0.0)),
        f("uRimV", g("rim_v", 0.0)),
        f("uShadeSteps", mat.shade_steps),
        f("uShadeFeather", mat.shade_feather),
        f("uShadeOffset", mat.shade_offset),
        f("uUnlit", mat.unlit),
        f("uAlphaTest", mat.alpha_test),
        f("uDitherCut", g("dither_cut", -1.0)),
        v2("uUVOffset", g("uv_offset", (0.0, 0.0))),
        v4("uHighlightXf", g("highlight_xf", (1.0, 1.0, 0.0, 0.0))),
        f("uHighlightOn", g("highlight_on", 1.0)),
        f("uSpecThreshold", g("specular_threshold", 0.0)),
        f("uNormalScale", g("normal_scale", 1.0)),
        i("uNormalEncoding", mat.normal_encoding),
        i("uToonFamily", g("toon_family", 0)),
        # Already in the shader's space — see ``AMBIENT_SHADOW_IS_GAMMA``.
        v3("uAmbientBase", _AMBIENT_BASE),
        v3("uAmbientFloor", _AMBIENT_FLOOR),
        v4("uAmbientShadow", (*_AMBIENT_FLOOR, _AMBIENT_SHADOW_A)),
        # toon_glasses_lod0: Color properties, converted like every other
        # material colour (the alpha is not a colour and stays as authored)
        v4("uGlassColor", _lin4(g("glass_color", (1.0,) * 4))),
        v4("uGlassColor4", _lin4(g("glass_color4", (1.0,) * 4))),
        v3("uGlassParams", tuple(g("glass_params", (0.0, 0.0, 1.0)))),
        v4("uGlassRampXf", tuple(g("glass_ramp_xf", (1.0, 1.0, 0.0, 0.0)))),
        v4("uViewRampXf", g("view_ramp_transform", (1.0, 1.0, 0.0, 0.0))),
        v4("uGlossXf", g("gloss_transform", (1.0, 1.0, 0.0, 0.0))),
        v4("uNormalXf", g("normal_transform", (1.0, 1.0, 0.0, 0.0))),
        f("uGlossHeight", g("gloss_height", 1.0)),
        # the liquid layer's numbers: levels, transforms and look are the shader's own space already
        v4("uLiquidLevels", tuple(g("liquid_levels", (0.0,) * 4))),
        f("uLiquidFace", g("liquid_face", 0.0)),
        v4("uLiquidTiling", g("liquid_tiling", (1.0, 1.0, 0.0, 0.0))),
        v4("uLiquidMaskXf", g("liquid_mask_xf", (1.0, 1.0, 0.0, 0.0))),
        v4("uLiquidDropXf", g("liquid_drop_xf", (1.0, 1.0, 0.0, 0.0))),
        v4("uLiquidNormalXf", g("liquid_normal_xf", (1.0, 1.0, 0.0, 0.0))),
        v3("uLiquidThin", tuple(g("liquid_look", (0.5, 0.397059, 0.242647, 0.85, 0.85, 0.85, 1.0))[:3])),
        v3("uLiquidFull", tuple(g("liquid_look", (0.5, 0.397059, 0.242647, 0.85, 0.85, 0.85, 1.0))[3:6])),
        f("uLiquidOpacity", g("liquid_look", (0.0,) * 6 + (1.0,))[6]),
        f("uLiquidRimPower", g("liquid_rim_power", 0.0)),
        i("uLiquidSolid", 1 if g("liquid_solid", False) else 0),
        f("uShadeFloor", g("shade_floor", 0.0)),
        f("uLitOverride", LIT_OVERRIDE),
        i("uUseTex", 1 if mat.texture else 0), i("uTex", 0),
        i("uUsePattern", 1 if mat.pattern else 0), i("uPattern", 1),
        i("uUseRamp", 1 if g("ramp", None) else 0), i("uRamp", 2),
        i("uUseNormal", 1 if g("normal_map", None) else 0), i("uNormalTex", 3),
    ]
    if mat.pattern:
        # only read while a pattern is bound; left as they were otherwise
        out += [v4("uPatternColor", _lin4(mat.pattern_color)),
                v4("uPatternXf", mat.pattern_transform)]
    for unit, attr, flag, name in MATERIAL_TEX_UNITS:
        out += [i(flag, 1 if g(attr, None) else 0), i(name, unit)]
    # a family with inputs of its own (``toon_family`` ≥ 20) names them itself
    out += list(g("extra_uniforms", ()))
    return out


def _model_map_keys(model) -> set:
    """Every cache key a draw of *model* will bind that ``resolve_texture`` can upload.

    GF2 source keys use their strict resolver. A bare int, ``None`` and other
    keys without a factory resolve to pinned white and are not waited on.
    """
    keys = set()
    for part in getattr(model, "parts", ()) or ():
        mat = getattr(part, "material", None)
        if mat is None:
            continue
        for _unit, attr in _MATERIAL_MAPS:
            key = getattr(mat, attr, None)
            if isinstance(key, str) and key:
                key = (key,)
            elif isinstance(key, (tuple, list)) and key:
                key = tuple(key)
            else:
                continue
            if key[0] in TEXTURE_FACTORY or key[0] in ('gf2map', 'gf2source'):
                keys.add(key)
    return keys


# Illusion's character light is a plain directional light parented into each
# scene, and **the scene author sets its angle** -- there is no single "Origin
# character light direction".  Read straight off the shipped scenes with
# ``tools/origin_scene_dump.py``; ``angle`` is the light transform's local
# rotation, and ``from_camera`` says whether its parent is the camera.
#
#   scene                          asset                     angle    parent
#   CustomScene (chara maker)      Koikatu level2            (0, 0)   Camera
#   ActionScene (life mode)        Koikatu level16           (20, 0)  ActionCamera
#   HSceneProc                     Koikatu level18           (0, 0)   Camera
#   ConfigScene                    Koikatu level9            (20, 25) Camera
#   LiveStage                      Koikatu level24           (12, 0)  Camera
#   StudioScene                    CharaStudio level1        (40, 180) StudioScene
#
# The maker's 0 is why the card thumbnail -- which is shot in CustomScene --
# has no diffuse terminator anywhere on the front of the body, and why §7.35
# read the game as "always a head light".  Studio's is the odd one out: its
# light hangs off the scene root, so the camera does not carry it and its
# angle is a world-space Unity Euler.
#
# ``track_head`` is the second half of the story, and it changes what the
# angle above even *means*.  Two of these scenes carry a ``CorrectLightAngle``
# component, which in ``LateUpdate`` does:
#
#     lightTrans.rotation = <chara's cf_s_head>.rotation
#     lightTrans.Rotate(correctRX + offset.x, correctRY + offset.y, 0)
#
# so the light is aimed off the **character's head**, not the camera, and the
# authored local rotation survives only as ``initRot`` -- the fallback taken
# when no single character is selectable.  Which of the two scenes actually
# runs it is decided by the component's enabled flag in the scene, not by the
# code: ActionScene's is **enabled** (correctRX 5, correctRY 190), and
# CharaStudio's is **disabled**, which is why Studio really does keep its
# world-fixed 40/180.  ``offset`` is only ever written by ADV commands, so it
# is zero outside a conversation.
#
# ``colour`` and ``intensity`` are the same light's, and they matter as much as
# the angle: the character shaders' final term is
# ``max(_LightColor0 * 0.6 + 0.4, _ambientshadowG)``, and ``_LightColor0`` is
# this light's ``color * intensity``.  Every shipped scene authors it at pure
# white, intensity 1 (LiveStage is the only one that is not, at 0.98), so that
# term is 1.0 and a fully-lit face comes out at its albedo -- which is exactly
# what ``reference/`` shows (cheek 255,232,224 against a 255,233,224 map).
# Feeding the *scene* light in here instead cost 10% of every character's
# brightness plus its warm tint, and that was the whole of §八 二 5's
# "skin is 46.7 steps darker".  The scene light still lights the map.
ORIGIN_CHARA_LIGHT = {
    "maker": ((0.0, 0.0), True, (), (1.0, 1.0, 1.0), 1.0),
    "life": ((20.0, 0.0), True, (5.0, 190.0), (1.0, 1.0, 1.0), 1.0),
    "h": ((0.0, 0.0), True, (), (1.0, 1.0, 1.0), 1.0),
    "config": ((20.0, 25.0), True, (), (1.0, 1.0, 1.0), 1.0),
    "live": ((12.0, 0.0), True, (), (1.0, 1.0, 1.0), 0.98),
    "studio": ((40.0, 180.0), False, (), (1.0, 1.0, 1.0), 1.0),
}


#: The character light of Endfield's own screens (``LightSettings.ef_lamp``:
#: what its programs are lit by on every screen), in the same shape, read from
#: the game's ``HGCharacterVolume`` (the pipeline's volume component for how
#: characters are lit; ``docs/updates`` §7.916).  ``ef_charinfo`` is the
#: character-information screen (``prefabs/charinfo/charinfo_volume.asset``):
#: ``charMainLightMode`` 1 — the main light follows the camera —
#: ``charCameraFollowMainLightBias`` (32°, 10°), ``charMainLightOverrideColor``
#: white, ``charMainLightMultiplier`` 0.9.
EF_CHARA_LIGHT = {
    "ef_charinfo": ((32.0, 10.0), True, (), (1.0, 1.0, 1.0), 0.9),
}


def apply_chara_light(light: LightSettings, scene: str) -> LightSettings:
    """Point *light*'s character light at whatever Origin's matching scene uses.

    Each of our screens stands in for one of Illusion's: the editor for
    CustomScene, life mode for ActionScene, Free H for HSceneProc.  Naming the
    scene keeps the number in :data:`ORIGIN_CHARA_LIGHT` and out of the UI, so a
    re-measure lands everywhere at once.  All three fields are written every
    time so switching screens cannot leave one behind.
    """
    entry = ORIGIN_CHARA_LIGHT.get(scene, ORIGIN_CHARA_LIGHT["life"])
    angle, from_camera, track, colour, intensity = entry
    light.chara_angle = angle
    light.chara_from_camera = from_camera
    light.chara_track_head = track
    light.chara_color = colour
    light.chara_intensity = intensity
    return light


def apply_editor_stage(scene: "RenderScene") -> "RenderScene":
    """Give *scene* the character editor's stage: its backdrop with the floor grid, its
    character light, and every camera effect (CustomScene's camera carries no
    ``CameraEffectorConfig``, so the player's switches do not reach it).

    The one owner of how the editor looks, for the screens that show a
    character as the editor does (the editor itself, Free H).
    """
    scene.background = Background()
    apply_chara_light(scene.light, "maker")
    scene.camera_effector = False
    return scene


def h_light_angle(vertical: float, horizontal: float, scene: str = "h") -> tuple:
    """The H screen's light sliders → ``chara_angle`` (§7.549).

    ``HSprite.OnValueVerticalChanged`` / ``OnValueHorizonChanged`` turn the
    light to ``Euler(initRot.x + vertical·360, initRot.y + horizontal·360, 0)``;
    both sliders run 0‥1 and start at 0, so the scene's own angle is the rest.
    """
    init = ORIGIN_CHARA_LIGHT.get(scene, ORIGIN_CHARA_LIGHT["h"])[0]
    return (float(init[0]) + 360.0 * float(vertical),
            float(init[1]) + 360.0 * float(horizontal))


def head_light_dir(head_rot, correct=(5.0, 190.0)) -> tuple:
    """``CorrectLightAngle``: the light's direction once it is aimed off a head.

    ``lightTrans.rotation = head.rotation`` then ``Rotate(x, y, 0)`` in self
    space, i.e. ``head.rotation * Euler(x, y, 0)``, and the direction is that
    rotation's forward.  With the shipped (5, 190) the light ends up travelling
    at roughly ``-head.forward``: coming *at* the face from about ten degrees
    to one side and five above.  That is the real reason the ``gt_ramp_01``
    step never lands on the face in game -- not a lucky camera angle, but a key
    light bolted to the head, so it holds wherever the camera goes.

    Mirroring note: our space is Unity's with X flipped, and a rotation carries
    across that mirror as ``Euler(X, Y, Z) -> Euler(X, -Y, -Z)`` with the
    composition order unchanged, which is the same conversion
    ``attach.card_move_locals`` uses for card transforms.  Forward is on the
    mirror axis, so it maps to itself.
    """
    q = m3.quat_from_euler(np.radians(float(correct[0])),
                           np.radians(-float(correct[1])), 0.0, order="YXZ")
    local = m3.quat_to_mat4(q)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    # Origin reads ``Transform.rotation``, which carries no scale -- and this bone
    # is the one the face shaper writes head size onto, so its matrix always
    # has some.  Normalising the basis is what keeps a wide-faced card from
    # tilting its own key light.
    basis = np.asarray(head_rot, dtype=np.float64)[:3, :3].astype(np.float64)
    lens = np.linalg.norm(basis, axis=0)
    if np.any(lens < 1e-9):
        return (0.0, -1.0, 0.0)
    fwd = (basis / lens) @ local
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return (0.0, -1.0, 0.0)
    return tuple(float(x) for x in fwd / n)


def scene_chara_light_dir(scene: "RenderScene") -> tuple:
    """The character-light direction for a whole scene, Origin's selection included.

    ``CorrectLightAngle`` only aims at a character when it can name exactly one
    (``dictEntryChara`` filtered to hiPoly, or an ADV-supplied condition);
    with none or several it writes ``initRot`` back instead.  Our equivalent of
    that count is how many draw items carry a head, so a two-hander or an empty
    map falls back to the authored camera-relative angle the same way.
    """
    li = scene.light
    if not li.chara_light:
        return li.direction
    if li.chara_track_head:
        heads = [it.head_rot for it in scene.items
                 if it.visible and it.head_rot is not None]
        if len(heads) == 1:
            return head_light_dir(heads[0], li.chara_track_head)
    return chara_light_dir(scene.camera, li.chara_angle, li.chara_from_camera)


def chara_light_dir(camera: Camera, angle=(0.0, 0.0),
                    from_camera: bool = True) -> tuple:
    """The direction Origin's character light travels, given the camera.

    ``_RampG`` (``gt_ramp_01``, 深いくっきり影) is a **binary step at N.L =
    0.5**, so where the terminator lands is not a matter of taste: a light much
    more than 60 degrees off the view axis puts that step across the middle of
    the face, and the mesh's own normal variation then breaks it into patches
    rather than one clean edge.  Measured on 帰国子女: the visible face's N.L
    averaged 0.504 under the fixed world light -- dead on the step.

    The game does not solve this in the shader, it aims a light of its own at
    the character; *angle* is how far off that light sits, and it is per scene
    (see :data:`ORIGIN_CHARA_LIGHT`).  With ``from_camera`` the camera carries the
    light and the angle tilts off the view axis; without it the angle is a
    world-space ``Quaternion.Euler(X, Y, 0)`` whose forward is the direction,
    which is how CharaStudio parents its own.
    """
    if not from_camera:
        # A world-space angle may need all three axes: Studio's own light is
        # (40, 180, 0), but once a scene's ``LightInfo.rot`` is folded in (see
        # scene_build) the product no longer has a zero Z.
        e = [float(a) for a in angle] + [0.0, 0.0, 0.0]
        # Unity Euler -> ours: X mirrored, so Y and Z negate (the same
        # conversion ``attach.card_move_locals`` applies to card transforms).
        q = m3.quat_from_euler(np.radians(e[0]), np.radians(-e[1]),
                               np.radians(-e[2]), order="YXZ")
        fwd = m3.quat_to_mat4(q)[:3, :3] @ np.array([0.0, 0.0, 1.0])
        n = np.linalg.norm(fwd)
        return tuple(float(x) for x in (fwd / n if n > 1e-9
                                        else np.array([0.0, -1.0, 0.0])))

    to_eye = np.asarray(camera.direction(), dtype=np.float64)
    n = np.linalg.norm(to_eye)
    to_eye = to_eye / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    ax = np.radians(float(angle[0])), np.radians(float(angle[1]))
    if ax[1]:                                        # Y: swing around world up
        c, s = np.cos(ax[1]), np.sin(ax[1])
        to_eye = np.array([c * to_eye[0] + s * to_eye[2], to_eye[1],
                           -s * to_eye[0] + c * to_eye[2]])
    if ax[0]:                                        # X: tilt about the camera's right
        right = np.cross(np.array([0.0, 1.0, 0.0]), to_eye)
        rn = np.linalg.norm(right)
        if rn > 1e-6:
            right /= rn
            # A positive Unity X euler tilts a transform's forward **down**
            # (``Euler(20,0,0)`` has forward ``(0, -sin20, cos20)``), and this
            # is rotating ``to_eye``, which points the other way — so the angle
            # negates.  Getting this backwards lights the character from below,
            # which reads as "no terminator" from the front just like a head
            # light does, only with the chin shadow inverted.
            c, s = np.cos(-ax[0]), np.sin(-ax[0])
            to_eye = (to_eye * c + np.cross(right, to_eye) * s
                      + right * np.dot(right, to_eye) * (1.0 - c))
    # camera.direction() points target -> eye, i.e. *towards* the light; the
    # shader wants the direction the light travels.
    return tuple(float(x) for x in -to_eye)


#: Does the engine gamma→linear ``_ambientshadowG`` on its way to the shader?
#:
#: **The one link in this chain that is not read off the install** (§7.126), and
#: it is not read because there is nothing to read.  §7.85's rule for "does the
#: engine convert this colour" is *type-driven*: the property is declared
#: ``Color`` in the shader's own table and carries no ``[Gamma]``.  The four
#: ``…G`` globals are **not declared properties at all** — ``origin_shader_dump
#: --props`` finds zero occurrences of ``_ambientshadowG`` / ``_LineColorG`` /
#: ``_RampG`` / ``_linewidthG`` in ``main_skin`` or ``main_opaque``; they are
#: bare uniforms ``ChaShader`` sets with ``Shader.SetGlobalColor``.  With no
#: declaration there is no type for the rule to apply, so the value is taken to
#: reach the shader as authored.
#:
#: ``reference/`` cannot settle it either, and that is worth recording rather
#: than fudging: the quantity it would decide is the shadow tint at ``lit = 0``,
#: and **Origin's own thigh never reaches it** — swept 1..4 px inside the
#: silhouette, Origin's floor÷lit climbs 0.773 → 0.858 (still descending at the
#: edge) where ours is flat at 0.643 → 0.666 (a real plateau).  A ruler whose
#: reading on the reference is still moving cannot choose between two values
#: 5% apart, so this follows the readable rule and says so.
#:
#: The two branches differ by about a level on screen once the conversion is
#: applied in the right place: ambient base 0.274 (as authored) against 0.261
#: (converted first, then the overlay).
AMBIENT_SHADOW_CONVERTED = False


def _ambient_base() -> tuple:
    """``_ambientshadowG`` folded through the overlay every Origin shader opens with.

    ``1 - (1 - a) * (1 - rgb)`` when ``a * 0.5 + 0.5 > 0.5``, else ``(1 + a) *
    rgb``.  With the shipped (0.019, 0.027, 0.039, 0.26) that is the screen
    branch and lands on about (0.274, 0.280, 0.289) — the dark blue-grey every
    character's shadows are built on.  Constant, so it is resolved once here
    rather than per fragment.

    **If the value is converted at all, it is converted here — on the colour,
    before the overlay — never on the result.**  The overlay is arithmetic the
    shader does on whatever it was handed, so converting its output is a
    different number entirely: 0.26 + 0.74·rgb evaluated in gamma and then
    decoded gives **0.060**, where decoding first and then evaluating gives
    **0.261**, a factor of four.  That mistake is visible rather than subtle —
    it quadruples the darkness the tint is hard-lit against and paints the
    shadowed side of every limb a flat saturated orange (§7.126).
    """
    r, g, b, a = TOON.ambient_shadow()
    rgb = np.array([r, g, b])
    if AMBIENT_SHADOW_CONVERTED:
        rgb = CS.colour_to_linear((*rgb, 1.0))[:3].astype(np.float64)
    out = (1.0 - (1.0 - a) * (1.0 - rgb) if a * 0.5 + 0.5 > 0.5
           else (1.0 + a) * rgb)
    return tuple(float(x) for x in np.clip(out, 0.0, 1.0))


def _ambient_floor() -> tuple:
    """``_ambientshadowG.rgb`` itself, in the space the shader reads it."""
    rgb = TOON.AMBIENT_SHADOW_RGB
    return _lin3(rgb) if AMBIENT_SHADOW_CONVERTED else tuple(float(x) for x in rgb)


_AMBIENT_BASE = _ambient_base()
_AMBIENT_FLOOR = _ambient_floor()
# ``_ambientshadowG.a`` — the raw alpha (``shadowDepth``); ``toon_glasses_lod0``
# hard-lights with it directly rather than through ``0.5*a + 0.5``
_AMBIENT_SHADOW_A = float(TOON.ambient_shadow()[3])

#: **Pin the ramp term, for a ruler that needs this pass's range** (§7.341).
#:
#: ``shaded`` is linear in ``lit`` and colour grading is monotone, so rendering
#: once at 0 and once at 1 brackets every colour ``main_*`` can produce on a
#: texel.  §7.126 needed exactly that ("no value of ``lit`` fits Origin's shadow
#: under the chin") and could only get it by hand, for one patch; ``ref_compare
#: --sweep-lit`` now asks it for every patch.
#:
#: Negative is off, and nothing on the drawing path ever writes it — the flag
#: lives here rather than on a material because the question is about the pass,
#: not about one surface.
LIT_OVERRIDE = -1.0

#: A path; the next shadow pass writes its depth map there (§7.343).
#: Off by default -- a shadow in the wrong place and a shadow that is
#: too dark look the same on the character and need opposite fixes.
SHADOW_DUMP = ""


#: Unity's ``Transparent`` queue, the tag the blended shaders here declare —
#: the order a part with no stored queue sorts at (§7.379).
TRANSPARENT_QUEUE = 3000


class Renderer:
    """Owns GL objects; must be used with its context current."""

    def __init__(self, samples: int = 4):
        self.native = NativeStage()
        self.samples = samples
        self.toon = Program(shaders.TOON_VS, shaders.TOON_FS, "toon")
        self.outline = Program(shaders.OUTLINE_VS, shaders.OUTLINE_FS, "outline")
        self.post = Program(shaders.POST_VS, shaders.POST_FS, "post")
        self.bloom = Program(shaders.POST_VS, shaders.BLOOM_FS, "bloom")
        self.line = Program(shaders.LINE_VS, shaders.LINE_FS, "line")
        self.sky = Program(shaders.SKY_VS, shaders.SKY_FS, "sky")
        self.sky_image = Program(shaders.SKY_VS, shaders.SKY_IMAGE_FS, "sky_image")
        self.shadow = Program(shaders.SHADOW_VS, shaders.SHADOW_FS,
                              "shadow")
        self.textures = TextureCache()
        self.white = self.textures.get_or_make(
            ("white",), lambda: np.ones((4, 4, 4), np.float32))
        # Renderer binds ``self.white`` every empty slot without re-resolve;
        # pin so LRU thrash during Free H (two full Origin bodies) cannot delete it.
        self.textures.pin(("white",))

        self._gpu: dict[int, dict[int, GPUMesh]] = {}
        self._msaa: Framebuffer | None = None
        self._resolve: Framebuffer | None = None
        self._bloom_fbo: Framebuffer | None = None
        self._shadow_fbo: Framebuffer | None = None
        self._light_mat = np.eye(4, dtype=np.float32)
        self._shadow_texel = 0.0
        self._shadow_bias = 0.0
        self._shadow_texel_world = 0.0
        self._empty_vao = GL.glGenVertexArrays(1)
        self._lines = LineBatch()
        self._grid_batch = LineBatch()
        self._grid_key = None
        self.stats = {"draws": 0, "tris": 0}
        # Full Origin armatures exceed uniform mat4[] limits — bone palette lives
        # in an RGBA32F texture (4×N: one mat4 column per texel).
        self._bone_tex = int(GL.glGenTextures(1))
        self._bone_tex_n = 0
        #: per drawn model (``id``): ``[texture, the matrices and rests last uploaded, bone count]`` --
        #: the outline, opaque and transparent passes each bind an item's bones, and its matrices
        #: are the same in all of them, so they go up once a frame (``_bone_texture``)
        self._item_bones: dict = {}
        self._bone_identity_uploaded = False
        # None = not looked up yet, 0 = looked up and Koikatsu is not installed
        self._lut_tex = None
        # the same for the grade's inverse (``_ungrade_texture``)
        self._ungrade_tex = None
        # this frame's (inverse-grade texture or None, strength), set with the surfaces
        # (``_setup_toon``) and read again by their outline (``_ef_outline``)
        self._ef_ungrade = (None, 0.0)
        # Milliseconds of texture upload a frame may spend before a model whose
        # maps are not all on the GPU yet sits the frame out (``_resident``).
        # ``None`` uploads whatever the frame draws: tools and offscreen
        # renders want every frame complete; a window sets a budget.
        self.upload_budget_ms: float | None = None

    # ------------------------------------------------------------------
    # GPU mesh cache
    # ------------------------------------------------------------------
    def _model_cache(self, model: Model) -> dict[int, GPUMesh]:
        return self._gpu.setdefault(id(model), {})

    def gpu_for(self, model: Model, part: Part) -> GPUMesh:
        if part.material.normal_encoding == 1 and len(part.mesh.tangents) != part.mesh.vertex_count:
            raise ValueError('GF2 normal component requires complete authored tangents')
        cache = self._model_cache(model)
        key = id(part)
        g = cache.get(key)
        if g is None:
            g = GPUMesh(part.mesh, dynamic=part.dynamic)
            cache[key] = g
        elif g.source is not part.mesh:
            # the part was handed another mesh (an outfit state, a piece taken
            # off): its triangles changed, not only where its vertices are
            g.upload(part.mesh, dynamic=part.dynamic)
        elif part.dynamic or part.stale:
            g.refresh(part.mesh)
        part.stale = False
        return g

    def mark_dirty(self, model: Model) -> None:
        self._release_bones(model)
        cache = self._gpu.pop(id(model), None)
        if cache:
            for g in cache.values():
                g.release()

    def clear_cache(self) -> None:
        self._release_bones()
        for cache in self._gpu.values():
            for g in cache.values():
                g.release()
        self._gpu.clear()

    # ------------------------------------------------------------------
    def _ensure_targets(self, w: int, h: int) -> bool:
        if self._msaa is None:
            # sRGB on both scene targets: that is Koikatsu's camera render
            # target, and it is where the forward passes' linear output gets
            # encoded (§7.126).  The bloom target is **not** sRGB — BLOOM_FS
            # writes gamma, because the threshold and the additive blend that
            # reads it are ours and were calibrated in that space.
            self._msaa = Framebuffer(w, h, samples=self.samples, name="msaa",
                                     srgb=True)
            if not self._msaa.valid and self.samples:
                self._msaa.release()
                self.samples = 0
                self._msaa = Framebuffer(w, h, samples=0, depth_texture=True,
                                         name="msaa0", srgb=True)
            self._resolve = Framebuffer(w, h, samples=0, depth_texture=True,
                                        name="resolve", mipmap=True, srgb=True)
            self._bloom_fbo = Framebuffer(max(w // 4, 1), max(h // 4, 1),
                                          samples=0, name="bloom")
        else:
            self._msaa.resize(w, h)
            self._resolve.resize(w, h)
            self._bloom_fbo.resize(max(w // 4, 1), max(h // 4, 1))
        return bool(self._msaa and self._msaa.valid and self._resolve and self._resolve.valid)

    # ------------------------------------------------------------------
    def render(self, scene: RenderScene, width: int, height: int,
               target_fbo: int | None = None,
               shift: tuple[float, float] = (0.0, 0.0)) -> None:
        """Draw *scene* into *target_fbo* (the bound framebuffer when ``None``).

        *shift* moves the camera target off the frame's centre, in NDC (a lens
        shift: the view is not rotated, the image is slid) — the viewport uses
        it to centre the subject in the part of the window no panel covers.
        """
        w, h = max(1, int(width)), max(1, int(height))
        if target_fbo is None:
            target_fbo = int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING))
        aspect = w / max(h, 1)
        view = scene.camera.view()
        proj = scene.camera.projection(aspect)
        if shift[0] or shift[1]:
            proj = m3.translate((float(shift[0]), float(shift[1]), 0.0)) @ proj
        eye = scene.camera.eye()
        frame = SceneView(scene, w, h, view, proj, eye, clock=scene.clock.snapshot())
        # Prepare every source before any shadow, clear or character draw.
        # Missing native resources cannot leave an apparently successful card
        # containing only the ordinary scene/room or half of a character.
        with self.native.prepare(scene.native_items, frame) as native:
            self._render_prepared(frame, target_fbo, native)

    def _render_prepared(self, frame: SceneView, target_fbo: int, native: PreparedScene) -> None:
        scene, w, h = frame.scene, frame.width, frame.height
        view, proj, eye = frame.view, frame.projection, frame.eye
        self.stats = {"draws": 0, "tris": 0}

        # The pass always runs now, and not for the grade: it is the **encode
        # boundary**.  The character passes write linear into an sRGB target
        # (Koikatsu's camera does the same), so something has to convert on the
        # way to a plain RGBA8 window, and that something is POST_FS's tail.
        # Our grade being off, and Illusion's LUT being absent, only neutralise
        # terms inside it (see ``_draw_post``).
        # **Before the scene target is bound**: this pass owns the
        # framebuffer while it runs, and it has to know what is on screen, so
        # the item list is built here rather than below.
        items = self._resident([it for it in scene.items if it.visible and it.model])
        self._shadow_on = self._render_shadow(items, scene)

        use_post = self._ensure_targets(w, h)
        if use_post:
            self._msaa.bind()
            # linear in, sRGB stored — and blending decodes/encodes around it,
            # which is the half a shader-side conversion cannot reproduce
            GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)
        else:
            # FBO creation failed, which on any GL 3.3 driver means the context
            # is broken.  Draw straight out rather than showing nothing; the
            # frame will be over-bright because nothing encodes it, which is a
            # visible failure rather than a subtle one (§一之一).
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
            GL.glViewport(0, 0, w, h)

        bg = scene.background
        if bg.mode == "transparent":
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        else:
            # the clear goes through the same encode, so the colour it is given
            # has to be linear like everything else the scene pass writes
            GL.glClearColor(*_lin3(bg.bottom), 1.0)
        GL.glClearStencil(0)
        GL.glStencilMask(0xFF)                       # or the clear is a no-op
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT
                   | GL.GL_STENCIL_BUFFER_BIT)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_BLEND)

        if bg.mode == "image" and bg.image_path:
            self._draw_sky_image(bg)
        elif bg.mode == "gradient":
            self._draw_sky(bg)

        # Nothing on screen asks for the cut-out unless some character's card
        # put a sheet in front of the fringe, and holding the fringe back costs
        # real time when it does: the stencil test on that geometry measures
        # 19.1 -> 22.3 ms of render for three characters (43 -> 38 FPS,
        # reproduced twice back to back).  So the whole detour is skipped when
        # the answer is no, which is 52% of the local library.
        cutout = any(part.foreground and part.material.stencil_write
                     for it in items for part in it.model.visible_parts())
        if scene.show_outline:
            self._draw_outlines(items, view, proj, w, h, eye,
                                defer_masked=cutout)
        masked = self._draw_opaque(items, scene, view, proj, eye, defer=cutout)
        if scene.show_outline and any(
                part.material.toon_family >= EF.FAMILY_MAIN
                for it in items for part in it.model.visible_parts()):
            self._ef_light = ef_lit_light(scene.light)
            self._draw_outlines(items, view, proj, w, h, eye, late=True)
        early = self._draw_foreground(masked, items, scene, view, proj, eye,
                                      w, h, scene.show_outline)
        if native.active:
            native_target = SceneTarget(int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)),
                                        self.samples if use_post else int(GL.glGetIntegerv(GL.GL_SAMPLES)),
                                        bool(GL.glIsEnabled(GL.GL_FRAMEBUFFER_SRGB)))
            native.opaque(native_target)
        self._draw_grid(scene, view, proj)
        if scene.overlay_lines is not None:
            self._draw_overlay(scene.overlay_lines, view, proj)
        self._draw_transparent(items, scene, view, proj, eye, skip=early)
        if native.active:
            native.transparent(native_target)

        if use_post:
            self._msaa.blit_to(self._resolve, color=True, depth=self.samples == 0)
            if self.samples > 0:
                self._blit_depth()
            fx = scene_effects(scene)
            if scene.post.enabled and scene.post.dof_strength > 0.001 and fx["DepthOfField"]:
                self._resolve.regenerate_mipmaps()
            if scene.post.enabled and scene.post.bloom > 0.001 and fx["Bloom"]:
                self._draw_bloom(scene)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
            GL.glViewport(0, 0, w, h)
            # POST_FS encodes for itself: the destination is whatever the caller
            # bound (a window, an offscreen grab) and is not sRGB-capable.
            GL.glDisable(GL.GL_FRAMEBUFFER_SRGB)
            self._draw_post(scene, w, h)

    # ------------------------------------------------------------------
    def _render_shadow(self, items, scene: "RenderScene") -> bool:
        """Depth from the character light, for the ramp term (§7.343).

        Returns whether the forward pass should read it.  Off, missing, or a
        scene with nothing to cast all leave ``uShadowOn`` at 0, which is the
        construction before this section bit for bit.

        **The frustum is fitted to what casts, every frame.**  A fixed volume
        has to be large enough for the tallest card this install has (2.22 m,
        §7.340) and then spends most of its texels on air; fitted, a 1024 map
        puts about 1.5 mm on the body of a single character, which is the scale
        the feature exists for (a fringe's shadow on a forehead).  The bounds
        come from ``model.bounds()``, which is the **bind** pose, so a posed
        arm can reach outside it -- hence the margin, and hence the guard in
        the shader that treats "outside the map" as lit rather than as shadow
        (一之一: the honest answer to "I cannot see there" is not "it is dark
        there").
        """
        li = scene.light
        if not (li.chara_light and li.chara_shadow) or not items:
            return False
        # **The skinning matrices, not ``model.bounds()``.**  The bind box is a
        # T-pose: on the test card it is 1.32 m across while the drawn figure
        # is 0.40, so a frustum fitted to it spends **3.3x** of the map's width
        # on air -- measured, the map came back 8.5% covered.  What follows the
        # pose is the bone palette the same frame is about to draw with; its
        # translations are where the joints ended up, and the reach past them
        # (skin, hair, a skirt) is bounded by the **bind** box's own half
        # diagonal, which is a measurement rather than a guess.
        pts = []
        for it in items:
            m = np.asarray(it.transform, dtype=np.float64)
            lo, hi = it.model.bounds()
            ok = bool(np.all(np.isfinite(lo)) and not np.allclose(lo, hi))
            sk = getattr(it, "skin", None)
            if sk is not None and len(sk) and ok:
                j = np.asarray(sk, dtype=np.float64).reshape(-1, 4, 4)[:, :3, 3]
                j = j @ m[:3, :3].T + m[:3, 3]
                # A tenth of the figure's **height**, not of the box's
                # diagonal: the diagonal carries the T-pose's arm span, which
                # is the very thing this is here to stop paying for.  What has
                # to fit is how far the surface reaches past its joints --
                # long hair off the skull is the widest of those, and on this
                # install it is under a tenth of stature.
                pad = 0.10 * float(abs(hi[1] - lo[1]))
                pts.append(j.min(axis=0) - pad)
                pts.append(j.max(axis=0) + pad)
                continue
            if not ok:
                continue
            for c in ((lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]),
                      (lo[0], hi[1], lo[2]), (hi[0], hi[1], lo[2]),
                      (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]),
                      (lo[0], hi[1], hi[2]), (hi[0], hi[1], hi[2])):
                pts.append((m @ np.array([c[0], c[1], c[2], 1.0]))[:3])
        if not pts:
            return False
        pts = np.asarray(pts, dtype=np.float64)

        size = max(64, int(li.chara_shadow_size))
        if self._shadow_fbo is None:
            self._shadow_fbo = Framebuffer(size, size, samples=0,
                                           depth_texture=True, name="shadow")
        elif (self._shadow_fbo.width, self._shadow_fbo.height) != (size, size):
            self._shadow_fbo.resize(size, size)
        if not self._shadow_fbo.valid:
            return False

        d = np.asarray(scene_chara_light_dir(scene), dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return False
        d /= n
        up = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.95 \
            else np.array([0.0, 0.0, 1.0])
        right = np.cross(up, -d)
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(-d, right)
        centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        R = np.stack([right, up, -d])                    # rows: light basis
        local = (pts - centre) @ R.T
        half = np.abs(local).max(axis=0) * 1.05 + 0.02
        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = R
        view[:3, 3] = -R @ centre
        # Orthographic: a character light in Unity is directional.
        proj = np.eye(4, dtype=np.float64)
        proj[0, 0] = 1.0 / half[0]
        proj[1, 1] = 1.0 / half[1]
        proj[2, 2] = -1.0 / half[2]
        self._light_mat = (proj @ view).astype(np.float32)
        # One texel across, in the light's own units, is what the PCF kernel
        # steps by and what the depth bias is scaled against -- both have to
        # follow the fitted volume or a close-up and a wide shot disagree.
        # Three numbers, all off the fitted volume, because a value tuned at
        # one zoom is wrong at the next: the PCF step is in **light UV**, the
        # normal offset is in **world** units, and the depth bias is in the
        # **[0, 1]** the sampler returns (the projection maps one world unit of
        # depth to ``0.5 / half[2]`` there).
        self._shadow_texel = 1.0 / float(size)
        world_texel = float(2.0 * max(half[0], half[1]) / size)
        self._shadow_texel_world = world_texel
        self._shadow_bias = float(2.0 * world_texel * 0.5 / max(half[2], 1e-6))

        prev = int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING))
        self._shadow_fbo.bind()
        GL.glDisable(GL.GL_FRAMEBUFFER_SRGB)
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDisable(GL.GL_BLEND)
        # **Front faces, not back.**  The usual trick for shadow acne is to
        # cast from back faces, and it needs closed geometry: a hair card and a
        # skirt panel are single sheets here, and casting from their back faces
        # drops their shadow entirely -- which is the one shadow this feature
        # is for.  The slope-scaled bias in the shader carries the acne instead.
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        pr = self.shadow.use()
        pr.set_mat4("uLightMatrix", self._light_mat)
        for it in items:
            pr.set_mat4("uModel", it.transform)
            self._bind_skin(pr, it)
            for part in it.model.visible_parts(shadow=True):
                mat = part.material
                # A blended surface does not occlude in Origin either -- its pass
                # has no depth write.  A cut-out one does, and it casts through
                # its holes here (the alpha is not sampled), which is visible
                # on lace and is written down rather than hidden (一之一).
                if self._is_transparent(mat):
                    continue
                self._part_skin(pr, part)
                self.gpu_for(it.model, part).draw()
                self.stats["draws"] += 1
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, prev)
        if SHADOW_DUMP:
            self._dump_shadow(SHADOW_DUMP)
        return True

    def _dump_shadow(self, path: str) -> None:
        """Write the light's depth out as an image (§7.343's own eyes).

        A shadow that lands in the wrong place looks, on the character, exactly
        like a shadow that is too dark — and the two need opposite fixes, so
        the map has to be lookable at.  Set :data:`SHADOW_DUMP` to a path.
        """
        import numpy as _np
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._shadow_fbo.depth)
        w, h = self._shadow_fbo.width, self._shadow_fbo.height
        buf = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT,
                               GL.GL_FLOAT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        a = _np.frombuffer(bytes(buf), dtype=_np.float32).reshape(h, w)
        a = _np.flipud(a)
        lo = float(a[a < 1.0].min()) if (a < 1.0).any() else 0.0
        img = _np.where(a < 1.0, (a - lo) / max(1.0 - lo, 1e-6), 1.0)
        try:
            from PIL import Image as _Im
            _Im.fromarray((img * 255).astype(_np.uint8)).save(path)
            print(f"[shadow] wrote {path}  {w}x{h}  "
                  f"覆蓋 {(a < 1.0).mean():.1%}  near {lo:.4f}")
        except Exception as exc:                     # noqa: BLE001
            print(f"[shadow] dump failed: {exc}")

    def _blit_depth(self) -> None:
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._msaa.fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._resolve.fbo)
        try:
            GL.glBlitFramebuffer(0, 0, self._msaa.width, self._msaa.height,
                                 0, 0, self._resolve.width, self._resolve.height,
                                 GL.GL_DEPTH_BUFFER_BIT, GL.GL_NEAREST)
        except Exception as exc:
            swallowed.note("gl.renderer.Renderer._blit_depth: GL.glBlitFramebuffer", exc)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # ------------------------------------------------------------------
    def _draw_sky(self, bg: Background) -> None:
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        p = self.sky.use()
        p.set_vec3("uTop", _lin3(bg.top))
        p.set_vec3("uBottom", _lin3(bg.bottom))
        p.set_float("uPower", bg.power)
        GL.glBindVertexArray(self._empty_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)

    def _draw_sky_image(self, bg: Background) -> None:
        """Full-screen textured backdrop (Origin scene thumb / UserData/bg)."""
        tid = self._bg_image_tex(bg.image_path)
        if tid is None:
            self._draw_sky(bg)
            return
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        p = self.sky_image.use()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
        p.set_int("uImage", 0)
        p.set_float("uDim", float(bg.image_dim))
        GL.glBindVertexArray(self._empty_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)

    def _bg_image_tex(self, path: str) -> int | None:
        from pathlib import Path as _P
        src = _P(path)
        if not path or not src.is_file():
            return None
        key = ("bg_image", str(src.resolve()), int(src.stat().st_mtime_ns))
        tid = self.textures.get(key)
        if tid is not None:
            return tid
        try:
            from PIL import Image
            im = Image.open(src).convert("RGBA")
            # Cap resolution so huge studio thumbs stay frame-budget friendly.
            im.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            # A photo/thumbnail is sRGB by definition, and the scene target
            # re-encodes, so decoding here is what leaves it unchanged.
            return self.textures.upload(key, arr, mipmap=True,
                                        wrap=GL.GL_CLAMP_TO_EDGE, srgb=True)
        except Exception as exc:  # noqa: BLE001
            swallowed.note("gl.renderer.Renderer._bg_image_tex: from PIL import Image", exc)
            return None

    def _grading_lut(self):
        """Illusion's ``Lut_Amount`` as a GL texture, or None when uninstalled.

        Clamped and unmipmapped: the strip is 32 slices side by side, so a
        mip or a repeat wrap would blend neighbouring slices -- i.e. blue
        values a whole step apart -- into each other at the seams.

        **Uploaded raw, never as sRGB.**  It is a lookup table indexed by a
        gamma-space colour and returning one, which is what Amplify's own pass
        does either side of it (``POST_FS``); decoding it here would grade
        against a curve the LUT was not fitted to.
        """
        if self._lut_tex is not None:
            return self._lut_tex or None
        from ..origin import grading
        arr = grading.lut()
        if arr is None:
            self._lut_tex = 0                        # asked once, not there
            return None
        rgba = np.ones((arr.shape[0], arr.shape[1], 4), np.float32)
        rgba[:, :, :3] = arr.astype(np.float32) / 255.0
        self._lut_tex = self.textures.upload(
            ("origin_grading_lut",), rgba, mipmap=False, wrap=GL.GL_CLAMP_TO_EDGE)
        self.textures.pin(("origin_grading_lut",))
        return self._lut_tex

    def _ungrade_texture(self) -> int | None:
        """The inverse of the colour grade as a texture, laid out like the grade; None without one."""
        if self._ungrade_tex is not None:
            return self._ungrade_tex or None
        from ..origin import grading
        arr = grading.inverse()
        if arr is None:
            self._ungrade_tex = 0
            return None
        rgba = np.ones((arr.shape[0], arr.shape[1], 4), np.float32)
        rgba[:, :, :3] = arr
        self._ungrade_tex = self.textures.upload(
            ("origin_grading_lut_inverse",), rgba, mipmap=False, wrap=GL.GL_CLAMP_TO_EDGE)
        self.textures.pin(("origin_grading_lut_inverse",))
        return self._ungrade_tex

    def _upload_bone_tex(self, mats: np.ndarray, rests: np.ndarray | None = None, tex: int | None = None) -> int:
        """Upload (N,4,4) skin matrices → RGBA32F texture, one row per bone. Returns N.

        Four texels per bone, the mat4's columns (GLSL mat4(c0,c1,c2,c3)).
        With *rests* — the bones' rest frames, for an item that has a
        dual-quaternion part — ten: the palette follows the matrix.  *tex* is
        the texture to fill (the shared one by default); the upload goes
        through the bone unit, so no material's map is unbound by it.
        """
        tex = self._bone_tex if tex is None else tex
        mats = np.ascontiguousarray(mats, dtype=np.float32)
        if mats.ndim != 3 or mats.shape[1:] != (4, 4):
            mats = mats.reshape(-1, 4, 4)
        n = int(mats.shape[0])
        if n < 1:
            n = 1
            mats = np.eye(4, dtype=np.float32)[None, ...]
            rests = None
        palette = skinning.dual_palette(mats, rests) if rests is not None else None
        pixels = skinning.palette_texels(mats, palette)
        GL.glActiveTexture(GL.GL_TEXTURE0 + BONE_TEX_UNIT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, pixels.shape[1], n, 0,
            GL.GL_RGBA, GL.GL_FLOAT, pixels)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        self._bone_tex_n = n
        return n

    def _bone_texture(self, item: DrawItem, rests) -> tuple[int, int]:
        """``(texture, bone count)`` holding *item*'s skin matrices, uploaded only when they changed.

        Compared by value, not by the array: a held pose hands the same
        matrices in a new array every frame, and a reused array can have
        been written in place.
        """
        key = id(item.model)
        held = self._item_bones.get(key)
        if held is None or held[3]() is not item.model:
            if held is not None:                    # the id is another model's now: its texture goes
                GL.glDeleteTextures(1, [held[0]])
            held = self._item_bones[key] = [int(GL.glGenTextures(1)), None, None, weakref.ref(item.model), 0]
        skin = np.asarray(item.skin)
        same = (held[1] is not None and held[1].shape == skin.shape and np.array_equal(held[1], skin)
                and ((rests is None and held[2] is None)
                     or (rests is not None and held[2] is not None and held[2].shape == rests.shape
                         and np.array_equal(held[2], rests))))
        if not same:
            held[4] = self._upload_bone_tex(skin, rests, tex=held[0])
            held[1] = skin.copy()
            held[2] = None if rests is None else np.array(rests)
        return held[0], held[4]

    def _release_bones(self, model=None) -> None:
        """Delete the bone textures of *model* (every model's with ``None``)."""
        keys = list(self._item_bones) if model is None else [id(model)]
        for key in keys:
            held = self._item_bones.pop(key, None)
            if held is not None:
                GL.glDeleteTextures(1, [held[0]])

    def _bind_skin(self, prog: Program, item: DrawItem) -> None:
        """Bind one item's bone matrices.  Once per item, not per part.

        The unit matters and it has to be one of its own.  This used to sit on
        unit 1, which every material then rebinds a page later — ``uPattern``
        in the main pass, ``uMaskTex`` in the outline one, and a 1×1 white
        texture when the material has neither.  The vertex shader was
        therefore building its ``mat4`` out of white texels for every part
        after the first, which sends the whole character to a garbage position:
        **every skinned draw in the app was broken**, and only the offline
        preview escaped it by drawing the bind pose with ``skin=None``.
        """
        # the head's frame, for the families that light a face by it
        # (``gl/shaders_endfield.py``); identity for anything without a head
        prog.set_mat4("uEfHead", item.head_rot if item.head_rot is not None
                      else _IDENTITY)
        self._dual_bound = False
        prog.set_int("uSkinDual", 0)
        if item.skin is not None and len(item.skin):
            rests = self._dual_rests(item)
            tex, n = self._bone_texture(item, rests)
            self._dual_bound = rests is not None
            GL.glActiveTexture(GL.GL_TEXTURE0 + BONE_TEX_UNIT)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            prog.set_int("uBoneTex", BONE_TEX_UNIT)
            prog.set_int("uBoneCount", n)
            prog.set_int("uSkinned", 1)
            self._skin_bound = True
            GL.glActiveTexture(GL.GL_TEXTURE0)
        else:
            prog.set_int("uSkinned", 0)
            prog.set_int("uBoneCount", 0)
            self._skin_bound = False

    @staticmethod
    def _dual_rests(item: DrawItem) -> np.ndarray | None:
        """The item's rest frames when one of its parts blends dual quaternions.

        ``None`` otherwise, which keeps every other item's bone texture at the
        four texels it always had.  The frames are the model's own skeleton's;
        an item posed by matrices of another length has no palette to make.
        """
        if not any(p.skinning != skinning.LINEAR for p in item.model.parts):
            return None
        skeleton = getattr(item.model, "skeleton", None)
        if skeleton is None or len(skeleton.bones) != len(item.skin):
            raise ValueError(f"{item.model.name}: dual-quaternion parts need the skeleton "
                             f"the skin matrices belong to")
        return skeleton.rest_frames()

    def _part_skin(self, prog: Program, part: Part) -> None:
        """Cloth-solved parts carry model-space positions and skip skinning."""
        want = 1 if (part.skinned and getattr(self, "_skin_bound", False)) else 0
        prog.set_int("uSkinned", want)
        dual = 0
        if want and getattr(self, "_dual_bound", False):
            dual = _SKIN_DUAL[part.skinning]
        prog.set_int("uSkinDual", dual)

    def _draw_outlines(self, items, view, proj, w, h, eye, pairs=None,
                       defer_masked: bool = True, late: bool = False) -> None:
        """Inverted-hull pass, drawn before the opaque one so the surface wins.

        *late* draws the other half: the ``HGRP/CharacterNPR*`` parts, whose
        hulls go **after** the surfaces because their stencil test reads what
        the surfaces stamped (``gl/shaders_endfield.py``).

        The hull is nudged away from the camera in clip space.  Where a garment
        folds sharply — the underbust crease, the navel — a back face can sit
        *nearer* than the front face and the outline then paints a dark wedge
        across the middle of the cloth.  ``glPolygonOffset`` fixes it too and is
        slope-aware, but it turns off hierarchical-Z: measured here it cost a
        third of the frame rate (108 FPS against 144 for one character), which
        is far too much for a bias this pass can get by with as a constant.

        ``main_hair_front``'s OUTLINE pass carries the **same** stencil test as
        its FORWARD one, and both live in the material's single queue slot, so
        the fringe's hull has to be held back with the fringe itself — a hull
        drawn here would write depth across the eyebrow before anything had
        stamped the stencil, and the cut-out would come out with a black line
        through it.  ``pairs`` is how :meth:`_draw_foreground` draws them later.
        """
        p = self.outline.use()
        p.set_mat4("uView", view)
        p.set_mat4("uProj", proj)
        p.set_vec2("uViewport", (w, h))
        p.set_vec3("uCameraPos", eye)
        p.set_float("uLineWidthG", TOON.line_width_g())
        p.set_float("uFovHalf", math.atan(1.0 / max(abs(float(proj[1, 1])), 1e-6)))
        GL.glCullFace(GL.GL_FRONT)
        # Grouped by item, not one entry per part: ``_bind_skin`` uploads the
        # bone texture and is meant to run once per character.  Handing it a
        # flat pair list ran it 21 times instead of 3 for three characters and
        # cost 7 FPS — which read as "the stencil test is expensive" until the
        # test was switched off and nothing changed.
        walk = ([(it, list(it.model.visible_parts())) for it in items]
                if pairs is None else _group_by_item(pairs))
        for it, parts in walk:
            # an item with no inked part (the shot fluid, a prop) has nothing to draw here: no skin to bind
            if not any(pt.material.cast_outline and pt.material.outline_width > 0.0 for pt in parts):
                continue
            p.set_mat4("uModel", it.transform)
            self._bind_skin(p, it)
            for part in parts:
                mat = part.material
                if not mat.cast_outline or mat.outline_width <= 0.0:
                    continue
                if pairs is None and defer_masked and mat.stencil_reject:
                    continue
                self._part_skin(p, part)
                if (mat.toon_family >= EF.FAMILY_MAIN) != late:
                    continue
                if late:
                    # after the surfaces, under the part's own stencil test:
                    # a fringe the eyes show through has no line there either
                    state = mat.stencil_state
                    if state is not None:
                        _stencil_begin((state[0], state[1], state[2], 0))
                    self._ef_outline(p, mat, it, self._ef_light)
                    self.gpu_for(it.model, part).draw_hull()      # without its folds (``mesh_gpu.index_buffer``)
                    self.stats["draws"] += 1
                    if state is not None:
                        _stencil_end()
                    continue
                origin_game = mat.toon_family != TOON.FAMILY_NONE
                p.set_int("uWidthMode", 1 if origin_game else 0)
                p.set_float("uWidth", mat.outline_width * it.outline_scale)
                p.set_float("uZBias", OUTLINE_Z_BIAS)
                p.set_vec4("uColor", _lin4(mat.outline_color))
                # main_hair is the one family whose outline reads the strand
                # colour; the rest resolve to flat black (see shaders.py).
                hair = mat.toon_family == TOON.FAMILY_HAIR
                p.set_int("uColorMode", 1 if hair else 0)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                want_tex = origin_game and (hair or mat.alpha_test > 0.0) and mat.texture
                if want_tex:
                    GL.glBindTexture(GL.GL_TEXTURE_2D,
                                     resolve_texture(self.textures, mat.texture))
                    p.set_int("uUseTex", 1)
                    p.set_int("uUseAlpha", 1)
                else:
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self.white)
                    p.set_int("uUseTex", 0)
                    p.set_int("uUseAlpha", 0)
                p.set_int("uTex", 0)
                p.set_int("uAlphaTex", 0)
                p.set_float("uAlphaTest", mat.alpha_test)
                GL.glActiveTexture(GL.GL_TEXTURE1)
                if origin_game and mat.shade_mask:
                    GL.glBindTexture(GL.GL_TEXTURE_2D,
                                     resolve_texture(self.textures, mat.shade_mask))
                    p.set_int("uUseMask", 1)
                else:
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self.white)
                    p.set_int("uUseMask", 0)
                p.set_int("uMaskTex", 1)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                self.gpu_for(it.model, part).draw()
                self.stats["draws"] += 1
        GL.glCullFace(GL.GL_BACK)

    def _ef_outline(self, p: Program, mat: Material, item: DrawItem, light) -> None:
        """One ``HGRP/CharacterNPR*`` part's outline state (``CharacterOutline``)."""
        p.set_int("uWidthMode", 2)
        p.set_float("uWidth", mat.outline_width * item.outline_scale)
        p.set_float("uOffsetZ", mat.outline_tone[2])
        p.set_int("uColorMode", 2)
        p.set_vec4("uColor", _lin4(mat.base_color))
        p.set_vec2("uTone", mat.outline_tone[:2])
        p.set_vec3("uLight", light)
        p.set_float("uAlphaTest", mat.alpha_test)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, resolve_texture(self.textures, mat.texture)
                         if mat.texture else self.white)
        p.set_int("uUseTex", 1 if mat.texture else 0)
        p.set_int("uTex", 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        mask = mat.detail_mask
        GL.glBindTexture(GL.GL_TEXTURE_2D, resolve_texture(self.textures, mask)
                         if mask else self.white)
        p.set_int("uUseMask", 1 if mask else 0)
        p.set_int("uMaskTex", 1)
        # the outline is that game's colour too: written through the grade's inverse like its
        # surfaces, or the grade crushes the toned-down skin to red lines (``shaders.OUTLINE_FS``)
        ungrade, strength = self._ef_ungrade
        GL.glActiveTexture(GL.GL_TEXTURE0 + UNGRADE_TEX_UNIT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, ungrade or self.white)
        p.set_int("uEfUngradeLut", UNGRADE_TEX_UNIT)
        p.set_float("uEfUngrade", strength)
        GL.glActiveTexture(GL.GL_TEXTURE0)

    def _material_uniforms(self, p: Program, mat: Material, light: LightSettings,
                           eye) -> None:
        """A material's uniforms and maps onto *p* for one draw.

        The uniform values are a function of the material alone, so they are
        worked out once per material and program and kept on the material with
        its revision (``Material._rev``); a draw only compares them with what
        the program holds (§7.555).  The maps are bound every draw: the cache
        may have re-uploaded one under a new name.
        """
        entries = self._material_entries(p, mat)
        held = p._val
        for loc, fn, args in entries:
            if held.get(loc) != args:
                held[loc] = args
                fn(loc, *args)
        # Unit 7 is the bone-matrix texture and is bound per *item*, not per
        # part — do not reuse it here (PROGRESS §八 一).
        # the liquid's three units only on a material that has them: the shader reads them behind their
        # ``uUse*`` flags, and binding white to three more units on every draw cost every scene
        wet = getattr(mat, "liquid_mask", None) is not None
        # the card game's programs read the table's maps only behind their ``uUse*`` switch, which is
        # off when the material has no such map (``_material_values``): nothing to bind for those.
        # Another game's programs (``toon_family`` 20 and up) read some of them behind switches of
        # their own, so they keep white bound in every unit as before
        flagged = mat.toon_family < 20
        for unit, attr in _MATERIAL_MAPS:
            key = getattr(mat, attr, None)
            if flagged and key is None and unit in _FLAGGED_UNITS:
                continue
            if wet or unit not in _LIQUID_UNITS:
                self._bind_map(unit, key)
        GL.glActiveTexture(GL.GL_TEXTURE0)

    def _resident(self, items: list) -> list:
        """*items* whose every map is on the GPU, uploading missing ones within the budget.

        The first frame of a freshly dressed character used to upload all of its
        maps at once -- 0.38 s of ``TextureCache.upload`` with the window frozen
        (§7.548).  With ``upload_budget_ms`` set, a model whose maps are not all
        up yet sits the frame out instead of drawing with white or another map
        in their place (CLAUDE.md 一之一), and at least one map goes up a frame
        so every model arrives.  ``None`` keeps the old behaviour: everything
        the frame draws is uploaded in it.
        """
        if self.upload_budget_ms is None:
            return items
        deadline = time.perf_counter() + self.upload_budget_ms / 1000.0
        uploaded = False
        out = []
        for it in items:
            ready = True
            for key in _model_map_keys(it.model):
                if self.textures.get(key) is not None:
                    continue
                if uploaded and time.perf_counter() > deadline:
                    ready = False
                    continue
                resolve_texture(self.textures, key)
                uploaded = True
            if ready:
                out.append(it)
        return out

    def _bind_map(self, unit: int, key) -> None:
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D,
                         resolve_texture(self.textures, key) if key else self.white)

    def _material_entries(self, p: Program, mat: Material) -> list:
        """``[(location, glUniform*, args)]`` of *mat* for *p*, from the material's own cache."""
        cache = mat.__dict__.setdefault("_uniforms", {})
        rev = (mat.__dict__.get("_rev", 0), LIT_OVERRIDE)
        hit = cache.get(p)
        if hit is not None and hit[0] == rev:
            return hit[1]
        fns = {"i": GL.glUniform1i, "f": GL.glUniform1f, "v2": GL.glUniform2f,
               "v3": GL.glUniform3f, "v4": GL.glUniform4f}
        entries = []
        for kind, name, value in _material_values(mat):
            loc = p.loc(name)
            if loc >= 0:
                entries.append((loc, fns[kind], value))
        # stored straight into ``__dict__``: an attribute set would move the
        # revision it is keyed on
        cache[p] = (rev, entries)
        return entries

    def _setup_toon(self, scene: RenderScene, view, proj, eye) -> Program:
        p = self.toon.use()
        p.set_mat4("uView", view)
        p.set_mat4("uProj", proj)
        vp = GL.glGetIntegerv(GL.GL_VIEWPORT)
        p.set_vec2("uDitherViewport", (float(vp[2]), float(vp[3])))
        li = scene.light
        p.set_vec3("uLightDir", li.direction)
        p.set_vec3("uCharaLightDir", scene_chara_light_dir(scene))
        p.set_vec3("uLightColor", _lin3(li.color))
        p.set_float("uLightIntensity", li.intensity)
        # _LightColor0 as the character shaders see it: the character light's
        # own colour times its own intensity, folded here so the fragment
        # program does one multiply fewer and cannot pick up the scene light's
        # Unity converts a light's colour like any other, then multiplies by
        # the intensity — the intensity is a scalar and stays outside.
        p.set_vec3("uCharaLightColor",
                   tuple(c * float(li.chara_intensity)
                         for c in _lin3(li.chara_color)))
        p.set_vec3("uAmbient", _lin3(li.ambient))
        p.set_vec3("uRimLightDir", li.rim_direction)
        p.set_vec3("uCameraPos", eye)
        # the camera's own axes in the world (the view matrix's rows)
        p.set_vec3("uCamRight", tuple(float(v) for v in view[0, :3]))
        p.set_vec3("uCamFwd", tuple(-float(v) for v in view[2, :3]))
        (_a, _c, _t, ef_colour, ef_intensity), volume = ef_lamp(li)
        p.set_vec3("uEfLightDir", ef_light_dir(scene))
        p.set_vec3("uEfLightColor", tuple(c * float(ef_intensity) for c in _lin3(ef_colour)))
        p.set_float("uEfShadowOff", 1.0 if volume.ignore_main_shadow else 0.0)
        ungrade = self._ungrade_texture() if scene.post.lut > 0.001 and self._grading_lut() else None
        self._ef_ungrade = (ungrade, float(scene.post.lut) if ungrade else 0.0)
        GL.glActiveTexture(GL.GL_TEXTURE0 + UNGRADE_TEX_UNIT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, ungrade or self.white)
        p.set_int("uEfUngradeLut", UNGRADE_TEX_UNIT)
        p.set_float("uEfUngrade", self._ef_ungrade[1])
        p.set_vec4("uEfScene", (_EF_SCENE[0] * volume.main_light, _EF_SCENE[1] * volume.env_shadow,
                                _EF_SCENE[2] * volume.env_light, _EF_SCENE[3]))
        p.set_vec4("uEfTint", (*volume.tint, 1.0 if volume.tint_on else 0.0))
        p.set_vec3("uEfTintSkin", volume.skin_tint)
        p.set_vec3("uEfAmbDir", volume.ambient_dir)
        p.set_vec3("uEfAmbGrad", volume.ambient_grad)
        p.set_vec2("uEfNdl", (volume.ndl_bias, volume.ndl_scale))
        p.set_int("uFogEnable", 1 if li.fog and scene_effects(scene)["Fog"] else 0)
        p.set_vec3("uFogColor", _lin3(li.fog_color))
        p.set_vec2("uFogRange", li.fog_range)
        on = bool(getattr(self, "_shadow_on", False)) and self._shadow_fbo
        p.set_float("uShadowOn", self._shadow_texel if on else 0.0)
        p.set_float("uShadowBias", self._shadow_bias)
        p.set_float("uShadowTexel", getattr(self, "_shadow_texel_world", 0.0))
        p.set_mat4("uLightMatrix", self._light_mat)
        # ``SHADOW_TEX_UNIT``: bound **per scene** rather than per material,
        # which is why it lives here and not in ``_material_uniforms``, on a
        # unit no material touches.  Getting a unit wrong here is silent: a
        # sampler pointing at one somebody else rebinds reads **another real
        # texture**, not black (六之六's eviction family) -- which is what
        # happened when the highlight took unit 9 (§7.553).
        GL.glActiveTexture(GL.GL_TEXTURE0 + SHADOW_TEX_UNIT)
        GL.glBindTexture(GL.GL_TEXTURE_2D,
                         self._shadow_fbo.depth if on else self.white)
        p.set_int("uShadowMap", SHADOW_TEX_UNIT)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        return p

    @staticmethod
    def _is_transparent(mat: Material) -> bool:
        # ``blend`` is Illusion's ``_Mode 3`` (SrcAlpha / OneMinusSrcAlpha,
        # ZWrite off).  It is independent of the cut-out: toon_eyew_lod0
        # materials both clip at 0.5 *and* blend, so a material can want the
        # blended pass while still carrying an alpha test.
        return mat.blend or (mat.base_color[3] < 0.999 and mat.alpha_test <= 0.01)

    def _draw_opaque(self, items, scene, view, proj, eye,
                     defer: bool = True) -> list:
        """Everything opaque except the front hair, which is held back.

        The fringe is the one surface in the game that reads the stencil
        buffer, and Illusion puts it at render queue **2850** — after the face
        (2350) and after the eye/eyebrow sheets when those are in their
        foreground state (2800‥2803), before them when they are not
        (2900‥2903).  Our two-pass split cannot express that on its own: opaque
        always precedes transparent, so the fringe would always win.  Holding
        it back and letting :meth:`_draw_foreground` place it is the same
        ordering the queue numbers state.
        """
        p = self._setup_toon(scene, view, proj, eye)
        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)
        masked: list = []
        for it in items:
            p.set_mat4("uModel", it.transform)
            self._bind_skin(p, it)
            for part in it.model.visible_parts():
                if self._is_transparent(part.material):
                    continue
                if defer and part.material.stencil_reject:
                    masked.append((it, part))
                    continue
                if part.material.two_sided:
                    GL.glDisable(GL.GL_CULL_FACE)
                stencil = part.material.stencil_state
                if stencil is not None:
                    _stencil_begin(stencil)
                self._part_skin(p, part)
                self._material_uniforms(p, part.material, scene.light, eye)
                self.gpu_for(it.model, part).draw()
                self.stats["draws"] += 1
                self.stats["tris"] += part.mesh.tri_count
                if stencil is not None:
                    _stencil_end()
                if part.material.two_sided:
                    GL.glEnable(GL.GL_CULL_FACE)
        return masked

    def _draw_foreground(self, masked, items, scene, view, proj, eye,
                         w: int = 0, h: int = 0, outline: bool = False) -> set:
        """The eye/eyebrow sheets that sit *in front of* the fringe, then it.

        ``Stencil { Ref 2, Comp Always, Pass Replace }`` on ``toon_eyew_lod0``
        / ``toon_eye_lod0`` / ``toon_textureanimation``, and
        ``Stencil { Ref 2, Comp NotEqual, Pass Keep }`` on ``main_hair_front``
        — the only four shaders in the whole install that touch the buffer.
        The sheets stamp where they draw (``discard`` skips the write, so the
        cut-out has the shape of the lash, not of its quad) and the fringe is
        rejected there.

        Returns the parts drawn here, so the ordinary transparent pass does not
        draw them a second time.  A character whose card leaves both settings
        off contributes nothing to the first list, and then this is exactly the
        old behaviour: the fringe draws, writes depth, and the sheets behind it
        fail ``GL_LEQUAL``.
        """
        early: set = set()
        sheets = []
        for it in items:
            for part in it.model.visible_parts():
                if not (part.foreground and part.material.stencil_write):
                    continue
                if part.mesh.vertex_count == 0:
                    continue
                sheets.append((it, part))
        if not sheets and not masked:
            return early
        p = self._setup_toon(scene, view, proj, eye)
        if sheets:
            # By item first so ``_group_by_item`` sees each character's sheets
            # in one run; within a character it is the declared queue order,
            # the same key ``_draw_transparent`` sorts on.  The order *between*
            # characters does not matter — no two of them overlap here.
            sheets.sort(key=lambda t: (id(t[0]), t[1].layer))
            GL.glEnable(GL.GL_STENCIL_TEST)
            GL.glDepthMask(GL.GL_FALSE)
            # Two draws where Illusion has one, and on purpose.  A discarded
            # fragment writes no stencil, so in the game the stamp has the shape
            # of the *ink*: ``toon_eyew_lod0`` discards below 0.5 in all four of
            # its fragment programs.  Our sheets do not clip — ``shader_clip``
            # gates on material keywords these shaders never declare, which is a
            # standing question with two callers of its own (§7.90) and not one
            # to settle on the way past.  So the mask is stamped by a
            # colour-less draw at the program's threshold, and the colour draw
            # that follows is bit-for-bit what the ordinary transparent pass
            # would have produced.  Changing it here instead would have thinned
            # every foreground eyebrow by 41% of its pixels and left the other
            # cards' alone — two different eyebrows for one card field.
            GL.glStencilFunc(GL.GL_ALWAYS, 2, 0xFF)
            GL.glStencilOp(GL.GL_KEEP, GL.GL_KEEP, GL.GL_REPLACE)
            GL.glStencilMask(0xFF)
            GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
            for it, parts in _group_by_item(sheets):
                p.set_mat4("uModel", it.transform)
                self._bind_skin(p, it)
                for part in parts:
                    if part.material.stencil_clip <= 0.0:
                        continue             # toon_eye_lod0: stamps its quad
                    self._part_skin(p, part)
                    self._material_uniforms(p, part.material, scene.light, eye)
                    p.set_float("uAlphaTest", part.material.stencil_clip)
                    self.gpu_for(it.model, part).draw()
                    self.stats["draws"] += 1
            # The iris shader has no discard at all, so its quad *is* the mask;
            # it stamps in its own colour draw below rather than here.
            GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            for it, parts in _group_by_item(sheets):
                p.set_mat4("uModel", it.transform)
                self._bind_skin(p, it)
                for part in parts:
                    # Only the blended sheets are drawn here.  The sclera comes
                    # out of ``_is_transparent`` as opaque (``cf_m_sirome_00``
                    # ships ``_Mode 0`` / ``_ZWrite 1`` even though its shader's
                    # pass blends), so it has already been drawn in
                    # ``_draw_opaque`` — ahead of the held-back fringe either
                    # way — and drawing it again here would double-blend it.
                    if not self._is_transparent(part.material):
                        continue
                    GL.glStencilMask(0xFF if part.material.stencil_clip <= 0.0
                                     else 0x00)
                    if part.material.two_sided:
                        GL.glDisable(GL.GL_CULL_FACE)
                    self._part_skin(p, part)
                    self._material_uniforms(p, part.material, scene.light, eye)
                    self.gpu_for(it.model, part).draw()
                    self.stats["draws"] += 1
                    early.add(id(part))
                    if part.material.two_sided:
                        GL.glEnable(GL.GL_CULL_FACE)
            GL.glStencilMask(0xFF)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
        if masked:
            GL.glEnable(GL.GL_STENCIL_TEST)
            GL.glStencilMask(0x00)               # the fringe writes no stencil
            GL.glStencilFunc(GL.GL_NOTEQUAL, 2, 0xFF)
            GL.glStencilOp(GL.GL_KEEP, GL.GL_KEEP, GL.GL_KEEP)
            if outline:
                self._draw_outlines(None, view, proj, w, h, eye, pairs=masked)
                p = self._setup_toon(scene, view, proj, eye)
            for it, parts in _group_by_item(masked):
                p.set_mat4("uModel", it.transform)
                self._bind_skin(p, it)
                for part in parts:
                    if part.material.two_sided:
                        GL.glDisable(GL.GL_CULL_FACE)
                    self._part_skin(p, part)
                    self._material_uniforms(p, part.material, scene.light, eye)
                    self.gpu_for(it.model, part).draw()
                    self.stats["draws"] += 1
                    self.stats["tris"] += part.mesh.tri_count
                    if part.material.two_sided:
                        GL.glEnable(GL.GL_CULL_FACE)
        GL.glStencilMask(0xFF)
        GL.glStencilFunc(GL.GL_ALWAYS, 0, 0xFF)
        GL.glDisable(GL.GL_STENCIL_TEST)
        return early

    def _draw_transparent(self, items, scene, view, proj, eye,
                          skip: set | None = None) -> None:
        queue = []
        for it in items:
            for part in it.model.visible_parts():
                if not self._is_transparent(part.material):
                    continue
                if part.mesh.vertex_count == 0:
                    continue
                if skip and id(part) in skip:
                    continue
                c = part.mesh.positions.mean(axis=0)
                wc = m3.transform_point(it.transform, c)
                z = float(np.linalg.norm(np.asarray(eye) - wc))
                queue.append((z, it, part))
        if not queue:
            return
        # Declared order first, distance only as a tie-break.  These sheets are
        # near-coplanar — the eye white, iris, lid shadow and lashes sit within
        # a millimetre of each other — so sorting them by camera distance picks
        # an order that flickers with the head's angle.  Unity does not either:
        # Illusion gives each material an explicit render queue (2900 eyebrow,
        # 2901 sclera, 2902 iris and lid shadow, 2903 lashes), which is what
        # ``Part.layer`` carries.
        # Origin's render queue first when the material carries one (§7.379):
        # the eyes are 2900‥2903, a glasses lens's shader tag is 4010 and the
        # wedding veil's own queue is 3600, so a lens or veil draws **after**
        # the eyes behind it — which ``Part.layer`` (accessories 3, face 10‥20)
        # had the other way round.  A part baked before the queue was stored
        # reads 3000, the Transparent tag every blended shader here starts from.
        queue.sort(key=lambda t: (t[2].material.render_queue
                                  if t[2].material.render_queue >= 0
                                  else TRANSPARENT_QUEUE,
                                  t[2].layer, -t[0]))
        p = self._setup_toon(scene, view, proj, eye)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_FALSE)
        cur = None
        for _, it, part in queue:
            if cur is not it:
                p.set_mat4("uModel", it.transform)
                self._bind_skin(p, it)
                cur = it
            if part.material.two_sided:
                GL.glDisable(GL.GL_CULL_FACE)
            # ``ZWrite`` is per pass in Unity: a veil writes depth, an iris
            # does not (``textures.BLEND_ZWRITE``, §7.376).
            GL.glDepthMask(GL.GL_TRUE if part.material.depth_write else GL.GL_FALSE)
            factors = part.material.blend_func
            if factors is not None:
                GL.glBlendFunc(*factors)
            if part.material.depth_less:
                GL.glDepthFunc(GL.GL_LESS)
            stencil = part.material.stencil_state
            if stencil is not None:
                _stencil_begin(stencil)
            self._part_skin(p, part)
            self._material_uniforms(p, part.material, scene.light, eye)
            self.gpu_for(it.model, part).draw()
            self.stats["draws"] += 1
            if stencil is not None:
                _stencil_end()
            if part.material.depth_less:
                GL.glDepthFunc(GL.GL_LEQUAL)
            if factors is not None:
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            if part.material.two_sided:
                GL.glEnable(GL.GL_CULL_FACE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

    # ------------------------------------------------------------------
    def _draw_grid(self, scene: RenderScene, view, proj) -> None:
        bg = scene.background
        if not (bg.grid and getattr(scene, "show_grid", True)):
            return
        key = (round(bg.grid_y, 4),)
        if self._grid_key != key:
            v, c = make_grid(4.0, 0.25, bg.grid_y)
            self._grid_batch.upload(v, c)
            self._grid_key = key
        p = self.line.use()
        p.set_mat4("uMVP", proj @ view)
        p.set_float("uAlpha", 1.0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_FALSE)
        self._grid_batch.draw()
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

    def _draw_overlay(self, lines, view, proj) -> None:
        verts, colors = lines
        if verts is None or len(verts) == 0:
            return
        self._lines.upload(np.asarray(verts, np.float32), np.asarray(colors, np.float32))
        p = self.line.use()
        p.set_mat4("uMVP", proj @ view)
        p.set_float("uAlpha", 1.0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDisable(GL.GL_DEPTH_TEST)
        self._lines.draw()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)

    # ------------------------------------------------------------------
    def _draw_bloom(self, scene: RenderScene) -> None:
        """Bright-pass + blur into the quarter-res target."""
        fb = self._bloom_fbo
        if fb is None or not fb.valid:
            return
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        fb.bind()
        p = self.bloom.use()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._resolve.color)
        p.set_int("uScene", 0)
        p.set_vec2("uTexel", (1.0 / fb.width, 1.0 / fb.height))
        p.set_float("uThreshold", scene.post.bloom_threshold)
        GL.glBindVertexArray(self._empty_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def _draw_post(self, scene: RenderScene, w: int, h: int) -> None:
        ps = scene.post
        # ``enabled = False`` means *our* grade is off, and the pass is then
        # running only to apply Illusion's LUT — so every term below has to be
        # neutralised rather than the pass skipped.
        ours = ps.enabled
        # the player's グラフィック switches, which Origin applies to every camera
        # carrying ``CameraEffectorConfig`` (§7.554)
        fx = scene_effects(scene)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        p = self.post.use()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._resolve.color)
        p.set_int("uScene", 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._resolve.depth or self.white)
        p.set_int("uDepth", 1)
        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D,
                         self._bloom_fbo.color if self._bloom_fbo else self.white)
        p.set_int("uBloomTex", 2)
        # bilinear along u so the two slice samples the shader lerps between
        # interpolate inside a slice, clamped so slice 0 and 31 do not wrap
        lut_tex = self._grading_lut()
        GL.glActiveTexture(GL.GL_TEXTURE3)
        GL.glBindTexture(GL.GL_TEXTURE_2D, lut_tex or self.white)
        p.set_int("uLut", 3)
        p.set_float("uLutAmount", ps.lut if lut_tex else 0.0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        p.set_vec2("uTexel", (1.0 / w, 1.0 / h))
        p.set_float("uBloom", ps.bloom if ours and fx["Bloom"] else 0.0)
        p.set_float("uBloomThreshold", ps.bloom_threshold)
        p.set_float("uVignette", ps.vignette if ours and fx["Vignette"] else 0.0)
        p.set_float("uSaturation", ps.saturation if ours else 1.0)
        p.set_float("uContrast", ps.contrast if ours else 1.0)
        p.set_float("uBrightness", ps.brightness if ours else 0.0)
        p.set_vec3("uTint", ps.tint if ours else (1.0, 1.0, 1.0))
        p.set_float("uDofStrength", ps.dof_strength if ours and fx["DepthOfField"] else 0.0)
        p.set_vec2("uDofRange", (ps.dof_focus, scene.camera.far))
        p.set_float("uChromatic", ps.chromatic if ours else 0.0)
        p.set_int("uSepia", 1 if (ours and ps.sepia) else 0)
        GL.glBindVertexArray(self._empty_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_DEPTH_TEST)

    # ------------------------------------------------------------------
    def release(self) -> None:
        self.native.release()
        self.clear_cache()
        for prog in (self.toon, self.outline, self.post, self.bloom, self.line,
                     self.sky, self.sky_image, self.shadow):
            prog.release()
        if getattr(self, "_bone_tex", 0):
            GL.glDeleteTextures(1, [self._bone_tex])
            self._bone_tex = 0
        if getattr(self, "_shadow_fbo", None) is not None:
            self._shadow_fbo.release()
            self._shadow_fbo = None
        self.textures.release(include_pinned=True)
        self._lines.release()
        self._grid_batch.release()
        if self._msaa:
            self._msaa.release()
        if self._resolve:
            self._resolve.release()
        if self._bloom_fbo:
            self._bloom_fbo.release()
        if self._empty_vao:
            GL.glDeleteVertexArrays(1, [self._empty_vao])
            self._empty_vao = 0
