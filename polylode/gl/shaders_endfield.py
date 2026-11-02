"""The ``HGRP/CharacterNPR*`` character lighting, as this renderer draws it.

Written from the forward passes' compiled programs (``tools/ef_shader_dump.py``;
the reading is ``docs/updates/2026-10-02/7.566.md``).  What is here is the
**character lighting by the main light**: material colour, shadow colour, the
diffuse ramp, the specular terms and the per-shader parts (skin, the face's SDF
shadow, hair, eye).  The pipeline around it — clustered point lights, the
irradiance volume, fog, rain and snow — is the game's renderer, not the
material's look, and is not reproduced.

**What the game sets per frame and this renderer has to supply.**  The programs
read sixteen ``_CharacterParams`` vectors, the light, the exposure and a shadow
mask from the pipeline; none of it is in the converted data.  :data:`SCENE`
is this renderer's answer, one named value each, and every one of them is
*ours*, not the game's:

* the light is the scene's character light (direction, colour × intensity);
* the ambient is the scene's ambient colour, used the way the game's
  "fixed ambient" mode uses its own (``P1.y`` ≥ 0.5: no irradiance volume);
* ``direct`` is 0.5 so that a white light at intensity 1 over a full ambient
  leaves a lit albedo unchanged — a calibration of ours;
* rim light, the N·L bias, the face's second SDF light and the weather are
  off (their strengths are 0), because nothing here says what a scene sets;
* the environment a metal reflects is uniform white: the game's
  ``_CharMaxCubemap`` is produced at run time and is not an asset.  (The game
  multiplies the cubemap by the ambient colour, and so does this.)

Vectors are in **Unity's handedness** inside the block (this project mirrors
X on import): the head frame, the SDF lookup and the hair tangent tie world
directions to texture space, and transcribing them verbatim only works in the
space they were written for.  Tangents come from screen-space derivatives —
the baked meshes carry none.
"""

from dataclasses import dataclass

#: ``Material.toon_family`` values of the four surface shaders and the overlay.
FAMILY_MAIN, FAMILY_SKIN, FAMILY_HAIR, FAMILY_EYE, FAMILY_OVERLAY = 20, 21, 22, 23, 24
#: A converted source's Blender Principled BSDF (``converted.runtime_items.PRINCIPLED``): not this
#: game's, but drawn on the same character path (its lamp, its ungrade), by :data:`FUNCTIONS`' ``efPrincipled``.
FAMILY_PRINCIPLED = 25
FAMILIES = {
    "HGRP/CharacterNPR": FAMILY_MAIN, "HGRP/CharacterNPR_Skin": FAMILY_SKIN,
    "HGRP/CharacterNPR_Hair": FAMILY_HAIR, "HGRP/CharacterNPR_Eye": FAMILY_EYE,
    "HGRP/CharacterNPR_OverlayShadow": FAMILY_OVERLAY, "Blender/PrincipledBSDF": FAMILY_PRINCIPLED}

#: Bits of ``uEfFlags``: the material keywords the programs branch on.
FLAGS = {
    "_DIFF_RAMP_ON": 1, "_SHADOW_LUT_TEX": 2, "_NORMALMAP": 4,
    "_METALLICSPECGLOSSMAP": 8, "_SPEC_RAMP_ON": 16, "_SDFLIGHTMAP": 32,
    "_EMOTION_MAP": 64, "_HIGHLIGHT_MAP": 128, "_MATCAP_ON": 256,
    "_EYE_HIGHLIGHT": 512, "_SPECULAR_LINE": 1024, "_EMISSION": 2048,
    "_SPECULAR_NORMALMAP": 4096, "_DRAW_UNDER_BROW": 8192, "_ANISOTROPY_SPECULAR_ON": 65536,
    "_CLEARCOAT": 131072}
#: Not a keyword of the game's: the part carries a colour field per vertex
#: (``endfield/shading.py``: ``VERTEX_FIELD``), which the skin program mixes
#: with its base map.
VERTEX_FIELD_FLAG = 16384
#: … and this one a tint per vertex that multiplies the base colour (``VERTEX_TINT``).
VERTEX_TINT_FLAG = 32768

#: This renderer's stand-ins for what the game's pipeline sets per frame.
SCENE = {
    "direct": 0.5,            # P0.y: scale on (ambient + light) where lit
    "shadow_intensity": 1.0,  # P0.z: scale on the shadow colour
    "ambient": 1.0,           # P0.w: scale on the ambient-only term
    "specular": 1.0,          # P13.w: scale on the specular terms
}

@dataclass(frozen=True)
class CharacterVolume:
    """What a scene of the game sets for how its characters are lit.

    The fields of the pipeline's ``HGCharacterVolume`` this renderer reads,
    each named for what it does in the programs here.  Which shader parameter
    a field reaches was read off the shaders, not off the pipeline's code:

    * the two fixed colours and the switch — the skin program reads its fixed
      ambient colour from one parameter and every other program from another
      (``_CharacterParams3`` / ``2``), behind one flag (``Params1.y``); the
      volume has exactly such a pair with one switch (``charShadowTintColor``,
      ``charSkinShadowTintColor``, ``charShadowTintControl``).  The ambient is
      this colour where a surface is unlit, which is what tints a shadow;
    * the ambient's gradient — a direction and ``(offset, scale, base)``
      (``Params6``, ``Params7``) against the volume's two vectors of that very
      shape (``charGlobalAmbientParam0`` / ``1``);
    * the three multipliers — on the lit term, the shadow colour and the
      ambient-only term (``Params0.y`` / ``.z`` / ``.w``) against
      ``charMainLightMultiplier``, ``charEnvShadowMultiplier`` and
      ``charEnvLightMultiplier``.  They scale this renderer's own stand-ins
      (:data:`SCENE`), which stand for the game's defaults of 1;
    * ``ndl_bias`` — the programs add ``Params11.w × Params12.x`` to N·L.
      That ``charMainLightRangeBias`` is the first and the second is 1 while
      the volume controls the main light is **inferred** from how the fields
      are grouped; the product is what the programs use.
    """
    main_light: float = 1.0
    env_light: float = 1.0
    env_shadow: float = 1.0
    tint_on: bool = False
    tint: tuple = (1.0, 1.0, 1.0)
    skin_tint: tuple = (1.0, 1.0, 1.0)
    ambient_dir: tuple = (0.0, 0.0, 0.0)
    ambient_grad: tuple = (0.0, 0.0, 1.0)        # offset, scale, base: 1 everywhere
    ndl_bias: float = 0.0
    ndl_scale: float = 0.0
    #: ``charIgnoreMainLightShadow``: the characters take no cast shadow of the main light
    ignore_main_shadow: bool = False


#: no volume: every term as this renderer drew it before it read one
NO_VOLUME = CharacterVolume()
#: The game's volumes this renderer knows, by the lamp that goes with each
#: (``gl/renderer.EF_CHARA_LIGHT``).  ``ef_charinfo``: the character-information
#: screen, ``prefabs/charinfo/charinfo_volume.asset`` (§7.916, §7.917).
VOLUMES = {
    "ef_charinfo": CharacterVolume(
        main_light=0.9, env_light=0.8, env_shadow=0.8, tint_on=True,
        tint=(0.7830188, 0.8293082, 1.0), skin_tint=(1.0, 0.78114647, 0.68490565),
        ambient_dir=(0.0, 1.0, 0.0), ambient_grad=(0.15, 1.5, 0.5),
        ndl_bias=-0.4, ndl_scale=1.0, ignore_main_shadow=True),
}

#: The inverse of the frame's colour grade, for every program of this game that writes a colour:
#: the surfaces (``efFragment``) and their outline (``shaders.OUTLINE_FS``, colour mode 2).  The
#: outline is the darkened base colour, so left out it is the one thing the grade still crushes —
#: skin lines came out red (§7.927's orange-red, on a colour with less light in it).
UNGRADE_UNIFORMS = """
uniform sampler2D uEfUngradeLut; // the inverse of the frame's colour grade (origin.grading.inverse)
uniform float uEfUngrade;    // how much of that grade the frame will get (0: none)
"""

UNGRADE = """
// The frame's colour grade is the other game's camera effect, and it is one
// pass over everything.  What these programs made is sent out as the colour
// that pass turns back into it: to gamma the way the pass does, through the
// inverse table (addressed as ``POST_FS.applyLut`` addresses its own), and
// back to linear.
vec3 efUngrade(vec3 linearRgb) {
    if (uEfUngrade <= 0.001) return linearRgb;
    vec3 c = clamp(max(1.055 * pow(max(linearRgb, vec3(0.0)), vec3(0.41666667)) - 0.055, vec3(0.0)),
                   0.0, 1.0);
    float b = c.b * 31.0;
    float f = fract(b);
    vec2 uv = vec2(c.r * 0.03027344 + 0.00048828 + floor(b) * 0.03125, c.g * 0.96875 + 0.015625);
    vec3 g = mix(texture(uEfUngradeLut, uv).rgb, texture(uEfUngradeLut, uv + vec2(0.03125, 0.0)).rgb, f);
    vec3 back = g * (g * (g * 0.305306 + 0.682171) + 0.012523);
    return mix(linearRgb, back, uEfUngrade);
}
"""

UNIFORMS = """
// ---- HGRP/CharacterNPR* (families 20..24, see gl/shaders_endfield.py) ------
uniform int   uEfFlags;
uniform vec4  uEfShadow;     // _ShadowColorBrightness, _ShadowColorSaturation, _BumpScale, _AlphaPremultiply
uniform vec4  uEfPbr;        // _Metallic, _Smoothness, _Specular, _SpecRampIridescentMode
uniform vec4  uEfSkin;       // _SkinRimOffScale, _FaceRimOffScale, _EmotionIndex, _EmotionBlend
uniform vec4  uEfRimColor;   // _SDFRimColor
uniform vec4  uEfVec;        // skin: _HighlightMapVector; eye: _ParallaxScale, _MatcapNormalScale
uniform vec4  uEfColorA;     // eye: _EyeHighLightColor; hair: _AnisotropyColor2
uniform vec4  uEfColorB;     // eye: _EyeScatteringColor
uniform vec4  uEfColorC;     // eye: _MatcapColor; main: _EmissionColor.rgb, _EmissionBrightness
uniform vec4  uEfAniso;      // main: _AnisotropyDirectionMain, _AnisotropyDirectionAdditional,
                             // _AnisotropyOffsetAdditional, _AnisotropyIntensityMultiplier
uniform vec4  uEfAnisoColor; // main: _AnisotropyColorAdditional.rgb, _AnisotropyUseGeometryTangent
uniform vec4  uEfCoat;       // main: _ClearCoatSmoothness, _ClearCoatMetallic, _ClearCoatNormalMode, -
uniform vec4  uEfCoatColor;  // main: _ClearCoatColor
uniform vec4  uEfHairA;      // _AnisotropyValue, _AnisotropyValue2, _AnisotropyIntensity, _AnisotropyEdgeFade
uniform vec4  uEfHairB;      // _AnisotropyDirX, _AnisotropyRange2, _SpecBumpScale, _UseLineMap
uniform vec4  uEfHairC;      // _LineAmount, _LineValue, _LineRange, _LineIntensity
uniform vec4  uEfHairD;      // _LineMap_ST
uniform vec4  uEfMisc;       // _LineSaturation, _UseGrayAsAlpha (overlay), _HairBrowMaskThreshold, -;
                             // skin: -, -, -, occlusion without a base map
uniform mat4  uEfHead;       // the head's frame as a turn from its rest, world space
uniform vec3  uCamFwd;       // the camera's forward, world space
uniform vec3  uCamRight;     // the camera's right, world space
uniform vec4  uEfScene;      // direct, shadow intensity, ambient, specular (SCENE x the volume's)
uniform vec4  uEfTint;       // the volume's fixed ambient colour; w: use it (CharacterVolume)
uniform vec3  uEfTintSkin;   // ... the skin program's
uniform vec3  uEfAmbDir;     // the ambient gradient's direction, Unity space
uniform vec3  uEfAmbGrad;    // ... its offset, scale, base
uniform vec2  uEfNdl;        // added to N.L (x times y); y also switches the backlight push off
uniform vec3  uEfLightDir;   // the way this game's own lamp travels (renderer.ef_light_dir)
uniform vec3  uEfLightColor; // ... its colour x intensity
uniform float uEfShadowOff;  // 1: no cast shadow of the main light (the volume's)
""" + UNGRADE_UNIFORMS

FUNCTIONS = """
const vec3 EF_LUMA = vec3(0.212673, 0.715152, 0.072175);

bool efOn(int bit) { return (uEfFlags & bit) != 0; }
vec3 efU(vec3 v) { return vec3(-v.x, v.y, v.z); }          // this space <-> Unity's
float efLuma(vec3 c) { return dot(c, EF_LUMA); }
vec3 efSat(vec3 c, float k) { float l = efLuma(c); return l + (c - l) * k; }
float efSmooth01(float x) { return x * x * (3.0 - 2.0 * x); }
float efMax3(vec3 c) { return max(c.r, max(c.g, c.b)); }
float efMin3(vec3 c) { return min(c.r, min(c.g, c.b)); }

// A ramp is read with a clamping sampler in the game; the lookup stays inside
// the first and last texel centres so the edge does not blend with its opposite.
vec4 efRamp(sampler2D ramp, float u, float v) {
    vec2 size = vec2(textureSize(ramp, 0));
    vec2 uv = clamp(vec2(u, v), 0.5 / size, 1.0 - 0.5 / size);
    return textureLod(ramp, uv, 0.0);
}

// _SHADOW_LUT_TEX: a 32-cubed colour table, 32 slices side by side, blue picks
// the slice.  The table is indexed by the sRGB-encoded base colour.
vec3 efShadowLut(sampler2D lut, vec3 linearRgb) {
    vec3 c = clamp(mix(linearRgb * 12.92,
                       1.055 * pow(max(linearRgb, vec3(0.0)), vec3(0.416667)) - 0.055,
                       step(vec3(0.0031308), linearRgb)), 0.0, 1.0);
    float slice = floor(c.b * 31.0);
    vec2 uv = vec2(slice / 32.0 + c.r * (31.0 / 1024.0) + 0.5 / 1024.0,
                   c.g * (31.0 / 32.0) + 0.5 / 32.0);
    vec3 lo = textureLod(lut, uv, 0.0).rgb;
    vec3 hi = textureLod(lut, uv + vec2(1.0 / 32.0, 0.0), 0.0).rgb;
    return mix(lo, hi, c.b * 31.0 - slice);
}

struct EfLight {
    vec3 L;          // toward the light
    vec3 Lc;         // light colour x intensity
    vec3 ambTint;    // the ambient's colour
    vec3 fwd;        // camera forward
    vec3 right;      // camera right
    float S;         // 1 = not in a cast shadow
};

// Without _DIFF_RAMP_ON the main shader's "ramp" is this step on N.L (and on
// N.forward for the view term), and has no colour of its own.
float efStep(float x) {
    return efSmooth01(clamp((x - 0.25) * 1.333333, 0.0, 1.0));
}

// The ambient's gradient over a surface: more from the side the volume names.
// The skin and eye programs take the normal laid flat for it.
float efAmbGrad(vec3 Nh) {
    return clamp(dot(Nh, uEfAmbDir) + uEfAmbGrad.x, 0.0, 1.0) * uEfAmbGrad.y + uEfAmbGrad.z;
}
vec3 efFlat(vec3 N) { return normalize(vec3(N.x, 0.000061, N.z)); }

// The light a surface gets, given how lit it is (0 shadow .. 1 lit).
vec3 efLighting(EfLight li, float lit, float grad) {
    vec3 amb = mix(li.ambTint, vec3(1.0), lit) * grad;
    vec3 direct = mix(vec3(efLuma(li.Lc)), li.Lc, lit);
    vec3 lightTerm = (amb + direct) * uEfScene.x;
    return mix(amb * uEfScene.z, lightTerm, li.S);
}

// Material colour -> the colour under this light, before the light multiplies it.
//   ramp    the diffuse ramp at N.L (rgb tone, a how lit)
//   viewA   the ramp's alpha at N.forward (1 for the shaders that have none)
//   occA    the occlusion channel
vec3 efDiffuse(vec3 albedo, vec3 shadowCol, vec4 ramp, float viewA, float occA,
               float S, out float lit, out float litF) {
    float chroma = efMax3(ramp.rgb) - efMin3(ramp.rgb);
    lit = min(ramp.a, occA);
    vec3 deep = efSat(shadowCol * 0.65, 1.2);
    vec3 c1 = mix(deep, shadowCol, clamp(viewA * occA + ramp.a, 0.0, 1.0));
    c1 = mix(c1, albedo, lit);
    vec3 c2 = c1 * mix(vec3(1.0), ramp.rgb, chroma);
    float k = clamp(efLuma(c1) / max(efLuma(c2), 0.001), 0.0, 1.5);
    vec3 castCol = mix(shadowCol, efSat(albedo, 1.2), viewA * occA);
    litF = mix(viewA * occA, lit, S);
    return mix(castCol, c2 * k, S);
}

// The half vector the specular terms use: not normalize(L + V).
vec3 efHalf(EfLight li, vec3 V) {
    vec3 tilt = normalize(vec3(li.fwd.x, mix(0.5, li.L.y, li.S), li.fwd.z));
    return normalize(li.L * li.S + 2.0 * tilt + V * (li.S + 2.0));
}

float efGgx(float ndh, float ndv, float alpha) {
    float a2 = alpha * alpha;
    float d = (ndh * a2 - ndh) * ndh + 1.0;
    float D = a2 / max(d * d, 1e-12);
    return clamp(D * 0.5 / (2.0 * ndv + alpha + 1e-4) - 6.1e-5, 0.0, 20.0);
}

// The main shader's stretched GGX (_ANISOTROPY_SPECULAR_ON): D of a lobe
// ax along T, ay along B, without its 1/pi, as the program writes it —
// (ax ay)^3 / ((ay T.H)^2 + (ax B.H)^2 + (ax ay N.H)^2)^2 — and 1 where the
// two sides are equal.
float efGgxAniso(vec3 T, vec3 B, vec3 N, vec3 H, float ax, float ay) {
    float a = ax * ay;
    vec3 v = vec3(dot(T, H) * ay, dot(B, H) * ax, dot(N, H) * a);
    float d = dot(v, v);
    d *= d;
    float a3 = a * a * a;
    return d != a3 ? a3 / d : 1.0;
}

// What the specular terms are lit by.
vec3 efSpecLight(vec3 lighting, float litF) {
    return lighting * (0.5 + 0.5 * litF) * mix(uEfScene.y, 1.0, litF);
}

vec3 efFinish(vec3 color) {
    float lum = efLuma(color);
    float k = clamp(lum - 0.5, 0.0, 0.5);
    return mix(vec3(lum), color, 1.0 + k * k);
}

// The environment's split-sum terms, as the main shader's rational fit.
vec2 efEnvBrdf(float ndv, float rough) {
    float r2 = rough * rough, r6 = r2 * r2 * r2;
    float v2 = ndv * ndv, v3 = v2 * ndv;
    float aN = (3.32707 * ndv + 0.036546) + (-9.04756 * ndv + 9.0632) * r2;
    float aD = (3.59685 * v2 - 1.36772 * v3 + 1.0)
             + (-16.3174 * v2 + 9.04401 + 9.22949 * v3) * r2
             + (5.56589 + 19.7886 * v2 - 20.2123 * v3) * r6;
    float bN = (-1.28514 * ndv + 0.99044) + (1.29678 - 0.755907 * ndv) * r2;
    float bD = (2.92338 * ndv + 59.4188 * v3 + 1.0)
             + (20.3225 - 27.0302 * ndv + 222.592 * v3) * r2
             + (626.13 * ndv + 316.627 * v3 + 121.563) * r6;
    return vec2(aN / aD, bN / bD);
}

vec3 efNormalMap(mat3 tbn, vec2 uv) {
    // DXT5nm-style: x is stored in r * a
    vec4 t = texture(uNormalTex, uv);
    vec2 xy = vec2(t.r * t.a, t.g) * 2.0 - 1.0;
    float z = sqrt(max(1.0 - min(dot(xy, xy), 1.0), 0.0));
    return normalize(tbn * vec3(xy * uEfShadow.z, z));
}

vec3 efShadowBase(vec3 base) {
    if (efOn(2)) return efShadowLut(uGlassRamp, base);
    return efSat(base * uEfShadow.x, uEfShadow.y);
}

// ---- HGRP/CharacterNPR ------------------------------------------------------
vec4 efMain(EfLight li, vec3 N, vec3 V, mat3 tbn) {
    vec4 base = texture(uTex, vUV) * uBaseColor;
    if (base.a < uAlphaTest) discard;
    // r metallic, g specular level, b occlusion, a smoothness
    vec4 mgm = efOn(8) ? texture(uShadeMask, vUV)
                       : vec4(uEfPbr.x, uEfPbr.z, 1.0, uEfPbr.y);
    vec3 Ngeo = N;
    if (efOn(4)) N = efNormalMap(tbn, vUV);
    float rough = 1.0 - mgm.a;
    float kd = 0.96 * (1.0 - mgm.r);
    vec3 albedo = base.rgb * kd;
    // anisotropic: the map's g is where the stretch is, and the level is 1 there times the multiplier
    bool aniso = efOn(65536);
    float specLevel = aniso ? 1.0 + mgm.g * (uEfAniso.w - 1.0) : mgm.g;
    vec3 f0 = mix(vec3(0.04 * specLevel), base.rgb, mgm.r);
    vec3 f0Plain = f0;
    vec3 shadowCol = efShadowBase(base.rgb) * kd * uEfScene.y;
    float alpha = max(rough * rough, 0.0078125);

    float ndl = clamp(dot(N, li.L) + uEfNdl.x * uEfNdl.y, -1.0, 1.0);
    // backlight: when the light is behind the subject as seen from the camera
    // the terminator is pushed round, unless the camera looks steeply up or down
    float back = clamp(-dot(normalize(li.L.xz), normalize(li.fwd.xz)), 0.0, 1.0);
    float level = efSmooth01(clamp(2.0 * (0.75 - abs(li.fwd.y)), 0.0, 1.0)) * (1.0 - uEfNdl.y);
    ndl = clamp(ndl + back * level * (0.5 - 0.5 * ndl * ndl), -1.0, 1.0);
    vec4 ramp = vec4(1.0, 1.0, 1.0, efStep(ndl));
    float viewA = efStep(dot(N, li.fwd));
    if (efOn(1)) {
        ramp = efRamp(uRamp, 0.5 + 0.5 * ndl, 0.5);
        viewA = efRamp(uRamp, 0.5 + 0.5 * dot(N, li.fwd), 0.5).a;
    }
    float lit, litF;
    vec3 diffuse = efDiffuse(albedo, shadowCol, ramp, viewA, mgm.b, li.S, lit, litF);
    vec3 lighting = efLighting(li, lit, efAmbGrad(N));

    float ndv = clamp(dot(N, V), 0.0, 1.0);
    vec3 H = efHalf(li, V);
    float ndh = dot(N, H);
    float a2 = alpha * alpha;
    float d = (ndh * a2 - ndh) * ndh + 1.0;
    float D = a2 / max(d * d, 1e-12);
    vec3 T = vec3(1.0, 0.0, 0.0), B = vec3(0.0, 1.0, 0.0);
    if (aniso) {
        // the mesh's tangent, or one round the world's up axis
        vec3 Ng = tbn[2];
        vec2 flat2 = vec2(-Ng.z, Ng.x) * inversesqrt(max(dot(Ng.xz, Ng.xz), 6.1e-5));
        T = uEfAnisoColor.w > 0.5 ? tbn[0] : vec3(flat2.x, 0.0, flat2.y);
        T = normalize(T - N * dot(T, N));
        B = cross(N, T);
        D = efGgxAniso(T, B, N, H, alpha * (1.0 + uEfAniso.x * mgm.g),
                       alpha * (1.0 - uEfAniso.x * mgm.g));
    }
    float spec = clamp(D * 0.5 / (2.0 * ndv + alpha + 1e-4) - 6.1e-5, 0.0, 20.0);
    vec3 specColor = f0;
    if (efOn(16)) {
        float dn = D * (a2 + 1e-4);
        vec3 sr = efRamp(uViewRamp, mix(dn, ndv * ndv, uEfPbr.w),
                         rough * (1.0 - mgm.r)).rgb;
        specColor = f0 * sr;
        f0 = mix(f0, f0 * sr, uEfPbr.w);
    }
    vec3 specular = spec * specColor;
    if (aniso) {
        // a second lobe: the half vector leant towards the eye, its own stretch, tinted
        vec3 H2 = normalize(H + V * uEfAniso.z);
        float D2 = efGgxAniso(T, B, N, H2, alpha * (1.0 + uEfAniso.y * mgm.g),
                              alpha * (1.0 - uEfAniso.y * mgm.g));
        specular += f0Plain * uEfAnisoColor.rgb * mgm.g * clamp(D2, 0.0, 20.0);
    }
    // _CLEARCOAT: a coat over the surface, its own normal (the mesh's or the mapped one), smoothness
    // and colour, where its mask is.  What it reflects (Schlick) is taken off the diffuse once and the
    // base specular twice; its own GGX lobe is added.  An environment term of its own below.
    float coat = efOn(131072) ? texture(uGloss, vUV).r : 0.0;
    vec3 coatF0 = vec3(0.0), coatF = vec3(0.0);
    vec3 Nc = N;
    float coatRough = 1.0 - uEfCoat.x, ndvc = 0.0;
    if (coat > 0.001) {
        Nc = normalize(mix(N, Ngeo, uEfCoat.z));
        ndvc = clamp(dot(Nc, V), 0.0, 1.0);
        coatF0 = (0.04 + 0.96 * uEfCoat.y) * uEfCoatColor.rgb;
        if (efOn(16)) coatF0 *= efRamp(uViewRamp, ndvc * ndvc, 0.5).rgb;
        float vdh = clamp(dot(V, H), 0.0, 1.0);
        float f = 1.0 - vdh, f5 = f * f * f * f * f;
        coatF = coatF0 * (1.0 - f5) + f5;
        float ca = max(coatRough * coatRough, 0.0078125);
        float ca2 = ca * ca, cndh = dot(Nc, H);
        float cd = (cndh * ca2 - cndh) * cndh + 1.0;
        cd *= cd;
        float cD = cd != ca2 ? ca2 / cd : 1.0;
        vec3 coatSpec = clamp(coat * coatF * cD * 0.5 / (2.0 * ndvc + ca + 1e-4), 0.0, 20.0);
        vec3 keep = 1.0 - coatF * coat;
        specular = specular * keep * keep + coatSpec;
        diffuse *= keep;
    }
    float premul = mix(1.0, base.a, uEfShadow.w);
    vec3 color = lighting * diffuse * premul
               + efSpecLight(lighting, litF) * specular * uEfScene.w;
    color = efFinish(color);
    if (efOn(2048))
        color += texture(uPattern, vUV).rgb * uEfColorC.rgb * uEfColorC.a * premul;
    // environment: the ambient colour stands in for the game's run-time cubemap
    vec2 ab = efEnvBrdf(ndv, rough);
    float e = max(ab.x + ab.y, 1e-4);
    vec3 dfg = f0 * ab.x + ab.y;
    vec3 env = dfg * (1.0 + f0 * (1.0 - e) / e) * li.ambTint;
    if (coat > 0.001) {
        vec2 cab = efEnvBrdf(ndvc, coatRough);
        float ce = max(cab.x + cab.y, 1e-4);
        env += coat * (coatF0 * cab.x + cab.y) * (1.0 + coatF0 * (1.0 - ce) / ce) * li.ambTint;
    }
    color += env * uEfScene.z * mix(uEfScene.y, 1.0, litF);
    return vec4(color, base.a);
}

// ---- HGRP/CharacterNPR_Skin (body, and the face with _SDFLIGHTMAP) ----------
vec3 efEmotion(vec3 base, vec2 uv) {
    if (!efOn(64)) return base;
    float i = uEfSkin.z * 0.5;
    vec2 cell = vec2(fract(i), floor(i) * 0.5);
    vec4 emo = texture(uPattern, uv * 0.5 + cell);
    return mix(base, emo.rgb, emo.a * uEfSkin.w);
}

vec4 efSkin(EfLight li, vec3 N, vec3 V, mat3 tbn) {
    vec4 tex = texture(uTex, vUV);
    vec3 base = tex.rgb * uBaseColor.rgb;
    // the occlusion channel; a surface with no base map samples white, and takes its level
    // from the material instead (1 when there is a map)
    float occ = tex.a * uEfMisc.w;
    if (efOn(16384)) {
        // a reconstructed surface's own colour field: the vertex's linear colour
        // where its alpha is 0, the base map where it is 1, the mix between
        // (and its occlusion likewise: the second UV's x is how much the field's own is occluded)
        base = mix(vColor.rgb, base, vColor.a);
        occ = mix(1.0 - vUV1.x, tex.a, vColor.a);
    }
    if (efOn(32768)) base *= vColor.rgb;         // a lent skin matched to this one's along the seams
    // the head's axes in Unity's space
    vec3 hx = efU(-uEfHead[0].xyz), hy = efU(uEfHead[1].xyz), hz = efU(uEfHead[2].xyz);
    vec4 mask = vec4(1.0);                           // _SDFMask; g = "not the SDF face"
    bool face = efOn(32);
    base = efEmotion(base, vUV);
    vec3 shadowBase = efShadowBase(base);
    if (face) mask = texture(uShadeMask, vUV);
    if (!face && efOn(4)) N = efNormalMap(tbn, vUV);

    // grazing-angle tint
    float fwdZ = 1.0;
    float rimK = uEfSkin.x;
    if (face) {
        vec3 f = vec3(dot(li.fwd, hx), dot(li.fwd, hy), dot(li.fwd, hz));
        f = normalize(f);
        fwdZ = f.z / max(length(f.xz), 1e-6);
        float towards = mix(clamp(fwdZ + 0.5, 0.0, 1.0), 1.0, mask.g) * mask.r;
        rimK = mix(uEfSkin.y, uEfSkin.x, mask.b) * towards;
    }
    float graze = clamp((1.0 - (0.85 * clamp(dot(N, V), 0.0, 1.0) + 0.15)) * rimK, 0.0, 1.0);
    base *= mix(vec3(1.0), uEfRimColor.rgb, graze);

    float metallic = uEfPbr.x;
    float kd = 0.96 * (1.0 - metallic);
    vec3 albedo = base * kd;
    float specLevel = (face ? mask.g : 1.0) * uEfPbr.z;
    vec3 f0 = mix(vec3(0.04 * specLevel), base, metallic);
    vec3 shadowCol = shadowBase * kd * uEfScene.y;
    float rough = 1.0 - uEfPbr.y;
    float alpha = max(rough * rough, 0.0078125);

    float ndl = clamp(dot(N, li.L) + uEfNdl.x * uEfNdl.y, -1.0, 1.0);
    if (face) {
        // the light in the head's horizontal plane; the map is authored for
        // light from one side and read mirrored for the other.  (Its blue
        // channel is a face normal the game's rim light uses; no rim here.)
        vec2 lh = normalize(vec2(dot(li.L, hx), dot(li.L, hz)));
        bool flip = lh.x > 0.0;
        vec4 lm = textureLod(uViewRamp, vec2(flip ? vUV.x : 1.0 - vUV.x, vUV.y), 0.0);
        float back = clamp(-dot(normalize(li.L.xz), normalize(li.fwd.xz)), 0.0, 1.0)
                   * clamp(-lh.y, 0.0, 1.0);
        float lz = back * (0.5 - 0.5 * lh.y * lh.y) + lh.y;
        float t = clamp(0.5 - 0.5 * lz, 0.001, 0.999);
        float lo = max(2.0 * t - 1.0, 0.0);
        float hi = min(2.0 * t, 1.0);
        float x = clamp((0.5 * (lm.r + lm.g) - lo) / (hi - lo), 0.0, 1.0);
        float sdfNdl = 2.0 * (efSmooth01(x) + max(0.5 * lz, 0.0)) - 1.0;
        ndl = mix(sdfNdl, ndl, mask.g);
    }
    vec4 ramp = efRamp(uRamp, 0.5 + 0.5 * ndl, 0.5);
    float lit, litF;
    // skin has no view ramp: the occlusion channel plays that part
    vec3 diffuse = efDiffuse(albedo, shadowCol, ramp, 1.0, occ, li.S, lit, litF);
    vec3 lighting = efLighting(li, lit, efAmbGrad(efFlat(N)));

    float ndv = clamp(dot(N, V), 0.0, 1.0);
    float spec = efGgx(dot(N, efHalf(li, V)), ndv, alpha);
    vec3 specLight = efSpecLight(lighting, litF);
    vec3 color = lighting * diffuse + specLight * spec * f0 * uEfScene.w;
    if (efOn(128)) {
        vec2 huv = vUV + vec2(dot(V, hx), dot(V, hy)) * uEfVec.xy;
        color += specLight * texture(uGloss, huv).rgb;
    }
    return vec4(efFinish(color), 1.0);
}

// ---- HGRP/CharacterNPR_Eye --------------------------------------------------
vec4 efEye(EfLight li, vec3 N, vec3 V, mat3 tbnU) {
    vec3 T = tbnU[0], B = tbnU[1];
    // _EYE_HIGHLIGHT is the iris: parallax, a domed normal and the tints.  The
    // other eye sheets (white, brow) are flat and plain.  In the converted
    // data it always comes with _MATCAP_ON.
    bool iris = efOn(512);
    vec2 p = fract(vUV) - 0.5;
    float d2 = dot(p, p);
    bool inside = d2 < 0.25;                          // the iris disc
    vec2 uv = vUV;
    vec3 Ne = N;
    vec3 dome = vec3(0.0, 0.0, 1.0);
    if (iris) {
        // parallax, fading towards the rim of the disc
        vec3 vt = normalize(vec3(dot(T, V), dot(B, V), dot(N, V)));
        float fade = efSmooth01(clamp((d2 - 0.25) * -5.0, 0.0, 1.0));
        uv = vUV - vt.xy * uEfVec.x * vec2(1.0, 0.25) * fade;
        // the cornea's dome, flat outside the disc
        vec2 q = fract(vUV) * 2.0 - 1.0;
        dome = vec3(q * -uEfVec.y, sqrt(max(1.0 - min(dot(q, q), 1.0), 0.0)));
        vec3 nt = inside ? dome * vec3(-0.125, -0.125, 1.0) : vec3(0.0, 0.0, 1.0);
        Ne = normalize(T * nt.x + B * nt.y + N * nt.z);
    }
    vec4 base = texture(uTex, uv) * uBaseColor;
    if (base.a < uAlphaTest) discard;
    vec3 shadowBase = efShadowBase(base.rgb);

    float kd = 0.96 * (1.0 - uEfPbr.x);
    vec3 albedo = base.rgb * kd;
    vec3 shadowCol = shadowBase * kd * uEfScene.y;
    // the light with its component along the head's up axis taken out
    vec3 hx = efU(-uEfHead[0].xyz), hy = efU(uEfHead[1].xyz), hz = efU(uEfHead[2].xyz);
    vec3 lo = normalize(vec3(dot(li.L, hx), dot(li.L, hy), dot(li.L, hz)));
    vec3 Lh = normalize(hx * lo.x + hz * lo.z);
    float ndl = clamp(dot(Ne, Lh), -1.0, 1.0);
    vec4 ramp = efRamp(uRamp, 0.5 + 0.5 * ndl, 0.5);
    float viewA = efRamp(uRamp, 0.5 + 0.5 * dot(Ne, li.fwd), 0.5).a;
    // highlight outside the disc, scattering by the map's alpha
    vec3 tint = iris ? (inside ? vec3(1.0) : uEfColorA.rgb)
                       * mix(vec3(1.0), uEfColorB.rgb, base.a)
                     : vec3(1.0);
    float lit = min(ramp.a, 1.0);
    float chroma = efMax3(ramp.rgb) - efMin3(ramp.rgb);
    vec3 deep = efSat(shadowCol * 0.65, 1.2);
    vec3 c1 = mix(deep, shadowCol, clamp(ramp.a + viewA, 0.0, 1.0));
    c1 = mix(c1, albedo * tint, lit);
    vec3 c2 = c1 * mix(vec3(1.0), ramp.rgb, chroma);
    float k = clamp(efLuma(c1) / max(efLuma(c2), 0.001), 0.0, 1.5);
    vec3 castCol = mix(shadowCol, iris ? albedo * tint : efSat(albedo, 1.2), viewA);
    vec3 diffuse = mix(castCol, c2 * k, li.S);
    float litF = mix(viewA, lit, li.S);
    vec3 lighting = efLighting(li, lit, efAmbGrad(efFlat(N)));
    float premul = mix(1.0, base.a, uEfShadow.w);
    vec3 color = lighting * diffuse * premul;
    if (efOn(256)) {
        // matcap by the dome's normal in view space
        vec3 nd = normalize(T * dome.x + B * dome.y + N * dome.z);
        vec3 up = cross(li.fwd, li.right);
        vec4 mc = texture(uGloss, vec2(dot(nd, li.right), dot(nd, up)) * 0.5 + 0.5);
        vec3 matcap = mc.rgb * uEfColorC.a + mc.a * uEfColorC.rgb;
        color += matcap * efSpecLight(lighting, litF);
    }
    return vec4(efFinish(color), base.a);
}

// ---- HGRP/CharacterNPR_Hair -------------------------------------------------
vec4 efHair(EfLight li, vec3 N, vec3 V, mat3 tbnU) {
    vec3 T = tbnU[0], B = tbnU[1], Ng = N;
    // _DRAW_UNDER_BROW: the opaque fringe is cut where its mask says the brow
    // and eye show; the blended copy of the same mesh is what covers them
    if (efOn(8192) && texture(uPattern, vUV).r < uEfMisc.z) discard;
    vec4 base = texture(uTex, vUV) * uBaseColor;
    if (base.a < uAlphaTest) discard;
    // r specular shift, g specular level, b occlusion, a second-lobe level
    vec4 mgm = efOn(8) ? texture(uShadeMask, vUV) : vec4(0.0, 1.0, 1.0, 1.0);
    vec3 Nspec = Ng;
    if (efOn(4096)) {
        // rg = the normal the diffuse uses, ba = the one the highlight uses
        vec4 sn = texture(uNormalTex, vUV) * 2.0 - 1.0;
        float z = sqrt(max(1.0 - min(dot(sn.xy, sn.xy), 1.0), 0.0));
        N = normalize(T * sn.x * uEfShadow.z + B * sn.y * uEfShadow.z + Ng * z);
        float zs = sqrt(max(1.0 - min(dot(sn.zw, sn.zw), 1.0), 0.0));
        Nspec = normalize(T * sn.z * uEfHairB.z + B * sn.w * uEfHairB.z + Ng * zs);
    } else if (efOn(4)) {
        // no split map: one normal, from _BumpMap, for both
        N = efNormalMap(tbnU, vUV);
        Nspec = N;
    }
    vec3 albedo = base.rgb * 0.96;
    vec3 shadowCol = efShadowBase(base.rgb) * 0.96 * uEfScene.y;

    float ndl = clamp(dot(N, li.L) + uEfNdl.x * uEfNdl.y, -1.0, 1.0);
    float back = clamp(-dot(normalize(li.L.xz), normalize(li.fwd.xz)), 0.0, 1.0);
    float level = efSmooth01(clamp(2.0 * (0.75 - abs(li.fwd.y)), 0.0, 1.0)) * (1.0 - uEfNdl.y);
    ndl = clamp(ndl + back * level * (0.5 - 0.5 * ndl * ndl), -1.0, 1.0);
    vec4 ramp = efRamp(uRamp, 0.5 + 0.5 * ndl, 0.5);
    float viewA = efRamp(uRamp, 0.5 + 0.5 * dot(N, li.fwd), 0.5).a;
    float lit, litF;
    vec3 diffuse = efDiffuse(albedo, shadowCol, ramp, viewA, mgm.b, li.S, lit, litF);
    vec3 lighting = efLighting(li, lit, efAmbGrad(N));
    vec3 specLight = efSpecLight(lighting, litF);

    // the strand direction: the head's up axis leaned by _AnisotropyDirX,
    // made perpendicular to the highlight normal
    vec3 hx = efU(-uEfHead[0].xyz), hy = efU(uEfHead[1].xyz), hz = efU(uEfHead[2].xyz);
    vec3 strand = normalize(hx * uEfHairB.x + hy);
    // across the strands; the map's red channel hands it over to the mesh's own
    // tangent (and its handedness)
    float tw = sign(dot(cross(Ng, T), B));
    vec3 across = mix(cross(Nspec, strand), T, mgm.r);
    vec3 tangent = cross(Nspec, across) * mix(1.0, tw, mgm.r);
    // the half vector leans on the view direction as the head sees it
    vec3 hv = hx * dot(V, hx) + hy * mix(0.5, li.L.y, li.S) + hz * dot(V, hz);
    vec3 H = normalize(normalize(hv * 2.0 + li.L * li.S) + V);
    // how squarely the highlight normal faces the camera, in the head's plane
    vec2 nP = normalize(vec2(dot(Nspec, hx), dot(Nspec, hz)));
    vec2 vP = normalize(vec2(dot(V, hx), dot(V, hz)));
    float facing = pow(clamp(dot(nP, vP), 0.0, 1.0), uEfHairA.w);

    vec3 t1 = normalize(Nspec * (uEfHairA.x * 2.0 - 1.0) + tangent);
    float th = dot(t1, H);
    float lobe = clamp(mgm.g * pow(max(sqrt(max(1.0 - th * th, 0.0)), 1e-4), 200.0), 0.0, 1.0);
    vec3 spec1 = vec3(lobe);
    if (efOn(16)) {
        float gate = facing * facing * (th > 0.0 ? 1.0 : 0.0);
        spec1 = efRamp(uViewRamp, lobe, gate).rgb * lobe;
    }
    spec1 *= facing;
    float peak = efMax3(spec1);
    vec3 t2 = normalize(Nspec * (uEfHairA.y * 2.0 - 1.0) + tangent);
    float th2 = dot(t2, H);
    float power2 = floor(max(1.0 - uEfHairB.y, 0.0) * 200.0);
    float lobe2 = pow(max(sqrt(max(1.0 - th2 * th2, 0.0)), 1e-4), power2) * facing;
    vec3 spec2 = mgm.a * uEfColorA.rgb * lobe2 * (1.0 - peak);
    vec3 spec = specLight * (spec1 * (0.04 * mgm.g) * uEfHairA.z * 5.0 + spec2)
              * uEfScene.w;

    vec3 lightDiffuse = lighting * diffuse;
    if (efOn(1024)) {
        // strand lines: a map, or procedural stripes, darken along the strands
        float stripe = ceil(max(fract(vUV.x * uEfHairC.x) - 0.5, 0.0));
        float lineTex = 1.0 - texture(uGloss, vUV * uEfHairD.xy + uEfHairD.zw).r;
        float lineK = mix(stripe, lineTex, uEfHairB.w);
        lineK = mix(1.0 - uEfHairC.w, 1.0, lineK);
        lineK = mix(lineK, 1.0, peak);
        vec3 t3 = normalize(Nspec * (uEfHairC.y * 2.0 - 1.0) + tangent);
        float th3 = dot(t3, H);
        float range = floor(max(1.0 - uEfHairC.z, 0.0) * 200.0);
        float band = pow(max(sqrt(max(1.0 - th3 * th3, 0.0)), 1e-4), range);
        float k = mgm.g * (lineK - 1.0) * band + 1.0;
        vec3 lined = lightDiffuse * k;
        lightDiffuse = mix(vec3(efLuma(lined)), lined, mix(uEfMisc.x, 1.0, k));
    }
    float premul = mix(1.0, base.a, uEfShadow.w);
    return vec4(efFinish(lightDiffuse * premul + spec), base.a);
}

// ---- HGRP/CharacterNPR_OverlayShadow: multiplied onto what is behind it -----
vec4 efOverlay() {
    vec4 t = texture(uTex, vUV);
    t = mix(t, vec4(1.0, 1.0, 1.0, t.r), uEfMisc.y);
    return t * uBaseColor;
}

float efCastShadow(vec3 Nw) {
    if (uShadowOn <= 0.0 || uEfShadowOff > 0.5) return 1.0;
    float sl = clamp(1.0 - abs(dot(Nw, uCharaLightDir)), 0.0, 1.0);
    vec3 wp = vWorld + Nw * uShadowTexel * (1.0 + 2.0 * sl);
    vec4 lp = uLightMatrix * vec4(wp, 1.0);
    vec3 luv = lp.xyz / max(lp.w, 1e-6) * 0.5 + 0.5;
    if (luv.z > 1.0 || luv.x <= 0.0 || luv.x >= 1.0 || luv.y <= 0.0 || luv.y >= 1.0)
        return 1.0;
    float bias = uShadowBias * (0.5 + 1.5 * sl);
    float occ = 0.0;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i)
            occ += (luv.z - bias > texture(uShadowMap, luv.xy + vec2(i, j) * uShadowOn).r)
                 ? 1.0 : 0.0;
    return 1.0 - occ / 9.0;
}

// Blender's Principled BSDF as a converted source declares it (family 25): the base map's colour and
// its alpha, Metallic, Roughness and Specular IOR Level (uEfPbr.xyz), dithered ("hashed") alpha when
// uEfPbr.w is 1, the map's alpha kept for blending when it is 2.  Lambert diffuse and a GGX lobe (Smith-Schlick visibility, Schlick Fresnel with
// F0 = 0.08 x the specular level, Blender's) under the character lamp, over the scene's ambient as a
// uniform environment.  In this renderer's space: no Unity-space terms are read.
// glTF's maps as the source wires them: _NORMALMAP (4) is Blender's tangent-space Normal Map node
// (rgb * 2 - 1 on T, B, N, mixed from N by its Strength, uEfShadow.z); _METALLICSPECGLOSSMAP (8) is
// one linear image whose green is the Roughness and blue the Metallic.
vec4 efPrincipled(EfLight li, vec3 Nw, mat3 tbn) {
    if (efOn(4)) {
        // Blender flips the tangent on a back face but rebuilds the bitangent from the flipped
        // normal, so the bitangent keeps its side; the frame here flips all three
        vec3 B = gl_FrontFacing ? tbn[1] : -tbn[1];
        vec3 t = texture(uNormalTex, vUV).rgb * 2.0 - 1.0;
        vec3 mapped = normalize(t.x * tbn[0] + t.y * B + t.z * Nw);
        Nw = normalize(mix(Nw, mapped, max(uEfShadow.z, 0.0)));
    }
    vec4 rm = efOn(8) ? texture(uShadeMask, vUV) : vec4(0.0, uEfPbr.y, uEfPbr.x, 1.0);
    vec4 base = texture(uTex, vUV) * uBaseColor;
    base.rgb = efEmotion(base.rgb, vUV1);       // original facial UV, separate from the baked skin atlas
    if (efOn(32768)) base.rgb *= vColor.rgb;     // a lent skin matched to this one's along the seams
    if (uEfPbr.w > 0.5 && uEfPbr.w < 1.5) {
        float noise = fract(sin(dot(floor(gl_FragCoord.xy), vec2(12.9898, 78.233))) * 43758.5453);
        if (base.a < max(noise, 1e-4)) discard;
    }
    vec3 L = normalize(-uEfLightDir);
    vec3 V = normalize(uCameraPos - vWorld);
    vec3 H = normalize(L + V);
    float ndl = max(dot(Nw, L), 0.0);
    float ndv = max(dot(Nw, V), 1e-4);
    float ndh = max(dot(Nw, H), 0.0);
    float metallic = clamp(rm.b, 0.0, 1.0);
    float a = max(rm.g * rm.g, 1e-3);
    float a2 = a * a;
    float den = ndh * ndh * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159265 * den * den);
    float k = a * 0.5;
    float G = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k));
    vec3 f0 = mix(vec3(0.08 * uEfPbr.z), base.rgb, metallic);
    vec3 F = f0 + (1.0 - f0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
    vec3 spec = D * G * F / max(4.0 * ndl * ndv, 1e-4);
    vec3 light = li.Lc * ndl * li.S;
    vec3 lit = base.rgb * (1.0 - metallic) * (uAmbient + light) + spec * light;
    // uEfPbr.w 2: blended by its coverage (``shading.coverage_blended``), what dithering converges to
    return vec4(lit, uEfPbr.w > 1.5 ? base.a : 1.0);
}

""" + UNGRADE + """

void efFragment() {
    if (uToonFamily == 24) { fragColor = efOverlay(); fragColor.rgb = efUngrade(fragColor.rgb); return; }
    vec3 Nw = normalize(vNormal);
    if (!gl_FrontFacing) Nw = -Nw;
    // the frame is built in this renderer's space, where the derivatives are
    // right-handed, and then carried over axis by axis
    mat3 tbn = cotangentFrame(Nw, vWorld, vUV);
    mat3 tbnU = mat3(efU(tbn[0]), efU(tbn[1]), efU(tbn[2]));
    EfLight li;
    li.L = efU(normalize(-uEfLightDir));
    li.Lc = uEfLightColor;
    li.ambTint = clamp(uAmbient / max(efMax3(uAmbient), 1e-4), 0.0, 1.0);
    // a volume's fixed ambient colour replaces the scene's: the skin program has its own
    if (uEfTint.w > 0.5) li.ambTint = uToonFamily == 21 ? uEfTintSkin : uEfTint.rgb;
    li.fwd = efU(normalize(uCamFwd));
    li.right = efU(normalize(uCamRight));
    li.S = efCastShadow(Nw);
    vec3 N = efU(Nw);
    vec3 V = efU(normalize(uCameraPos - vWorld));
    vec4 c;
    if (uToonFamily == 25) c = efPrincipled(li, Nw, tbn);
    else if (uToonFamily == 20) c = efMain(li, N, V, tbnU);
    else if (uToonFamily == 21) c = efSkin(li, N, V, tbnU);
    else if (uToonFamily == 22) c = efHair(li, N, V, tbnU);
    else c = efEye(li, N, V, tbnU);
    if (uFogEnable == 1) {
        float f = clamp((vDepth - uFogRange.x) / max(uFogRange.y - uFogRange.x, 0.001), 0.0, 1.0);
        c.rgb = mix(c.rgb, uFogColor, f * 0.85);
    }
    c.rgb = efUngrade(c.rgb);
    fragColor = c;
}
"""
