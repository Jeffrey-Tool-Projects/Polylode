"""GLSL sources for the cel-shading pipeline.

**Everything that draws into the scene target writes linear.**  Koikatsu is a
Linear project, so its character passes run on decoded texels and its camera
render target is sRGB — the encode is the hardware's, at the framebuffer write
(``gl.framebuffer.Framebuffer(srgb=True)`` + ``GL_FRAMEBUFFER_SRGB``).  That is
the boundary this file is written against: no fragment program here encodes its
own output, because doing it in the shader would also blend in the wrong space.

The one program that *does* convert is the post pass, and both conversions in it
are transcribed from the game's own ``Hidden/Amplify Color/Base``
(``Koikatu_Data/resources.assets``) rather than from the sRGB spec — see
:data:`POST_FS`.
"""

from . import shaders_endfield as _EF

# Amplify Color's own pair, straight off the pass the game runs.  They are
# **not** the piecewise sRGB curve the hardware uses; they are Unity's
# shader-side approximations, and Illusion's LUT is fitted to *these*::
#
#     log r0.xyz, r0.xyzx                     |  mad r1, c, 0.305306, 0.682171
#     mul r0.xyz, r0.xyzx, l(0.416667)        |  mad r1, c, r1, 0.012523
#     exp r0.xyz, r0.xyzx                     |  mul o0, c, r1
#     mad r0.xyz, r0.xyzx, l(1.055), l(-0.055)
#     max r0.xyz, r0.xyzx, l(0)
#
# ``encodeSrgb`` is separate and *is* the spec curve: it stands in for the final
# blit to the back buffer, which our post target (a plain RGBA8) cannot do in
# hardware.
_COLORSPACE_BLOCK = """
vec3 amplifyToGamma(vec3 c) {
    return max(1.055 * pow(max(c, vec3(0.0)), vec3(0.41666667)) - 0.055,
               vec3(0.0));
}
vec3 amplifyToLinear(vec3 c) {
    return c * (c * (c * 0.305306 + 0.682171) + 0.012523);
}
vec3 encodeSrgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92,
               1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
               step(vec3(0.0031308), c));
}
"""

# Soft cap only for debug tooling — skinning uses a bone-matrix *texture*,
# so the real armature can be the full Origin set (~400+) with no uniform limit.
MAX_BONES = 4096

_SKIN_BLOCK = """
// Bone palette as RGBA32F texture: width=4 (mat4 columns), height=nBones.
// Avoids GL uniform component caps so Origin full armatures load without collapse.
// An item with a dual-quaternion part has ten texels per bone
// (mesh/skinning.py: palette_texels): 4 the quaternion (w, x, y, z), 5 its dual
// part, 6..8 the scale matrix's rows, 9.x whether the bone is scaled.
uniform sampler2D uBoneTex;
uniform int       uSkinned;
uniform int       uBoneCount;
// 0: the weighted sum of the bones' matrices; 1: of their rigid motions;
// 2: each vertex aSkinDual of the second and the rest of the first.
uniform int       uSkinDual;

// The inverse transpose times the determinant: what carries a normal.  Its
// columns are the cross products of the matrix's, so a mirroring matrix turns
// the normal round with the triangle and a flattened one still gives a direction.
mat3 cofactor(mat3 m) {
    return mat3(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1]));
}

vec4 boneTexel(int idx, int column) {
    return texelFetch(uBoneTex, ivec2(column, clamp(idx, 0, max(uBoneCount - 1, 0))), 0);
}

// One influence of the dual-quaternion blend (mesh/skinning.py: _blended).  A
// scaled bone's scale becomes the translation it gives this vertex; the sum is
// taken towards the hemisphere of what has been summed so far.
// scaleT is the transpose of the summed scale (the texels are its rows), with
// scaleWeight the weight that went into it; unscaled bones are added as identity
// at the end.
void dualAdd(float bone, float weight, vec3 p, inout vec4 real, inout vec4 dual,
             inout mat3 scaleT, inout float scaleWeight) {
    if (weight == 0.0) return;
    int idx = int(bone);
    vec4 q = boneTexel(idx, 4);
    vec4 t = boneTexel(idx, 5);
    if (boneTexel(idx, 9).x > 0.5) {
        vec4 hp = vec4(p, 1.0);
        vec4 r0 = boneTexel(idx, 6), r1 = boneTexel(idx, 7), r2 = boneTexel(idx, 8);
        vec3 d = vec3(dot(r0, hp), dot(r1, hp), dot(r2, hp)) - p;
        scaleT += weight * mat3(r0.xyz, r1.xyz, r2.xyz);
        scaleWeight += weight;
        t.x -= 0.5 * (q.y * d.x + q.z * d.y + q.w * d.z);
        t.y += 0.5 * (q.x * d.x + q.z * d.z - q.w * d.y);
        t.z += 0.5 * (q.x * d.y + q.w * d.x - q.y * d.z);
        t.w += 0.5 * (q.x * d.z + q.y * d.y - q.z * d.x);
    }
    if (dot(q, real) < 0.0) weight = -weight;
    real += weight * q;
    dual += weight * t;
}

// The blend's own division by the total weight cancels against the squared
// length the result is divided by, so neither is done before the end.  The
// surface follows the vertex's deform matrix, rotation · scale.  False when the
// vertex has no weight at all.
bool dualDeform(in vec3 p, out vec3 op, out mat3 deform) {
    vec4 real = vec4(0.0), dual = vec4(0.0);
    mat3 scaleT = mat3(0.0);
    float scaleWeight = 0.0;
    dualAdd(aBoneIdx.x, aBoneWt.x, p, real, dual, scaleT, scaleWeight);
    dualAdd(aBoneIdx.y, aBoneWt.y, p, real, dual, scaleT, scaleWeight);
    dualAdd(aBoneIdx.z, aBoneWt.z, p, real, dual, scaleT, scaleWeight);
    dualAdd(aBoneIdx.w, aBoneWt.w, p, real, dual, scaleT, scaleWeight);
    dualAdd(aBoneIdx1.x, aBoneWt1.x, p, real, dual, scaleT, scaleWeight);
    dualAdd(aBoneIdx1.y, aBoneWt1.y, p, real, dual, scaleT, scaleWeight);
    dualAdd(aBoneIdx1.z, aBoneWt1.z, p, real, dual, scaleT, scaleWeight);
    dualAdd(aBoneIdx1.w, aBoneWt1.w, p, real, dual, scaleT, scaleWeight);
    float len2 = dot(real, real);
    if (len2 <= 0.0) return false;
    float w = real.x, x = real.y, y = real.z, z = real.w;
    mat3 m = mat3(w * w + x * x - y * y - z * z, 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
                  2.0 * (x * y - w * z), w * w + y * y - x * x - z * z, 2.0 * (y * z + w * x),
                  2.0 * (x * z + w * y), 2.0 * (y * z - w * x), w * w + z * z - x * x - y * y);
    vec3 shift = 2.0 * vec3(-dual.x * x + w * dual.y - dual.z * z + y * dual.w,
                            -dual.x * y + dual.y * z - x * dual.w + w * dual.z,
                            -dual.x * z + x * dual.z + w * dual.w - dual.y * y);
    op = (m * p + shift) / len2;
    deform = m / len2;
    if (scaleWeight > 0.0) {
        float total = dot(aBoneWt, vec4(1.0)) + dot(aBoneWt1, vec4(1.0));
        deform = deform * transpose((scaleT + (total - scaleWeight) * mat3(1.0)) / total);
    }
    return true;
}

mat4 boneMatrix(int idx) {
    int i = clamp(idx, 0, max(uBoneCount - 1, 0));
    return mat4(
        texelFetch(uBoneTex, ivec2(0, i), 0),
        texelFetch(uBoneTex, ivec2(1, i), 0),
        texelFetch(uBoneTex, ivec2(2, i), 0),
        texelFetch(uBoneTex, ivec2(3, i), 0)
    );
}

// A tangent goes through the vertex's deform matrix and a normal through its
// cofactor, which keeps it square to the surface when bones are scaled unevenly
// and on the side the triangle faces under a mirroring one.  Under a rotation
// or an even scale it is the same direction.
void skinFrame(in vec3 p, in vec3 n, in vec3 t, out vec3 op, out vec3 on, out vec3 ot,
               out vec3 sourceNormal) {
    if (uSkinned == 0) { op = p; on = n; ot = t; sourceNormal = n; return; }
    vec3 dualPoint = p;
    mat3 dualMatrix = mat3(1.0);
    bool turned = uSkinDual != 0 && dualDeform(p, dualPoint, dualMatrix);
    if (uSkinDual == 1) {
        if (!turned) { op = p; on = n; ot = t; sourceNormal = n; return; }
        op = dualPoint;
        on = cofactor(dualMatrix) * n;
        sourceNormal = on;
        ot = dualMatrix * t;
        return;
    }
    mat4 m = boneMatrix(int(aBoneIdx.x)) * aBoneWt.x
           + boneMatrix(int(aBoneIdx.y)) * aBoneWt.y
           + boneMatrix(int(aBoneIdx.z)) * aBoneWt.z
           + boneMatrix(int(aBoneIdx.w)) * aBoneWt.w;
    if (dot(aBoneWt1, vec4(1.0)) > 0.0) {
        m += boneMatrix(int(aBoneIdx1.x)) * aBoneWt1.x
           + boneMatrix(int(aBoneIdx1.y)) * aBoneWt1.y
           + boneMatrix(int(aBoneIdx1.z)) * aBoneWt1.z
           + boneMatrix(int(aBoneIdx1.w)) * aBoneWt1.w;
    }
    op = (m * vec4(p, 1.0)).xyz;
    mat3 l = mat3(m);
    if (turned) {
        op += aSkinDual * (dualPoint - op);
        l += aSkinDual * (dualMatrix - l);
    }
    on = cofactor(l) * n;
    ot = l * t;
    // Authored linear-skin channels retain their vector transport contract.
    sourceNormal = uSkinDual == 0 ? l * n : on;
}

void skin(in vec3 p, in vec3 n, out vec3 op, out vec3 on) {
    vec3 unusedTangent;
    vec3 unusedSourceNormal;
    skinFrame(p, n, vec3(0.0), op, on, unusedTangent, unusedSourceNormal);
}
"""

_ATTRIBS = """
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aColor;
layout(location = 4) in vec4 aBoneIdx;
layout(location = 5) in vec4 aBoneWt;
layout(location = 6) in vec2 aUV1;
layout(location = 7) in vec4 aBoneIdx1;
layout(location = 8) in vec4 aBoneWt1;
layout(location = 9) in vec4 aSourceTangent;
layout(location = 10) in float aSourceTangentPresent;
layout(location = 11) in float aSkinDual;
layout(location = 12) in vec4 aOutlineNormal;
"""

TOON_VS = "#version 330 core\n" + _ATTRIBS + _SKIN_BLOCK + """
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorld;
out vec3 vNormal;
out vec2 vUV;
out vec2 vUV1;
out vec4 vColor;
out float vDepth;
out vec4 vSourceTangent;
out vec3 vSourceNormal;
out vec3 vSourceBinormal;
flat out float vSourceTangentPresent;

void main() {
    vec3 sp, sn, st, sourceNormal;
    skinFrame(aPos, aNormal, aSourceTangent.xyz, sp, sn, st, sourceNormal);
    vec4 world = uModel * vec4(sp, 1.0);
    vWorld  = world.xyz;
    vNormal = normalize(mat3(uModel) * sn);
    vSourceTangentPresent = aSourceTangentPresent;
    vSourceTangent = vec4(0.0);
    vSourceNormal = vec3(0.0);
    vSourceBinormal = vec3(0.0);
    if (aSourceTangentPresent > 0.5) {
        // Source object handedness is independent of the weighted bone blend's determinant.
        mat3 linear = mat3(uModel);
        vec3 normal = transpose(inverse(linear)) * sourceNormal;
        normal = length(normal) > 0.0 ? normalize(normal) : vec3(0.0);
        vec3 tangent = linear * st;
        tangent = length(tangent) > 0.0 ? normalize(tangent) : vec3(0.0);
        float handedness = aSourceTangent.w * sign(determinant(linear));
        vSourceTangent = vec4(tangent, handedness);
        vSourceNormal = normal;
        vSourceBinormal = cross(normal, tangent) * handedness;
    }
    vUV     = aUV;
    vUV1    = aUV1;
    vColor  = aColor;
    vec4 viewPos = uView * world;
    vDepth = -viewPos.z;
    gl_Position = uProj * viewPos;
}
"""

TOON_FS = """#version 330 core
in vec3 vWorld;
in vec3 vNormal;
in vec2 vUV;
in vec2 vUV1;
in vec4 vColor;
in float vDepth;
in vec4 vSourceTangent;
in vec3 vSourceNormal;
in vec3 vSourceBinormal;
flat in float vSourceTangentPresent;
out vec4 fragColor;

uniform vec4  uBaseColor;
uniform vec4  uShadowColor;
uniform vec4  uSpecColor;
uniform float uSpecPower;
uniform float uSpecStrength;
uniform vec4  uRimColor;
uniform float uRimStrength;
uniform float uRimPower;
// Illusion's specular block, which shares no property with the Blinn pair
// above: _SpecularPower is a 0..1 strength that also sets the exponent,
// _SpecularPowerNail floors the specular map, _notusetexspecular switches the
// texture lobe off and _rimV scales the rim.
uniform float uOriginSpecular;
uniform float uSpecNail;
uniform float uNoTexSpec;
uniform float uRimV;
uniform float uShadeSteps;
uniform float uShadeFeather;
uniform float uShadeOffset;
uniform float uUnlit;
uniform float uAlphaTest;
// ``main_item_ditherd``'s ``_Cutoff`` (negative: not that shader) and the
// target size its dither reads the pixel row from (§7.1199)
uniform float uDitherCut;
uniform vec2  uDitherViewport;
// Illusion's eye look-at is a texture offset, not a bone: EyeLookMaterialControll
// writes its ``offset`` into the iris material's _MainTex/_overtex1/_overtex2
// every frame, mirrored in X for the right eye.  The iris map takes it as a
// shift of its UV0 lookup; the two highlights are their own map on UV1, read at
// ``uv1 * xy + zw`` so they can also shake on their own (§7.403).
uniform vec2  uUVOffset;
uniform sampler2D uHighlight;
uniform int   uUseHighlight;
// toon_glasses_lod0 (family 8, §7.413)
uniform vec4  uGlassColor;      // _Color (linear)
uniform vec4  uGlassColor4;     // _Color4 (linear rgb, alpha as authored)
uniform vec3  uGlassParams;     // _ColorInverse, _ColorSort, _RimPower
uniform sampler2D uGlassRamp;   // _GlassRamp; unbound = black (declared default)
uniform int   uUseGlassRamp;
uniform vec4  uGlassRampXf;
uniform vec4  uAmbientShadow;   // _ambientshadowG as the shader reads it
uniform vec4  uHighlightXf;
uniform float uHighlightOn;     // _isHighLight

uniform vec3  uLightDir;       // pointing *from* the light toward the scene
// Origin's characters are lit by their own light, aimed from the camera, and the
// families below use it instead of the scene's sun.  Not a stylistic choice:
// _RampG is a hard step at N.L = 0.5, so an off-axis light drops that step
// across the face.  See Renderer.chara_light_dir.
uniform vec3  uCharaLightDir;
uniform vec3  uLightColor;
uniform float uLightIntensity;
uniform vec3  uCharaLightColor;   // _LightColor0: the chara light, colour*intensity
uniform vec3  uAmbient;
uniform vec3  uRimLightDir;
uniform vec3  uCameraPos;

uniform sampler2D uTex;
uniform int   uUseTex;
uniform sampler2D uPattern;
uniform int   uUsePattern;
// Illusion's own toon inputs, read off the material rather than assumed:
//   _AnotherRamp  a 128x8 shading ramp -- the band shape is a lookup, not a step
//   _NormalMap    surface detail; tangents come from screen-space derivatives
//                 so no extra vertex attribute is needed
//   _SpeclarHeight where the highlight starts (0.95-0.98 = a very tight one)
uniform sampler2D uRamp;
uniform int   uUseRamp;
uniform sampler2D uNormalTex;
uniform int   uUseNormal;
uniform float uNormalScale;
uniform int   uNormalEncoding;
uniform float uSpecThreshold;
uniform vec4  uPatternColor;
uniform vec4  uPatternXf;      // scale.xy, offset.xy

// ---- Illusion's toon model, transcribed from the forward passes ---------
// 0 = this project's own procedural materials (the banded model below);
// 1 = main_skin, 2 = main_opaque / main_alpha / main_item, 3 = main_hair.
uniform int   uToonFamily;
uniform vec3  uAmbientBase;    // _ambientshadowG resolved through its overlay
uniform vec3  uAmbientFloor;   // _ambientshadowG.rgb, the light's lower bound
// R = multiplier on the diffuse term, G = ink, B = outline width scale,
// A = the _AnotherRamp blend weight, which doubles as the scale on the shadow
//     tint (one register in the bytecode serves both).
uniform sampler2D uShadeMask;
uniform int   uUseShadeMask;
// _DetailMask as itself, resampled through its own _ST.  main_opaque reads
// four raw channels the packing above has no room for, and reads .r a second
// time at a parallax-offset uv, which no packing can carry at all.
uniform sampler2D uDetailMask;
uniform int   uUseDetail;
uniform sampler2D uViewRamp;   // _AnotherRamp
uniform int   uUseViewRamp;
uniform vec4  uViewRampXf;
uniform vec4  uNormalXf;
uniform sampler2D uGloss;      // _HairGloss
uniform int   uUseGloss;
uniform vec4  uGlossXf;
uniform float uGlossHeight;    // _SpeclarHeight
uniform float uShadeFloor;     // 1 - _ShadowExtend (main_hair's second term)
uniform float uLitOverride;    // <0 off; otherwise pins the ramp term (§7.341)
uniform sampler2D uShadowMap;  // the character light's depth (§7.343)
uniform mat4  uLightMatrix;    // world -> light clip
uniform float uShadowOn;       // 0 = off, else one texel in light UV
uniform float uShadowBias;
uniform float uShadowTexel;   // one shadow texel in world units

uniform int   uFogEnable;
uniform vec3  uFogColor;
uniform vec2  uFogRange;
// ``main_skin``'s liquid layer (``origin.figure.siru``, D13): ``_liquidmask`` (which skin is which
// region), ``_Texture2`` (the drops: .r at level 1, .g at level 2), ``_Texture3`` (their normals,
// unpacked like ``_NormalMap``), the levels ``UpdateSiru`` writes, and ``siru.FluidLook``.
uniform sampler2D uLiquidMask;
uniform int   uUseLiquidMask;
uniform sampler2D uLiquidDrop;
uniform int   uUseLiquidDrop;
uniform sampler2D uLiquidNormal;
uniform int   uUseLiquidNormal;
uniform vec4  uLiquidLevels;    // _liquidftop, _liquidfbot, _liquidbtop, _liquidbbot
uniform float uLiquidFace;      // _liquidface
uniform vec4  uLiquidTiling;    // _LiquidTiling as uv * xy + zw (its .zw scale, .xy offset)
uniform vec4  uLiquidMaskXf;
uniform vec4  uLiquidDropXf;
uniform vec4  uLiquidNormalXf;
uniform vec3  uLiquidThin;      // the liquid's colour at no coverage ...
uniform vec3  uLiquidFull;      // ... and at full
uniform float uLiquidOpacity;
uniform float uLiquidRimPower;  // the skin's _rimpower, which only the liquid's rim reads
uniform int   uLiquidSolid;     // 1: the whole surface is liquid (the shot fluid's mesh)
""" + _EF.UNIFORMS + """
float bands(float x) {
    float steps = max(uShadeSteps, 1.0);
    float f = max(uShadeFeather, 0.0008);
    float scaled = clamp(x, 0.0, 1.0) * steps;
    float lo = floor(scaled);
    float frac = scaled - lo;
    float s = smoothstep(0.5 - f * steps * 0.5, 0.5 + f * steps * 0.5, frac);
    return clamp((lo + s) / steps, 0.0, 1.0);
}

// Photoshop "hard light" with *blend* as the upper layer — the blend every
// Origin character shader uses to fold _ambientshadowG together with either the
// material's _ShadowColor or the albedo's own hue.
vec3 hardLight(vec3 base, vec3 blend) {
    vec3 mult   = 2.0 * base * blend;
    vec3 screen = 1.0 - (2.0 - 2.0 * blend) * (1.0 - base);
    return clamp(mix(mult, screen, step(vec3(0.5), blend)), 0.0, 1.0);
}

// The colour's own hue at full value and *k* of its saturation.  main_skin
// and main_opaque build their shadow tint out of this rather than a fixed
// colour, which is why a red dress shadows red and skin shadows warm.
// (hue, saturation) of a colour, the way these shaders compute it.  Factored
// out because main_skin builds **two** different tints from the *same* hue and
// they differ only in the offsets and the shaping — recomputing it separately
// for each is how the two would drift apart.
vec2 originHueSat(vec3 c) {
    float mx = max(c.r, max(c.g, c.b));
    float mn = min(c.r, min(c.g, c.b));
    float d  = mx - mn;
    float h  = 0.0;
    if (d > 1e-6) {
        if (mx == c.r)      h = mod((c.g - c.b) / d, 6.0);
        else if (mx == c.g) h = (c.b - c.r) / d + 2.0;
        else                h = (c.r - c.g) / d + 4.0;
        h /= 6.0;
    }
    return vec2(h, (mx > 1e-6) ? d / mx : 0.0);
}

vec3 hueTint(vec3 c, float k) {
    vec2 hs = originHueSat(c);
    vec3 t = fract(vec3(hs.x) + vec3(0.0, -1.0 / 3.0, 1.0 / 3.0));
    t = clamp(abs(1.0 - 2.0 * t) * 3.0 - 1.0, 0.0, 1.0);
    return 1.0 + hs.y * k * (t - 1.0);
}

// main_skin's **second** hue rebuild — the one its textured specular darkens
// with, and it is not ``hueTint`` with another k::
//
//     add r1.xzw, |hue|, l(-0.080000, 0, -0.413333, 0.253333)
//     …fract / |1-2x|*3-1 / -1…                    -> [-1, 0]
//     mad r1.xyz, r1.xyzx, l(0.4), l(1.0)          -> [0.6, 1]
//     mad r1.xyz, r1.xyzx, l(0.97), l(-1.0)        -> [-0.418, -0.03]
//
// Two differences that matter: the hue is rotated by **-0.08** rather than
// sitting on the (0, -1/3, +1/3) triad, and the result is **not** scaled by the
// colour's saturation the way the shadow tint is (there is no ``r0.w`` multiply
// in this chain).  Using the shadow tint here would put the darkening on the
// wrong channels and make it vanish on grey skin.
vec3 originSpecTint(vec3 c) {
    vec3 t = fract(vec3(originHueSat(c).x)
                   + vec3(-0.08, -0.413333, 0.253333));
    t = clamp(abs(1.0 - 2.0 * t) * 3.0 - 1.0, 0.0, 1.0) - 1.0;
    return (t * 0.4 + 1.0) * 0.97 - 1.0;
}

// Cotangent frame from derivatives (Mikkelsen): lets a normal map work on
// meshes without an authored tangent frame (the legacy material path).
mat3 cotangentFrame(vec3 N, vec3 p, vec2 uv) {
    vec3 dp1 = dFdx(p);
    vec3 dp2 = dFdy(p);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)) + 1e-8);
    return mat3(T * invmax, B * invmax, N);
}
// The tangent-space normal the legacy path bent N with (``(0, 0, 1)`` without a map), for the
// liquid layer to blend its own normal against.
vec3 gBaseNT = vec3(0.0, 0.0, 1.0);

// ``main_skin``'s liquid amount for one level: the level-1 drops up to 1, the level-2 ones above.
float liquidAmount(float lv, vec2 d) {
    return max(clamp(lv, 0.0, 1.0) * d.x, clamp(lv - 1.0, 0.0, 1.0) * d.y);
}

// ``main_skin``'s liquid coverage (sub-program 8.0): each region of ``_liquidmask`` takes its own
// level -- R alone the front top, G alone the front bottom, B alone the back top, R and G together the
// back bottom, G and B together the face -- and the two mixed ones are remapped from a tenth.
float liquidCoverage(vec2 luv) {
    vec2 d = texture(uLiquidDrop, luv * uLiquidDropXf.xy + uLiquidDropXf.zw).rg;
    vec3 m = texture(uLiquidMask, vUV * uLiquidMaskXf.xy + uLiquidMaskXf.zw).rgb;
    float c = min(liquidAmount(uLiquidLevels.x, d), m.r - max(m.g, m.b));
    c = max(c, min(m.g - max(m.b, m.r), liquidAmount(uLiquidLevels.y, d)));
    c = max(c, min(m.b - max(m.g, m.r), liquidAmount(uLiquidLevels.z, d)));
    c = max(c, min(min(m.g, m.r) * 1.111111 - 0.111111, liquidAmount(uLiquidLevels.w, d)));
    c = max(c, min(min(m.b, m.g) * 1.111111 - 0.111111, liquidAmount(uLiquidFace, d)));
    return clamp(c, 0.0, 1.0);
}

// A legacy-path normal-map texel as a tangent-space normal: RGB, or RG with z rebuilt
// (``uNormalEncoding`` 2), the xy scaled by ``uNormalScale``; not normalised.
vec3 tangentNormal(vec3 texel) {
    vec3 nt = texel * 2.0 - 1.0;
    if (uNormalEncoding == 2)
        nt.z = sqrt(max(1.0 - min(dot(nt.xy, nt.xy), 1.0), 0.0));
    nt.xy *= uNormalScale;
    return nt;
}

// GFCharForward/GFCharTransFront: only N changes sign on back faces.
// Source T and B are interpolated independently, without orthogonalization.
vec3 surfaceNormal(out mat3 tbn) {
    if (uNormalEncoding == 1) {
        vec3 N = gl_FrontFacing ? vSourceNormal : -vSourceNormal;
        tbn = mat3(vSourceTangent.xyz, vSourceBinormal, N);
        if (uUseNormal == 0) return normalize(N);
        vec4 texel = texture(uNormalTex, vUV * uNormalXf.xy + uNormalXf.zw);
        vec2 xy = vec2(texel.r * texel.a, 1.0 - texel.g) * 2.0 - 1.0;
        float z = sqrt(1.0 - min(dot(xy, xy), 1.0));
        return normalize(vSourceTangent.xyz * xy.x + vSourceBinormal * xy.y + N * z);
    }
    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;
    tbn = cotangentFrame(N, vWorld, vUV);
    if (uUseNormal == 1) {
        vec3 nt = tangentNormal(texture(uNormalTex, vUV * uNormalXf.xy + uNormalXf.zw).xyz);
        gBaseNT = nt;
        N = normalize(tbn * nt);
    }
    return N;
}
""" + _EF.FUNCTIONS + """
void main() {
    // families 20 and up are another game's character shaders, with their own
    // inputs from the first line on (gl/shaders_endfield.py)
    if (uToonFamily >= 20) { efFragment(); return; }
    vec4 albedo = uBaseColor * vColor;
    vec2 uv = vUV + uUVOffset;
    if (uUseTex == 1) {
        // textures that carry a design ship full colour; masks ship white RGB
        vec4 t = texture(uTex, uv);
        albedo.rgb *= t.rgb;
        albedo.a   *= t.a;
    }
    // ``toon_eye_lod0``: ``w = max(hi2 * _overcolor2.a, hi1 * _overcolor1.a)
    // * _isHighLight``, ``rgb = lerp(rgb, hi.rgb, w)``, ``a = max(a, w)`` —
    // and the same ``w`` is added to ``lit`` below.
    float hlW = 0.0;
    if (uUseHighlight == 1) {
        vec4 hl = texture(uHighlight, vUV1 * uHighlightXf.xy + uHighlightXf.zw);
        hlW = hl.a * uHighlightOn;
        albedo.rgb = mix(albedo.rgb, hl.rgb, hlW);
        albedo.a = max(albedo.a, hlW);
    }
    if (uUsePattern == 1) {
        vec2 puv = vUV * uPatternXf.xy + uPatternXf.zw;
        vec4 pt = texture(uPattern, puv);
        albedo.rgb = mix(albedo.rgb, uPatternColor.rgb, pt.a * uPatternColor.a);
    }
    if (albedo.a < uAlphaTest) discard;
    if (uDitherCut >= 0.0) {
        // ``main_item_ditherd``: the pixel's column and its row counted from
        // the top (the program flips y), each mod 4, pick a 4x4 Bayer
        // threshold T = B[y][x] / 17; ``discard`` when
        // ``round_ne(T + a - _Cutoff/2 - 1.5) < 0``.
        int bx = int(floor(gl_FragCoord.x)) & 3;
        int by = (int(uDitherViewport.y) - 1 - int(floor(gl_FragCoord.y))) & 3;
        const float B[16] = float[16](1.0, 13.0, 4.0, 16.0,  9.0, 5.0, 12.0, 8.0,
                                      3.0, 15.0, 2.0, 14.0, 11.0, 7.0, 10.0, 6.0);
        float t = B[by * 4 + bx] / 17.0;
        if (roundEven(t + albedo.a - uDitherCut * 0.5 - 1.5) < 0.0) discard;
    }

    mat3 tbn;
    vec3 N = surfaceNormal(tbn);
    // ``main_skin``'s liquid layer: its coverage, and its normals bent into the skin's by it -- the
    // reoriented blend of the program's ``r4``/``r7``, ``base + c * (rnm(base, liquid) - base)`` -- so the
    // skin under the liquid is lit through the liquid's normal too.
    // ``uLiquidSolid``: the shot fluid's own surface (``game.intimate.fluid_shot``), all liquid
    float liq = (uToonFamily == 1 && uLiquidSolid == 1) ? 1.0 : 0.0;
    if (uToonFamily == 1 && uLiquidSolid == 0 && uUseLiquidMask == 1 && uUseLiquidDrop == 1
            && max(max(max(uLiquidLevels.x, uLiquidLevels.y), max(uLiquidLevels.z, uLiquidLevels.w)),
                   uLiquidFace) > 0.0) {
        vec2 luv = vUV * uLiquidTiling.xy + uLiquidTiling.zw;
        liq = liquidCoverage(luv);
        if (liq > 0.0 && uUseLiquidNormal == 1 && uNormalEncoding != 1) {
            vec3 ln = texture(uLiquidNormal, luv * uLiquidNormalXf.xy + uLiquidNormalXf.zw).xyz * 2.0 - 1.0;
            ln.z = sqrt(max(1.0 - min(dot(ln.xy, ln.xy), 1.0), 0.0));
            vec3 t = gBaseNT + vec3(0.0, 0.0, 1.0);
            vec3 u = ln * vec3(-1.0, -1.0, 1.0);
            vec3 rnm = t * dot(t, u) / t.z - u;
            N = normalize(tbn * mix(gBaseNT, rnm, liq));
        }
    }
    vec3 L = normalize(-(uToonFamily != 0 ? uCharaLightDir : uLightDir));
    vec3 V = normalize(uCameraPos - vWorld);
    vec3 H = normalize(L + V);

    vec3 shaded;
    if (uToonFamily == 8) {
        // ``toon_glasses_lod0``'s forward fragment, register by register
        // (``bo_head_00`` sub-program 12, §7.366 五 / §7.413).  It reads
        // ``_MainTex`` raw — no vertex colour, no base colour — and its
        // output is premultiplied (``Blend One OneMinusSrcAlpha``); returning
        // ``(c, a)`` under our ``SrcAlpha OneMinusSrcAlpha`` is the same sum.
        vec4 t8 = (uUseTex == 1) ? texture(uTex, uv) : vec4(1.0);
        float ndl8 = dot(N, L);
        vec3 R8 = reflect(-V, N);
        float gu = abs(dot(R8, H)) * (N.x + N.y + N.z) * 0.5 + 0.5;
        vec3 g8 = (uUseGlassRamp == 1)
            ? texture(uGlassRamp, vec2(gu) * uGlassRampXf.xy + uGlassRampXf.zw).rgb
            : vec3(0.0);
        float rim8 = clamp(2.0 * pow(1.0 - max(dot(N, V), 0.0), uGlassParams.z)
                           - 1.0, 0.0, 1.0);
        float spec8 = clamp(abs(dot(N, H)) * 3.999999 - 3.799999, 0.0, 1.0);
        vec3 diff8 = (uGlassColor.rgb - uGlassColor4.rgb) * (1.0 - uGlassParams.y);
        vec3 alb8 = t8.rgb * (uGlassColor4.rgb + uGlassParams.x * diff8);
        // hardLight(hardLight(_asG.rgb, _asG.a), _ShadowColor) — the inner
        // blend is ``_asG.a`` itself, not the 0.5*a+0.5 of main_skin / the iris
        vec3 tint8 = hardLight(hardLight(uAmbientShadow.rgb, vec3(uAmbientShadow.a)),
                               uShadowColor.rgb);
        tint8 = mix(tint8, vec3(1.0), abs(ndl8));
        vec3 sort8 = uGlassColor4.rgb + (1.0 - uGlassParams.x) * diff8;
        vec3 c8 = (g8 + rim8 + spec8) * sort8 + rim8 * uAmbientShadow.rgb
                + alb8 * tint8;
        c8 *= max(uCharaLightColor * 0.6 + 0.4, uAmbientShadow.rgb);
        c8 = clamp(c8 * uGlassColor4.a + tint8 * alb8, 0.0, 1.0);
        c8 *= max(ndl8, 0.0) * uCharaLightColor + uAmbient;
        float a8 = t8.a * uGlassColor4.a;
        vec3 fc8 = c8;
        if (uFogEnable == 1) {
            float f = clamp((vDepth - uFogRange.x)
                            / max(uFogRange.y - uFogRange.x, 0.001), 0.0, 1.0);
            fc8 = mix(fc8, uFogColor, f * 0.85);
        }
        fragColor = vec4(fc8, a8);
        return;
    }
    if (uToonFamily != 0) {
        // ------------------------------------------------------------------
        // Illusion's forward pass.  The three families differ in exactly two
        // places: where the shadow tint comes from, and how the per-texel
        // shading term combines with the light ramp.  Everything else --
        // the ramp lookup, the ink, the final light term -- is shared.
        // ------------------------------------------------------------------
        vec4 m = (uUseShadeMask == 1) ? texture(uShadeMask, vUV) : vec4(1.0);
        // _DetailMask as itself, for the terms the packing cannot hold.  An
        // unbound slot is **black** — the property table's "black" default is
        // RGBA(0,0,0,0), not opaque black — and every ``1 - dm.x`` below then
        // reads 1, i.e. no effect.
        vec4 dm = (uUseDetail == 1) ? texture(uDetailMask, vUV) : vec4(0.0);
        float ndl = dot(N, L);
        // _RampG is sampled with the raw N.L (clamped by the wrap mode), not
        // a half-lambert: the shipped ramp already places the terminator.
        // **The iris does not go through the ramp** (§7.345).  Its program
        // never samples ``_RampG`` -- so the hard step at N.L = 0.5 that puts
        // a terminator on a cheek has no counterpart on an eye, and forcing
        // one there is what made every card's iris read too dark on its far
        // side.  The material may still carry a ramp; what decides is the
        // program, which is a fact about the asset (一之二).
        // The sclera beside it (6) does sample the ramp (§7.359).
        // Family 7 (``toon_nose_lod0``) has **no lighting at all**: its whole
        // fragment program is four instructions and reads only ``_MainTex``
        // and ``_Color`` — no N.L, no ramp, no tint, no ``_LightColor0``
        // (§7.366).  It is an alpha-blended decal, so ``lit`` is 1 and the
        // terms below that are not multiplies by 1 are excluded by name.
        // The iris adds its highlight weight before the saturate
        // (``mad r0.w, _isHighLight, hi, max(N.L, 0)``), so a catchlight is
        // lit whatever the N.L under it (§7.403).  The
        // ``_expression.a * _exppower`` term beside it is 0 until an eye
        // pattern binds the overlay, which we do not draw.
        float lit = (uToonFamily == 7) ? 1.0
            : (uToonFamily == 5) ? clamp(max(ndl, 0.0) + hlW, 0.0, 1.0)
            : (uUseRamp == 1)
            ? texture(uRamp, vec2(clamp(ndl, 0.0, 1.0), 0.5)).r
            : clamp(ndl, 0.0, 1.0);

        if (uToonFamily == 5 || uToonFamily == 6) {
            // Nothing else touches it: no view ramp (the program samples one
            // texture for the iris, one for the expression overlay and two for
            // the highlights, and none of them is a ramp) and no LineMask
            // multiply (it declares neither).  The sclera samples its map and
            // the ramp and nothing more.
            lit = clamp(lit, 0.0, 1.0);
        } else if (uToonFamily == 3) {
            // main_hair: _AnotherRamp is a *view* ramp and the two combine
            // by min().
            if (uUseViewRamp == 1) {
                float nv = dot(N, V);
                lit = min(lit, texture(uViewRamp,
                        vec2(nv * uViewRampXf.x + uViewRampXf.z, 0.5)).r);
            }
            lit = clamp(lit, 0.0, 1.0);
        } else {
            // main_skin / main_opaque: the LineMask/DetailMask term multiplies
            // the ramp.  Where a second ramp is bound it is sampled by |N.H|
            // and blended in by the mask's alpha.
            if (uUseViewRamp == 1) {
                float nh = abs(dot(N, H));
                float other = texture(uViewRamp,
                        vec2(nh * uViewRampXf.x + uViewRampXf.z, 0.5)).r;
                lit = mix(lit, other, m.a);
            }
            lit = clamp(lit * m.r, 0.0, 1.0);
        }
        // **The character light's shadow** (§7.343), transcribed from
        // ``main_skin``'s shadowed variant: the map is remapped by
        // ``saturate(2s - 1)`` **before** it multiplies the ramp rather than
        // being multiplied in.  That remap is a deepening -- a map value of
        // 0.5 is already full shadow -- and a plain multiply draws a penumbra
        // far paler than the game's.
        //
        // ``max`` with the face's exemption is **not** here: that term is
        // ``_NormalMask.z * _FaceShadowG`` and we bind neither, so writing
        // ``max(0, s)`` would be the same expression with a factor silently
        // pinned to its neutral value.  It is left out and reported (一之一).
        // **A diagnostic, off by default** (§7.341), and it pins the
        // **ramp** — §7.343's shadow multiplies whatever it leaves, so
        // ``--lit 1`` renders the shadow alone.  ``--lit 0`` is still
        // the floor of the whole pass either way (a shadow can only
        // multiply 0 by something).  ``shaded`` is linear in
        // ``lit`` (``mix(tint * lin, lin, lit)``) and colour grading is
        // monotone, so pinning it to 0 and to 1 brackets **every** colour this
        // pass can put on a texel — which is the question §7.126 had to answer
        // by hand ("no value of lit fits Origin's shadow under the chin") and
        // could only answer for one patch.  Doing it here rather than
        // re-deriving the formula in numpy is the point: a second
        // implementation of a shading pass drifts from the one that draws
        // (七), and this one is the one that draws.
        if (uLitOverride >= 0.0) lit = uLitOverride;
        if (uShadowOn > 0.0) {
            // **Normal offset before the projection, not depth bias after
            // it.**  A depth bias big enough to clear the acne on a cheek is
            // several texels deep, and it detaches every contact shadow by the
            // same amount; moving the *sample* along the surface normal costs
            // the shadow only where the surface is nearly edge-on to the
            // light, which is exactly where the acne is.  ``uShadowTexel`` is
            // one texel in **world** units, so this follows the fitted frustum
            // rather than being a number tuned at one zoom.
            float sl = clamp(1.0 - abs(dot(N, uCharaLightDir)), 0.0, 1.0);
            vec3 wp = vWorld + N * uShadowTexel * (1.0 + 2.0 * sl);
            vec4 lp = uLightMatrix * vec4(wp, 1.0);
            vec3 luv = lp.xyz / max(lp.w, 1e-6) * 0.5 + 0.5;
            if (luv.z <= 1.0 && luv.x > 0.0 && luv.x < 1.0
                    && luv.y > 0.0 && luv.y < 1.0) {
                float bias = uShadowBias * (0.5 + 1.5 * sl);
                float occ = 0.0;
                for (int j = -1; j <= 1; ++j)
                    for (int i = -1; i <= 1; ++i) {
                        float d = texture(uShadowMap,
                                          luv.xy + vec2(i, j) * uShadowOn).r;
                        occ += (luv.z - bias > d) ? 1.0 : 0.0;
                    }
                float s = 1.0 - occ / 9.0;
                lit *= clamp(2.0 * s - 1.0, 0.0, 1.0);
            }
        }

        // Three different hue rebuilds, one per family, all read off their own
        // bytecode.  main_skin runs over the albedo alone and keeps 0.66 of its
        // saturation; main_opaque runs over ``albedo * _ShadowColor`` and keeps
        // **half** (``mul r3.z, r3.z, l(0.5)``) — which is why a garment's own
        // shadow colour has to reach the tint even though nothing multiplies by
        // it later; main_emblem builds it out of its own constants: an upper
        // layer scaled by 0.8 instead of 0.9, and a *shader-local*
        // ``_ambientshadow`` that nothing in the game ever writes, so the
        // badge's shadow is hard-lit against black.
        vec3 tint = (uToonFamily == 3 || uToonFamily == 5 || uToonFamily == 6)
            // main_hair and toon_eye_lod0 build the same one: the ambient
            // hard-lit against the material's own ``_shadowcolor``.  §7.345
            // read the iris's bytecode and it is this expression to the
            // instruction, ``movc_sat`` included, and toon_eyew_lod0 is the
            // same again (§7.359).
            ? hardLight(uAmbientBase, uShadowColor.rgb)
            : (uToonFamily == 4)
            ? hardLight(vec3(0.0), 0.8 * hueTint(albedo.rgb, 0.5))
            : (uToonFamily == 2)
            ? hardLight(uAmbientBase,
                        0.9 * hueTint(albedo.rgb * uShadowColor.rgb, 0.5))
            : hardLight(uAmbientBase, 0.9 * hueTint(albedo.rgb, 0.66));
        if (uToonFamily == 2) {
            // main_opaque scales the shadow colour by the **complement** of the
            // value the ramp blend uses -- the register is negated in between
            // (``add r3.xz, -r3.xxzx, l(1,0,1,0)``) -- floored by
            // 1 - _ShadowExtendAnother.  Reading it uncomplemented turns any
            // material with an unbound _LineMask fully black in shadow, which
            // is what the teeth did before this line was corrected.
            tint = clamp(max(1.0 - m.a, uShadeFloor) * tint, 0.0, 1.0);
        }

        vec3 lin = albedo.rgb;
        if (uToonFamily == 3) {
            // main_hair applies the shadow tint **twice**, with two different
            // factors: once mixing the shaded albedo (by lit * (1 - d.g)) and
            // once as a plain multiply (by max(lit, 1 - d.g, 1 - _ShadowExtend)).
            // Collapsing them into one term loses the second, which is what
            // keeps a strand's own line work readable on the lit side.
            float sA = clamp(lit * m.r, 0.0, 1.0);
            float sB = clamp(max(max(lit, m.r), uShadeFloor), 0.0, 1.0);
            float spec = 0.0;
            if (uUseGloss == 1) {
                // The strand highlight: _HairGloss sampled twice a few texels
                // apart along v, differenced.  v walks with the view angle
                // through _SpeclarHeight, which is what makes the band slide
                // across the hair as the head turns.
                float gu = ndl * 0.005 + vUV1.x;
                float gv = (uGlossHeight - 1.0) * dot(tbn[1], V) + vUV1.y;
                float g1 = texture(uGloss, vec2(gu * uGlossXf.x + uGlossXf.z,
                                                (gv + 0.008) * uGlossXf.y + uGlossXf.w)).r;
                float g2 = texture(uGloss, vec2(gu * uGlossXf.x + uGlossXf.z,
                                                gv * uGlossXf.y + uGlossXf.w)).r;
                lin = albedo.rgb * mix(vec3(1.0), tint, (g1 - g2) * 0.5);
                spec = g2 * (lit * 0.5 + 0.5);
            }
            shaded = (mix(tint * lin, lin, sA) + spec) * mix(tint, vec3(1.0), sB);
        } else {
            // The specular goes into the colour the ramp then mixes, not on
            // top of the mix: ``mad r2.xzw, r1.xxyz, r4.xxyz, r2.xxzw`` builds
            // albedo + lobe *first*, and only line 250 lerps it against the
            // shadow tint.  Adding it afterwards lights the shadow side too.
            if (uToonFamily != 4) {
                // The analytic lobe, and it is one property doing three jobs:
                //   spec = saturate(P * (N.H)^(256*P) * _SpecularColor.a)
                // With P = 0 -- which is every garment on the test card and
                // ``cf_m_body`` too -- there is no lobe at all.
                float ndh = max(dot(N, H), 0.0);
                float spec = clamp(uOriginSpecular * pow(ndh, 256.0 * uOriginSpecular)
                                   * uSpecColor.a, 0.0, 1.0);
                if (uToonFamily == 2) {
                    // The *texture* lobe, which is what _SpeclarHeight is for:
                    // _DetailMask.r read again through a parallax step along
                    // the view direction, scaled by (_SpeclarHeight - 1) * 0.8.
                    vec2 puv = vUV + (uSpecThreshold - 1.0) * 0.8
                             * vec2(dot(tbn[0], V), dot(tbn[1], V));
                    float dr = (uUseDetail == 1)
                        ? texture(uDetailMask, puv).r : 0.0;
                    float dsq = min(dr * dr, 1.0);
                    float texSpec = min(min(lit, clamp(dr - dsq, 0.0, 1.0)),
                                        min(uOriginSpecular,
                                            dot(uSpecColor.rgb,
                                                vec3(0.3, 0.59, 0.11))));
                    // the specular *map*: _DetailMask.a floored by
                    // _SpecularPowerNail, bounded by its own complement
                    float sMap = min(max(dm.a, uSpecNail),
                                     min(dsq, max(1.0 - dm.a, uOriginSpecular)));
                    vec3 withTex = lin * (1.0 + texSpec * (2.0 * tint - 2.0))
                        + uSpecColor.rgb * (spec * (1.0 - dm.r) + sMap);
                    vec3 plain = lin + uSpecColor.rgb * spec * (1.0 - dm.r);
                    lin = clamp(mix(withTex, plain, uNoTexSpec), 0.0, 1.0);
                } else if (uToonFamily == 1) {
                    // main_skin's textured lobe.  ``_notusetexspecular`` is a
                    // **lerp weight**, not a switch -- the pass ends with
                    // ``mad_sat r1, _notusetexspecular, analytic - textured,
                    // textured`` -- and ``cf_m_body`` carries 0, so the path
                    // the game keeps is this one and the analytic lobe above is
                    // the one it throws away.  Reading it the other way round
                    // is why this was recorded as "cannot show a difference,
                    // _SpecularColor.a is 0": that alpha only ever gated the
                    // analytic term.
                    //
                    //     t   = saturate(_DetailMask(uv').r * 5/3)
                    //     p   = max(dm.a * _SpecularPower,
                    //               (1 - dm.a) * _SpecularPowerNail)
                    //     k   = min(dm.a, min(lit, min(luma(_SpecularColor),
                    //                                  (t - t^3) * p)))
                    //     out = albedo * (1 + k * specTint) + p * t^3 * _SpecularColor
                    //
                    // On the stock body this is nearly inert -- ``cf_body_00_dm``
                    // has r = 0 on 99.2% of its texels, so the added lobe's 99th
                    // percentile is 0.0008 (a fifth of a level).  It is here
                    // because a modded skin map that paints .r gets a highlight
                    // the game draws and we did not.
                    vec2 puv = vUV + (uSpecThreshold - 1.0) * 0.8
                             * vec2(dot(tbn[0], V), dot(tbn[1], V));
                    float t = clamp(((uUseDetail == 1)
                                     ? texture(uDetailMask, puv).r : 0.0)
                                    * (5.0 / 3.0), 0.0, 1.0);
                    float t3 = t * t * t;
                    float p  = max(dm.a * uOriginSpecular,
                                   (1.0 - dm.a) * uSpecNail);
                    float k  = min(min(dm.a, lit),
                                   min(dot(uSpecColor.rgb,
                                           vec3(0.3, 0.59, 0.11)),
                                       (t - t3) * p));
                    vec3 withTex = lin * (1.0 + k * originSpecTint(albedo.rgb))
                                 + uSpecColor.rgb * (p * t3);
                    lin = clamp(mix(withTex, lin + uSpecColor.rgb * spec,
                                    uNoTexSpec), 0.0, 1.0);
                } else {
                    lin += uSpecColor.rgb * spec;
                }
            }
            shaded = mix(tint * lin, lin, lit);
            if (liq > 0.0) {
                // The liquid's own colour, lit by its own terms (the end of sub-program 8.0): a colour
                // from thin to full by its coverage, a soft diffuse ``(N.L+0.5)*(0.15,0.2,0.3)+(0.85,0.8,0.7)``,
                // the ``0.5 * (N.H)^256`` highlight and the rim the skin keeps for it, mixed over the shaded
                // skin by the coverage -- before the light term and the ink, as there.
                float ndlL = max(dot(N, L), 0.0) + 0.5;
                vec3 diffL = ndlL * vec3(0.15, 0.2, 0.3) + vec3(0.85, 0.8, 0.7);
                float specL = 0.5 * pow(max(dot(N, H), 0.0), 256.0);
                float rimL = clamp(min(pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0),
                                           1.0 + 9.0 * uLiquidRimPower) * 5.0 - 2.0,
                                       (1.0 - dm.b) * 2.5 - 1.5), 0.0, 1.0) * (1.0 - dm.g);
                vec3 liquid = (uLiquidThin + (uLiquidFull - uLiquidThin) * liq) * diffL + specL + rimL * uRimV;
                shaded = mix(shaded, liquid, liq * uLiquidOpacity);
            }
        }

        if (uToonFamily == 2) {
            // main_opaque's rim, whole: the exponent is 1 + 9 * _rimpower, the
            // colour is _SpecularColor scaled by _rimV, the (1 - _DetailMask.b)
            // term is remapped so that only the top third of the channel shows
            // anything, the lobe is capped at 0.6, and it is masked a second
            // time by (1 - _DetailMask.g).  Ours used _rimpower as a strength
            // and a fixed exponent of 3, i.e. a far wider band.
            float rim = pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0),
                            1.0 + 9.0 * uRimStrength);
            rim = clamp(rim * (2.777778 * (1.0 - dm.b) - 1.777778), 0.0, 0.6);
            // ``mad_sat``: the clamp is before the light term, not after
            shaded = clamp(shaded + uSpecColor.rgb * rim * uRimV * (1.0 - dm.g),
                           0.0, 1.0);
        } else if (uRimStrength > 0.0) {
            float rim = pow(1.0 - max(dot(N, V), 0.0), uRimPower);
            shaded += uRimColor.rgb * rim * uRimStrength * m.b;
        }

        // _LineColorG.a is 0 (lineDepth 1.0), so the line colour the ink
        // blends toward collapses to black and the ink is a plain multiply.
        shaded *= m.g;
        // main_opaque applies the shadow tint a **second** time, and by the
        // raw shading term rather than by the ramp-blended one
        // (``mad r2.xyz, r2.yyyy, r2.xzwx, r3.xzwx`` -> lerp(tint, 1, shade)).
        // It is 1 wherever both mask maps are black, which is why collapsing
        // the two into the single mix above went unnoticed: it only shows on a
        // garment that carries line work.  main_hair has the same pair and
        // models it as sA / sB above.
        if (uToonFamily == 2) shaded *= mix(tint, vec3(1.0), m.r);
        // main_hair / main_opaque / main_item all carry the same
        // ``frontFacing ? 1.0 : 0.66``, which is what keeps the inside of a
        // hair shell or an open jacket from reading as brightly as the
        // outside.  main_skin has no such term (its 0.66 scales saturation).
        // main_skin has no such term (its 0.66 scales saturation) and
        // main_emblem's program does not declare is_front_face at all.
        // Neither eye program (5, 6) declares SV_IsFrontFace either (§7.359),
        // and the noseline (7) declares no inputs but POSITION and TEXCOORD0.
        if (!gl_FrontFacing && uToonFamily != 1 && uToonFamily != 4
                && uToonFamily != 5 && uToonFamily != 6
                && uToonFamily != 7) shaded *= 0.66;
        // ``max(_LightColor0 * 0.6 + 0.4, _ambientshadowG)``, straight off the
        // forward pass.  _LightColor0 is the **character** light -- Origin gives it
        // its own directional light per scene and ships every one of them at
        // white / 1.0, so a fully lit surface comes out at its albedo.  The
        // scene's own light and ambient are not in this expression: they light
        // the map, and folding them in here dimmed and warmed every character
        // by the scene's mood (§八 二 5).
        // main_emblem floors the same expression with its own (unwritten, and
        // therefore zero) ``_ambientshadow`` rather than the global.
        // Family 7 has no light term either -- and this one is **not** a
        // multiply by 1 (the floor alone is 0.4), so leaving the noseline in
        // it darkened a line the game draws at its texel value.
        if (uToonFamily != 7) {
            shaded *= (uToonFamily == 4)
                ? max(uCharaLightColor * 0.6 + 0.4, vec3(0.0))
                : max(uCharaLightColor * 0.6 + 0.4, uAmbientFloor);
        }
    } else {
        float ndl = dot(N, L) * 0.5 + 0.5;      // half-lambert reads better on toon
        float lit;
        if (uUseRamp == 1) {
            // the ramp *is* the band shape; sampling it is what makes the
            // terminator look like the game's instead of a hard step
            lit = texture(uRamp, vec2(clamp(ndl + uShadeOffset, 0.0, 1.0), 0.5)).r;
        } else {
            lit = bands(ndl + uShadeOffset);
        }

        // energy-conserving cel model: the band tints the albedo, then one
        // combined light term scales it, so ambient never blows out the highlight
        vec3 shadeTint = mix(uShadowColor.rgb, vec3(1.0), lit);
        shaded = albedo.rgb * shadeTint * (uLightColor * uLightIntensity + uAmbient);

        // toon specular: a small hard highlight, kept off the shadow side
        float ndh = max(dot(N, H), 0.0);
        float spec;
        if (uSpecThreshold > 0.0) {
            // _SpeclarHeight is a cut-off on N.H, not an exponent
            spec = smoothstep(uSpecThreshold, mix(uSpecThreshold, 1.0, 0.35), ndh);
        } else {
            spec = smoothstep(0.52, 0.68, pow(ndh, uSpecPower));
        }
        shaded += uSpecColor.rgb * spec * uSpecStrength * lit
                * (1.0 - 0.85 * max(max(shaded.r, shaded.g), shaded.b));

        // rim / light-wrap along the silhouette
        float rim = pow(1.0 - max(dot(N, V), 0.0), uRimPower);
        rim *= smoothstep(-0.25, 0.55, dot(N, normalize(-uRimLightDir)));
        shaded += uRimColor.rgb * rim * uRimStrength;
    }

    vec3 finalCol = mix(shaded, albedo.rgb, clamp(uUnlit, 0.0, 1.0));

    if (uFogEnable == 1) {
        float f = clamp((vDepth - uFogRange.x) / max(uFogRange.y - uFogRange.x, 0.001), 0.0, 1.0);
        finalCol = mix(finalCol, uFogColor, f * 0.85);
    }
    fragColor = vec4(finalCol, albedo.a);
}
"""

# --------------------------------------------------------------------------
# The character light's shadow (§7.343).
#
# Origin's life-scene character light is ``shadows=soft`` with a 1024 cull, and its
# ``main_*`` forward bases carry a variant that folds a **screen-space** shadow
# map into the ramp term::
#
#     s   = saturate(2 * shadow - 1)                 ; the remap, not a multiply
#     s   = max(_NormalMask.z * _FaceShadowG, s)     ; the face's own exemption
#     lit = s * ramp(N.L)
#
# (§7.341 read that off ``main_skin``'s only shadowed variant; the unshadowed
# one writes ``max(r6.y, 1.0)`` in the same slot, which is ``shadow = 1``
# substituted in -- that pair is the transcription's own calibration.)
#
# Unity builds its screen-space map by rendering the light's depth and then
# resolving it against the camera's; sampling the light's depth directly in the
# forward pass reaches the same value with one buffer instead of two, and the
# softness Unity gets from its filter is the PCF kernel here.
#
# **Depth only, and it still needs the skin block**: what casts is the same
# skinned geometry that draws, so a fixed-function depth pass would put every
# character in bind pose and hang the shadow off the body.
SHADOW_VS = "#version 330 core\n" + _ATTRIBS + _SKIN_BLOCK + """
uniform mat4 uModel;
uniform mat4 uLightMatrix;

void main() {
    vec3 sp, sn;
    skin(aPos, aNormal, sp, sn);
    gl_Position = uLightMatrix * (uModel * vec4(sp, 1.0));
}
"""

SHADOW_FS = """#version 330 core
void main() { }
"""


OUTLINE_VS = "#version 330 core\n" + _ATTRIBS + _SKIN_BLOCK + """
uniform mat4  uModel;
uniform mat4  uView;
uniform mat4  uProj;
uniform float uWidth;      // pixels (procedural) or a scale on Origin's formula
uniform vec2  uViewport;
uniform float uZBias;
uniform vec3  uCameraPos;
// Origin's own outline width: the vertex program extrudes along the normal by
//   (0.3 + 0.1 * distanceToCamera) * _linewidthG * 0.005 * (1 - _DetailMask.b)
// in object space, divided by the object scale so the result is uniform in
// world space.  Hair additionally scales it by the albedo's alpha, which is
// what stops a hull forming around the transparent part of a strand sheet.
uniform int   uWidthMode;     // 0 = screen-space pixels, 1 = Origin world-space,
                              // 2 = HGRP/CharacterNPR's clip-space rule
uniform float uFovHalf;       // half the vertical field of view, radians
uniform float uOffsetZ;       // _OutlineOffsetZ
uniform float uLineWidthG;
uniform sampler2D uMaskTex;
uniform int   uUseMask;
uniform sampler2D uAlphaTex;
uniform int   uUseAlpha;

out vec2 vUV;

void main() {
    vec3 sp, sn;
    // CharacterOutline with _OutlineAverageNormal: the hull goes along the surface's averaged
    // normal (TEXCOORD2 in that game, Mesh.outline_normals here), not split at its hard edges;
    // .w is 0 on a mesh without one.  An array not enabled reads (0, 0, 0, 1): the unit vector is the
    // test, so a draw without the stream keeps its normal
    bool averaged = uWidthMode == 2 && aOutlineNormal.w > 0.5 && dot(aOutlineNormal.xyz, aOutlineNormal.xyz) > 0.25;
    skin(aPos, averaged ? aOutlineNormal.xyz : aNormal, sp, sn);
    vUV = aUV;
    vec4 world = uModel * vec4(sp, 1.0);
    vec3 wn = normalize(mat3(uModel) * sn);
    if (uWidthMode == 1) {
        float dist = length(uCameraPos - world.xyz);
        float w = (0.3 + 0.1 * dist) * uLineWidthG * 0.005 * uWidth;
        if (uUseMask == 1) w *= textureLod(uMaskTex, aUV, 0.0).b;
        if (uUseAlpha == 1) w *= textureLod(uAlphaTex, aUV, 0.0).a;
        world.xyz += wn * w;
        vec4 clip = uProj * uView * world;
        clip.z += uZBias * clip.w;
        gl_Position = clip;
        return;
    }
    vec4 clip = uProj * uView * world;
    vec3 clipN = normalize(mat3(uProj) * (mat3(uView) * wn));
    if (uWidthMode == 2) {
        // CharacterOutline: the normal's direction on screen, a width that is
        // constant in the world up to a distance and then on screen, never
        // under a pixel; then pushed back by a tenth of _OutlineOffsetZ.
        vec2 dir = normalize((uProj * uView * vec4(wn, 0.0)).xy);
        dir.x *= uViewport.y / max(uViewport.x, 1.0);
        float mask = (uUseMask == 1) ? textureLod(uMaskTex, aUV, 0.0).r : 1.0;
        float pushMask = (uUseMask == 1) ? textureLod(uMaskTex, aUV, 0.0).g : 1.0;
        vec2 off = dir * (0.3926991 / uFovHalf) * uWidth
                 * clamp(clip.w * uFovHalf * 4.583663, 0.0, 1.0) * 0.005;
        vec2 least = min(1.5707963 / uFovHalf, max(clip.w, 0.0)) / max(uViewport, vec2(1.0));
        off = mix(off, sign(dir) * least, lessThan(abs(off), least));
        clip.xy += off * mask;          // the mask scales the clamped width: 0 = no line
        float back = 0.1 * uOffsetZ * pushMask;
        vec4 pushed = uProj * vec4(0.0, 0.0, -(clip.w + back), 1.0);
        clip.z = pushed.z / pushed.w * clip.w;
        gl_Position = clip;
        return;
    }
    vec2 px = clipN.xy / max(uViewport, vec2(1.0)) * 2.0;
    clip.xy += px * uWidth * clip.w;
    clip.z += uZBias * clip.w;
    gl_Position = clip;
}
"""

OUTLINE_FS = """#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform vec4 uColor;
// 0 = flat uColor (this project's procedural parts, and every Origin material
//     whose outline pass hard-lights against the global _LineColorG, whose
//     alpha is 0 and therefore resolves to black).
// 1 = main_hair, whose outline pass hard-lights the strand colour against the
//     material's own _LineColor instead of the global one.
// 2 = HGRP/CharacterNPR: the base colour toned down — uColor.rgb is the base
//     colour, uColor.a unused, uTone = (brightness, saturation) — and lit like
//     a fully lit surface (uLight); then, like every program of that game,
//     sent out through the inverse of the frame's colour grade (efUngrade).
uniform int   uColorMode;
uniform sampler2D uTex;
uniform int   uUseTex;
uniform float uAlphaTest;
uniform vec2  uTone;
uniform vec3  uLight;
""" + _EF.UNGRADE_UNIFORMS + _EF.UNGRADE + """

void main() {
    if (uColorMode == 2) {
        vec4 t = (uUseTex == 1) ? texture(uTex, vUV) : vec4(1.0);
        if (t.a * uColor.a < uAlphaTest) discard;
        vec3 c = t.rgb * uColor.rgb * uTone.x;
        float l = dot(c, vec3(0.212673, 0.715152, 0.072175));
        fragColor = vec4(efUngrade((l + (c - l) * uTone.y) * 0.96 * uLight), 1.0);
        return;
    }
    if (uUseTex == 1) {
        vec4 t = texture(uTex, vUV);
        if (t.a < uAlphaTest) discard;
        if (uColorMode == 1) {
            vec3 b = t.rgb;
            vec3 mult = 2.0 * b * uColor.rgb;
            vec3 screen = 1.0 - (2.0 - 2.0 * uColor.rgb) * (1.0 - b);
            fragColor = vec4(clamp(mix(mult, screen, step(vec3(0.5), uColor.rgb)),
                                   0.0, 1.0), 1.0);
            return;
        }
    }
    fragColor = uColor;
}
"""

# --------------------------------------------------------------------------
# fullscreen post-processing
# --------------------------------------------------------------------------
POST_VS = """#version 330 core
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Bloom runs in its own quarter-resolution target: extracting and blurring at
# 1/16 the pixel count costs a fraction of a millisecond, where blurring at full
# resolution (or generating a mip chain every frame) costs milliseconds.
BLOOM_FS = "#version 330 core\n" + _COLORSPACE_BLOCK + """
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uScene;
uniform vec2  uTexel;        // texel size of THIS (quarter-res) target
uniform float uThreshold;
void main() {
    vec3 acc = vec3(0.0);
    float total = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 o = vec2(float(x), float(y)) * uTexel * 1.6;
            float w = (x == 0 && y == 0) ? 2.0 : ((x == 0 || y == 0) ? 1.0 : 0.55);
            // gamma, because the threshold and the additive blend in POST_FS
            // are ours and were calibrated there; the scene target is sRGB so
            // the fetch arrives linear
            acc += amplifyToGamma(texture(uScene, vUV + o).rgb) * w;
            total += w;
        }
    }
    vec3 c = acc / max(total, 0.0001);
    fragColor = vec4(max(c - vec3(uThreshold), vec3(0.0)), 1.0);
}
"""

POST_FS = "#version 330 core\n" + _COLORSPACE_BLOCK + """
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uScene;
uniform sampler2D uDepth;
uniform vec2  uTexel;

uniform float uBloom;
uniform float uBloomThreshold;
uniform float uVignette;
uniform float uSaturation;
uniform float uContrast;
uniform float uBrightness;
uniform vec3  uTint;
uniform float uDofStrength;
uniform vec2  uDofRange;
uniform float uChromatic;
uniform int   uSepia;
uniform sampler2D uBloomTex;
uniform sampler2D uLut;
uniform float uLutAmount;      // 0 = off; Origin runs this at 1

// The scene target is sRGB, so ``texture`` hands back **linear** — which is
// exactly what Unity hands an image effect in a Linear project.  Amplify Color
// converts to gamma before it does anything else, and the whole chain below
// (Illusion's LUT, then our own grade, which was calibrated there) runs in that
// space; ``main`` converts back at the end.
vec3 sampleScene(vec2 uv) { return amplifyToGamma(texture(uScene, uv).rgb); }

// Illusion's colour grade: Hidden/Amplify Color/Base, transcribed from the
// DXBC in resources.assets rather than from Amplify's published source, so
// the constants are the ones the game actually runs:
//
//   mul r1.x, r0.z, l(31.0)          frc r1.x, r1.x
//   mad r0.xyz, r0.xyzx, l(0.030273, 0.968750, 31.0, 0), l(0.000488, 0.015625, 0, 0)
//   add r0.z, -r1.x, r0.z            mad r0.xy, r0.zzzz, l(0.031250, 0, 0, 0), r0.xyxx
//   add r0.zw, r0.xxxy, l(0, 0, 0.031250, 0)
//   two samples, lerp by r1.x
//
// i.e. a 32-cube unrolled into a 1024x32 strip: 31/1024 and 31/32 to land on
// texel centres, 1/32 to step one slice.  The sizes are baked into the
// bytecode, not read from a uniform, so they are constants here too.
vec3 applyLut(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    float b = c.b * 31.0;
    float f = fract(b);
    vec2 uv = vec2(c.r * 0.03027344 + 0.00048828 + floor(b) * 0.03125,
                   c.g * 0.96875 + 0.015625);
    return mix(texture(uLut, uv).rgb,
               texture(uLut, uv + vec2(0.03125, 0.0)).rgb, f);
}

// Blur from a mip level. A 3x3 tap at lod 3 covers a wider radius than a 7x7
// tap at full resolution while doing a ninth of the fetches — on an integrated
// GPU the post pass is bandwidth bound, and this is what buys the frame budget.
vec3 blur(vec2 uv, float radius, float lod) {
    vec3 acc = vec3(0.0);
    float total = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 o = vec2(float(x), float(y)) * uTexel * radius;
            float w = (x == 0 && y == 0) ? 2.0 : ((x == 0 || y == 0) ? 1.0 : 0.6);
            acc += amplifyToGamma(textureLod(uScene, uv + o, lod).rgb) * w;
            total += w;
        }
    }
    return acc / max(total, 0.0001);
}

void main() {
    vec3 col;
    if (uChromatic > 0.001) {
        vec2 d = (vUV - 0.5) * uChromatic * 0.01;
        col = vec3(sampleScene(vUV + d).r, sampleScene(vUV).g, sampleScene(vUV - d).b);
    } else {
        col = sampleScene(vUV);
    }

    if (uDofStrength > 0.001) {
        float z = texture(uDepth, vUV).r;
        float ndc = z * 2.0 - 1.0;
        float lin = (2.0 * uDofRange.x * uDofRange.y)
                  / (uDofRange.y + uDofRange.x - ndc * (uDofRange.y - uDofRange.x));
        float coc = clamp(abs(lin - uDofRange.x) * uDofStrength, 0.0, 1.0);
        if (coc > 0.004) col = mix(col, blur(vUV, 5.0, 2.0), coc);
    }

    if (uBloom > 0.001) {
        col += texture(uBloomTex, vUV).rgb * uBloom;
    }

    // Placed here because Unity runs image effects in component order and on
    // the ActionCamera -- the one the game is played through -- the order is
    // BloomAndFlares, *AmplifyColorEffect*, SunShafts, Vignette, DOF, Blur,
    // SepiaTone.  (The chara maker's camera runs Amplify *before* bloom; the
    // two scenes disagree, and this follows the play mode.)  Everything below
    // is our own grade, not Origin's, so it stacks on top of the LUT.
    if (uLutAmount > 0.001) {
        col = mix(col, applyLut(col), uLutAmount);
    }

    col = (col - 0.5) * uContrast + 0.5 + uBrightness;
    float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(l), col, uSaturation);
    if (uSepia == 1) col = mix(col, vec3(l) * vec3(1.07, 0.94, 0.78), 0.75);
    col *= uTint;

    if (uVignette > 0.001) {
        vec2 d = (vUV - 0.5) * 2.0;
        float v = 1.0 - dot(d, d) * uVignette * 0.45;
        col *= clamp(v, 0.0, 1.0);
    }
    // Amplify's tail, then the back-buffer encode.  The game does both: the
    // pass ends by converting to linear and Unity's final blit writes through
    // an sRGB surface.  Our post target is a plain RGBA8, so the second half
    // is done here rather than by the hardware.
    fragColor = vec4(encodeSrgb(amplifyToLinear(clamp(col, 0.0, 1.0))), 1.0);
}
"""

# --------------------------------------------------------------------------
# unlit helpers (grid, gizmos, backdrop)
# --------------------------------------------------------------------------
LINE_VS = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uMVP;
out vec4 vColor;
void main() { vColor = aColor; gl_Position = uMVP * vec4(aPos, 1.0); }
"""

LINE_FS = "#version 330 core\n" + _COLORSPACE_BLOCK + """
in vec4 vColor;
out vec4 fragColor;
uniform float uAlpha;
// Grid and gizmo colours are authored as the bytes we want on screen, and the
// scene target encodes on write, so they are decoded here — otherwise every
// overlay comes out lighter than it was picked.  ``amplifyToLinear`` rather
// than the spec curve only because it is the one already in this file; the two
// differ by under a level and nothing measures an overlay.
void main() { fragColor = vec4(amplifyToLinear(vColor.rgb), vColor.a * uAlpha); }
"""

SKY_VS = """#version 330 core
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 1.0, 1.0);
}
"""

SKY_IMAGE_FS = """#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uImage;
uniform float uDim;
void main() {
    // vUV: y=0 bottom in our sky triangle; image UVs usually top-left origin
    vec2 uv = vec2(vUV.x, 1.0 - vUV.y);
    vec3 col = texture(uImage, uv).rgb;
    fragColor = vec4(col * (1.0 - uDim), 1.0);
}
"""

SKY_FS = """#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform vec3 uTop;
uniform vec3 uBottom;
uniform float uPower;
void main() {
    float t = pow(clamp(vUV.y, 0.0, 1.0), uPower);
    fragColor = vec4(mix(uBottom, uTop, t), 1.0);
}
"""
