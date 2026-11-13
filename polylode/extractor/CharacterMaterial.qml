import QtQuick
import QtQuick3D

CustomMaterial {
    property var original
    property int game: 1
    property string sourceFamily: ""
    property bool rampAvailable: false
    property bool surfaceAvailable: false
    property bool shadowLutAvailable: false
    property bool rampSrgb: false
    property bool surfaceSrgb: false
    property bool shadowLutSrgb: false
    property real intensity: 1
    // the view's light from above (preview.qml), over the source's own lighting
    property real topLight: 0
    property vector4d tint: Qt.vector4d(original.baseColor.r, original.baseColor.g, original.baseColor.b, original.baseColor.a)
    property bool hasBase: false
    property vector4d baseUv: Qt.vector4d(1, 1, 0, 0)
    property real baseRotation: 0
    property TextureInput baseMap: TextureInput { texture: Texture { source: "" } }
    property TextureInput rampMap: TextureInput { texture: Texture { source: ""; tilingModeHorizontal: Texture.ClampToEdge; tilingModeVertical: Texture.ClampToEdge } }
    property bool hasRamp: rampAvailable && ((game === 1 && sourceFamily === "endfield") || (game === 2 && sourceFamily === "gf2"))
    property bool hasSurface: surfaceAvailable && ((game === 1 && sourceFamily === "endfield") || (game === 2 && sourceFamily === "gf2"))
    property real emission: 0
    property real backShadowOffset: -0.1
    property real frontShadowOffset: 0.4
    property real shadowProcess: 0.5
    property real shadowWidth: 0.01
    property real pbrRate: 1
    property real specularPower: 5
    property real metalSpecularPower: 5000
    // Explicit neutral preview illumination; these are not claimed game scene colors.
    property vector3d sceneLight: Qt.vector3d(1, 1, 1)
    property vector3d sceneShadow: Qt.vector3d(.18, .18, .18)
    property TextureInput surfaceMap: TextureInput { texture: Texture { source: "" } }
    property TextureInput shadowLutMap: TextureInput { texture: Texture { source: ""; tilingModeHorizontal: Texture.ClampToEdge; tilingModeVertical: Texture.ClampToEdge } }
    property real shadowBrightness: 0.5
    property real shadowSaturation: 1
    property real smoothness: 0.5
    property real metallic: 0
    property real specularLevel: 0
    // the eye's own program (character_styles.eye_style): 1 Endfield's iris, 2 GF2's eye, 3 Wuthering Waves' base
    // pass (its lit colour is the base map, its unlit emission eyeEmission); drawn when the style is its game's
    property int eyeMode: 0
    property bool eyeOn: eyeMode !== 0 && eyeMode === game
    property bool eyeHighlight: false
    property bool eyeMatcapOn: false
    property vector4d eyeHighlightColor: Qt.vector4d(1, 1, 1, 1)
    property vector4d eyeScattering: Qt.vector4d(1, 1, 1, 1)
    property vector4d matcapColor: Qt.vector4d(1, 1, 1, 1)
    property real eyeParallax: 0
    property real matcapNormalScale: 1
    property real eyeShadowIntensity: 0.25
    property real eyeSpecularIntensity: 0
    property vector4d eyeFinalTint: Qt.vector4d(1, 1, 1, 1)
    property real eyeAddScale: 1
    property real eyeEmissionStrength: 0
    property bool hasEyeBase: false
    property bool hasEyeMatcap: false
    property bool hasEyeSpecular: false
    property bool hasEyeAdd: false
    property bool hasEyeMultiply: false
    property bool hasEyeEmission: false
    property bool eyeMatcapSrgb: false
    property bool eyeSpecularSrgb: false
    property bool eyeAddSrgb: false
    property bool eyeEmissionSrgb: false
    property TextureInput eyeBaseMap: TextureInput { texture: Texture { source: "" } }
    property TextureInput eyeMatcapMap: TextureInput { texture: Texture { source: ""; tilingModeHorizontal: Texture.ClampToEdge; tilingModeVertical: Texture.ClampToEdge } }
    property TextureInput eyeSpecularMap: TextureInput { texture: Texture { source: "" } }
    property TextureInput eyeAddMap: TextureInput { texture: Texture { source: "" } }
    property TextureInput eyeMultiplyMap: TextureInput { texture: Texture { source: "" } }
    property TextureInput eyeEmissionMap: TextureInput { texture: Texture { source: "" } }
    property real cutoff: original.alphaMode === PrincipledMaterial.Mask ? original.alphaCutoff : 0
    shadingMode: CustomMaterial.Shaded
    cullMode: original.cullMode
    sourceBlend: original.alphaMode === PrincipledMaterial.Blend ? CustomMaterial.SrcAlpha : CustomMaterial.NoBlend
    destinationBlend: original.alphaMode === PrincipledMaterial.Blend ? CustomMaterial.OneMinusSrcAlpha : CustomMaterial.NoBlend
    fragmentShader: "character.frag"
}
