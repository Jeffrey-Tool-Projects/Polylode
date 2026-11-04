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
    property real cutoff: original.alphaMode === PrincipledMaterial.Mask ? original.alphaCutoff : 0
    shadingMode: CustomMaterial.Shaded
    cullMode: original.cullMode
    sourceBlend: original.alphaMode === PrincipledMaterial.Blend ? CustomMaterial.SrcAlpha : CustomMaterial.NoBlend
    destinationBlend: original.alphaMode === PrincipledMaterial.Blend ? CustomMaterial.OneMinusSrcAlpha : CustomMaterial.NoBlend
    fragmentShader: "character.frag"
}
