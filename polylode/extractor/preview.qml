// The model view: one model file (the preview's .glb, or any file the viewer opens), lit and framed, turned with
// the mouse (§7.1455).  The viewer page sets wireframe, backdrop and light, and calls look() for a fixed view.
import QtQuick
import QtQuick3D
import QtQuick3D.AssetUtils
import QtQuick3D.Helpers

Rectangle {
    id: root
    color: backdrop
    property url source: ""
    property color backdrop: "#1e2129"
    property bool wireframe: false
    // a multiplier on the lights
    property real light: 1.0
    // the light from straight above (a scene light, and the same over the character material's own)
    property real topLight: 0.6
    property string characterStyle: "studio"
    property var nativeMaterials: []
    property var styledModels: []
    property string styleStatus: ""
    property string styleUnavailableText: "此模型無法套用角色著色；目前使用攝影棚"
    Component { id: characterMaterial; CharacterMaterial {} }
    onCharacterStyleChanged: applyStyle()
    function clearStyles() {
        for (var i = 0; i < styledModels.length; ++i) {
            var item = styledModels[i];
            item.model.materials = item.originals;
            for (var j = 0; j < item.adapted.length; ++j) item.adapted[j].destroy();
        }
        styledModels = [];
    }
    function collectModels(node, result) {
        if (node.materials !== undefined) {
            var originals = [];
            for (var m = 0; m < node.materials.length; ++m) originals.push(node.materials[m]);
            result.push({model: node, originals: originals, adapted: []});
        }
        for (var i = 0; i < node.children.length; ++i) collectModels(node.children[i], result);
    }
    function applyStyle() {
        for (var i = 0; i < styledModels.length; ++i) {
            var item = styledModels[i];
            item.model.materials = characterStyle === "studio" || !item.adapted.length ? item.originals : item.adapted;
            for (var j = 0; j < item.adapted.length; ++j)
                item.adapted[j].game = characterStyle === "wuwa" ? 3 : characterStyle === "gf2" ? 2 : 1;
        }
    }
    function prepareStyles() {
        clearStyles();
        var models = [];
        collectModels(loader, models);
        styledModels = models;
        styleStatus = "unavailable";
        if (models.length !== nativeMaterials.length) return;
        for (var i = 0; i < models.length; ++i)
            if (models[i].originals.length !== nativeMaterials[i].length) return;
        for (var i = 0; i < models.length; ++i) {
            var item = models[i];
            for (var j = 0; j < item.originals.length; ++j) {
                var data = nativeMaterials[i][j];
                var mat = characterMaterial.createObject(loader, {original: item.originals[j],
                    intensity: Qt.binding(function() { return root.light; }),
                    topLight: Qt.binding(function() { return root.topLight; }),
                    shadowBrightness: data.shadowBrightness === undefined ? .5 : data.shadowBrightness,
                    shadowSaturation: data.shadowSaturation === undefined ? 1 : data.shadowSaturation,
                    smoothness: data.smoothness === undefined ? .5 : data.smoothness,
                    metallic: data.metallic || 0, specularLevel: data.specularLevel || 0,
                    sourceFamily: data.family || "", rampAvailable: Boolean(data.ramp),
                    surfaceAvailable: Boolean(data.surface), shadowLutAvailable: Boolean(data.shadowLut),
                    rampSrgb: Boolean(data.rampSrgb), surfaceSrgb: Boolean(data.surfaceSrgb), shadowLutSrgb: Boolean(data.shadowLutSrgb),
                    hasBase: Boolean(data.baseTexture), emission: data.emission || 0});
                mat.baseMap.texture.source = data.baseTexture || "";
                if (data.baseUv) mat.baseUv = Qt.vector4d(data.baseUv[0], data.baseUv[1], data.baseUv[2], data.baseUv[3]);
                mat.baseRotation = data.baseRotation || 0;
                mat.rampMap.texture.source = data.ramp || "";
                mat.surfaceMap.texture.source = data.surface || "";
                mat.shadowLutMap.texture.source = data.shadowLut || "";
                var parameters = ["backShadowOffset", "frontShadowOffset", "shadowProcess", "shadowWidth",
                                  "pbrRate", "specularPower", "metalSpecularPower"];
                for (var p = 0; p < parameters.length; ++p)
                    if (data[parameters[p]] !== undefined) mat[parameters[p]] = data[parameters[p]];
                item.adapted.push(mat);
            }
        }
        styleStatus = "";
        applyStyle();
    }
    property string hint: "選一個模型後按「預覽」"
    // a file loaded and framed: the view's process then looks for its poses (view_host.Poses)
    signal loaded()
    // set from Python in the chosen language (preview.PreviewPanel.apply_texts)
    property string hintSource: ""
    property string loadingText: "載入中…"
    property string errorText: "讀不進這個檔案："
    // the loader's state as the window reads it: "", "loading", "ready" or the loader's error
    property string state_: loader.source.toString() !== source.toString() ? (source.toString() ? "loading" : "")
                           : loader.status === RuntimeLoader.Success ? "ready"
                           : loader.status === RuntimeLoader.Error ? root.errorText + loader.errorString
                           : source.toString() ? "loading" : ""

    // RuntimeLoader reads the file on this thread and says nothing until it is done: the loader is
    // emptied at once and given the file a frame later, so the moving bar below is drawn before the
    // read holds the window
    onSourceChanged: {
        clearStyles();
        loader.source = "";
        deferred.restart();
    }
    Timer {
        id: deferred
        interval: 50
        onTriggered: loader.source = root.source
    }
    // the model's box as the window measured it from a glTF file's own data (preview.gltf_bounds): the
    // loader's own bounds are all zero for a skinned model, and every character is one; without it, the
    // loader's bounds, or the boxes of the scene's models added up (measured)
    property bool framed: false
    property vector3d frameLow: Qt.vector3d(0, 0, 0)
    property vector3d frameHigh: Qt.vector3d(0, 0, 0)
    property vector3d measuredLow: Qt.vector3d(0, 0, 0)
    property vector3d measuredHigh: Qt.vector3d(0, 0, 0)
    property bool loaderBoxEmpty: loader.bounds.maximum.minus(loader.bounds.minimum).length() <= 0
    property vector3d low: framed ? frameLow : loaderBoxEmpty ? measuredLow : loader.bounds.minimum
    property vector3d high: framed ? frameHigh : loaderBoxEmpty ? measuredHigh : loader.bounds.maximum
    property vector3d centre: Qt.vector3d((low.x + high.x) / 2, (low.y + high.y) / 2, (low.z + high.z) / 2)
    property real size: measured ? Math.max(high.x - low.x, high.y - low.y, high.z - low.z, 0.001) : 3
    // whether there is a box to frame by at all
    property bool measured: high.minus(low).length() > 0

    // the boxes of every model under *node*, in the scene's frame, grown into measuredLow/High
    function measure(node, box) {
        if (node.bounds !== undefined && node.mapPositionToScene !== undefined) {
            var a = node.bounds.minimum, b = node.bounds.maximum;
            if (b.minus(a).length() > 0) {
                for (var i = 0; i < 8; ++i) {
                    var p = node.mapPositionToScene(Qt.vector3d(i & 1 ? b.x : a.x, i & 2 ? b.y : a.y, i & 4 ? b.z : a.z));
                    box.low = box.low ? Qt.vector3d(Math.min(box.low.x, p.x), Math.min(box.low.y, p.y), Math.min(box.low.z, p.z)) : p;
                    box.high = box.high ? Qt.vector3d(Math.max(box.high.x, p.x), Math.max(box.high.y, p.y), Math.max(box.high.z, p.z)) : p;
                }
            }
        }
        for (var c = 0; c < node.children.length; ++c)
            measure(node.children[c], box);
    }

    // the model from *yaw* and *pitch* degrees (0, 0: from the front), centred on its box and as close as
    // lets the whole box show in the view, whichever way the view is wider
    function look(yaw, pitch) {
        if (!root.measured) {
            // no box (a file the window could not measure): the origin, from a fixed distance
            origin.position = Qt.vector3d(0, 0, 0);
            origin.eulerRotation = Qt.vector3d(pitch, yaw, 0);
            camera.position = Qt.vector3d(0, 0, 3);
            camera.eulerRotation = Qt.vector3d(0, 0, 0);
            return;
        }
        var half = Math.tan(camera.fieldOfView * Math.PI / 360);    // the vertical half view, as a slope
        var aspect = Math.max(view.width, 1) / Math.max(view.height, 1);
        var tall = (high.y - low.y) / 2;
        var wide = Math.max(high.x - low.x, high.z - low.z) / 2;      // any side may face the camera
        var deep = Math.max(high.x - low.x, high.z - low.z) / 2;
        var distance = Math.max(tall / half, wide / (half * aspect)) * 1.12 + deep;
        origin.position = root.centre;
        origin.eulerRotation = Qt.vector3d(pitch, yaw, 0);
        camera.position = Qt.vector3d(0, 0, Math.max(distance, root.size * 0.5));
        camera.eulerRotation = Qt.vector3d(0, 0, 0);
    }

    // framed again from the angle it is seen from now: a pose shown elsewhere than the rest (view_host.Poses)
    function refit() {
        look(origin.eulerRotation.y, origin.eulerRotation.x);
    }

    View3D {
        id: view
        anchors.fill: parent
        camera: camera
        environment: SceneEnvironment {
            clearColor: root.backdrop
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
            debugSettings: DebugSettings { wireframeEnabled: root.wireframe }
        }

        Node {
            id: origin
            objectName: "origin"
            // the model's centre: the camera orbits it
            position: root.centre
            PerspectiveCamera {
                id: camera
                objectName: "camera"
                fieldOfView: 30
                clipNear: root.size / 1000
                clipFar: root.size * 100
                // placed by root.look() once a model is loaded
                z: root.size * 2.2
            }
        }

        DirectionalLight { eulerRotation.x: -35; eulerRotation.y: 30; brightness: 1.1 * root.light }
        DirectionalLight { eulerRotation.x: -10; eulerRotation.y: 200; brightness: 0.45 * root.light }
        DirectionalLight { eulerRotation.x: 60; brightness: 0.25 * root.light }
        // from above, straight down: heads, shoulders and the tops of things
        DirectionalLight { eulerRotation.x: -90; brightness: root.topLight * root.light }

        RuntimeLoader {
            id: loader
            objectName: "loader"
            // set by root.onSourceChanged, a frame after the file is chosen
            // a newly loaded model is seen from the front, framed to its box, whatever the last one was turned to
            onStatusChanged: {
                if (status !== RuntimeLoader.Success)
                    return;
                if (!root.framed && root.loaderBoxEmpty) {
                    var box = {};
                    root.measure(loader, box);
                    root.measuredLow = box.low ? box.low : Qt.vector3d(0, 0, 0);
                    root.measuredHigh = box.high ? box.high : Qt.vector3d(0, 0, 0);
                }
                root.look(0, 0);
                root.prepareStyles();
                root.loaded();
            }
        }
    }

    OrbitCameraController {
        id: orbit
        anchors.fill: parent
        origin: origin
        camera: camera
    }

    // the controller pans only with Ctrl and the left button; the right button pans too, through its own calls
    Item {
        anchors.fill: parent
        DragHandler {
            target: null
            acceptedButtons: Qt.RightButton
            onActiveChanged: {
                if (active)
                    orbit.startPan(Qt.vector2d(centroid.position.x, centroid.position.y));
                else
                    orbit.endPan();
            }
            onCentroidChanged: orbit.panEvent(Qt.vector2d(centroid.position.x, centroid.position.y))
        }
    }

    Text {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.margins: 8
        color: "#e9ecf2"
        wrapMode: Text.WordWrap
        visible: root.characterStyle !== "studio" && root.styleStatus !== "" && root.state_ === "ready"
        text: root.styleUnavailableText
    }
    Text {
        id: message
        anchors.centerIn: parent
        visible: root.state_ !== "ready"
        color: Qt.colorEqual(root.backdrop, "#e9ecf2") ? "#3a3f4c" : "#c9ced8"
        font.pixelSize: 14
        text: root.state_ === "" ? root.hint : root.state_ === "loading" ? root.loadingText : root.state_
    }

    // while a file loads: a moving bar (the loader reports no count, only that it is done)
    Rectangle {
        id: track
        objectName: "loadingBar"
        visible: root.state_ === "loading"
        anchors.top: message.bottom
        anchors.topMargin: 12
        anchors.horizontalCenter: parent.horizontalCenter
        width: Math.min(260, root.width * 0.6)
        height: 6
        radius: 3
        clip: true
        color: Qt.colorEqual(root.backdrop, "#e9ecf2") ? "#c9ced8" : "#333846"
        Rectangle {
            width: track.width / 3
            height: track.height
            radius: track.radius
            color: "#2f6feb"
            NumberAnimation on x {
                running: track.visible
                from: -track.width / 3
                to: track.width
                duration: 1100
                loops: Animation.Infinite
            }
        }
    }
}
