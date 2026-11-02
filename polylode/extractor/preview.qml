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
    // a multiplier on the three lights
    property real light: 1.0
    property string hint: "選一個模型後按「預覽」"
    // set from Python in the chosen language (preview.PreviewPanel.apply_texts)
    property string hintSource: ""
    property string loadingText: "載入中…"
    property string errorText: "讀不進這個檔案："
    // the loader's state as the window reads it: "", "loading", "ready" or the loader's error
    property string state_: loader.status === RuntimeLoader.Success ? "ready"
                           : loader.status === RuntimeLoader.Error ? root.errorText + loader.errorString
                           : source.toString() ? "loading" : ""
    property vector3d low: loader.bounds.minimum
    property vector3d high: loader.bounds.maximum
    property vector3d centre: Qt.vector3d((low.x + high.x) / 2, (low.y + high.y) / 2, (low.z + high.z) / 2)
    property real size: Math.max(high.x - low.x, high.y - low.y, high.z - low.z, 0.01)

    // the model from *yaw* and *pitch* degrees (0, 0: from the front), framed again and centred
    function look(yaw, pitch) {
        origin.position = root.centre;
        origin.eulerRotation = Qt.vector3d(pitch, yaw, 0);
        camera.position = Qt.vector3d(0, 0, root.size * 2.2);
        camera.eulerRotation = Qt.vector3d(0, 0, 0);
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
                // far enough that the whole height fits a 30° view, with a margin
                z: root.size * 2.2
            }
        }

        DirectionalLight { eulerRotation.x: -35; eulerRotation.y: 30; brightness: 1.1 * root.light }
        DirectionalLight { eulerRotation.x: -10; eulerRotation.y: 200; brightness: 0.45 * root.light }
        DirectionalLight { eulerRotation.x: 60; brightness: 0.25 * root.light }

        RuntimeLoader {
            id: loader
            source: root.source
            // a newly loaded model is seen from the front, whatever the last one was turned to
            onStatusChanged: if (status === RuntimeLoader.Success) root.look(0, 0)
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
        anchors.centerIn: parent
        visible: root.state_ !== "ready"
        color: Qt.colorEqual(root.backdrop, "#e9ecf2") ? "#3a3f4c" : "#c9ced8"
        font.pixelSize: 14
        text: root.state_ === "" ? root.hint : root.state_ === "loading" ? root.loadingText : root.state_
    }
}
