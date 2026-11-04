import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from polylode.extractor.morph_playback import Track
from polylode.interchange import glb


@pytest.mark.parametrize("mode,expected", [("LINEAR", .25), ("STEP", 0), ("CUBICSPLINE", .15625)])
def test_weight_interpolation_and_endpoints(mode, expected):
    values = np.array([[0.], [1.]])
    if mode == "CUBICSPLINE":
        values = np.array([[[0.], [0.], [0.]], [[0.], [1.], [0.]]])
    track = Track([], np.array([0., 2.]), values, mode)
    assert track.sample(.5)[0] == pytest.approx(expected)
    assert track.sample(-1)[0] == 0
    assert track.sample(3)[0] == 1


def test_preview_plays_late_morph_and_resets_when_switching(tmp_path):
    from test_interchange_glb import _scene
    document, arrays = _scene(tmp_path)
    document["parts"][0]["shapes"] = [f"shape{i}" for i in range(10)]
    for i in range(10):
        arrays[f"p0_s{i}_idx"] = np.array([3])
        arrays[f"p0_s{i}_delta"] = np.array([[0, 0, .1 * (i + 1)]], np.float32)
    document["animations"] = [{"name": "blink", "rate": 2, "frames": 3},
                              {"name": "neutral", "rate": 2, "frames": 3}]
    for i in range(2):
        arrays[f"anim{i}"] = np.repeat(arrays["bone_rest"][None], 3, axis=0)
        arrays[f"p0_anim{i}_weights"] = np.zeros((3, 10), np.float32)
    arrays["p0_anim0_weights"][:, 9] = [0, .5, 1]
    path = tmp_path / "late-morph.glb"
    glb.write(document, arrays, tmp_path, path)
    script = '''
import sys
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from polylode.extractor.preview import Offscreen
from polylode.extractor.view_host import Poses
app = QGuiApplication([])
view = Offscreen()
assert not view.failure, view.failure
assert view.load(Path(sys.argv[1])) == "ready"
poses = Poses()
assert poses.read(view.root) == ["blink", "neutral"]
assert poses.morphs is not None
timeline, player = poses.found[0][1:]
player.stop()
timeline.setProperty("currentFrame", 750.)
targets = [track.targets[9] for track in poses.morphs.tracks[0]]
assert targets
assert all(abs(target.property("weight") - .75) < 1e-5 for target in targets)
assert poses.play(1)
assert all(target.property("weight") == 0 for target in targets)
poses.clear()
assert poses.morphs is None
print("late morph playback and reset verified")
'''
    env = dict(os.environ, QSG_RHI_BACKEND="opengl")
    if sys.platform == "win32":
        import PySide6
        env["QT_QPA_PLATFORM"] = "windows"
        env["PATH"] = str(Path(PySide6.__file__).parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run([sys.executable, "-c", script, str(path)], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
