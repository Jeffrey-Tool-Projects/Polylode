"""A Wuthering Waves character's display poses from the reader's decoded sequences (``wuthering_waves/poses``)."""
import json

import numpy as np
from scipy.spatial.transform import Rotation

from polylode.endfield.rigscene import LEAST_SCALE
from polylode.wuthering_waves import poses as W


def test_expression_curves_follow_source_time_and_all_matching_material_parts():
    record = {"name": "smile", "rate": 60, "rate_scale": 2,
              "curves": {"names": ["smile", "blink"], "rate": 15, "values": [[0, 1], [.2, .2]]}}
    parts = [{"shapes": ["blink", "smile", "other"]}, {"shapes": ["smile"]}, {"shapes": []}]
    weights = W.shape_weights(record, parts, 3)
    np.testing.assert_allclose(weights[0], [[.2, 0, 0], [.2, .5, 0], [.2, 1, 0]])
    np.testing.assert_allclose(weights[1][:, 0], [0, .5, 1])
    assert 2 not in weights


def test_undecoded_expression_curves_are_not_silently_dropped():
    import pytest
    with pytest.raises(ValueError, match="not decoded"):
        W.shape_weights({"name": "face", "curves": {"names": ["smile"], "values": None}}, [], 2)


def test_decoded_reader_playback_rate_does_not_accelerate_the_expression_samples(tmp_path):
    record = {"name": "face", "rate": 60, "rate_scale": 2, "frames": 3, "bones": ["root"],
              "curves": {"names": ["smile"], "rate": 30, "values": [[0, .5, 1]]}}
    (tmp_path / "0.json").write_text(json.dumps(record))
    np.zeros((3, 1, 10), dtype="<f4").tofile(tmp_path / "0.bin")
    decoded, frames = W.read(tmp_path, 0)
    np.testing.assert_allclose(W.shape_weights(decoded, [{"shapes": ["smile"]}], len(frames))[0][:, 0], [0, .5, 1])


def test_reader_without_expression_capability_requests_an_update(tmp_path, monkeypatch):
    import pytest
    from polylode.extractor import wuwa_source as source
    from polylode.extractor.sources import ExtractorError
    package = "Client/Content/Aki/Character/Role/FemaleXL/Xin/R2T1XinMd10011/Model/R2T1XinMd10011.uasset"
    index = tmp_path / "index.txt"
    index.write_text(W.ui_folder(package) + "Ui_Loop.uasset\n", encoding="utf-8")
    monkeypatch.setattr(source, "archive_index", lambda settings, log: index)
    monkeypatch.setattr(source, "_reader", lambda settings, mode, folder, listing, log:
                        (folder / "0.json").write_text(json.dumps({"name": "idle", "frames": 2})))
    with pytest.raises(ExtractorError, match="表情"):
        source._poses_of(None, package, tmp_path, tmp_path / "poses", lambda message: None)


def test_every_wuwa_pose_on_the_models_own_skeleton_is_taken_held_ones_first():
    package = "Client/Content/Aki/Character/Role/FemaleXL/Xin/R2T1XinMd10011/Model/R2T1XinMd10011.uasset"
    assert W.ui_folder(package) == "Client/Content/Aki/Character/Role/FemaleXL/Xin/CommonAnim/UiAnim/"
    rest = np.array([0, 0, 0, 1, 0, 0, 10, 1, 1, 1], dtype=np.float64)
    loop = np.stack([rest, rest + [0, 0, 0, 0, 0, 0, 3, 0, 0, 0], rest])[:, None, :]
    start = np.stack([rest, rest + [0, 0, 0, 0, 0, 0, 3, 0, 0, 0]])[:, None, :]
    own = {"bones": ["Hips"], "keyed": [True], "additive": False,
           "skeleton": "Client/Content/Aki/Character/Role/FemaleXL/Xin/R2T1XinMd10011/Model/R2T1XinMd10011_Skeleton"}
    other = dict(own, skeleton="Client/Content/Aki/Character/Role/FemaleXL/BaiLian/R2T1BaiLianMd10011/Model/S")
    picked = W.chosen([(own, start), (own, loop), (other, loop), (dict(own, additive=True), loop)], package)
    assert [f for _r, f in picked] == [loop, start]                       # the transition too, after the held pose
    assert W.holds(own, loop) and not W.holds(own, start)


def test_a_wuwa_frame_is_made_as_the_cases_rest_is(tmp_path):
    # one bone 1 m up (Unreal: 100 cm on Z), turned 90° about Z; its own keys give the case's rest back
    rest_q = Rotation.from_euler("z", 90, degrees=True).as_quat()
    record = {"bones": ["Hips"], "keyed": [True]}
    native = np.eye(4)
    native[:3, :3] = W.C @ Rotation.from_quat(rest_q).as_matrix() @ W.C.T
    native[:3, 3] = [0, 0, 1]
    values = np.array([[[*rest_q, 0, 0, 100, 1, 1, 1]]])
    frames = W.frames(record, values, [{"name": "Hips", "parent": -1}], native[None])
    assert np.allclose(frames[0, 0], native, atol=1e-6)
    # a second bone the clip does not key keeps its native local under the moved parent
    child = native @ np.array([[1, 0, 0, 0], [0, 1, 0, .2], [0, 0, 1, 0], [0, 0, 0, 1.]])
    moved = np.array([[[*rest_q, 0, 0, 150, 1, 1, 1]]])
    frames = W.frames(record, moved, [{"name": "Hips", "parent": -1}, {"name": "Spine", "parent": 0}],
                      np.stack([native, child]))
    assert np.allclose(frames[0, 1, :3, 3], child[:3, 3] + [0, 0, .5], atol=1e-6)


def test_the_readers_frames_are_read_back_as_written(tmp_path):
    record = {"name": "Ui_Weapon_Loop", "bones": ["A", "B"], "keyed": [True, False], "frames": 3}
    (tmp_path / "0.json").write_text(json.dumps(record), encoding="utf-8")
    values = np.arange(3 * 2 * 10, dtype="<f4")
    values.tofile(tmp_path / "0.bin")
    back, frames = W.read(tmp_path, 0)
    assert back["name"] == "Ui_Weapon_Loop" and frames.shape == (3, 2, 10) and frames[2, 1, 9] == values[-1]


def test_an_acl_buffer_is_decoded_onto_the_bones_its_tracks_drive(tmp_path):
    import struct
    from polylode.endfield import acl
    # a compressed_tracks header: 2 tracks, 3 samples at 30 a second (the rest of the buffer is the decoder's)
    header = struct.pack("<IIIHBBIIf", 64, 0, acl.TAG, 10, 0, 0, 2, 3, 30.0)
    (tmp_path / "0.acl").write_bytes(header + bytes(32))
    reference = [[0, 0, 0, 1, 1, 2, 3, 1, 1, 1], [0, 0, 0, 1, 4, 5, 6, 1, 1, 1], [0, 0, 0, 1, 7, 8, 9, 1, 1, 1]]
    record = {"name": "Ui_Weapon_Loop", "bones": ["Root", "Hips", "Spine"], "tracks": [2, 0], "reference": reference,
              "rate_scale": 1.0}
    (tmp_path / "0.json").write_text(json.dumps(record), encoding="utf-8")

    class Decoder:
        def decode(self, buffer, width):
            assert width == acl.TRANSFORM
            values = np.zeros((3, 2, 10), dtype=np.float32)
            values[..., 3] = 1
            values[:, 0, 4:7] = [[10, 0, 0], [11, 0, 0], [12, 0, 0]]          # track 0 drives Spine
            values[:, 1, 4:7] = [20, 0, 0]                                     # track 1 drives Root
            values[..., 7:] = 1
            return values

    back, frames = W.read(tmp_path, 0, Decoder())
    assert back["frames"] == 3 and back["rate"] == 30.0 and back["keyed"] == [True, False, True]
    assert np.allclose(frames[:, 2, 4], [10, 11, 12]) and np.allclose(frames[:, 0, 4], 20)
    assert np.allclose(frames[:, 1, 4:7], [4, 5, 6])                           # Hips: no track, its ref pose


def test_old_reader_reports_that_poses_cannot_be_exported(tmp_path, monkeypatch):
    import pytest
    from polylode.extractor import wuwa_source as source
    from polylode.extractor.sources import ExtractorError

    package = "Client/Content/Aki/Character/Role/FemaleXL/Xin/R2T1XinMd10011/Model/R2T1XinMd10011.uasset"
    index = tmp_path / "index.txt"
    index.write_text(W.ui_folder(package) + "Ui_Loop.uasset\n", encoding="utf-8")
    monkeypatch.setattr(source, "archive_index", lambda settings, log: index)

    def old_reader(*args):
        raise ExtractorError("Unknown mode")

    monkeypatch.setattr(source, "_reader", old_reader)
    with pytest.raises(ExtractorError, match="reader|\u8b80\u53d6\u5668"):
        source._poses_of(None, package, tmp_path, tmp_path / "poses", lambda message: None)


def test_missing_acl_decoder_reports_that_poses_cannot_be_exported(tmp_path, monkeypatch):
    import pytest
    from polylode.extractor import wuwa_source as source
    from polylode.extractor.sources import ExtractorError

    package = "Client/Content/Aki/Character/Role/FemaleXL/Xin/R2T1XinMd10011/Model/R2T1XinMd10011.uasset"
    index = tmp_path / "index.txt"
    index.write_text(W.ui_folder(package) + "Ui_Loop.uasset\n", encoding="utf-8")
    monkeypatch.setattr(source, "archive_index", lambda settings, log: index)
    monkeypatch.setattr(source, "_reader", lambda settings, mode, folder, listing, log: (folder / "0.acl").touch())
    monkeypatch.setattr(source, "require", lambda *args: (_ for _ in ()).throw(ExtractorError("missing")))
    with pytest.raises(ExtractorError, match="ACL.*AnimeStudio"):
        source._poses_of(None, package, tmp_path, tmp_path / "poses", lambda message: None)


def test_a_bone_a_clip_scales_to_nothing_is_drawn_that_small_and_keeps_the_clips_turn():
    # a transition hides a part by scaling its bone to 0 (Xin's weapon in Ui_Chip_End): drawn at its rest size it
    # stood out whole; as a singular frame it had no rotation left and the export failed
    record = {"bones": ["Hips", "Prop"], "keyed": [True, True]}
    bones = [{"name": "Hips", "parent": -1}, {"name": "Prop", "parent": 0}]
    native = np.stack([np.eye(4), np.eye(4)])
    native[1, :3, 3] = [0, 0, .5]
    turn = Rotation.from_euler("z", 30, degrees=True).as_quat()
    values = np.array([[[0, 0, 0, 1, 0, 0, 0, 1, 1, 1], [*turn, 0, 0, 50, 0, 0, 0]]], dtype=np.float64)
    frames = W.frames(record, values, bones, native)
    assert np.isfinite(frames).all()
    size = np.linalg.svd(frames[0, 1, :3, :3].astype(np.float64), compute_uv=False)
    assert np.allclose(size, LEAST_SCALE, rtol=1e-3)                  # hidden as the game hides it, not singular
    shown = W.frames(record, np.concatenate([values[:, :, :7], np.ones((1, 2, 3))], axis=2), bones, native)
    assert np.allclose(frames[0, 1, :3, 3], shown[0, 1, :3, 3], atol=1e-6)    # where the clip puts it
    turned = frames[0, 1, :3, :3] / LEAST_SCALE
    assert np.allclose(turned, shown[0, 1, :3, :3], atol=1e-3)                # turned as the clip turns it
    # a bone the clip grows is drawn grown, its child carried with it
    big = values.copy()
    big[0, 1, 7:10] = 2.0
    grown = W.frames(record, big, bones, native)
    assert np.allclose(np.linalg.svd(grown[0, 1, :3, :3].astype(np.float64), compute_uv=False), 2.0, atol=1e-5)


def test_a_runtime_folder_without_dotnet_10_is_said_plainly_not_as_a_bare_winerror(tmp_path, monkeypatch):
    """The .NET field named the game's folder: the reader failed as FileNotFoundError [WinError 2] in a traceback."""
    import pytest
    from polylode.extractor import wuwa_source as WS
    from polylode.extractor.sources import ExtractorError
    from polylode.wuthering_waves import extract
    game = tmp_path / "Wuthering Waves"
    game.mkdir()
    monkeypatch.setattr(WS, "find_dotnet", lambda: None)
    with pytest.raises(ExtractorError, match="不是 .NET 10 執行環境.*安裝 .NET 10 Runtime"):
        WS.runtime_for(tmp_path / "wuwa.dll", game)
    runtime = tmp_path / "dotnet"
    (runtime / "shared" / "Microsoft.NETCore.App" / "10.0.1").mkdir(parents=True)
    (runtime / "dotnet.exe").write_bytes(b"MZ")
    monkeypatch.setattr(WS, "find_dotnet", lambda: runtime)
    with pytest.raises(ExtractorError, match="欄位清空就會用它"):              # the one found, offered
        WS.runtime_for(tmp_path / "wuwa.dll", game)
    WS.runtime_for(tmp_path / "wuwa.dll", runtime)                               # a runtime: nothing said
    WS.runtime_for(tmp_path / "wuwa.exe", game)                                  # an apphost runs on its own
    # a program that is not there at all: the reader's own error, not a traceback
    monkeypatch.setattr(extract, "derived_root", lambda: tmp_path, raising=False)
    with pytest.raises(extract.ReaderError, match="無法啟動讀取器"):
        extract.run(tmp_path / "missing.exe", runtime, tmp_path, tmp_path / "k.json", tmp_path / "out", "metadata")
