from polylode.wuthering_waves.extract import install_stamp


def test_the_stamp_counts_every_archive_the_reader_mounts_the_hd_pack_too(tmp_path):
    content = tmp_path / "Client" / "Content"
    (content / "Paks").mkdir(parents=True)
    (content / "HD").mkdir()
    (content / "Aki" / "Cursor").mkdir(parents=True)
    (content / "Paks" / "pakchunk0-WindowsNoEditor.pak").write_bytes(b"a" * 10)
    (content / "Paks" / "pakchunk0-WindowsNoEditor.sig").write_bytes(b"s")
    (content / "Aki" / "Cursor" / "CursorHi.png").write_bytes(b"p" * 100)
    before = install_stamp(tmp_path)
    assert before[:2] == [2, 11]                      # loose files beside the archives are not archives
    (content / "HD" / "pakchunk101-HD-WindowsNoEditor.pak").write_bytes(b"h" * 5)
    after = install_stamp(tmp_path)
    assert after[:2] == [3, 16] and after != before   # an update of the HD pack alone is an update
