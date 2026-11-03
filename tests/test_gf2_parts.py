"""GF2's swappable outfit parts read from the game's tables (``gf2/parts``), and its combat variants
(``gf2/presentation.combat_variants``): which of an outfit's versions a model is shown in."""
import struct

from polylode.gf2 import parts, presentation


def varint(n):
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def field(number, value):
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    if isinstance(value, str):
        value = value.encode()
    return varint(number << 3 | 2) + varint(len(value)) + value


def entry(**fields):
    return b"".join(field(int(k[1:]), v) for k, v in fields.items())


def table(path, rows):
    """A table as the game writes one: a 4-byte length, the index (field 3: id, (offset, length)), the rows."""
    body, index = b"", b""
    for ident, entries in rows.items():
        row = b"".join(field(1, e) for e in entries)
        index += field(3, field(1, ident) + field(2, field(1, len(body)) + field(2, len(row))))
        body += row
    head = field(1, 10) + field(2, 1) + index
    path.write_bytes(struct.pack("<I", len(head)) + head + body)


def tables(tmp_path):
    folder = tmp_path / parts.TABLES
    folder.mkdir(parents=True)
    table(folder / "ModelConfigData.bytes", {10550: [entry(f1=105501, f2="NikketaSSR01"),
                                                     entry(f1=105502, f2="NikketaSSR0102")]})
    table(folder / "PartsGroupData.bytes", {55020: [entry(f1=550201, f2=1, f5=1105502),
                                                    entry(f1=550202, f2=2, f5=1105502)]})
    table(folder / "PartsListData.bytes", {
        5502010: [entry(f1=55020101, f2=1, f4=1, f7=550201), entry(f1=55020102, f2=2, f5=2210, f7=550201)],
        5502020: [entry(f1=55020201, f2=1, f4=1, f7=550202), entry(f1=55020202, f2=2, f5=2212, f7=550202)]})
    table(folder / "PartsResourceData.bytes", {
        55020101: [entry(f1=1, f2="P1_body"), entry(f1=2, f2="P1_body1")], 55020102: [entry(f1=3, f2="P2_body")],
        55020201: [entry(f1=4, f2="P1_hip2")], 55020202: [entry(f1=5, f2="P2_hip2")]})


def test_the_parts_worn_are_each_slots_free_choice_of_the_outfit_the_model_is(tmp_path):
    tables(tmp_path)
    picked = parts.choices(tmp_path, "c_NikketaSSR0102_slg_skin_48e81872d7")
    assert picked == {"outfit": 1105502, "worn": ["P1_body", "P1_body1", "P1_hip2"], "spare": ["P2_body", "P2_hip2"]}
    assert parts.choices(tmp_path, "c_NikketaSSR01_slg_skin") is None      # its outfit has no slots
    assert parts.choices(tmp_path / "elsewhere", "c_NikketaSSR0102_slg_skin") is None   # no tables: nothing said
    names = ["c_X_slg_P1_body_lod0", "c_X_slg_P1_body1_lod0", "c_X_slg_P2_body_lod0_Fight",
             "c_X_slg_P1_hip2_trans_lod0", "c_X_slg_P2_hip2_lod0", "c_X_slg_body_lod0"]
    hidden = parts.hidden([{"name": n} for n in names], picked)
    assert [n for n in names if hidden[n]] == ["c_X_slg_P2_body_lod0_Fight", "c_X_slg_P2_hip2_lod0"]


def test_an_outfits_dormitory_versions_are_hidden_and_the_dormitory_model_keeps_them():
    def hidden(names, model="c_X_slg_skin_0123456789"):
        flags = presentation.combat_variants([{"name": n} for n in names], model)
        return [n for n in names if flags[n]]

    assert hidden(["a_lod0", "a_lod0_Fight", "a_lod0_Dorm", "c_lod1_Fight", "c_lod1_Dorm", "e_lod0_General"])         == ["a_lod0_Dorm", "c_lod1_Dorm"]
    # Springfield SSR0101: combat heels on one renderer, dormitory sandals on another — worn through each other
    assert hidden(["s_cloth2_lod0", "s_cloth2_lod0_Fight", "s_body_lod0", "s_body_lod0_Dorm"]) == ["s_body_lod0_Dorm"]
    # Zhaohui SSR0101 spells the combat suffix _fight, Nagant SR0101 _Flight
    assert hidden(["z_cloth_lod0", "z_cloth_lod0_fight", "z_cloth_lod0_Dorm"]) == ["z_cloth_lod0_Dorm"]
    assert hidden(["n_boby_lod0", "n_boby_lod0_Flight", "n_boby_lod0_Dorm"], "c_NagantSR0101_skin") == ["n_boby_lod0_Dorm"]
    # the dormitory model wears them
    assert hidden(["d_body_lod0", "d_body_lod0_Dorm"], "c_PhaetusaDorm_slg_skin_41237cd1b9") == []
    assert hidden(["d_body_lod0", "d_body_lod0_Dorm"], "c_VepleyDorm_skin") == []
