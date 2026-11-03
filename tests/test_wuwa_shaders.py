"""Wuthering Waves' compiled shaders read and run (``wuthering_waves/shaders``): the preshader programs and the
assembly interpreter, on programs written as the game's are."""
import struct

import numpy as np
import pytest

from polylode.wuthering_waves.shaders import preshader, vm


def f4(*values):
    return bytes([preshader.CONSTANT]) + struct.pack("<4f", *values)


def param(index):
    return bytes([preshader.SCALAR]) + struct.pack("<H", index)


def test_preshaders_as_the_games_programs_are_written():
    scalars = [0.0] * 30
    scalars[17], scalars[19], scalars[12] = 0.5333, 0.0, 0.25
    # Height × 0.01, (1 − Plane) × −1 — read from the fox tail's material
    assert preshader.evaluate(param(17) + f4(.01, .01, .01, .01) + bytes([preshader.MUL]), scalars, [])[0] == pytest.approx(0.005333)
    one_minus = f4(1, 1, 1, 1) + param(19) + bytes([preshader.SUB]) + f4(-1, -1, -1, -1) + bytes([preshader.MUL])
    assert preshader.evaluate(one_minus, scalars, [])[0] == pytest.approx(-1.0)
    # p × 2π → (sin, cos): Sin, then Cos, joined by AppendVector 1
    turn = param(12) + f4(*[2 * np.pi] * 4) + bytes([preshader.MUL])
    sincos = turn + bytes([preshader.SIN]) + turn + bytes([preshader.COS, preshader.APPEND, 1])
    assert preshader.evaluate(sincos, scalars, [])[:2] == pytest.approx([1.0, 0.0], abs=1e-6)
    # v.xyz normalised: three copies, Dot 4, Sqrt, Div
    xyz = bytes([preshader.VECTOR]) + struct.pack("<H", 0) + bytes([preshader.SWIZZLE, 3, 0, 1, 2, 255])
    unit = preshader.evaluate(xyz * 3 + bytes([preshader.DOT, 4, preshader.SQRT, preshader.DIV]), scalars, [(3, 0, 4, 9)])
    assert unit[:3] == pytest.approx([0.6, 0.0, 0.8]) and unit[3] == 0.0
    with pytest.raises(preshader.PreshaderError):
        preshader.evaluate(bytes([200]), scalars, [])                    # not read here: refused, not guessed
    with pytest.raises(preshader.PreshaderError):
        preshader.evaluate(bytes([preshader.ADD]), scalars, [])          # past its stack


def run(asm, inputs, buffers=None, textures=None):
    program = vm.parse("ps_5_0\n" + asm)
    n = len(next(iter(inputs.values())))
    machine = vm.Machine(program, n, inputs, buffers or {}, textures or {})
    discarded = machine.run()
    return machine, discarded


def test_the_interpreter_runs_branches_loops_bits_and_discards_per_lane():
    v0 = np.array([[0.2, 0, 0, 0], [0.7, 0, 0, 0], [0.5, 0, 0, 0]], np.float32)
    asm = """
mov r0.x, l(0)
ge r1.x, v0.x, l(0.400000)
and r1.x, r1.x, l(0x3f800000)
if_nz r1.x
  mul_sat r2.x, v0.x, l(2.000000)
else
  mov r2.x, l(-1.000000)
endif
mov r3.x, l(0)
loop
  ige r3.y, r3.x, l(3)
  breakc_nz r3.y
  iadd r3.x, r3.x, l(1)
  add r0.x, r0.x, v0.x
endloop
udiv null, r4.x, l(7), l(5)
lt r5.x, r2.x, l(0.000000)
discard_nz r5.x
ret
"""
    machine, discarded = run(asm, {0: v0})
    r = lambda k: machine.registers[k].view(np.float32)
    assert r(1)[:, 0].tolist() == [0.0, 1.0, 1.0]                     # the comparison's bits and 1.0's
    assert discarded.tolist() == [True, False, False]                  # the else lane went negative: discarded
    assert r(2)[1:, 0].tolist() == [1.0, 1.0]                          # saturated
    assert r(0)[1:, 0] == pytest.approx([2.1, 1.5])                    # three passes of the loop
    assert machine.registers[4][1, 0] == 2                             # 7 mod 5


def test_the_interpreter_samples_with_the_resource_swizzle_and_reads_per_lane_buffers():
    image = np.zeros((2, 2, 4), np.float32)
    image[..., 0], image[..., 1] = 0.25, 0.75                          # R and G everywhere
    v0 = np.array([[0.5, 0.5, 0, 0]] * 2, np.float32)
    asm = """
sample_l_indexable(texture2d)(float,float,float,float) r0.z, v0.xyxx, t0.yzxw, s0, l(0)
mul r1.x, cb0[1].x, l(2.000000)
"""
    buffer = np.zeros((2, 2, 4))
    buffer[:, 1, 0] = [3.0, 5.0]                                       # one value per lane
    machine, _ = run(asm, {0: v0}, {0: buffer}, {0: image})
    assert machine.registers[0].view(np.float32)[:, 2] == pytest.approx([0.25, 0.25])   # .z of .yzxw is R
    assert machine.registers[1].view(np.float32)[:, 0].tolist() == [6.0, 10.0]
    with pytest.raises(vm.ShaderError):
        run("bogus_op r0.x, v0.x\n", {0: v0})                         # an instruction not run here: refused


def test_overlapping_uvs_with_another_coverage_get_a_map_of_their_own():
    from polylode.wuthering_waves import shader_bake as SB
    square = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]], np.float64)
    uv = np.concatenate([square, square, square])                       # three islands on the same texels
    rows, columns, which, _weights = SB._texels(uv, (8, 8), unique=False)
    covered = np.where(which >= 4, 1.0, 0.0)                           # the third island differs: opaque cards
    maps, layer = SB._layers(rows, columns, which, covered, len(uv), (8, 8))
    assert layer.tolist() == [0, 0, 0, 0, 1, 1]                        # the first two (mirrored) stay together
    assert np.nanmax(maps[0]) == 0.0 and np.nanmin(maps[1]) == 1.0
    same = np.zeros_like(covered)
    maps, layer = SB._layers(rows, columns, which, same, len(uv), (8, 8))
    assert len(maps) == 1 and not layer.any()                          # agreeing overlaps share one map


def test_overlap_layers_move_to_parts_of_their_own():
    from polylode.wuthering_waves import shader_bake as SB
    document = {"parts": [{"name": "Tail", "materials": ["Fur"], "shapes": ["Wag"]},
                          {"name": "Body", "materials": ["Skin"], "shapes": []}]}
    co = np.arange(18, dtype=np.float32).reshape(6, 3)
    arrays = {"p0_co": co, "p0_no": co, "p0_uv0": co[:, :2], "p0_tri": np.array([[0, 1, 2], [3, 4, 5], [0, 2, 3]]),
              "p0_slot": np.zeros(3, np.int64), "p0_s0_idx": np.array([1, 4]), "p0_s0_delta": np.ones((2, 3)),
              "p1_co": co[:3], "p1_no": co[:3], "p1_uv0": co[:3, :2], "p1_tri": np.array([[0, 1, 2]]),
              "p1_slot": np.zeros(1, np.int64)}
    SB._move_layers(document, arrays, "Fur", np.array([0, 1, 0]))
    assert [p["name"] for p in document["parts"]] == ["Tail", "Body", "Tail_Overlap1"]
    assert document["parts"][2]["materials"] == ["Fur_Overlap1"] and document["parts"][0]["materials"] == ["Fur"]
    assert arrays["p0_tri"].tolist() == [[0, 1, 2], [0, 2, 3]] and len(arrays["p0_co"]) == 4
    assert arrays["p2_co"].tolist() == co[3:].tolist() and arrays["p2_tri"].tolist() == [[0, 1, 2]]
    assert arrays["p2_s0_idx"].tolist() == [1] and document["parts"][2]["shapes"] == ["Wag"]   # the shape follows
    assert arrays["p1_tri"].tolist() == [[0, 1, 2]]                    # the other part untouched


def test_overlapping_samples_run_the_shader_only_where_they_can_differ():
    from polylode.wuthering_waves import shader_bake as SB
    square = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]], np.float64)
    rows, columns, which, _weights = SB._texels(np.concatenate([square] * 3), (16, 16), unique=False)
    truth = np.where(which >= 4, 1.0, 0.0)                             # the third island: opaque cards
    ran = []

    def run(indices):
        ran.extend(indices.tolist())
        return truth[indices]
    covered = SB._covered(run, rows, columns, which, 6, 16)
    assert np.array_equal(covered, truth)
    ran = np.array(ran)
    assert len(set(ran.tolist())) == len(ran)                          # no sample run twice
    assert (which[ran] >= 4).sum() == (which >= 4).sum()               # the differing island run in full
    assert (which[ran] // 2 == 1).sum() <= 2 * SB.PROBES               # the mirrored one only probed


def test_min_and_max_give_the_other_operand_for_a_nan_as_direct3d_does():
    v0 = np.array([[np.nan, 0.25, 0, 0]], np.float32)
    machine, _ = run("max r0.x, v0.x, v0.y\nmin r0.y, v0.x, v0.y\nmax r0.z, v0.y, v0.x\n", {0: v0})
    assert machine.registers[0].view(np.float32)[0, :3].tolist() == [0.25, 0.25, 0.25]


def test_float_to_integer_clamps_to_its_type_as_direct3d_does():
    v0 = np.array([[np.nan, -3.5, 3e9, 0]], np.float32)
    machine, _ = run("ftoi r0.xyz, v0.xyzx\nftou r1.xyz, v0.xyzx\n", {0: v0})
    assert machine.registers[0].view(np.int32)[0, :3].tolist() == [0, -3, 2 ** 31 - 1]
    assert machine.registers[1][0, :3].tolist() == [0, 0, 3000000000]


def test_the_interpreter_keeps_what_a_program_writes_to_its_outputs():
    v0 = np.array([[0.25, 0.5, 0, 0]], np.float32)
    machine, _ = run("mul o0.xy, v0.xyxx, l(2.000000, 4.000000, 0, 0)\nmov o1.z, l(3.000000)\n", {0: v0})
    assert machine.outputs[0].view(np.float32)[0, :2].tolist() == [0.5, 2.0]
    assert machine.outputs[1].view(np.float32)[0, 2] == 3.0 and 0 not in machine.registers


def test_an_opcode_with_brackets_holding_a_space_and_a_vertex_stage_header_are_read():
    program = vm.parse("vs_5_0\nld_structured_indexable(structured_buffer, stride=16)(mixed,mixed,mixed,mixed) "
                       "r7.xyzw, r6.y, l(0), t0.xyzw\nmov o1.xyzw, v4.xyzw\n")
    assert [i.op for i in program] == ["ld_structured_indexable", "mov"]
    assert [o.kind for o in program[0].operands] == ["r", "r", "l", "t"]


def test_scalars_sharing_a_buffer_row_each_keep_their_own_word():
    from polylode.endfield import shader_code as SC, vfx_bake as VB

    class Variant:
        params = None

        def registers(self, stage="ps"):
            return {3: ("UnityPerMaterial", [("_A", 80, 1, 1, 0), ("_B", 84, 1, 1, 0), ("_C", 96, 1, 4, 0)])}, {}, {}
    asm = "dcl_constantbuffer CB3[7], immediateIndexed\nmov r0.xy, cb3[5].xyxx\nmov r1.xyzw, cb3[6].xyzw\n"
    record = {"floats": {"_A": 0.25, "_B": 1.0}, "colors": {"_C": [1, 2, 3, 4]}, "maps": {}}
    buffers = VB._buffers(Variant(), record, {}, asm, "ps")
    assert buffers[3][5].tolist() == [0.25, 1.0, 0.0, 0.0] and buffers[3][6].tolist() == [1, 2, 3, 4]
    del SC
