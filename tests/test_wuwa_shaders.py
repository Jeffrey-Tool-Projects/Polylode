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
