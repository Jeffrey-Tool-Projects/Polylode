"""Unreal 4.26 material preshaders: the small stack programs that fill a material's uniform buffer.

Each opcode is a byte (Unreal 4.26's ``EMaterialPreshaderOpcode``), some followed by operands; every value on
the stack is a float4.  The numbering was checked against what the programs of Wuthering Waves' materials do:
``p × 2π, Sin, p × 2π, Cos, AppendVector 1`` builds (sin, cos); ``v.xyz, v.xyz, v.xyz, Dot 4, Sqrt, Div``
normalises; ``1, p, Sub, −1, Mul`` is p − 1 — and every program of the character materials read must end
with exactly its last byte and one value.  An opcode not handled here stops the evaluation
(``PreshaderError``): a value guessed into a uniform buffer is a material drawn wrong with nothing saying so.
"""
from __future__ import annotations

import struct

import numpy as np

(NOP, ZERO, CONSTANT, SCALAR, VECTOR, ADD, SUB, MUL, DIV, FMOD, MIN, MAX, CLAMP, SIN, COS, TAN, ASIN, ACOS,
 ATAN, ATAN2, DOT, CROSS, SQRT, RCP, LENGTH, NORMALIZE, SATURATE, ABS, FLOOR, CEIL, ROUND, TRUNC, SIGN, FRAC,
 FRACTIONAL, SWIZZLE, APPEND, TEXTURE_SIZE, TEXEL_SIZE) = range(39)

_BINARY = {ADD: np.add, SUB: np.subtract, MUL: np.multiply, MIN: np.minimum, MAX: np.maximum,
           FMOD: np.fmod, ATAN2: np.arctan2}
_UNARY = {SIN: np.sin, COS: np.cos, TAN: np.tan, ASIN: lambda a: np.arcsin(np.clip(a, -1, 1)),
          ACOS: lambda a: np.arccos(np.clip(a, -1, 1)), ATAN: np.arctan, SQRT: lambda a: np.sqrt(np.maximum(a, 0)),
          RCP: lambda a: np.divide(1.0, a, out=np.zeros(4), where=a != 0), SATURATE: lambda a: np.clip(a, 0, 1),
          ABS: np.abs, FLOOR: np.floor, CEIL: np.ceil, ROUND: np.round, TRUNC: np.trunc, SIGN: np.sign,
          FRAC: lambda a: a - np.floor(a), FRACTIONAL: lambda a: a - np.trunc(a)}


class PreshaderError(ValueError):
    pass


def evaluate(code: bytes, scalars: list[float], vectors: list[tuple], texture_size=None) -> np.ndarray:
    """*code* run with the material's scalar and vector parameter values (by the code's indices): its float4.

    *texture_size* answers ``TextureSize``／``TexelSize``: the (width, height) of the texture the material
    references at an index (its ``ReferencedTextures``), as the instance binds it.
    """
    stack: list[np.ndarray] = []
    at = 0
    try:
        while at < len(code):
            op = code[at]
            at += 1
            if op == NOP:
                continue
            if op == ZERO:
                stack.append(np.zeros(4))
            elif op == CONSTANT:
                stack.append(np.array(struct.unpack_from("<4f", code, at)))
                at += 16
            elif op == SCALAR:
                (index,) = struct.unpack_from("<H", code, at)
                at += 2
                stack.append(np.full(4, float(scalars[index])))
            elif op == VECTOR:
                (index,) = struct.unpack_from("<H", code, at)
                at += 2
                stack.append(np.array(vectors[index], dtype=float))
            elif op in _BINARY:
                b, a = stack.pop(), stack.pop()
                stack.append(_BINARY[op](a, b))
            elif op == DIV:
                b, a = stack.pop(), stack.pop()
                stack.append(np.divide(a, b, out=np.zeros(4), where=b != 0))
            elif op == CLAMP:
                high, low, a = stack.pop(), stack.pop(), stack.pop()
                stack.append(np.minimum(np.maximum(a, low), high))
            elif op in _UNARY:
                stack.append(_UNARY[op](stack.pop()))
            elif op == DOT:
                count = code[at]                  # the operands' component count
                at += 1
                b, a = stack.pop(), stack.pop()
                stack.append(np.full(4, float(np.dot(a[:count], b[:count]))))
            elif op == CROSS:
                b, a = stack.pop(), stack.pop()
                stack.append(np.append(np.cross(a[:3], b[:3]), 0.0))
            elif op == LENGTH:
                stack.append(np.full(4, float(np.linalg.norm(stack.pop()))))
            elif op == NORMALIZE:
                a = stack.pop()
                size = np.linalg.norm(a)
                stack.append(a / size if size else a)
            elif op == SWIZZLE:
                count, picks = code[at], code[at + 1:at + 5]
                at += 5
                a = stack.pop()
                stack.append(np.array([a[picks[i]] if i < count and picks[i] < 4 else 0.0 for i in range(4)]))
            elif op in (TEXTURE_SIZE, TEXEL_SIZE):
                # a parameter (name, index, association; a name-less one is the referenced texture itself), then
                # the referenced texture's index: 2 + 4 + 1 + 4 bytes
                (reference,) = struct.unpack_from("<i", code, at + 7)
                at += 11
                if texture_size is None:
                    raise PreshaderError(f"a texture's size is asked for (referenced texture {reference}) and none given")
                width, height = texture_size(reference)
                stack.append(np.array([width, height, 0.0, 0.0]) if op == TEXTURE_SIZE
                             else np.array([1.0 / width, 1.0 / height, 0.0, 0.0]))
            elif op == APPEND:
                count = code[at]
                at += 1
                b, a = stack.pop(), stack.pop()
                stack.append(np.concatenate([a[:count], b])[:4])
            else:
                raise PreshaderError(f"preshader opcode {op} at {at - 1} is not read here")
    except IndexError:
        raise PreshaderError("a preshader read past its stack") from None
    if len(stack) != 1:
        raise PreshaderError(f"a preshader left {len(stack)} values")
    return stack[0]
