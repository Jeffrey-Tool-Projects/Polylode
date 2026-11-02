"""Shader program wrapper with a uniform-location cache and a uniform-value cache."""
from __future__ import annotations

import numpy as np
from OpenGL import GL


class ShaderError(RuntimeError):
    pass


class Program:
    def __init__(self, vs: str, fs: str, name: str = "program"):
        self.name = name
        self.id = self._link(vs, fs)
        self._reset_uniform_cache()

    def _reset_uniform_cache(self) -> None:
        self._loc: dict[str, int] = {}
        # A uniform keeps its value in the program until it is set again, so a
        # set to the value it already holds is a wasted call -- and the
        # renderer re-sets every material field per draw: ~7,000 ``glUniform*``
        # a frame for three characters, through PyOpenGL, most to the value
        # already there (§7.551).  Only scalars and vectors: a matrix changes
        # per draw and comparing it costs what the call does.
        self._val: dict[int, tuple] = {}

    @classmethod
    def from_linked(cls, program_id: int, name: str = 'program') -> Program:
        """Take ownership of an already linked program in the current context."""
        if type(program_id) is not int or program_id <= 0 or not GL.glIsProgram(program_id):
            raise ValueError('cannot adopt a missing GL program')
        if not GL.glGetProgramiv(program_id, GL.GL_LINK_STATUS):
            raise ShaderError('cannot adopt an unlinked GL program')
        result = cls.__new__(cls)
        result.name, result.id = name, program_id
        result._reset_uniform_cache()
        return result

    def _compile(self, src: str, kind) -> int:
        sid = GL.glCreateShader(kind)
        GL.glShaderSource(sid, src)
        GL.glCompileShader(sid)
        if not GL.glGetShaderiv(sid, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(sid)
            log = log.decode() if isinstance(log, bytes) else str(log)
            kindname = "vertex" if kind == GL.GL_VERTEX_SHADER else "fragment"
            numbered = "\n".join(f"{i + 1:4d}| {l}" for i, l in enumerate(src.splitlines()))
            raise ShaderError(f"[{self.name}] {kindname} shader failed:\n{log}\n{numbered}")
        return sid

    def _link(self, vs: str, fs: str) -> int:
        v = self._compile(vs, GL.GL_VERTEX_SHADER)
        f = self._compile(fs, GL.GL_FRAGMENT_SHADER)
        pid = GL.glCreateProgram()
        GL.glAttachShader(pid, v)
        GL.glAttachShader(pid, f)
        GL.glLinkProgram(pid)
        if not GL.glGetProgramiv(pid, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(pid)
            log = log.decode() if isinstance(log, bytes) else str(log)
            raise ShaderError(f"[{self.name}] link failed:\n{log}")
        GL.glDeleteShader(v)
        GL.glDeleteShader(f)
        return pid

    # -- use ---------------------------------------------------------------
    def use(self) -> "Program":
        GL.glUseProgram(self.id)
        return self

    def loc(self, name: str) -> int:
        if name not in self._loc:
            self._loc[name] = GL.glGetUniformLocation(self.id, name)
        return self._loc[name]

    # -- uniform setters ---------------------------------------------------
    def _changed(self, loc: int, value: tuple) -> bool:
        """Whether *loc* needs *value* sent (and remember it as sent)."""
        if loc < 0 or self._val.get(loc) == value:
            return False
        self._val[loc] = value
        return True

    def set_int(self, name: str, v: int):
        l = self.loc(name)
        val = (int(v),)
        if self._changed(l, val):
            GL.glUniform1i(l, *val)

    def set_float(self, name: str, v: float):
        l = self.loc(name)
        val = (float(v),)
        if self._changed(l, val):
            GL.glUniform1f(l, *val)

    def set_vec2(self, name: str, v):
        l = self.loc(name)
        val = (float(v[0]), float(v[1]))
        if self._changed(l, val):
            GL.glUniform2f(l, *val)

    def set_vec3(self, name: str, v):
        l = self.loc(name)
        val = (float(v[0]), float(v[1]), float(v[2]))
        if self._changed(l, val):
            GL.glUniform3f(l, *val)

    def set_vec4(self, name: str, v):
        l = self.loc(name)
        val = (float(v[0]), float(v[1]), float(v[2]), float(v[3]) if len(v) > 3 else 1.0)
        if self._changed(l, val):
            GL.glUniform4f(l, *val)

    def set_mat4(self, name: str, m: np.ndarray):
        l = self.loc(name)
        if l >= 0:
            GL.glUniformMatrix4fv(l, 1, GL.GL_TRUE,
                                  np.ascontiguousarray(m, dtype=np.float32))

    def set_mat4_array(self, name: str, mats: np.ndarray):
        l = self.loc(name)
        if l >= 0:
            arr = np.ascontiguousarray(mats, dtype=np.float32)
            GL.glUniformMatrix4fv(l, len(arr), GL.GL_TRUE, arr)

    def release(self):
        if self.id:
            GL.glDeleteProgram(self.id)
            self.id = 0
        self._val.clear()
