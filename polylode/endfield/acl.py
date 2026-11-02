"""Decode an ACL-compressed track buffer with the exporter's own library.

The game stores a clip's curves as Animation Compression Library
``compressed_tracks`` buffers (tag ``0xac11ac11``): one of transform tracks,
one of scalar tracks, one for root motion.  The exporter ships native
decoders for several builds of that library; the one that reads this game's
buffers is ``acldb_zzz`` (measured: 273 tracks × 661 samples decoded to unit
quaternions and translations within 0.01 mm of the prefab's rest pose on the
bones that do not move, §7.564).  It is the user's file, loaded from beside
their exporter — nothing of it is bundled.

Import side only: the run time reads the decoded clips (:mod:`.clips`).
"""
from __future__ import annotations

import ctypes
import struct
from pathlib import Path

import numpy as np

#: The library and where it sits relative to the exporter's executable.
LIBRARY = Path("x64") / "acldb_zzz.dll"
TAG = 0xAC11AC11
#: Floats per sample of a transform track (rotation xyzw, translation xyz,
#: scale xyz) and of a scalar track.
TRANSFORM, SCALAR = 10, 1
_HEADER = "<IIIHBBIIf"


class AclError(RuntimeError):
    """The buffer is not what the decoder was measured against, or it failed."""


class _Clip(ctypes.Structure):
    _fields_ = [("values", ctypes.c_void_p), ("values_count", ctypes.c_int),
                ("times", ctypes.c_void_p), ("times_count", ctypes.c_int)]


def header(buffer: bytes) -> tuple[int, int, float]:
    """``(tracks, samples, sample rate)`` of a ``compressed_tracks`` buffer."""
    if len(buffer) < struct.calcsize(_HEADER):
        raise AclError(f"{len(buffer)} bytes is too short for an ACL buffer")
    (_size, _hash, tag, _version, _algorithm, _kind,
     tracks, samples, rate) = struct.unpack_from(_HEADER, buffer, 0)
    if tag != TAG:
        raise AclError(f"tag {tag:08x} is not an ACL compressed_tracks buffer")
    return int(tracks), int(samples), float(rate)


class Decoder:
    """The exporter's ACL library, loaded once."""

    def __init__(self, exporter: Path):
        path = Path(exporter).parent / LIBRARY
        if not path.is_file():
            raise AclError(f"ACL library not found beside the exporter: {path}")
        self._dll = ctypes.CDLL(str(path))

    def decode(self, buffer: bytes, width: int) -> np.ndarray:
        """Every sample of every track: ``(samples, tracks, width)`` float32.

        *width* is :data:`TRANSFORM` or :data:`SCALAR`; transform tracks are in
        Unity space, local to each track's parent.  Raises :class:`AclError`
        when the library returns a different amount than the header and
        *width* declare — a decoder that disagrees with the buffer about its
        own size has not decoded it.
        """
        tracks, samples, _rate = header(buffer)
        # the library reads 16-byte aligned memory
        holder = ctypes.create_string_buffer(len(buffer) + 32)
        aligned = (ctypes.addressof(holder) + 15) // 16 * 16
        ctypes.memmove(aligned, buffer, len(buffer))
        out = _Clip()
        self._dll.DecompressTracks(ctypes.c_void_p(aligned), ctypes.c_void_p(0),
                                   ctypes.c_void_p(0), ctypes.byref(out))
        try:
            want = samples * tracks * width
            if out.values_count != want or out.times_count != samples:
                raise AclError(f"decoded {out.values_count} values and "
                               f"{out.times_count} times, header says {want} "
                               f"and {samples}")
            flat = np.ctypeslib.as_array(
                ctypes.cast(out.values, ctypes.POINTER(ctypes.c_float)),
                shape=(want,))
            return flat.reshape(samples, tracks, width).copy()
        finally:
            self._dll.Dispose(ctypes.byref(out))
