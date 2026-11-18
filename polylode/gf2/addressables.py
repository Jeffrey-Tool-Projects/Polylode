"""GF2's Addressables content catalog: where the game finds an asset it loads by path (IslandLife §7.1510).

GF2 3.0 loads its character models through Unity Addressables.  The install keeps the
content catalog of the downloaded bundles in ``LocalCache/Data/AssetBundles_Windows`` as
``catalog_main_<n>.bin``, where *n* is the last field of ``CacheResVersion`` in
``LocalCache/Data/GameConfig.cfg`` (3.0.5388.0.29003 → ``catalog_main_29003.bin``).  It is
Addressables' binary catalog (``BinaryStorageBuffer``, magic ``0x0DE38942``):

* the keys are the asset paths, hashed: ``BitConverter.ToString(MD5(UTF-16LE(path)))``
  (:func:`address`), 199 599 of the 262 344 keys of 3.0 (the rest are the bundles' own names);
* a key's location gives the provider (``BundledAssetProvider``), the asset's type, the
  internal id — the asset's GUID, which is the bundle's container key of that asset — and
  one dependency, the bundle's location, whose internal id is the bundle file's name.

So a path such as a ``RoleMeshRes`` ``MeshResPath`` (:mod:`.role_mesh`) is resolved the way
the game resolves it: path → key → GUID and bundle file → the bundle's container entry.
Nothing is matched by asset name.  The game's C# is not readable here (``global-metadata.dat``
is encrypted, ``GameAssembly.dll`` is packed), so the rule is read from the catalog itself: a
``MeshResPath`` hashes to a key whose location is a bundled ``Mesh``, and its GUID is the ``Mesh``
container of one of the bundles the location names (checked on ``c_AsteriaSSR01_slg_skin_model``).
"""
from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = 0x0DE38942
#: the catalog layouts this reader was checked against (GF2 3.0's is version 1)
VERSIONS = (1,)
NONE = 0xFFFFFFFF
UNICODE = 0x80000000
DYNAMIC = 0x40000000
CLEAR = 0x3FFFFFFF
STRING_TYPE = "System.String"
#: the folder holding the downloaded bundles and their catalog, under the install
BUNDLES = "GF2_Exilium_Data/LocalCache/Data/AssetBundles_Windows"
CONFIG = "GF2_Exilium_Data/LocalCache/Data/GameConfig.cfg"


class CatalogError(ValueError):
    """The catalog is absent, of an unknown layout, or does not answer a lookup."""


def address(path: str) -> str:
    """The catalog key of an asset *path*: the MD5 of its UTF-16LE bytes, as ``BitConverter.ToString`` writes it."""
    return "-".join(f"{b:02X}" for b in hashlib.md5(path.encode("utf-16-le")).digest())


def catalog_path(install: Path) -> Path:
    """The catalog of the install's current resources: ``catalog_main_<n>.bin`` with *n* the last field of
    ``CacheResVersion`` (an update downloads a new catalog; an old one left beside it is not this version's)."""
    config = Path(install) / CONFIG
    if not config.is_file():
        raise CatalogError(f"{config}: no GameConfig.cfg, the resource version is unknown")
    version = None
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CacheResVersion:"):
            version = line.split(":", 1)[1].strip()
    if not version or not re.fullmatch(r"[\d.]+", version):
        raise CatalogError(f"{config}: no CacheResVersion")
    path = Path(install) / BUNDLES / f"catalog_main_{version.rsplit('.', 1)[-1]}.bin"
    if not path.is_file():
        raise CatalogError(f"{path}: the catalog of CacheResVersion {version} is not in the install")
    return path


@dataclass
class Location:
    """One catalog location (Addressables' ``ResourceLocation``)."""
    primary_key: str | None
    internal_id: str | None
    provider: str | None
    type: str | None
    dependencies: list = field(default_factory=list)


class ContentCatalog:
    """An Addressables binary content catalog, read whole (GF2 3.0's: 53 MB, 262 344 keys)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        if len(self.data) < 32:
            raise CatalogError(f"{self.path}: truncated catalog")
        magic, version, keys = struct.unpack_from("<3I", self.data, 0)
        if magic != MAGIC:
            raise CatalogError(f"{self.path}: not an Addressables binary catalog (magic {magic:#x})")
        if version not in VERSIONS:
            raise CatalogError(f"{self.path}: catalog layout version {version} has not been read")
        self.version = version
        self._keys = None
        self._keys_at = keys
        self.other_keys = 0

    # -- the storage buffer ---------------------------------------------------------------------------------------
    def _u32(self, at: int) -> int:
        if not 0 <= at <= len(self.data) - 4:
            raise CatalogError(f"{self.path}: offset {at:#x} outside the catalog")
        return struct.unpack_from("<I", self.data, at)[0]

    def _array(self, at: int) -> tuple:
        """A value array of uint32: its byte size is the uint32 before it."""
        size = self._u32(at - 4)
        if size % 4 or at + size > len(self.data):
            raise CatalogError(f"{self.path}: bad array at {at:#x}")
        return struct.unpack_from(f"<{size // 4}I", self.data, at)

    def _plain(self, at: int) -> str | None:
        if at == NONE:
            return None
        unicode = bool(at & UNICODE)
        at &= CLEAR
        size = self._u32(at - 4)
        raw = self.data[at:at + size]
        if len(raw) != size:
            raise CatalogError(f"{self.path}: string at {at:#x} runs past the end")
        return raw.decode("utf-16-le" if unicode else "ascii")

    def string(self, at: int, separator: str) -> str | None:
        """A string id: plain (ASCII, or UTF-16 with the high flag) or a chain of parts joined by *separator*."""
        if at == NONE:
            return None
        if not at & DYNAMIC:
            return self._plain(at)
        parts, seen = [], set()
        while at != NONE:
            if at in seen:
                raise CatalogError(f"{self.path}: dynamic string loops at {at:#x}")
            seen.add(at)
            part, at = struct.unpack_from("<2I", self.data, at & CLEAR)
            parts.append(self._plain(part))
        return separator.join(parts)

    def type_name(self, at: int) -> str | None:
        """A serialized type (assembly name id, class name id): the class name."""
        if at == NONE:
            return None
        _assembly, name = struct.unpack_from("<2I", self.data, at)
        return self.string(name, ".")

    # -- keys and locations ---------------------------------------------------------------------------------------
    def keys(self) -> dict[str, int]:
        """``{string key: location set offset}``; keys of another type are counted in :attr:`other_keys`."""
        if self._keys is None:
            table = self._array(self._keys_at)
            keys, types = {}, {}
            for name_at, locations in zip(table[0::2], table[1::2]):
                type_at, object_at = struct.unpack_from("<2I", self.data, name_at)
                if type_at not in types:
                    types[type_at] = self.type_name(type_at)
                if types[type_at] != STRING_TYPE:
                    self.other_keys += 1
                    continue
                key = self.string(self._u32(object_at), "/")
                if key in keys and keys[key] != locations:
                    raise CatalogError(f"{self.path}: key {key} listed twice")
                keys[key] = locations
            self._keys = keys
        return self._keys

    def location(self, at: int, depth: int = 0) -> Location:
        primary, internal, provider, dependencies, _hash, _extra, kind = struct.unpack_from("<7I", self.data, at)
        found = Location(self.string(primary, "/"), self.string(internal, "/"), self.string(provider, "."),
                         self.type_name(kind))
        if dependencies != NONE:
            if depth > 4:
                raise CatalogError(f"{self.path}: dependency chain deeper than 4 at {at:#x}")
            found.dependencies = [self.location(d, depth + 1) for d in self._array(dependencies)]
        return found

    def locate(self, key: str) -> list[Location]:
        """The locations of *key*; an empty list when the catalog has no such key."""
        at = self.keys().get(key)
        return [] if at is None else [self.location(location) for location in self._array(at)]


_OPEN: dict = {}


def opened(install: Path) -> ContentCatalog:
    """The install's current catalog (:func:`catalog_path`), read once per process for a given file state."""
    path = catalog_path(install)
    stat = path.stat()
    stamp = (str(path), stat.st_size, stat.st_mtime_ns)
    if stamp not in _OPEN:
        _OPEN.clear()
        _OPEN[stamp] = ContentCatalog(path)
    return _OPEN[stamp]
