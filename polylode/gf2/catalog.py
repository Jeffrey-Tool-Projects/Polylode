"""Locate serialized files by CAB identity, with lazy dependency loading."""
from __future__ import annotations

import json
from pathlib import Path

import UnityPy
from UnityPy.enums import FileType
from UnityPy.files import SerializedFile
from UnityPy.helpers.ImportHelper import check_file_type

from .bundles import configure_unity, members


def cab_key(name):
    return name.replace('\\', '/').rsplit('/', 1)[-1].lower()


class Catalog:
    def __init__(self, install: Path, index: Path):
        self.install = install
        self.unity_version = configure_unity(install)
        self.rows = {}
        for line in index.read_text(encoding='utf8').splitlines():
            row = json.loads(line)
            self.rows[row['path']] = row
        self.cabs = {}
        self.entries = []
        directory_index = index.with_name('cab_index.jsonl')
        if directory_index.exists():
            for line in directory_index.read_text(encoding='utf8').splitlines():
                row = json.loads(line)
                for member in row['members']:
                    for cab in member['cabs']:
                        self.cabs[cab_key(cab['name'])] = (row['path'], member['offset'])
        for row in self.rows.values():
            if row.get('error'):
                continue
            for member in row['members']:
                for cab in member['cabs']:
                    self.cabs[cab_key(cab['name'])] = (row['path'], member['offset'])
                for entry in member['containers']:
                    if entry['type'] == 'GameObject':
                        self.entries.append(dict(entry, path=row['path'], offset=member['offset']))

    def environment(self):
        return Assets(self)


class Assets(UnityPy.Environment):
    def __init__(self, catalog: Catalog):
        super().__init__(path='')
        self.catalog = catalog
        self.loaded = set()

    def find_file(self, name, is_dependency=True):
        key = cab_key(name)
        existing = self.get_cab(key)
        if existing is not None:
            return existing
        location = self.catalog.cabs.get(key)
        if location is None:
            if key in ('unity default resources', 'unity_builtin_extra'):
                return self.load_native(key, is_dependency)
            raise FileNotFoundError(f'GF2 serialized file absent from local index: {name}')
        self.load_member(*location)
        result = self.get_cab(key)
        if result is None:
            raise FileNotFoundError(f'indexed member does not contain {name}')
        return result

    def load_native(self, name: str, is_dependency: bool = True) -> SerializedFile:
        """Read an installed built-in serialized file; reject undecoded NP data.

        Native files bypass the bundle decoder and retain their raw installation
        path in source provenance. An opaque resource stream is never substituted
        for the serialized objects referenced by a prefab.
        """
        if name not in ('unity default resources', 'unity_builtin_extra'):
            raise ValueError(f'unsupported GF2 native resource: {name}')
        relative = f'GF2_Exilium_Data/Resources/{name}'
        path = self.catalog.install / relative
        raw = path.read_bytes()
        if raw.startswith(b'NP'):
            raise ValueError(f'GF2 native resource {relative}: NP format requires an explicit decoder')
        kind, _ = check_file_type(raw)
        if kind != FileType.AssetsFile:
            raise ValueError(f'GF2 native resource {relative}: not a Unity serialized file')
        result = self.load_file(raw, name=name, is_dependency=is_dependency)
        self.loaded.add((relative, 0))
        return result

    def load_member(self, relative, offset):
        if (relative, offset) in self.loaded:
            return
        path = self.catalog.install / relative
        for at, data in members(path):
            if at == offset:
                self.load_file(data, name=f'{relative}@{offset}')
                self.loaded.add((relative, offset))
                return
        raise ValueError(f'{path}: no member at {offset}')
