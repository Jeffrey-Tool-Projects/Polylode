"""GF2's separate local store, using the project's existing bake schema."""
import os
from contextlib import contextmanager
from pathlib import Path

from ..core.paths import derived_root
from ..origin import bake

SOURCE = 'gf2'


def root():
    value = os.environ.get('ISLANDLIFE_GF2_DIR')
    return Path(value) if value else derived_root() / 'GF2'


@contextmanager
def opened():
    with bake.store(root()):
        yield


def read_item(ident):
    from . import source_tangents
    with opened():
        item = bake.read_item('chara', ident)
    return source_tangents.attach(root(), str(ident), item)


def tangent_shape_frames(ident: str, source_node: str) -> list[dict] | None:
    """Read the exact own-source expression frames; None means supplementary data is unavailable."""
    from . import source_tangents
    return source_tangents.shape_frames(root(), ident, source_node)


def read_extra(ident):
    with opened():
        return bake.named_table(SOURCE, 'chara', ident)[1]


def texture_path(cab, name):
    with opened():
        entry = bake.named_entry(SOURCE, cab, name)
        if entry is None or entry.get('absent'):
            raise FileNotFoundError(f'GF2 texture not stored: {cab}/{name}')
        return root() / entry['file']
