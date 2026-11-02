"""The source files Blender is handed: Blender runs them as ``.py`` files, so a packaged program carries them.

The converters start Blender with a worker script beside their own module
(``gf2/blender_worker.py``) that puts the project's root on Blender's path and
also runs ``tools/ef_fbx_blender.py``; the GF2 export also signs its result
with those files' bytes.  A compiled program has no ``.py`` files, so
``build_extractor.py`` copies the package and that script beside the program
at the same relative paths, and :func:`missing` says what is not there.
"""
from __future__ import annotations

from pathlib import Path

from ..core.paths import frozen

#: Files the converters read as source, relative to the project (or program) root.
REQUIRED = ("polylode/gf2/blender_worker.py", "polylode/gf2/export.py", "polylode/gf2/fbx_tangents.py",
            "polylode/gf2/precision.py", "polylode/endfield/fbxfile.py", "polylode/endfield/rigscene.py",
            "tools/ef_fbx_blender.py", "tools/wuwa_source_blender.py")


#: Third-party modules the build leaves as source beside the program (``build_extractor.py``): UnityPy's
#: generated class table is a 31 MB C file the C compiler cannot optimise within 9 GB.
SOURCE_MODULES = ("UnityPy.classes.generated",)


def root() -> Path:
    """Where the package's own files are: the checkout, or the program's folder once built."""
    return Path(__file__).resolve().parents[2]


def source_path(name: str) -> Path:
    return root().joinpath(*name.split(".")).with_suffix(".py")


def missing() -> list[str]:
    return ([rel for rel in REQUIRED if not (root() / rel).is_file()]
            + [name for name in SOURCE_MODULES if frozen() and not source_path(name).is_file()])


class _SourceModules:
    """Finds :data:`SOURCE_MODULES` at their file beside the program: the packaged program's own
    import machinery does not look for a ``.py`` inside a compiled package."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name not in SOURCE_MODULES:
            return None
        import importlib.util
        return importlib.util.spec_from_file_location(name, source_path(name))


def load_source_modules() -> None:
    """In the packaged program, import :data:`SOURCE_MODULES` from their files (before anything imports them)."""
    import sys
    if frozen() and _SourceModules not in sys.meta_path:
        sys.meta_path.insert(0, _SourceModules)
