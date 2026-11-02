"""Optional runtime loading of Koikatsu-derived assets.

Runtime reads only project-local trees:

  assets_cache/   — mirrored abdata bundles + zipmods
  UserData/chara/ — character cards

Use ``tools/sync_origin_local.py <install>`` once to fill the cache from a game
install.  Nothing ships Illusion content in the repo itself.
"""
from .install import (  # noqa: F401
    Install, autodetect, check, configured, enabled, set_path,
    cache_dir, local_install, sync_from_install,
)
from .lists import AssetIndex, AssetRef, index  # noqa: F401
from .meshimport import clear_cache, find_meshes, list_meshes, load_mesh, load_texture  # noqa: F401
from .scene import (  # noqa: F401
    list_install_scenes, list_install_bgs, read_scene, is_kstudio,
    install_scene_dir, install_bg_dir,
)
from .studio_catalog import catalog as studio_catalog  # noqa: F401
