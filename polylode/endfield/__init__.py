"""Arknights: Endfield actors, converted into this project's baked format.

The same split as :mod:`polylode.origin`: the *import side* reads the user's
own game data on the user's machine and writes ``bake``-format archives into
its own store, ``derived_root()/Endfield`` (:mod:`.store`); the run time
(:mod:`.actor`) reads only those archives.
Nothing converted is tracked or shipped (``著作權.md``, ``CLAUDE.md`` 四之一).

The game's bundles are an encrypted VFS that UnityPy cannot open, so the import
side does not read them itself.  It drives an external exporter (AnimeStudio's
CLI, which the user supplies) into a staging folder of per-object JSON and PNG
files and converts those:

``cabmap``     which bundle slot holds a serialized file, and its dependencies
``assetmap``   which prefabs exist, by their container path
``exporter``   one exporter run: bundle slots in, staged objects out
``staged``     staged objects, addressed by ``(serialized file, path id)``
``prefab``     a staged prefab: nodes, renderers, LOD levels
``unitymesh``  a staged ``Mesh`` as arrays in this project's space
``material``   a staged ``Material`` and the textures it binds
``convert``    prefab → ``origin.prefab.Piece`` list
``components`` the Avatar's human-bone table and the cloth / collider components
``acl``        the exporter's ACL decoder, for clip buffers
``clips``      animation clips: decoded, stored, played back on an actor
``links``      which items belong to which character, by declared character id
``morph``      facial morphs: a character's face controls and the shared
               emotions, converted — and, at run time, weights → a face pose
``store``      the ``Endfield/`` store: items and textures, in ``bake``'s format
``pipeline``   the whole pass, resumable per prefab

and at run time, on top of :mod:`.actor`:

``shading``    a piece's material as the renderer's ``Material``: which of the
               transcribed ``HGRP/CharacterNPR*`` families draws it, and how
``muscle``     a clip's humanoid muscle curves as bone rotations, by the item's
               own Avatar definition
``retarget``   this project's poses carried onto an item's own skeleton
``controller`` an item as a scene object: placed, posed, drawn

**Every reference is followed, none is guessed.**  A renderer's mesh and
materials, a material's textures and a skinned mesh's bones are ``PPtr``s, and
a ``PPtr``'s file id is an index into the dependency list the CAB map records
for that file.  Names are never matched — the export this replaces paired
meshes with textures by file name and found a texture for 2 of 7 meshes.
"""
