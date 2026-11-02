"""Writing the converted scenes as interchange files without Blender: glTF binary (:mod:`.glb`).

Every converter hands Blender the same scene: a document (bones with parents,
parts with their material slots, UV sets, shapes, materials with their maps)
and arrays (``bone_rest`` world rest frames, ``p<i>_co`` … per part), in
Blender's space — right-handed, Z up, metres (``tools/ef_fbx_blender.py``).
The writers here read that same scene.
"""
