"""A standalone tool that scans the user's own game installs for character models and exports them.

Each source (:mod:`.sources`) scans its install for the models this project
may take (the same identity and age scope as every other conversion here) and
exports the ones the user picks as textured Blender and/or FBX files, through
the project's own converters.  The window (:mod:`.app`) runs every export as a
job of its own (:mod:`.jobs`) in a second process of the same program, so a
converter that fails or runs out of memory leaves the window and the queue
alone.  Nothing it writes is tracked or shipped: the files go where the user
says.
"""
