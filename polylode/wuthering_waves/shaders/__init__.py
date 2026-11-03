"""Wuthering Waves' compiled material shaders, read and run: what a material draws, from the game's own code.

The game ships no material graphs; its cooked materials carry the compiled result.  These modules read it:

* :mod:`.archive` — ``ShaderArchive-*.ushaderbytecode``: which shaders a shader map holds, and their DXBC;
* :mod:`.inline` — a material instance's inline shader map: its uniform expressions (preshaders), the
  parameters they read and the textures it binds, out of the instance's package;
* :mod:`.preshader` — those expressions evaluated with the instance's own parameter values;
* :mod:`.dxbc` — a shader's assembly (``d3dcompiler_47``), its resource table and uniform buffers;
* :mod:`.vm` — an interpreter that runs a pixel shader's assembly over many points at once.
"""
