"""The whole pass: pick prefabs, stage what they reference, bake each one.

Five exporter stages, each loading only the bundle slots the previous one
proved are needed:

1. **hierarchy** — every selected prefab's own slots: GameObjects, Transforms,
   ``LODGroup``s, MonoBehaviours and Animators (one run for all of them);
2. **scripts and avatars** — the files the MonoBehaviours' script pointers and
   the Animators' avatar pointers resolve to (class names, human-bone tables);
3. **materials** — the files the drawn renderers' material pointers resolve to;
4. **textures** — the files those materials' texture pointers resolve to, as
   PNG plus a type-tree dump (colour space, wrap mode);
5. **meshes** — per prefab, because decoded meshes are large: staged,
   converted, baked, and the staging removed.

A prefab already in the store is skipped unless *force*, so a pass that was
interrupted picks up where it stopped.
"""
from __future__ import annotations

import shutil
import struct
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from ..core.paths import derived_root
from . import acl, clips, model, morph
from . import components as K
from . import prefab as P
from . import store
from .assetmap import AssetMap, ClipRow
from .cabmap import CabIndex
from .convert import (Assets, ConvertError, NothingToConvert, Report,
                      build_pieces, drawn, node_names)
from .exporter import ExportError, Exporter, Secondary
from .material import MaterialError
from .staged import Ref, Staged
from .unitymesh import MeshError

Log = Callable[[str], None] | None

_GAMEPLAY = "assets/beyond/dynamicassets/gameplay"


@dataclass(frozen=True)
class Kind:
    """Where this game keeps one kind of prefab, and where its bake goes."""
    category: str              # category in the store
    directory: str             # container directory
    #: True: every prefab below the directory.  False: only the ones directly
    #: in it — its subfolders are other kinds (``uimodels/decoitems``).
    nested: bool = False


KINDS: dict[str, Kind] = {
    "chara": Kind("chara", f"{_GAMEPLAY}/actors/postmodels/characters"),
    "deco": Kind("deco", f"{_GAMEPLAY}/prefabs/uimodels/decoitems"),
    "weapon": Kind("weapon", f"{_GAMEPLAY}/prefabs/weapons"),
    "attachment": Kind("attachment", f"{_GAMEPLAY}/prefabs/attachment"),
    "npc_accessory": Kind("npc_accessory",
                          "assets/beyond/arts/entity/npc/accessory", nested=True),
    "npc_part": Kind("npc_part", f"{_GAMEPLAY}/npc/generated", nested=True),
    "uimodel": Kind("uimodel", f"{_GAMEPLAY}/prefabs/uimodels"),
}

#: every error a single prefab can fail with without stopping the pass
_PREFAB_ERRORS = (ConvertError, P.PrefabError, MeshError, MaterialError,
                  K.ComponentError, LookupError, ExportError)


def stage_root() -> Path:
    """Where staged exports and the asset-map index live (never shipped)."""
    return derived_root() / "assets_cache" / "endfield"


@dataclass(frozen=True)
class Sources:
    """The user's exporter and the maps it built from their install."""
    exe: Path
    cab_maps: tuple            # CAB map files, installed layer first
    asset_maps: tuple          # asset map JSON documents

    @property
    def cab_map_dir(self) -> Path:
        return Path(self.cab_maps[0]).parent


#: Unity's two data roots, installed layer first: what the game downloads
#: (``persistentDataPath``) replaces what it shipped with (``streamingAssetsPath``).
_LAYERS = ("StreamingAssets", "Persistent")


def sources_from_export(exe: Path, meta: Path) -> Sources:
    """The maps of an Endfield research kit export, given its ``meta`` folder.

    The kit writes ``cab_map/endfield_<layer>_assets.bin`` and
    ``<Layer>/asset_map/endfield_<layer>_assets.json``.  The installed layer is
    required; the downloaded one is taken when present.  Raises
    ``FileNotFoundError`` naming the first missing file.
    """
    cab_maps, asset_maps = [], []
    for layer in _LAYERS:
        stem = f"endfield_{layer.lower()}_assets"
        cab = Path(meta) / "cab_map" / f"{stem}.bin"
        assets = Path(meta) / layer / "asset_map" / f"{stem}.json"
        if cab.is_file() and assets.is_file():
            cab_maps.append(cab)
            asset_maps.append(assets)
        elif layer == _LAYERS[0]:
            raise FileNotFoundError(cab if not cab.is_file() else assets)
    return Sources(Path(exe), tuple(cab_maps), tuple(asset_maps))


@dataclass
class Job:
    """One prefab to convert."""
    kind: str
    container: str
    slots: list

    @property
    def category(self) -> str:
        return KINDS[self.kind].category

    @property
    def ident(self) -> str:
        """The container path below the kind's directory, without ``.prefab``.

        A file name alone is not an identity in a nested kind: two NPC part
        folders each hold a ``p_npc_boy_ear_angel_a_01_variant.prefab``.
        """
        below = self.container[len(KINDS[self.kind].directory) + 1:]
        return below[:-len(".prefab")]


#: How many clips one exporter run stages (each decodes to about a megabyte
#: of JSON).
CLIP_BATCH = 60


@dataclass
class ClipResult:
    done: dict = field(default_factory=dict)       # clip id → (frames, tracks)
    failed: dict = field(default_factory=dict)     # clip id → reason
    skipped: int = 0                               # already stored
    other: list = field(default_factory=list)      # clip ids not kept as ACL


#: Where the game keeps its facial-morph assets: a config per Avatar, and the
#: emotions and poses every face shares.
MORPH_ROOT = "assets/beyond/dynamicassets/gamedata/skeletalmorph"


@dataclass
class MorphResult:
    configs: dict = field(default_factory=dict)    # config name → (model, bones, controls)
    unlinked: list = field(default_factory=list)   # configs whose Avatar is no one model's
    failed: dict = field(default_factory=dict)     # asset name → reason
    unread: dict = field(default_factory=dict)     # config name → the record not decoded
    emotions: int = 0
    poses: int = 0


@dataclass
class Result:
    done: dict = field(default_factory=dict)       # "category/ident" → Report
    failed: dict = field(default_factory=dict)     # "category/ident" → reason
    skipped: list = field(default_factory=list)    # already baked
    empty: list = field(default_factory=list)      # no mesh renderer at all
    refused: dict = field(default_factory=dict)    # "category/ident" → why *admit* turned it away


def select(assets: AssetMap, kinds: Iterable[str], only: str = "") -> list[Job]:
    """The prefabs of *kinds* whose name contains *only* (all when empty)."""
    jobs = []
    for kind in kinds:
        spec = KINDS[kind]
        found = assets.prefabs(f"{spec.directory}/%.prefab")
        for container, slots in found.items():
            if not spec.nested and "/" in container[len(spec.directory) + 1:]:
                continue
            job = Job(kind, container, slots)
            if only.lower() in job.ident.lower():
                jobs.append(job)
    return jobs


class Pipeline:
    """Runs the five stages for a list of jobs."""

    def __init__(self, sources: Sources, log: Log = None):
        self.log = log or (lambda _msg: None)
        self.cabs = CabIndex.load(sources.cab_maps)
        self.assets = AssetMap.open(list(sources.asset_maps),
                                    stage_root() / "assetmap.sqlite", self.log)
        self.exporter = Exporter(Path(sources.exe), sources.cab_map_dir)
        self.staged = Staged()
        self.stage = stage_root() / "stage"

    # -- stage 1 ------------------------------------------------------------
    def _hierarchies(self, jobs: list[Job], result: Result) -> dict:
        """job key → Prefab, for every job whose hierarchy staged cleanly."""
        entries, owners = [], {}
        for job in jobs:
            for slot in job.slots:
                cab = self.cabs.at(*slot)
                if cab is not None:
                    entries.append(self.cabs.entry(cab))
                    owners.setdefault(_key(job), []).append(cab)
        self.staged.add(self.exporter.run(
            entries, self.stage / "hierarchy",
            ["GameObject", "Transform", "LODGroup", "MonoBehaviour", "Animator"],
            secondary=Secondary("Dump", self.stage / "hierarchy_dump", ("GameObject",)),
            log=self.log))
        with_objects = {ref.cab for ref in self.staged.of_type("GameObject")}
        prefabs = {}
        for job in jobs:
            cabs = [c for c in owners.get(_key(job), []) if c in with_objects]
            try:
                if len(cabs) != 1:
                    raise P.PrefabError(
                        f"{len(cabs)} of its files hold GameObjects, expected 1")
                prefabs[_key(job)] = P.read_prefab(self.staged, self.cabs, cabs[0])
            except _PREFAB_ERRORS as exc:
                result.failed[_key(job)] = f"hierarchy: {exc}"
        return prefabs

    # -- stage 2 ------------------------------------------------------------
    def _scripts_and_avatars(self, prefabs: dict) -> None:
        wanted = set()
        for pf in prefabs.values():
            for ref in self.staged.of_type("MonoBehaviour", pf.cab):
                script = self.cabs.pointer(pf.cab, self.staged.json(ref).get("m_Script"))
                if script is not None:
                    wanted.add(script.cab)
            avatar = K.animator_avatar(self.staged, self.cabs, pf)
            if avatar is not None:
                wanted.add(avatar.cab)
        self._stage_files(wanted, "scripts", ["MonoScript", "Avatar"])

    def _extra(self, pf: P.Prefab) -> dict:
        """The JSON stored beside the item (:mod:`.components`)."""
        names = node_names(pf)
        kept, seen = K.kept_components(self.staged, self.cabs, pf, names)
        extra = {"components": kept, "classes": seen,
                 "paths": {names[tid]: P.node_path(pf, tid) for tid in pf.nodes}}
        avatar = K.animator_avatar(self.staged, self.cabs, pf)
        if avatar is not None and self.staged.has(avatar):
            extra["avatar"] = {"cab": avatar.cab, "path_id": avatar.path_id}
            document = self.staged.json(avatar)
            extra["humanoid"] = K.humanoid(document, pf, names)
            extra["human"] = K.human_definition(document, pf, names)
            container = self._model_of(pf)
            if container:
                extra["model"] = container
        return extra

    def _model_of(self, pf: P.Prefab) -> str:
        """The model container of the Avatar *pf* is skinned on (staged), or ``""`` when it is not one container's."""
        avatar = K.animator_avatar(self.staged, self.cabs, pf)
        if avatar is None or not self.staged.has(avatar):
            return ""
        entry = self.cabs.entry(avatar.cab)
        containers = self.assets.containers_at(entry.source, entry.offset)
        return containers[0] if len(containers) == 1 else ""

    # -- stages 3 and 4 -----------------------------------------------------
    def _stage_files(self, cabs: set, out: str, types: list,
                     export_type: str = "JSON",
                     secondary: Secondary | None = None) -> None:
        if cabs:
            self.staged.add(self.exporter.run(
                [self.cabs.entry(c) for c in sorted(cabs)], self.stage / out,
                types, export_type, secondary, self.log))

    def _materials(self, prefabs: dict) -> None:
        wanted = {ref.cab for pf in prefabs.values() for node in drawn(pf)[0]
                  for ref in node.renderer.materials if ref is not None}
        self._stage_files(wanted, "materials", ["Material"])

    def _textures(self, prefabs: dict, converter: Assets) -> None:
        wanted = set()
        for pf in prefabs.values():
            for node in drawn(pf)[0]:
                for ref in node.renderer.materials:
                    if ref is not None and self.staged.has(ref):
                        wanted |= {t.ref.cab for t in converter.material(ref).textures}
        self._stage_files(
            wanted, "textures_dump", ["Texture2D"], "Dump",
            Secondary("Convert", self.stage / "textures", ("Texture2D",)))

    # -- stage 5 ------------------------------------------------------------
    def _bake(self, job: Job, pf: P.Prefab, converter: Assets, force: bool) -> Report:
        mesh_dir = self.stage / "meshes"
        shutil.rmtree(mesh_dir, ignore_errors=True)
        wanted = {node.renderer.mesh.cab for node in drawn(pf)[0]
                  if node.renderer.mesh is not None}
        self._stage_files(wanted, "meshes", ["Mesh"])
        # a model's item and every texture it names go to its own Charas folder
        own = store.writing(job.ident) if job.category in store.MODEL_CATEGORIES else nullcontext()
        try:
            with own:
                pieces, nodes, report = build_pieces(pf, converter, force)
                store.write_item(job.category, job.ident, pieces, nodes,
                                 self._extra(pf))
        finally:
            shutil.rmtree(mesh_dir, ignore_errors=True)
        return report

    # -- model containers ---------------------------------------------------
    def model_containers(self) -> list[str]:
        """Every actor model container the asset map lists, sorted."""
        return sorted(self.assets.prefabs(model.MODELS))

    def read_model(self, container: str) -> tuple[list, dict, dict]:
        """A model container's meshes on its Avatar's skeleton (:mod:`.model`).

        ``(pieces, nodes, extra)``, not written to the store: the pieces have
        no material, so they are not an item a scene can draw.  Raises
        ``LookupError`` for a container the asset map does not list and
        :class:`model.ModelError` unless it holds exactly one Avatar.
        """
        slots = self.assets.prefabs(container).get(container)
        if not slots:
            raise LookupError(f"{container} is not in the asset map")
        cabs = [cab for cab in (self.cabs.at(*slot) for slot in slots) if cab]
        out = self.stage / "model"
        shutil.rmtree(out, ignore_errors=True)
        staged = Staged()
        try:
            staged.add(self.exporter.run([self.cabs.entry(cab) for cab in cabs],
                                         out, ["Mesh", "Avatar"], log=self.log))
            avatars = list(staged.of_type("Avatar"))
            if len(avatars) != 1:
                raise model.ModelError(f"{container}: {len(avatars)} Avatars, "
                                       f"expected 1")
            pieces, nodes, extra = model.build(
                staged.json(avatars[0]),
                [staged.json(ref) for ref in staged.of_type("Mesh")],
                Path(container).stem, container)
            extra['avatar'] = {'cab': avatars[0].cab, 'path_id': avatars[0].path_id}
            return pieces, nodes, extra
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def run(self, jobs: list[Job], force: bool = False,
            keep_stage: bool = False, admit=None) -> Result:
        """Convert *jobs*; one prefab's failure is recorded, not raised.

        *keep_stage* leaves the staged hierarchy, materials and textures on
        disk afterwards (the meshes are removed per prefab either way) — for
        looking at what the exporter wrote when a prefab fails.

        *admit(job, model)* — *model* the container of the Avatar the prefab
        is skinned on — returns why a prefab is not converted, or ``None``.
        It is asked once the hierarchies and Avatars are staged and before
        any material, texture or mesh is: a refused prefab is dropped there
        (``Result.refused``) and nothing of it reaches the store.
        """
        result = Result()
        todo = []
        for job in jobs:
            if not force and store.has(job.category, job.ident):
                result.skipped.append(_key(job))
            else:
                todo.append(job)
        if not todo:
            return result
        shutil.rmtree(self.stage, ignore_errors=True)
        prefabs = self._hierarchies(todo, result)
        converter = Assets(self.staged, self.cabs, self._shader_name)
        self._scripts_and_avatars(prefabs)
        if admit is not None:
            for job in todo:
                pf = prefabs.get(_key(job))
                reason = None if pf is None else admit(job, self._model_of(pf))
                if reason:
                    result.refused[_key(job)] = reason
                    del prefabs[_key(job)]
                    self.log(f"{_key(job)}: not converted — {reason}")
        self._materials(prefabs)
        self._textures(prefabs, converter)
        for index, job in enumerate(todo, 1):
            pf = prefabs.get(_key(job))
            if pf is None:
                continue
            try:
                report = self._bake(job, pf, converter, force)
            except NothingToConvert:
                result.empty.append(_key(job))
                self.log(f"[{index}/{len(todo)}] {_key(job)}: no mesh renderer")
                continue
            except _PREFAB_ERRORS as exc:
                result.failed[_key(job)] = str(exc)
                self.log(f"[{index}/{len(todo)}] {_key(job)}: FAILED {exc}")
                continue
            result.done[_key(job)] = report
            self.log(f"[{index}/{len(todo)}] {_key(job)}: {report.pieces} pieces, "
                     f"{report.vertices} vertices, {report.textures} textures")
        if not keep_stage:
            shutil.rmtree(self.stage, ignore_errors=True)
        return result

    def _shader_name(self, ref: Ref) -> str | None:
        entry = self.cabs.entry(ref.cab)
        return self.assets.shader_name(entry.source, entry.offset, ref.path_id)

    # -- links --------------------------------------------------------------
    def char_ids(self) -> list[str]:
        """Every character id a UI-model prefab declares (``CharUIModelMono``).

        Stages only those prefabs' hierarchies and scripts; nothing is baked.
        """
        result = Result()
        shutil.rmtree(self.stage, ignore_errors=True)
        prefabs = self._hierarchies(select(self.assets, ["uimodel"]), result)
        self._scripts_and_avatars(prefabs)
        ids = set()
        for pf in prefabs.values():
            kept, _seen = K.kept_components(self.staged, self.cabs, pf,
                                            node_names(pf))
            ids |= {str(c["_charId"]) for c in kept.get("CharUIModelMono", [])}
        shutil.rmtree(self.stage, ignore_errors=True)
        return sorted(ids)

    # -- clips --------------------------------------------------------------
    def clip_rows(self, models: Iterable[str], groups: Iterable[str]) -> list[ClipRow]:
        """The clips filed with each model and with its body type's ``common``.

        *models* are model asset paths (an item's ``extra["model"]``:
        ``…/actor/<type>/<name>/models/<file>``); *groups* are subfolders of
        ``animations`` (``3c``, ``ui``, …), ``.`` for the clips directly in
        it, or ``all`` for everything below it.
        """
        found: dict[tuple, ClipRow] = {}
        for asset in models:
            art = asset.rsplit("/models/", 1)[0]
            for base in (art, f"{art.rsplit('/', 1)[0]}/common"):
                for group in groups:
                    for row in self._clips_in(f"{base}/animations", group):
                        found[(row.source, row.offset, row.path_id)] = row
        return sorted(found.values(), key=lambda r: (r.container, r.name))

    def _clips_in(self, folder: str, group: str) -> list[ClipRow]:
        if group == "all":
            return self.assets.clips(f"{folder}/%")
        if group != ".":
            return self.assets.clips(f"{folder}/{group}/%")
        return [row for row in self.assets.clips(f"{folder}/%")
                if "/" not in row.container[len(folder) + 1:]]

    def convert_clips(self, rows: list[ClipRow], force: bool = False,
                      batch: int = CLIP_BATCH) -> ClipResult:
        """Decode *rows* into the store; one clip's failure is recorded."""
        result = ClipResult()
        listed = clips.index()
        todo = []
        for row in rows:
            ident = clips.clip_id(row.container, row.name)
            if not force and ident in listed:
                result.skipped += 1
            else:
                todo.append((ident, row))
        decoder = acl.Decoder(self.exporter.exe) if todo else None
        stage = self.stage / "clips"
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            shutil.rmtree(stage, ignore_errors=True)
            staged = Staged()
            cabs = {self.cabs.at(row.source, row.offset) for _ident, row in chunk}
            staged.add(self.exporter.run(
                [self.cabs.entry(c) for c in sorted(c for c in cabs if c)],
                stage, ["AnimationClip"], log=self.log))
            for ident, row in chunk:
                ref = Ref(str(self.cabs.at(row.source, row.offset)), row.path_id)
                try:
                    clip = clips.convert(staged.json(ref), decoder)
                    clips.write(ident, row.container, clip)
                except clips.NotAcl:
                    result.other.append(ident)
                    continue
                except (clips.ClipError, acl.AclError, LookupError) as exc:
                    result.failed[ident] = str(exc)
                    continue
                result.done[ident] = (clip.frames, len(clip.hashes))
            self.log(f"clips {min(start + batch, len(todo))}/{len(todo)}: "
                     f"{len(result.done)} stored, {len(result.failed)} failed")
        shutil.rmtree(stage, ignore_errors=True)
        return result


    # -- facial morphs ---------------------------------------------------------
    def _stage_morphs(self, like: str, out: str, raw: bool) -> tuple[Staged, list[Ref]]:
        """Export the scripted objects under *like*; the refs are the decoded ones."""
        staged = Staged()
        slots = {slot for found in self.assets.prefabs(like).values() for slot in found}
        cabs = sorted(cab for cab in (self.cabs.at(*slot) for slot in slots) if cab)
        if not cabs:
            return staged, []
        staged.add(self.exporter.run(
            [self.cabs.entry(c) for c in cabs], self.stage / out, ["MonoBehaviour"],
            secondary=Secondary("Raw", self.stage / f"{out}_raw", ("MonoBehaviour",))
            if raw else None, log=self.log))
        return staged, [ref for ref in staged.of_type("MonoBehaviour")
                        if staged.has(ref, ".json")]

    def convert_morphs(self) -> MorphResult:
        """Convert every face and ear config, and the shared emotions and poses.

        A config is filed under the model asset its ``avatar`` pointer belongs
        to — the same Avatar a converted character's Animator uses — so a
        character finds its face by that reference, not by its name.
        """
        result = MorphResult()
        shutil.rmtree(self.stage, ignore_errors=True)
        staged, refs = self._stage_morphs(f"{MORPH_ROOT}/skeletalmorphcfg/%", "morph_cfg", True)
        for ref in refs:
            doc = staged.json(ref)
            name = str(doc.get("m_Name") or "")
            if "basePoseConfig" not in (doc.get("data") or {}):
                continue                                   # not a bone config (lip-sync table)
            try:
                table = morph.config(doc, staged.path(ref, ".dat").read_bytes())
            except morph.Unread as exc:
                result.unread[name] = str(exc)
                continue
            except (morph.MorphError, LookupError, KeyError, struct.error) as exc:
                result.failed[name] = str(exc)
                continue
            avatar = self.cabs.pointer(ref.cab, doc.get("avatar"))
            models = []
            if avatar is not None:
                entry = self.cabs.entry(avatar.cab)
                models = self.assets.containers_at(entry.source, entry.offset)
            if len(models) != 1:
                result.unlinked.append(name)
                continue
            table["avatar"] = {"cab": avatar.cab, "path_id": avatar.path_id}
            table["model"] = models[0]
            morph.write_config(name, models[0], table)
            result.configs[name] = (models[0], len(table["bones"]), len(table["controls"]))
        tables = {}
        for kind in ("emotion", "pose"):
            staged, refs = self._stage_morphs(
                f"{MORPH_ROOT}/skeletalmorphanim/{kind}/%", f"morph_{kind}", False)
            tables[kind] = {}
            for ref in refs:
                doc = staged.json(ref)
                name = str(doc.get("m_Name") or "")
                try:
                    tables[kind][name] = morph.pose(doc)
                except (KeyError, TypeError, ValueError) as exc:
                    result.failed[name] = str(exc)
        morph.write_poses(tables["emotion"], tables["pose"])
        result.emotions, result.poses = len(tables["emotion"]), len(tables["pose"])
        shutil.rmtree(self.stage, ignore_errors=True)
        return result


def _key(job: Job) -> str:
    return f"{job.category}/{job.ident}"
