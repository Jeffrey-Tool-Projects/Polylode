"""One run of the external exporter: bundle slots in, staged objects out.

The exporter is AnimeStudio's CLI built with the Endfield research kit's
additions (``--filter_data``, ``--export_manifest_jsonl``, the secondary
export).  It is the user's own tool at a path the user gives; nothing of it is
bundled.  A run loads only the bundle slots it is handed and writes every
object of the requested types, plus one manifest row per written file naming
the serialized file and path id it came from — which is what lets
:mod:`.staged` address objects exactly rather than by file name.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .cabmap import CabEntry

Log = Callable[[str], None] | None

GAME = "ArknightsEndfield"


class ExportError(RuntimeError):
    """The exporter could not be run, failed, or wrote no manifest."""


@dataclass(frozen=True)
class Secondary:
    """A second export over the same loaded objects (e.g. textures as PNG)."""
    export_type: str
    out_dir: Path
    types: tuple[str, ...]


@dataclass(frozen=True)
class Exporter:
    """Where the exporter is and the CAB maps it resolves references with."""
    exe: Path
    cab_map_dir: Path

    def run(self, entries: Iterable[CabEntry], out_dir: Path,
            types: Sequence[str], export_type: str = "JSON",
            secondary: Secondary | None = None, log: Log = None) -> list[dict]:
        """Export *types* from the slots of *entries*; returns the manifest rows.

        One process per VFS root (the exporter takes one input folder).  Each
        row is ``{output, type, name, pathId, serializedFile, …}`` with
        ``output`` made absolute.  Raises :class:`ExportError` on a non-zero
        exit or a missing manifest, never returning a partial listing.
        """
        by_root: dict[Path, list[CabEntry]] = {}
        for entry in entries:
            by_root.setdefault(entry.root, []).append(entry)
        rows: list[dict] = []
        for index, (root, group) in enumerate(sorted(by_root.items())):
            rows += self._run_root(root, group, out_dir, index, types,
                                   export_type, secondary, log)
        return rows

    def _run_root(self, root: Path, group: list[CabEntry], out_dir: Path,
                  index: int, types: Sequence[str], export_type: str,
                  secondary: Secondary | None, log: Log) -> list[dict]:
        out_dir.mkdir(parents=True, exist_ok=True)
        filter_path = out_dir / f"filter_{index}.json"
        manifest_path = out_dir / f"manifest_{index}.jsonl"
        manifest_path.unlink(missing_ok=True)
        # The exporter indexes the items by Source and reads only their Offset
        # (``AssetsManager.EnsureFilterDataIndex``); the other three keys are
        # its schema's, and ``Type`` has to parse as one of its class ids.
        slots = sorted({(str(e.source), e.offset) for e in group})
        filter_path.write_text(json.dumps(
            [{"Source": s, "Offset": o, "Name": "", "PathID": 0,
              "Type": "AssetBundle"} for s, o in slots]), encoding="utf-8")
        cmd = [str(self.exe), str(root), str(out_dir), "--game", GAME,
               "--export_type", export_type, "--group_assets", "ByType",
               "--cab_map_dir", str(self.cab_map_dir),
               "--filter_data", str(filter_path),
               "--export_manifest_jsonl", str(manifest_path),
               "--logger_flags", "Warning", "Error",
               "--types", *types]
        if secondary is not None:
            secondary.out_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--secondary_export_type", secondary.export_type,
                    "--secondary_export_path", str(secondary.out_dir),
                    "--secondary_types", *secondary.types]
        if log:
            log(f"exporter: {len(slots)} slots from {root.name} → "
                f"{', '.join(types)}")
        if not self.exe.is_file():
            raise ExportError(f"exporter not found: {self.exe}")
        done = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", check=False)
        if done.returncode != 0:
            raise ExportError(f"exporter exit {done.returncode}: "
                              f"{(done.stdout + done.stderr)[-2000:]}")
        if not manifest_path.is_file():
            raise ExportError(f"exporter wrote no manifest: {manifest_path}")
        return _manifest_rows(manifest_path, out_dir)


def _manifest_rows(path: Path, out_dir: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("kind") != "output":
                continue
            row["output"] = (out_dir / row["output"]).resolve()
            rows.append(row)
    return rows
