"""Read Koikatsu Studio scene cards and background images from a user install.

A Origin studio scene is a PNG thumbnail with binary scene data after IEND and a
trailing ``【KStudio】`` marker.  Full object-tree reconstruction is out of
scope (props reference abdata ids this project does not ship).  What *is*
portable and useful:

* the embedded character cards (same ChaFile stream as ``UserData/chara``)
* the PNG thumbnail, usable as a backdrop
* pure background images under ``UserData/bg``

Nothing is copied into the project; files are read from the configured install.
"""
from __future__ import annotations

import struct
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from ..core import swallowed

KSTUDIO_MARKER = "【KStudio】"
CHARA_MARKERS = (
    "【KoiKatuChara】",
    "【KoiKatuCharaSP】",
    "【KoiKatuCharaSun】",
    "【EroMakeChara】",
)


@dataclass
class ExtractedCard:
    """One character card pulled out of a studio scene."""
    index: int
    marker: str
    png_bytes: bytes
    # filled lazily by ``materialise`` / callers that import
    name: str = ""
    sex: int = 0          # IslandLife convention: 0 female, 1 male


@dataclass
class OriginSceneInfo:
    path: Path
    version: str = ""
    thumbnail_png: bytes = b""
    cards: list[ExtractedCard] = field(default_factory=list)
    marker: str = KSTUDIO_MARKER

    @property
    def label(self) -> str:
        try:
            root = _install_root()
        except Exception as exc:  # noqa: BLE001
            swallowed.note("origin.scene.OriginSceneInfo.label: _install_root", exc)
            root = None
        if root is not None:
            try:
                return self.path.relative_to(root / "UserData" / "studio" / "scene").as_posix()
            except ValueError:               # a scene picked from outside the install
                pass
        return self.path.name


def _install():
    from .install import configured, autodetect, set_path
    inst = configured()
    if not inst.valid:
        auto = autodetect()
        if auto is not None and auto.valid:
            set_path(auto.root)
            inst = auto
    return inst


def _install_root() -> Path | None:
    inst = _install()
    return inst.root if inst.valid else None


def install_scene_dir() -> Path | None:
    """``UserData/studio/scene`` under the configured Origin install, or None."""
    root = _install_root()
    if root is None:
        return None
    p = root / "UserData" / "studio" / "scene"
    return p if p.is_dir() else None


def install_bg_dir() -> Path | None:
    """``UserData/bg`` under the configured Origin install, or None."""
    root = _install_root()
    if root is None:
        return None
    p = root / "UserData" / "bg"
    return p if p.is_dir() else None


def list_install_scenes(*, limit: int = 200) -> list[Path]:
    """PNG studio scenes from the install, newest first (nested packs included)."""
    folder = install_scene_dir()
    if folder is None:
        return []
    cards = [p for p in folder.rglob("*.png") if p.is_file() and is_kstudio(p)]
    cards.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cards[: max(1, int(limit))]


def list_install_bgs(*, limit: int = 120) -> list[Path]:
    """Plain background images from ``UserData/bg`` (no KStudio payload required)."""
    folder = install_bg_dir()
    if folder is None:
        return []
    imgs = [p for p in folder.rglob("*.png") if p.is_file()]
    imgs.sort(key=lambda p: p.name.lower())
    return imgs[: max(1, int(limit))]


def is_kstudio(path: Path) -> bool:
    """True when the file carries a ``【KStudio】`` marker (cheap tail scan)."""
    path = Path(path)
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - 4096))
            tail = f.read()
        return KSTUDIO_MARKER.encode("utf-8") in tail
    except OSError:
        return False


def _png_end(data: bytes) -> int:
    """Byte offset just past the IEND chunk (start of trailing payload), or 0."""
    from ..chara.cardimport import png_iend_end
    return png_iend_end(data) or 0


def thumbnail_png_bytes(path: Path) -> bytes:
    """PNG bytes of the scene thumbnail only (payload stripped)."""
    data = Path(path).read_bytes()
    end = _png_end(data)
    return data[:end] if end else data


def _read7_len(data: bytes, i: int) -> tuple[int, int]:
    n = 0
    shift = 0
    while i < len(data):
        b = data[i]
        i += 1
        n |= (b & 0x7F) << shift
        if not (b & 0x80):
            return n, i
        shift += 7
        if shift > 35:
            break
    return 0, i


def _find_card_starts(data: bytes) -> list[tuple[int, str]]:
    """Return (byte_offset of int32 productNo, marker_string) for each card."""
    found: list[tuple[int, str]] = []
    for marker in CHARA_MARKERS:
        mb = marker.encode("utf-8")
        idx = 0
        while True:
            j = data.find(mb, idx)
            if j < 0:
                break
            # BinaryWriter: 7-bit length prefix immediately before the marker
            # For all our markers length fits in one byte.
            if j >= 5 and data[j - 1] == len(mb):
                start = j - 5          # int32 product + 1-byte len
                # sanity: product number is a small positive int on real cards
                prod = struct.unpack_from("<i", data, start)[0]
                if 0 < prod < 10_000:
                    found.append((start, marker))
            idx = j + 1
    found.sort(key=lambda t: t[0])
    # de-dupe overlapping starts
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for off, m in found:
        if off in seen:
            continue
        seen.add(off)
        out.append((off, m))
    return out


def _minimal_png() -> bytes:
    """1×1 transparent PNG used as a carrier for extracted card payloads."""
    # Build a valid tiny PNG so cardimport's IEND scan lands before the card blob.
    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)  # 1×1 RGBA
    # filter byte 0 + 4 transparent samples
    raw = b"\x00\x00\x00\x00\x00"
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


def extract_character_pngs(path: Path) -> list[ExtractedCard]:
    """Pull every embedded Illusion character card out of a studio scene PNG."""
    data = Path(path).read_bytes()
    starts = _find_card_starts(data)
    if not starts:
        return []
    kstudio = data.find(KSTUDIO_MARKER.encode("utf-8"))
    ends = [s for s, _ in starts[1:]] + [kstudio if kstudio > 0 else len(data)]
    carrier = _minimal_png()
    out: list[ExtractedCard] = []
    for i, ((start, marker), end) in enumerate(zip(starts, ends)):
        if end <= start:
            continue
        blob = data[start:end]
        # Trim trailing zeros that pad object records — keep a small tail so
        # MessagePack sizes still resolve, but drop obvious filler.
        while len(blob) > 64 and blob[-1] == 0 and blob[-32:] == b"\x00" * 32:
            blob = blob[:-16]
        out.append(ExtractedCard(
            index=i, marker=marker, png_bytes=carrier + blob))
    return out


def materialise_cards(cards: list[ExtractedCard], *, folder: Path | None = None
                      ) -> list[Path]:
    """Write extracted card PNGs to ``folder`` (or a temp dir); return paths."""
    if folder is None:
        folder = Path(tempfile.mkdtemp(prefix="il_originscene_"))
    else:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for c in cards:
        p = folder / f"scene_char_{c.index:02d}.png"
        p.write_bytes(c.png_bytes)
        paths.append(p)
    return paths


def enrich_cards(cards: list[ExtractedCard]) -> list[ExtractedCard]:
    """Import each extracted card to fill ``name`` / ``sex`` (best-effort)."""
    if not cards:
        return cards
    from ..chara import cardimport
    from ..chara import params as P
    paths = materialise_cards(cards)
    for c, path in zip(cards, paths):
        try:
            info = cardimport.detect(path)
            if not info.get("supported"):
                continue
            prm, _ = cardimport.import_card(path)
            c.name = P.display_name(prm) or f"角色{c.index + 1}"
            c.sex = int(prm.get("param.sex", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            swallowed.note("origin.scene.enrich_cards: info = cardimport.detect(path)", exc)
            c.name = f"角色{c.index + 1}"
    return cards


def read_scene(path: Path, *, enrich: bool = True) -> OriginSceneInfo:
    """Parse a Origin studio scene PNG into thumbnail + embedded character cards."""
    path = Path(path)
    data = path.read_bytes()
    end = _png_end(data)
    thumb = data[:end] if end else b""
    payload = data[end:] if end else b""
    marker_b = KSTUDIO_MARKER.encode("utf-8")
    if payload.endswith(marker_b):
        body = payload[:-len(marker_b)]
    else:
        body = payload
    version = ""
    if body:
        try:
            n, o = _read7_len(body, 0)
            if 0 < n < 32:
                version = body[o:o + n].decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            swallowed.note("origin.scene.read_scene: n, o = _read7_len(body, 0)", exc)
            version = ""
    cards = extract_character_pngs(path)
    if enrich:
        cards = enrich_cards(cards)
    return OriginSceneInfo(
        path=path, version=version, thumbnail_png=thumb, cards=cards)


def write_thumbnail(path: Path, dest: Path) -> Path:
    """Save the scene's pure PNG thumbnail to ``dest``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(thumbnail_png_bytes(path))
    return dest


def average_color(png_bytes: bytes) -> tuple[float, float, float]:
    """Cheap average RGB in 0..1 for sky tint when image bg is unavailable."""
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        im.thumbnail((64, 64))
        arr = list(im.getdata())
        if not arr:
            return (0.45, 0.48, 0.55)
        r = sum(p[0] for p in arr) / (len(arr) * 255.0)
        g = sum(p[1] for p in arr) / (len(arr) * 255.0)
        b = sum(p[2] for p in arr) / (len(arr) * 255.0)
        return (float(r), float(g), float(b))
    except Exception as exc:  # noqa: BLE001
        swallowed.note("origin.scene.average_color: from PIL import Image", exc)
        return (0.45, 0.48, 0.55)
