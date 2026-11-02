"""Character cards: a PNG thumbnail with the parameter payload appended.

Layout, mirroring how Illusion stores cards:

    <complete PNG file bytes, ending in IEND>
    <marker>            e.g. b"[IslandLifeChara]"
    <uint32 little-endian payload length>
    <zlib-compressed UTF-8 JSON>

Image viewers ignore everything past IEND, so a card is still a normal PNG.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

from ..core.i18n import tr

CHARA_MARKER = b"[IslandLifeChara]"
#: A converted character's card: which character it is and how it is edited (:func:`save_actor`)
ACTOR_MARKER = b"[IslandLifeActor]"
ACTOR_VERSION = 2
SCENE_MARKER = b"[IslandLifeScene]"
FORMAT_VERSION = 1

MIN_AGE = 18
ALLOWED_STYLES = ("anime", "toon", "cel", "stylised", "stylized")


class CardError(RuntimeError):
    pass


class CardPolicyError(CardError):
    """Raised when a card fails the adult-only / art-style policy check."""


# --------------------------------------------------------------------------
# low-level PNG payload IO
# --------------------------------------------------------------------------
def png_bytes(image) -> bytes:
    """Accept a QImage, a PIL image, a numpy array or raw PNG bytes."""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if hasattr(image, "save") and hasattr(image, "sizeInBytes"):     # QImage
        from PySide6.QtCore import QBuffer, QByteArray, QIODevice
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.WriteOnly)
        image.save(buf, "PNG")
        buf.close()
        return bytes(ba.data())
    if isinstance(image, np.ndarray):
        from PIL import Image
        arr = image
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(arr)
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def write_card_png(path: Path, image, payload: dict, marker: bytes = CHARA_MARKER) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png = png_bytes(image)
    blob = zlib.compress(json.dumps(payload, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8"), 6)
    with open(path, "wb") as f:
        f.write(png)
        f.write(marker)
        f.write(struct.pack("<I", len(blob)))
        f.write(blob)
    return path


def read_card_png(path: Path, marker: bytes = CHARA_MARKER) -> dict | None:
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CardError(tr("無法讀取檔案: {error}").format(error=exc)) from exc
    return payload_of(data, marker)


def payload_of(data: bytes, marker: bytes = CHARA_MARKER) -> dict | None:
    """The payload *data* (a card file's bytes) carries after *marker*, or None without one."""
    i = data.rfind(marker)
    if i < 0:
        return None
    off = i + len(marker)
    if off + 4 > len(data):
        raise CardError(tr("卡片資料損毀（長度欄位不完整）"))
    (n,) = struct.unpack("<I", data[off:off + 4])
    blob = data[off + 4:off + 4 + n]
    if len(blob) != n:
        raise CardError(tr("卡片資料損毀（內容長度不符）"))
    try:
        return json.loads(zlib.decompress(blob).decode("utf-8"))
    except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CardError(tr("卡片資料無法解析: {error}").format(error=exc)) from exc


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------
def validate_payload(payload: dict, strict: bool = True) -> list[str]:
    """Return a list of policy problems; empty means the card is acceptable."""
    problems = []
    params = payload.get("params") or {}
    try:
        age = int(params.get("param.age", 0))
    except (TypeError, ValueError):
        age = 0
    if age < MIN_AGE:
        problems.append(tr("角色年齡為 {age}，本作僅接受 {minimum} 歲以上的角色").format(age=age, minimum=MIN_AGE))
    style = str(params.get("param.style", "anime")).strip().lower()
    if style not in ALLOWED_STYLES:
        problems.append(tr("角色風格 '{style}' 不被支援，本作僅支援動漫／卡通風格").format(style=style))
    if strict and payload.get("format") != "polylode.chara":
        problems.append(tr("卡片格式標記不正確"))
    return problems


def enforce_policy(payload: dict, source: str = "") -> dict:
    problems = validate_payload(payload)
    if problems:
        text = (tr("此卡片不符合本作的內容規範（{source}）：\n- {problems}") if source
                else tr("此卡片不符合本作的內容規範：\n- {problems}"))
        raise CardPolicyError(text.format(source=source, problems="\n- ".join(problems)))
    return payload


# --------------------------------------------------------------------------
# character cards
# --------------------------------------------------------------------------
def build_payload(params: dict, sex: int = 0, extra: dict | None = None) -> dict:
    from . import params as P
    clean = P.normalise(dict(params))
    clean.setdefault("param.style", "anime")
    payload = {
        "format": "polylode.chara",
        "version": FORMAT_VERSION,
        "sex": int(sex),
        "name": P.display_name(clean),
        "params": clean,
        "policy": {"min_age": MIN_AGE, "style": clean.get("param.style", "anime"),
                   "adult_only": True},
    }
    if extra:
        payload["extra"] = extra
    return payload


def save_character(path: Path, params: dict, image, sex: int = 0,
                   extra: dict | None = None) -> Path:
    payload = build_payload(params, sex, extra)
    enforce_policy(payload, str(path))
    return write_card_png(path, image, payload, CHARA_MARKER)


def load_character(path: Path) -> tuple[dict, dict]:
    """Returns (params, payload). Raises CardPolicyError if non-compliant."""
    payload = read_card_png(path, CHARA_MARKER)
    if payload is None:
        raise CardError(tr("這不是本程式的角色卡（PNG 內沒有角色資料）"))
    enforce_policy(payload, Path(path).name)
    from . import params as P
    return P.normalise(dict(payload.get("params") or {})), payload


def peek_character(path: Path) -> dict | None:
    """Read the header of a card without validating — used to list a folder."""
    try:
        payload = read_card_png(path, CHARA_MARKER)
    except CardError:
        return None
    if not payload:
        return None
    params = payload.get("params") or {}
    return {"name": payload.get("name", "?"),
            "sex": payload.get("sex", 0),
            "age": params.get("param.age", 0),
            "personality": params.get("param.personality", 0),
            "club": params.get("param.club", 0),
            "compliant": not validate_payload(payload),
            "path": Path(path)}


# --------------------------------------------------------------------------
def list_characters(folder: Path) -> list[dict]:
    folder = Path(folder)
    if not folder.exists():
        return []
    out = []
    for p in sorted(folder.glob("*.png"), key=lambda x: x.name.lower()):
        info = peek_character(p)
        if info:
            out.append(info)
    return out


def own_cards(folder: Path) -> list[Path]:
    """This program's character cards in *folder* and one nested level, by name.

    An original-format card there is not one of them: those are test material and
    no screen lists them (owner, 2026-10-17, §7.1174).
    """
    from ..core.paths import list_cards
    out = []
    for p in list_cards(Path(folder)):
        try:
            if CHARA_MARKER in p.read_bytes():
                out.append(p)
        except OSError:
            continue
    return out


def is_wearable(path: Path) -> bool:
    """Whether *path* is a character card of ours that carries an original-format card to dress from.

    Only those can be worn (CLAUDE.md 一之三: the card is where a character's looks are);
    a raw original-format card is test material, and a card of ours without one has no looks.
    """
    from .cardimport import IMPORTED
    try:
        payload = payload_of(Path(path).read_bytes())
    except (OSError, CardError):
        return False
    return bool(((payload or {}).get("extra") or {}).get(IMPORTED))


def wearable_cards(folder: Path) -> list[Path]:
    """The cards in *folder* :func:`is_wearable` accepts."""
    return [p for p in own_cards(folder) if is_wearable(p)]


# --------------------------------------------------------------------------
# converted characters
# --------------------------------------------------------------------------
def save_actor(path: Path, image, category: str, ident: str, name: str, edits: dict,
               sex: int = 0) -> Path:
    """Write a converted character's card: the picture, then which character it is and its edits.

    A converted character has no parameters of its own; the card names it
    in the store (*category*/*ident*) and carries what the character maker
    changed on it (``ActorController.card_edits``).
    """
    if not category or not ident:
        raise CardError(tr("轉換角色卡少了角色的名稱"))
    # The original Endfield writer remains version one. Source-qualified cards
    # use converted.cards and version two, so older programs refuse them.
    payload = {"kind": "actor", "version": 1, "category": str(category), "ident": str(ident),
               "name": str(name or ident), "sex": int(sex), "edits": dict(edits or {})}
    return write_card_png(path, image, payload, ACTOR_MARKER)


def read_actor(path: Path) -> dict | None:
    """A converted character's card payload, or None when *path* is not one."""
    payload = read_card_png(path, ACTOR_MARKER)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise CardError(tr("轉換角色卡的內容不完整"))
    if payload.get("kind") != "actor" or not payload.get("category") or not payload.get("ident")             or not isinstance(payload.get("edits", {}), dict):
        raise CardError(tr("轉換角色卡的內容不完整"))
    version = payload.get("version", 0)
    if type(version) is not int or version < 0:
        raise CardError(tr("轉換角色卡的內容不完整"))
    if version > ACTOR_VERSION:
        raise CardError(tr("這張轉換角色卡是較新版本的程式存的"))
    if version == 2:
        from ..converted.store import SOURCES
        import re
        if (payload.get("source") not in SOURCES or payload.get("category") != "body"
                or not re.fullmatch("[0-9a-f]{64}", str(payload.get("warehouse_sha256", "")))):
            raise CardError(tr("轉換角色卡的內容不完整"))
    elif payload.get("source", "endfield") != "endfield":
        raise CardError(tr("轉換角色卡的內容不完整"))
    return payload


def list_actor_cards(folder: Path) -> list[dict]:
    """The converted characters' cards in *folder*, each with ``"actor": True``."""
    folder = Path(folder)
    if not folder.exists():
        return []
    out = []
    for p in sorted(folder.glob("*.png"), key=lambda x: x.name.lower()):
        try:
            payload = read_actor(p)
        except CardError:
            continue
        if payload:
            out.append({"name": payload.get("name", p.stem), "sex": payload.get("sex", 0), "age": None,
                        "personality": None, "club": None, "compliant": True, "path": p, "actor": True,
                        "ident": payload["ident"]})
    return out
