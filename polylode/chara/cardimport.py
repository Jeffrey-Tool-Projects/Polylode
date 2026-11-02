"""Import Koikatsu / Koikatsu Sunshine character cards.

An Illusion card is a PNG thumbnail followed by a .NET ``BinaryWriter`` stream:

    <PNG bytes ... IEND + CRC>
    int32   loadProductNo
    string  marker            "【KoiKatuChara】"   (7-bit length prefix, UTF-8)
    string  version
    int32   faceImageLength   + bytes
    int32   blockHeaderLength + MessagePack block header {lstInfo: [{name,pos,size}]}
    int64   dataLength        + concatenated sections ("Custom", "Parameter", ...)

**What can and cannot be imported.** Illusion cards address content by asset ID
— ``hairId``, ``topId``, ``headId`` index their game's asset database. This
project generates all geometry procedurally and ships no such assets, so those
IDs have no meaning here and are recorded but not applied.  What *is* portable
are the named, unit-valued fields: identity, personality, and the colours.  The
importer states exactly what it took, so nothing is silently invented.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core import swallowed
from ..core.i18n import N_, tr

ORIGIN_MARKERS = (
    "【KoiKatuChara】",
    "【KoiKatuCharaSP】",
    "【KoiKatuCharaSun】",
    # EmotionCreators is the Koikatsu engine with a different front end and
    # writes the same ChaFile layout, so the reader handles it unchanged.  Its
    # item ids index EC's own lists, though, so anything it references that
    # Koikatsu doesn't ship simply won't resolve.
    "【EroMakeChara】",
)

# Every Illusion card marker we can recognise, so an unsupported file gets a
# useful answer instead of "not a card". Filenames are never consulted — people
# rename cards constantly.
KNOWN_MARKERS = [
    ("【KoiKatuChara】", "角色卡", True),
    ("【KoiKatuCharaSP】", "角色卡（SP 格式）", True),
    ("【KoiKatuCharaSun】", "角色卡（Sunshine 格式）", True),
    ("[IslandLifeChara]", "IslandLife 角色卡", True),
    ("【KoiKatuClothes】", "服裝卡", False),
    ("[IslandLifeCoord]", "IslandLife 服裝卡", False),
    ("【KStudio】", "工作室場景檔", True),
    ("[IslandLifeScene]", "IslandLife 場景檔", True),
    ("[IslandLifePose]", "IslandLife 姿勢檔", False),
    ("【EroMakeChara】", "EmotionCreators 角色卡", True),
    ("【AIS_Chara】", "AI 少女 角色卡", False),
    ("【AIS_Clothes】", "AI 少女 服裝卡", False),
    ("【HCChara】", "Honey Select 2 角色卡", False),
    ("【HCPChara】", "Honey Select 2 角色卡", False),
    ("【PHChara】", "PlayHome 角色卡", False),
    ("【SBChara】", "Summer Vacation 角色卡", False),
]


def detect(path: Path) -> dict:
    """Identify a PNG by its embedded payload, never by its name.

    Returns ``{kind, label, supported, marker, reason}``; ``supported`` says
    whether this project can actually load it.
    """
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        return {"kind": "error", "label": "無法讀取檔案", "supported": False,
                "marker": "", "reason": str(exc)}
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return {"kind": "not_png", "label": "不是 PNG 檔", "supported": False,
                "marker": "", "reason": "檔頭不是 PNG"}
    for marker, label, supported in KNOWN_MARKERS:
        if marker.encode("utf-8") in data:
            kind = ("polylode" if marker.startswith("[IslandLife")
                    else "illusion")
            return {"kind": kind, "label": label, "supported": supported,
                    "marker": marker,
                    "reason": "" if supported else "本作不支援此類型"}
    return {"kind": "plain_png", "label": "一般 PNG（沒有角色資料）",
            "supported": False, "marker": "",
            "reason": "檔案裡沒有任何已知的角色卡資料"}


class CardImportError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# minimal MessagePack reader (only the subset Illusion cards use)
# --------------------------------------------------------------------------
class _MsgPack:
    def __init__(self, data: bytes):
        self.d = data
        self.i = 0

    def _u(self, fmt: str, n: int):
        v = struct.unpack_from(fmt, self.d, self.i)[0]
        self.i += n
        return v

    def read(self):
        if self.i >= len(self.d):
            raise CardImportError(tr("MessagePack 資料提前結束"))
        b = self.d[self.i]
        self.i += 1
        if b <= 0x7F:
            return b                                   # positive fixint
        if b >= 0xE0:
            return b - 0x100                           # negative fixint
        if 0x80 <= b <= 0x8F:
            return self._map(b & 0x0F)                 # fixmap
        if 0x90 <= b <= 0x9F:
            return self._array(b & 0x0F)               # fixarray
        if 0xA0 <= b <= 0xBF:
            return self._str(b & 0x1F)                 # fixstr
        if b == 0xC0:
            return None
        if b == 0xC2:
            return False
        if b == 0xC3:
            return True
        if b == 0xC4:
            return self._bin(self._u("<B", 1))
        if b == 0xC5:
            return self._bin(self._u(">H", 2))
        if b == 0xC6:
            return self._bin(self._u(">I", 4))
        if b in (0xC7, 0xC8, 0xC9):                    # ext — skip payload
            # One length, read at its own width.  This used to be a dict of
            # the three ``_u`` calls indexed by ``b`` — and a dict literal
            # evaluates every value, so the cursor moved 7 bytes, not 1/2/4,
            # and everything after an ext was read out of step (§7.466).
            n = (self._u("<B", 1) if b == 0xC7 else
                 self._u(">H", 2) if b == 0xC8 else self._u(">I", 4))
            self.i += 1 + n
            return None
        if b == 0xCA:
            return self._u(">f", 4)
        if b == 0xCB:
            return self._u(">d", 8)
        if b == 0xCC:
            return self._u("<B", 1)
        if b == 0xCD:
            return self._u(">H", 2)
        if b == 0xCE:
            return self._u(">I", 4)
        if b == 0xCF:
            return self._u(">Q", 8)
        if b == 0xD0:
            return self._u("<b", 1)
        if b == 0xD1:
            return self._u(">h", 2)
        if b == 0xD2:
            return self._u(">i", 4)
        if b == 0xD3:
            return self._u(">q", 8)
        if 0xD4 <= b <= 0xD8:                          # fixext 1..16
            n = 1 << (b - 0xD4)
            self.i += 1 + n
            return None
        if b == 0xD9:
            return self._str(self._u("<B", 1))
        if b == 0xDA:
            return self._str(self._u(">H", 2))
        if b == 0xDB:
            return self._str(self._u(">I", 4))
        if b == 0xDC:
            return self._array(self._u(">H", 2))
        if b == 0xDD:
            return self._array(self._u(">I", 4))
        if b == 0xDE:
            return self._map(self._u(">H", 2))
        if b == 0xDF:
            return self._map(self._u(">I", 4))
        raise CardImportError(tr("不支援的 MessagePack 標記 0x{b:02X}").format(b=b))

    def _str(self, n: int) -> str:
        s = self.d[self.i:self.i + n].decode("utf-8", errors="replace")
        self.i += n
        return s

    def _bin(self, n: int) -> bytes:
        s = self.d[self.i:self.i + n]
        self.i += n
        return s

    def _array(self, n: int) -> list:
        return [self.read() for _ in range(n)]

    def _map(self, n: int) -> dict:
        out = {}
        for _ in range(n):
            k = self.read()
            out[k if isinstance(k, str) else str(k)] = self.read()
        return out


def _unpack(data: bytes):
    return _MsgPack(data).read()


#: What :func:`_unpack` raises on bytes that are not a MessagePack value: its
#: own "ran out" / "unknown marker", ``struct`` reading past the end, and the
#: recursion a run of nested-array markers in random data turns into.
MSGPACK_ERRORS = (CardImportError, struct.error, RecursionError)


# --------------------------------------------------------------------------
# .NET BinaryReader helpers
# --------------------------------------------------------------------------
class _Reader:
    def __init__(self, data: bytes, pos: int = 0):
        self.d = data
        self.i = pos

    def i32(self) -> int:
        v = struct.unpack_from("<i", self.d, self.i)[0]
        self.i += 4
        return v

    def i64(self) -> int:
        v = struct.unpack_from("<q", self.d, self.i)[0]
        self.i += 8
        return v

    def string(self) -> str:
        """.NET length-prefixed string (7-bit encoded length)."""
        n = 0
        shift = 0
        while True:
            b = self.d[self.i]
            self.i += 1
            n |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
            if shift > 35:
                raise CardImportError(tr("字串長度前綴損毀"))
        s = self.d[self.i:self.i + n].decode("utf-8", errors="replace")
        self.i += n
        return s

    def bytes(self, n: int) -> bytes:
        b = self.d[self.i:self.i + n]
        self.i += n
        return b


def png_iend_end(data: bytes) -> int | None:
    """Offset just past a PNG's ``IEND`` chunk, found by walking the chunk list.

    Each chunk is length, type, data, CRC; the walk never reads inside one, so
    ``IEND`` appearing in compressed image data by chance cannot end it early
    (a search for the four letters can — §7.467).  ``None`` when *data* has no
    PNG signature or the list runs off the end before an ``IEND``.  Shared by
    the card reader and the studio-scene reader: the trailing payload of both
    starts here.
    """
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    i = 8
    while i + 12 <= len(data):
        n = struct.unpack_from(">I", data, i)[0]
        kind = data[i + 4:i + 8]
        i += 12 + n
        if kind == b"IEND":
            return i if i <= len(data) else None
    return None


def _png_end(data: bytes) -> int:
    """Where the card data starts: :func:`png_iend_end`, else a search for ``IEND``."""
    end = png_iend_end(data)
    if end is not None:
        return end
    i = data.find(b"IEND")
    if i < 0:
        raise CardImportError(tr("這不是 PNG 檔（找不到 IEND）"))
    return i + 8               # IEND + CRC


# --------------------------------------------------------------------------
@dataclass
class OriginCard:
    marker: str = ""
    version: str = ""
    sections: dict = field(default_factory=dict)
    # Some sections are nested containers rather than plain MessagePack, so the
    # undecoded bytes are kept for readers that know the inner layout.
    raw: dict = field(default_factory=dict)
    thumbnail: bytes = b""
    # the whole file as read: what ``cardwrite`` patches when the maker saves
    # the card back, so every block it does not touch stays byte for byte
    data: bytes = b""

    @property
    def custom(self) -> dict:
        return self.sections.get("Custom") or {}

    @property
    def parameter(self) -> dict:
        return self.sections.get("Parameter") or {}

    def face(self) -> dict:
        return self.custom.get("face") or {}

    def body(self) -> dict:
        return self.custom.get("body") or {}

    def hair(self) -> dict:
        return self.custom.get("hair") or {}


def without_status(card: OriginCard) -> OriginCard:
    """*card* as ``LoadCharaFile(…, noLoadStatus: true)`` leaves it: the ``Status`` block not read.

    Free H (``FreeHClassRoomCharaFile.Start``), Studio's ``AddObjectFemale.Add`` and the maker's
    file list all load that way, so the character starts from ``ChaFileStatus.MemberInit`` —
    mouth not fixed and fully open, eyes fully open, indoor shoes, every garment on — whatever
    the card saved (§7.1206).  The readers already treat a card without the block as those defaults.
    """
    from dataclasses import replace
    return replace(card, sections={k: v for k, v in (card.sections or {}).items() if k != "Status"})


def _find_header(data: bytes, start: int, window: int = 512):
    """Locate the ``lstInfo`` index and where the data blob begins.

    The fields between the version string and the index differ per game:
    Koikatsu writes a face thumbnail, EmotionCreators writes two GUIDs and a
    flag instead.  Rather than encode each layout, scan the next few hundred
    bytes for a length-prefixed MessagePack map that actually contains
    ``lstInfo`` -- there is only one, so this cannot pick the wrong thing.
    """
    starts = [start]
    try:                            # Koikatsu puts a face thumbnail here first
        face_len = struct.unpack_from("<i", data, start)[0]
        if 0 < face_len < len(data) - start:
            starts.append(start + 4 + face_len)
    except struct.error:
        pass

    for begin in starts:
        limit = min(len(data) - 4, begin + window)
        for off in range(begin, limit):
            try:
                n = struct.unpack_from("<i", data, off)[0]
            except struct.error:
                break
            if not (16 < n < 1 << 20) or off + 4 + n > len(data):
                continue
            try:
                header = _unpack(data[off + 4:off + 4 + n])
            except MSGPACK_ERRORS:
                continue                             # not a header here; keep scanning
            if isinstance(header, dict) and header.get("lstInfo"):
                return header, off + 4 + n + 8       # skip the i64 total length
    return None, start


def locate(data: bytes) -> tuple[int, str, str, dict, int]:
    """``(end of the thumbnail PNG, marker, version, header, data start)`` of a card file.

    The block positions in ``header["lstInfo"]`` are relative to the data
    start, so the thumbnail in front can change size without touching them.
    """
    start = _png_end(data)
    idx, marker = -1, ""
    for m in ORIGIN_MARKERS:
        j = data.find(m.encode("utf-8"), start)
        if j >= 0:
            idx, marker = j, m
            break
    if idx < 0:
        raise CardImportError(tr("這不是角色卡（找不到卡片標記）"))

    # step back over the 7-bit length prefix that precedes the marker
    r = _Reader(data, idx)
    r.i = idx - 1
    while r.i > start and data[r.i] & 0x80:
        r.i -= 1
    r.string()                      # marker
    version = r.string()

    header, base = _find_header(data, r.i)
    if header is None:
        raise CardImportError(tr("找不到卡片索引（lstInfo）"))
    return start, marker, version, header, base


#: where an IslandLife character card keeps the original-format card it was imported from: the
#: maker saves only its own format (owner, 2026-10-16), and the card's looks live in that card
IMPORTED = "imported_card"


def imported_bytes(data: bytes) -> bytes | None:
    """The original-format card an IslandLife character card's bytes *data* carry, or None."""
    import base64

    from . import card as card_io
    try:
        payload = card_io.payload_of(data)
    except card_io.CardError:
        return None
    kept = ((payload or {}).get("extra") or {}).get(IMPORTED)
    return base64.b64decode(kept) if kept else None


def read_origin_card(path: Path) -> OriginCard:
    """The original-format card at *path*, or the one the IslandLife character card there carries."""
    data = Path(path).read_bytes()
    if not any(m.encode("utf-8") in data for m in ORIGIN_MARKERS):
        data = imported_bytes(data) or data
    return origin_card_of(data)


def origin_card_of(data: bytes) -> OriginCard:
    """The original-format card whose file bytes are *data*."""
    start, marker, version, header, base = locate(data)
    card = OriginCard(thumbnail=data[:start], marker=marker, version=version, data=data)

    for info in (header.get("lstInfo") or []):
        name = info.get("name")
        pos = int(info.get("pos", 0))
        size = int(info.get("size", 0))
        blob = data[base + pos: base + pos + size]
        if not blob:
            continue
        card.raw[name] = blob
        try:
            card.sections[name] = (_read_custom(blob) if name == "Custom"
                                   else _unpack(blob))
        except (CardImportError, struct.error, IndexError):
            card.sections[name] = {"_raw_bytes": len(blob)}
    return card


def _read_custom(blob: bytes) -> dict:
    """The Custom block is three int32-length-prefixed MessagePack objects."""
    r = _Reader(blob, 0)
    out = {}
    for key in ("face", "body", "hair"):
        n = r.i32()
        out[key] = _unpack(r.bytes(n))
    return out


# --------------------------------------------------------------------------
# mapping onto the IslandLife schema
# --------------------------------------------------------------------------
BLOOD = ["A", "B", "O", "AB"]


def _rgb(v) -> tuple | None:
    """Illusion colours are {r,g,b,a} maps or 4-element arrays of 0..1 floats."""
    if isinstance(v, dict):
        try:
            return (float(v.get("r", 1)), float(v.get("g", 1)), float(v.get("b", 1)))
        except (TypeError, ValueError):
            return None
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return tuple(float(x) for x in v[:3])
        except (TypeError, ValueError):
            return None
    return None


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _origin_height_metres(card) -> float | None:
    """Height the card asks for, read from Illusion's own height curve.

    Optional on purpose: card conversion has to keep working with no asset
    library present (the maker imports sliders from a bare card), so any
    failure here just leaves the raw slider value in place.
    """
    try:
        from ..origin.figure.shape_apply import card_height
        return card_height(card)
    except Exception as exc:                         # noqa: BLE001
        swallowed.note("chara.cardimport._origin_height_metres: from ..origin.figure.shape_apply import card_h", exc)
        return None


def convert(card: OriginCard, params: dict | None = None) -> tuple[dict, list[str], list[str]]:
    """Map a Origin card onto IslandLife parameters.

    Returns (params, imported, skipped) where the two lists name exactly which
    fields were taken and which were recognised but cannot apply here.
    """
    from . import params as P
    p = P.normalise(dict(params) if params else P.defaults())
    took: list[str] = []
    skipped: list[str] = []

    par = card.parameter
    face = card.face()
    body = card.body()
    hair = card.hair()

    # ---- identity ----------------------------------------------------
    def text(key, dst, label):
        v = par.get(key)
        if isinstance(v, str) and v.strip():
            p[dst] = v.strip()
            took.append(label)

    text("lastname", "param.last_name", tr("姓"))
    text("firstname", "param.first_name", tr("名"))
    text("nickname", "param.nickname", tr("暱稱"))

    sex = par.get("sex")
    if sex is not None:
        p["param.sex"] = 1 if int(sex) == 0 else 0     # Origin: 0 = male
        took.append(tr("性別"))
    for key, dst, lo, hi, label in (("birthMonth", "param.birth_month", 1, 12, tr("生月")),
                                    ("birthDay", "param.birth_day", 1, 31, tr("生日"))):
        v = par.get(key)
        if v is not None:
            p[dst] = int(np.clip(int(v), lo, hi))
            took.append(label)
    bt = par.get("bloodType")
    if bt is not None and 0 <= int(bt) < len(BLOOD):
        p["param.blood"] = int(bt)
        took.append(tr("血型"))
    pers = par.get("personality")
    if pers is not None:
        # personality IDs are game-specific; keep the index inside our range
        p["param.personality"] = int(pers) % len(P.PERSONALITIES)
        # …but keep the original too.  Origin ships **46** personalities (including
        # negative ids for the special characters) and the H expression tables
        # are keyed by that number — ``HFaceList_c{personality}_*`` — so the
        # ``% 30`` above would silently pick another character's faces.  This
        # key is deliberately not in ``params.py``'s schema: it is a source
        # value we carry, not a slider (§7.131).
        p["param.origin_personality"] = int(pers)
        took.append(tr("個性（依編號對應）"))

    # ---- colours -----------------------------------------------------
    def colour(src, dst, label):
        c = _rgb(src)
        if c:
            p[dst] = list(c)
            took.append(label)
            return True
        return False

    makeup = face.get("baseMakeup") if isinstance(face.get("baseMakeup"), dict) else {}
    colour(body.get("skinMainColor"), "body.skin_color", tr("膚色"))
    if colour(makeup.get("cheekColor"), "face.blush_color", tr("腮紅色")):
        alpha = makeup.get("cheekColor")
        if isinstance(alpha, list) and len(alpha) >= 4:
            p["face.blush"] = float(np.clip(float(alpha[3]), 0.0, 1.0))

    pupil = face.get("pupil")
    if isinstance(pupil, list) and pupil and isinstance(pupil[0], dict):
        colour(pupil[0].get("baseColor"), "face.iris_color", tr("瞳色"))
        # Illusion layers subColor over baseColor through a gradient mask; the
        # mask itself is a game texture, but the two colours and the blend
        # amount are in the card, and reproduce most of the look
        colour(pupil[0].get("subColor"), "face.iris_grad_color", tr("瞳孔漸層色"))
        # ``gradBlend`` is ``create_eye``'s ``_Blend`` and **not** clamped:
        # ``ChangeSettingEye`` hands it to ``SetFloat`` raw and 99 local cards
        # sit outside [0, 1] (§7.345).  Clipping it here made the parameter a
        # different number from the one the eye is composited with.
        gb = _num(pupil[0].get("gradBlend"))
        if gb is not None:
            p["face.iris_gradient"] = float(gb)
            took.append(tr("瞳孔漸層強度"))

    parts = hair.get("parts")
    if isinstance(parts, list) and parts and isinstance(parts[0], dict):
        first = parts[0]
        colour(first.get("baseColor"), "hair.color", tr("髮色"))

    # ---- numeric fields with the same meaning on both sides -----------
    sv = body.get("shapeValueBody")
    if isinstance(sv, list) and sv:
        h = _num(sv[0])
        if h is not None:
            # shapeValueBody entries are plain 0..1 maker sliders, not the
            # -1..1-with-0-neutral encoding some other Illusion fields use: no
            # shipped card has a negative entry and 1.0 does occur.  Rescaling
            # them as if they were -1..1 mapped every card into the middle of
            # our height range, so tall and short characters came out alike.
            #
            # The slider is not linear in metres either — it drives a baked
            # curve on ``cf_n_height``.  Read that curve when the asset library
            # is around so the parameter lands on the height the grafted
            # armature will actually be built at; without it, fall back to the
            # raw slider rather than refusing to import.
            p["body.height"] = float(np.clip(h, 0.0, 1.0))
            metres = _origin_height_metres(card)
            if metres is not None:
                from ..mesh.body import height_to_slider
                p["body.height"] = height_to_slider(metres)
                # keep the displayed height honest — it is read straight off
                # the same curve the armature is grafted at
                p["param.height_cm"] = int(round(metres * 100.0))
            took.append(tr("身高"))
    for key, dst, label in (("bustSoftness", "body.bust_softness", tr("胸部柔軟")),
                            ("bustWeight", "body.bust_angle_v", tr("胸部下垂")),
                            ("skinGlossPower", "body.skin_gloss", tr("皮膚光澤"))):
        v = _num(body.get(key))
        if v is not None:
            p[dst] = float(np.clip(v, 0.0, 1.0))
            took.append(label)

    # ---- recognised but not applicable -------------------------------
    for key, label in (("headId", N_("臉部模型編號")), ("skinId", N_("膚質貼圖編號")),
                       ("detailId", N_("臉部細節貼圖")), ("eyebrowId", N_("眉毛樣式編號")),
                       ("eyelineUpId", N_("眼線樣式編號")), ("noseId", N_("鼻子模型編號")),
                       ("hlUpId", N_("眼睛高光貼圖")), ("whiteId", N_("眼白貼圖")),
                       ("moleId", N_("痣貼圖")),
                       # dressing reads these off the card itself; the maker
                       # rows they used to fill moved nothing (§7.550)
                       ("eyebrowColor", N_("眉色")), ("eyelineColor", N_("眼線／睫毛色")),
                       ("whiteBaseColor", N_("眼白色")), ("pupilWidth", N_("眼寬")),
                       ("pupilHeight", N_("眼高"))):
        if key in face:
            skipped.append(tr(label))
    if "lipColor" in makeup:
        skipped.append(tr("唇色"))
    for key, label in (("nipId", N_("乳暈貼圖")), ("paintId", N_("彩繪貼圖")),
                       ("sunburnId", N_("曬痕貼圖")), ("underhairId", N_("體毛貼圖")),
                       ("typeBone", N_("骨骼類型"))):
        if key in body:
            skipped.append(tr(label))
    if isinstance(parts, list):
        skipped.append(tr("髮型編號（{n} 個部位，引用原作素材）").format(n=len(parts)))
    if isinstance(sv, list) and len(sv) > 1:
        skipped.append(tr("體型滑桿陣列（{n} 項，索引語意屬原作）").format(n=len(sv)))
    svf = face.get("shapeValueFace")
    if isinstance(svf, list):
        skipped.append(tr("臉部滑桿陣列（{n} 項，索引語意屬原作）").format(n=len(svf)))
    if "Coordinate" in card.sections:
        skipped.append(tr("全部服裝與飾品（引用原作素材編號）"))
    if "KKEx" in card.sections:
        ex = card.sections["KKEx"]
        if isinstance(ex, dict) and ex:
            skipped.append(tr("外掛擴充資料（{n} 個外掛）").format(n=len(ex)))

    # this project only accepts adult, anime-styled characters
    p["param.age"] = max(18, int(p.get("param.age", 18) or 18))
    p["param.style"] = "anime"
    P.normalise(p)
    return p, took, skipped


def import_card(path: Path, params: dict | None = None):
    """Read a Origin card and convert it. Returns (params, report dict)."""
    card = read_origin_card(path)
    p, took, skipped = convert(card, params)
    report = {
        "marker": card.marker,
        "version": card.version,
        "sections": sorted(card.sections),
        "imported": took,
        "skipped": skipped,
        "name": f"{p.get('param.last_name','')} {p.get('param.first_name','')}".strip(),
    }
    return p, report


def is_origin_card(path: Path) -> bool:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return False
    return any(m.encode("utf-8") in data for m in ORIGIN_MARKERS)


def describe(report: dict) -> str:
    lines = [tr("來源：{marker} v{version}").format(marker=report["marker"], version=report["version"]),
             tr("區段：{sections}").format(sections=", ".join(report["sections"]) or "—"),
             "",
             tr("已匯入 {n} 項：").format(n=len(report["imported"])) + tr("、").join(report["imported"])]
    if report["skipped"]:
        lines += ["",
                  tr("以下項目引用原作素材／滑桿索引，本作無對應資料，維持預設："),
                  "・" + "\n・".join(report["skipped"])]
    return "\n".join(lines)
