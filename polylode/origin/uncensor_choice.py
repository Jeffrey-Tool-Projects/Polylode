"""Which uncensored body, penis and balls UncensorSelector puts on a card (#93).

Read off ``KK_UncensorSelector.dll`` (``origin_il_dump --dll``):

* The card's plugin block ``com.deathweasel.bepinex.uncensorselector`` names
  ``BodyGUID`` / ``PenisGUID`` / ``BallsGUID`` (each may be null) and
  ``DisplayPenis`` / ``DisplayBalls``.
* ``get_BodyData``: a named GUID that is in ``BodyDictionary`` is used, unless the
  body is for the other sex and ``Genderbender Allowed`` is off (studio skips that
  check); otherwise ``DefaultData.GetDefaultOrRandomBody``.  Penis and balls have
  no sex check.
* ``GetDefaultOrRandom*``: the setting ``Default Female Body`` (and the five like
  it) is a display label; a label that names an entry gives that entry, ``Random``
  (or anything unknown) draws one at random, and when nothing may be drawn the
  built-in ``Default.Body.Female`` / ``Default.Body.Male`` / ``Default.Penis`` /
  ``Default.Balls``.
* ``GetRandom*``: the dictionary's entries in their insertion order — the
  built-ins first (``PopulateUncensorLists``; they do not allow random), then every
  manifest's ``<KK_UncensorSelector>`` entries in Sideloader's load order — kept
  when ``allowRandom`` is not ``false``/``0``, the GUID is not in the
  ``Random Excluded`` setting for the character's sex, and (bodies only) the body
  is for that sex.  The pick is ``new System.Random(seed).Next(count)`` with
  ``seed = birthDay + personality + (int)(voicePitch * 100)``
  (``GetRandomNumber``), ``voicePitch = Lerp(0.94, 1.06, voiceRate)``.
* ``System.Random`` is the game's own Mono one (``Koikatu_Data/Managed/mscorlib.dll``):
  Knuth's subtractive generator with ``inextp`` starting at 31 — .NET's starts at
  21, so a .NET port would pick other bodies.

Sex here is the game's: 0 male, 1 female.
"""
from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

PLUGIN_GUID = "com.deathweasel.bepinex.uncensorselector"
DEFAULT_BODY = {0: "Default.Body.Male", 1: "Default.Body.Female"}
DEFAULT_PENIS = "Default.Penis"
DEFAULT_BALLS = "Default.Balls"
KINDS = ("body", "penis", "balls")

_MBIG = 2147483647
_MSEED = 161803398


class MonoRandom:
    """``System.Random(int seed)`` as the game's Mono ``mscorlib`` implements it."""

    def __init__(self, seed: int, first_inextp: int = 31):
        seeds = [0] * 56
        mj = _MSEED - abs(int(seed))
        seeds[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            seeds[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += _MBIG
            mj = seeds[ii]
        for _k in range(1, 5):
            for i in range(1, 56):
                seeds[i] -= seeds[1 + (i + 30) % 55]
                if seeds[i] < 0:
                    seeds[i] += _MBIG
        self._seeds = seeds
        self._inext = 0
        self._inextp = first_inextp       # 31 in the game's Mono; .NET Framework's is 21

    def internal(self) -> int:
        """The next raw value in ``[0, int.MaxValue)``."""
        self._inext = 1 if self._inext + 1 >= 56 else self._inext + 1
        self._inextp = 1 if self._inextp + 1 >= 56 else self._inextp + 1
        value = self._seeds[self._inext] - self._seeds[self._inextp]
        if value < 0:
            value += _MBIG
        self._seeds[self._inext] = value
        return value

    def sample(self) -> float:
        return self.internal() * 4.656612875245797e-10

    def next(self, max_value: int) -> int:
        """``Next(int maxValue)``: ``(int)(Sample() * maxValue)``."""
        return int(self.sample() * max_value)


def voice_pitch(voice_rate: float) -> float:
    """``ChaFileParameter.voicePitch``: ``Mathf.Lerp(0.94f, 1.06f, voiceRate)``, kept as a float."""
    t = min(1.0, max(0.0, float(voice_rate)))
    lo, hi = np.float32(0.94), np.float32(1.06)
    return float(np.float32(float(lo) + (float(hi) - float(lo)) * t))


def random_seed(parameter: dict) -> int:
    """``DefaultData.GetRandomNumber``'s seed for a card's ``Parameter`` block.

    ``(int)(voicePitch * 100f)`` is not stored before the ``conv.i4``, so the product is taken
    at the runtime's internal precision; this takes it in double, as the old Mono JIT does.
    """
    pitch = voice_pitch(parameter.get("voiceRate", 0.5) or 0.0)
    return int(parameter.get("birthDay", 0) or 0) + int(parameter.get("personality", 0) or 0) \
        + int(pitch * 100.0)


@dataclass(frozen=True)
class Entry:
    """One ``<body>`` / ``<penis>`` / ``<balls>`` of a manifest's ``<KK_UncensorSelector>``."""
    kind: str
    guid: str
    display: str
    sex: int | None            # bodies only: 1 female, 0 male
    allow_random: bool
    bundle: str                # ``abdata``-relative file, "" for the built-ins
    asset: str
    archive: str = ""          # the zipmod it came from, "" for the built-ins
    main_tex: str = ""         # a body's ``<oo_base><mainTex>`` ("" when not declared)
    color_mask: str = ""       # a body's ``<oo_base><colorMask>``
    additional: tuple = ()     # a body's ``<additionalPart>`` renderer names (``BodyData.AdditionalParts``)
    #: ``<colorMatch>`` rows (``ColorMatchPart``): ``(object, materialCreate, mainTex, colorMask)``
    color_match: tuple = ()
    normals: str = ""          # a body's ``<oo_base><normals>`` (``BodyData.Normals``; "" = the plugin's default)
    uncensor_overlay: str = ""     # a body's ``<oo_base><uncensorOverlay>`` (a ``BodyOver`` skin layer)
    uncensor_underlay: str = ""    # a body's ``<oo_base><uncensorUnderlay>`` (a ``BodyUnder`` skin layer)

    @property
    def label(self) -> str:
        """The settings' display label (``PopulateUncensorLists``)."""
        if self.kind == "body":
            return f"[{'Female' if self.sex == 1 else 'Male'}] {self.display}"
        return self.display


def _flag(text: str | None) -> bool:
    """``allowRandom``: off only for ``false`` / ``0``, lower-cased and not trimmed (``BodyData..ctor``)."""
    return (text or "").lower() not in ("false", "0")


def manifest_entries(xml: bytes, archive: str = "") -> list[Entry]:
    """The entries a zipmod manifest declares, in document order."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    section = root.find("KK_UncensorSelector")
    out: list[Entry] = []
    if section is None:
        return out
    for kind in KINDS:
        for item in section.findall(kind):
            guid = (item.findtext("guid") or "").strip()
            if not guid:
                continue
            files = item.find("oo_base") if kind == "body" else item
            sex = None
            if kind == "body":
                sex = 1 if (item.findtext("sex") or "").lower() == "female" else 0
            text = (lambda tag: ((files.findtext(tag) if files is not None else "") or "").strip())
            out.append(Entry(kind, guid, (item.findtext("displayName") or guid).strip(), sex,
                             _flag(item.findtext("allowRandom")), text("file"), text("asset"), archive,
                             text("mainTex") if kind == "body" else "",
                             text("colorMask") if kind == "body" else "",
                             tuple(t.text.strip() for t in item.iter("additionalPart") if t.text and t.text.strip())
                             if kind == "body" else (),
                             tuple(tuple((c.findtext(k) or "").strip()
                                         for k in ("object", "materialCreate", "mainTex", "colorMask"))
                                   for c in item.iter("colorMatch")),
                             text("normals") if kind == "body" else "",
                             text("uncensorOverlay") if kind == "body" else "",
                             text("uncensorUnderlay") if kind == "body" else ""))
    return out


def _built_ins() -> list[Entry]:
    return [Entry("body", DEFAULT_BODY[0], "Default Body M", 0, False, "", ""),
            Entry("body", DEFAULT_BODY[1], "Default Body F", 1, False, "", ""),
            Entry("penis", DEFAULT_PENIS, "Mosaic Penis", None, False, "", ""),
            Entry("balls", DEFAULT_BALLS, "Mosaic Balls", None, False, "", "")]


@dataclass
class Catalog:
    """Every entry, per kind, in ``BodyDictionary`` order; a later GUID replaces an earlier one in place."""
    entries: dict = field(default_factory=lambda: {k: {} for k in KINDS})
    #: ``MigrationDictionary``: a version-1 card's ``UncensorGUID`` → ``(body, penis, balls)`` GUIDs
    migrations: dict = field(default_factory=dict)

    def add(self, entry: Entry) -> None:
        self.entries[entry.kind][entry.guid] = entry

    def get(self, kind: str, guid: str | None) -> Entry | None:
        return self.entries[kind].get(guid) if guid else None


def manifest_migrations(xml: bytes) -> dict:
    """``{guidUncensor: (guidBody, guidPenis, guidBalls)}`` of a manifest's ``<migration>`` rows (``MigrationData``).

    ``PopulateUncensorLists`` skips a row without ``guidUncensor`` or ``guidBody``; the
    other two may be missing (None).
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {}
    section = root.find("KK_UncensorSelector")
    out: dict = {}
    for row in (section.findall("migration") if section is not None else ()):
        old, body = row.get("guidUncensor"), row.get("guidBody")
        if old and body:
            out[old] = (body, row.get("guidPenis") or None, row.get("guidBalls") or None)
    return out


def catalog_of(archives) -> Catalog:
    """The catalog the given ``[(guid, zipmod path)]`` (Sideloader's load order) give."""
    cat = Catalog()
    for e in _built_ins():
        cat.add(e)
    for _guid, archive in archives:
        try:
            with zipfile.ZipFile(archive) as z:
                name = next((n for n in z.namelist() if n.lower() == "manifest.xml"), None)
                xml = z.read(name) if name else b""
        except (OSError, zipfile.BadZipFile, KeyError):
            continue
        for e in manifest_entries(xml, str(archive)):
            cat.add(e)
        cat.migrations.update(manifest_migrations(xml))
    return cat


def installed_catalog(mods_dir: Path) -> Catalog:
    """:func:`catalog_of` the zipmods Sideloader loads from *mods_dir*."""
    from .sideloader import loaded_archives
    return catalog_of(loaded_archives(Path(mods_dir)))


@dataclass(frozen=True)
class Settings:
    """The player's UncensorSelector settings (``BepInEx/config``); the plugin's own defaults when unset."""
    genderbender: bool = True
    defaults: dict = field(default_factory=dict)       # (kind, sex) -> display label
    excluded: dict = field(default_factory=dict)       # (kind, sex) -> frozenset of GUIDs
    female_balls: bool = False                         # ``Default Female Balls Display``

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Settings":
        """From :func:`plugins.parse_cfg` of ``com.deathweasel.bepinex.uncensorselector.cfg``."""
        defaults, excluded = {}, {}
        for sex, word in ((0, "Male"), (1, "Female")):
            for kind in KINDS:
                defaults[(kind, sex)] = cfg.get(f"Config.Default {word} {kind.title()}", "Random")
                raw = cfg.get(f"Random Excluded.{word} {kind.title()}", "")
                excluded[(kind, sex)] = frozenset(
                    s.strip() for s in raw.replace(";", ",").split(",") if s.strip())
        return cls(cfg.get("Config.Genderbender Allowed", "true").strip().lower() == "true",
                   defaults, excluded,
                   cfg.get("Config.Default Female Balls Display", "false").strip().lower() == "true")


@dataclass(frozen=True)
class CardChoice:
    """What a card's plugin block names (``UncensorSelectorController.OnReload``); None where it says nothing."""
    body: str | None = None
    penis: str | None = None
    balls: str | None = None
    display_penis: bool | None = None
    display_balls: bool | None = None


def card_choice(card, migrations: dict | None = None) -> CardChoice:
    """The card's own choice; all unnamed when it carries no block.

    ``OnReload`` reads a version-1 block for ``DisplayBalls`` and a legacy
    ``UncensorGUID``, which ``MigrationDictionary`` (*migrations*,
    ``Catalog.migrations``) turns into the three GUIDs — a migrated penis
    shows (``DisplayPenis`` true); an unknown one names nothing.  Any other
    version is read for the three GUIDs and both display flags, each when
    present.
    """
    block = (getattr(card, "sections", {}).get("KKEx") or {}).get(PLUGIN_GUID)
    version = block[0] if isinstance(block, list) and block else None
    data = block[1] if isinstance(block, list) and len(block) > 1 else block
    if not isinstance(data, dict):
        return CardChoice()
    name = lambda k: (str(data[k]) if data.get(k) else None)  # noqa: E731
    flag = lambda k: (bool(data[k]) if k in data and data[k] is not None else None)  # noqa: E731
    if version == 1:
        hit = (migrations or {}).get(str(data.get("UncensorGUID") or "").strip()) if data.get("UncensorGUID") else None
        if hit is None:
            return CardChoice(display_balls=flag("DisplayBalls"))
        body, penis, balls = hit
        return CardChoice(body, penis, balls, True if penis else None, flag("DisplayBalls"))
    return CardChoice(name("BodyGUID"), name("PenisGUID"), name("BallsGUID"),
                      flag("DisplayPenis"), flag("DisplayBalls"))


def _default_or_random(cat: Catalog, settings: Settings, kind: str, sex: int, seed: int) -> Entry | None:
    label = settings.defaults.get((kind, sex), "Random")
    named = next((e for e in cat.entries[kind].values() if e.label == label), None)
    if named is not None:
        return named
    excluded = settings.excluded.get((kind, sex), frozenset())
    pool = [e for e in cat.entries[kind].values()
            if e.allow_random and e.guid not in excluded and (kind != "body" or e.sex == sex)]
    if pool:
        return pool[MonoRandom(seed).next(len(pool))]
    fallback = DEFAULT_BODY[sex] if kind == "body" else DEFAULT_PENIS if kind == "penis" else DEFAULT_BALLS
    return cat.get(kind, fallback)


def resolve(cat: Catalog, settings: Settings, choice: CardChoice, sex: int, parameter: dict,
            studio: bool = False) -> dict:
    """``{kind: Entry}`` UncensorSelector loads for a character of *sex* whose card names *choice*."""
    seed = random_seed(parameter)
    out = {}
    for kind in KINDS:
        hit = cat.get(kind, getattr(choice, kind))
        if hit is not None and kind == "body" and not studio and not settings.genderbender and hit.sex != sex:
            hit = None
        out[kind] = hit if hit is not None else _default_or_random(cat, settings, kind, sex, seed)
    return out


def shown(choice: CardChoice, sex: int, settings: Settings) -> tuple[bool, bool]:
    """``(shaft, balls)`` drawn on a character of *sex* (the game's: 0 man) whose card says *choice*.

    ``OnReload`` starts ``DisplayPenis`` at "is a man" and ``DisplayBalls`` at
    true for a man and ``Default Female Balls Display`` for a woman; the
    card's own flags replace them when it has them.  ``ReloadCharacterPenis``
    sets ``visibleSonAlways`` from the first, which ``UpdateVisible`` makes
    the ``S_Son`` object (``n_dankon``) active by — and ``o_dan_f`` hangs
    under it, so the second (``ReloadCharacterBalls``: that object's own
    active state) shows nothing without the first.  Nothing in the H scene
    sets ``visibleSonAlways`` back.
    """
    penis = choice.display_penis if choice.display_penis is not None else sex == 0
    balls = choice.display_balls if choice.display_balls is not None else (sex == 0 or settings.female_balls)
    return penis, penis and balls


def worn(cat: Catalog, settings: Settings, choice: CardChoice, sex: int, parameter: dict,
         studio: bool = False) -> dict:
    """:func:`resolve`, with a woman's penis and balls kept only where she shows them (None otherwise).

    A man's stay as resolved — they name his archive whatever is drawn;
    what is drawn is :func:`shown`.
    """
    out = resolve(cat, settings, choice, sex, parameter, studio)
    if sex == 1:
        penis, balls = shown(choice, sex, settings)
        if not penis:
            out["penis"] = None
        if not balls:
            out["balls"] = None
    return out


def shown_for_card(card, sex: int) -> tuple[bool, bool]:
    """:func:`shown` for *card*; a man's both and a woman's neither when the install runs no UncensorSelector."""
    got = selector()
    if got is None:
        return sex == 0, sex == 0
    return shown(card_choice(card, got[0].migrations), sex, got[1])


#: Where the converter keeps the catalog and settings (``bake.write_named_table``, source ``bake.INSTALL``)
TABLE = ("uncensor_selector", "catalog")
_CACHE: dict = {}


def to_payload(value) -> dict | None:
    """``(Catalog, Settings)`` as JSON; None (the install runs no UncensorSelector) stays None."""
    if value is None:
        return None
    cat, settings = value
    return {"entries": [asdict(e) for kind in KINDS for e in cat.entries[kind].values()],
            "migrations": [[k, *v] for k, v in cat.migrations.items()],
            "genderbender": settings.genderbender, "female_balls": settings.female_balls,
            "defaults": [[k, s, v] for (k, s), v in settings.defaults.items()],
            "excluded": [[k, s, sorted(v)] for (k, s), v in settings.excluded.items()]}


def from_payload(data: dict) -> tuple[Catalog, Settings]:
    """:func:`to_payload` back, the entries in their order."""
    cat = Catalog()
    for row in data.get("entries") or []:
        cat.add(Entry(**{**row, "additional": tuple(row.get("additional") or ()),
                         "color_match": tuple(tuple(r) for r in row.get("color_match") or ())}))
    cat.migrations = {r[0]: tuple(r[1:4]) for r in data.get("migrations") or []}
    return cat, Settings(bool(data.get("genderbender", True)),
                         {(k, int(s)): v for k, s, v in data.get("defaults") or []},
                         {(k, int(s)): frozenset(v) for k, s, v in data.get("excluded") or []},
                         bool(data.get("female_balls", False)))


def selector(from_bundle: bool = False):
    """``(Catalog, Settings)`` of the install, or None when it does not run UncensorSelector.

    The run time reads the converter's table; ``from_bundle`` (the converter, or a
    machine whose bake has no table yet) reads the synced zipmods and settings.
    """
    from . import bake, plugins
    if not from_bundle:
        if "v" in _CACHE:
            return _CACHE["v"]
        recorded, payload = bake.named_table(bake.INSTALL, *TABLE)
        if recorded:
            _CACHE["v"] = None if payload is None else from_payload(payload)
            return _CACHE["v"]
    if not plugins.installed(PLUGIN_GUID):
        value = None
    else:
        from .sideloader import mods_search_roots
        root = next((r for r in mods_search_roots() if r.is_dir()), None)
        cat = installed_catalog(root) if root is not None else catalog_of([])
        value = (cat, Settings.from_cfg(plugins.settings(PLUGIN_GUID)))
    if not from_bundle:
        _CACHE["v"] = value
    return value


def all_additional_parts(from_bundle: bool = False) -> frozenset:
    """``AllAdditionalParts``: every body's ``<additionalPart>`` names — the parts a body without them destroys."""
    got = selector(from_bundle)
    if got is None:
        return frozenset()
    return frozenset(n for e in got[0].entries["body"].values() for n in e.additional)


def for_card(card, sex: int, studio: bool = False) -> dict | None:
    """:func:`resolve` for *card* (a ``cardimport.OriginCard``) as a character of the game's *sex*;
    None when the install runs no UncensorSelector."""
    got = selector()
    if got is None:
        return None
    cat, settings = got
    return worn(cat, settings, card_choice(card, cat.migrations), sex, getattr(card, "parameter", {}) or {}, studio)
