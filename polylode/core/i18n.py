"""Which language the screens speak, and the words in it.

The program's strings are written in Traditional Chinese (:data:`SOURCE`);
each shown string goes through :func:`tr`, which gives it in the language
chosen.  English is the default (owner, 2026-10-16); a system whose
interface language is Chinese gets the Chinese, unless a language was picked
in the settings (``UserData/config/ui.json``).  The language is read once, at
start-up: a screen is built in one language, so a change shows on the next
start.

A string with no English entry shows as written: ``tests/test_i18n.py``
finds every ``tr("…")`` and ``N_("…")`` in the package and fails on one the
table lacks.  A string kept in a constant is marked with :func:`N_` where it
is written and shown with ``tr(CONSTANT)`` where it is used.  A string with
values in it is a template: ``tr("{n} 片").format(n=count)``.  One word with two meanings
takes a context: ``tr("顯示", "settings page")``.
"""
from __future__ import annotations

import locale
import os
import sys

#: the language the program's strings are written in
SOURCE = "zh-TW"
ENGLISH = "en"
#: follow the system's interface language
AUTO = "auto"
#: the languages a screen can speak, by code, each named in itself
LANGUAGES = {ENGLISH: "English", SOURCE: "繁體中文"}

_current: list[str] = []


def _config():
    from .paths import Config
    return Config("ui")


def chosen() -> str:
    """The language picked in the settings: a code of :data:`LANGUAGES`, or :data:`AUTO`."""
    value = str(_config().get("language", AUTO))
    return value if value in LANGUAGES or value == AUTO else AUTO


def choose(language: str) -> None:
    """Keep *language* (a code of :data:`LANGUAGES` or :data:`AUTO`) for the next start."""
    if language not in LANGUAGES and language != AUTO:
        raise ValueError(f"no such language: {language}")
    _config().set("language", language)


def system_language() -> str:
    """The system's interface language as a code of :data:`LANGUAGES`: Chinese, or else English."""
    name = ""
    if sys.platform == "win32":
        try:
            import ctypes
            name = locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage(), "")
        except (AttributeError, OSError):
            name = ""
    if not name:
        for key in ("LC_ALL", "LC_MESSAGES", "LANG"):
            if os.environ.get(key):
                name = os.environ[key]
                break
    if not name:
        name = locale.getlocale()[0] or ""
    return SOURCE if name.lower().startswith("zh") else ENGLISH


def resolve(picked: str, system: str) -> str:
    """The language spoken: *picked* unless it is :data:`AUTO`, then *system*."""
    return system if picked == AUTO else picked


def current() -> str:
    """The language the screens speak this run (read once)."""
    if not _current:
        _current.append(resolve(chosen(), system_language()))
    return _current[0]


def N_(text: str) -> str:
    """*text* as written: marks a string kept in a constant that is shown through :func:`tr` later."""
    return text


def tr(text: str, context: str = "") -> str:
    """*text* (written in :data:`SOURCE`) in the language spoken; as written when there is no entry.

    *context* tells apart one Chinese word with two meanings: ``("settings page", "顯示")`` is
    "Display" where the bare ``"顯示"`` is the option "Shown".  Without an entry for the pair the
    word's own entry answers.
    """
    if current() == SOURCE:
        return text
    from .lang_en import WORDS
    if context and (context, text) in WORDS:
        return WORDS[(context, text)]
    return WORDS.get(text, text)
