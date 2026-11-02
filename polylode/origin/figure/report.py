"""Dress result types shared by the figure pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

from ...core.i18n import tr


@dataclass
class Attached:
    name: str
    category: str
    source: str
    tris: int


@dataclass
class DressReport:
    attached: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    replaced: list = field(default_factory=list)
    # Things that make the result unusable rather than merely incomplete —
    # no body, no asset library.  ``missing`` is a list of items that did not
    # resolve, which a card can legitimately have a few of; these are not that.
    errors: list = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.errors.append(reason)
        self.missing.append(reason)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        """One line for a status bar: the error if there is one, else counts."""
        if self.errors:
            return tr("載入失敗：") + tr("；").join(self.errors[:2])
        if not self.attached:
            return tr("沒有可套用的原作素材")
        note = tr("已載入 {count} 件原作素材").format(count=len(self.attached))
        if self.missing:
            note += tr("，{count} 項找不到").format(count=len(self.missing))
        return note

    def describe(self) -> str:
        lines = []
        if self.errors:
            lines.append(tr("載入失敗："))
            for e in self.errors:
                lines.append(f"  ! {e}")
            lines.append("")
        if self.attached:
            lines.append(tr("已套用 {count} 件原作素材：").format(count=len(self.attached)))
            for a in self.attached:
                lines.append(f"  · {a.category:12s} {a.name}  "
                             + tr("({tris:,} 面, {source})").format(tris=a.tris, source=a.source))
        if self.missing:
            lines.append("")
            lines.append(tr("以下 {count} 項找不到：").format(count=len(self.missing)))
            for m in self.missing[:20]:
                lines.append(f"  · {m}")
            if len(self.missing) > 20:
                lines.append("  · " + tr("…另外 {count} 項").format(count=len(self.missing) - 20))
        return "\n".join(lines) or tr("沒有可套用的原作素材。")
