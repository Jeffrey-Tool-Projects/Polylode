"""The progress of a long step, shown under what it belongs to: a scan, a preview's conversion, an export.

A job reports through its ``PROGRESS`` lines (``jobs.ProgressLines``): a counted step fills the bar, a step
that cannot be counted (one converter call, Blender's read-back) keeps it moving, and the step's name is
written above it.  Hidden while nothing runs.
"""
from __future__ import annotations

from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from . import jobs
from .i18n import tr


class StepBar(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        self.label = QLabel()
        self.label.setObjectName("muted")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.bar = QProgressBar()
        # a moving bar (range 0..0) shows no text, so the step's name is the label's, the count the bar's
        self.bar.setFormat("%v / %m  (%p%)")
        layout.addWidget(self.bar)
        self.text = ""
        self.hide()

    def start(self, text: str) -> None:
        """A step begins and nothing is counted yet: the bar moves."""
        self.text = text
        self.label.setText(tr(text))
        self.bar.setRange(0, 0)
        self.show()

    def report(self, done: int, total: int, text: str = "") -> None:
        if text and text != self.text:
            self.text = text
            self.label.setText(tr(text))
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(max(0, min(done, total)))
        else:
            self.bar.setRange(0, 0)
        self.show()

    def take(self, line: str) -> bool:
        """*line* of a job's output, shown when it is a ``PROGRESS`` line; True when it was one."""
        kind, _, rest = line.partition(" ")
        if kind != "PROGRESS":
            return False
        reported = jobs.read_progress(rest)
        if reported is not None:
            self.report(*reported)
        return True

    def stop(self) -> None:
        self.hide()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.text = ""
