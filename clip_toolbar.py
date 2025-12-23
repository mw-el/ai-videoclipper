"""Toolbar for clip editing operations."""

from PyQt6.QtWidgets import QToolBar, QLabel, QSpinBox, QComboBox
from PyQt6.QtCore import pyqtSignal, Qt


class ClipToolbar(QToolBar):
    """Toolbar with buttons for clip editing operations."""

    set_start_clicked = pyqtSignal()
    set_end_clicked = pyqtSignal()
    duplicate_clicked = pyqtSignal()
    split_clicked = pyqtSignal()
    export_all_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Clip Tools", parent)
        self._init_actions()

    def _init_actions(self):
        """Initialize toolbar actions."""
        self.addAction("📍 Set Start", self.set_start_clicked.emit)
        self.addAction("📍 Set End", self.set_end_clicked.emit)
        self.addSeparator()
        self.addAction("⧉ Duplicate", self.duplicate_clicked.emit)
        self.addAction("➗ Split", self.split_clicked.emit)
        self.addSeparator()
        self.addAction("💾 Export All", self.export_all_clicked.emit)
