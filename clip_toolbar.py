"""Toolbar for clip editing operations."""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal


class ClipToolbar(QWidget):
    """Toolbar with buttons for clip editing operations."""

    set_start_clicked = pyqtSignal()
    set_end_clicked = pyqtSignal()
    duplicate_clicked = pyqtSignal()
    split_clicked = pyqtSignal()
    export_all_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize toolbar UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Create buttons
        btn_set_start = QPushButton("📍 Set Start")
        btn_set_start.clicked.connect(self.set_start_clicked.emit)
        layout.addWidget(btn_set_start)

        btn_set_end = QPushButton("📍 Set End")
        btn_set_end.clicked.connect(self.set_end_clicked.emit)
        layout.addWidget(btn_set_end)

        layout.addSpacing(10)

        btn_duplicate = QPushButton("⧉ Duplicate")
        btn_duplicate.clicked.connect(self.duplicate_clicked.emit)
        layout.addWidget(btn_duplicate)

        btn_split = QPushButton("➗ Split")
        btn_split.clicked.connect(self.split_clicked.emit)
        layout.addWidget(btn_split)

        layout.addSpacing(10)

        btn_export_all = QPushButton("💾 Export All")
        btn_export_all.clicked.connect(self.export_all_clicked.emit)
        layout.addWidget(btn_export_all)

        layout.addStretch()
