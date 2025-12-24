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
    load_config_clicked = pyqtSignal()
    save_config_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize toolbar UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Create buttons (left side) - all same height
        btn_set_start = QPushButton("Start")
        btn_set_start.clicked.connect(self.set_start_clicked.emit)
        btn_set_start.setMaximumWidth(50)
        btn_set_start.setMinimumHeight(32)
        layout.addWidget(btn_set_start)

        btn_set_end = QPushButton("End")
        btn_set_end.clicked.connect(self.set_end_clicked.emit)
        btn_set_end.setMaximumWidth(50)
        btn_set_end.setMinimumHeight(32)
        layout.addWidget(btn_set_end)

        btn_duplicate = QPushButton("⧉")
        btn_duplicate.clicked.connect(self.duplicate_clicked.emit)
        btn_duplicate.setMaximumWidth(32)
        btn_duplicate.setMinimumHeight(32)
        layout.addWidget(btn_duplicate)

        btn_split = QPushButton("➗")
        btn_split.clicked.connect(self.split_clicked.emit)
        btn_split.setMaximumWidth(32)
        btn_split.setMinimumHeight(32)
        layout.addWidget(btn_split)

        # Stretch to push load/save to the right side
        layout.addStretch()

        btn_load_scenes = QPushButton("Load Clips")
        btn_load_scenes.clicked.connect(self.load_config_clicked.emit)
        btn_load_scenes.setMaximumWidth(70)
        btn_load_scenes.setMinimumHeight(32)
        layout.addWidget(btn_load_scenes)

        btn_save_scenes = QPushButton("Save Clips")
        btn_save_scenes.clicked.connect(self.save_config_clicked.emit)
        btn_save_scenes.setMaximumWidth(70)
        btn_save_scenes.setMinimumHeight(32)
        layout.addWidget(btn_save_scenes)

        btn_run = QPushButton("▶ Run")
        btn_run.clicked.connect(self.export_all_clicked.emit)
        btn_run.setMaximumWidth(60)
        btn_run.setMinimumHeight(32)
        btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        layout.addWidget(btn_run)
