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

        # Note: Start, End, Duplicate, Split buttons are now in the left edit toolbar
        # This toolbar now only contains Load, Save, and Run buttons

        # Stretch to push load/save/run to the right side
        layout.addStretch()

        btn_load_clips = QPushButton("Load Clips")
        btn_load_clips.clicked.connect(self.load_config_clicked.emit)
        btn_load_clips.setMinimumWidth(90)
        btn_load_clips.setMinimumHeight(32)
        layout.addWidget(btn_load_clips)

        btn_save_clips = QPushButton("Save Clips")
        btn_save_clips.clicked.connect(self.save_config_clicked.emit)
        btn_save_clips.setMinimumWidth(90)
        btn_save_clips.setMinimumHeight(32)
        layout.addWidget(btn_save_clips)

        btn_run = QPushButton("▶ Run")
        btn_run.clicked.connect(self.export_all_clicked.emit)
        btn_run.setMaximumWidth(60)
        btn_run.setMinimumHeight(32)
        btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        layout.addWidget(btn_run)
