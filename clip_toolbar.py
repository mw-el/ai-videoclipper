"""Toolbar for clip editing operations."""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal, QSize
from design.icon_manager import IconManager
from design.style_manager import StyleManager


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
        layout.setSpacing(6)

        # Note: Start, End, Duplicate, Split buttons are now in the left edit toolbar
        # This toolbar now only contains Load, Save, and Run buttons

        # Stretch to push load/save/run to the right side
        layout.addStretch()

        # Load Clips button with folder_open icon
        self.btn_load_clips = QPushButton()
        self.btn_load_clips.setIcon(IconManager.create_icon('folder_open', color='#666666', size=18))
        self.btn_load_clips.setIconSize(QSize(18, 18))
        self.btn_load_clips.setToolTip("Load clips configuration from file")
        self.btn_load_clips.clicked.connect(self.load_config_clicked.emit)
        StyleManager.apply_button_style(self.btn_load_clips)
        self.btn_load_clips.setMinimumWidth(36)
        self.btn_load_clips.setMinimumHeight(36)
        layout.addWidget(self.btn_load_clips)

        # Save Clips button with save icon
        self.btn_save_clips = QPushButton()
        self.btn_save_clips.setIcon(IconManager.create_icon('save', color='#666666', size=18))
        self.btn_save_clips.setIconSize(QSize(18, 18))
        self.btn_save_clips.setToolTip("Save clips configuration to file")
        self.btn_save_clips.clicked.connect(self.save_config_clicked.emit)
        StyleManager.apply_button_style(self.btn_save_clips)
        self.btn_save_clips.setMinimumWidth(36)
        self.btn_save_clips.setMinimumHeight(36)
        layout.addWidget(self.btn_save_clips)

        # Run button with play_arrow icon
        self.btn_run = QPushButton()
        self.btn_run.setIcon(IconManager.create_icon('play_arrow', color='white', size=18))
        self.btn_run.setIconSize(QSize(18, 18))
        self.btn_run.setToolTip("Export all clips")
        self.btn_run.clicked.connect(self.export_all_clicked.emit)
        StyleManager.apply_primary_button_style(self.btn_run)
        self.btn_run.setMinimumWidth(40)
        self.btn_run.setMinimumHeight(36)
        layout.addWidget(self.btn_run)
