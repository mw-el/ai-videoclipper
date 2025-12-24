"""Toolbar for clip editing operations."""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal, QSize
from design.icon_manager import IconManager
from design.style_manager import StyleManager, Colors


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
        layout.setSpacing(10)

        # Note: Start, End, Duplicate, Split buttons are now in the left edit toolbar
        # This toolbar now only contains Load, Save, and Run buttons

        # Stretch to push load/save/run to the right side
        layout.addStretch()

        # Load Clips button with folder_open icon
        self.btn_load_clips = QPushButton()
        self.btn_load_clips.setIcon(IconManager.create_icon('folder_open', color='white', size=18))
        self.btn_load_clips.setIconSize(QSize(18, 18))
        self.btn_load_clips.setToolTip("Load clips configuration from file")
        self.btn_load_clips.clicked.connect(self.load_config_clicked.emit)
        StyleManager.apply_colored_icon_button_style(self.btn_load_clips, Colors.BRIGHT_BLUE)
        self.btn_load_clips.setMinimumWidth(StyleManager.BUTTON_MIN_SIZE)
        self.btn_load_clips.setMinimumHeight(StyleManager.BUTTON_MIN_SIZE)
        layout.addWidget(self.btn_load_clips)

        # Save Clips button with save icon
        self.btn_save_clips = QPushButton()
        self.btn_save_clips.setIcon(IconManager.create_icon('save', color='white', size=18))
        self.btn_save_clips.setIconSize(QSize(18, 18))
        self.btn_save_clips.setToolTip("Save clips configuration to file")
        self.btn_save_clips.clicked.connect(self.save_config_clicked.emit)
        StyleManager.apply_colored_icon_button_style(self.btn_save_clips, Colors.GREEN)
        self.btn_save_clips.setMinimumWidth(StyleManager.BUTTON_MIN_SIZE)
        self.btn_save_clips.setMinimumHeight(StyleManager.BUTTON_MIN_SIZE)
        layout.addWidget(self.btn_save_clips)

        # Run button with movie icon (film strip)
        self.btn_run = QPushButton()
        self.btn_run.setIcon(IconManager.create_icon('movie', color='white', size=18))
        self.btn_run.setIconSize(QSize(18, 18))
        self.btn_run.setToolTip("Export all clips")
        self.btn_run.clicked.connect(self.export_all_clicked.emit)
        StyleManager.apply_colored_icon_button_style(self.btn_run, Colors.BRIGHT_GREEN)
        self.btn_run.setMinimumWidth(StyleManager.BUTTON_MIN_SIZE)
        self.btn_run.setMinimumHeight(StyleManager.BUTTON_MIN_SIZE)
        layout.addWidget(self.btn_run)
