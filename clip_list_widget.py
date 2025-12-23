"""Clip list widget for displaying and selecting clips."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton,
    QHBoxLayout, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont

from clip_model import Clip
from time_utils import format_timestamp


class ClipListWidget(QWidget):
    """Left panel widget displaying list of clips."""

    clip_selected = pyqtSignal(int)  # Emitted when clip is selected (index)
    new_clip_requested = pyqtSignal()  # Emitted when New Clip button clicked
    delete_clip_requested = pyqtSignal(int)  # Emitted when Delete button clicked (index)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clips: list[Clip] = []
        self.current_clip_index: int = -1
        self._init_ui()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Clip list
        self.clip_list = QListWidget()
        self.clip_list.itemSelectionChanged.connect(self._on_clip_selected)
        layout.addWidget(self.clip_list)

        # Bottom button bar
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(4, 4, 4, 4)
        button_layout.setSpacing(4)

        self.new_clip_btn = QPushButton("➕ New")
        self.new_clip_btn.setMaximumHeight(28)
        self.new_clip_btn.clicked.connect(self.new_clip_requested.emit)
        button_layout.addWidget(self.new_clip_btn)

        self.delete_clip_btn = QPushButton("🗑️ Delete")
        self.delete_clip_btn.setMaximumHeight(28)
        self.delete_clip_btn.clicked.connect(self._on_delete_clicked)
        self.delete_clip_btn.setEnabled(False)
        button_layout.addWidget(self.delete_clip_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def set_clips(self, clips: list[Clip]):
        """Set clips to display in list."""
        import logging
        logger = logging.getLogger("ai_videoclipper")
        logger.info(f"[CLIP_LIST] set_clips() called with {len(clips)} clips")

        self.clips = clips
        self.clip_list.clear()
        self.current_clip_index = -1

        for i, clip in enumerate(clips):
            try:
                self._add_clip_row(i, clip)
                logger.info(f"[CLIP_LIST] Added clip row {i + 1}/{len(clips)}")
            except Exception as e:
                logger.error(f"[CLIP_LIST] Failed to add clip row {i}: {e}", exc_info=True)
                raise

        # Select first clip by default
        if clips:
            logger.info(f"[CLIP_LIST] Setting current row to 0...")
            try:
                self.clip_list.setCurrentRow(0)
                logger.info(f"[CLIP_LIST] ✓ Current row set")
            except Exception as e:
                logger.error(f"[CLIP_LIST] Failed to set current row: {e}", exc_info=True)
                raise

    def _add_clip_row(self, index: int, clip: Clip):
        """Add a clip row to the list."""
        # Format: "Clip 1\n00:15 - 00:42\nSegments 5-12"
        start_time = format_timestamp(clip.start_time)
        end_time = format_timestamp(clip.end_time)
        segment_range = f"Segments {clip.segment_start_index + 1}-{clip.segment_end_index + 1}"

        text = f"Clip {index + 1}\n{start_time} – {end_time}\n{segment_range}"

        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, index)  # Store clip index

        # Set fixed height for each row (3 lines)
        height = self.get_item_height()
        item.setSizeHint(QSize(200, height))

        self.clip_list.addItem(item)

    def get_item_height(self) -> int:
        """Get height of a single clip row."""
        # Approximate: font height * 3 lines + padding
        return 70

    def _on_clip_selected(self):
        """Handle clip selection."""
        import logging
        logger = logging.getLogger("ai_videoclipper")

        current_item = self.clip_list.currentItem()
        if current_item is None:
            logger.info("[CLIP_LIST] No item selected")
            self.current_clip_index = -1
            self.delete_clip_btn.setEnabled(False)
            return

        try:
            self.current_clip_index = current_item.data(Qt.ItemDataRole.UserRole)
            logger.info(f"[CLIP_LIST] Clip selected at index {self.current_clip_index}")
            self.delete_clip_btn.setEnabled(True)
            self.clip_selected.emit(self.current_clip_index)
        except Exception as e:
            logger.error(f"[CLIP_LIST] Error in _on_clip_selected: {e}", exc_info=True)
            raise

    def _on_delete_clicked(self):
        """Handle delete button click."""
        if self.current_clip_index >= 0:
            self.delete_clip_requested.emit(self.current_clip_index)

    def get_current_clip(self) -> Clip | None:
        """Get currently selected clip."""
        if 0 <= self.current_clip_index < len(self.clips):
            return self.clips[self.current_clip_index]
        return None

    def select_clip(self, index: int):
        """Select a clip by index."""
        if 0 <= index < self.clip_list.count():
            self.clip_list.setCurrentRow(index)
