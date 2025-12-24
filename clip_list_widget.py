"""Clip list widget for displaying and selecting clips."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QMenu, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPoint
from PyQt6.QtGui import QFont

from clip_model import Clip
from time_utils import format_timestamp
from srt_viewer import SRTViewer
from design.icon_manager import IconManager
from design.style_manager import StyleManager, Colors


class ClipListWidget(QWidget):
    """Left panel widget displaying list of clips."""

    clip_selected = pyqtSignal(int)  # Emitted when clip is selected (index)
    new_clip_requested = pyqtSignal()  # Emitted when New Clip button clicked
    delete_clip_requested = pyqtSignal(int)  # Emitted when Delete button clicked (index)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clips: list[Clip] = []
        self.current_clip_index: int = -1
        self.srt_viewer: SRTViewer | None = None  # Will be set by parent
        self._init_ui()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Clip list with context menu
        self.clip_list = QListWidget()
        self.clip_list.itemSelectionChanged.connect(self._on_clip_selected)
        self.clip_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.clip_list.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.clip_list)

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
        import logging
        logger = logging.getLogger("ai_videoclipper")

        # Calculate duration in seconds
        duration_seconds = clip.end_time - clip.start_time

        # Format: "Clip 1 (45s)\n00:15 - 00:42\nSegments 5-12"
        start_time = format_timestamp(clip.start_time)
        end_time = format_timestamp(clip.end_time)
        segment_range = f"Segments {clip.segment_start_index + 1}-{clip.segment_end_index + 1}"

        text = f"Clip {index + 1} ({duration_seconds:.0f}s)\n{start_time} – {end_time}\n{segment_range}"

        # Add clip description/name if available (from Claude or manual input)
        if clip.text and clip.text.strip() and clip.text != "Full Video":
            text += f"\n\n{clip.text}"
        else:
            # Fallback: show start and end segment texts
            start_text = ""
            end_text = ""
            if self.srt_viewer:
                start_seg = self.srt_viewer.get_segment_at_index(clip.segment_start_index)
                end_seg = self.srt_viewer.get_segment_at_index(clip.segment_end_index)
                if start_seg:
                    start_text = start_seg.text[:50]  # Limit to 50 chars
                if end_seg:
                    end_text = end_seg.text[-50:]  # Last 50 chars, right-aligned

            if start_text:
                text += f"\n{start_text}"
            if end_text:
                text += f"\n...\n{end_text}"

        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, index)  # Store clip index

        # Set variable height for each row (adjust based on content)
        height = self.get_item_height()
        item.setSizeHint(QSize(200, height))

        self.clip_list.addItem(item)
        logger.debug(f"[CLIP_LIST] Added clip {index + 1} with description")

    def get_item_height(self) -> int:
        """Get height of a single clip row."""
        # Approximate: font height * 7-8 lines + padding (for clip description)
        return 140

    def _on_clip_selected(self):
        """Handle clip selection."""
        import logging
        logger = logging.getLogger("ai_videoclipper")

        current_item = self.clip_list.currentItem()
        if current_item is None:
            logger.info("[CLIP_LIST] No item selected")
            self.current_clip_index = -1
            return

        try:
            self.current_clip_index = current_item.data(Qt.ItemDataRole.UserRole)
            logger.info(f"[CLIP_LIST] Clip selected at index {self.current_clip_index}")
            self.clip_selected.emit(self.current_clip_index)
        except Exception as e:
            logger.error(f"[CLIP_LIST] Error in _on_clip_selected: {e}", exc_info=True)
            raise

    def _on_context_menu(self, position: QPoint):
        """Show context menu for clip operations."""
        current_item = self.clip_list.itemAt(position)

        menu = QMenu()

        # Always allow "New Clip"
        new_action = menu.addAction("New Clip")
        new_action.triggered.connect(self.new_clip_requested.emit)

        # Only allow "Delete Clip" if a clip is selected
        if current_item is not None and self.current_clip_index >= 0:
            menu.addSeparator()
            delete_action = menu.addAction("Delete Clip")
            delete_action.triggered.connect(
                lambda: self.delete_clip_requested.emit(self.current_clip_index)
            )

        # Show menu at cursor position
        menu.exec(self.clip_list.mapToGlobal(position))

    def get_current_clip(self) -> Clip | None:
        """Get currently selected clip."""
        if 0 <= self.current_clip_index < len(self.clips):
            return self.clips[self.current_clip_index]
        return None

    def select_clip(self, index: int):
        """Select a clip by index."""
        if 0 <= index < self.clip_list.count():
            self.clip_list.setCurrentRow(index)

    def update_clip_display(self, clip_index: int):
        """Update display of a specific clip (when segment texts change)."""
        import logging
        logger = logging.getLogger("ai_videoclipper")

        if not (0 <= clip_index < len(self.clips)):
            logger.warning(f"[CLIP_LIST] Invalid clip index for update: {clip_index}")
            return

        try:
            clip = self.clips[clip_index]
            start_time = format_timestamp(clip.start_time)
            end_time = format_timestamp(clip.end_time)
            segment_range = f"Segments {clip.segment_start_index + 1}-{clip.segment_end_index + 1}"

            # Get start and end segment texts
            start_text = ""
            end_text = ""
            if self.srt_viewer:
                start_seg = self.srt_viewer.get_segment_at_index(clip.segment_start_index)
                end_seg = self.srt_viewer.get_segment_at_index(clip.segment_end_index)
                if start_seg:
                    start_text = start_seg.text[:50]
                if end_seg:
                    end_text = end_seg.text[-50:]

            text = f"Clip {clip_index + 1}\n{start_time} – {end_time}\n{segment_range}"
            if start_text:
                text += f"\n{start_text}"
            if end_text:
                text += f"\n...\n{end_text}"

            # Update the item
            item = self.clip_list.item(clip_index)
            if item:
                item.setText(text)
                logger.debug(f"[CLIP_LIST] Updated clip {clip_index + 1} display")
        except Exception as e:
            logger.error(f"[CLIP_LIST] Error updating clip display {clip_index}: {e}", exc_info=True)
