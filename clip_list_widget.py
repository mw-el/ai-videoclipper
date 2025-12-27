"""Clip list widget for displaying and selecting clips."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QMenu, QPushButton, QLabel
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

        # Custom styling: light green background for selected items instead of red
        self.clip_list.setStyleSheet("""
            QListWidget::item:selected {
                background-color: #c3f0ca;
                color: #000000;
            }
            QListWidget::item:hover {
                background-color: #e8f5e9;
            }
            QListWidget {
                background-color: #ffffff;
                border: none;
            }
        """)

        layout.addWidget(self.clip_list)

        self.setLayout(layout)

    def set_clips(self, clips: list[Clip]):
        """Set clips to display in list."""
        import logging
        logger = logging.getLogger("ai_videoclipper")
        logger.info(f"[CLIP_LIST] set_clips() called with {len(clips)} clips")

        # Sort clips chronologically by start_time
        sorted_clips = sorted(clips, key=lambda c: c.start_time)

        self.clips = sorted_clips
        self.clip_list.clear()
        self.current_clip_index = -1

        for i, clip in enumerate(sorted_clips):
            try:
                self._add_clip_row(i, clip)
                logger.info(f"[CLIP_LIST] Added clip row {i + 1}/{len(sorted_clips)}")
            except Exception as e:
                logger.error(f"[CLIP_LIST] Failed to add clip row {i}: {e}", exc_info=True)
                raise

        # Select first clip by default
        if sorted_clips:
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

        # Format timestamps and segment info
        start_time = format_timestamp(clip.start_time)
        end_time = format_timestamp(clip.end_time)

        # Combine segment range with time to save space: "Seg 5-12: 00:15 - 00:42"
        segment_nums = f"Seg {clip.segment_start_index + 1}-{clip.segment_end_index + 1}"

        # Build clip title with optional score
        clip_title = f"Clip {index + 1} ({duration_seconds:.0f}s)"
        if clip.score is not None:
            clip_title += f" [Score: {clip.score:.1f}]"

        # Build HTML-formatted text for rich formatting
        html_parts = [
            f"<b>{clip_title}</b><br>",
            f"{segment_nums}: {start_time} – {end_time}"
        ]

        # Add clip description/name if available (from Claude or manual input)
        if clip.text and clip.text.strip() and clip.text != "Full Video":
            # Parse clip.text to format title, summary, and justification
            lines = clip.text.split('\n')

            if len(lines) >= 1:
                # First line: Title (bold)
                title = lines[0].strip()
                html_parts.append(f"<br><br><b>{title}</b>")

                # Remaining lines: Look for "Zusammenfassung:" and "Begründung:"
                remaining_text = '\n'.join(lines[1:])

                # Split by paragraphs (double newline)
                paragraphs = remaining_text.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue

                    # Check if it's a justification (Begründung)
                    if para.lower().startswith('begründung:') or 'begründung' in para.lower():
                        # Make justification italic
                        html_parts.append(f"<br><br><i>{para}</i>")
                    else:
                        # Regular text (summary)
                        html_parts.append(f"<br><br>{para}")
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
                html_parts.append(f"<br>{start_text}")
            if end_text:
                html_parts.append(f"<br>...<br>{end_text}")

        # Create list item with HTML support
        item = QListWidgetItem()
        label = QLabel(''.join(html_parts))
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setContentsMargins(8, 8, 8, 8)

        # Set size policy to ensure label expands properly
        from PyQt6.QtWidgets import QSizePolicy
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        item.setData(Qt.ItemDataRole.UserRole, index)  # Store clip index

        # Calculate variable height based on actual content
        # Use QLabel's sizeHint to get required height for the content
        label_width = self.clip_list.width() - 40  # Account for scrollbar and margins
        label.setMaximumWidth(label_width)
        content_height = label.sizeHint().height()

        # Set minimum height and add padding
        min_height = 80
        height = max(min_height, content_height + 20)  # +20 for padding

        item.setSizeHint(QSize(self.clip_list.width() - 20, height))

        self.clip_list.addItem(item)
        self.clip_list.setItemWidget(item, label)  # Use label as custom widget
        logger.debug(f"[CLIP_LIST] Added clip {index + 1} with HTML-formatted description, height: {height}px")

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
            duration_seconds = clip.end_time - clip.start_time
            start_time = format_timestamp(clip.start_time)
            end_time = format_timestamp(clip.end_time)

            # Combine segment range with time (same format as _add_clip_row)
            segment_nums = f"Seg {clip.segment_start_index + 1}-{clip.segment_end_index + 1}"

            # Build clip title with optional score (same as in _add_clip_row)
            clip_title = f"Clip {clip_index + 1} ({duration_seconds:.0f}s)"
            if clip.score is not None:
                clip_title += f" [Score: {clip.score:.1f}]"

            # Build HTML-formatted text (same as in _add_clip_row)
            html_parts = [
                f"<b>{clip_title}</b><br>",
                f"{segment_nums}: {start_time} – {end_time}"
            ]

            # Add clip description if available
            if clip.text and clip.text.strip() and clip.text != "Full Video":
                lines = clip.text.split('\n')
                if len(lines) >= 1:
                    title = lines[0].strip()
                    html_parts.append(f"<br><br><b>{title}</b>")

                    remaining_text = '\n'.join(lines[1:])
                    paragraphs = remaining_text.split('\n\n')
                    for para in paragraphs:
                        para = para.strip()
                        if not para:
                            continue
                        if para.lower().startswith('begründung:') or 'begründung' in para.lower():
                            html_parts.append(f"<br><br><i>{para}</i>")
                        else:
                            html_parts.append(f"<br><br>{para}")
            else:
                # Fallback: show start and end segment texts
                start_text = ""
                end_text = ""
                if self.srt_viewer:
                    start_seg = self.srt_viewer.get_segment_at_index(clip.segment_start_index)
                    end_seg = self.srt_viewer.get_segment_at_index(clip.segment_end_index)
                    if start_seg:
                        start_text = start_seg.text[:50]
                    if end_seg:
                        end_text = end_seg.text[-50:]

                if start_text:
                    html_parts.append(f"<br>{start_text}")
                if end_text:
                    html_parts.append(f"<br>...<br>{end_text}")

            # Update the label widget (NOT the item text!)
            item = self.clip_list.item(clip_index)
            if item:
                label = self.clip_list.itemWidget(item)
                if label and isinstance(label, QLabel):
                    label.setText(''.join(html_parts))

                    # Recalculate height based on new content
                    label_width = self.clip_list.width() - 40
                    label.setMaximumWidth(label_width)
                    content_height = label.sizeHint().height()
                    min_height = 80
                    height = max(min_height, content_height + 20)
                    item.setSizeHint(QSize(self.clip_list.width() - 20, height))

                    logger.debug(f"[CLIP_LIST] Updated clip {clip_index + 1} display (height: {height}px)")
        except Exception as e:
            logger.error(f"[CLIP_LIST] Error updating clip display {clip_index}: {e}", exc_info=True)
