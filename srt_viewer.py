from __future__ import annotations

from typing import Dict, List

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QTextCharFormat, QSyntaxHighlighter, QTextCursor, QTextDocument
from PyQt6.QtWidgets import QTextEdit, QWidget, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QLabel

from srt_utils import SrtSegment
from time_utils import format_srt_timestamp


class SRTSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, document) -> None:
        super().__init__(document)
        self.index_format = QTextCharFormat()
        self.index_format.setForeground(QColor("#3b5bdb"))
        self.time_format = QTextCharFormat()
        self.time_format.setForeground(QColor("#2b8a3e"))

    def highlightBlock(self, text: str) -> None:
        stripped = text.strip()
        if stripped.isdigit():
            self.setFormat(0, len(text), self.index_format)
            return
        if "-->" in text:
            self.setFormat(0, len(text), self.time_format)


class SRTViewer(QTextEdit):
    marker_changed = pyqtSignal(float, float)
    segment_clicked = pyqtSignal(int)  # Emitted when user clicks on a segment
    highlight_range_changed = pyqtSignal(int, int)  # Emitted when highlight range changes (start_idx, end_idx)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        # Set white background to prevent orange/red default background
        self.setStyleSheet("QTextEdit { background-color: #ffffff; color: #000000; }")
        self._highlighter = SRTSyntaxHighlighter(self.document())
        self._segments: List[SrtSegment] = []
        self._block_to_segment: Dict[int, int] = {}
        self._segment_to_blocks: Dict[int, List[int]] = {}
        self._current_segment_index = None
        self._highlighted_range_start = None  # For range highlighting (clip boundaries)
        self._highlighted_range_end = None
        self._hook_start_index = None  # Hook segment start index
        self._hook_end_index = None    # Hook segment end index

    def set_segments(self, segments: List[SrtSegment]) -> None:
        import logging
        logger = logging.getLogger("ai_videoclipper")
        logger.info(f"[SRT_VIEWER] set_segments() called with {len(segments)} segments")

        self._segments = list(segments)
        self._block_to_segment.clear()
        self._segment_to_blocks.clear()

        lines = []
        line_number = 0
        for seg_index, segment in enumerate(self._segments):
            block_indexes = []
            lines.append(str(segment.index))
            self._block_to_segment[line_number] = seg_index
            block_indexes.append(line_number)
            line_number += 1

            lines.append(
                f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}"
            )
            self._block_to_segment[line_number] = seg_index
            block_indexes.append(line_number)
            line_number += 1

            text_lines = segment.text.splitlines() if segment.text else [""]
            for text_line in text_lines:
                lines.append(text_line)
                self._block_to_segment[line_number] = seg_index
                block_indexes.append(line_number)
                line_number += 1

            lines.append("")
            line_number += 1
            self._segment_to_blocks[seg_index] = block_indexes

        self.setPlainText("\n".join(lines).strip() + "\n")
        self._current_segment_index = None
        self.setExtraSelections([])

        # Apply bold formatting to hook segments
        self._apply_hook_formatting()

        logger.info(f"[SRT_VIEWER] Display updated with {len(self._segments)} segments and {len(lines)} display lines")

    def highlight_for_time(self, seconds: float) -> None:
        if not self._segments:
            return
        for idx, segment in enumerate(self._segments):
            if segment.start <= seconds <= segment.end:
                if idx != self._current_segment_index:
                    self.highlight_segment(idx)
                return
        if self._current_segment_index is not None:
            self.setExtraSelections([])
            self._current_segment_index = None

    def highlight_segment(self, segment_index: int, auto_scroll: bool = True) -> None:
        blocks = self._segment_to_blocks.get(segment_index)
        if not blocks:
            return
        doc = self.document()
        first_block = doc.findBlockByNumber(blocks[0])
        last_block = doc.findBlockByNumber(blocks[-1])
        if not first_block.isValid() or not last_block.isValid():
            return

        cursor = QTextCursor(first_block)
        cursor.setPosition(last_block.position() + last_block.length() - 1, QTextCursor.MoveMode.KeepAnchor)

        selection = QTextEdit.ExtraSelection()
        selection.cursor = cursor
        # Set format explicitly to override any default selection colors
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("#fff3bf"))  # Yellow highlight for current position
        fmt.setForeground(QColor("#000000"))  # Black text on yellow background
        selection.format = fmt
        self.setExtraSelections([selection])
        self._current_segment_index = segment_index

        # Auto-scroll to show highlighted segment at top
        if auto_scroll:
            # Position cursor at start of first block
            scroll_cursor = QTextCursor(first_block)
            self.setTextCursor(scroll_cursor)
            self.ensureCursorVisible()

    def highlight_segment_range(self, start_index: int, end_index: int, auto_scroll: bool = True) -> None:
        """Highlight a range of segments (for showing active clip).

        Args:
            start_index: First segment index (inclusive)
            end_index: Last segment index (inclusive)
            auto_scroll: If True, scroll so start_index appears at top
        """
        import logging
        logger = logging.getLogger("ai_videoclipper")

        try:
            logger.info(f"[SRT_VIEWER] highlight_segment_range() called: start_index={start_index}, end_index={end_index}, auto_scroll={auto_scroll}")
            logger.info(f"[SRT_VIEWER] Total segments available: {len(self._segments)}")

            # Validate indices
            if start_index < 0 or end_index < 0:
                logger.error(f"[SRT_VIEWER] Invalid indices: start_index={start_index}, end_index={end_index}")
                self.setExtraSelections([])
                return

            if start_index >= len(self._segments) or end_index >= len(self._segments):
                logger.error(f"[SRT_VIEWER] Indices out of range: start_index={start_index}, end_index={end_index}, total_segments={len(self._segments)}")
                self.setExtraSelections([])
                return

            if start_index > end_index:
                logger.error(f"[SRT_VIEWER] start_index > end_index: {start_index} > {end_index}")
                self.setExtraSelections([])
                return

            self._highlighted_range_start = start_index
            self._highlighted_range_end = end_index
            logger.info(f"[SRT_VIEWER] Range validated, collecting blocks...")

            # Collect all blocks in range
            all_blocks = []
            for seg_idx in range(start_index, end_index + 1):
                blocks = self._segment_to_blocks.get(seg_idx, [])
                logger.debug(f"[SRT_VIEWER] Segment {seg_idx}: {len(blocks)} blocks")
                all_blocks.extend(blocks)

            logger.info(f"[SRT_VIEWER] Collected {len(all_blocks)} total blocks for range")

            if not all_blocks:
                logger.warning(f"[SRT_VIEWER] No blocks found for range {start_index}-{end_index}")
                self.setExtraSelections([])
                return

            doc = self.document()
            first_block = doc.findBlockByNumber(all_blocks[0])
            last_block = doc.findBlockByNumber(all_blocks[-1])

            logger.info(f"[SRT_VIEWER] first_block valid={first_block.isValid()}, last_block valid={last_block.isValid()}")

            if not first_block.isValid() or not last_block.isValid():
                logger.error(f"[SRT_VIEWER] Invalid blocks after lookup")
                self.setExtraSelections([])
                return

            # Create selection for range
            cursor = QTextCursor(first_block)
            cursor.setPosition(last_block.position() + last_block.length() - 1, QTextCursor.MoveMode.KeepAnchor)

            selection = QTextEdit.ExtraSelection()
            selection.cursor = cursor
            # Set format explicitly to override any default selection colors
            fmt = QTextCharFormat()
            fmt.setBackground(QColor("#c3f0ca"))  # Green highlight for active clip
            fmt.setForeground(QColor("#000000"))  # Black text on green background
            selection.format = fmt
            self.setExtraSelections([selection])
            logger.info(f"[SRT_VIEWER] Highlight applied successfully")

            # Emit signal to notify about range change
            self.highlight_range_changed.emit(start_index, end_index)

            # Auto-scroll to show start segment at top
            if auto_scroll:
                logger.debug(f"[SRT_VIEWER] Auto-scrolling...")
                self.verticalScrollBar().setValue(0)  # Reset scroll
                self.setTextCursor(cursor)
                self.ensureCursorVisible()
                logger.info(f"[SRT_VIEWER] Auto-scroll complete")

        except Exception as e:
            import logging
            logger = logging.getLogger("ai_videoclipper")
            logger.exception(f"[SRT_VIEWER] EXCEPTION in highlight_segment_range(): {e}")

    def mousePressEvent(self, event) -> None:
        cursor = self.cursorForPosition(event.pos())
        block_number = cursor.blockNumber()
        segment_index = self._block_to_segment.get(block_number)
        if segment_index is None:
            super().mousePressEvent(event)
            return
        segment = self._segments[segment_index]
        self.marker_changed.emit(segment.start, segment.end)
        self.segment_clicked.emit(segment_index)
        super().mousePressEvent(event)

    def get_segment_at_index(self, index: int) -> SrtSegment | None:
        """Get segment by index."""
        if 0 <= index < len(self._segments):
            return self._segments[index]
        return None

    def get_current_highlight_range(self) -> tuple[int, int] | None:
        """Get current highlight range (start_index, end_index) or None."""
        if self._highlighted_range_start is not None and self._highlighted_range_end is not None:
            return (self._highlighted_range_start, self._highlighted_range_end)
        return None

    def set_hook_range(self, start_index: int | None, end_index: int | None) -> None:
        """Set which segments are the hook (will be shown in bold).

        Args:
            start_index: First segment index of hook (inclusive), or None to clear
            end_index: Last segment index of hook (inclusive), or None to clear
        """
        self._hook_start_index = start_index
        self._hook_end_index = end_index
        self._apply_hook_formatting()

    def _apply_hook_formatting(self) -> None:
        """Apply bold formatting to hook segments."""
        if (self._hook_start_index is None or
            self._hook_end_index is None or
            not self._segments):
            return

        # Create cursor and format for bold text
        cursor = QTextCursor(self.document())
        bold_format = QTextCharFormat()
        bold_format.setFontWeight(700)  # Bold
        bold_format.setForeground(QColor("#1a5490"))  # Darker blue for hook

        # Apply bold to all blocks in hook range
        for seg_idx in range(self._hook_start_index, self._hook_end_index + 1):
            blocks = self._segment_to_blocks.get(seg_idx, [])
            for block_num in blocks:
                block = self.document().findBlockByNumber(block_num)
                if block.isValid():
                    cursor.setPosition(block.position())
                    cursor.movePosition(
                        QTextCursor.MoveOperation.EndOfBlock,
                        QTextCursor.MoveMode.KeepAnchor
                    )
                    cursor.mergeCharFormat(bold_format)

    def clear(self) -> None:
        """Clear all content and reset state."""
        self.setPlainText("")
        self._segments = []
        self._block_to_segment.clear()
        self._segment_to_blocks.clear()
        self._current_segment_index = None
        self._highlighted_range_start = None
        self._highlighted_range_end = None
        self._hook_start_index = None
        self._hook_end_index = None
        self.setExtraSelections([])

    def search_text(self, search_term: str, forward: bool = True) -> bool:
        """Search for text in SRT content.

        Args:
            search_term: Text to search for
            forward: If True, search forward; if False, search backward

        Returns:
            True if match found, False otherwise
        """
        if not search_term:
            return False

        flags = QTextDocument.FindFlag(0)
        if not forward:
            flags |= QTextDocument.FindFlag.FindBackward

        # Get current cursor position
        cursor = self.textCursor()

        # Search from current position
        found_cursor = self.document().find(search_term, cursor, flags)

        if not found_cursor.isNull():
            self.setTextCursor(found_cursor)
            self.ensureCursorVisible()
            return True
        else:
            # Wrap around search
            if forward:
                cursor = QTextCursor(self.document())  # Start from beginning
            else:
                cursor = QTextCursor(self.document())
                cursor.movePosition(QTextCursor.MoveOperation.End)  # Start from end

            found_cursor = self.document().find(search_term, cursor, flags)
            if not found_cursor.isNull():
                self.setTextCursor(found_cursor)
                self.ensureCursorVisible()
                return True

        return False


class SRTViewerWithSearch(QWidget):
    """SRT Viewer with integrated search bar."""

    # Forward signals from SRTViewer
    marker_changed = pyqtSignal(float, float)
    segment_clicked = pyqtSignal(int)
    highlight_range_changed = pyqtSignal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Search bar
        search_layout = QHBoxLayout()
        search_layout.setSpacing(4)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("SRT durchsuchen...")
        self.search_input.returnPressed.connect(self._search_next)
        search_layout.addWidget(self.search_input)

        self.prev_button = QPushButton("◀")
        self.prev_button.setMaximumWidth(40)
        self.prev_button.setToolTip("Vorheriges Ergebnis")
        self.prev_button.clicked.connect(self._search_previous)
        search_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("▶")
        self.next_button.setMaximumWidth(40)
        self.next_button.setToolTip("Nächstes Ergebnis")
        self.next_button.clicked.connect(self._search_next)
        search_layout.addWidget(self.next_button)

        self.search_status = QLabel("")
        self.search_status.setStyleSheet("color: #666; font-size: 10px;")
        search_layout.addWidget(self.search_status)

        layout.addLayout(search_layout)

        # SRT Viewer
        self.srt_viewer = SRTViewer()
        layout.addWidget(self.srt_viewer)

        # Forward signals
        self.srt_viewer.marker_changed.connect(self.marker_changed)
        self.srt_viewer.segment_clicked.connect(self.segment_clicked)
        self.srt_viewer.highlight_range_changed.connect(self.highlight_range_changed)

    def _search_next(self) -> None:
        search_term = self.search_input.text()
        if search_term:
            found = self.srt_viewer.search_text(search_term, forward=True)
            self.search_status.setText("Gefunden" if found else "Nicht gefunden")

    def _search_previous(self) -> None:
        search_term = self.search_input.text()
        if search_term:
            found = self.srt_viewer.search_text(search_term, forward=False)
            self.search_status.setText("Gefunden" if found else "Nicht gefunden")

    # Delegate methods to SRTViewer
    def set_segments(self, segments: List[SrtSegment]) -> None:
        self.srt_viewer.set_segments(segments)

    def highlight_for_time(self, seconds: float) -> None:
        self.srt_viewer.highlight_for_time(seconds)

    def highlight_segment(self, segment_index: int, auto_scroll: bool = True) -> None:
        self.srt_viewer.highlight_segment(segment_index, auto_scroll)

    def highlight_segment_range(self, start_index: int, end_index: int, auto_scroll: bool = True) -> None:
        self.srt_viewer.highlight_segment_range(start_index, end_index, auto_scroll)

    def get_segment_at_index(self, index: int) -> SrtSegment | None:
        return self.srt_viewer.get_segment_at_index(index)

    def get_current_highlight_range(self) -> tuple[int, int] | None:
        return self.srt_viewer.get_current_highlight_range()

    def set_hook_range(self, start_index: int | None, end_index: int | None) -> None:
        self.srt_viewer.set_hook_range(start_index, end_index)

    def clear(self) -> None:
        self.srt_viewer.clear()
        self.search_input.clear()
        self.search_status.setText("")
