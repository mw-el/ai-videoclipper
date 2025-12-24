from __future__ import annotations

from typing import Dict, List

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QSyntaxHighlighter, QTextCursor
from PyQt6.QtWidgets import QTextEdit

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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._highlighter = SRTSyntaxHighlighter(self.document())
        self._segments: List[SrtSegment] = []
        self._block_to_segment: Dict[int, int] = {}
        self._segment_to_blocks: Dict[int, List[int]] = {}
        self._current_segment_index = None
        self._highlighted_range_start = None  # For range highlighting (clip boundaries)
        self._highlighted_range_end = None

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

    def highlight_segment(self, segment_index: int) -> None:
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
        selection.format.setBackground(QColor("#fff3bf"))
        self.setExtraSelections([selection])
        self._current_segment_index = segment_index

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
            selection.format.setBackground(QColor("#c3f0ca"))  # Green highlight for active clip
            self.setExtraSelections([selection])
            logger.info(f"[SRT_VIEWER] Highlight applied successfully")

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
