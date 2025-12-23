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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._highlighter = SRTSyntaxHighlighter(self.document())
        self._segments: List[SrtSegment] = []
        self._block_to_segment: Dict[int, int] = {}
        self._segment_to_blocks: Dict[int, List[int]] = {}
        self._current_segment_index = None

    def set_segments(self, segments: List[SrtSegment]) -> None:
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

    def mousePressEvent(self, event) -> None:
        cursor = self.cursorForPosition(event.pos())
        block_number = cursor.blockNumber()
        segment_index = self._block_to_segment.get(block_number)
        if segment_index is None:
            super().mousePressEvent(event)
            return
        segment = self._segments[segment_index]
        self.marker_changed.emit(segment.start, segment.end)
        super().mousePressEvent(event)
