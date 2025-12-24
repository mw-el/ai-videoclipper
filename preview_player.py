from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl, QSize, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

from time_utils import format_timestamp


class SegmentAwareSlider(QSlider):
    """Custom slider that handles segment-based keyboard navigation."""

    segment_jump_requested = pyqtSignal(bool)  # True = forward, False = backward

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Visual feedback for focused state
        self.setStyleSheet("""
            QSlider:focus {
                border: 2px solid #4CAF50;
                border-radius: 3px;
                padding: 2px;
            }
        """)

    def keyPressEvent(self, event) -> None:
        """Handle arrow key presses for segment navigation."""
        if event.key() == Qt.Key.Key_Left:
            self.segment_jump_requested.emit(False)  # backward
            event.accept()
        elif event.key() == Qt.Key.Key_Right:
            self.segment_jump_requested.emit(True)  # forward
            event.accept()
        else:
            super().keyPressEvent(event)


class PreviewPlayer(QWidget):
    marker_changed = pyqtSignal(float, float)
    position_changed = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

        self._duration_ms = 0
        self._start_ms = 0
        self._end_ms = 0
        self._segments = []  # Store segment boundaries for keyboard navigation

        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)
        # Set minimum size for video widget (30% larger), maintain 16:9 aspect ratio
        self.video_widget.setMinimumHeight(260)
        self.video_widget.setMinimumWidth(int(260 * 16 / 9))

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.player.setPosition)

        self.time_label = QLabel("00:00:00 / 00:00:00")

        self.start_slider = SegmentAwareSlider(Qt.Orientation.Horizontal)
        self.start_slider.setRange(0, 0)
        self.start_slider.valueChanged.connect(self._on_start_changed)
        self.start_slider.sliderMoved.connect(self._on_start_slider_moved)
        self.start_slider.segment_jump_requested.connect(self._on_start_segment_jump)

        self.end_slider = SegmentAwareSlider(Qt.Orientation.Horizontal)
        self.end_slider.setRange(0, 0)
        self.end_slider.valueChanged.connect(self._on_end_changed)
        self.end_slider.sliderMoved.connect(self._on_end_slider_moved)
        self.end_slider.segment_jump_requested.connect(self._on_end_segment_jump)

        self._build_layout()

        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.playbackStateChanged.connect(self._on_state_changed)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Playback controls
        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.position_slider, stretch=1)
        controls.addWidget(self.time_label)
        layout.addLayout(controls)

        # Clip markers sliders
        markers_layout = QVBoxLayout()
        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Start"))
        start_row.addWidget(self.start_slider, stretch=1)
        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("End"))
        end_row.addWidget(self.end_slider, stretch=1)
        markers_layout.addLayout(start_row)
        markers_layout.addLayout(end_row)
        layout.addLayout(markers_layout)

    def load_media(self, path: str) -> None:
        self.player.setSource(QUrl.fromLocalFile(path))

    def toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def seek_seconds(self, seconds: float) -> None:
        self.player.setPosition(int(seconds * 1000))

    def set_markers(self, start_seconds: float, end_seconds: float) -> None:
        if self._duration_ms <= 0:
            return
        start_ms = max(0, min(int(start_seconds * 1000), self._duration_ms))
        end_ms = max(0, min(int(end_seconds * 1000), self._duration_ms))
        if end_ms < start_ms:
            end_ms = start_ms
        self._set_marker_values(start_ms, end_ms)

    def _set_marker_values(self, start_ms: int, end_ms: int) -> None:
        self._start_ms = start_ms
        self._end_ms = end_ms
        self.start_slider.blockSignals(True)
        self.end_slider.blockSignals(True)
        self.start_slider.setValue(start_ms)
        self.end_slider.setValue(end_ms)
        self.start_slider.blockSignals(False)
        self.end_slider.blockSignals(False)
        self.marker_changed.emit(start_ms / 1000.0, end_ms / 1000.0)

    def _on_duration_changed(self, duration: int) -> None:
        self._duration_ms = duration
        self.position_slider.setRange(0, duration)
        self.start_slider.setRange(0, duration)
        self.end_slider.setRange(0, duration)
        self._set_marker_values(0, duration)
        self._update_time_label(0)

    def _on_position_changed(self, position: int) -> None:
        if self._end_ms and position >= self._end_ms:
            self.player.pause()
            self.player.setPosition(self._end_ms)
            position = self._end_ms
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)
        self._update_time_label(position)
        self.position_changed.emit(position / 1000.0)

    def _on_state_changed(self, state) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")

    def _on_start_changed(self, value: int) -> None:
        if value > self._end_ms:
            self._end_ms = value
            self.end_slider.blockSignals(True)
            self.end_slider.setValue(value)
            self.end_slider.blockSignals(False)
        self._start_ms = value
        self.marker_changed.emit(self._start_ms / 1000.0, self._end_ms / 1000.0)

    def _on_end_changed(self, value: int) -> None:
        if value < self._start_ms:
            self._start_ms = value
            self.start_slider.blockSignals(True)
            self.start_slider.setValue(value)
            self.start_slider.blockSignals(False)
        self._end_ms = value
        self.marker_changed.emit(self._start_ms / 1000.0, self._end_ms / 1000.0)

    def _update_time_label(self, position: int) -> None:
        current = format_timestamp(position / 1000.0)
        total = format_timestamp(self._duration_ms / 1000.0) if self._duration_ms else "00:00:00"
        self.time_label.setText(f"{current} / {total}")

    def set_segments(self, segments) -> None:
        """Store segment boundaries for keyboard-based navigation."""
        self._segments = [(seg.start * 1000, seg.end * 1000) for seg in segments]

    def _on_start_slider_moved(self, value: int) -> None:
        """Handle start slider moved by user (not value change)."""
        self.start_slider.setFocus()

    def _on_end_slider_moved(self, value: int) -> None:
        """Handle end slider moved by user (not value change)."""
        self.end_slider.setFocus()

    def _on_start_segment_jump(self, forward: bool) -> None:
        """Handle start slider segment jump request from keyboard."""
        self._move_start_to_segment(backward=not forward)

    def _on_end_segment_jump(self, forward: bool) -> None:
        """Handle end slider segment jump request from keyboard."""
        self._move_end_to_segment(backward=not forward)

    def _move_start_to_segment(self, backward: bool = False) -> None:
        """Move start marker to previous/next segment boundary."""
        current_pos = self._start_ms

        # Find matching segment boundary
        if backward:
            # Move to previous segment start
            for seg_start, seg_end in reversed(self._segments):
                if seg_start < current_pos:
                    self.start_slider.blockSignals(True)
                    self.start_slider.setValue(int(seg_start))
                    self.start_slider.blockSignals(False)
                    self._on_start_changed(int(seg_start))
                    return
        else:
            # Move to next segment start
            for seg_start, seg_end in self._segments:
                if seg_start > current_pos:
                    self.start_slider.blockSignals(True)
                    self.start_slider.setValue(int(seg_start))
                    self.start_slider.blockSignals(False)
                    self._on_start_changed(int(seg_start))
                    return

    def _move_end_to_segment(self, backward: bool = False) -> None:
        """Move end marker to previous/next segment boundary."""
        current_pos = self._end_ms

        # Find matching segment boundary
        if backward:
            # Move to previous segment end
            for seg_start, seg_end in reversed(self._segments):
                if seg_end < current_pos:
                    self.end_slider.blockSignals(True)
                    self.end_slider.setValue(int(seg_end))
                    self.end_slider.blockSignals(False)
                    self._on_end_changed(int(seg_end))
                    return
        else:
            # Move to next segment end
            for seg_start, seg_end in self._segments:
                if seg_end > current_pos:
                    self.end_slider.blockSignals(True)
                    self.end_slider.setValue(int(seg_end))
                    self.end_slider.blockSignals(False)
                    self._on_end_changed(int(seg_end))
                    return
