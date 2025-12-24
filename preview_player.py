from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl, QSize, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QFontMetrics

from time_utils import format_timestamp
from design.icon_manager import IconManager
from design.style_manager import StyleManager, Colors


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

    def mousePressEvent(self, event) -> None:
        """Ensure focus (highlight) on click for keyboard navigation."""
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)


class PreviewPlayer(QWidget):
    marker_changed = pyqtSignal(float, float)
    user_marker_changed = pyqtSignal(float, float)
    position_changed = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.player = None
        self.audio_output = None

        self._duration_ms = 0
        self._start_ms = 0
        self._end_ms = 0
        self._segments = []  # Store segment boundaries for keyboard navigation
        self._user_marker_change = False

        self.video_widget = QVideoWidget(self)
        # Set size for video widget: 360px wide × 203px high (16:9 aspect ratio)
        self.video_widget.setMinimumHeight(203)
        self.video_widget.setMinimumWidth(360)
        self.video_widget.setMaximumWidth(360)

        self.play_button = QPushButton()
        self.play_button.setIcon(IconManager.create_icon('play_arrow', color='white', size=20))
        self.play_button.setIconSize(QSize(20, 20))
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setToolTip("Play/Pause video")
        StyleManager.apply_colored_icon_button_style(self.play_button, Colors.BRIGHT_GREEN)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self._on_position_slider_moved)

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

        self._build_marker_widget()
        self._build_layout()

        self._player_connected = False

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Playback controls
        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(self.play_button)
        controls.addWidget(self.position_slider, stretch=1)
        controls.addWidget(self.time_label)
        controls.setAlignment(self.play_button, Qt.AlignmentFlag.AlignVCenter)
        controls.setAlignment(self.position_slider, Qt.AlignmentFlag.AlignVCenter)
        controls.setAlignment(self.time_label, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(controls)

        self.position_slider.setStyleSheet("margin-top: 2px;")

    def _build_marker_widget(self) -> None:
        self.marker_widget = QWidget()
        markers_layout = QVBoxLayout(self.marker_widget)
        markers_layout.setContentsMargins(0, 0, 0, 0)
        markers_layout.setSpacing(4)
        label_width = self.play_button.sizeHint().width()
        font_metrics = QFontMetrics(self.time_label.font())
        trailing_width = self.time_label.sizeHint().width()
        start_row = QHBoxLayout()
        start_row.setSpacing(6)
        start_label = QLabel("Start")
        start_label.setFixedWidth(label_width)
        start_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        start_row.addWidget(start_label)
        start_row.addWidget(self.start_slider, stretch=1)
        self.start_marker_label = QLabel("Start: --: --:--:--")
        self.start_marker_label.setFixedWidth(trailing_width)
        self.start_marker_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        start_row.addWidget(self.start_marker_label)
        end_row = QHBoxLayout()
        end_row.setSpacing(6)
        end_label = QLabel("End")
        end_label.setFixedWidth(label_width)
        end_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        end_row.addWidget(end_label)
        end_row.addWidget(self.end_slider, stretch=1)
        self.end_marker_label = QLabel("End: --: --:--:--")
        self.end_marker_label.setFixedWidth(trailing_width)
        self.end_marker_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        end_row.addWidget(self.end_marker_label)
        markers_layout.addLayout(start_row)
        markers_layout.addLayout(end_row)

    def load_media(self, path: str) -> None:
        self._ensure_player()
        if self.player:
            self.player.setSource(QUrl.fromLocalFile(path))

    def toggle_play(self) -> None:
        self._ensure_player()
        if not self.player:
            return
        from PyQt6.QtMultimedia import QMediaPlayer

        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            # Before playing, check if we're within the clip range
            current_pos = self.player.position()
            if current_pos < self._start_ms or current_pos >= self._end_ms:
                # Jump to start marker if we're outside the clip range
                self.player.setPosition(self._start_ms)
            self.player.play()

    def seek_seconds(self, seconds: float) -> None:
        self._ensure_player()
        if self.player:
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
        self._update_marker_labels()

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
        from PyQt6.QtMultimedia import QMediaPlayer

        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("")
            self.play_button.setIcon(IconManager.create_icon('pause', color='white', size=20))
        else:
            self.play_button.setText("")
            self.play_button.setIcon(IconManager.create_icon('play_arrow', color='white', size=20))

    def _on_start_changed(self, value: int) -> None:
        if value > self._end_ms:
            self._end_ms = value
            self.end_slider.blockSignals(True)
            self.end_slider.setValue(value)
            self.end_slider.blockSignals(False)
        self._start_ms = value
        self.marker_changed.emit(self._start_ms / 1000.0, self._end_ms / 1000.0)
        self._update_marker_labels()
        if self._user_marker_change:
            self.user_marker_changed.emit(self._start_ms / 1000.0, self._end_ms / 1000.0)
            self._user_marker_change = False

    def _on_end_changed(self, value: int) -> None:
        if value < self._start_ms:
            self._start_ms = value
            self.start_slider.blockSignals(True)
            self.start_slider.setValue(value)
            self.start_slider.blockSignals(False)
        self._end_ms = value
        self.marker_changed.emit(self._start_ms / 1000.0, self._end_ms / 1000.0)
        self._update_marker_labels()
        if self._user_marker_change:
            self.user_marker_changed.emit(self._start_ms / 1000.0, self._end_ms / 1000.0)
            self._user_marker_change = False

    def _update_time_label(self, position: int) -> None:
        current = format_timestamp(position / 1000.0)
        total = format_timestamp(self._duration_ms / 1000.0) if self._duration_ms else "00:00:00"
        self.time_label.setText(f"{current} / {total}")

    def set_segments(self, segments) -> None:
        """Store segment boundaries for keyboard-based navigation."""
        self._segments = [(seg.start * 1000, seg.end * 1000) for seg in segments]
        self._update_marker_labels()

    def _find_segment_index_for_time(self, seconds: float, prefer_start: bool) -> int | None:
        if not self._segments:
            return None
        target_ms = seconds * 1000.0
        if prefer_start:
            for i, (seg_start, seg_end) in enumerate(self._segments):
                if seg_end >= target_ms:
                    return i
            return len(self._segments) - 1
        for i in range(len(self._segments) - 1, -1, -1):
            seg_start, seg_end = self._segments[i]
            if seg_start <= target_ms:
                return i
        return 0

    def _update_marker_labels(self) -> None:
        if not hasattr(self, "start_marker_label"):
            return
        start_seconds = self._start_ms / 1000.0
        end_seconds = self._end_ms / 1000.0
        start_idx = self._find_segment_index_for_time(start_seconds, prefer_start=True)
        end_idx = self._find_segment_index_for_time(end_seconds, prefer_start=False)
        start_prefix = f"{start_idx + 1}" if start_idx is not None else "--"
        end_prefix = f"{end_idx + 1}" if end_idx is not None else "--"
        self.start_marker_label.setText(f"Start: {start_prefix}: {format_timestamp(start_seconds)}")
        self.end_marker_label.setText(f"End: {end_prefix}: {format_timestamp(end_seconds)}")

    def ensure_player(self) -> None:
        """Public hook to initialize the media player after GUI start."""
        self._ensure_player()

    def _ensure_player(self) -> None:
        if self.player is not None:
            return
        from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)
        if not self._player_connected:
            self.player.durationChanged.connect(self._on_duration_changed)
            self.player.positionChanged.connect(self._on_position_changed)
            self.player.playbackStateChanged.connect(self._on_state_changed)
            self._player_connected = True

    def _on_position_slider_moved(self, position: int) -> None:
        self._ensure_player()
        if self.player:
            self.player.setPosition(position)

    def _on_start_slider_moved(self, value: int) -> None:
        """Handle start slider moved by user (not value change)."""
        self._user_marker_change = True
        self.start_slider.setFocus()

    def _on_end_slider_moved(self, value: int) -> None:
        """Handle end slider moved by user (not value change)."""
        self._user_marker_change = True
        self.end_slider.setFocus()

    def _on_start_segment_jump(self, forward: bool) -> None:
        """Handle start slider segment jump request from keyboard."""
        self._user_marker_change = True
        if not self._move_start_to_segment(backward=not forward):
            self._user_marker_change = False

    def _on_end_segment_jump(self, forward: bool) -> None:
        """Handle end slider segment jump request from keyboard."""
        self._user_marker_change = True
        if not self._move_end_to_segment(backward=not forward):
            self._user_marker_change = False

    def _move_start_to_segment(self, backward: bool = False) -> bool:
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
                    return True
        else:
            # Move to next segment start
            for seg_start, seg_end in self._segments:
                if seg_start > current_pos:
                    self.start_slider.blockSignals(True)
                    self.start_slider.setValue(int(seg_start))
                    self.start_slider.blockSignals(False)
                    self._on_start_changed(int(seg_start))
                    return True
        return False

    def _move_end_to_segment(self, backward: bool = False) -> bool:
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
                    return True
        else:
            # Move to next segment end
            for seg_start, seg_end in self._segments:
                if seg_end > current_pos:
                    self.end_slider.blockSignals(True)
                    self.end_slider.setValue(int(seg_end))
                    self.end_slider.blockSignals(False)
                    self._on_end_changed(int(seg_end))
                    return True
        return False
