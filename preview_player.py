from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl, QSize, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

from time_utils import format_timestamp


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

        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.player.setPosition)

        self.time_label = QLabel("00:00:00 / 00:00:00")

        self.start_slider = QSlider(Qt.Orientation.Horizontal)
        self.start_slider.setRange(0, 0)
        self.start_slider.valueChanged.connect(self._on_start_changed)

        self.end_slider = QSlider(Qt.Orientation.Horizontal)
        self.end_slider.setRange(0, 0)
        self.end_slider.valueChanged.connect(self._on_end_changed)

        self._build_layout()

        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.playbackStateChanged.connect(self._on_state_changed)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # TOP: Video player with 16:9 aspect ratio, 200px height
        # Width = 200 * 16/9 = 355.56px
        self.video_widget.setMinimumHeight(200)
        self.video_widget.setMaximumHeight(200)
        self.video_widget.setMinimumWidth(int(200 * 16 / 9))
        self.video_widget.setMaximumWidth(int(200 * 16 / 9))
        layout.addWidget(self.video_widget, alignment=Qt.AlignmentFlag.AlignHCenter)

        # MIDDLE/BOTTOM: Horizontal layout with sliders on left
        main_content = QHBoxLayout()

        # LEFT: Sliders (2/3 width)
        sliders_widget = QWidget()
        sliders_layout = QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.position_slider, stretch=1)
        controls.addWidget(self.time_label)
        sliders_layout.addLayout(controls)

        markers_layout = QVBoxLayout()
        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Start"))
        start_row.addWidget(self.start_slider, stretch=1)
        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("End"))
        end_row.addWidget(self.end_slider, stretch=1)
        markers_layout.addLayout(start_row)
        markers_layout.addLayout(end_row)
        sliders_layout.addLayout(markers_layout)

        main_content.addWidget(sliders_widget, stretch=2)

        layout.addLayout(main_content)

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
