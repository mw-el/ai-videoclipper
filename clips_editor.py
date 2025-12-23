from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QDockWidget,
    QSplitter,
)

from faster_whisper_transcriber import FasterWhisperTranscriber
from clip_model import Clip, ClipsAIWrapper
from logger import setup_logging
from preview_player import PreviewPlayer
from srt_viewer import SRTViewer
from time_utils import format_timestamp

logger = logging.getLogger("ai_videoclipper")


class Worker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            import time
            logger.info(f"[WORKER] START: {self.func.__name__} at {time.strftime('%H:%M:%S')}")
            print(f"[WORKER THREAD] Starting {self.func.__name__}", file=__import__('sys').stderr, flush=True)
            result = self.func(*self.args, **self.kwargs)
            logger.info(f"[WORKER] COMPLETE: {self.func.__name__} at {time.strftime('%H:%M:%S')}")
            print(f"[WORKER THREAD] Completed {self.func.__name__}", file=__import__('sys').stderr, flush=True)
        except Exception as exc:
            logger.exception(f"[WORKER] ERROR in {self.func.__name__}: {exc}")
            print(f"[WORKER THREAD] ERROR: {exc}", file=__import__('sys').stderr, flush=True)
            self.error.emit(str(exc))
        else:
            self.finished.emit(result)


class ClipListItemWidget(QWidget):
    export_requested = pyqtSignal(int)

    def __init__(self, clip: Clip, index: int, parent=None) -> None:
        super().__init__(parent)
        self.clip = clip
        self.index = index

        title = QLabel(
            f"Clip {index + 1} [{format_timestamp(clip.start_time)}-{format_timestamp(clip.end_time)}]"
        )
        title.setStyleSheet("font-weight: bold")
        subtitle = QLabel(clip.text)
        subtitle.setWordWrap(True)

        export_button = QPushButton("Export")
        export_button.clicked.connect(lambda: self.export_requested.emit(index))

        text_layout = QVBoxLayout()
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)

        layout = QHBoxLayout(self)
        layout.addLayout(text_layout)
        layout.addStretch()
        layout.addWidget(export_button)


class ClipEditor(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI VideoClipper")
        self.resize(1400, 900)

        logger.info("Initializing AI VideoClipper")

        logger.info("Setting up faster-whisper transcriber...")
        self.transcriber = FasterWhisperTranscriber(progress_callback=self._log_progress)
        logger.info("✓ Transcriber initialized")

        self.clip_wrapper = ClipsAIWrapper()

        self.video_path: Path | None = None
        self.transcription = None
        self.clips: list[Clip] = []
        self.output_dir = Path(__file__).resolve().parent / "output" / "clips"

        self._threads: list[QThread] = []

        # Setup logging window
        log_file = Path(__file__).resolve().parent / "logs" / "ai_videoclipper.log"
        self.logger, self.log_emitter = setup_logging(log_file)
        logger.info(f"Logging to: {log_file}")

        self._build_ui()

    def _build_ui(self) -> None:
        container = QWidget(self)
        main_layout = QVBoxLayout(container)

        top_bar = QHBoxLayout()
        self.open_button = QPushButton("Select File")
        self.open_button.clicked.connect(self.select_file)
        self.status_label = QLabel("Status: idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top_bar.addWidget(self.open_button)
        top_bar.addStretch()
        top_bar.addWidget(self.status_label)

        main_layout.addLayout(top_bar)

        body_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        self.clip_list = QListWidget()
        self.clip_list.currentRowChanged.connect(self.preview_clip)

        self.export_all_button = QPushButton("Export All Clips")
        self.export_all_button.setEnabled(False)
        self.export_all_button.clicked.connect(self.export_all)

        self.output_button = QPushButton("Select Output Folder")
        self.output_button.clicked.connect(self.select_output_dir)
        self.output_label = QLabel(str(self.output_dir))
        self.output_label.setWordWrap(True)

        left_layout.addWidget(QLabel("Clips"))
        left_layout.addWidget(self.clip_list, stretch=1)
        left_layout.addWidget(self.export_all_button)
        left_layout.addWidget(self.output_button)
        left_layout.addWidget(self.output_label)

        right_layout = QVBoxLayout()
        self.preview_player = PreviewPlayer()
        self.preview_player.position_changed.connect(self.srt_viewer_highlight)

        self.srt_viewer = SRTViewer()
        self.srt_viewer.marker_changed.connect(self.on_marker_changed)

        right_layout.addWidget(self.preview_player, stretch=2)
        right_layout.addWidget(QLabel("Transcript"))
        right_layout.addWidget(self.srt_viewer, stretch=1)

        body_layout.addLayout(left_layout, stretch=1)
        body_layout.addLayout(right_layout, stretch=2)

        main_layout.addLayout(body_layout)

        # Add logging dock widget
        log_dock = QDockWidget("Logs", self)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(150)
        self.log_display.setStyleSheet("background-color: #1e1e1e; color: #e0e0e0; font-family: monospace;")
        log_dock.setWidget(self.log_display)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)

        # Connect logging signal to display
        self.log_emitter.log_message.connect(self.on_log_message)

        self.setCentralWidget(container)

    def on_log_message(self, message: str) -> None:
        """Display log message in the log window."""
        self.log_display.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _log_progress(self, message: str) -> None:
        """Log progress messages from transcriber."""
        logger.info(f"[PROGRESS] {message}")

    def select_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.mp3 *.wav);;All Files (*)",
        )
        if not file_path:
            return
        self.video_path = Path(file_path)
        logger.info(f"Selected video file: {file_path}")
        self.status_label.setText("Status: transcribing")
        self.export_all_button.setEnabled(False)
        self.clips = []
        self.clip_list.clear()
        self.srt_viewer.clear()
        logger.info("Loading video for preview...")
        self.preview_player.load_media(file_path)

        logger.info("Starting transcription worker...")
        self._run_worker(self._transcribe_and_find_clips, self.on_transcription_ready, self.on_error)

    def _transcribe_and_find_clips(self):
        logger.info(f"Transcribing: {self.video_path}")
        logger.info("=" * 60)
        result = self.transcriber.transcribe(str(self.video_path))
        logger.info("=" * 60)
        logger.info(f"✓ Transcription complete: {len(result.segments)} segments")

        logger.info("Finding clips using ClipsAI...")
        clips = self.clip_wrapper.find_clips(result.segments, max_clips=6)
        logger.info(f"Found {len(clips)} clips")
        return result, clips

    def on_transcription_ready(self, payload) -> None:
        logger.info("Processing transcription results...")
        result, clips = payload
        self.transcription = result
        self.clips = clips

        logger.info(f"Setting up viewer with {len(result.segments)} segments and {len(clips)} clips")

        # Display segments in the SRT viewer
        logger.info("Displaying transcript in viewer...")
        self.srt_viewer.set_segments(result.segments)
        logger.info(f"✓ Transcript displayed: {len(result.segments)} segments visible")

        # Populate clips
        self.populate_clips()
        self.export_all_button.setEnabled(bool(self.clips))
        self.status_label.setText(f"Status: {len(self.clips)} clips found")
        logger.info(f"✓ Ready to export {len(self.clips)} clips")

    def populate_clips(self) -> None:
        self.clip_list.clear()
        for idx, clip in enumerate(self.clips):
            item = QListWidgetItem()
            widget = ClipListItemWidget(clip, idx)
            widget.export_requested.connect(self.export_clip)
            item.setSizeHint(widget.sizeHint())
            self.clip_list.addItem(item)
            self.clip_list.setItemWidget(item, widget)

    def preview_clip(self, index: int) -> None:
        if index < 0 or index >= len(self.clips):
            return
        clip = self.clips[index]
        self.preview_player.set_markers(clip.start_time, clip.end_time)
        self.preview_player.seek_seconds(clip.start_time)
        self.status_label.setText(f"Status: previewing clip {index + 1}")

    def export_clip(self, index: int) -> None:
        if not self.video_path:
            return
        if index < 0 or index >= len(self.clips):
            return
        clip = self.clips[index]
        self.status_label.setText(f"Status: exporting clip {index + 1}")
        self._run_worker(
            lambda: self._export_single_clip(index, clip),
            self.on_export_done,
            self.on_error,
        )

    def _export_single_clip(self, index: int, clip: Clip) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"clip_{index + 1:02d}.mp4"
        self.clip_wrapper.trim_clip(self.video_path, clip.start_time, clip.end_time, output_path)
        return output_path

    def export_all(self) -> None:
        if not self.video_path or not self.clips:
            return
        self.status_label.setText("Status: exporting all clips")
        self._run_worker(self._export_all_clips, self.on_export_done, self.on_error)

    def _export_all_clips(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clip_wrapper.trim_all_clips(self.video_path, self.clips, self.output_dir)
        return self.output_dir

    def on_export_done(self, output_path) -> None:
        self.status_label.setText(f"Status: export complete ({output_path})")

    def on_marker_changed(self, start_seconds: float, end_seconds: float) -> None:
        self.preview_player.set_markers(start_seconds, end_seconds)
        self.preview_player.seek_seconds(start_seconds)

    def srt_viewer_highlight(self, seconds: float) -> None:
        self.srt_viewer.highlight_for_time(seconds)

    def select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if directory:
            self.output_dir = Path(directory)
            self.output_label.setText(str(self.output_dir))

    def on_error(self, message: str) -> None:
        logger.error(f"Error occurred: {message}")
        self.status_label.setText("Status: error")
        QMessageBox.critical(self, "Error", message)

    def _run_worker(self, func, on_finished, on_error) -> None:
        thread = QThread(self)
        worker = Worker(func)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._threads.remove(thread) if thread in self._threads else None)
        self._threads.append(thread)
        thread.start()


def main() -> None:
    app = QApplication(sys.argv)
    window = ClipEditor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
