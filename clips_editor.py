from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
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
from clip_list_widget import ClipListWidget
from clip_toolbar import ClipToolbar
from new_clip_dialog import NewClipDialog

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
            logger.info(f"[WORKER] Result type: {type(result)}")
            if hasattr(result, '__len__'):
                logger.info(f"[WORKER] Result length: {len(result)}")
            print(f"[WORKER THREAD] Completed {self.func.__name__}", file=__import__('sys').stderr, flush=True)
        except Exception as exc:
            logger.exception(f"[WORKER] ERROR in {self.func.__name__}: {exc}")
            print(f"[WORKER THREAD] ERROR: {exc}", file=__import__('sys').stderr, flush=True)
            self.error.emit(str(exc))
        else:
            logger.info(f"[WORKER] Emitting finished signal with result...")
            self.finished.emit(result)
            logger.info(f"[WORKER] Finished signal emitted")


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
        self.output_dir: Path | None = None  # Will be set based on source video location
        self.last_clicked_segment_index: int = -1  # For Set Start/End operations

        self._threads: list[QThread] = []

        # Setup logging window
        log_file = Path(__file__).resolve().parent / "logs" / "ai_videoclipper.log"
        self.logger, self.log_emitter = setup_logging(log_file)
        logger.info(f"Logging to: {log_file}")

        self._build_ui()

    def _build_ui(self) -> None:
        container = QWidget(self)
        main_layout = QVBoxLayout(container)

        # Top bar: Select File button and Status label
        top_bar = QHBoxLayout()
        self.open_button = QPushButton("Select File")
        self.open_button.clicked.connect(self.select_file)
        self.status_label = QLabel("Status: idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top_bar.addWidget(self.open_button)
        top_bar.addStretch()
        top_bar.addWidget(self.status_label)

        main_layout.addLayout(top_bar)

        # Body: Two-column layout (clips on left, SRT viewer on right)
        body_layout = QHBoxLayout()

        # LEFT PANEL: Clip list (narrow, 200px)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.clip_list_widget = ClipListWidget()
        self.clip_list_widget.clip_selected.connect(self._on_clip_selected)
        self.clip_list_widget.new_clip_requested.connect(self._on_new_clip)
        self.clip_list_widget.delete_clip_requested.connect(self._on_delete_clip)

        left_layout.addWidget(self.clip_list_widget)

        self.output_label = QLabel("Output: (auto-determined from video location)")
        self.output_label.setWordWrap(True)
        self.output_label.setMaximumHeight(60)
        left_layout.addWidget(self.output_label)

        # RIGHT PANEL: Toolbar + SRT Viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar with editing buttons
        self.clip_toolbar = ClipToolbar()
        self.clip_toolbar.set_start_clicked.connect(self._on_set_start)
        self.clip_toolbar.set_end_clicked.connect(self._on_set_end)
        self.clip_toolbar.duplicate_clicked.connect(self._on_duplicate_clip)
        self.clip_toolbar.split_clicked.connect(self._on_split_clip)
        self.clip_toolbar.export_all_clicked.connect(self.export_all)
        right_layout.addWidget(self.clip_toolbar)

        # SRT Viewer
        self.srt_viewer = SRTViewer()
        self.srt_viewer.marker_changed.connect(self.on_marker_changed)
        self.srt_viewer.segment_clicked.connect(self._on_srt_segment_clicked)
        right_layout.addWidget(self.srt_viewer)

        # Add panels to body layout
        body_layout.addWidget(left_panel, stretch=0)
        body_layout.setStretchFactor(left_panel, 0)
        # Set minimum width for left panel
        left_panel.setMinimumWidth(200)
        left_panel.setMaximumWidth(250)

        body_layout.addWidget(right_panel, stretch=1)

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

        # Determine output directory based on source video location
        self._setup_output_dir()

        self.status_label.setText("Status: transcribing")
        self.export_all_button.setEnabled(False)
        self.clips = []
        self.clip_list.clear()
        self.srt_viewer.clear()
        logger.info("Loading video for preview...")
        self.preview_player.load_media(file_path)

        logger.info("Starting transcription worker...")
        self._run_worker(self._transcribe_and_find_clips, self.on_transcription_ready, self.on_error)

    def _setup_output_dir(self) -> None:
        """Set up output directory based on source video location."""
        if not self.video_path:
            logger.error("Video path not set")
            return

        # Create folder: {video_name}_clips in same directory as video
        video_dir = self.video_path.parent
        video_name = self.video_path.stem  # Name without extension
        self.output_dir = video_dir / f"{video_name}_clips"

        logger.info(f"Output directory will be: {self.output_dir}")
        self.output_label.setText(f"Output: {self.output_dir}")

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
        logger.info(f"[CALLBACK] on_transcription_ready called with payload type: {type(payload)}")

        try:
            result, clips = payload
            logger.info(f"[CALLBACK] Unpacked payload: result={type(result)}, clips={type(clips)}")
        except Exception as e:
            logger.error(f"[CALLBACK] Failed to unpack payload: {e}")
            raise

        self.transcription = result
        self.clips = clips

        logger.info(f"[CALLBACK] Result has {len(result.segments)} segments, {len(clips)} clips found")
        logger.info(f"[CALLBACK] First segment: {result.segments[0] if result.segments else 'NO SEGMENTS'}")

        logger.info(f"Setting up viewer with {len(result.segments)} segments and {len(clips)} clips")

        # Display segments in the SRT viewer
        logger.info("[DISPLAY] Calling srt_viewer.set_segments()...")
        try:
            self.srt_viewer.set_segments(result.segments)
            logger.info(f"[DISPLAY] ✓ set_segments() completed")
            logger.info(f"[DISPLAY] SRT viewer now has {len(self.srt_viewer._segments)} segments")
        except Exception as e:
            logger.error(f"[DISPLAY] Failed to set segments: {e}")
            import traceback
            logger.error(f"[DISPLAY] Traceback: {traceback.format_exc()}")
            raise

        logger.info(f"✓ Transcript displayed: {len(result.segments)} segments visible")

        # Populate clips
        logger.info("[CLIPS] Populating clips list...")
        self.populate_clips()
        logger.info(f"[CLIPS] ✓ Populated {len(self.clips)} clips in list widget")

        self.export_all_button.setEnabled(bool(self.clips))
        self.status_label.setText(f"Status: {len(self.clips)} clips found")
        logger.info(f"✓ Ready to export {len(self.clips)} clips")

    def populate_clips(self) -> None:
        """Populate the clip list widget with clips."""
        self.clip_list_widget.set_clips(self.clips)
        # Select first clip and show it in SRT viewer
        if self.clips:
            self._on_clip_selected(0)

    def _on_clip_selected(self, clip_index: int) -> None:
        """Handle clip selection from clip list."""
        if clip_index < 0 or clip_index >= len(self.clips):
            return
        clip = self.clips[clip_index]
        # Highlight the clip's segment range in the SRT viewer
        self.srt_viewer.highlight_segment_range(
            clip.segment_start_index,
            clip.segment_end_index,
            auto_scroll=True
        )
        self.status_label.setText(f"Status: clip {clip_index + 1} selected")

    def _on_srt_segment_clicked(self, segment_index: int) -> None:
        """Handle segment click in SRT viewer. Remember for Set Start/End."""
        self.last_clicked_segment_index = segment_index
        logger.info(f"[CLIP_EDIT] User clicked segment {segment_index + 1}")

    def _on_set_start(self) -> None:
        """Set the start boundary of current clip to last clicked segment."""
        current_clip = self.clip_list_widget.get_current_clip()
        if not current_clip or not hasattr(self, 'last_clicked_segment_index'):
            logger.warning("[CLIP_EDIT] No clip selected or no segment clicked")
            return
        if self.last_clicked_segment_index >= len(self.transcription.segments):
            return
        seg = self.transcription.segments[self.last_clicked_segment_index]
        current_clip.start_time = seg.start
        current_clip.segment_start_index = self.last_clicked_segment_index
        logger.info(f"[CLIP_EDIT] Set clip start to segment {self.last_clicked_segment_index + 1} at {seg.start}s")
        self._on_clip_selected(self.clip_list_widget.current_clip_index)

    def _on_set_end(self) -> None:
        """Set the end boundary of current clip to last clicked segment."""
        current_clip = self.clip_list_widget.get_current_clip()
        if not current_clip or not hasattr(self, 'last_clicked_segment_index'):
            logger.warning("[CLIP_EDIT] No clip selected or no segment clicked")
            return
        if self.last_clicked_segment_index >= len(self.transcription.segments):
            return
        seg = self.transcription.segments[self.last_clicked_segment_index]
        current_clip.end_time = seg.end
        current_clip.segment_end_index = self.last_clicked_segment_index
        logger.info(f"[CLIP_EDIT] Set clip end to segment {self.last_clicked_segment_index + 1} at {seg.end}s")
        self._on_clip_selected(self.clip_list_widget.current_clip_index)

    def _on_duplicate_clip(self) -> None:
        """Duplicate the currently selected clip."""
        current_idx = self.clip_list_widget.current_clip_index
        if current_idx < 0 or current_idx >= len(self.clips):
            return
        clip_to_dup = self.clips[current_idx]
        # Create a new clip with same boundaries
        new_clip = Clip(
            start_time=clip_to_dup.start_time,
            end_time=clip_to_dup.end_time,
            text=clip_to_dup.text,
            score=clip_to_dup.score,
            segment_start_index=clip_to_dup.segment_start_index,
            segment_end_index=clip_to_dup.segment_end_index,
        )
        self.clips.insert(current_idx + 1, new_clip)
        self.populate_clips()
        logger.info(f"[CLIP_EDIT] Duplicated clip {current_idx + 1}")

    def _on_split_clip(self) -> None:
        """Split the current clip at the last clicked segment."""
        current_idx = self.clip_list_widget.current_clip_index
        if current_idx < 0 or current_idx >= len(self.clips):
            return
        if not hasattr(self, 'last_clicked_segment_index'):
            logger.warning("[CLIP_EDIT] No segment clicked for split")
            return
        clip = self.clips[current_idx]
        split_seg = self.transcription.segments[self.last_clicked_segment_index]

        # Create second half of split clip
        new_clip = Clip(
            start_time=split_seg.start,
            end_time=clip.end_time,
            text=clip.text,
            score=clip.score,
            segment_start_index=self.last_clicked_segment_index,
            segment_end_index=clip.segment_end_index,
        )
        # Adjust first half
        clip.end_time = split_seg.start
        clip.segment_end_index = self.last_clicked_segment_index - 1

        self.clips.insert(current_idx + 1, new_clip)
        self.populate_clips()
        logger.info(f"[CLIP_EDIT] Split clip {current_idx + 1} at segment {self.last_clicked_segment_index + 1}")

    def _on_delete_clip(self, clip_index: int) -> None:
        """Delete a clip from the list."""
        if 0 <= clip_index < len(self.clips):
            del self.clips[clip_index]
            self.populate_clips()
            logger.info(f"[CLIP_EDIT] Deleted clip {clip_index + 1}")

    def _on_new_clip(self) -> None:
        """Create a new clip by selecting segment range."""
        if not self.transcription or not self.transcription.segments:
            QMessageBox.warning(self, "No Transcription", "No transcription available to create clips from.")
            return

        max_seg_idx = len(self.transcription.segments) - 1
        dialog = NewClipDialog(max_seg_idx, self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            start_idx, end_idx = dialog.get_segment_range()
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            # Get time boundaries from segments
            start_seg = self.transcription.segments[start_idx]
            end_seg = self.transcription.segments[end_idx]

            # Combine text from all segments in range
            text_parts = []
            for i in range(start_idx, end_idx + 1):
                text_parts.append(self.transcription.segments[i].text)
            clip_text = " ".join(text_parts)

            # Create new clip
            new_clip = Clip(
                start_time=start_seg.start,
                end_time=end_seg.end,
                text=clip_text,
                score=None,
                segment_start_index=start_idx,
                segment_end_index=end_idx,
            )
            self.clips.append(new_clip)
            self.populate_clips()
            logger.info(f"[CLIP_EDIT] Created new clip from segments {start_idx + 1}-{end_idx + 1}")

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

    def _get_clip_folder_name(self, index: int, clip: Clip) -> str:
        """
        Generate folder name from first 5-6 words of clip text.
        Falls back to clip number if text is too short.
        """
        if not clip.text:
            return f"clip_{index + 1:02d}"

        # Get first 5-6 words from clip text
        words = clip.text.split()[:6]
        if not words:
            return f"clip_{index + 1:02d}"

        # Join words and sanitize for filesystem
        folder_name = "_".join(words)
        # Remove non-alphanumeric chars except underscore and hyphen
        folder_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in folder_name)
        # Clean up multiple underscores
        while '__' in folder_name:
            folder_name = folder_name.replace('__', '_')
        folder_name = folder_name.strip('_')

        return folder_name if folder_name else f"clip_{index + 1:02d}"

    def _export_single_clip(self, index: int, clip: Clip) -> Path:
        if not self.output_dir:
            logger.error("Output directory not set")
            raise RuntimeError("Output directory not set")

        # Create main clips folder
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subfolder for this clip based on its text
        clip_folder_name = self._get_clip_folder_name(index, clip)
        clip_folder = self.output_dir / clip_folder_name
        clip_folder.mkdir(parents=True, exist_ok=True)

        # Export video clip
        video_path = clip_folder / f"clip_{index + 1:02d}.mp4"
        logger.info(f"Exporting clip {index + 1} video to: {video_path}")
        self.clip_wrapper.trim_clip(self.video_path, clip.start_time, clip.end_time, video_path)

        # Export SRT file for this clip
        from srt_utils import SrtSegment, segments_to_srt_text
        clip_segment = SrtSegment(
            index=1,
            start=0,  # SRT in clip starts at 0
            end=clip.end_time - clip.start_time,
            text=clip.text
        )
        srt_content = segments_to_srt_text([clip_segment])
        srt_path = clip_folder / f"clip_{index + 1:02d}.srt"
        logger.info(f"Saving clip {index + 1} SRT to: {srt_path}")
        srt_path.write_text(srt_content, encoding="utf-8")

        logger.info(f"✓ Exported clip {index + 1} to: {clip_folder}")
        return clip_folder

    def export_all(self) -> None:
        if not self.video_path or not self.clips:
            return
        self.status_label.setText("Status: exporting all clips")
        self._run_worker(self._export_all_clips, self.on_export_done, self.on_error)

    def _export_all_clips(self) -> Path:
        if not self.output_dir:
            logger.error("Output directory not set")
            raise RuntimeError("Output directory not set")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exporting all {len(self.clips)} clips to: {self.output_dir}")

        for idx, clip in enumerate(self.clips):
            logger.info(f"Exporting clip {idx + 1}/{len(self.clips)}...")
            self._export_single_clip(idx, clip)

        logger.info(f"✓ All {len(self.clips)} clips exported to: {self.output_dir}")
        return self.output_dir

    def on_export_done(self, output_path) -> None:
        self.status_label.setText(f"Status: export complete ({output_path})")

    def on_marker_changed(self, start_seconds: float, end_seconds: float) -> None:
        self.preview_player.set_markers(start_seconds, end_seconds)
        self.preview_player.seek_seconds(start_seconds)

    def srt_viewer_highlight(self, seconds: float) -> None:
        self.srt_viewer.highlight_for_time(seconds)

    def on_error(self, message: str) -> None:
        logger.error(f"Error occurred: {message}")
        self.status_label.setText("Status: error")
        QMessageBox.critical(self, "Error", message)

    def _run_worker(self, func, on_finished, on_error) -> None:
        logger.info(f"[RUN_WORKER] Starting worker thread for {func.__name__}")
        logger.info(f"[RUN_WORKER] on_finished={on_finished.__name__ if hasattr(on_finished, '__name__') else on_finished}")
        logger.info(f"[RUN_WORKER] on_error={on_error.__name__ if hasattr(on_error, '__name__') else on_error}")

        thread = QThread(self)
        worker = Worker(func)
        worker.moveToThread(thread)

        logger.info(f"[RUN_WORKER] Worker moved to thread")

        thread.started.connect(worker.run)
        logger.info(f"[RUN_WORKER] Connected thread.started -> worker.run")

        worker.finished.connect(on_finished)
        logger.info(f"[RUN_WORKER] Connected worker.finished -> on_finished")

        worker.error.connect(on_error)
        logger.info(f"[RUN_WORKER] Connected worker.error -> on_error")

        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._threads.remove(thread) if thread in self._threads else None)

        self._threads.append(thread)
        logger.info(f"[RUN_WORKER] Starting thread...")
        thread.start()
        logger.info(f"[RUN_WORKER] Thread started (total threads: {len(self._threads)})")


def main() -> None:
    app = QApplication(sys.argv)
    window = ClipEditor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
