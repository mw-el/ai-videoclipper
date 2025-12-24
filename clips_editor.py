from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
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
        logger.info(f"[WORKER] Worker created for {func.__name__} with args={args}, kwargs={kwargs}")

    def run(self) -> None:
        try:
            import time
            print(f"[WORKER THREAD ACTUAL RUN CALLED] {self.func.__name__}", file=__import__('sys').stderr, flush=True)
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

        # Setup logging FIRST before any logging calls
        log_file = Path(__file__).resolve().parent / "logs" / "ai_videoclipper.log"
        self.logger, self.log_emitter = setup_logging(log_file)
        logger.info(f"Logging to: {log_file}")

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
        self.auto_scene_detection: bool = False  # Scene detection mode (False=Manual, True=Auto)

        self._threads: list[QThread] = []

        self._build_ui()

    def _build_ui(self) -> None:
        container = QWidget(self)
        main_layout = QVBoxLayout(container)

        # Top bar: Select File button, Mode checkboxes, and Status label
        top_bar = QHBoxLayout()
        self.open_button = QPushButton("Select File")
        self.open_button.clicked.connect(self.select_file)
        top_bar.addWidget(self.open_button)

        # Scene detection mode checkboxes (mutually exclusive)
        top_bar.addSpacing(20)
        label_mode = QLabel("Scene Detection:")
        top_bar.addWidget(label_mode)

        self.checkbox_manual = QCheckBox("Manual (default)")
        self.checkbox_manual.setChecked(True)
        self.checkbox_manual.stateChanged.connect(self._on_manual_mode_toggled)
        top_bar.addWidget(self.checkbox_manual)

        self.checkbox_auto = QCheckBox("Auto (ClipsAI)")
        self.checkbox_auto.setChecked(False)
        self.checkbox_auto.stateChanged.connect(self._on_auto_mode_toggled)
        top_bar.addWidget(self.checkbox_auto)

        top_bar.addStretch()
        self.status_label = QLabel("Status: idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top_bar.addWidget(self.status_label)

        main_layout.addLayout(top_bar)

        # MAIN BODY: Horizontal layout with sliders/SRT on left, video+clips on right
        body_layout = QHBoxLayout()

        # LEFT PANEL: Sliders and SRT Viewer (much broader, reaches to video)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Preview player (contains sliders)
        self.preview_player = PreviewPlayer()
        left_layout.addWidget(self.preview_player)

        # SRT Viewer (fills remaining space)
        self.srt_viewer = SRTViewer()
        self.srt_viewer.marker_changed.connect(self.on_marker_changed)
        self.srt_viewer.segment_clicked.connect(self._on_srt_segment_clicked)
        left_layout.addWidget(self.srt_viewer, stretch=1)

        body_layout.addWidget(left_panel, stretch=1)

        # RIGHT PANEL: Video preview (top) + Buttons (under video) + Clip list
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Video preview at top
        self.preview_player.video_widget.setMinimumHeight(200)
        self.preview_player.video_widget.setMinimumWidth(int(200 * 16 / 9))
        self.preview_player.video_widget.setMaximumHeight(200)
        right_layout.addWidget(self.preview_player.video_widget, stretch=0)

        # Toolbar with editing buttons (under video)
        self.clip_toolbar = ClipToolbar()
        self.clip_toolbar.set_start_clicked.connect(self._on_set_start)
        self.clip_toolbar.set_end_clicked.connect(self._on_set_end)
        self.clip_toolbar.duplicate_clicked.connect(self._on_duplicate_clip)
        self.clip_toolbar.split_clicked.connect(self._on_split_clip)
        self.clip_toolbar.export_all_clicked.connect(self.export_all)
        self.clip_toolbar.load_config_clicked.connect(self.load_clips_config)
        self.clip_toolbar.save_config_clicked.connect(self.save_clips_config)
        self.clip_toolbar.setContentsMargins(8, 4, 8, 4)  # Padding left/right
        self.clip_toolbar.setMinimumHeight(40)
        self.clip_toolbar.setMaximumHeight(40)
        right_layout.addWidget(self.clip_toolbar, stretch=0)

        # Clip list widget (fills remaining space)
        self.clip_list_widget = ClipListWidget()
        self.clip_list_widget.srt_viewer = self.srt_viewer  # Link SRT viewer for text extraction
        self.clip_list_widget.clip_selected.connect(self._on_clip_selected)
        self.clip_list_widget.new_clip_requested.connect(self._on_new_clip)
        self.clip_list_widget.delete_clip_requested.connect(self._on_delete_clip)
        # When highlight range changes in SRT viewer, update currently selected clip display
        self.srt_viewer.highlight_range_changed.connect(
            lambda start, end: self.clip_list_widget.update_clip_display(self.clip_list_widget.current_clip_index)
            if self.clip_list_widget.current_clip_index >= 0 else None
        )
        right_layout.addWidget(self.clip_list_widget, stretch=1)

        body_layout.addWidget(right_panel, stretch=0)

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
        self.clips = []
        self.srt_viewer.clear()
        logger.info("Loading video for preview...")
        self.preview_player.load_media(file_path)

        logger.info("Starting transcription worker...")
        self._run_worker(self._transcribe_and_find_clips, self.on_transcription_ready, self.on_error)

    def _on_manual_mode_toggled(self, state: int) -> None:
        """Handle manual mode checkbox toggle."""
        if state:  # Manual checkbox is checked
            self.auto_scene_detection = False
            self.checkbox_auto.blockSignals(True)
            self.checkbox_auto.setChecked(False)
            self.checkbox_auto.blockSignals(False)
            logger.info("Scene detection mode: MANUAL (create full transcript + default full-video scene)")

    def _on_auto_mode_toggled(self, state: int) -> None:
        """Handle auto mode checkbox toggle."""
        if state:  # Auto checkbox is checked
            self.auto_scene_detection = True
            self.checkbox_manual.blockSignals(True)
            self.checkbox_manual.setChecked(False)
            self.checkbox_manual.blockSignals(False)
            logger.info("Scene detection mode: AUTO (ClipsAI auto-detect clips)")

    def _setup_output_dir(self) -> None:
        """Set up output directory based on source video location with timestamp."""
        if not self.video_path:
            logger.error("Video path not set")
            return

        # Create folder: {video_name}_clips/{timestamp} in same directory as video
        video_dir = self.video_path.parent
        video_name = self.video_path.stem  # Name without extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = video_dir / f"{video_name}_clips" / timestamp

        logger.info(f"Output directory will be: {self.output_dir}")

    def _transcribe_and_find_clips(self):
        logger.info(f"Transcribing: {self.video_path}")
        logger.info("=" * 60)
        result = self.transcriber.transcribe(str(self.video_path))
        logger.info("=" * 60)
        logger.info(f"✓ Transcription complete: {len(result.segments)} segments")

        if self.auto_scene_detection:
            logger.info("Finding clips using ClipsAI (AUTO mode)...")
            clips = self.clip_wrapper.find_clips(result.segments, max_clips=6)
            logger.info(f"Found {len(clips)} clips")
        else:
            logger.info("Creating default full-video clip (MANUAL mode)...")
            if result.segments:
                # Create a single clip spanning the entire video
                first_segment = result.segments[0]
                last_segment = result.segments[-1]
                default_clip = Clip(
                    start_time=first_segment.start,
                    end_time=last_segment.end,
                    text="Full Video",
                    score=None,
                    segment_start_index=0,
                    segment_end_index=len(result.segments) - 1
                )
                clips = [default_clip]
                logger.info("Created 1 default full-video clip")
            else:
                logger.warning("No segments found in transcription")
                clips = []

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

        # Setup preview player with segments for keyboard navigation
        logger.info("[PREVIEW] Setting segments for keyboard navigation...")
        try:
            self.preview_player.set_segments(result.segments)
            logger.info(f"[PREVIEW] ✓ Segments set for slider keyboard navigation")
        except Exception as e:
            logger.error(f"[PREVIEW] Failed to set segments: {e}")
            import traceback
            logger.error(f"[PREVIEW] Traceback: {traceback.format_exc()}")

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
        try:
            self.populate_clips()
            logger.info(f"[CLIPS] ✓ Populated {len(self.clips)} clips in list widget")
        except Exception as e:
            logger.error(f"[CLIPS] Failed to populate clips: {e}")
            import traceback
            logger.error(f"[CLIPS] Traceback: {traceback.format_exc()}")
            raise

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

    def _text_to_slug(self, text: str) -> str:
        """
        Convert text to a filesystem-safe slug.
        Takes first few words and converts to lowercase with underscores.
        """
        if not text:
            return ""

        # Get first 3-4 words
        words = text.split()[:4]
        if not words:
            return ""

        # Join words and sanitize for filesystem
        slug = "_".join(words).lower()
        # Remove non-alphanumeric chars except underscore and hyphen
        slug = ''.join(c if c.isalnum() or c in '-_' else '' for c in slug)
        # Clean up multiple underscores
        while '__' in slug:
            slug = slug.replace('__', '_')
        slug = slug.strip('_')

        return slug if slug else ""

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

    def _get_clip_slug(self, clip: Clip) -> str:
        """
        Get slug from the first segment that falls within the clip's time range.
        """
        if not self.transcription or not self.transcription.segments:
            return ""

        # Find first segment within clip time range
        for segment in self.transcription.segments:
            if segment.start >= clip.start_time:
                # Use first few words from this segment
                return self._text_to_slug(segment.text)

        # Fallback to clip text if no segments found
        return self._text_to_slug(clip.text)

    def _export_clip_srt(self, clip: Clip, srt_path: Path) -> None:
        """
        Export SRT file for a clip with all segments within the clip's time range.
        Adjusts timing so the clip starts at 0:00:00.
        """
        from srt_utils import SrtSegment, segments_to_srt_text

        if not self.transcription or not self.transcription.segments:
            logger.warning(f"No transcription segments available for SRT export")
            return

        # Collect all segments that fall within the clip's time range
        clip_segments = []
        for segment in self.transcription.segments:
            # Check if segment overlaps with clip time range
            if segment.end > clip.start_time and segment.start < clip.end_time:
                # Adjust timing: subtract clip start time so clip starts at 0
                adjusted_start = max(0, segment.start - clip.start_time)
                adjusted_end = min(clip.end_time - clip.start_time, segment.end - clip.start_time)

                adjusted_segment = SrtSegment(
                    index=len(clip_segments) + 1,
                    start=adjusted_start,
                    end=adjusted_end,
                    text=segment.text
                )
                clip_segments.append(adjusted_segment)

        if not clip_segments:
            logger.warning(f"No segments found for clip in time range [{clip.start_time}, {clip.end_time}]")
            return

        # Generate SRT content and write to file
        srt_content = segments_to_srt_text(clip_segments)
        logger.info(f"Saving clip SRT with {len(clip_segments)} segments to: {srt_path}")
        srt_path.write_text(srt_content, encoding="utf-8")
        logger.info(f"✓ SRT exported: {len(clip_segments)} segments, adjusted timing from clip start")

    def _export_single_clip(self, index: int, clip: Clip) -> Path:
        if not self.output_dir:
            logger.error("Output directory not set")
            raise RuntimeError("Output directory not set")

        # Create main clips folder
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get slug from first segment in the clip
        clip_slug = self._get_clip_slug(clip)
        if not clip_slug:
            clip_slug = f"clip_{index + 1:02d}"

        # Create folder using the slug name
        clip_folder = self.output_dir / clip_slug
        clip_folder.mkdir(parents=True, exist_ok=True)

        # Export video clip - use slug for video filename
        video_path = clip_folder / f"{clip_slug}.mp4"
        logger.info(f"Exporting clip {index + 1} video to: {video_path}")
        self.clip_wrapper.trim_clip(self.video_path, clip.start_time, clip.end_time, video_path)

        # Export SRT file with all segments within the clip, adjusted timing
        srt_path = clip_folder / f"{clip_slug}.srt"
        self._export_clip_srt(clip, srt_path)

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

    def load_clips_config(self) -> None:
        """Load clip configuration from JSON file."""
        if not self.transcription:
            QMessageBox.warning(self, "No Transcription", "Please transcribe a video first before loading clips configuration.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Clips Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        try:
            logger.info(f"[CONFIG] Loading clips configuration from: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            mode = config.get("mode", "auto")
            logger.info(f"[CONFIG] Configuration mode: {mode}")

            if mode == "auto":
                self._load_auto_clips(config)
            elif mode == "manual":
                self._load_manual_clips(config)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            logger.info(f"[CONFIG] ✓ Successfully loaded {len(self.clips)} clips")
            self.populate_clips()
            self.status_label.setText(f"Status: loaded {len(self.clips)} clips from config")

        except Exception as e:
            logger.error(f"[CONFIG] Failed to load clips config: {e}", exc_info=True)
            QMessageBox.critical(self, "Error Loading Config", f"Failed to load clips configuration:\n{str(e)}")

    def _load_auto_clips(self, config: dict) -> None:
        """Load auto-detected clips configuration."""
        max_clips = config.get("max_clips", 6)
        logger.info(f"[CONFIG] Auto-mode: finding clips with max_clips={max_clips}")

        # Use ClipsAI to find clips
        clips = self.clip_wrapper.find_clips(self.transcription.segments, max_clips=max_clips)
        clips = self.clip_wrapper._add_segment_indices(clips, self.transcription.segments)
        self.clips = clips
        logger.info(f"[CONFIG] ✓ Auto-mode found {len(clips)} clips")

    def _load_manual_clips(self, config: dict) -> None:
        """Load manually defined clips configuration."""
        selection_type = config.get("selection_type")
        clips_data = config.get("clips", [])

        if not selection_type:
            raise ValueError("Manual mode requires 'selection_type' field")

        if selection_type == "time":
            self._load_manual_clips_by_time(clips_data)
        elif selection_type == "segments":
            self._load_manual_clips_by_segments(clips_data)
        else:
            raise ValueError(f"Unknown selection_type: {selection_type}")

    def _load_manual_clips_by_time(self, clips_data: list) -> None:
        """Load clips defined by start_time and end_time."""
        logger.info(f"[CONFIG] Manual-mode (by time): loading {len(clips_data)} clips")

        clips = []
        for i, clip_data in enumerate(clips_data):
            start_time = float(clip_data.get("start_time"))
            end_time = float(clip_data.get("end_time"))
            name = clip_data.get("name", f"Clip {i + 1}")

            if end_time <= start_time:
                raise ValueError(f"Clip {i + 1}: end_time must be greater than start_time")

            clip = Clip(start_time=start_time, end_time=end_time, text=name)
            clips.append(clip)
            logger.info(f"[CONFIG]   Clip {i + 1}: {name} ({format_timestamp(start_time)} - {format_timestamp(end_time)})")

        # Add segment indices
        clips = self.clip_wrapper._add_segment_indices(clips, self.transcription.segments)
        self.clips = clips
        logger.info(f"[CONFIG] ✓ Loaded {len(clips)} clips by time")

    def _load_manual_clips_by_segments(self, clips_data: list) -> None:
        """Load clips defined by segment indices (1-indexed)."""
        logger.info(f"[CONFIG] Manual-mode (by segments): loading {len(clips_data)} clips")

        segments = self.transcription.segments
        clips = []

        for i, clip_data in enumerate(clips_data):
            start_seg = int(clip_data.get("start_segment")) - 1  # Convert to 0-indexed
            end_seg = int(clip_data.get("end_segment")) - 1      # Convert to 0-indexed
            name = clip_data.get("name", f"Clip {i + 1}")

            if start_seg < 0 or end_seg >= len(segments):
                raise ValueError(f"Clip {i + 1}: segment index out of range (valid: 1-{len(segments)})")

            if end_seg < start_seg:
                raise ValueError(f"Clip {i + 1}: end_segment must be >= start_segment")

            # Get timing from segments
            start_time = segments[start_seg].start
            end_time = segments[end_seg].end

            clip = Clip(
                start_time=start_time,
                end_time=end_time,
                text=name,
                segment_start_index=start_seg,
                segment_end_index=end_seg
            )
            clips.append(clip)
            logger.info(f"[CONFIG]   Clip {i + 1}: {name} (segments {start_seg + 1}-{end_seg + 1})")

        self.clips = clips
        logger.info(f"[CONFIG] ✓ Loaded {len(clips)} clips by segments")

    def save_clips_config(self) -> None:
        """Save current clips to a JSON configuration file."""
        if not self.clips:
            QMessageBox.warning(self, "No Clips", "No clips to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Clips Configuration",
            "clips.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        try:
            logger.info(f"[CONFIG] Saving {len(self.clips)} clips to: {file_path}")

            # Create config with both time and segment information
            config = {
                "mode": "manual",
                "selection_type": "time",
                "clips": []
            }

            for i, clip in enumerate(self.clips):
                clip_entry = {
                    "name": clip.text or f"Clip {i + 1}",
                    "start_time": round(clip.start_time, 3),
                    "end_time": round(clip.end_time, 3)
                }
                config["clips"].append(clip_entry)
                logger.info(f"[CONFIG]   Clip {i + 1}: {clip_entry['name']} ({clip_entry['start_time']}s - {clip_entry['end_time']}s)")

            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info(f"[CONFIG] ✓ Successfully saved clips configuration")
            self.status_label.setText(f"Status: saved clips config to {Path(file_path).name}")
            QMessageBox.information(self, "Success", f"Clips configuration saved to:\n{file_path}")

        except Exception as e:
            logger.error(f"[CONFIG] Failed to save clips config: {e}", exc_info=True)
            QMessageBox.critical(self, "Error Saving Config", f"Failed to save clips configuration:\n{str(e)}")

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
        logger.info(f"[RUN_WORKER] ============ STARTING WORKER THREAD ============")
        logger.info(f"[RUN_WORKER] Function: {func.__name__}")
        logger.info(f"[RUN_WORKER] on_finished: {on_finished.__name__ if hasattr(on_finished, '__name__') else on_finished}")
        logger.info(f"[RUN_WORKER] on_error: {on_error.__name__ if hasattr(on_error, '__name__') else on_error}")

        thread = QThread(self)
        logger.info(f"[RUN_WORKER] Created QThread: {thread}")

        worker = Worker(func)
        logger.info(f"[RUN_WORKER] Created Worker: {worker}")

        worker.moveToThread(thread)
        logger.info(f"[RUN_WORKER] Worker moved to thread")

        thread.started.connect(worker.run)
        logger.info(f"[RUN_WORKER] ✓ Connected thread.started -> worker.run")

        worker.finished.connect(on_finished)
        logger.info(f"[RUN_WORKER] ✓ Connected worker.finished -> on_finished")

        worker.error.connect(on_error)
        logger.info(f"[RUN_WORKER] ✓ Connected worker.error -> on_error")

        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._threads.remove(thread) if thread in self._threads else None)

        self._threads.append(thread)
        logger.info(f"[RUN_WORKER] About to start thread (total threads before: {len(self._threads)})")
        thread.start()
        logger.info(f"[RUN_WORKER] ✓✓✓ THREAD STARTED! ✓✓✓ (total threads: {len(self._threads)})")


def main() -> None:
    try:
        print("\n" + "="*80, file=sys.stderr, flush=True)
        print("[MAIN] Starting AI VideoClipper...", file=sys.stderr, flush=True)
        print("="*80, file=sys.stderr, flush=True)

        try:
            print("[MAIN] Creating QApplication...", file=sys.stderr, flush=True)
            app = QApplication(sys.argv)
            print("[MAIN] ✓ QApplication created", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[MAIN] FATAL: Failed to create QApplication: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        try:
            print("[MAIN] Creating ClipEditor window...", file=sys.stderr, flush=True)
            window = ClipEditor()
            print("[MAIN] ✓ ClipEditor window created successfully", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[MAIN] FATAL: Failed to create ClipEditor: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        try:
            print("[MAIN] Showing window...", file=sys.stderr, flush=True)
            window.show()
            print("[MAIN] ✓ Window shown", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[MAIN] ERROR showing window: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        print("[MAIN] Starting event loop...", file=sys.stderr, flush=True)
        print("="*80 + "\n", file=sys.stderr, flush=True)
        exit_code = app.exec()
        print(f"\n[MAIN] Event loop exited with code {exit_code}", file=sys.stderr, flush=True)
        sys.exit(exit_code)

    except Exception as e:
        print(f"\n[MAIN] FATAL UNCAUGHT EXCEPTION: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
