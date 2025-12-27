from __future__ import annotations

from pathlib import Path
import logging
import json
import os
import pty
import re
import shutil
import subprocess

from PyQt6.QtCore import Qt, QSize, QSocketNotifier, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
    QDialog,
    QDialogButtonBox,
    QSlider,
    QFormLayout,
    QComboBox,
    QInputDialog,
)

from design.icon_manager import IconManager
from design.style_manager import StyleManager
from srt_utils import parse_srt
from scene_detection_pipeline import (
    FaceAnalysisError,
    SceneDetectionConfig,
    SceneDetectionPipeline,
)

logger = logging.getLogger("ai_videoclipper")


class TerminalOutput(QPlainTextEdit):
    def __init__(self, on_key, parent=None) -> None:
        super().__init__(parent)
        self._on_key = on_key
        self.setReadOnly(True)
        # Enable word wrap so long lines are visible
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)

    def keyPressEvent(self, event) -> None:
        if self._on_key(event):
            return
        super().keyPressEvent(event)


class SceneWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(object)

    def __init__(self, func, *args, **kwargs) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            result = self.func(*self.args, **self.kwargs)
        except Exception as exc:
            self.error.emit(exc)
        else:
            self.finished.emit(result)


class ClaudeTerminalWidget(QWidget):
    def __init__(self, work_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self._work_dir = work_dir
        self._process = None
        self._master_fd = None
        self._slave_fd = None
        self._notifier = None

        self._output = TerminalOutput(self._handle_key)
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._output.setFont(font)

        self._status_label = QLabel("Idle")
        self._status_label.setStyleSheet("color: #666; padding: 2px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._output, stretch=1)
        layout.addWidget(self._status_label)

    def start(self) -> None:
        if self._process is not None:
            return
        if shutil.which("claude") is None:
            self._status_label.setText("Claude CLI not found in PATH.")
            return
        self._master_fd, self._slave_fd = pty.openpty()
        os.set_blocking(self._master_fd, False)
        banner = (
            'echo "========================================"; '
            'echo "Claude Code Terminal"; '
            'echo "========================================"; '
            'echo ""; '
        )
        # Use --headless mode to avoid interactive prompts
        # The CLAUDE_CONTEXT_DIR environment variable tells Claude where to find context
        cmd = f"{banner} claude --headless; exec bash"

        env = os.environ.copy()
        # Set Claude context directory if it exists
        context_dir = self._work_dir / ".claude-context"
        if context_dir.exists():
            env["CLAUDE_CONTEXT_DIR"] = str(context_dir)

        self._process = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdin=self._slave_fd,
            stdout=self._slave_fd,
            stderr=self._slave_fd,
            cwd=str(self._work_dir),
            env=env,
            text=False,
        )
        self._status_label.setText("Running")
        self._notifier = QSocketNotifier(self._master_fd, QSocketNotifier.Type.Read, self)
        self._notifier.activated.connect(self._read_output)

    def stop(self) -> None:
        if self._process is None:
            return
        if self._notifier:
            self._notifier.setEnabled(False)
            self._notifier.deleteLater()
            self._notifier = None
        try:
            self._process.terminate()
            self._process.wait(timeout=2)
        except Exception:
            self._process.kill()
        self._process = None
        self._close_fds()
        self._status_label.setText("Stopped")

    def set_work_dir(self, work_dir: Path) -> None:
        if self._process is not None:
            return
        self._work_dir = work_dir

    def _close_fds(self) -> None:
        for fd in (self._master_fd, self._slave_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._master_fd = None
        self._slave_fd = None

    def _read_output(self) -> None:
        if self._master_fd is None:
            return
        while True:
            try:
                data = os.read(self._master_fd, 4096)
            except BlockingIOError:
                return
            except OSError:
                self.stop()
                return
            if not data:
                return
            text = data.decode("utf-8", errors="replace")
            text = _strip_ansi(text)
            self._output.moveCursor(QTextCursor.MoveOperation.End)
            self._output.insertPlainText(text)
            self._output.ensureCursorVisible()

    def _handle_key(self, event) -> bool:
        if self._master_fd is None:
            return False
        key = event.key()
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_C:
                self._write_bytes(b"\x03")
                return True
            if key == Qt.Key.Key_D:
                self._write_bytes(b"\x04")
                return True
        if key == Qt.Key.Key_Backspace:
            self._write_bytes(b"\x7f")
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._write_bytes(b"\r")
            return True
        if key == Qt.Key.Key_Tab:
            self._write_bytes(b"\t")
            return True
        if key == Qt.Key.Key_Left:
            self._write_bytes(b"\x1b[D")
            return True
        if key == Qt.Key.Key_Right:
            self._write_bytes(b"\x1b[C")
            return True
        if key == Qt.Key.Key_Up:
            self._write_bytes(b"\x1b[A")
            return True
        if key == Qt.Key.Key_Down:
            self._write_bytes(b"\x1b[B")
            return True
        text = event.text()
        if text:
            self._write_bytes(text.encode("utf-8"))
            return True
        return False

    def _write_bytes(self, data: bytes) -> None:
        try:
            os.write(self._master_fd, data)
        except OSError:
            self.stop()

    def send_text(self, text: str) -> None:
        if not text:
            return
        self._write_bytes(text.encode("utf-8"))
        self._write_bytes(b"\r")

    def is_running(self) -> bool:
        return self._process is not None

    def status_text(self) -> str:
        return self._status_label.text()

    def closeEvent(self, event) -> None:
        self.stop()
        super().closeEvent(event)


class ClaudePanel(QWidget):
    # Signal emitted when scene data is extracted from Claude response
    scene_data_received = pyqtSignal(dict)  # Emits parsed JSON scene data
    # Signal emitted when clips config should be loaded from Claude output
    load_clips_config = pyqtSignal(dict)  # Emits clips configuration JSON

    @staticmethod
    def _build_scene_detection_icon():
        """Build combined movie+search icon for scene detection button."""
        size = 18
        spacing = 2
        try:
            from PyQt6.QtGui import QIcon
            left_icon = IconManager.create_icon("movie", color="white", size=size)
            right_icon = IconManager.create_icon("search", color="white", size=size)

            pixmap = QPixmap(size * 2 + spacing, size)
            pixmap.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pixmap)
            left_icon.paint(painter, 0, 0, size, size, Qt.AlignmentFlag.AlignCenter)
            right_icon.paint(painter, size + spacing, 0, size, size, Qt.AlignmentFlag.AlignCenter)
            painter.end()
            return QIcon(pixmap)
        except Exception as e:
            print(f"Warning: Could not create scene detection icon: {e}")
            return IconManager.create_icon("search", color="white", size=18)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._prompt_path = Path(__file__).resolve().parent / "scene-detection-prompt.txt"
        self._work_dir = Path(__file__).resolve().parents[1]
        self._video_path: Path | None = None
        self._srt_path: Path | None = None
        self._context_dir: Path | None = None
        self._pending_prompt = False
        self._threads: list[QThread] = []
        self._scene_workers: list[SceneWorker] = []
        self._scene_settings = {
            "weight_text": 0.40,
            "weight_cut": 0.10,
            "weight_audio": 0.20,
            "weight_face": 0.30,
            "incomplete_thought_penalty": 0.08,
            "min_clip_seconds": 30.0,
            "max_clip_seconds": 300.0,
            "speech_ratio_min": 0.2,
            "whisper_bad_ratio_max": 0.8,
            "top_k_max": 7,
        }
        self._scene_settings_name = "default"
        self._scene_settings_dialog: QDialog | None = None
        self._scene_settings_controls = {}
        self._scene_settings_preview_timer: QTimer | None = None
        self._ensure_default_settings_saved()

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #666; padding: 4px;")
        self._analysis_status_label = QLabel("")
        self._analysis_status_label.setStyleSheet("color: #666; padding: 2px 4px; font-size: 11px;")

        self._prompt_button = QPushButton()
        self._prompt_button.setToolTip("Scene Selection Prompt")
        self._prompt_button.clicked.connect(self.send_scene_selection_prompt)
        self._prompt_button.setIcon(self._build_scene_detection_icon())
        self._prompt_button.setIconSize(QSize(38, 18))
        from design.style_manager import Colors
        # Use custom styling for wider button with dark background
        self._prompt_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BLUE};
                border: none;
                border-radius: 4px;
                padding: 3px;
                min-height: 32px;
                min-width: 76px;
            }}
            QPushButton:hover {{
                background-color: {Colors.BRIGHT_BLUE};
            }}
            QPushButton:disabled {{
                background-color: #595959;
            }}
        """)
        self._prompt_button.setEnabled(False)

        self._start_button = QPushButton("Start Claude")
        self._start_button.setToolTip("Start Claude in headless mode")
        self._start_button.clicked.connect(self.start_claude)
        StyleManager.apply_button_style(self._start_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setToolTip("Stop Claude process")
        self._stop_button.clicked.connect(self.stop_claude)
        StyleManager.apply_button_style(self._stop_button)
        self._stop_button.setEnabled(False)

        # Clear button to clear terminal output
        self._clear_button = QPushButton()
        self._clear_button.setIcon(IconManager.create_icon('close', color='white', size=18))
        self._clear_button.setIconSize(QSize(18, 18))
        self._clear_button.setToolTip("Clear terminal output")
        self._clear_button.clicked.connect(self._clear_terminal)
        from design.style_manager import Colors
        StyleManager.apply_colored_icon_button_style(self._clear_button, Colors.DARK_GRAY)

        # Copy button to copy terminal output
        self._copy_button = QPushButton()
        self._copy_button.setIcon(IconManager.create_icon('content_copy', color='white', size=18))
        self._copy_button.setIconSize(QSize(18, 18))
        self._copy_button.setToolTip("Copy results to clipboard")
        self._copy_button.clicked.connect(self._copy_results)
        StyleManager.apply_colored_icon_button_style(self._copy_button, Colors.DARK_GRAY)

        # Load from Claude button - extracts JSON config from terminal output
        self._load_claude_button = QPushButton()
        self._load_claude_button.setIcon(IconManager.create_icon('download', color='white', size=18))
        self._load_claude_button.setIconSize(QSize(18, 18))
        self._load_claude_button.setToolTip("Load clips config from Claude output")
        self._load_claude_button.clicked.connect(self._load_config_from_output)
        StyleManager.apply_colored_icon_button_style(self._load_claude_button, Colors.BRIGHT_GREEN)

        self._settings_button = QPushButton()
        self._settings_button.setIcon(IconManager.create_icon('settings', color='white', size=18))
        self._settings_button.setIconSize(QSize(18, 18))
        self._settings_button.setToolTip("Scene detection sliders")
        self._settings_button.clicked.connect(self._open_scene_settings)
        StyleManager.apply_colored_icon_button_style(self._settings_button, Colors.DARK_GRAY)

        # Analysis step buttons
        self._srt_button = QPushButton("SRT")
        self._srt_button.setToolTip("Generate or load SRT transcription")
        self._srt_button.clicked.connect(self._run_srt_analysis)
        StyleManager.apply_button_style(self._srt_button)
        self._srt_button.setEnabled(False)

        self._audio_button = QPushButton("Audio")
        self._audio_button.setToolTip("Run audio analysis (VAD, RMS, pauses)")
        self._audio_button.clicked.connect(self._run_audio_analysis)
        StyleManager.apply_button_style(self._audio_button)
        self._audio_button.setEnabled(False)

        self._face_button = QPushButton("Face")
        self._face_button.setToolTip("Run face detection and expressivity analysis")
        self._face_button.clicked.connect(self._run_face_analysis)
        StyleManager.apply_button_style(self._face_button)
        self._face_button.setEnabled(False)

        # Content type selector
        self._content_type_combo = QComboBox()
        self._content_type_combo.addItem("Erklärvideos", "qa")
        self._content_type_combo.addItem("Social Media", "social")
        self._content_type_combo.setToolTip("Select content type for scene detection")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        header_row = QHBoxLayout()
        header = QLabel("Claude")
        header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setStyleSheet("padding: 6px 0px; font-weight: bold;")
        header_row.addWidget(header)
        header_row.addStretch()
        header_row.addWidget(self._load_claude_button)
        header_row.addWidget(self._copy_button)
        header_row.addWidget(self._clear_button)
        header_row.addWidget(self._settings_button)
        header_row.addWidget(self._start_button)
        header_row.addWidget(self._stop_button)
        layout.addLayout(header_row)

        # Analysis control row
        analysis_row = QHBoxLayout()
        analysis_row.setSpacing(4)
        analysis_row.addWidget(QLabel("Analyse:"))
        analysis_row.addWidget(self._srt_button)
        analysis_row.addWidget(self._audio_button)
        analysis_row.addWidget(self._face_button)
        analysis_row.addStretch()
        analysis_row.addWidget(self._content_type_combo)
        analysis_row.addWidget(self._prompt_button)
        layout.addLayout(analysis_row)

        self._terminal = ClaudeTerminalWidget(self._work_dir)
        layout.addWidget(self._terminal, stretch=1)

        # Prompt input row
        prompt_row = QHBoxLayout()
        prompt_row.setSpacing(4)
        self._prompt_input = QLineEdit()
        self._prompt_input.setPlaceholderText("Enter follow-up prompt...")
        self._prompt_input.returnPressed.connect(self._send_custom_prompt)
        prompt_row.addWidget(self._prompt_input, stretch=1)

        self._send_button = QPushButton()
        self._send_button.setIcon(IconManager.create_icon('send', color='white', size=18))
        self._send_button.setIconSize(QSize(18, 18))
        self._send_button.setToolTip("Send prompt")
        self._send_button.clicked.connect(self._send_custom_prompt)
        StyleManager.apply_colored_icon_button_style(self._send_button, Colors.BLUE)
        prompt_row.addWidget(self._send_button)

        layout.addLayout(prompt_row)
        layout.addWidget(self._status_label)
        layout.addWidget(self._analysis_status_label)

    def _load_prompt(self, filename: str) -> str:
        path = Path(__file__).resolve().parent / filename
        if not path.exists():
            self._status_label.setText(f"Prompt missing: {path.name}")
            return ""
        return path.read_text(encoding="utf-8").strip()

    def start_claude(self) -> None:
        self._write_context_files()
        if self._context_dir:
            self._terminal.set_work_dir(self._context_dir)
        self._terminal.start()
        self._sync_controls()

    def stop_claude(self) -> None:
        self._terminal.stop()
        self._sync_controls()

    def send_scene_selection_prompt(self) -> None:
        """Run scene detection in headless mode and display results."""
        if not self._srt_path or not self._srt_path.exists():
            QMessageBox.information(self, "Scene Detection", "Bitte warten Sie auf die Erstellung der Untertitel.")
            return

        # Update status
        self._status_label.setText("Running scene detection...")
        self._prompt_button.setEnabled(False)

        # Run Claude in headless mode
        QTimer.singleShot(100, self._run_scene_detection_headless)

    def set_context(self, video_path: Path | None, srt_path: Path | None) -> None:
        self._video_path = video_path
        self._srt_path = srt_path
        ready = self._srt_path is not None and self._srt_path.exists()
        has_video = self._video_path is not None and self._video_path.exists()

        # Enable analysis buttons when video is loaded
        self._srt_button.setEnabled(has_video)
        self._audio_button.setEnabled(has_video)
        self._face_button.setEnabled(has_video)

        # Enable scene selection button when SRT is ready
        self._prompt_button.setEnabled(ready)

        if ready:
            self._status_label.setText("Subtitles ready.")
        else:
            self._status_label.setText("Waiting for subtitles.")
        self._update_analysis_status()
        self._write_context_files()

    def _run_srt_analysis(self) -> None:
        """Generate or load SRT transcription."""
        if not self._video_path:
            QMessageBox.information(self, "SRT Analysis", "Kein Video geladen.")
            return

        # TODO: Implement SRT generation/loading
        # For now, just show message
        QMessageBox.information(
            self,
            "SRT Analysis",
            "SRT-Generierung wird implementiert.\n\n"
            "Aktuell: SRT wird automatisch bei Video-Auswahl erstellt."
        )

    def _run_audio_analysis(self) -> None:
        """Run audio analysis (VAD, RMS, pauses)."""
        if not self._video_path:
            QMessageBox.information(self, "Audio Analysis", "Kein Video geladen.")
            return

        if not self._context_dir:
            QMessageBox.warning(self, "Audio Analysis", "Kontext-Verzeichnis nicht verfügbar.")
            return

        self._status_label.setText("Running audio analysis...")
        # Run just audio analysis step
        from scene_detection_pipeline import SceneDetectionPipeline, SceneDetectionConfig

        config = self._build_scene_config(face_required=False, face_enabled=False)
        pipeline = SceneDetectionPipeline(self._context_dir, config)

        try:
            audio_data = pipeline._analyze_audio(self._video_path)
            logger.info(f"[AUDIO] Analysis complete: {len(audio_data.get('speech_segments', []))} speech segments")
            self._status_label.setText("✓ Audio analysis complete")
            self._update_analysis_status()
        except Exception as e:
            logger.error(f"[AUDIO] Analysis failed: {e}")
            self._status_label.setText(f"Audio analysis failed: {e}")

    def _run_face_analysis(self) -> None:
        """Run face detection and expressivity analysis."""
        if not self._video_path:
            QMessageBox.information(self, "Face Analysis", "Kein Video geladen.")
            return

        if not self._context_dir:
            QMessageBox.warning(self, "Face Analysis", "Kontext-Verzeichnis nicht verfügbar.")
            return

        self._status_label.setText("Running face analysis...")
        # Run just face analysis step
        from scene_detection_pipeline import SceneDetectionPipeline, SceneDetectionConfig, FaceAnalysisError

        config = self._build_scene_config(face_required=True, face_enabled=True)
        pipeline = SceneDetectionPipeline(self._context_dir, config)

        try:
            face_data = pipeline._analyze_face(self._video_path)
            if face_data:
                logger.info(f"[FACE] Analysis complete: {len(face_data.get('series', []))} frames")
                self._status_label.setText("✓ Face analysis complete")
            else:
                self._status_label.setText("Face analysis skipped (no face detected)")
            self._update_analysis_status()
        except FaceAnalysisError as e:
            logger.error(f"[FACE] Analysis failed: {e}")
            QMessageBox.warning(self, "Face Analysis", f"Face analysis failed:\n\n{e}")
            self._status_label.setText("Face analysis failed")
        except Exception as e:
            logger.error(f"[FACE] Unexpected error: {e}")
            self._status_label.setText(f"Face analysis error: {e}")

    def _run_scene_detection_headless(self) -> None:
        """Run Claude Code in headless mode for scene detection."""
        prompt = self._load_prompt("scene-detection-prompt.txt")
        if not prompt:
            self._status_label.setText("Error: Prompt file not found")
            self._prompt_button.setEnabled(True)
            return

        if not self._video_path or not self._video_path.exists():
            self._status_label.setText("Error: Video file not found")
            self._prompt_button.setEnabled(True)
            return
        if not self._srt_path or not self._srt_path.exists():
            self._status_label.setText("Error: SRT file not found")
            self._prompt_button.setEnabled(True)
            return

        # Ensure context files are written
        self._write_context_files()

        if not self._context_dir:
            self._status_label.setText("Error: Context directory not available")
            self._prompt_button.setEnabled(True)
            return

        self._run_scene_detection_job(face_required=True, face_enabled=True)

    def _run_scene_detection_job(self, face_required: bool, face_enabled: bool) -> None:
        logger.info(
            "[SCENE] Starting scene detection job (face_required=%s, face_enabled=%s)",
            face_required,
            face_enabled,
        )
        worker = SceneWorker(self._scene_detection_task, face_required, face_enabled)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_scene_detection_finished)
        worker.error.connect(self._on_scene_detection_error)
        worker.finished.connect(lambda _result, w=worker: self._remove_scene_worker(w))
        worker.error.connect(lambda _error, w=worker: self._remove_scene_worker(w))
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._scene_workers.append(worker)
        self._threads.append(thread)
        thread.start()

    def _scene_detection_task(self, face_required: bool, face_enabled: bool) -> dict:
        scoring_prompt = self._load_prompt("scene-detection-prompt.txt")
        discovery_prompt = self._load_prompt("scene-discovery-prompt.txt")
        if not scoring_prompt:
            raise RuntimeError("Scoring prompt file not found")
        if not discovery_prompt:
            raise RuntimeError("Discovery prompt file not found")
        if not self._video_path or not self._video_path.exists():
            raise RuntimeError("Video file not found")
        if not self._srt_path or not self._srt_path.exists():
            raise RuntimeError("SRT file not found")
        if not self._context_dir:
            raise RuntimeError("Context directory not available")

        logger.info("[SCENE] Running Claude discovery")
        discovery_prompt = discovery_prompt.replace("__SRT_PATH__", str(self._srt_path))
        discovery_cmd = [
            "claude",
            "-p", discovery_prompt,
            "--allowedTools", "Read,Grep",
        ]
        discovery_result = subprocess.run(
            discovery_cmd,
            cwd=str(self._context_dir) if self._context_dir else str(self._work_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        discovery_output = discovery_result.stdout if discovery_result.returncode == 0 else discovery_result.stderr
        discovery_payloads = self._parse_json_payloads(discovery_output)
        discovery_candidates = next(
            (payload for payload in discovery_payloads if isinstance(payload, list)),
            None,
        )
        if not isinstance(discovery_candidates, list) or not discovery_candidates:
            raise RuntimeError("No candidates found in discovery output")

        pipeline, bundle = self._prepare_scene_bundle_from_list(
            face_required=face_required,
            face_enabled=face_enabled,
            candidates=discovery_candidates,
        )
        if not bundle.get("enriched_candidates"):
            raise RuntimeError("No clip candidates found")

        candidates_path = self._analysis_artifact_path("analysis_clip_candidates_enriched")
        if candidates_path is None:
            raise RuntimeError("Scene analysis artifacts missing for current video")
        scoring_prompt = (
            scoring_prompt.replace("__CANDIDATES_PATH__", str(candidates_path.resolve()))
            .replace("__MIN_CLIP_SECONDS__", str(int(self._scene_settings["min_clip_seconds"])))
            .replace("__MAX_CLIP_SECONDS__", str(int(self._scene_settings["max_clip_seconds"])))
        )

        cmd = [
            "claude",
            "-p", scoring_prompt,
            "--allowedTools", "Read,Grep",
        ]

        logger.info("[SCENE] Running Claude scoring")
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._context_dir) if self._context_dir else str(self._work_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Scene detection timed out after 5 minutes") from exc

        output = result.stdout if result.returncode == 0 else result.stderr
        payloads = self._parse_json_payloads(output)
        scores = next((payload for payload in payloads if isinstance(payload, list)), None)
        if isinstance(scores, list) and scores:
            ranking = pipeline.score_and_rank(bundle, scores)
            refined = pipeline.refine_cuts(
                bundle,
                ranking["top_candidates"],
                self._video_path,
            )
            config = pipeline.build_clip_config(refined)
            output += "\n=== Final Clips Config ===\n```json\n"
            output += json.dumps(config, indent=2, ensure_ascii=True)
            output += "\n```\n"
        else:
            output += "\n⚠ No valid scoring JSON found. Skipping ranking.\n"

        return {"output": output}

    def _on_scene_detection_finished(self, result: dict) -> None:
        output = result.get("output", "")
        self._terminal._output.appendPlainText("\n=== Scene Detection Results ===\n")
        self._terminal._output.appendPlainText(output)
        self._terminal._output.appendPlainText("\n=== End Results ===\n")
        self._extract_scene_data(output)
        self._status_label.setText("✓ Scene detection complete")
        self._update_analysis_status()
        self._prompt_button.setEnabled(True)
        logger.info("[SCENE] Scene detection job completed")

    def _on_scene_detection_error(self, error: Exception) -> None:
        logger.error("[SCENE] Scene detection job failed: %s", error)
        if isinstance(error, FaceAnalysisError):
            message = (
                "Gesichtsanalyse fehlgeschlagen.\n\n"
                f"{error}\n\n"
                "Ohne Gesichtsanalyse weitermachen?"
            )
            choice = QMessageBox.question(
                self,
                "Gesichtsanalyse fehlgeschlagen",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice == QMessageBox.StandardButton.Yes:
                self._terminal._output.appendPlainText(
                    "\n⚠ Face analysis skipped. Continuing without face features.\n"
                )
                self._run_scene_detection_job(face_required=False, face_enabled=False)
                return
        self._status_label.setText("Error: Scene detection failed")
        self._terminal._output.appendPlainText(f"\n⚠ Scene error: {error}\n")
        self._prompt_button.setEnabled(True)

    def _remove_scene_worker(self, worker: SceneWorker) -> None:
        try:
            self._scene_workers.remove(worker)
        except ValueError:
            pass

    def _prepare_scene_bundle(
        self,
        face_required: bool,
        face_enabled: bool,
    ) -> tuple[SceneDetectionPipeline, dict]:
        config = self._build_scene_config(face_required, face_enabled)
        pipeline = SceneDetectionPipeline(self._context_dir, config)
        bundle = pipeline.prepare_candidates(self._video_path, self._srt_path)
        return pipeline, bundle

    def _prepare_scene_bundle_from_list(
        self,
        face_required: bool,
        face_enabled: bool,
        candidates: list[dict],
    ) -> tuple[SceneDetectionPipeline, dict]:
        config = self._build_scene_config(face_required, face_enabled)
        pipeline = SceneDetectionPipeline(self._context_dir, config)
        bundle = pipeline.prepare_candidates_from_list(
            self._video_path,
            self._srt_path,
            candidates,
        )
        return pipeline, bundle

    def _build_scene_config(self, face_required: bool, face_enabled: bool) -> SceneDetectionConfig:
        max_clips = int(self._scene_settings.get("top_k_max", 7))
        top_k_min = 0 if max_clips <= 0 else min(3, max_clips)
        return SceneDetectionConfig(
            min_clip_seconds=float(self._scene_settings["min_clip_seconds"]),
            max_clip_seconds=float(self._scene_settings["max_clip_seconds"]),
            face_required=face_required,
            face_enabled=face_enabled,
            speech_ratio_min=float(self._scene_settings["speech_ratio_min"]),
            whisper_bad_ratio_max=float(self._scene_settings["whisper_bad_ratio_max"]),
            incomplete_thought_penalty=float(self._scene_settings["incomplete_thought_penalty"]),
            weight_text=float(self._scene_settings["weight_text"]),
            weight_cut=float(self._scene_settings["weight_cut"]),
            weight_audio=float(self._scene_settings["weight_audio"]),
            weight_face=float(self._scene_settings["weight_face"]),
            top_k_min=top_k_min,
            top_k_max=max_clips,
            whisperx_env_name="ai-videoclipper-whisperx",
            whisperx_require_env=True,
        )

    def _open_scene_settings(self) -> None:
        if self._scene_settings_dialog is None:
            self._scene_settings_dialog = QDialog(self)
            self._scene_settings_dialog.setWindowTitle("Scene Detection Settings")
            self._scene_settings_dialog.setModal(True)
            self._scene_settings_dialog.setMinimumWidth(420)
            layout = QVBoxLayout(self._scene_settings_dialog)

            preset_row = QHBoxLayout()
            preset_label = QLabel("Preset")
            preset_combo = QComboBox()
            preset_combo.setEditable(False)
            preset_row.addWidget(preset_label)
            preset_row.addWidget(preset_combo, stretch=1)

            save_button = QPushButton("Save")
            save_as_button = QPushButton("Save As")
            load_button = QPushButton("Load")
            preset_row.addWidget(save_button)
            preset_row.addWidget(save_as_button)
            preset_row.addWidget(load_button)
            layout.addLayout(preset_row)

            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

            controls = {}

            def add_slider(key, label, min_val, max_val, step, scale=1.0, suffix=""):
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMinimum(min_val)
                slider.setMaximum(max_val)
                slider.setSingleStep(step)
                slider.setPageStep(step)
                value_label = QLabel()

                def update_label(value):
                    scaled = value / scale
                    if scale == 1.0:
                        text = f"{scaled:.0f}{suffix}"
                    else:
                        text = f"{scaled:.2f}{suffix}"
                    value_label.setText(text)

                slider.valueChanged.connect(update_label)
                row = QHBoxLayout()
                row.addWidget(slider, stretch=1)
                row.addWidget(value_label)
                form.addRow(label, row)
                controls[key] = (slider, scale, value_label, update_label)
                return slider

            add_slider("weight_text", "Content Gewicht", 0, 100, 1, scale=100.0)
            add_slider("weight_cut", "Cut Gewicht", 0, 100, 1, scale=100.0)
            add_slider("weight_audio", "Audio Gewicht", 0, 100, 1, scale=100.0)
            add_slider("weight_face", "Face Gewicht", 0, 100, 1, scale=100.0)
            add_slider("incomplete_thought_penalty", "Incomplete Penalty", 0, 30, 1, scale=100.0)
            add_slider("min_clip_seconds", "Min Clip (s)", 10, 180, 5, scale=1.0)
            add_slider("max_clip_seconds", "Max Clip (s)", 60, 600, 10, scale=1.0)
            add_slider("speech_ratio_min", "Speech Ratio Min", 0, 100, 1, scale=100.0)
            add_slider("whisper_bad_ratio_max", "Whisper Bad Max", 0, 100, 1, scale=100.0)

            max_clips_slider = add_slider("top_k_max", "Max Clips", 0, 20, 1, scale=1.0)

            layout.addLayout(form)

            preview_label = QLabel("Clip-Vorschlaege: Analyse fehlt")
            preview_label.setStyleSheet("color: #555; padding-top: 6px;")
            layout.addWidget(preview_label)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(self._scene_settings_dialog.close)
            layout.addWidget(buttons)

            self._scene_settings_controls = {
                "combo": preset_combo,
                "save": save_button,
                "save_as": save_as_button,
                "load": load_button,
                "preview": preview_label,
                "sliders": controls,
                "max_clips": max_clips_slider,
            }

            for slider, *_rest in controls.values():
                slider.valueChanged.connect(self._on_scene_setting_changed)

            save_button.clicked.connect(self._save_scene_settings)
            save_as_button.clicked.connect(self._save_scene_settings_as)
            load_button.clicked.connect(self._load_scene_settings)
            preset_combo.currentTextChanged.connect(self._on_preset_selected)

        self._sync_scene_settings_controls()
        self._scene_settings_dialog.show()
        self._refresh_scene_settings_preview()

    def _scene_settings_store_path(self) -> Path:
        return Path.home() / ".local" / "share" / "ai-videoclipper" / "scene_settings.json"

    def _load_scene_settings_store(self) -> dict:
        path = self._scene_settings_store_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_scene_settings_store(self, payload: dict) -> None:
        path = self._scene_settings_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _ensure_default_settings_saved(self) -> None:
        store = self._load_scene_settings_store()
        store["default"] = self._scene_settings.copy()
        self._save_scene_settings_store(store)

    def _sync_scene_settings_controls(self) -> None:
        controls = self._scene_settings_controls
        if not controls:
            return
        store = self._load_scene_settings_store()
        combo: QComboBox = controls["combo"]
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(sorted(store.keys()))
        if self._scene_settings_name in store:
            combo.setCurrentText(self._scene_settings_name)
        combo.blockSignals(False)

        sliders = controls["sliders"]
        for key, (slider, scale, value_label, update_label) in sliders.items():
            value = self._scene_settings.get(key, 0.0)
            slider.blockSignals(True)
            slider.setValue(int(round(value * scale)))
            slider.blockSignals(False)
            update_label(slider.value())

        self._sync_max_clips_slider()
        self._refresh_scene_settings_preview()

    def _sync_max_clips_slider(self) -> None:
        controls = self._scene_settings_controls
        if not controls:
            return
        max_clips_slider: QSlider = controls["max_clips"]
        max_allowed = self._compute_max_clips_limit()
        max_clips_slider.blockSignals(True)
        max_clips_slider.setMaximum(max_allowed)
        current = min(int(self._scene_settings.get("top_k_max", 0)), max_allowed)
        max_clips_slider.setValue(current)
        max_clips_slider.blockSignals(False)
        self._update_scene_settings_from_controls()

    def _compute_max_clips_limit(self) -> int:
        duration_minutes = self._get_video_duration_minutes()
        if duration_minutes <= 0:
            return 20
        return max(0, int(duration_minutes / 3))

    def _get_video_duration_minutes(self) -> float:
        if not self._srt_path or not self._srt_path.exists():
            return 0.0
        segments = parse_srt(str(self._srt_path))
        if not segments:
            return 0.0
        duration = max(seg.end for seg in segments)
        return duration / 60.0

    def _on_scene_setting_changed(self) -> None:
        self._update_scene_settings_from_controls()
        self._schedule_scene_settings_preview()

    def _update_scene_settings_from_controls(self) -> None:
        controls = self._scene_settings_controls
        if not controls:
            return
        sliders = controls["sliders"]
        for key, (slider, scale, _label, _update) in sliders.items():
            self._scene_settings[key] = slider.value() / scale
        self._scene_settings["top_k_max"] = controls["max_clips"].value()

    def _schedule_scene_settings_preview(self) -> None:
        if self._scene_settings_preview_timer is None:
            self._scene_settings_preview_timer = QTimer(self)
            self._scene_settings_preview_timer.setSingleShot(True)
            self._scene_settings_preview_timer.timeout.connect(self._refresh_scene_settings_preview)
        self._scene_settings_preview_timer.start(250)

    def _refresh_scene_settings_preview(self) -> None:
        controls = self._scene_settings_controls
        if not controls:
            return
        label: QLabel = controls["preview"]
        status = self._estimate_clip_count()
        if status is None:
            label.setText("Clip-Vorschlaege: Analyse fehlt")
        else:
            label.setText(f"Clip-Vorschlaege: {status}")

    def _estimate_clip_count(self) -> Optional[int]:
        if not self._context_dir:
            return None
        if not self._video_path:
            return None
        enriched_path = self._analysis_artifact_path("analysis_clip_candidates_enriched")
        scores_path = self._analysis_artifact_path("analysis_llm_scores")
        meta_path = self._analysis_artifact_path("analysis_meta")
        if not enriched_path or not scores_path or not meta_path:
            return None
        if not enriched_path.exists() or not scores_path.exists() or not meta_path.exists():
            return None
        try:
            enriched = json.loads(enriched_path.read_text(encoding="utf-8"))
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        config = self._build_scene_config(face_required=False, face_enabled=False)
        pipeline = SceneDetectionPipeline(self._context_dir, config)
        scored = pipeline._apply_llm_scores(enriched, scores)
        ranked = pipeline._rank_candidates(scored)
        final = pipeline._apply_diversity(ranked)
        if len(final) < config.top_k_min:
            final = ranked[: config.top_k_min]
        final = final[: config.top_k_max]
        return len(final)

    def _analysis_artifact_path(self, suffix: str) -> Optional[Path]:
        if not self._video_path:
            return None
        return self._video_path.parent / f"{self._video_path.stem}_{suffix}.json"

    def _update_analysis_status(self) -> None:
        if not self._video_path:
            self._analysis_status_label.setText("")
            return
        artifacts = [
            ("Audio", "analysis_audio_pack"),
            ("VAD", "analysis_audio_vad"),
            ("Pauses", "analysis_audio_pauses"),
            ("RMS", "analysis_audio_rms"),
            ("Face", "analysis_face"),
            ("Candidates", "analysis_clip_candidates"),
            ("Enriched", "analysis_clip_candidates_enriched"),
            ("Scores", "analysis_llm_scores"),
            ("Cutlist", "analysis_cutlist"),
        ]
        available = []
        for label, suffix in artifacts:
            path = self._analysis_artifact_path(suffix)
            if path and path.exists():
                available.append(label)
        if not available:
            self._analysis_status_label.setText("Analyse: keine Daten geladen")
        else:
            self._analysis_status_label.setText("Analyse geladen: " + ", ".join(available))

    def _save_scene_settings(self) -> None:
        store = self._load_scene_settings_store()
        store[self._scene_settings_name] = self._scene_settings.copy()
        self._save_scene_settings_store(store)
        self._sync_scene_settings_controls()

    def _save_scene_settings_as(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Save Settings As",
            "Name for new preset:",
        )
        name = (name or "").strip()
        if not ok or not name:
            return
        store = self._load_scene_settings_store()
        store[name] = self._scene_settings.copy()
        self._scene_settings_name = name
        self._save_scene_settings_store(store)
        self._sync_scene_settings_controls()

    def _load_scene_settings(self) -> None:
        controls = self._scene_settings_controls
        if not controls:
            return
        name = controls["combo"].currentText().strip()
        store = self._load_scene_settings_store()
        if name not in store:
            return
        self._scene_settings = store[name].copy()
        self._scene_settings_name = name
        self._sync_scene_settings_controls()
        self._refresh_scene_settings_preview()

    def _on_preset_selected(self, name: str) -> None:
        if not name:
            return
        store = self._load_scene_settings_store()
        if name not in store:
            return
        self._scene_settings = store[name].copy()
        self._scene_settings_name = name
        self._sync_scene_settings_controls()

    def _extract_scene_data(self, output: str) -> None:
        """Extract JSON scene data from Claude output and emit signal."""
        payloads = self._parse_json_payloads(output)
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            # Check if it's a clips config (new format)
            if 'clips' in payload and 'mode' in payload:
                num_clips = len(payload.get('clips', []))
                print(f"[CLAUDE] ✓ Extracted clips config with {num_clips} clips")
                # Emit as clips config, not scene data
                self.load_clips_config.emit(payload)
                return
            # Legacy format: cut_points (old scene detection)
            if 'cut_points' in payload:
                print(f"[CLAUDE] ✓ Extracted {len(payload.get('cut_points', []))} scene cut points")
                self.scene_data_received.emit(payload)
                return
        if payloads:
            print(f"[CLAUDE] ⚠ Unknown JSON format")

    def _parse_json_payloads(self, output: str) -> list[object]:
        """Extract JSON payloads from Claude output (code blocks or raw JSON)."""
        payloads: list[object] = []
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, output, re.DOTALL)
        for match in matches:
            try:
                payloads.append(json.loads(match))
            except json.JSONDecodeError:
                continue
        if payloads:
            return payloads
        stripped = output.strip()
        if not stripped:
            return payloads
        try:
            payloads.append(json.loads(stripped))
        except json.JSONDecodeError:
            pass
        return payloads

    def _queue_prompt_send(self) -> None:
        """Legacy method - kept for backwards compatibility."""
        prompt = self._load_prompt("scene-detection-prompt.txt")
        if not prompt:
            return
        if self._terminal.is_running() and self._pending_prompt:
            self._pending_prompt = False
            QTimer.singleShot(1200, lambda: self._terminal.send_text(prompt))
        elif self._terminal.is_running():
            self._terminal.send_text(prompt)
        self._status_label.setText("Prompt sent to terminal.")
        self._sync_controls()

    def _write_context_files(self) -> None:
        context_dir = self._resolve_context_dir()
        if context_dir is None:
            return
        context_dir.mkdir(parents=True, exist_ok=True)
        claude_dir = context_dir / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)

        claude_md = self._build_claude_md()
        (context_dir / "CLAUDE.md").write_text(claude_md, encoding="utf-8")

        session = {
            "video_path": str(self._video_path) if self._video_path else None,
            "srt_path": str(self._srt_path) if self._srt_path else None,
        }
        (context_dir / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")

        settings = {
            "permissions": {
                "allow": [
                    f"Read:{context_dir}/**",
                ],
                "deny": [],
            }
        }
        if self._video_path:
            settings["permissions"]["allow"].append(f"Read:{self._video_path}")
        if self._srt_path:
            settings["permissions"]["allow"].append(f"Read:{self._srt_path}")
        (claude_dir / "settings.local.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8"
        )

    def _resolve_context_dir(self) -> Path | None:
        if self._video_path:
            context_dir = self._video_path.parent / ".claude-context"
        else:
            context_dir = self._work_dir / ".claude-context"
        self._context_dir = context_dir
        return context_dir

    def _build_claude_md(self) -> str:
        from datetime import datetime

        lines = [
            "# AI VideoClipper - Video Editing Session",
            "",
            "## Current Working Files",
        ]
        if self._video_path:
            lines.append(f"- **Video:** `{self._video_path}`")
        if self._srt_path:
            lines.append(f"- **Subtitle (SRT):** `{self._srt_path}`")

        lines.extend([
            "",
            "## Session Information",
            f"- **Session Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Working Directory:** `{self._work_dir}`",
            "",
            "## Available Operations",
            "### Scene Detection & Analysis",
            "- Candidates are precomputed locally from audio/VAD + transcript",
            "- Candidate file: `scene_detection/clip_candidates_enriched.json`",
            "- Claude scores content/packaging only (no boundary edits)",
            "- Output strict JSON as instructed in the prompt",
            "",
            "### File Reference Syntax",
            "When referencing files in your analysis, use:",
        ])

        if self._video_path:
            lines.append(f"- Video file: `@{self._video_path}`")
        if self._srt_path:
            lines.append(f"- SRT file: `@{self._srt_path}`")

        lines.extend([
            "",
            "## Output Format Guidelines",
            "- Use absolute paths for all file references",
            "- For scene detection scoring: return strict JSON array as prompted",
            "- Do not invent or adjust timestamps/boundaries",
            "- Include SRT segment indices only if explicitly requested",
            "",
            "## Clips Configuration Format (for direct import)",
            "Only use this format when explicitly asked for clip configs.",
            "",
            "### Format Option 1: By Time (seconds)",
            "```json",
            "{",
            '  "mode": "manual",',
            '  "selection_type": "time",',
            '  "clips": [',
            "    {",
            '      "name": "Scene title or description",',
            '      "start_time": 23.5,',
            '      "end_time": 125.8',
            "    }",
            "  ]",
            "}",
            "```",
            "",
            "### Format Option 2: By Segment Numbers (1-indexed)",
            "```json",
            "{",
            '  "mode": "manual",',
            '  "selection_type": "segments",',
            '  "clips": [',
            "    {",
            '      "name": "Scene title or description",',
            '      "start_segment": 8,',
            '      "end_segment": 36',
            "    }",
            "  ]",
            "}",
            "```",
            "",
            "**Important:**",
            "- Times are in seconds (float), not HH:MM:SS format",
            "- Segment numbers are 1-indexed (first segment = 1)",
            "- Use segments when working from SRT analysis",
            "- Use time when specifying precise timestamps",
            "",
            "## Important Notes",
            "- All file paths are absolute and pre-configured",
            "- SRT timing must stay synchronized with video",
            "- Scene detection prompt available at: `scene-detection-prompt.txt`",
        ])

        return "\n".join(lines)

    def _sync_controls(self) -> None:
        running = self._terminal.is_running()
        self._start_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._status_label.setText(self._terminal.status_text())

    def _copy_results(self) -> None:
        """Copy terminal output to clipboard."""
        from PyQt6.QtWidgets import QApplication
        text = self._terminal._output.toPlainText()
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self._status_label.setText("✓ Results copied to clipboard")
        # Reset status after 2 seconds
        QTimer.singleShot(2000, lambda: self._status_label.setText(self._terminal.status_text()))

    def _load_config_from_output(self) -> None:
        """Extract clips config JSON from terminal output and load it."""
        import logging
        logger = logging.getLogger("ai_videoclipper")

        output = self._terminal._output.toPlainText()
        logger.info(f"[CLAUDE] Attempting to extract config from {len(output)} chars of output")

        payloads = self._parse_json_payloads(output)
        logger.info(f"[CLAUDE] Found {len(payloads)} JSON payloads in output")

        if not payloads:
            self._status_label.setText("⚠ No JSON found in output")
            logger.warning("[CLAUDE] No JSON payloads found in terminal output")
            QTimer.singleShot(2000, lambda: self._status_label.setText(self._terminal.status_text()))
            QMessageBox.information(self, "No JSON Found", "No JSON configuration found in Claude's output.\n\nMake sure Claude returned a JSON code block with clips configuration.")
            return

        # Use the LAST JSON payload found (most recent)
        config = payloads[-1]
        if not isinstance(config, dict):
            self._status_label.setText("⚠ Invalid clips config format")
            logger.error("[CLAUDE] JSON payload is not an object")
            QTimer.singleShot(2000, lambda: self._status_label.setText(self._terminal.status_text()))
            QMessageBox.warning(self, "Invalid Format", "JSON payload must be an object with clips configuration.")
            return
        try:
            logger.info(f"[CLAUDE] Successfully parsed JSON: {config.keys()}")

            # Validate that it's a clips config (has required fields)
            if "mode" not in config:
                self._status_label.setText("⚠ Invalid clips config format")
                logger.error(f"[CLAUDE] Config missing 'mode' field. Keys: {config.keys()}")
                QTimer.singleShot(2000, lambda: self._status_label.setText(self._terminal.status_text()))
                QMessageBox.warning(self, "Invalid Format", "JSON is missing required 'mode' field.\n\nExpected format:\n{\n  \"mode\": \"manual\",\n  \"selection_type\": \"time\",\n  \"clips\": [...]\n}")
                return

            # Emit signal to parent to load this config
            num_clips = len(config.get('clips', []))
            logger.info(f"[CLAUDE] ✓ Extracted clips config with {num_clips} clips")
            logger.info(f"[CLAUDE] Emitting load_clips_config signal")
            self.load_clips_config.emit(config)
            self._status_label.setText(f"✓ Loaded {num_clips} clips from Claude")
            QTimer.singleShot(3000, lambda: self._status_label.setText(self._terminal.status_text()))

        except Exception as e:
            self._status_label.setText(f"⚠ JSON parse error")
            logger.error(f"[CLAUDE] JSON parse error: {e}")
            QTimer.singleShot(2000, lambda: self._status_label.setText(self._terminal.status_text()))
            QMessageBox.critical(self, "JSON Parse Error", f"Failed to parse JSON:\n{str(e)}\n\nCheck the Claude output for syntax errors.")

    def _clear_terminal(self) -> None:
        """Clear the terminal output."""
        self._terminal._output.clear()
        self._status_label.setText("✓ Terminal cleared")
        QTimer.singleShot(2000, lambda: self._status_label.setText(self._terminal.status_text()))

    def _send_custom_prompt(self) -> None:
        """Send custom prompt in headless mode."""
        prompt_text = self._prompt_input.text().strip()
        if not prompt_text:
            return

        # Clear input field
        self._prompt_input.clear()

        # Update status
        self._status_label.setText("Running custom prompt...")

        # Run Claude in headless mode with custom prompt
        QTimer.singleShot(100, lambda: self._run_custom_prompt_headless(prompt_text))

    def _run_custom_prompt_headless(self, prompt_text: str) -> None:
        """Run Claude Code in headless mode with custom prompt."""
        # Ensure context files are written
        self._write_context_files()

        # Build Claude command with headless mode
        cmd = [
            "claude",
            "-p", prompt_text,
            "--allowedTools", "Read,Grep",
        ]

        try:
            # Run in context directory
            result = subprocess.run(
                cmd,
                cwd=str(self._context_dir) if self._context_dir else str(self._work_dir),
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            # Display result in terminal
            output = result.stdout if result.returncode == 0 else result.stderr
            self._terminal._output.appendPlainText(f"\n=== Custom Prompt ===\n{prompt_text}\n")
            self._terminal._output.appendPlainText("\n=== Response ===\n")
            self._terminal._output.appendPlainText(output)
            self._terminal._output.appendPlainText("\n=== End Response ===\n")

            # Try to extract JSON if present
            self._extract_scene_data(output)

            self._status_label.setText("✓ Prompt complete")

        except subprocess.TimeoutExpired:
            self._status_label.setText("Error: Prompt timeout")
            self._terminal._output.appendPlainText("\n⚠ Prompt timed out after 2 minutes\n")
        except Exception as e:
            self._status_label.setText(f"Error: {str(e)}")
            self._terminal._output.appendPlainText(f"\n⚠ Error: {e}\n")


_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[A-Za-z]|"
    r"\x1b\][^\x07]*\x07|"
    r"\x1b\][^\x1b]*\x1b\\|"
    r"\r"
)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)
