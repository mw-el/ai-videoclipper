from __future__ import annotations

from pathlib import Path
import json
import os
import pty
import re
import shutil
import subprocess

from PyQt6.QtCore import Qt, QSize, QSocketNotifier, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from design.icon_manager import IconManager
from design.style_manager import StyleManager


class TerminalOutput(QPlainTextEdit):
    def __init__(self, on_key, parent=None) -> None:
        super().__init__(parent)
        self._on_key = on_key
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

    def keyPressEvent(self, event) -> None:
        if self._on_key(event):
            return
        super().keyPressEvent(event)


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
        # Auto-answer "yes" to Claude Code startup confirmation
        cmd = f"{banner} echo 'yes' | claude; exec bash"
        self._process = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdin=self._slave_fd,
            stdout=self._slave_fd,
            stderr=self._slave_fd,
            cwd=str(self._work_dir),
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

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #666; padding: 4px;")

        self._prompt_button = QPushButton()
        self._prompt_button.setToolTip("Scene Selection Prompt")
        self._prompt_button.clicked.connect(self.send_scene_selection_prompt)
        self._prompt_button.setIcon(self._build_scene_detection_icon())
        self._prompt_button.setIconSize(QSize(38, 18))
        StyleManager.apply_icon_button_style(self._prompt_button)
        self._prompt_button.setMinimumWidth(StyleManager.BUTTON_MIN_SIZE * 2 + 8)
        self._prompt_button.setEnabled(False)

        self._start_button = QPushButton("Start Claude")
        self._start_button.clicked.connect(self.start_claude)
        StyleManager.apply_button_style(self._start_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.clicked.connect(self.stop_claude)
        StyleManager.apply_button_style(self._stop_button)
        self._stop_button.setEnabled(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        header_row = QHBoxLayout()
        header = QLabel("Claude")
        header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setStyleSheet("padding: 6px 0px; font-weight: bold;")
        header_row.addWidget(header)
        header_row.addStretch()
        header_row.addWidget(self._prompt_button)
        header_row.addWidget(self._start_button)
        header_row.addWidget(self._stop_button)
        layout.addLayout(header_row)
        self._terminal = ClaudeTerminalWidget(self._work_dir)
        layout.addWidget(self._terminal, stretch=1)
        layout.addWidget(self._status_label)

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
        if not self._srt_path or not self._srt_path.exists():
            QMessageBox.information(self, "Scene Detection", "Bitte warten Sie auf die Erstellung der Untertitel.")
            return
        if not self._terminal.is_running():
            self._pending_prompt = True
            self.start_claude()
        self._queue_prompt_send()

    def set_context(self, video_path: Path | None, srt_path: Path | None) -> None:
        self._video_path = video_path
        self._srt_path = srt_path
        ready = self._srt_path is not None and self._srt_path.exists()
        self._prompt_button.setEnabled(ready)
        if ready:
            self._status_label.setText("Subtitles ready.")
        else:
            self._status_label.setText("Waiting for subtitles.")
        self._write_context_files()

    def _queue_prompt_send(self) -> None:
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
            "- Analyze SRT file to identify scene boundaries",
            "- Detect dialogue breaks and natural pause points",
            "- Identify topic transitions and content changes",
            "- Suggest optimal clip cut points with timestamps",
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
            "- Timestamps format: HH:MM:SS.mmm",
            "- Provide clear reasoning for each suggestion",
            "- Include SRT segment indices when relevant",
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


_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[A-Za-z]|"
    r"\x1b\][^\x07]*\x07|"
    r"\x1b\][^\x1b]*\x1b\\|"
    r"\r"
)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)
