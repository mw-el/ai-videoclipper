"""Dialog for creating a new clip."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton
)
from PyQt6.QtCore import Qt


class NewClipDialog(QDialog):
    """Dialog to create a new clip by selecting segment range."""

    def __init__(self, max_segment_index: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Clip")
        self.setModal(True)
        self.max_segment_index = max_segment_index
        self.start_segment = 0
        self.end_segment = min(5, max_segment_index)
        self._init_ui()

    def _init_ui(self):
        """Initialize dialog UI."""
        layout = QVBoxLayout()

        # Start segment selection
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start Segment:"))
        self.start_spinbox = QSpinBox()
        self.start_spinbox.setMinimum(0)
        self.start_spinbox.setMaximum(self.max_segment_index)
        self.start_spinbox.setValue(0)
        self.start_spinbox.valueChanged.connect(self._on_start_changed)
        start_layout.addWidget(self.start_spinbox)
        layout.addLayout(start_layout)

        # End segment selection
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End Segment:"))
        self.end_spinbox = QSpinBox()
        self.end_spinbox.setMinimum(0)
        self.end_spinbox.setMaximum(self.max_segment_index)
        self.end_spinbox.setValue(min(5, self.max_segment_index))
        self.end_spinbox.valueChanged.connect(self._on_end_changed)
        end_layout.addWidget(self.end_spinbox)
        layout.addLayout(end_layout)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Create")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _on_start_changed(self, value):
        """Keep start <= end."""
        self.start_segment = value
        if value > self.end_spinbox.value():
            self.end_spinbox.setValue(value)

    def _on_end_changed(self, value):
        """Keep end >= start."""
        self.end_segment = value
        if value < self.start_spinbox.value():
            self.start_spinbox.setValue(value)

    def get_segment_range(self) -> tuple[int, int]:
        """Get selected segment range (start, end inclusive)."""
        return self.start_segment, self.end_segment
