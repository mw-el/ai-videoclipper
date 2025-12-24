"""Style manager for consistent UI styling across the application."""

from PyQt6.QtWidgets import QPushButton, QWidget


class StyleManager:
    """Manages consistent styling for UI elements."""

    # Global button stylesheet
    BUTTON_STYLE = """
    QPushButton {
        background-color: #f5f5f5;
        color: #333333;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
        min-height: 36px;
    }
    QPushButton:hover {
        background-color: #ececec;
        border: 1px solid #b0b0b0;
    }
    QPushButton:pressed {
        background-color: #dcdcdc;
        border: 1px solid #a0a0a0;
    }
    QPushButton:focus {
        outline: none;
        border: 1px solid #4CAF50;
    }
    """

    # Icon-only button style (square, smaller padding)
    ICON_BUTTON_STYLE = """
    QPushButton {
        background-color: #f5f5f5;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 4px;
        min-height: 36px;
        min-width: 36px;
    }
    QPushButton:hover {
        background-color: #ececec;
        border: 1px solid #b0b0b0;
    }
    QPushButton:pressed {
        background-color: #dcdcdc;
        border: 1px solid #a0a0a0;
    }
    QPushButton:focus {
        outline: none;
        border: 2px solid #4CAF50;
    }
    """

    # Primary action button (green, like "Run")
    PRIMARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #4CAF50;
        color: white;
        border: 1px solid #45a049;
        border-radius: 4px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: bold;
        min-height: 36px;
    }
    QPushButton:hover {
        background-color: #45a049;
        border: 1px solid #3d8b40;
    }
    QPushButton:pressed {
        background-color: #3d8b40;
        border: 1px solid #367534;
    }
    QPushButton:focus {
        outline: none;
        border: 2px solid #2e5c2b;
    }
    """

    @classmethod
    def apply_button_style(cls, button: QPushButton) -> None:
        """Apply default button style."""
        button.setStyleSheet(cls.BUTTON_STYLE)

    @classmethod
    def apply_icon_button_style(cls, button: QPushButton) -> None:
        """Apply icon-only button style (square, equal dimensions)."""
        button.setStyleSheet(cls.ICON_BUTTON_STYLE)
        button.setMinimumWidth(36)
        button.setMinimumHeight(36)
        button.setMaximumWidth(36)
        button.setMaximumHeight(36)

    @classmethod
    def apply_primary_button_style(cls, button: QPushButton) -> None:
        """Apply primary action button style (green)."""
        button.setStyleSheet(cls.PRIMARY_BUTTON_STYLE)
