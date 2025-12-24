"""Style manager for consistent UI styling across the application."""

from PyQt6.QtWidgets import QPushButton, QWidget, QApplication


class Colors:
    """Standard color palette for the application (CSS variables)."""

    # Dark Siena - Primary dark color
    DARK_SIENA = "#220c10"
    DARK_SIENA_600 = "#39141b"

    # Blue - Secondary colors
    BLUE = "#2176ae"
    BRIGHT_BLUE = "#0eb1d2"

    # Green - Accent/success colors
    GREEN = "#688e26"
    BRIGHT_GREEN = "#98ce00"

    # Orange - Warning color
    ORANGE = "#ff7b33"

    # Paper - Light background
    PAPER = "#fffdf9"
    DARK_PAPER = "#d7ac7f"

    # Red - Error color
    RED = "#ff3333"

    # Yellow - Info/highlight color
    YELLOW = "#fdcb34"

    # Pink - Secondary accent
    PINK = "#cb20c5"

    # Grayscale
    LIGHT_GRAY = "#f5f5f5"
    MEDIUM_GRAY = "#ececec"
    DARK_GRAY = "#666666"
    BORDER_GRAY = "#d0d0d0"
    DARK_BORDER_GRAY = "#b0b0b0"


class StyleManager:
    """Manages consistent styling for UI elements."""

    BUTTON_MIN_SIZE = 32

    # Global CSS stylesheet with color variables
    GLOBAL_STYLE = f"""
    /* Root style with color variables */
    * {{
        --dark-siena: {Colors.DARK_SIENA};
        --dark-siena-600: {Colors.DARK_SIENA_600};
        --blue: {Colors.BLUE};
        --bright-blue: {Colors.BRIGHT_BLUE};
        --green: {Colors.GREEN};
        --bright-green: {Colors.BRIGHT_GREEN};
        --orange: {Colors.ORANGE};
        --paper: {Colors.PAPER};
        --dark-paper: {Colors.DARK_PAPER};
        --red: {Colors.RED};
        --yellow: {Colors.YELLOW};
        --pink: {Colors.PINK};
        --light-gray: {Colors.LIGHT_GRAY};
        --medium-gray: {Colors.MEDIUM_GRAY};
        --dark-gray: {Colors.DARK_GRAY};
        --border-gray: {Colors.BORDER_GRAY};
        --dark-border-gray: {Colors.DARK_BORDER_GRAY};
    }}
    """

    # Global button stylesheet
    BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {Colors.LIGHT_GRAY};
        color: {Colors.DARK_GRAY};
        border: none;
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 12px;
        min-height: {BUTTON_MIN_SIZE}px;
    }}
    QPushButton:hover {{
        background-color: {Colors.MEDIUM_GRAY};
    }}
    QPushButton:pressed {{
        background-color: {Colors.DARK_BORDER_GRAY};
    }}
    QPushButton:focus {{
        outline: none;
    }}
    QPushButton:disabled {{
        background-color: #595959;
        color: white;
    }}
    """

    # Icon-only button style (square, smaller padding)
    ICON_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {Colors.LIGHT_GRAY};
        border: none;
        border-radius: 4px;
        padding: 3px;
        min-height: {BUTTON_MIN_SIZE}px;
        min-width: {BUTTON_MIN_SIZE}px;
    }}
    QPushButton:hover {{
        background-color: {Colors.MEDIUM_GRAY};
    }}
    QPushButton:pressed {{
        background-color: {Colors.DARK_BORDER_GRAY};
    }}
    QPushButton:focus {{
        outline: none;
    }}
    QPushButton:disabled {{
        background-color: #595959;
    }}
    """

    # Primary action button (green, like "Run")
    PRIMARY_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {Colors.BRIGHT_GREEN};
        color: white;
        border: none;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: bold;
        min-height: {BUTTON_MIN_SIZE}px;
    }}
    QPushButton:hover {{
        background-color: {Colors.GREEN};
    }}
    QPushButton:pressed {{
        background-color: {Colors.DARK_SIENA_600};
    }}
    QPushButton:focus {{
        outline: none;
    }}
    QPushButton:disabled {{
        background-color: #595959;
        color: white;
    }}
    """

    @staticmethod
    def _darken_hex_color(hex_color: str, amount: float = 0.12) -> str:
        """Return a slightly darkened hex color for hover states."""
        if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
            return hex_color
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
        except ValueError:
            return hex_color
        scale = max(0.0, min(1.0, 1.0 - amount))
        r = max(0, min(255, int(r * scale)))
        g = max(0, min(255, int(g * scale)))
        b = max(0, min(255, int(b * scale)))
        return f"#{r:02x}{g:02x}{b:02x}"

    @classmethod
    def apply_button_style(cls, button: QPushButton) -> None:
        """Apply default button style."""
        button.setStyleSheet(cls.BUTTON_STYLE)

    @classmethod
    def apply_icon_button_style(cls, button: QPushButton) -> None:
        """Apply icon-only button style (square, equal dimensions)."""
        button.setStyleSheet(cls.ICON_BUTTON_STYLE)
        button.setMinimumWidth(cls.BUTTON_MIN_SIZE)
        button.setMinimumHeight(cls.BUTTON_MIN_SIZE)
        button.setMaximumWidth(cls.BUTTON_MIN_SIZE)
        button.setMaximumHeight(cls.BUTTON_MIN_SIZE)

    @classmethod
    def apply_primary_button_style(cls, button: QPushButton) -> None:
        """Apply primary action button style (green)."""
        button.setStyleSheet(cls.PRIMARY_BUTTON_STYLE)

    @classmethod
    def apply_colored_icon_button_style(cls, button: QPushButton, bg_color: str, icon_color: str = "white") -> None:
        """Apply colored icon button style with no border, just background color.

        Args:
            button: The button to style
            bg_color: Background color (hex string)
            icon_color: Icon color (hex string, default white)
        """
        hover_color = cls._darken_hex_color(bg_color)
        style = f"""
        QPushButton {{
            background-color: {bg_color};
            border: none;
            border-radius: 4px;
            padding: 3px;
            min-height: {cls.BUTTON_MIN_SIZE}px;
            min-width: {cls.BUTTON_MIN_SIZE}px;
        }}
        QPushButton:hover {{
            background-color: {hover_color};
            padding: 2px;
            min-height: {cls.BUTTON_MIN_SIZE + 1}px;
            min-width: {cls.BUTTON_MIN_SIZE + 1}px;
        }}
        QPushButton:pressed {{
            opacity: 0.7;
            background-color: {bg_color};
        }}
        QPushButton:focus {{
            outline: none;
        }}
        QPushButton:disabled {{
            background-color: #595959;
        }}
        """
        button.setStyleSheet(style)
        button.setMinimumWidth(cls.BUTTON_MIN_SIZE)
        button.setMinimumHeight(cls.BUTTON_MIN_SIZE)
        button.setMaximumWidth(36)
        button.setMaximumHeight(36)

    @classmethod
    def apply_global_style(cls, app: QApplication) -> None:
        """Apply global stylesheet to the application with color variables."""
        app.setStyleSheet(cls.GLOBAL_STYLE)
