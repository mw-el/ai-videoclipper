"""Material Design Icons manager for PyQt6 applications."""

from pathlib import Path
from PyQt6.QtGui import QFont, QIcon, QPixmap, QPainter, QColor
from PyQt6.QtCore import QSize


class MaterialIcon:
    """Material Design Icon definition with Unicode codepoint."""

    def __init__(self, codepoint: str, name: str):
        """
        Initialize a Material Design icon.

        Args:
            codepoint: Unicode codepoint (hex string, e.g., 'e145')
            name: Human-readable name (e.g., 'play_arrow')
        """
        self.codepoint = codepoint
        self.name = name
        self.char = chr(int(codepoint, 16))

    def __str__(self) -> str:
        return self.char


class IconManager:
    """Manages Material Design Icons and provides icon creation methods."""

    # Material Design Icons - Line/Outlined variants (Unicode codepoints)
    ICONS = {
        'play_arrow': MaterialIcon('e037', 'play_arrow'),
        'pause': MaterialIcon('e047', 'pause'),
        'edit': MaterialIcon('e3c9', 'edit'),  # Edit/pencil
        'add': MaterialIcon('e145', 'add'),
        'delete': MaterialIcon('e872', 'delete'),
        'check': MaterialIcon('e5ca', 'check'),
        'close': MaterialIcon('e5cd', 'close'),
        'settings': MaterialIcon('e8b8', 'settings'),
        'save': MaterialIcon('e161', 'save'),  # Save/disk icon
        'folder_open': MaterialIcon('e2c8', 'folder_open'),  # Open/load file
        'upload': MaterialIcon('e2c6', 'upload'),  # Upload
        'download': MaterialIcon('e2be', 'download'),  # Download
        'folder': MaterialIcon('e2c7', 'folder'),
        'file': MaterialIcon('e226', 'file'),
        'search': MaterialIcon('e8b6', 'search'),
        'menu': MaterialIcon('e5d2', 'menu'),
        'more_vert': MaterialIcon('e5d4', 'more_vert'),
        'content_copy': MaterialIcon('e14d', 'content_copy'),  # Duplicate/Copy
        'call_split': MaterialIcon('e129', 'call_split'),  # Split into two calls
        'cut': MaterialIcon('e14f', 'cut'),  # Cut/scissors for split
        'unfold_more': MaterialIcon('e5d8', 'unfold_more'),  # Split/expand icon
        'split_screen': MaterialIcon('e188', 'split_screen'),  # Split screen icon
        'split_scene': MaterialIcon('e2d8', 'split_scene'),  # Split scene icon
        'keyboard_arrow_left': MaterialIcon('e314', 'keyboard_arrow_left'),  # Start marker
        'keyboard_arrow_right': MaterialIcon('e315', 'keyboard_arrow_right'),  # End marker
        'skip_previous': MaterialIcon('e045', 'skip_previous'),
        'skip_next': MaterialIcon('e044', 'skip_next'),
        'movie': MaterialIcon('e02c', 'movie'),  # Film strip/reel icon
    }

    _font: QFont | None = None
    _font_size: int = 18

    @classmethod
    def initialize(cls, font_path: str | None = None, size: int = 18) -> None:
        """
        Initialize the icon manager with Material Design font.

        Args:
            font_path: Path to MaterialIcons-Regular.ttf. If None, uses default.
            size: Default font size for icons (default: 18)
        """
        if font_path is None:
            font_path = str(Path(__file__).parent / 'fonts' / 'MaterialIcons-Regular.ttf')

        cls._font_size = size
        cls._font = QFont()

        # Try to load the font
        if Path(font_path).exists():
            from PyQt6.QtGui import QFontDatabase
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id >= 0:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    cls._font.setFamily(families[0])
                    cls._font.setPixelSize(size)
                    cls._font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
            else:
                print(f"Warning: Could not load Material Design Icons font from {font_path}")
                cls._font = None
        else:
            print(f"Warning: Font file not found at {font_path}")
            cls._font = None

    @classmethod
    def get_font(cls, size: int | None = None) -> QFont:
        """Get the Material Design Icons font with optional size override."""
        if cls._font is None:
            raise RuntimeError("IconManager not initialized. Call IconManager.initialize() first.")

        font = QFont(cls._font)
        if size is not None:
            font.setPixelSize(size)
        return font

    @classmethod
    def get_icon_char(cls, icon_name: str) -> str:
        """
        Get the character for an icon by name.

        Args:
            icon_name: Icon name (e.g., 'play_arrow')

        Returns:
            Unicode character representing the icon
        """
        if icon_name not in cls.ICONS:
            raise ValueError(f"Unknown icon: {icon_name}")
        return str(cls.ICONS[icon_name])

    @classmethod
    def create_icon(cls, icon_name: str, color: QColor | str = "black",
                   size: int | None = None) -> QIcon:
        """
        Create a QIcon from a Material Design icon.

        Args:
            icon_name: Icon name (e.g., 'play_arrow')
            color: Color for the icon (default: black). Can be hex string or QColor.
            size: Icon size in pixels (default: _font_size)

        Returns:
            QIcon ready to use with QPushButton, etc.
        """
        if cls._font is None:
            raise RuntimeError("IconManager not initialized. Call IconManager.initialize() first.")

        if isinstance(color, str):
            color = QColor(color)

        if size is None:
            size = cls._font_size

        # Create a pixmap with the icon character
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))  # Transparent background

        painter = QPainter(pixmap)
        font = cls.get_font(size)
        painter.setFont(font)
        painter.setPen(color)

        # Draw the icon character centered
        icon_char = cls.get_icon_char(icon_name)
        rect = pixmap.rect()
        painter.drawText(rect, 1, icon_char)  # 1 = Qt.AlignmentFlag.AlignCenter

        painter.end()

        return QIcon(pixmap)
