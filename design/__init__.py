"""Design system package for AI VideoClipper.

This package contains all design-related functionality:
- Icon management (Material Design Icons)
- Style management (colors, buttons, global styling)
- Assets (fonts, icons)
"""

from design.icon_manager import IconManager
from design.style_manager import StyleManager, Colors

__all__ = ['IconManager', 'StyleManager', 'Colors']
