# src/teval/viz/__init__.py
from . import static

# Try to import interactive, but fail gracefully if dependencies are missing
try:
    from . import interactive
except ImportError:
    pass

__all__ = ["static", "interactive"]