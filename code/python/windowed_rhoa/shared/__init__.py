"""Shared package primitives for the exploratory windowed-RhoA pipeline."""

from .parameters import AllParams, default_params


__all__ = ["AllParams", "apply_publication_style", "default_params"]


def __getattr__(name: str):
    """Keep plotting dependencies optional for equation-only kernel imports."""
    if name == "apply_publication_style":
        from .figure_style import apply_publication_style

        return apply_publication_style
    raise AttributeError(name)
