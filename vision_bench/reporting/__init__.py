"""Reporting and chart generation for benchmark runs."""

from importlib import import_module

_EXPORTS = {
    "evaluate": ("dataset_metrics", "evaluate"),
    "field_scores": ("dataset_metrics", "field_scores"),
    "generate_comparison_report": ("report", "generate"),
    "generate_dataset_report": ("dataset_metrics", "generate"),
    "ratio": ("dataset_metrics", "ratio"),
}

__all__ = [
    "evaluate",
    "field_scores",
    "generate_comparison_report",
    "generate_dataset_report",
    "ratio",
]


def __getattr__(name):
    """Load a public reporting function only when a caller requests it."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, attribute)
