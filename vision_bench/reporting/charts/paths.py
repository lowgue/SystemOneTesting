"""Shared locations and report-relative names for generated charts."""

from pathlib import Path


def chart_directory(output, category):
    """Create and return the output folder for one chart category."""
    directory = Path(output) / "charts" / category
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def chart_reference(category, stem):
    """Return the report-relative path for a chart file."""
    return f"charts/{category}/{stem}"
