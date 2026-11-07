"""Chart generation grouped by the kind of comparison they present."""

from .comparison import generate_performance_charts
from .quality import generate_quality_charts
from .summary import generate_summary_charts

__all__ = [
    "generate_performance_charts",
    "generate_quality_charts",
    "generate_summary_charts",
]
