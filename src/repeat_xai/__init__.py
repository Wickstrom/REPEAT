"""
REPEAT

A framework for uncertainty estimation in representation learning
explainability.
"""

from repeat_xai.repeat import REPEAT
from repeat_xai.thresholds import (
    THRESHOLD_METHODS,
    threshold_li,
    threshold_mean,
    threshold_otsu,
    threshold_triangle,
)

__version__ = "0.1.0"
__author__ = "Kristoffer Wickstrøm"
__credits__ = "UiT The Arctic University of Norway"

__all__ = [
    "REPEAT",
    "THRESHOLD_METHODS",
    "threshold_li",
    "threshold_mean",
    "threshold_otsu",
    "threshold_triangle",
]
