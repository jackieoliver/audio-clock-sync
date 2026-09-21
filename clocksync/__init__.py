"""Recover clock offsets between recording devices from the ambient audio they both heard."""
from .core import Match, Offset, Hypothesis, cross_correlate, measure_offset, rank_hypotheses

__all__ = ["Match", "Offset", "Hypothesis", "cross_correlate", "measure_offset", "rank_hypotheses"]
__version__ = "0.1.0"
