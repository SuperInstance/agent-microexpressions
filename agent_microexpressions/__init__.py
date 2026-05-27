"""
agent-microexpressions — Detect subtle behavioral signals in AI agent interactions.

Analyze agent outputs for micro-expression analogs: hesitation, confidence,
uncertainty, deflection, enthusiasm, and evasion. Encode signals as numerical
vectors, aggregate them into behavioral profiles, and track changes over time.
"""

from .signals import (
    SignalType,
    Signal,
    SignalStrength,
)
from .detector import MicroexpressionDetector
from .encoder import SignalEncoder
from .aggregator import SignalAggregator, BehavioralProfile
from .timeline import Timeline, TimelineEntry

__version__ = "0.1.0"
__all__ = [
    "SignalType",
    "Signal",
    "SignalStrength",
    "MicroexpressionDetector",
    "SignalEncoder",
    "SignalAggregator",
    "BehavioralProfile",
    "Timeline",
    "TimelineEntry",
]
