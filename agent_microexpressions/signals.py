"""Signal types and data structures for agent micro-expression analysis."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any


class SignalType(enum.Enum):
    """Kinds of behavioral micro-expressions detectable in agent output."""

    HESITATION = "hesitation"
    CONFIDENCE = "confidence"
    UNCERTAINTY = "uncertainty"
    DEFLECTION = "deflection"
    ENTHUSIASM = "enthusiasm"
    EVASION = "evasion"


class SignalStrength(enum.Enum):
    """Qualitative strength of a detected signal."""

    NONE = "none"
    FAINT = "faint"
    MODERATE = "moderate"
    STRONG = "strong"
    OVERWHELMING = "overwhelming"

    @classmethod
    def from_score(cls, score: float) -> SignalStrength:
        """Map a 0–1 score to a qualitative strength band."""
        if score < 0.05:
            return cls.NONE
        if score < 0.25:
            return cls.FAINT
        if score < 0.55:
            return cls.MODERATE
        if score < 0.80:
            return cls.STRONG
        return cls.OVERWHELMING


@dataclass(frozen=True)
class Signal:
    """A single detected behavioral signal.

    Attributes:
        signal_type: The category of micro-expression.
        score: Normalized intensity in [0, 1].
        strength: Qualitative band derived from *score*.
        indicators: Concrete textual evidence that contributed to detection.
        metadata: Arbitrary extra information.
    """

    signal_type: SignalType
    score: float
    strength: SignalStrength = field(default=None)
    indicators: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp score
        score = max(0.0, min(1.0, self.score))
        object.__setattr__(self, "score", score)
        if math.isnan(score):
            object.__setattr__(self, "score", 0.0)
        # Derive strength
        if self.strength is None:
            object.__setattr__(self, "strength", SignalStrength.from_score(self.score))
