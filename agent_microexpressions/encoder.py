"""SignalEncoder — convert behavioral signals into fixed-length numerical vectors.

Each of the six signal types maps to a dimension.  A secondary set of
derived features (sentiment polarity, verbosity ratio, formality, and
question density) can optionally be appended for richer representations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .signals import Signal, SignalType

# Canonical ordering — vector indices correspond to this list
_SIGNAL_ORDER: list[SignalType] = [
    SignalType.HESITATION,
    SignalType.CONFIDENCE,
    SignalType.UNCERTAINTY,
    SignalType.DEFLECTION,
    SignalType.ENTHUSIASM,
    SignalType.EVASION,
]

SIGNAL_DIM = len(_SIGNAL_ORDER)
EXTENDED_DIM = SIGNAL_DIM + 4  # + sentiment, verbosity, formality, question_density


@dataclass(frozen=True)
class EncodedVector:
    """A numerical representation of a set of behavioral signals.

    Attributes:
        values: The raw float values.
        labels: Human-readable labels for each dimension.
    """

    values: tuple[float, ...]
    labels: tuple[str, ...]

    @property
    def dim(self) -> int:
        return len(self.values)

    def cosine_similarity(self, other: EncodedVector) -> float:
        """Cosine similarity between two vectors of matching dimension."""
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        dot = sum(a * b for a, b in zip(self.values, other.values))
        mag_a = math.sqrt(sum(a * a for a in self.values))
        mag_b = math.sqrt(sum(b * b for b in other.values))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def euclidean_distance(self, other: EncodedVector) -> float:
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.values, other.values)))


class SignalEncoder:
    """Encode a collection of :class:`Signal` objects into a numerical vector.

    Parameters:
        extended: If *True*, append four derived text-analysis features
            (sentiment polarity, verbosity, formality, question density)
            extracted from the original text.
    """

    def __init__(self, *, extended: bool = False) -> None:
        self.extended = extended

    def encode_signals(
        self,
        signals: list[Signal],
        text: str | None = None,
    ) -> EncodedVector:
        """Produce a vector from *signals*.

        When *extended* is ``True`` and *text* is provided, four additional
        features are appended.
        """
        lookup = {s.signal_type: s.score for s in signals}
        values: list[float] = [lookup.get(st, 0.0) for st in _SIGNAL_ORDER]
        labels: list[str] = [st.value for st in _SIGNAL_ORDER]

        if self.extended and text is not None:
            ext = _text_features(text)
            values.extend(ext)
            labels.extend(["sentiment", "verbosity", "formality", "question_density"])

        return EncodedVector(values=tuple(values), labels=tuple(labels))


# ---------------------------------------------------------------------------
# Text-derived extended features
# ---------------------------------------------------------------------------

_POSITIVE = {"good", "great", "excellent", "positive", "happy", "success", "love", "amazing", "fantastic"}
_NEGATIVE = {"bad", "poor", "negative", "unhappy", "fail", "error", "wrong", "terrible", "awful"}
_FORMAL = {"therefore", "however", "moreover", "furthermore", "consequently", "thus", "hence", "accordingly"}


def _text_features(text: str) -> list[float]:
    words = text.lower().split()
    wc = len(words) or 1
    sentences = [s for s in __import__("re").split(r"[.!?]+", text) if s.strip()]
    sc = len(sentences) or 1

    # Sentiment polarity ∈ [-1, 1] → normalise to [0, 1]
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    total = pos + neg or 1
    sentiment = (pos - neg) / total  # [-1, 1]
    sentiment_norm = (sentiment + 1.0) / 2.0

    # Verbosity: avg words per sentence, cap at 1.0 via sigmoid-ish
    verbosity = min(1.0, (wc / sc) / 30.0)

    # Formality
    formals = sum(1 for w in words if w in _FORMAL)
    formality = min(1.0, formals / 5.0)

    # Question density
    questions = text.count("?")
    question_density = min(1.0, questions / sc)

    return [sentiment_norm, verbosity, formality, question_density]
