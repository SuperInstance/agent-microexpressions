"""SignalAggregator — combine multiple signals into behavioral profiles."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .signals import Signal, SignalType, SignalStrength
from .encoder import SignalEncoder, EncodedVector


@dataclass
class BehavioralProfile:
    """An aggregated snapshot of an agent's behavioral signals.

    Attributes:
        agent_id: Identifier of the agent this profile describes.
        dominant_signal: The signal type with the highest score.
        dominant_strength: Qualitative strength of the dominant signal.
        signal_scores: Mapping of each signal type to its aggregated score.
        vector: Numerical encoding of the profile.
        anomaly_flags: Any detected anomalies (e.g. extreme sentiment).
        timestamp: Unix timestamp of when the profile was created.
        sample_count: How many responses contributed to this profile.
    """

    agent_id: str
    dominant_signal: SignalType
    dominant_strength: SignalStrength
    signal_scores: dict[SignalType, float]
    vector: EncodedVector
    anomaly_flags: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)
    sample_count: int = 1

    def summary(self) -> str:
        """Human-readable one-line summary."""
        parts = [f"[{self.agent_id}]"]
        parts.append(f"dominant: {self.dominant_signal.value}({self.dominant_strength.value})")
        if self.anomaly_flags:
            parts.append(f"anomalies: {', '.join(self.anomaly_flags)}")
        parts.append(f"samples: {self.sample_count}")
        return " | ".join(parts)


class SignalAggregator:
    """Aggregate per-response signals into a running behavioral profile.

    The aggregator maintains an exponential moving average per signal type so
    that recent observations carry more weight than older ones.

    Parameters:
        smoothing: EMA smoothing factor (0–1).  Higher values weight recent
            observations more heavily.  ``0.3`` means 30% new, 70% history.
        encoder: The :class:`SignalEncoder` used to produce vectors.  A
            default encoder is created if not supplied.
    """

    def __init__(
        self,
        smoothing: float = 0.3,
        encoder: SignalEncoder | None = None,
    ) -> None:
        self.smoothing = max(0.0, min(1.0, smoothing))
        self.encoder = encoder or SignalEncoder()
        # agent_id → running scores
        self._profiles: dict[str, dict[SignalType, float]] = {}
        self._sample_counts: dict[str, int] = {}
        # Store last text per agent for extended encoding
        self._last_text: dict[str, str | None] = {}

    def ingest(
        self,
        agent_id: str,
        signals: list[Signal],
        text: str | None = None,
    ) -> BehavioralProfile:
        """Add a new observation for *agent_id* and return the updated profile."""
        current = self._profiles.get(agent_id, {})
        new_scores = {s.signal_type: s.score for s in signals}

        # EMA update
        for st in SignalType:
            old = current.get(st, 0.0)
            new = new_scores.get(st, 0.0)
            current[st] = old * (1 - self.smoothing) + new * self.smoothing

        self._profiles[agent_id] = current
        self._sample_counts[agent_id] = self._sample_counts.get(agent_id, 0) + 1
        self._last_text[agent_id] = text

        return self.get_profile(agent_id)

    def get_profile(self, agent_id: str) -> BehavioralProfile:
        """Return the current behavioral profile for *agent_id*."""
        scores = self._profiles.get(agent_id, {})
        if not scores:
            raise KeyError(f"No data for agent '{agent_id}'")

        dominant = max(scores, key=lambda st: scores.get(st, 0.0))
        dominant_score = scores[dominant]

        # Build synthetic signals for encoder
        synth_signals = [
            Signal(signal_type=st, score=sc)
            for st, sc in scores.items()
        ]
        vector = self.encoder.encode_signals(synth_signals, self._last_text.get(agent_id))

        # Anomaly detection
        anomalies: list[str] = []
        if scores.get(SignalType.ENTHUSIASM, 0) > 0.8:
            anomalies.append("extreme_enthusiasm")
        if scores.get(SignalType.EVASION, 0) > 0.7:
            anomalies.append("high_evasion")
        if scores.get(SignalType.CONFIDENCE, 0) < 0.05 and self._sample_counts.get(agent_id, 0) > 3:
            anomalies.append("chronic_low_confidence")
        if scores.get(SignalType.UNCERTAINTY, 0) > 0.7:
            anomalies.append("high_uncertainty")

        return BehavioralProfile(
            agent_id=agent_id,
            dominant_signal=dominant,
            dominant_strength=SignalStrength.from_score(dominant_score),
            signal_scores=dict(scores),
            vector=vector,
            anomaly_flags=tuple(anomalies),
            sample_count=self._sample_counts.get(agent_id, 0),
        )

    def agents(self) -> list[str]:
        """Return all tracked agent IDs."""
        return list(self._profiles.keys())

    def reset(self, agent_id: str | None = None) -> None:
        """Reset profile data.  If *agent_id* is ``None``, reset all."""
        if agent_id is None:
            self._profiles.clear()
            self._sample_counts.clear()
            self._last_text.clear()
        else:
            self._profiles.pop(agent_id, None)
            self._sample_counts.pop(agent_id, None)
            self._last_text.pop(agent_id, None)
