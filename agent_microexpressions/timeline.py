"""Timeline — track behavioral signal changes across conversation turns."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .signals import Signal, SignalType
from .detector import MicroexpressionDetector, DetectionContext
from .encoder import SignalEncoder, EncodedVector


@dataclass(frozen=True)
class TimelineEntry:
    """A single point in a behavioral timeline.

    Attributes:
        turn_index: 0-based conversation turn number.
        signals: Detected signals at this turn.
        vector: Numerical encoding of the signals.
        agent_id: The agent that produced this turn (if tracked).
        latency_ms: Response latency for this turn.
    """

    turn_index: int
    signals: tuple[Signal, ...]
    vector: EncodedVector
    agent_id: str | None = None
    latency_ms: float | None = None

    def dominant_signal(self) -> Signal:
        """Return the signal with the highest score."""
        return max(self.signals, key=lambda s: s.score)


class Timeline:
    """Track how an agent's behavioral signals evolve over a conversation.

    Parameters:
        detector: The detector used for each new turn.  A default is created
            if not supplied.
        encoder: The encoder used for each entry's vector representation.
    """

    def __init__(
        self,
        detector: MicroexpressionDetector | None = None,
        encoder: SignalEncoder | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.detector = detector or MicroexpressionDetector()
        self.encoder = encoder or SignalEncoder()
        self.agent_id = agent_id
        self._entries: list[TimelineEntry] = []

    # -- mutators -----------------------------------------------------------

    def add_turn(
        self,
        text: str,
        latency_ms: float | None = None,
    ) -> TimelineEntry:
        """Analyse a new agent response and append it to the timeline.

        Returns the newly created :class:`TimelineEntry`.
        """
        turn_index = len(self._entries)
        ctx = DetectionContext(
            latency_ms=latency_ms,
            turn_index=turn_index,
            agent_id=self.agent_id,
            previous_text=(
                self._entries[-1].signals[0].metadata.get("_text")
                if self._entries and self._entries[-1].signals
                else None
            ),
        )
        signals = self.detector.detect(text, ctx)
        # Stash text so previous_turn lookup can find it if needed
        for s in signals:
            s.metadata["_text"] = text  # type: ignore[misc]

        vector = self.encoder.encode_signals(signals, text)

        entry = TimelineEntry(
            turn_index=turn_index,
            signals=tuple(signals),
            vector=vector,
            agent_id=self.agent_id,
            latency_ms=latency_ms,
        )
        self._entries.append(entry)
        return entry

    # -- accessors ----------------------------------------------------------

    @property
    def entries(self) -> Sequence[TimelineEntry]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> TimelineEntry:
        return self._entries[index]

    # -- analysis -----------------------------------------------------------

    def drift_at(self, turn_index: int) -> float:
        """Compute behavioral drift between *turn_index* and the previous turn.

        Drift is the Euclidean distance between the two encoded vectors.
        Returns 0.0 for the first turn.
        """
        if turn_index <= 0 or turn_index >= len(self._entries):
            return 0.0
        return self._entries[turn_index - 1].vector.euclidean_distance(
            self._entries[turn_index].vector,
        )

    def total_drift(self) -> float:
        """Sum of per-turn drift across the entire timeline."""
        return sum(self.drift_at(i) for i in range(1, len(self._entries)))

    def average_drift(self) -> float:
        """Average per-turn drift."""
        n = max(1, len(self._entries) - 1)
        return self.total_drift() / n

    def trajectory(self) -> dict[SignalType, list[float]]:
        """Per-signal score trajectory across all turns.

        Returns a dict mapping each signal type to a list of scores,
        one per turn.
        """
        result: dict[SignalType, list[float]] = {st: [] for st in SignalType}
        for entry in self._entries:
            lookup = {s.signal_type: s.score for s in entry.signals}
            for st in SignalType:
                result[st].append(lookup.get(st, 0.0))
        return result

    def dominant_signal_series(self) -> list[SignalType]:
        """List of the dominant signal at each turn."""
        return [entry.dominant_signal().signal_type for entry in self._entries]

    def volatility(self) -> float:
        """Standard deviation of per-turn drift — a measure of instability."""
        if len(self._entries) < 3:
            return 0.0
        drifts = [self.drift_at(i) for i in range(1, len(self._entries))]
        mean = sum(drifts) / len(drifts)
        variance = sum((d - mean) ** 2 for d in drifts) / len(drifts)
        return math.sqrt(variance)

    def reset(self) -> None:
        """Clear all timeline entries."""
        self._entries.clear()
