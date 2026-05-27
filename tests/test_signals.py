"""Tests for agent_microexpressions.signals."""

from agent_microexpressions.signals import Signal, SignalType, SignalStrength


class TestSignalStrength:
    def test_from_score_none(self):
        assert SignalStrength.from_score(0.0) == SignalStrength.NONE
        assert SignalStrength.from_score(0.01) == SignalStrength.NONE

    def test_from_score_faint(self):
        assert SignalStrength.from_score(0.05) == SignalStrength.FAINT
        assert SignalStrength.from_score(0.24) == SignalStrength.FAINT

    def test_from_score_moderate(self):
        assert SignalStrength.from_score(0.25) == SignalStrength.MODERATE
        assert SignalStrength.from_score(0.54) == SignalStrength.MODERATE

    def test_from_score_strong(self):
        assert SignalStrength.from_score(0.55) == SignalStrength.STRONG
        assert SignalStrength.from_score(0.79) == SignalStrength.STRONG

    def test_from_score_overwhelming(self):
        assert SignalStrength.from_score(0.80) == SignalStrength.OVERWHELMING
        assert SignalStrength.from_score(1.0) == SignalStrength.OVERWHELMING


class TestSignal:
    def test_creation(self):
        s = Signal(signal_type=SignalType.CONFIDENCE, score=0.7)
        assert s.signal_type == SignalType.CONFIDENCE
        assert s.score == 0.7
        assert s.strength == SignalStrength.STRONG
        assert s.indicators == ()
        assert s.metadata == {}

    def test_score_clamped_high(self):
        s = Signal(signal_type=SignalType.HESITATION, score=5.0)
        assert s.score == 1.0

    def test_score_clamped_low(self):
        s = Signal(signal_type=SignalType.HESITATION, score=-1.0)
        assert s.score == 0.0

    def test_frozen(self):
        s = Signal(signal_type=SignalType.ENTHUSIASM, score=0.5)
        try:
            s.score = 0.9  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_with_indicators(self):
        s = Signal(
            signal_type=SignalType.EVASION,
            score=0.6,
            indicators=("evasion_phrases(2)",),
            metadata={"agent": "gpt-4"},
        )
        assert len(s.indicators) == 1
        assert s.metadata["agent"] == "gpt-4"
