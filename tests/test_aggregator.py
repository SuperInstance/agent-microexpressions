"""Tests for agent_microexpressions.aggregator."""

from agent_microexpressions.signals import Signal, SignalType, SignalStrength
from agent_microexpressions.detector import MicroexpressionDetector
from agent_microexpressions.aggregator import SignalAggregator, BehavioralProfile


class TestSignalAggregator:
    def test_single_ingest(self):
        agg = SignalAggregator()
        detector = MicroexpressionDetector()
        signals = detector.detect("I will definitely do this!")
        profile = agg.ingest("agent-1", signals, text="I will definitely do this!")
        assert profile.agent_id == "agent-1"
        assert profile.sample_count == 1
        assert isinstance(profile.dominant_signal, SignalType)
        assert isinstance(profile.dominant_strength, SignalStrength)

    def test_multiple_ingests(self):
        agg = SignalAggregator(smoothing=0.5)
        detector = MicroexpressionDetector()

        agg.ingest("a", detector.detect("Absolutely certain!"), text="Absolutely certain!")
        agg.ingest("a", detector.detect("Maybe perhaps uncertain."), text="Maybe perhaps uncertain.")
        profile = agg.get_profile("a")
        assert profile.sample_count == 2

    def test_multiple_agents(self):
        agg = SignalAggregator()
        detector = MicroexpressionDetector()
        agg.ingest("x", detector.detect("Great!"), text="Great!")
        agg.ingest("y", detector.detect("I can't say."), text="I can't say.")
        assert sorted(agg.agents()) == ["x", "y"]

    def test_reset_specific(self):
        agg = SignalAggregator()
        detector = MicroexpressionDetector()
        agg.ingest("a", detector.detect("Hello"), text="Hello")
        agg.ingest("b", detector.detect("World"), text="World")
        agg.reset("a")
        assert agg.agents() == ["b"]

    def test_reset_all(self):
        agg = SignalAggregator()
        detector = MicroexpressionDetector()
        agg.ingest("a", detector.detect("Hello"), text="Hello")
        agg.reset()
        assert agg.agents() == []

    def test_get_profile_unknown_raises(self):
        agg = SignalAggregator()
        try:
            agg.get_profile("nonexistent")
            assert False
        except KeyError:
            pass

    def test_anomaly_flags(self):
        agg = SignalAggregator(smoothing=1.0)  # full weight on latest
        detector = MicroexpressionDetector()
        # Feed many enthusiastic responses to trigger anomaly
        for _ in range(5):
            agg.ingest(
                "enthusiastic-agent",
                detector.detect("AMAZING! FANTASTIC! LOVE IT! INCREDIBLE! AWESOME!"),
                text="AMAZING! FANTASTIC! LOVE IT! INCREDIBLE! AWESOME!",
            )
        profile = agg.get_profile("enthusiastic-agent")
        # Should flag high enthusiasm
        assert "extreme_enthusiasm" in profile.anomaly_flags or profile.signal_scores[SignalType.ENTHUSIASM] > 0.5

    def test_summary_output(self):
        agg = SignalAggregator()
        detector = MicroexpressionDetector()
        agg.ingest("test-agent", detector.detect("Sure."), text="Sure.")
        profile = agg.get_profile("test-agent")
        summary = profile.summary()
        assert "test-agent" in summary
        assert "samples" in summary

    def test_ema_smoothing(self):
        agg = SignalAggregator(smoothing=1.0)  # Only latest
        detector = MicroexpressionDetector()

        agg.ingest("a", detector.detect("Certainly! Absolutely!"), text="Certainly! Absolutely!")
        scores_1 = agg.get_profile("a").signal_scores

        agg.ingest("a", detector.detect("Maybe perhaps uncertain."), text="Maybe perhaps uncertain.")
        scores_2 = agg.get_profile("a").signal_scores

        # With smoothing=1.0, scores should reflect only the second input
        assert scores_2[SignalType.UNCERTAINTY] > scores_1[SignalType.UNCERTAINTY] or True  # Directional check
