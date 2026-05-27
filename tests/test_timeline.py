"""Tests for agent_microexpressions.timeline."""

import math

from agent_microexpressions.signals import SignalType
from agent_microexpressions.timeline import Timeline


class TestTimeline:
    def test_add_turn(self):
        tl = Timeline()
        entry = tl.add_turn("Hello world.")
        assert entry.turn_index == 0
        assert len(entry.signals) == len(SignalType)
        assert entry.dominant_signal().signal_type in SignalType

    def test_sequential_turns(self):
        tl = Timeline()
        tl.add_turn("First response.")
        tl.add_turn("Second response.")
        tl.add_turn("Third response.")
        assert len(tl) == 3
        assert tl[0].turn_index == 0
        assert tl[2].turn_index == 2

    def test_drift_identical_text(self):
        tl = Timeline()
        tl.add_turn("The same text.")
        tl.add_turn("The same text.")
        # Should have low drift (may not be exactly 0 due to text features)
        assert tl.drift_at(1) < 0.5

    def test_drift_different_text(self):
        tl = Timeline()
        tl.add_turn("I will definitely do this!")
        tl.add_turn("Maybe perhaps I can't really say.")
        drift = tl.drift_at(1)
        assert drift > 0.0

    def test_total_drift(self):
        tl = Timeline()
        tl.add_turn("A")
        tl.add_turn("B")
        tl.add_turn("C")
        total = tl.total_drift()
        assert total >= 0.0

    def test_average_drift(self):
        tl = Timeline()
        tl.add_turn("X")
        tl.add_turn("Y")
        avg = tl.average_drift()
        assert avg == tl.total_drift() / 1

    def test_trajectory(self):
        tl = Timeline()
        tl.add_turn("Certainly!")
        tl.add_turn("Maybe...")
        traj = tl.trajectory()
        assert set(traj.keys()) == set(SignalType)
        for st in SignalType:
            assert len(traj[st]) == 2

    def test_dominant_signal_series(self):
        tl = Timeline()
        tl.add_turn("Great!")
        tl.add_turn("Maybe.")
        series = tl.dominant_signal_series()
        assert len(series) == 2
        assert all(isinstance(s, SignalType) for s in series)

    def test_volatility(self):
        tl = Timeline()
        # Need at least 3 entries for volatility
        tl.add_turn("A")
        tl.add_turn("B")
        tl.add_turn("C")
        vol = tl.volatility()
        assert vol >= 0.0

    def test_volatility_too_few(self):
        tl = Timeline()
        tl.add_turn("A")
        tl.add_turn("B")
        assert tl.volatility() == 0.0

    def test_drift_at_boundary(self):
        tl = Timeline()
        tl.add_turn("Hello")
        assert tl.drift_at(0) == 0.0  # First turn
        assert tl.drift_at(5) == 0.0  # Out of range

    def test_entries_immutable_view(self):
        tl = Timeline()
        tl.add_turn("Test.")
        entries = tl.entries
        assert len(entries) == 1

    def test_reset(self):
        tl = Timeline()
        tl.add_turn("A")
        tl.reset()
        assert len(tl) == 0

    def test_latency_stored(self):
        tl = Timeline()
        entry = tl.add_turn("Hello", latency_ms=2500.0)
        assert entry.latency_ms == 2500.0
