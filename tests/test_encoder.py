"""Tests for agent_microexpressions.encoder."""

import math

from agent_microexpressions.signals import Signal, SignalType
from agent_microexpressions.encoder import SignalEncoder, EncodedVector, SIGNAL_DIM, EXTENDED_DIM


def _make_signals(**overrides: float) -> list[Signal]:
    """Create one signal per type with optional score overrides."""
    result = []
    for st in SignalType:
        score = overrides.get(st.value, 0.0)
        result.append(Signal(signal_type=st, score=score))
    return result


class TestEncodedVector:
    def test_dim(self):
        v = EncodedVector(values=(0.1, 0.2, 0.3), labels=("a", "b", "c"))
        assert v.dim == 3

    def test_cosine_similarity_identical(self):
        v = EncodedVector(values=(0.5, 0.5, 0.5), labels=("x", "y", "z"))
        assert abs(v.cosine_similarity(v) - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal(self):
        a = EncodedVector(values=(1.0, 0.0), labels=("a", "b"))
        b = EncodedVector(values=(0.0, 1.0), labels=("a", "b"))
        assert abs(a.cosine_similarity(b)) < 1e-9

    def test_euclidean_distance(self):
        a = EncodedVector(values=(0.0, 0.0), labels=("a", "b"))
        b = EncodedVector(values=(3.0, 4.0), labels=("a", "b"))
        assert abs(a.euclidean_distance(b) - 5.0) < 1e-9

    def test_dimension_mismatch_raises(self):
        a = EncodedVector(values=(1.0,), labels=("a",))
        b = EncodedVector(values=(1.0, 2.0), labels=("a", "b"))
        try:
            a.cosine_similarity(b)
            assert False
        except ValueError:
            pass


class TestSignalEncoder:
    def test_basic_encoding_dimension(self):
        encoder = SignalEncoder()
        signals = _make_signals()
        vec = encoder.encode_signals(signals)
        assert vec.dim == SIGNAL_DIM
        assert len(vec.labels) == SIGNAL_DIM

    def test_labels_match_signal_types(self):
        encoder = SignalEncoder()
        signals = _make_signals()
        vec = encoder.encode_signals(signals)
        for label, st in zip(vec.labels, [
            SignalType.HESITATION, SignalType.CONFIDENCE,
            SignalType.UNCERTAINTY, SignalType.DEFLECTION,
            SignalType.ENTHUSIASM, SignalType.EVASION,
        ]):
            assert label == st.value

    def test_scores_preserved(self):
        encoder = SignalEncoder()
        signals = _make_signals(hesitation=0.8, confidence=0.3)
        vec = encoder.encode_signals(signals)
        assert vec.values[0] == 0.8  # hesitation is first
        assert vec.values[1] == 0.3  # confidence is second

    def test_extended_encoding(self):
        encoder = SignalEncoder(extended=True)
        signals = _make_signals()
        vec = encoder.encode_signals(signals, text="Hello world. Good!")
        assert vec.dim == EXTENDED_DIM
        assert vec.labels[-1] == "question_density"

    def test_extended_without_text(self):
        encoder = SignalEncoder(extended=True)
        signals = _make_signals()
        vec = encoder.encode_signals(signals, text=None)
        # Without text, extended features are not appended
        assert vec.dim == SIGNAL_DIM

    def test_missing_signals_default_zero(self):
        encoder = SignalEncoder()
        vec = encoder.encode_signals([Signal(signal_type=SignalType.CONFIDENCE, score=0.9)])
        assert vec.values[1] == 0.9  # confidence
        assert vec.values[0] == 0.0  # hesitation (missing)
