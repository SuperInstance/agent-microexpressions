"""Tests for agent_microexpressions.detector."""

from agent_microexpressions.detector import MicroexpressionDetector, DetectionContext
from agent_microexpressions.signals import SignalType


def _detect(text: str, latency_ms: float | None = None) -> dict[SignalType, float]:
    ctx = DetectionContext(latency_ms=latency_ms)
    detector = MicroexpressionDetector()
    signals = detector.detect(text, ctx)
    return {s.signal_type: s.score for s in signals}


class TestMicroexpressionDetector:
    def test_returns_all_signal_types(self):
        scores = _detect("Hello world.")
        assert set(scores.keys()) == set(SignalType)

    def test_confident_text(self):
        scores = _detect(
            "I will absolutely and definitely solve this problem clearly. "
            "The solution is certain and obvious."
        )
        assert scores[SignalType.CONFIDENCE] > 0.1

    def test_hedging_text(self):
        scores = _detect(
            "Maybe perhaps it might possibly seem somewhat likely that "
            "this could apparently be roughly correct."
        )
        assert scores[SignalType.HESITATION] > 0.1
        assert scores[SignalType.UNCERTAINTY] > 0.1

    def test_enthusiastic_text(self):
        scores = _detect(
            "This is amazing! I love it! Fantastic work! AWESOME results!"
        )
        assert scores[SignalType.ENTHUSIASM] > 0.2

    def test_evasive_text(self):
        scores = _detect(
            "I can't really say. It's hard to tell. "
            "That's beyond the scope of what I can discuss."
        )
        assert scores[SignalType.EVASION] > 0.1

    def test_deflection_text(self):
        scores = _detect(
            "That said, moving on to other matters. "
            "Anyway, let's focus on something else."
        )
        assert scores[SignalType.DEFLECTION] > 0.1

    def test_high_latency_boosts_hesitation(self):
        scores_normal = _detect("I think maybe this works.", latency_ms=1500)
        scores_slow = _detect("I think maybe this works.", latency_ms=8000)
        assert scores_slow[SignalType.HESITATION] > scores_normal[SignalType.HESITATION]

    def test_empty_text(self):
        scores = _detect("")
        for st in SignalType:
            assert st in scores
            assert 0.0 <= scores[st] <= 1.0

    def test_indicators_populated(self):
        detector = MicroexpressionDetector()
        signals = detector.detect(
            "Perhaps maybe this is possibly correct. I think so."
        )
        all_indicators = []
        for s in signals:
            all_indicators.extend(s.indicators)
        assert len(all_indicators) > 0

    def test_scores_bounded(self):
        texts = [
            "No.",
            "Perhaps maybe might could possibly seem somewhat apparently roughly "
            "probably typically usually sometimes allegedly reportedly presumably "
            "arguably supposedly approximately.",
            "DEFINITELY CERTAINLY ABSOLUTELY CLEARLY OBVIOUSLY UNDOUBTEDLY!!!!!",
        ]
        for text in texts:
            scores = _detect(text)
            for score in scores.values():
                assert 0.0 <= score <= 1.0
