"""MicroexpressionDetector — analyze agent response text for behavioral signals.

The detector examines latency (when provided), word choice, hedging language,
structural patterns, and sentiment to produce a list of :class:`Signal` objects.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Sequence

from .signals import Signal, SignalType, SignalStrength


# ---------------------------------------------------------------------------
# Word / pattern lists
# ---------------------------------------------------------------------------

_HEDGE_WORDS = frozenset([
    "maybe", "perhaps", "possibly", "might", "could", "seem", "seems",
    "apparently", "somewhat", "arguably", "presumably", "supposedly",
    "allegedly", "reportedly", "roughly", "approximately", "likely",
    "probably", "usually", "typically", "often", "sometimes",
])

_CERTAINTY_WORDS = frozenset([
    "definitely", "certainly", "absolutely", "clearly", "obviously",
    "undoubtedly", "unquestionably", "plainly", "surely", "guaranteed",
    "always", "never", "must", "will", "shall",
])

_ENTHUSIASM_WORDS = frozenset([
    "great", "excellent", "amazing", "fantastic", "wonderful", "awesome",
    "love", "exciting", "thrilled", "delighted", "incredible", "perfect",
    "brilliant", "outstanding", "superb", "phenomenal",
])

_NEGATIVE_WORDS = frozenset([
    "bad", "poor", "terrible", "awful", "horrible", "unfortunately",
    "sadly", "fail", "error", "wrong", "problem", "issue", "negative",
    "unhappy", "disappointing", "frustrating",
])

_DEFLECTION_PATTERNS = [
    re.compile(r"\b(that said|having said that|be that as it may)\b", re.I),
    re.compile(r"\b(on the other hand|at the same time)\b", re.I),
    re.compile(r"\b(let me (?:re)?phrase\b)", re.I),
    re.compile(r"\bI('d| would) (?:say|argue|suggest) (?:that )?rather\b", re.I),
    re.compile(r"\b(moving on|anyway|in any case)\b", re.I),
]

_EVASION_PATTERNS = [
    re.compile(r"\b(I(?:'m| am) not (?:sure|certain|able) (?:I|to))\b", re.I),
    re.compile(r"\b(I (?:can't|cannot|won't) (?:really |necessarily )?(?:say|tell|answer|comment|discuss))\b", re.I),
    re.compile(r"\b(it(?:'s| is) (?:hard|difficult) to say)\b", re.I),
    re.compile(r"\b(I'd (?:rather|prefer) not)\b", re.I),
    re.compile(r"\b(no comment)\b", re.I),
    re.compile(r"\b(beyond (?:the )?scope)\b", re.I),
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split()) or 1


def _sentence_count(text: str) -> int:
    return max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))


def _match_count(text: str, words: frozenset[str]) -> int:
    return sum(1 for w in text.lower().split() if w in words)


def _pattern_matches(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    hits: list[str] = []
    for pat in patterns:
        for m in pat.finditer(text):
            hits.append(m.group(0))
    return hits


# ---------------------------------------------------------------------------
# Detection context
# ---------------------------------------------------------------------------

@dataclass
class DetectionContext:
    """Optional metadata about the agent response being analyzed.

    Attributes:
        latency_ms: Wall-clock latency of the response (None if unavailable).
        turn_index: Which turn in the conversation (0-based).
        agent_id: Identifier for the agent that produced the response.
        previous_text: The agent's previous response, if any.
    """

    latency_ms: float | None = None
    turn_index: int = 0
    agent_id: str | None = None
    previous_text: str | None = None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class MicroexpressionDetector:
    """Detect behavioral micro-expressions in a single agent response.

    Parameters:
        latency_baseline_ms: Typical response latency; deviations above this
            amplify :attr:`SignalType.HESITATION`.
        hedge_weight: Multiplier for hedging signal sensitivity.
    """

    def __init__(
        self,
        latency_baseline_ms: float = 1500.0,
        hedge_weight: float = 1.0,
    ) -> None:
        self.latency_baseline_ms = latency_baseline_ms
        self.hedge_weight = hedge_weight

    # -- public API ----------------------------------------------------------

    def detect(
        self,
        text: str,
        context: DetectionContext | None = None,
    ) -> list[Signal]:
        """Analyze *text* and return all detected :class:`Signal` objects."""
        ctx = context or DetectionContext()
        signals: list[Signal] = []

        signals.append(self._hesitation(text, ctx))
        signals.append(self._confidence(text, ctx))
        signals.append(self._uncertainty(text, ctx))
        signals.append(self._deflection(text, ctx))
        signals.append(self._enthusiasm(text, ctx))
        signals.append(self._evasion(text, ctx))

        return signals

    # -- individual signal detectors -----------------------------------------

    def _hesitation(self, text: str, ctx: DetectionContext) -> Signal:
        indicators: list[str] = []

        # Latency factor
        latency_score = 0.0
        if ctx.latency_ms is not None and self.latency_baseline_ms > 0:
            ratio = ctx.latency_ms / self.latency_baseline_ms
            if ratio > 2.0:
                latency_score = min(1.0, (ratio - 2.0) / 4.0)
                indicators.append(f"high_latency({ctx.latency_ms:.0f}ms)")

        # Hedge word factor
        hedge_count = _match_count(text, _HEDGE_WORDS)
        wc = _word_count(text)
        hedge_ratio = hedge_count / wc
        hedge_score = min(1.0, hedge_ratio * 12.0) * self.hedge_weight

        # Ellipsis / trailing off
        ellipsis_count = len(re.findall(r"\.{2,}", text))
        ellipsis_score = min(1.0, ellipsis_count / 3.0)
        if ellipsis_count:
            indicators.append(f"ellipsis({ellipsis_count})")
        if hedge_count:
            indicators.append(f"hedge_words({hedge_count})")

        score = _blend(latency_score * 0.35, hedge_score * 0.45, ellipsis_score * 0.20)
        return Signal(
            signal_type=SignalType.HESITATION,
            score=score,
            indicators=tuple(indicators),
            metadata={"latency_ms": ctx.latency_ms},
        )

    def _confidence(self, text: str, ctx: DetectionContext) -> Signal:
        indicators: list[str] = []

        certainty_count = _match_count(text, _CERTAINTY_WORDS)
        wc = _word_count(text)
        certainty_ratio = certainty_count / wc
        certainty_score = min(1.0, certainty_ratio * 15.0)
        if certainty_count:
            indicators.append(f"certainty_words({certainty_count})")

        # Declarative sentences vs questions
        sc = _sentence_count(text)
        questions = text.count("?")
        declarative = max(0, sc - questions)
        decl_ratio = declarative / sc if sc else 0.5
        if questions > sc * 0.5:
            indicators.append(f"many_questions({questions})")

        # Short first-person statements ("I will", "I can")
        first_person_assertions = len(
            re.findall(r"\bI (?:will|can|am going to|plan to)\b", text, re.I)
        )
        assertion_score = min(1.0, first_person_assertions / 3.0)
        if first_person_assertions:
            indicators.append(f"assertions({first_person_assertions})")

        score = _blend(
            certainty_score * 0.40,
            decl_ratio * 0.30,
            assertion_score * 0.30,
        )
        return Signal(
            signal_type=SignalType.CONFIDENCE,
            score=score,
            indicators=tuple(indicators),
        )

    def _uncertainty(self, text: str, ctx: DetectionContext) -> Signal:
        indicators: list[str] = []

        # Hedge words
        hedge_count = _match_count(text, _HEDGE_WORDS)
        wc = _word_count(text)
        hedge_ratio = hedge_count / wc

        # Questions
        questions = text.count("?")
        sc = _sentence_count(text)
        question_ratio = questions / sc if sc else 0.0

        # "I think", "I believe", "I'm not sure"
        uncertainty_phrases = len(
            re.findall(
                r"\bI (?:think|believe|guess|suppose|assume|reckon)\b|\bnot sure\b|\buncertain\b",
                text,
                re.I,
            )
        )

        if hedge_count:
            indicators.append(f"hedge_words({hedge_count})")
        if questions:
            indicators.append(f"questions({questions})")
        if uncertainty_phrases:
            indicators.append(f"uncertainty_phrases({uncertainty_phrases})")

        score = _blend(
            min(1.0, hedge_ratio * 12.0) * 0.35,
            min(1.0, question_ratio) * 0.30,
            min(1.0, uncertainty_phrases / 3.0) * 0.35,
        )
        return Signal(
            signal_type=SignalType.UNCERTAINTY,
            score=score,
            indicators=tuple(indicators),
        )

    def _deflection(self, text: str, ctx: DetectionContext) -> Signal:
        matches = _pattern_matches(text, _DEFLECTION_PATTERNS)
        # Topic shifting: many distinct subjects / bullet re-directions
        bullet_redirects = len(re.findall(r"\b(?:anyway|moving on|let's focus|back to)\b", text, re.I))

        indicators: list[str] = []
        if matches:
            indicators.append(f"deflection_phrases({len(matches)})")
        if bullet_redirects:
            indicators.append(f"redirects({bullet_redirects})")

        score = _blend(
            min(1.0, len(matches) / 3.0) * 0.65,
            min(1.0, bullet_redirects / 2.0) * 0.35,
        )
        return Signal(
            signal_type=SignalType.DEFLECTION,
            score=score,
            indicators=tuple(indicators),
        )

    def _enthusiasm(self, text: str, ctx: DetectionContext) -> Signal:
        indicators: list[str] = []

        # Enthusiasm words
        enth_count = _match_count(text, _ENTHUSIASM_WORDS)
        wc = _word_count(text)
        enth_ratio = enth_count / wc
        if enth_count:
            indicators.append(f"enthusiasm_words({enth_count})")

        # Exclamation marks
        excl = len(re.findall(r"!", text))
        excl_score = min(1.0, excl / 3.0)
        if excl:
            indicators.append(f"exclamations({excl})")

        # Capitalised emphasis (ALL-CAPS words >2 chars)
        caps_words = len(re.findall(r"\b[A-Z]{3,}\b", text))
        caps_score = min(1.0, caps_words / 3.0)
        if caps_words:
            indicators.append(f"caps_words({caps_words})")

        score = _blend(
            min(1.0, enth_ratio * 15.0) * 0.50,
            excl_score * 0.30,
            caps_score * 0.20,
        )
        return Signal(
            signal_type=SignalType.ENTHUSIASM,
            score=score,
            indicators=tuple(indicators),
        )

    def _evasion(self, text: str, ctx: DetectionContext) -> Signal:
        matches = _pattern_matches(text, _EVASION_PATTERNS)
        indicators: list[str] = []

        if matches:
            indicators.append(f"evasion_phrases({len(matches)})")

        # Very short answers to potentially complex questions
        wc = _word_count(text)
        brevity_score = 0.0
        if wc < 8:
            brevity_score = min(1.0, (8 - wc) / 8.0)
            indicators.append(f"very_brief({wc}_words)")

        # Refusal to answer
        refusals = len(re.findall(r"\b(?:can't|cannot|won't|refuse|unable) (?:answer|discuss|comment|help|provide)\b", text, re.I))
        if refusals:
            indicators.append(f"refusals({refusals})")

        score = _blend(
            min(1.0, len(matches) / 2.0) * 0.45,
            brevity_score * 0.25,
            min(1.0, refusals / 2.0) * 0.30,
        )
        return Signal(
            signal_type=SignalType.EVASION,
            score=score,
            indicators=tuple(indicators),
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _blend(*weighted_scores: float) -> float:
    """Combine pre-weighted scores and normalise to [0, 1]."""
    total = sum(weighted_scores)
    return min(1.0, max(0.0, total))
