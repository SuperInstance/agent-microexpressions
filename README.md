# agent-microexpressions

**Detect and analyze subtle behavioral signals in AI agent interactions.**

Analogous to human micro-expressions — those fleeting facial cues that reveal
true feelings — this library detects behavioral "tells" in AI agent outputs:
hesitation, confidence, uncertainty, deflection, enthusiasm, and evasion.

Part of the [SuperInstance fleet](https://github.com/SuperInstance).

## Installation

```bash
pip install agent-microexpressions
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Detect signals in agent output

```python
from agent_microexpressions import MicroexpressionDetector

detector = MicroexpressionDetector()
signals = detector.detect("I will absolutely and definitely solve this clearly!")

for signal in signals:
    print(f"{signal.signal_type.value:15s} score={signal.score:.3f} strength={signal.strength.value}")
```

Output:
```
hesitation      score=0.000 strength=none
confidence      score=0.467 strength=strong
uncertainty     score=0.000 strength=none
deflection      score=0.000 strength=none
enthusiasm      score=0.000 strength=none
evasion         score=0.000 strength=none
```

### Encode signals as vectors

```python
from agent_microexpressions import MicroexpressionDetector, SignalEncoder

detector = MicroexpressionDetector()
encoder = SignalEncoder(extended=True)

signals = detector.detect("Perhaps maybe this could work. Good idea!")
vector = encoder.encode_signals(signals, text="Perhaps maybe this could work. Good idea!")

print(f"Dimensions: {vector.dim}")
print(f"Values: {[round(v, 3) for v in vector.values]}")
print(f"Labels: {vector.labels}")
```

### Track behavioral profiles across agents

```python
from agent_microexpressions import MicroexpressionDetector, SignalAggregator

detector = MicroexpressionDetector()
aggregator = SignalAggregator(smoothing=0.3)

# Ingest responses from different agents
aggregator.ingest("gpt-4", detector.detect("Certainly! Here's the solution."), text="Certainly! Here's the solution.")
aggregator.ingest("gpt-4", detector.detect("Maybe try a different approach?"), text="Maybe try a different approach?")
aggregator.ingest("claude", detector.detect("I can't really discuss that."), text="I can't really discuss that.")

for agent_id in aggregator.agents():
    profile = aggregator.get_profile(agent_id)
    print(profile.summary())
    print(f"  Anomalies: {profile.anomaly_flags}")
```

### Timeline analysis across conversation turns

```python
from agent_microexpressions import Timeline

timeline = Timeline(agent_id="assistant")

timeline.add_turn("I will definitely help you with that!")
timeline.add_turn("Hmm, maybe there's a different approach...")
timeline.add_turn("I'm not sure I can answer that.")
timeline.add_turn("Anyway, let's move on to something else.")

print(f"Total drift: {timeline.total_drift():.4f}")
print(f"Average drift: {timeline.average_drift():.4f}")
print(f"Volatility: {timeline.volatility():.4f}")

# Per-signal trajectory
trajectory = timeline.trajectory()
for signal_type, scores in trajectory.items():
    print(f"  {signal_type.value:15s}: {[round(s, 2) for s in scores]}")
```

## Architecture

```
agent_microexpressions/
├── __init__.py       # Public API
├── signals.py        # Signal types, strength bands, data structures
├── detector.py       # MicroexpressionDetector — text → signals
├── encoder.py        # SignalEncoder — signals → numerical vectors
├── aggregator.py     # SignalAggregator — signals → behavioral profiles
└── timeline.py       # Timeline — track signal evolution across turns
```

### Signal Types

| Signal | Description |
|--------|-------------|
| **Hesitation** | Detected via hedging words, high latency, ellipsis usage |
| **Confidence** | Certainty words, declarative sentences, first-person assertions |
| **Uncertainty** | Hedge words, questions, "I think/believe/guess" phrases |
| **Deflection** | Topic-shifting phrases, redirects, conversational pivots |
| **Enthusiasm** | Positive words, exclamation marks, ALL-CAPS emphasis |
| **Evasion** | Refusal phrases, brevity, "can't comment" patterns |

## API Reference

### `MicroexpressionDetector(latency_baseline_ms=1500.0, hedge_weight=1.0)`

Analyze text and optionally latency context for behavioral signals.

```python
signals = detector.detect(text, context=DetectionContext(latency_ms=3000))
```

### `SignalEncoder(extended=False)`

Encode signals into fixed-length numerical vectors. Basic: 6 dimensions (one per signal). Extended: 10 dimensions (+ sentiment, verbosity, formality, question density).

### `SignalAggregator(smoothing=0.3)`

Maintain exponential moving average profiles per agent. Detect anomalies like extreme enthusiasm, high evasion, or chronic low confidence.

### `Timeline(detector=None, encoder=None, agent_id=None)`

Track behavioral drift across conversation turns. Provides total drift, average drift, volatility, per-signal trajectories, and dominant signal series.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT © SuperInstance
