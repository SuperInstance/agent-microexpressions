# agent-microexpressions — Behavioral Signal Detection

**Detect and analyze subtle behavioral signals in AI agent interactions — hesitation, confidence, uncertainty, deflection, enthusiasm, evasion.**

## What This Gives You

- **Signal detection** — identify behavioral analogs: hesitation, confidence, uncertainty, deflection, enthusiasm, evasion
- **Signal encoding** — convert detected signals into numerical vectors for quantitative analysis
- **Behavioral profiles** — aggregate signals over time into per-agent behavioral profiles
- **Change tracking** — monitor how an agent's behavioral patterns shift over sessions
- **Pure Python** — no ML model required, rule-based detection

## Quick Start

```bash
pip install agent-microexpressions
```

```python
from agent_microexpressions import MicroexpressionDetector, SignalEncoder

# Detect signals in agent output
detector = MicroexpressionDetector()
signals = detector.detect("""
I think this should work, but I'm not entirely sure.
Let me try a different approach...
Actually, the first approach was correct.
""")

for signal in signals:
    print(f"{signal.type}: {signal.strength:.2f} — {signal.context}")
# HESITATION: 0.8 — "I think... but I'm not entirely sure"
# CONFIDENCE: 0.3 — "should work"
# DEFLECTION: 0.6 — "Let me try a different approach"

# Encode as vectors
encoder = SignalEncoder()
vector = encoder.encode(signals)
print(vector)  # [0.8, 0.3, 0.1, 0.6, 0.0, 0.0]

# Build behavioral profile over time
from agent_microexpressions import SignalAggregator
agg = SignalAggregator(agent_id="agent-3")
agg.add_session(signals)
profile = agg.profile()
print(f"Dominant signal: {profile.dominant}")
print(f"Stability: {profile.stability:.2f}")
```

## API Reference

### `MicroexpressionDetector` — `detect(text) → list[Signal]`
### `Signal(type, strength, context, position)` · `SignalType` — HESITATION, CONFIDENCE, UNCERTAINTY, DEFLECTION, ENTHUSIASM, EVASION
### `SignalStrength` — LOW, MODERATE, HIGH
### `SignalEncoder` — `encode(signals) → list[float]`
### `SignalAggregator(agent_id)` — `add_session(signals)`, `profile() → BehavioralProfile`

## How It Fits

The behavioral analysis layer for the [SuperInstance fleet](https://github.com/SuperInstance). Detects when agents are uncertain, evasive, or overconfident — before those signals become problems.

- **[agent-therapy](https://github.com/SuperInstance/agent-therapy)** — Health monitoring (uses microexpression data)
- **[cocapn-explain](https://github.com/SuperInstance/cocapn-explain)** — Explainability (behavioral context)
- **[agent-whisper](https://github.com/SuperInstance/agent-whisper)** — Adjusts whispers based on detected signals

## Testing

```bash
pytest tests/
```

## Installation

```bash
pip install agent-microexpressions
```

Python 3.10+. MIT license.
