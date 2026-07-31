from __future__ import annotations


def attempt_score(confidence: float, stability: float, framing: float) -> int:
    weighted = (confidence * 0.70) + (stability * 0.15) + (framing * 0.15)
    return round(max(0.0, min(weighted, 1.0)) * 100)

