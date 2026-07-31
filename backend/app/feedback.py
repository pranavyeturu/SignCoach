from __future__ import annotations


def feedback_for(
    *,
    hand_detected: bool,
    framing_score: float,
    predicted: str | None,
    confidence: float,
    target: str | None,
    stable: bool,
) -> str:
    if not hand_detected:
        return "Move your hand into frame."
    if framing_score < 0.45:
        return "Move closer to the camera."
    if not predicted or confidence < 0.55:
        return "Hold steady and face your palm toward the camera."
    if target and predicted != target:
        return f"Detected {predicted}. Target is {target}."
    if target and not stable:
        return "Correct. Hold for one second."
    if target:
        return "Nice. Sign confirmed."
    return f"Detected {predicted}."
