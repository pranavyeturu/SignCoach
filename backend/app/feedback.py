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
        detail = confusing_pair_feedback(target, predicted)
        if detail:
            return detail
        return f"Detected {predicted}. Target is {target}."
    if target and not stable:
        return "Correct. Hold for one second."
    if target:
        return "Nice. Sign confirmed."
    return f"Detected {predicted}."


def confusing_pair_feedback(target: str, predicted: str) -> str | None:
    pair = (target.upper(), predicted.upper())
    messages = {
        ("R", "U"): "Detected U. Target is R. Cross index and middle fingers more clearly.",
        ("U", "R"): "Detected R. Target is U. Keep index and middle fingers together, not crossed.",
        ("T", "A"): "Detected A. Target is T. Place thumb between index and middle fingers.",
        ("A", "T"): "Detected T. Target is A. Keep thumb along the side of the fist.",
    }
    return messages.get(pair)
