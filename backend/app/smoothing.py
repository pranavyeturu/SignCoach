from __future__ import annotations

from collections import Counter, deque
from time import monotonic


class PredictionSmoother:
    def __init__(self, window_size: int = 10, threshold: float = 0.55, hold_seconds: float = 1.0) -> None:
        self.predictions: deque[tuple[str, float]] = deque(maxlen=window_size)
        self.threshold = threshold
        self.hold_seconds = hold_seconds
        self._candidate: str | None = None
        self._candidate_since = monotonic()

    def reset(self) -> None:
        self.predictions.clear()
        self._candidate = None
        self._candidate_since = monotonic()

    def add(self, label: str, confidence: float) -> dict[str, object]:
        if confidence < self.threshold:
            # A single blurry frame should not erase an otherwise steady hold.
            # No-hand frames still call reset() from the request pipeline.
            return {"label": None, "confidence": confidence, "stable": False, "holdProgress": 0.0}

        self.predictions.append((label, confidence))
        counts = Counter(item[0] for item in self.predictions)
        winner, votes = counts.most_common(1)[0]
        majority = votes >= max(2, (len(self.predictions) // 2) + 1)

        if not majority:
            self._candidate = None
            return {"label": None, "confidence": confidence, "stable": False, "holdProgress": 0.0}

        if winner != self._candidate:
            self._candidate = winner
            self._candidate_since = monotonic()

        hold_progress = min((monotonic() - self._candidate_since) / self.hold_seconds, 1.0)
        winner_confidences = [score for item, score in self.predictions if item == winner]
        return {
            "label": winner,
            "confidence": round(sum(winner_confidences) / len(winner_confidences), 4),
            "stable": hold_progress >= 1.0,
            "holdProgress": round(hold_progress, 3),
        }
