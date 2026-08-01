from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from app.sequence_features import sequence_features


@dataclass(frozen=True)
class PhrasePrediction:
    label: str
    confidence: float
    top_predictions: list[dict[str, float | str]]


class PhraseSignPredictor:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model = None
        self.error: str | None = None
        self.reload()

    def reload(self) -> None:
        self.model = None
        self.error = None
        if not self.model_path.exists():
            self.error = "No trained MS-ASL phrase model found."
            return
        try:
            self.model = joblib.load(self.model_path)
            if not hasattr(self.model, "predict_proba"):
                raise ValueError("Model must support predict_proba().")
        except Exception as exc:
            self.error = f"Could not load phrase model: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, frames: list[list[dict[str, float]]]) -> PhrasePrediction | None:
        if self.model is None:
            return None
        features = sequence_features(frames).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        classes = np.asarray(self.model.classes_)
        order = np.argsort(probabilities)[::-1][:3]
        return PhrasePrediction(
            label=str(classes[order[0]]),
            confidence=float(probabilities[order[0]]),
            top_predictions=[
                {"label": str(classes[index]), "confidence": round(float(probabilities[index]), 4)}
                for index in order
            ],
        )
