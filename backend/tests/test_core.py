import unittest

import numpy as np

from app.features import normalize_landmarks
from app.scoring import attempt_score
from app.smoothing import PredictionSmoother


class CoreTests(unittest.TestCase):
    def test_landmark_normalization_is_translation_invariant(self):
        base = [{"x": i / 20, "y": i / 40, "z": i / 80} for i in range(21)]
        shifted = [{"x": p["x"] + 2, "y": p["y"] - 3, "z": p["z"] + 1} for p in base]
        self.assertTrue(np.allclose(normalize_landmarks(base), normalize_landmarks(shifted), atol=1e-6))

    def test_attempt_score(self):
        self.assertEqual(attempt_score(1, 1, 1), 100)
        self.assertEqual(attempt_score(0, 0, 0), 0)

    def test_smoother_rejects_low_confidence(self):
        result = PredictionSmoother().add("A", 0.4)
        self.assertIsNone(result["label"])

    def test_low_confidence_frame_does_not_erase_prediction_history(self):
        smoother = PredictionSmoother()
        smoother.add("A", 0.9)
        smoother.add("A", 0.9)
        smoother.add("A", 0.4)
        result = smoother.add("A", 0.9)
        self.assertEqual(result["label"], "A")


if __name__ == "__main__":
    unittest.main()
