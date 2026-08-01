import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from app.calibration import append_calibration_sample
from app.disambiguation import a_t_shape_cue
from app.features import normalize_landmarks
from app.feedback import confusing_pair_feedback
from app.landmarks import physical_handedness_from_mediapipe
from app.main import confidence_threshold_for
from app.phrases import analyze_phrase, phrase_catalog
from app.scoring import attempt_score
from app.smoothing import PredictionSmoother


class CoreTests(unittest.TestCase):
    def test_landmark_normalization_is_translation_invariant(self):
        base = [{"x": (i % 5) / 10, "y": (i // 5) / 10, "z": i / 100} for i in range(21)]
        shifted = [{"x": p["x"] + 2, "y": p["y"] - 3, "z": p["z"] + 1} for p in base]
        self.assertTrue(np.allclose(normalize_landmarks(base), normalize_landmarks(shifted), atol=1e-3))
        self.assertGreater(len(normalize_landmarks(base)), 63)

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

    def test_handedness_is_corrected_for_raw_camera_frame(self):
        self.assertEqual(physical_handedness_from_mediapipe("Left"), "Right")
        self.assertEqual(physical_handedness_from_mediapipe("Right"), "Left")
        self.assertIsNone(physical_handedness_from_mediapipe(None))

    def test_calibration_sample_is_saved_by_letter(self):
        landmarks = [{"x": i / 20, "y": i / 30, "z": i / 40} for i in range(21)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.csv"
            with patch("app.calibration.CALIBRATION_LANDMARKS_PATH", path):
                summary = append_calibration_sample("a", landmarks)
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["byLetter"], {"A": 1})

    def test_confusing_pairs_have_specific_feedback(self):
        self.assertIn("Cross index", confusing_pair_feedback("R", "U"))
        self.assertIn("thumb between", confusing_pair_feedback("T", "A"))

    def test_confusing_letters_require_higher_confirmation_confidence(self):
        self.assertEqual(confidence_threshold_for("R", "R"), 0.65)
        self.assertEqual(confidence_threshold_for("B", "B"), 0.55)

    def test_a_t_shape_cue_detects_thumb_geometry(self):
        landmarks = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)]
        for index, (x, y) in {
            0: (0.0, 0.0),
            4: (0.55, 0.1),
            5: (0.05, 0.05),
            8: (0.0, 0.6),
            9: (0.12, 0.05),
            12: (0.12, 0.6),
            20: (0.2, 0.2),
        }.items():
            landmarks[index] = {"x": x, "y": y, "z": 0.0}
        self.assertEqual(a_t_shape_cue(landmarks).label, "T")

    def test_phrase_catalog_includes_starter_signs(self):
        ids = {phrase["id"] for phrase in phrase_catalog()}
        self.assertIn("yes", ids)
        self.assertIn("no", ids)

    def test_yes_phrase_detects_nodding_motion(self):
        frames = []
        ys = [0.2, 0.25, 0.31, 0.24, 0.2, 0.26, 0.32, 0.24, 0.2]
        for y in ys:
            frame = [{"x": 0.5, "y": y, "z": 0.0} for _ in range(21)]
            for index in (4, 8, 12, 16, 20):
                frame[index] = {"x": 0.5 + index * 0.001, "y": y + index * 0.001, "z": 0.0}
            frames.append(frame)
        result = analyze_phrase("yes", frames)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
