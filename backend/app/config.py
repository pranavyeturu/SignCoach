from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "archive" / "asl_alphabet_train" / "asl_alphabet_train"
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "signcoach_model.joblib"
PROCESSED_LANDMARKS_PATH = DATA_DIR / "processed" / "landmarks.csv"
CALIBRATION_LANDMARKS_PATH = DATA_DIR / "calibration" / "landmarks.csv"
METRICS_DIR = DATA_DIR / "metrics"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MAX_FRAME_WIDTH = 960
MAX_FRAME_HEIGHT = 720
