from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "archive" / "asl_alphabet_train" / "asl_alphabet_train"
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "signcoach_model.joblib"
PHRASE_MODEL_PATH = MODEL_DIR / "phrase_sign_model.joblib"
PROCESSED_LANDMARKS_PATH = DATA_DIR / "processed" / "landmarks.csv"
PHRASE_FEATURES_PATH = DATA_DIR / "processed" / "msasl_phrase_features.csv"
CALIBRATION_LANDMARKS_PATH = DATA_DIR / "calibration" / "landmarks.csv"
METRICS_DIR = DATA_DIR / "metrics"
MSASL_DIR = ROOT_DIR / "MS-ASL"
MSASL_MANIFEST_PATH = DATA_DIR / "msasl" / "phrase_manifest.json"
MSASL_VIDEO_DIR = DATA_DIR / "msasl" / "videos"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MAX_FRAME_WIDTH = 960
MAX_FRAME_HEIGHT = 720
