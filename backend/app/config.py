from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "archive" / "asl_alphabet_train" / "asl_alphabet_train"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MAX_FRAME_WIDTH = 960
MAX_FRAME_HEIGHT = 720

