import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "outputs" / "features"
MODELS_DIR = ROOT / "outputs" / "models"

# Default names (you can override with CLI args)
METADATA_CSV = DATA_DIR / "metadata.csv"  # user said they'll use this as metadata.csv
CONTEXT_VIDEO_DIR = DATA_DIR / "context_videos"
UTTERANCE_VIDEO_DIR = DATA_DIR / "utterance_videos"

# feature subfolders
TEXT_FEAT_DIR = FEATURES_DIR / "text"
AUDIO_FEAT_DIR = FEATURES_DIR / "audio"
VISUAL_FEAT_DIR = FEATURES_DIR / "visual"

# Ensure folders exist at runtime
for p in [FEATURES_DIR, MODELS_DIR, TEXT_FEAT_DIR, AUDIO_FEAT_DIR, VISUAL_FEAT_DIR]:
    os.makedirs(p, exist_ok=True)
