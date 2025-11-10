import os
from pathlib import Path
import pandas as pd

def read_metadata(csv_path):
    df = pd.read_csv(csv_path)
    # Guarantee columns expected exist
    # Some CSVs might have trailing spaces in headers; normalize
    df.columns = [c.strip() for c in df.columns]
    return df

def key_to_scene(key):
    # KEY examples:
    # - context segment: 1_10004_c_00
    # - utterance: 1_10004_u
    # We want the scene id: 1_10004
    if isinstance(key, float) and pd.isna(key):
        return None
    key = str(key)
    if "_c_" in key:
        return key.split("_c_")[0]
    if key.endswith("_u"):
        return key[:-2]
    # fallback: take first two underscore chunks
    parts = key.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return key

def key_is_utterance(key):
    return str(key).endswith("_u")

def scene_to_context_filename(scene):
    return f"{scene}_c.mp4"

def scene_to_utterance_filename(scene):
    return f"{scene}_u.mp4"

def parse_timestamp_to_seconds(ts):
    """
    Parse time strings like '0:06' or '00:01:23' to seconds (float).
    """
    if pd.isna(ts := ts):
        return None
    s = str(ts).strip()
    # some values may already be seconds numeric
    try:
        return float(s)
    except Exception:
        pass

    parts = s.split(":")
    try:
        parts = [float(p) for p in parts]
    except:
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None
