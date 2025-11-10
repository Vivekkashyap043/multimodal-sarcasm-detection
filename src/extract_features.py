"""
Feature extraction script.

Saves features to out_dir/<modality>/<KEY>.npy

- text: uses 'SENTENCE' column, key used is KEY (expected _u for utterance, or scene id)
- audio: extracts from utterance video file if available, otherwise scene context video
- visual: extracts single-frame features from video file (utterance first, fallback to context)

Run:
python src/extract_features.py --data_dir data --out_dir outputs/features --device cpu
"""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd

from .features import TextFeatureExtractor, extract_audio_feature, VisualFeatureExtractor, save_feature

def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    metadata = pd.read_csv(data_dir / args.metadata)
    metadata.columns = [c.strip() for c in metadata.columns]

    # Initialize extractors
    text_extractor = TextFeatureExtractor(device=args.device)
    visual_extractor = VisualFeatureExtractor(device=args.device)

    # iterate rows and extract features per KEY
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        key = str(row['KEY'])
        scene = key
        # for _c_XX rows, scene is before _c_
        if "_c_" in key:
            scene = key.split("_c_")[0]
        if key.endswith("_u"):
            scene = key[:-2]

        # TEXT: prefer utterance sentence when KEY is utterance; otherwise the same sentence
        if 'SENTENCE' in row and isinstance(row['SENTENCE'], str):
            txt = str(row['SENTENCE'])
        else:
            txt = ""

        # Save text features keyed by KEY (so text features for _u rows are saved at KEY)
        txt_vec = text_extractor.extract(txt)
        save_feature(out_dir, key, txt_vec, 'text')

        # AUDIO: try utterance-level video then scene context video
        # map KEY -> video paths
        utt_vid = data_dir / "utterance_videos" / f"{scene}_u.mp4"
        ctx_vid = data_dir / "context_videos" / f"{scene}_c.mp4"

        if utt_vid.exists():
            audio_vec = extract_audio_feature(utt_vid)
            save_feature(out_dir, key, audio_vec, 'audio')
        elif ctx_vid.exists():
            audio_vec = extract_audio_feature(ctx_vid)
            save_feature(out_dir, key, audio_vec, 'audio')
        else:
            # zero vector
            save_feature(out_dir, key, np.zeros(64, dtype=np.float32), 'audio')

        # VISUAL: use end time if available to sample frame from context; else use middle frame
        end_time_sec = None
        if 'END_TIME' in metadata.columns:
            val = row.get('END_TIME', None)
            try:
                # handle formats like 0:06
                if isinstance(val, str) and ":" in val:
                    parts = [float(x) for x in val.split(":")]
                    if len(parts) == 2:
                        end_time_sec = parts[0] * 60 + parts[1]
                    elif len(parts) == 3:
                        end_time_sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
                else:
                    end_time_sec = float(val)
            except Exception:
                end_time_sec = None

        # prefer utterance video for visual features
        if utt_vid.exists():
            vis_vec = visual_extractor.extract(utt_vid, time_sec=end_time_sec)
            save_feature(out_dir, key, vis_vec, 'visual')
        elif ctx_vid.exists():
            vis_vec = visual_extractor.extract(ctx_vid, time_sec=end_time_sec)
            save_feature(out_dir, key, vis_vec, 'visual')
        else:
            save_feature(out_dir, key, np.zeros(512, dtype=np.float32), 'visual')

    print("Feature extraction finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs/features")
    parser.add_argument("--metadata", type=str, default="metadata.csv")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
