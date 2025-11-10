import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import key_to_scene, key_is_utterance, scene_to_context_filename, scene_to_utterance_filename, parse_timestamp_to_seconds

class MustardDataset(Dataset):
    """
    Dataset that returns multimodal features for each utterance KEY.
    Expects features saved under features_dir/<modality>/<KEY>.npy
    Modalities: text, audio, visual
    """
    def __init__(self, metadata_csv, features_dir, context_video_dir, utterance_video_dir, split=None, keys=None):
        self.df = pd.read_csv(metadata_csv)
        self.df.columns = [c.strip() for c in self.df.columns]
        # Filter only rows that are utterances (KEY endswith _u)
        self.df = self.df[self.df['KEY'].astype(str).str.endswith("_u")]
        self.df = self.df.reset_index(drop=True)

        if keys is not None:
            self.df = self.df[self.df['KEY'].isin(keys)].reset_index(drop=True)

        self.features_dir = Path(features_dir)
        self.context_video_dir = Path(context_video_dir)
        self.utterance_video_dir = Path(utterance_video_dir)

        # prepare labels
        # Ensure Sarcasm column exists (may be named different). Accept 'Sarcasm' or 'sarcasm'
        if 'Sarcasm' in self.df.columns:
            self.df['label'] = self.df['Sarcasm'].fillna(0).astype(int)
        elif 'sarcasm' in self.df.columns:
            self.df['label'] = self.df['sarcasm'].fillna(0).astype(int)
        else:
            # fallback: create zeros
            self.df['label'] = 0

    def __len__(self):
        return len(self.df)

    def _load_feat(self, modality, key):
        p = self.features_dir / modality / f"{key}.npy"
        if p.exists():
            return np.load(str(p))
        # fallback: check scene-level names (scene id)
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = row['KEY']
        label = int(row['label'])
        # try to load features by KEY (utterance-level)
        text = self._load_feat('text', key)
        audio = self._load_feat('audio', key)
        visual = self._load_feat('visual', key)

        # if any modality missing, try scene-level fallback by scene id
        if text is None:
            scene = key[:-2] if str(key).endswith("_u") else key
            text = self._load_feat('text', scene)
        if audio is None:
            scene = key[:-2] if str(key).endswith("_u") else key
            audio = self._load_feat('audio', scene)
        if visual is None:
            scene = key[:-2] if str(key).endswith("_u") else key
            visual = self._load_feat('visual', scene)

        # final fallback: zeros
        if text is None:
            text = np.zeros(768, dtype=np.float32)
        if audio is None:
            audio = np.zeros(64, dtype=np.float32)
        if visual is None:
            visual = np.zeros(512, dtype=np.float32)

        sample = {
            'key': key,
            'text': torch.from_numpy(text).float(),
            'audio': torch.from_numpy(audio).float(),
            'visual': torch.from_numpy(visual).float(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        return sample
