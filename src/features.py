import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import librosa
import cv2
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from torchvision import models, transforms

# Text extractor using DistilBERT pooled output
class TextFeatureExtractor:
    def __init__(self, model_name="distilbert-base-uncased", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def extract(self, sentence, max_length=64):
        if not isinstance(sentence, str) or sentence.strip() == "":
            # return zeros
            return np.zeros(self.model.config.hidden_size, dtype=np.float32)
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
            # distilbert: last_hidden_state; use mean pooling
            if hasattr(out, "last_hidden_state"):
                vec = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            elif hasattr(out, "pooler_output"):
                vec = out.pooler_output.squeeze().cpu().numpy()
            else:
                vec = out[0].mean(dim=1).squeeze().cpu().numpy()
        return vec

# Audio extractor using librosa: returns log-mel spectrogram mean pooled
def extract_audio_feature(video_path, sr=16000, n_mels=64):
    # librosa can load mp4 via ffmpeg backend if available.
    try:
        y, _ = librosa.load(str(video_path), sr=sr, mono=True)
    except Exception:
        # try extracting audio via cv2 fallback (read frame-by-frame and ignore) -> return zeros
        return np.zeros(n_mels, dtype=np.float32)
    if y.size == 0:
        return np.zeros(n_mels, dtype=np.float32)
    # compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    # mean pooling over time -> shape (n_mels,)
    feat = S_db.mean(axis=1)
    # normalize
    if np.all(np.isfinite(feat)):
        feat = (feat - feat.mean()) / (feat.std() + 1e-9)
    else:
        feat = np.zeros(n_mels, dtype=np.float32)
    return feat.astype(np.float32)

# Visual extractor: sample central frame or end-time frame and pass through ResNet18
class VisualFeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        r = models.resnet18(pretrained=True)
        # remove final layer
        modules = list(r.children())[:-1]
        self.backbone = torch.nn.Sequential(*modules).to(device)
        self.backbone.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # DO NOT set color normalization values in case of differences; but typical normalization:
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def extract(self, video_path, time_sec=None):
        # open video and pick frame at time_sec (if provided) else middle frame
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return np.zeros(512, dtype=np.float32)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        selected_frame = None
        if time_sec is not None:
            frame_idx = int(min(frame_count - 1, max(0, time_sec * fps)))
        else:
            frame_idx = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return np.zeros(512, dtype=np.float32)
        # convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        inp = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.backbone(inp).squeeze().cpu().numpy()
        # feat shape (512,)
        return feat.astype(np.float32)

# Utilities to save/load features as numpy files
def save_feature(out_dir, key, arr, modality):
    out_dir = Path(out_dir)
    (out_dir / modality).mkdir(parents=True, exist_ok=True)
    out_path = out_dir / modality / f"{key}.npy"
    np.save(str(out_path), arr)

def load_feature(out_dir, key, modality):
    p = Path(out_dir) / modality / f"{key}.npy"
    if not p.exists():
        return None
    return np.load(str(p))