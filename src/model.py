import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.fc(x)

class FusionModel(nn.Module):
    def __init__(self, text_dim=768, audio_dim=64, visual_dim=512, hidden_dim=256, out_dim=2):
        super().__init__()
        self.text_enc = ModalityEncoder(text_dim, hidden_dim)
        self.audio_enc = ModalityEncoder(audio_dim, hidden_dim)
        self.visual_enc = ModalityEncoder(visual_dim, hidden_dim)

        # fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, text, audio, visual):
        t = self.text_enc(text)
        a = self.audio_enc(audio)
        v = self.visual_enc(visual)
        cat = torch.cat([t, a, v], dim=1)
        logits = self.fusion(cat)
        return logits
