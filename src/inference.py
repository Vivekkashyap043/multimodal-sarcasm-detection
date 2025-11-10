import torch
import numpy as np
import argparse
import os
from src.features import TextFeatureExtractor, extract_audio_feature, VisualFeatureExtractor
from src.model import FusionModel
from torch.nn.functional import softmax

def load_feature_or_extract(video_path, sentence, device):
    """
    Extract text, audio, and visual features from a given video + sentence.
    """
    # Text
    print("Extracting text features...")
    text_extractor = TextFeatureExtractor(device=device)
    text_feat = text_extractor.extract(sentence)
    text_feat = text_feat.squeeze(0).cpu().numpy()

    # Audio
    print("Extracting audio features...")
    audio_feat = extract_audio_feature(video_path)

    # Visual
    print("Extracting visual features...")
    visual_extractor = VisualFeatureExtractor(device=device)
    visual_feat = visual_extractor.extract(video_path)
    visual_feat = visual_feat.squeeze(0).cpu().numpy()

    return text_feat, audio_feat, visual_feat


def predict(model_path, video_path, sentence, device="cpu"):
    """
    Load the model, extract features, and predict sarcasm.
    """
    # Load trained model
    print(f"Loading model from {model_path} ...")
    model = FusionModel(text_dim=768, audio_dim=64, visual_dim=512, hidden_dim=256, out_dim=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Extract features
    text_feat, audio_feat, visual_feat = load_feature_or_extract(video_path, sentence, device)

    # Convert to tensors
    text_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).to(device)
    audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).to(device)
    visual_tensor = torch.tensor(visual_feat, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(text_tensor, audio_tensor, visual_tensor)
        probs = softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_conf = probs[0, pred_class].item()

    label = "Sarcastic" if pred_class == 1 else "Not Sarcastic"
    print(f"\n🧠 Prediction: {label} (confidence: {pred_conf:.2f})")
    return label, pred_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sarcasm Detection Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--video_path", type=str, required=True, help="Path to utterance video file (.mp4)")
    parser.add_argument("--sentence", type=str, required=True, help="Text transcript of the utterance")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    predict(args.model_path, args.video_path, args.sentence, args.device)
