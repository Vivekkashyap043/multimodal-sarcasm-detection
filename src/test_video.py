import os
import torch
import numpy as np
import argparse
import subprocess
from torch.nn.functional import softmax
from src.features import TextFeatureExtractor, extract_audio_feature, VisualFeatureExtractor
from src.model import FusionModel

# For audio extraction
from pydub import AudioSegment
from pydub.utils import which

# ✅ OpenAI Whisper for offline transcription
import whisper

# --- Explicitly set ffmpeg and ffprobe paths (Windows fix) ---
FFMPEG_PATH = r"D:\ffmpeg-8.0\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
FFPROBE_PATH = r"D:\ffmpeg-8.0\ffmpeg-8.0-essentials_build\bin\ffprobe.exe"

FFMPEG_PATH = os.path.normpath(FFMPEG_PATH)
FFPROBE_PATH = os.path.normpath(FFPROBE_PATH)

if not os.path.exists(FFMPEG_PATH):
    raise FileNotFoundError(f"❌ FFmpeg not found at {FFMPEG_PATH}")
if not os.path.exists(FFPROBE_PATH):
    raise FileNotFoundError(f"❌ FFprobe not found at {FFPROBE_PATH}")

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

print(f"🔧 Using ffmpeg: {AudioSegment.converter}")
print(f"🔧 Using ffprobe: {AudioSegment.ffprobe}")

try:
    subprocess.run([FFPROBE_PATH, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print("✅ FFprobe test succeeded.")
except Exception as e:
    print(f"⚠️ FFprobe test failed: {e}")

# ------------------------------------------------------------

# ✅ Load Whisper model globally (so it’s reused for multiple videos)
print("\n🎙️ Loading Whisper model (base)... This may take a moment.")
WHISPER_MODEL = whisper.load_model("base")
print("✅ Whisper model loaded successfully.\n")


def extract_audio_from_video(video_path, output_dir):
    """Extract audio from video using ffmpeg subprocess (saves in project folder)."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_out = os.path.join(output_dir, f"{base_name}.wav")

    print("🎧 Extracting audio from video using ffmpeg subprocess...")
    command = [
        FFMPEG_PATH, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_out
    ]
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"✅ Audio extracted successfully to: {audio_out}")
        return audio_out
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed: {e.stderr.decode()}")
        raise RuntimeError("Audio extraction failed. Check FFmpeg installation or video format.")


def transcribe_audio(audio_path):
    """Convert audio speech → text using OpenAI Whisper (offline)."""
    print("🧠 Transcribing speech using OpenAI Whisper (offline)...")
    try:
        # Use preloaded Whisper model
        result = WHISPER_MODEL.transcribe(audio_path, fp16=False)

        # Extract text and language
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")

        if text:
            print(f"\n🗣️ Detected Language: {language.upper()}")
            print(f"🗣️ Transcribed Text Detected in Video:\n   \"{text}\"")
        else:
            print("⚠️ Whisper could not detect speech or the clip was too short.")
            text = " "

        return text

    except Exception as e:
        print(f"❌ Whisper transcription failed: {e}")
        return " "


def extract_modal_features(video_path, sentence, device):
    """Extract text, audio, and visual features from custom video."""
    print("🎬 Extracting multimodal features from video...")

    # Text features
    text_extractor = TextFeatureExtractor(device=device)
    text_feat = text_extractor.extract(sentence)
    text_feat = text_feat.squeeze(0).cpu().numpy() if isinstance(text_feat, torch.Tensor) else np.array(text_feat)

    # Audio features
    audio_feat = extract_audio_feature(video_path)
    audio_feat = audio_feat.detach().cpu().numpy() if isinstance(audio_feat, torch.Tensor) else np.array(audio_feat)

    # Visual features
    visual_extractor = VisualFeatureExtractor(device=device)
    visual_feat = visual_extractor.extract(video_path)
    visual_feat = visual_feat.squeeze(0).cpu().numpy() if isinstance(visual_feat, torch.Tensor) else np.array(visual_feat)

    return text_feat, audio_feat, visual_feat


def predict_sarcasm(model_path, video_path, sentence=None, device="cpu", threshold=0.65):
    """Run end-to-end sarcasm detection on a custom video."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Video not found: {video_path}")

    # Extract & transcribe audio if needed
    if not sentence or sentence.strip() == "":
        print("🔊 Extracting audio and transcribing speech...")
        audio_dir = os.path.join(os.getcwd(), "test_audio")
        audio_path = extract_audio_from_video(video_path, audio_dir)
        sentence = transcribe_audio(audio_path)
        if not sentence:
            sentence = " "  # fallback

    # Load model
    print(f"\n📦 Loading trained model from {model_path}")
    model = FusionModel(text_dim=768, audio_dim=64, visual_dim=512, hidden_dim=256, out_dim=2)
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state" in checkpoint:
        print("📦 Loaded checkpoint contains optimizer and epoch info — extracting model_state.")
        checkpoint = checkpoint["model_state"]

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Extract multimodal features
    text_feat, audio_feat, visual_feat = extract_modal_features(video_path, sentence, device)

    # Convert to tensors
    text_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).to(device)
    audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).to(device)
    visual_tensor = torch.tensor(visual_feat, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(text_tensor, audio_tensor, visual_tensor)
        probs = softmax(logits, dim=1)
        sarcasm_prob = probs[0, 1].item()

    # ✅ Apply threshold (0.60 for your current model accuracy)
    label = "Sarcastic" if sarcasm_prob >= threshold else "Not Sarcastic"

    print(f"\n🧠 Prediction: {label}")
    print(f"   Sarcasm probability: {sarcasm_prob:.2f}  |  Threshold: {threshold:.2f}\n")

    return label, sarcasm_prob, sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test your own video for sarcasm detection")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--video_path", type=str, required=True, help="Path to your video (.mp4)")
    parser.add_argument("--sentence", type=str, default="", help="Optional sentence (if known)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--threshold", type=float, default=0.65, help="Probability threshold for sarcasm classification")
    args = parser.parse_args()

    predict_sarcasm(args.model_path, args.video_path, args.sentence, args.device, args.threshold)
