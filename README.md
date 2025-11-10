# Multimodal Sarcasm Detection — Documentation

Comprehensive documentation for the multimodal sarcasm detection scaffold in this repository.

This project is a compact, runnable scaffold that extracts per-utterance multimodal features (text, audio, visual), trains a simple fusion model, and supports evaluation and inference. It follows a MUStARD++-style layout but is purposely minimal and easy to extend.

## Highlights

- Multimodal feature extraction:
  - Text: DistilBERT pooled representation (768d)
  - Audio: librosa mel-spectrogram mean pooling (64d)
  - Visual: ResNet18 single-frame pooled features (512d)
- Simple per-modality encoders + concatenation fusion model
- Scripts for feature extraction, training, evaluation and testing custom videos
- Files and outputs use a straightforward, reproducible layout (npy features, model checkpoints)

## Table of contents

- Project layout
- Dependencies (what and why)
- Data / metadata format
- Feature extraction (how each modality is handled)
- Common feature format & missing-data handling
- Dataset and dataloader behavior
- Model architecture and training loop
- Evaluation and inference
- How to run (quick commands)
- Troubleshooting, tips, and next steps

## Project layout

Top-level (relevant files/folders):

- `data/` — input metadata and video folders
  - `metadata.csv` — dataset metadata (KEY, SENTENCE, optionally START_TIME/END_TIME, Sarcasm label, etc.)
  - `context_videos/` — scene-level videos (expected: `<scene>_c.mp4`)
  - `utterance_videos/` — utterance-level videos (expected: `<scene>_u.mp4`)
- `outputs/`
  - `features/` — extracted features saved as `npy` files per modality: `features/<modality>/<KEY>.npy`
    - `text/`, `audio/`, `visual/`
  - `models/` — saved model checkpoints (e.g. `best_model.pth`, `epoch1.pth`)
  - `predictions.csv` — evaluation output (written by `src/evaluate.py`)
- `src/` — Python package containing the code
  - `config.py` — (minimal) constants and paths used across code
  - `dataset.py` — `MustardDataset` that loads npy features and labels
  - `features.py` — feature extractor utilities (text, audio, visual) and helpers to save/load
  - `extract_features.py` — CLI script to run full dataset feature extraction
  - `model.py` — `ModalityEncoder` and `FusionModel` implementation
  - `train.py` — training script (split, train, validate, save best checkpoints)
  - `evaluate.py` — evaluation script that writes `predictions.csv` and prints metrics
  - `inference.py` — a small inference helper that loads model and predicts given features
  - `test_video.py` — end-to-end local video testing: audio extraction, Whisper transcription (optional), feature extraction, model inference
  - `utils.py` — helper functions for key/scene parsing and small utilities
- `requirements.txt` — Python dependencies used by the project
- `steps.txt` — short runnable sequence describing the typical workflow

## Dependencies (what they do)

This project uses the following major libraries. The list in `requirements.txt` is authoritative for version pins; highlights here explain usage:

- PyTorch (`torch`, `torchvision`, `torchaudio`)
  - Model building and training. `torchvision` provides ResNet18 used for visual features.
- Transformers (`transformers`, `tokenizers`)
  - DistilBERT tokenizer & model to extract text embeddings (pooled/mean pooled from last hidden state).
- NumPy / pandas / scikit-learn
  - Data handling, feature arrays, and metrics (F1, accuracy, classification report).
- OpenCV (`opencv-python`) and Pillow
  - Video frame reading and image processing for visual feature extraction.
- librosa, soundfile
  - Audio loading and mel-spectrogram extraction for audio features.
- pydub + ffmpeg
  - Audio extraction from video in `test_video.py` (Windows-specific ffmpeg path handling is included).
- openai-whisper
  - Optional, offline transcription for custom video testing (`test_video.py`).
- tqdm
  - Progress bars used in extraction and training loops.

Why these choices?

- DistilBERT: small but effective sentence-level text embedding with minimal setup.
- ResNet18: widely-available image backbone; we remove the final classification layer and use the pooled 512-dim representation.
- librosa mels + mean pooling: compact, robust audio representation for short utterances.

## Data / metadata format

- `metadata.csv` is expected to have at least the following columns (case-sensitive after normalization in code):
  - `KEY` — unique identifier per row. Examples: `1_10004_c_00` (context segment), `1_10004_u` (utterance)
  - `SENTENCE` — text transcript (optional; extract_features prefers this for text features)
  - `Sarcasm` or `sarcasm` (optional) — label used for training/evaluation. If absent, labels default to `0`.
  - `END_TIME` (optional) — used by `extract_features.py` to sample a frame at a time offset when available; supports formats like `0:06` or `00:01:23`.

Naming conventions for videos (used by extraction scripts):

- Context (scene-level) video: `<scene>_c.mp4` (example: `1_10004_c.mp4`) in `data/context_videos/`
- Utterance-level video: `<scene>_u.mp4` (example: `1_10004_u.mp4`) in `data/utterance_videos/`

Keys ending with `_u` are considered utterance-level examples for training and evaluation. The dataset loader filters the metadata to only include these utterance rows by default.

## Feature extraction — modalities & implementation details

All extracted features are saved as numpy arrays at `outputs/features/<modality>/<KEY>.npy` by `src/extract_features.py`.

1) Text
  - Implemented in `src/features.py` as `TextFeatureExtractor`.
  - Uses `transformers.AutoTokenizer` and `AutoModel` (DistilBERT by default).
  - Tokenizes `SENTENCE` to a fixed max length (default 64 tokens), runs the model and mean-pools last hidden state to a 768-dim vector.
  - If the sentence is empty or missing, returns a zeros vector of length 768.

2) Audio
  - Function `extract_audio_feature` (in `src/features.py`) attempts to use `librosa.load` on the video file (ffmpeg-backed reader) at 16 kHz.
  - Computes mel-spectrogram (n_mels=64), converts to dB, and mean-pools across time to produce a 64-dim vector.
  - The vector is normalized (zero mean, unit-ish variance) and returned as float32.
  - If audio extraction fails (missing video or load error), a 64-dim zero vector is returned.

3) Visual
  - `VisualFeatureExtractor` uses `torchvision.models.resnet18(pretrained=True)` with the final FC removed.
  - The extractor samples one frame (preferred: frame at `END_TIME` in metadata; fallback: middle frame) using OpenCV, resizes to 224x224, applies ImageNet normalization, and extracts a 512-dim pooled feature.
  - If video reading fails, returns a zeros vector of length 512.

Notes on sampling strategy

- `extract_features.py` prefers utterance-level video (`<scene>_u.mp4`) for audio and visual extraction. If unavailable, it falls back to the context/scene video (`<scene>_c.mp4`).
- Text features are saved keyed by the metadata `KEY` (utterance keys will produce utterance-level text features).

## Common feature format & missing-data handling

- Feature files written as numpy `.npy` arrays. Location convention: `outputs/features/<modality>/<KEY>.npy`.
- Default dimensions used in code:
  - text: 768
  - audio: 64
  - visual: 512
- `MustardDataset` (`src/dataset.py`) attempts to load features for the utterance `KEY`. If a modality file for the exact `KEY` is missing it will attempt to load the scene-level features (strip `_u`), and if still missing it substitutes a zeros vector of the appropriate dimensionality.

This makes the model resilient to missing modalities and inconsistent dataset layouts.

## Dataset and dataloader behavior

- `MustardDataset` reads `metadata.csv`, filters rows with keys that end with `_u` (utterances), and constructs samples with the following fields:
  - `key`: the KEY string
  - `text`: torch.FloatTensor (shape [768])
  - `audio`: torch.FloatTensor (shape [64])
  - `visual`: torch.FloatTensor (shape [512])
  - `label`: torch.LongTensor containing 0/1 sarcasm label (defaults to 0 if no label column present)
- The `collate_fn` used by `train.py` / `evaluate.py` stacks modality tensors into batch tensors and returns labels and keys.

Edge behaviors:

- If `Sarcasm` column is absent, dataset assigns label `0` to all rows (useful for inference-only runs).
- The dataset and training split logic is simple: `sklearn.model_selection.train_test_split` stratifies by `Sarcasm` if present.

## Model architecture

Location: `src/model.py`

Overview:

- Each modality has a small encoder (`ModalityEncoder`) — a linear layer to `hidden_dim`, ReLU, and dropout.
- Encoded modality vectors are concatenated and passed through a fusion MLP: Linear -> ReLU -> Dropout -> Linear(out_dim).

Default sizes (configurable via CLI args in `train.py`):

- text_dim = 768
- audio_dim = 64
- visual_dim = 512
- hidden_dim = 256
- out_dim = 2 (binary classification: sarcastic / not sarcastic)

Forward pass summary:

1. text -> text encoder -> t (hidden_dim)
2. audio -> audio encoder -> a (hidden_dim)
3. visual -> visual encoder -> v (hidden_dim)
4. cat = concat([t,a,v]) -> fusion MLP -> logits

Loss: CrossEntropyLoss; optimizer: Adam (default lr 2e-4).

## Training

Script: `src/train.py`

Key behavior:

- Reads `metadata.csv` and builds a list of `KEY`s that end with `_u`.
- Performs a single randomized train/validation split (default test_size=0.2) stratified on `Sarcasm` if present.
- Creates `DataLoader` instances and trains for `--epochs` saving `epoch{n}.pth` each epoch and `best_model.pth` when validation F1 improves.
- Training metrics logged per epoch: training loss, F1 (weighted), and accuracy; same for validation.

CLI highlights (defaults shown in `train.py`):

- --data_dir (default: data)
- --features_dir (default: outputs/features)
- --metadata (default: metadata.csv)
- --output_dir (default: outputs/models)
- --epochs (default: 6)
- --batch_size (default: 8)
- --lr (default: 2e-4)
- --device (default: cpu)

Quality note: The training loop is intentionally simple and suitable for experimentation on small datasets. For larger runs:

- Add learning-rate scheduling, checkpointing with best validation epoch, gradient clipping, mixed precision, and more robust logging.

## Evaluation

Script: `src/evaluate.py`

Behavior:

- Loads the saved checkpoint (`--model_path`). The checkpoint can be either a raw `model_state` dict or the full checkpoint with `model_state` / `optimizer_state` / `epoch`. `evaluate.py` expects the `model_state` content and will load it into the `FusionModel`.
- Runs the model on the dataset (utterance `_u` rows) and prints accuracy, weighted F1, and a scikit-learn classification report.
- Writes a `predictions.csv` with columns: `KEY`, `pred`, `true` to `--output_dir` (default: `outputs/`).

## Inference & testing custom videos

Two helper scripts exist:

1. `src/inference.py`
   - Small helper: expects precomputed features or a given video+sentence and performs feature extraction (calls functions from `src/features.py`) then loads a model and predicts.
   - Simpler than `test_video.py`, useful for ad-hoc runs where you already have features.

2. `src/test_video.py`
   - Full end-to-end helper for local custom videos.
   - Steps performed:
     - Extract audio from the provided video using ffmpeg (Windows: the script includes explicit ffmpeg path handling — update the paths in the script to your local ffmpeg install or adjust `AudioSegment.converter`/`AudioSegment.ffprobe`).
     - Optionally transcribe audio to text using OpenAI Whisper (offline). Whisper model is loaded once at the module level; this requires network to download the model once and disk to cache it.
     - Extract the three modality features (Text via DistilBERT, audio via librosa, visual via ResNet18 single-frame) and feed them to the model.
     - Output: label, sarcasm probability, and transcribed sentence.

Important: `test_video.py` contains platform-specific ffmpeg paths (the repo example uses `D:\\ffmpeg-8.0\\...`). Update those two constants to the location of your ffmpeg binaries on Windows or remove the explicit declaration to rely on system PATH.

## How to run (quick commands)

Below are the recommended steps (see `steps.txt` for a condensed sequence). Replace `cpu` with `cuda` if you have a GPU and corresponding PyTorch build.

1) Create and activate virtual environment and install requirements

PowerShell example:

```powershell
# create venv (if you want)
python -m venv venv
venv\\Scripts\\Activate.ps1
# install CPU torch per project notes (if needed) then install remaining requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

2) Extract features for the dataset

```powershell
python -m src.extract_features --data_dir data --out_dir outputs/features --metadata metadata.csv --device cpu
```

This writes `outputs/features/text/<KEY>.npy`, `outputs/features/audio/<KEY>.npy`, and `outputs/features/visual/<KEY>.npy`.

3) Train the model

```powershell
python -m src.train --data_dir data --features_dir outputs/features --metadata metadata.csv --output_dir outputs/models --epochs 5 --batch_size 8 --device cpu
```

4) Evaluate

```powershell
python -m src.evaluate --data_dir data --features_dir outputs/features --metadata metadata.csv --model_path outputs/models/best_model.pth --output_dir outputs --device cpu
```

5) Test a custom video (end-to-end)

```powershell
python -m src.test_video --model_path outputs/models/best_model.pth --video_path D:\\projects\\sarcasm\\multimodal-sarcasm-detection\\videos\\non-sar1.mp4 --device cpu
```

Notes about GPU vs CPU

- Default CLI flags use `--device cpu`. If you have CUDA-enabled PyTorch installed, pass `--device cuda` to use GPU for the ResNet / DistilBERT feature extraction and training.

## Checkpoints and formats

- Training saves per-epoch checkpoints `epoch{n}.pth` and `best_model.pth`. Each saved file is a dictionary containing at least `model_state`. The training script saves both `model_state` and `optimizer_state`.
- `evaluate.py` expects a checkpoint with `model_state` (it loads the contained dict into `model.load_state_dict`). If a raw state dict is supplied, adjust accordingly (the scripts attempt to handle both formats in places).

## Edge cases & how they're handled

- Missing modality files: dataset falls back to scene-level feature (strip `_u`) and ultimately to a zero vector of the modality's expected dim.
- Missing transcript text: text extractor returns zeros for empty or missing sentences.
- Nonexistent videos for audio/visual: the extractors return zero vectors (safe fallback for training and inference).

## Troubleshooting

- FFmpeg not found (common on Windows): update ffmpeg path constants in `src/test_video.py` or ensure `ffmpeg` and `ffprobe` are on your PATH. The `test_video.py` script includes explicit checks and will raise helpful errors.
- Whisper model download: the first run of Whisper will download the selected model (e.g., `base`) and may take time; ensure network access and disk space.
- Out-of-memory on GPU: try running extraction and training with `--device cpu` or reduce `batch_size`.

## Suggested improvements / next steps

- Feature improvements:
  - Instead of single-frame visual features, extract short clip features (I3D, temporal pooling, or frame-level sequences + transformer).
  - Use a pretrained speech embedding model (e.g., Wav2Vec 2.0) instead of mel spectrogram mean pooling.
  - Improve text encoder by using better pooling (CLS + attention), or a larger BERT variant.

- Model / training improvements:
  - Add learning-rate scheduling, early stopping, and gradient clipping.
  - Add better logging (TensorBoard / Weights & Biases).
  - Replace naive fusion MLP with cross-modal attention or gating mechanisms.

- Dataset / reproducibility:
  - Add deterministic data loaders and seeds across numpy/torch/cudnn.
  - Add unit tests for feature extraction and dataset loading.

## Small contract / expected input-output (short)

- Inputs: `data/metadata.csv`, videos in `data/context_videos/` and `data/utterance_videos/`.
- Outputs: `outputs/features/<modality>/<KEY>.npy`, trained checkpoints in `outputs/models/`, evaluation `outputs/predictions.csv`.

## Files of interest (quick pointer)

- `src/extract_features.py` — feature extraction entry point
- `src/features.py` — code for text/audio/visual feature extraction and save/load
- `src/dataset.py` — `MustardDataset` and how features are assembled into samples
- `src/model.py` — `FusionModel` and `ModalityEncoder`
- `src/train.py` — training loop and CLI
- `src/evaluate.py` — evaluation and predictions CSV
- `src/test_video.py` — full local video testing flow (ffmpeg + Whisper + inference)

## Completion / verification

This README was generated after inspecting the repository source files (feature extractors, dataset loader, model, training and evaluation scripts). If you'd like, I can:

- Add a short example `notebook/` showing extraction → training → evaluation on a small subset.
- Add unit tests that run a tiny end-to-end smoke test using artificial data and random features.
- Add automatic validation that `outputs/features` contains expected modalities/dims before training.

If you want any of those, tell me which one to implement and I'll add it in the repo.

---

(End of README)
# Multimodal Sarcasm Detection (MUStARD_Plus_Plus-based scaffold)

This repository is a runnable scaffold to train a simple multimodal sarcasm detection model based on the MUStARD++ dataset layout.  
It expects the dataset layout as described in the project:

data/
├─ mustard++_text.csv
├─ final_context_videos/
│ └─ 1_10004_c.mp4
└─ final_utterance_videos/
└─ 1_10004_u.mp4

markdown
Copy code

**High level steps**

1. Create Python environment and install requirements:
   ```bash
   pip install -r requirements.txt
Place mustard++_text.csv in data/ and put videos in data/final_context_videos/ and data/final_utterance_videos/.

Extract features (text/audio/visual):

bash
Copy code
python src/extract_features.py --data_dir data --out_dir outputs/features --device cuda
Train:

bash
Copy code
python src/train.py --data_dir data --features_dir outputs/features --output_dir outputs/models --epochs 10 --batch_size 8 --device cuda
Evaluate:

bash
Copy code
python src/evaluate.py --data_dir data --features_dir outputs/features --model_path outputs/models/best_model.pth --device cuda
Notes

The scripts are intentionally simple and suitable for small-scale experiments and debugging. For production or larger-scale training, adapt batching, caching, augmentations, reproducibility, and multi-GPU logic.

Feature extraction uses DistilBERT (text), librosa (audio), and ResNet18 (visual single-frame features). You can replace these with more advanced extractors if desired.

yaml
Copy code

---

# `requirements.txt`

torch>=1.12
torchvision
torchaudio
transformers>=4.0
numpy
pandas
scikit-learn
opencv-python
librosa
tqdm
python-dotenv

yaml
Copy code

---

# `.gitignore`

pycache/
*.pyc
outputs/
.idea/
.vscode/
.env

yaml
Copy code

---

# `src/__init__.py`

```python
# package marker