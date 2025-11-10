#!/usr/bin/env bash
# Example run script (UNIX)
set -e

# 1) install requirements (uncomment if needed)
# pip install -r requirements.txt

# 2) extract features
python src/extract_features.py --data_dir data --out_dir outputs/features --metadata mustard++_text.csv --device cpu

# 3) train (small epochs)
python src/train.py --data_dir data --features_dir outputs/features --metadata mustard++_text.csv --output_dir outputs/models --epochs 3 --batch_size 8 --device cpu

# 4) evaluate
python src/evaluate.py --data_dir data --features_dir outputs/features --metadata mustard++_text.csv --model_path outputs/models/best_model.pth --device cpu --output_dir outputs
