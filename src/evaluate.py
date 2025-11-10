import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from .dataset import MustardDataset
from .model import FusionModel
from tqdm import tqdm

def collate_fn(batch):
    text = torch.stack([b['text'] for b in batch])
    audio = torch.stack([b['audio'] for b in batch])
    visual = torch.stack([b['visual'] for b in batch])
    labels = torch.tensor([b['label'].item() if hasattr(b['label'],'item') else int(b['label']) for b in batch], dtype=torch.long)
    keys = [b['key'] for b in batch]
    return {'text': text, 'audio': audio, 'visual': visual, 'label': labels, 'key': keys}

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    # load model
    model = FusionModel(text_dim=args.text_dim, audio_dim=args.audio_dim, visual_dim=args.visual_dim, hidden_dim=args.hidden_dim, out_dim=2)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()

    ds = MustardDataset(metadata_csv=Path(args.data_dir) / args.metadata,
                        features_dir=args.features_dir,
                        context_video_dir=Path(args.data_dir) / "context_videos",
                        utterance_video_dir=Path(args.data_dir) / "utterance_videos")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    preds = []
    trues = []
    keys_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            labels = batch['label'].to(device)
            logits = model(text, audio, visual)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds.extend(pred)
            trues.extend(labels.cpu().numpy().tolist())
            keys_all.extend(batch['key'])
    print("Accuracy:", accuracy_score(trues, preds))
    print("F1 (weighted):", f1_score(trues, preds, average='weighted'))
    print(classification_report(trues, preds))

    # write predictions CSV
    import pandas as pd
    out_df = pd.DataFrame({"KEY": keys_all, "pred": preds, "true": trues})
    out_df.to_csv(Path(args.output_dir) / "predictions.csv", index=False)
    print("Saved predictions to", Path(args.output_dir) / "predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--features_dir", type=str, default="outputs/features")
    parser.add_argument("--metadata", type=str, default="metadata.csv")
    parser.add_argument("--model_path", type=str, default="outputs/models/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--audio_dim", type=int, default=64)
    parser.add_argument("--visual_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
