import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

from .dataset import MustardDataset
from .model import FusionModel

from torch import nn, optim
from sklearn.metrics import f1_score, accuracy_score

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    # batch is list of samples
    text = torch.stack([b['text'] for b in batch])
    audio = torch.stack([b['audio'] for b in batch])
    visual = torch.stack([b['visual'] for b in batch])
    labels = torch.tensor([b['label'].item() if hasattr(b['label'],'item') else int(b['label']) for b in batch], dtype=torch.long)
    keys = [b['key'] for b in batch]
    return {'text': text, 'audio': audio, 'visual': visual, 'label': labels, 'key': keys}

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds = []
    trues = []
    for batch in tqdm(loader, desc="train"):
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(text, audio, visual)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        trues.extend(labels.detach().cpu().numpy().tolist())
    return np.mean(losses), f1_score(trues, preds, average='weighted'), accuracy_score(trues, preds)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    preds = []
    trues = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            labels = batch['label'].to(device)

            logits = model(text, audio, visual)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())
    return np.mean(losses), f1_score(trues, preds, average='weighted'), accuracy_score(trues, preds)

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # prepare dataset keys by loading metadata and filtering utterance keys
    import pandas as pd
    df = pd.read_csv(Path(args.data_dir) / args.metadata)
    df.columns = [c.strip() for c in df.columns]
    df_u = df[df['KEY'].astype(str).str.endswith("_u")].reset_index(drop=True)
    keys = df_u['KEY'].tolist()

    # simple train/val split
    train_keys, val_keys = train_test_split(keys, test_size=0.2, random_state=args.seed, stratify=df_u['Sarcasm'] if 'Sarcasm' in df_u.columns else None)

    train_ds = MustardDataset(metadata_csv=Path(args.data_dir) / args.metadata,
                              features_dir=args.features_dir,
                              context_video_dir=Path(args.data_dir) / "context_videos",
                              utterance_video_dir=Path(args.data_dir) / "utterance_videos",
                              keys=train_keys)
    val_ds = MustardDataset(metadata_csv=Path(args.data_dir) / args.metadata,
                              features_dir=args.features_dir,
                              context_video_dir=Path(args.data_dir) / "context_videos",
                              utterance_video_dir=Path(args.data_dir) / "utterance_videos",
                              keys=val_keys)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # model
    model = FusionModel(text_dim=args.text_dim, audio_dim=args.audio_dim, visual_dim=args.visual_dim, hidden_dim=args.hidden_dim, out_dim=2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_f1, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train loss: {train_loss:.4f} f1: {train_f1:.4f} acc: {train_acc:.4f}")

        val_loss, val_f1, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Val loss: {val_loss:.4f} f1: {val_f1:.4f} acc: {val_acc:.4f}")

        # save
        ckpt_path = Path(args.output_dir) / f"epoch{epoch}.pth"
        torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch}, str(ckpt_path))
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = Path(args.output_dir) / "best_model.pth"
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'epoch': epoch}, str(best_path))
            print(f"Saved best model to {best_path}")

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--features_dir", type=str, default="outputs/features")
    parser.add_argument("--metadata", type=str, default="metadata.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/models")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--audio_dim", type=int, default=64)
    parser.add_argument("--visual_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    args = parser.parse_args()
    main(args)
