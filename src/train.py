import os
import math
import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import TransformerModel
from src.data import prepare_dataloaders

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_loss(logits, targets, pad_id, criterion):
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    loss = criterion(logits_flat, targets_flat)
    non_pad = (targets_flat != pad_id).sum().item()
    avg_loss = loss / non_pad
    return loss, torch.exp(avg_loss)

def train_epoch(model, loader, optimizer, pad_id, device, criterion, grad_clip=1.0):
    model.train()
    total_loss, total_tokens = 0, 0
    for batch in tqdm(loader, desc="Training"):
        src, tgt = batch["src_input"].to(device), batch["tgt_input"].to(device)
        dec_in, target = tgt[:, :-1], tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, dec_in)
        loss, _ = compute_loss(logits, target, pad_id, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        total_tokens += (target != pad_id).sum().item()
    return total_loss / total_tokens

def evaluate(model, loader, pad_id, device, criterion):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            src, tgt = batch["src_input"].to(device), batch["tgt_input"].to(device)
            dec_in, target = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, dec_in)
            loss, _ = compute_loss(logits, target, pad_id, criterion)
            total_loss += loss.item()
            total_tokens += (target != pad_id).sum().item()
    return total_loss / total_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--limit_train_samples", type=int, default=48880)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print("加载数据中...")
    train_loader, val_loader, tokenizer = prepare_dataloaders(args.max_len, args.batch_size, args.limit_train_samples)
    pad_id = tokenizer.pad_token_id

    model = TransformerModel(
        vocab=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_id=pad_id
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')

    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}")
        train_loss = train_epoch(model, train_loader, optimizer, pad_id, args.device, criterion)
        val_loss = evaluate(model, val_loader, pad_id, args.device, criterion)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        torch.save(model.state_dict(), f"{args.save_dir}/epoch{epoch}.pt")

    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
    print("训练完成，结果保存在:", args.save_dir)

if __name__ == "__main__":
    main()
