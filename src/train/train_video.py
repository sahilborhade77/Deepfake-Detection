import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r"E:\Deepfake_Detection\deep")
from src.models.video_detector import build_video_model
from src.data.video_loader import get_video_dataloader

# ── Config ───────────────────────────────────────
EPOCHS     = 15
BATCH_SIZE = 4
LR         = 0.0001
SAVE_PATH  = r"E:\Deepfake_Detection\deep\models\video_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

def train():
    print(f"Using device: {DEVICE}\n")

    train_loader = get_video_dataloader("Train",      batch_size=BATCH_SIZE)
    val_loader   = get_video_dataloader("Validation", batch_size=BATCH_SIZE)

    model     = build_video_model(pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ── TRAIN ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for i, (frames, labels) in enumerate(train_loader):
            frames = frames.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(frames)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            preds          = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

            # Progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f}")

        # ── VALIDATE ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(DEVICE)
                labels = labels.unsqueeze(1).to(DEVICE)

                outputs = model(frames)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item()
                preds        = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        # ── Metrics ──
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss   / len(val_loader)
        t_acc  = 100 * train_correct / train_total
        v_acc  = 100 * val_correct   / val_total

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        scheduler.step(v_loss)

        print(f"\nEpoch [{epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✅ Best model saved! Val Acc: {v_acc:.2f}%")

    print(f"\n🏆 Training complete! Best Val Acc: {best_val_acc:.2f}%")
    plot_history(history)


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"],   label="Val Loss")
    ax1.set_title("Video — Loss"); ax1.legend()

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"],   label="Val Acc")
    ax2.set_title("Video — Accuracy"); ax2.legend()

    plt.tight_layout()
    plt.savefig(r"E:\Deepfake_Detection\deep\models\video_training_plot.png")
    print("📊 Video training plot saved!")
    plt.show()


if __name__ == "__main__":
    train()