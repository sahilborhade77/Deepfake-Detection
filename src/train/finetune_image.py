"""
finetune_image.py
-----------------
Fine-tunes image_model.pth on the updated dataset (new personal / celebrity
real-face images included).

Optimised for RTX 2050 4 GB:
  • Mixed-precision (AMP) — new torch.amp API
  • Batch size 16 (safe for 4 GB with EfficientNet-B4 @ 224×224)
  • EfficientNet blocks 0-4 frozen → only blocks 5-6 + classifier trained
  • LR = 2e-5 (5× lower than original) to avoid catastrophic forgetting
  • Saves to models/image_model_finetuned.pth (original NOT overwritten)

Usage:
    cd E:\\Deepfake_Detection\\deep
    python src/train/finetune_image.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast   # ← new unified API
from contextlib import nullcontext
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(r"E:\Deepfake_Detection\deep")
from src.models.image_detector import build_model
from src.data.image_loader import get_dataloader

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS      = 10
BATCH_SIZE  = 16       # safe for RTX 2050 4 GB with AMP
LR          = 2e-5     # 5× lower than original to prevent forgetting
PATIENCE    = 4        # epochs before LR drops
assert torch.cuda.is_available(), "❌ GPU (CUDA) required but not available!"
DEVICE      = torch.device("cuda")

MODEL_DIR   = r"E:\Deepfake_Detection\deep\models"
LOAD_PATH   = os.path.join(MODEL_DIR, "image_model.pth")
SAVE_PATH   = os.path.join(MODEL_DIR, "image_model_finetuned.pth")
PLOT_PATH   = os.path.join(MODEL_DIR, "finetune_training_plot.png")
# ─────────────────────────────────────────────────────────────────────────────


def freeze_backbone(model, freeze_blocks: int = 5):
    """
    Freeze EfficientNet-B4 blocks 0-4. Unfreeze blocks 5-6 + head.
    timm names them: model.blocks[0] … model.blocks[6]
    """
    for name, param in model.model.named_parameters():
        parts = name.split(".")
        trainable = False
        if parts[0] == "blocks":
            try:
                if int(parts[1]) >= freeze_blocks:
                    trainable = True
            except (ValueError, IndexError):
                pass
        elif parts[0] in ("head", "classifier", "global_pool",
                          "conv_head", "bn2", "act2"):
            trainable = True
        param.requires_grad = trainable

    for param in model.classifier.parameters():
        param.requires_grad = True

    frozen     = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable  = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Frozen params   : {frozen:,}")
    print(f"  Trainable params: {trainable:,}\n")


def run_epoch(model, loader, criterion, optimizer, scaler, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training" if training else "Validation", leave=False)
    ctx = torch.no_grad() if not training else nullcontext()
    with ctx:
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda"):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            preds       = (torch.sigmoid(outputs) > 0.5).float()
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def plot_history(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss");     ax1.legend(); ax1.set_xlabel("Epoch")
    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"],   label="Val Acc")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=100)
    print(f"📊  Plot saved → {PLOT_PATH}")
    plt.close(fig)


def train():
    print(f"\n{'='*55}")
    print(f"  FINE-TUNING IMAGE MODEL")
    print(f"  Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Epochs  : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"{'='*55}\n")

    if not os.path.exists(LOAD_PATH):
        print(f"❌  Pretrained model not found at:\n    {LOAD_PATH}")
        sys.exit(1)

    print(f"✅  Loading weights from: {LOAD_PATH}\n")

    train_loader = get_dataloader("Train",      batch_size=BATCH_SIZE)
    val_loader   = get_dataloader("Validation", batch_size=BATCH_SIZE)

    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))
    print("✅  Pretrained weights loaded.\n")

    freeze_backbone(model, freeze_blocks=5)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=0.5)
    scaler    = GradScaler(device="cuda")

    history      = {"train_loss": [], "val_loss": [],
                    "train_acc": [],  "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        t_loss, t_acc = run_epoch(model, train_loader, criterion,
                                  optimizer, scaler, training=True)
        v_loss, v_acc = run_epoch(model, val_loader, criterion,
                                  optimizer, scaler, training=False)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        scheduler.step(v_loss)

        tag = ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), SAVE_PATH)
            tag = "  ✅ saved"

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.2f}%  |  "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.2f}%{tag}")

        # Print LR so we can see if scheduler triggered
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < LR:
            print(f"          ↳ LR reduced to {current_lr:.2e}")

    print(f"\n🏆  Done! Best Val Acc: {best_val_acc:.2f}%")
    print(f"    Saved → {SAVE_PATH}")
    plot_history(history)


if __name__ == "__main__":
    train()
