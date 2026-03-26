import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.models.frequency_detector import extract_fft_features

BASE = r"E:\Deepfake_Detection\deep\data\images"

class FrequencyDataset(Dataset):
    def __init__(self, split):
        self.samples = []

        for label, cls in enumerate(["Real", "Fake"]):
            folder = os.path.join(BASE, split, cls)
            if not os.path.exists(folder):
                print(f"  ⚠️ Missing: {folder}")
                continue
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(folder, f), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img      = Image.open(path).convert("RGB")
        features = extract_fft_features(img)
        return features, torch.tensor(label, dtype=torch.float32)


def get_frequency_dataloader(split, batch_size=64):
    dataset = FrequencyDataset(split)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "Train"),
        num_workers=0,
        pin_memory=False
    )
    print(f"{split} — {len(dataset)} images, {len(loader)} batches")
    return loader


# ── Quick test ───────────────────────────────────
if __name__ == "__main__":
    print("Testing frequency loader...\n")
    train_loader = get_frequency_dataloader("Train", batch_size=64)
    val_loader   = get_frequency_dataloader("Validation", batch_size=64)

    features, labels = next(iter(train_loader))
    print(f"\nBatch shape  : {features.shape}")
    print(f"Labels       : {labels[:8]}")
    print("\n✅ Frequency loader working!")