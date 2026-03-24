import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

BASE = r"E:\Deepfake_Detection\deep\data\audio"

class AudioDataset(Dataset):
    def __init__(self, split, n_mfcc=40, max_len=100):
        self.n_mfcc  = n_mfcc
        self.max_len = max_len
        self.samples = []  # (filepath, label)

        # 0 = Real, 1 = Fake
        for label, cls in enumerate(["Real", "Fake"]):
            folder = os.path.join(BASE, split, cls)
            if not os.path.exists(folder):
                print(f"  ⚠️  Missing: {folder}")
                continue
            for f in os.listdir(folder):
                if f.endswith(".wav"):
                    self.samples.append(
                        (os.path.join(folder, f), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load audio
        audio, sr = librosa.load(path, sr=22050, mono=True)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.n_mfcc
        )

        # Pad or trim to fixed length
        if mfcc.shape[1] < self.max_len:
            pad = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
        else:
            mfcc = mfcc[:, :self.max_len]

        # Normalize
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

        # Shape: [1, n_mfcc, max_len]
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        return mfcc, torch.tensor(label, dtype=torch.float32)


def get_audio_dataloader(split, batch_size=8):
    dataset = AudioDataset(split)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "Train")
    )
    print(f"{split} — {len(dataset)} audio clips, "
          f"{len(loader)} batches")
    return loader


# ── Quick test ───────────────────────────────────
if __name__ == "__main__":
    print("Testing audio loaders...\n")
    train_loader = get_audio_dataloader("Train")
    val_loader   = get_audio_dataloader("Validation")
    test_loader  = get_audio_dataloader("Test")

    mfcc, labels = next(iter(train_loader))
    print(f"\nBatch shape  : {mfcc.shape}")
    print(f"Labels       : {labels}")
    print("\n✅ Audio loader working!")