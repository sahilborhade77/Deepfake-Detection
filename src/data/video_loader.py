import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE       = r"E:\Deepfake_Detection\deep\data\videos"
FAKE_TYPES = ["Deepfakes", "Face2Face", "FaceShifter",
              "FaceSwap", "NeuralTextures"]
FRAMES_PER_VIDEO = 5

class VideoDataset(Dataset):
    def __init__(self, split, frames_per_video=FRAMES_PER_VIDEO):
        self.frames_per_video = frames_per_video
        self.samples = []  # (list_of_frame_paths, label)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Real videos — label 0
        real_root = os.path.join(BASE, "Real")
        if os.path.exists(real_root):
            for video_folder in sorted(os.listdir(real_root)):
                video_path = os.path.join(real_root, video_folder)
                if not os.path.isdir(video_path):
                    continue
                frames = sorted([
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.endswith(".png")
                ])
                if len(frames) >= frames_per_video:
                    # Sample evenly spaced frames
                    indices = np.linspace(
                        0, len(frames) - 1,
                        frames_per_video, dtype=int
                    )
                    self.samples.append(
                        ([frames[i] for i in indices], 0)
                    )

        # Fake videos — label 1
        for fake_type in FAKE_TYPES:
            fake_root = os.path.join(BASE, fake_type)
            if not os.path.exists(fake_root):
                continue
            for video_folder in sorted(os.listdir(fake_root)):
                video_path = os.path.join(fake_root, video_folder)
                if not os.path.isdir(video_path):
                    continue
                frames = sorted([
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.endswith(".png")
                ])
                if len(frames) >= frames_per_video:
                    indices = np.linspace(
                        0, len(frames) - 1,
                        frames_per_video, dtype=int
                    )
                    self.samples.append(
                        ([frames[i] for i in indices], 1)
                    )

        # Shuffle then split
        random.seed(42)
        random.shuffle(self.samples)
        total   = len(self.samples)
        n_train = int(total * 0.70)
        n_val   = int(total * 0.15)

        if split == "Train":
            self.samples = self.samples[:n_train]
        elif split == "Validation":
            self.samples = self.samples[n_train:n_train + n_val]
        else:
            self.samples = self.samples[n_train + n_val:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            frames.append(img)
        frames = torch.stack(frames)
        return frames, torch.tensor(label, dtype=torch.float32)


def get_video_dataloader(split, batch_size=4):
    dataset = VideoDataset(split)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "Train"),
        num_workers=0,
        pin_memory=True
    )
    print(f"{split} — {len(dataset)} videos, "
          f"{len(loader)} batches")
    return loader


# ── Quick test ───────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    print("Testing video loaders...\n")
    train_loader = get_video_dataloader("Train")
    val_loader   = get_video_dataloader("Validation")
    test_loader  = get_video_dataloader("Test")

    frames, labels = next(iter(train_loader))
    print(f"\nBatch shape  : {frames.shape}")
    print(f"Labels       : {labels}")
    print("\n✅ Video loader working!")