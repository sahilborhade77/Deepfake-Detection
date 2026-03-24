import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BASE = r"E:\Deepfake_Detection\deep\data\images"

def get_transforms(split):
    if split == "Train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloader(split, batch_size=16, num_workers=0):
    folder = os.path.join(BASE, split)
    # ImageFolder assigns labels based on sorted class names
    # (e.g. ['Fake', 'Real'] => Fake=0, Real=1)
    # Our expected label mapping: 0=REAL, 1=FAKE.
    # Swap labels to enforce this convention in training/evaluation.
    def relabel(idx):
        return 1 - idx

    dataset = datasets.ImageFolder(
        root=folder,
        transform=get_transforms(split),
        target_transform=relabel
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "Train"),
        num_workers=0,
        pin_memory=False
    )
    print(f"{split} — {len(dataset)} images, "
          f"{len(loader)} batches, "
          f"classes_before={['Fake','Real']} classes_after={dataset.classes} (mapped 0=REAL,1=FAKE)")
    return loader


if __name__ == "__main__":
    print("Testing data loaders...\n")
    train_loader = get_dataloader("Train")
    val_loader   = get_dataloader("Validation")
    test_loader  = get_dataloader("Test")

    images, labels = next(iter(train_loader))
    print(f"\nBatch shape  : {images.shape}")
    print(f"Labels shape : {labels.shape}")
    print(f"Label values : {labels}")
    print("\n✅ Data loader working!")


























































