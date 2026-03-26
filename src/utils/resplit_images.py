import os
import shutil
import random

BASE = r"E:\Deepfake_Detection\deep\data\images"
TEMP = r"E:\Deepfake_Detection\deep\data\images_temp"
SPLITS = ["Train", "Validation", "Test"]
CLASSES = ["Real", "Fake"]
RATIOS = (0.70, 0.15, 0.15)

random.seed(42)

def get_all_files(cls):
    all_files = []
    for split in SPLITS:
        folder = os.path.join(BASE, split, cls)
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith((".jpg", ".jpeg", ".png")):
                    all_files.append(os.path.join(folder, f))
    return all_files

print("=" * 50)
print("   RESPLITTING IMAGES — 70 / 15 / 15")
print("=" * 50)

for cls in CLASSES:
    print(f"\n🔄 Processing {cls}...")

    # Step 1 — collect all file paths
    all_files = get_all_files(cls)
    random.shuffle(all_files)
    total = len(all_files)

    # Step 2 — calculate split sizes
    n_train = int(total * RATIOS[0])
    n_val   = int(total * RATIOS[1])
    n_test  = total - n_train - n_val

    train_files = all_files[:n_train]
    val_files   = all_files[n_train:n_train + n_val]
    test_files  = all_files[n_train + n_val:]

    # Step 3 — copy everything to temp first
    temp_cls = os.path.join(TEMP, cls)
    os.makedirs(temp_cls, exist_ok=True)

    print(f"  Copying to temp folder...")
    for i, f in enumerate(all_files):
        ext = os.path.splitext(f)[1]
        shutil.copy(f, os.path.join(temp_cls, f"{i}{ext}"))

    # Step 4 — clear original folders
    for split in SPLITS:
        folder = os.path.join(BASE, split, cls)
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    # Step 5 — copy from temp into correct splits
    temp_files = [os.path.join(temp_cls, f) for f in os.listdir(temp_cls)]

    print(f"  Distributing into splits...")
    for f in temp_files[:n_train]:
        shutil.copy(f, os.path.join(BASE, "Train", cls))
    for f in temp_files[n_train:n_train + n_val]:
        shutil.copy(f, os.path.join(BASE, "Validation", cls))
    for f in temp_files[n_train + n_val:]:
        shutil.copy(f, os.path.join(BASE, "Test", cls))

    print(f"  Total : {total}")
    print(f"  Train : {n_train}")
    print(f"  Val   : {n_val}")
    print(f"  Test  : {n_test}")

# Step 6 — delete temp folder
shutil.rmtree(TEMP)
print("\n" + "=" * 50)
print("✅ Resplit complete! Temp folder cleaned up.")
print("=" * 50)