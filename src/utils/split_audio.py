import os
import shutil
import random

BASE_REAL = r"E:\Deepfake_Detection\deep\data\audio\REAL"
BASE_FAKE = r"E:\Deepfake_Detection\deep\data\audio\FAKE"
OUT      = r"E:\Deepfake_Detection\deep\data\audio"
RATIOS   = (0.70, 0.15, 0.15)
random.seed(42)

def split_class(src_folder, cls_name):
    files = [f for f in os.listdir(src_folder) if f.endswith(".wav")]
    random.shuffle(files)
    total  = len(files)
    n_train = int(total * RATIOS[0])
    n_val   = int(total * RATIOS[1])

    splits = {
        "Train":      files[:n_train],
        "Validation": files[n_train:n_train + n_val],
        "Test":       files[n_train + n_val:]
    }

    for split, split_files in splits.items():
        dest = os.path.join(OUT, split, cls_name)
        os.makedirs(dest, exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(src_folder, f), dest)
        print(f"  {split:<12} {cls_name}: {len(split_files)}")

print("=" * 50)
print("   SPLITTING AUDIO — 70 / 15 / 15")
print("=" * 50)
print("\n🔊 REAL audio:")
split_class(BASE_REAL, "Real")
print("\n🔊 FAKE audio:")
split_class(BASE_FAKE, "Fake")
print("\n" + "=" * 50)
print("✅ Audio split complete!")
print("=" * 50)