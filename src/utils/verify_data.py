import os

BASE = r"E:\Deepfake_Detection\deep\data"

def count_files(folder, ext=None):
    if not os.path.exists(folder):
        return f"MISSING ❌"
    files = os.listdir(folder)
    if ext:
        files = [f for f in files if f.endswith(ext)]
    return len(files)

print("=" * 50)
print("   DEEPFAKE DETECTION — DATA VERIFICATION")
print("=" * 50)

# ── IMAGES ──────────────────────────────────────
print("\n📁 IMAGES")
splits = ["Train", "Validation", "Test"]
for split in splits:
    real = count_files(os.path.join(BASE, "images", split, "Real"), ".jpg")
    fake = count_files(os.path.join(BASE, "images", split, "Fake"), ".jpg")
    print(f"  {split:<12} Real: {real:<6} Fake: {fake}")

# ── AUDIO ────────────────────────────────────────
print("\n🔊 AUDIO")
real = count_files(os.path.join(BASE, "audio", "REAL"), ".wav")
fake = count_files(os.path.join(BASE, "audio", "FAKE"), ".wav")
print(f"  Real: {real}   Fake: {fake}")

# ── VIDEOS ───────────────────────────────────────
print("\n🎬 VIDEOS")
fake_types = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
for ft in fake_types:
    path = os.path.join(BASE, "videos", ft)
    count = count_files(path)
    print(f"  {ft:<20} Files: {count}")

real_path = os.path.join(BASE, "videos", "Real")
print(f"  {'Real':<20} Files: {count_files(real_path)}")

print("\n" + "=" * 50)
print("✅ Verification complete!")
print("=" * 50)