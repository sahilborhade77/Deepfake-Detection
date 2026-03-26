"""
face_gate.py
=============
Pre-check gate — runs BEFORE deepfake detection.
Rejects anime, cartoons, objects, landscapes, etc.
Only lets real human face images through to the detector.

Usage:
    from face_gate import FaceGate
    gate = FaceGate()
    result = gate.check("image.jpg")
    if result["pass"]:
        # run your deepfake detector
    else:
        print(result["reason"])  # "No human face detected"
"""

import cv2
import numpy as np
from PIL import Image
import torch


# ─────────────────────────────────────────────
# METHOD 1: OpenCV Face Detector (fast, no GPU needed)
# ─────────────────────────────────────────────

def load_face_cascade():
    """Load OpenCV's built-in Haar cascade for frontal faces."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_faces_opencv(image_path: str, min_confidence: float = 0.3) -> dict:
    """
    Fast face detection using OpenCV Haar cascades.
    Works offline, no extra packages.

    Returns:
        dict with keys: faces_found (int), face_boxes (list), method (str)
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"faces_found": 0, "face_boxes": [], "method": "opencv"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = load_face_cascade()

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),        # ignore tiny detections
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    face_boxes = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return {
        "faces_found": len(face_boxes),
        "face_boxes":  face_boxes,
        "method":      "opencv_haar"
    }


# ─────────────────────────────────────────────
# METHOD 2: facenet-pytorch (more accurate, already installed)
# ─────────────────────────────────────────────

def detect_faces_mtcnn(image_path: str) -> dict:
    """
    Accurate face detection using MTCNN (facenet-pytorch).
    Better at detecting real human faces vs anime/drawings.

    You already have facenet-pytorch installed (from Phase 1 setup).
    """
    try:
        from facenet_pytorch import MTCNN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40)

        img = Image.open(image_path).convert("RGB")
        boxes, probs = mtcnn.detect(img)

        if boxes is None or probs is None:
            return {"faces_found": 0, "face_boxes": [], "method": "mtcnn", "probs": []}

        # Filter by confidence
        MIN_PROB = 0.85  # MTCNN is strict — anime faces score much lower
        valid = [(b, p) for b, p in zip(boxes, probs) if p is not None and p >= MIN_PROB]

        face_boxes = [{"x": int(b[0]), "y": int(b[1]),
                       "w": int(b[2]-b[0]), "h": int(b[3]-b[1])} for b, _ in valid]
        face_probs = [float(p) for _, p in valid]

        return {
            "faces_found": len(face_boxes),
            "face_boxes":  face_boxes,
            "probs":       face_probs,
            "method":      "mtcnn"
        }

    except ImportError:
        # Fall back to OpenCV if facenet_pytorch not available
        return detect_faces_opencv(image_path)


# ─────────────────────────────────────────────
# METHOD 3: Skin tone heuristic (extra anime filter)
# ─────────────────────────────────────────────

def has_realistic_skin_tone(image_path: str, threshold: float = 0.04) -> bool:
    """
    Checks if image contains realistic skin-tone pixels.
    Anime images often lack natural skin color distribution.

    Skin tone in HSV: H in [0,25] or [335,360], S in [0.2,0.8], V > 0.4
    threshold = minimum fraction of pixels that must be skin-toned (default 4%)
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    img_resized = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # Skin tone mask in HSV
    lower1 = np.array([0,  30, 80],  dtype=np.uint8)
    upper1 = np.array([20, 200, 255], dtype=np.uint8)
    lower2 = np.array([165, 30, 80],  dtype=np.uint8)
    upper2 = np.array([180, 200, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    skin_mask = cv2.bitwise_or(mask1, mask2)

    skin_fraction = skin_mask.sum() / (255 * 224 * 224)
    return skin_fraction >= threshold


# ─────────────────────────────────────────────
# MAIN GATE CLASS
# ─────────────────────────────────────────────

class FaceGate:
    """
    Pre-check gate that must pass before running deepfake detection.

    Logic:
        1. Run MTCNN face detection (strict, trained on real faces)
        2. If MTCNN finds 0 faces → try OpenCV as fallback
        3. If still 0 faces → REJECT (anime, cartoon, no face, etc.)
        4. If face found → check skin tone heuristic
        5. If all pass → allow deepfake detection to run

    Example:
        gate = FaceGate()
        result = gate.check("luffy.jpg")
        # result["pass"] = False
        # result["reason"] = "No human face detected — anime/cartoon/object images are not supported"
    """

    def __init__(self, require_skin: bool = True, verbose: bool = True):
        self.require_skin = require_skin
        self.verbose = verbose

    def check(self, image_path: str) -> dict:
        """
        Run all checks on one image.

        Returns dict:
            pass       : bool   — True = proceed with deepfake detection
            reason     : str    — human-readable explanation
            faces      : int    — number of faces found
            face_boxes : list   — bounding boxes
            method     : str    — which detector was used
            skin_check : bool   — did skin heuristic pass
        """
        if self.verbose:
            print(f"[FaceGate] Checking: {image_path}")

        # ── Step 1: MTCNN (preferred) ──
        detection = detect_faces_mtcnn(image_path)
        faces_found = detection["faces_found"]

        # ── Step 2: Fallback to OpenCV if MTCNN found nothing ──
        if faces_found == 0:
            detection = detect_faces_opencv(image_path)
            faces_found = detection["faces_found"]

        # ── Step 3: No face at all → reject ──
        if faces_found == 0:
            return {
                "pass":       False,
                "reason":     "No human face detected — anime, cartoons, objects, and landscapes are not supported. Please upload a real photo of a person.",
                "faces":      0,
                "face_boxes": [],
                "method":     detection["method"],
                "skin_check": False,
            }

        # ── Step 4: Skin tone check (catches anime with detected face outline) ──
        skin_ok = has_realistic_skin_tone(image_path) if self.require_skin else True

        if not skin_ok:
            return {
                "pass":       False,
                "reason":     "Image does not appear to contain a real human face. Anime, illustrated, and AI-art images are not supported.",
                "faces":      faces_found,
                "face_boxes": detection["face_boxes"],
                "method":     detection["method"],
                "skin_check": False,
            }

        # ── Step 5: All checks passed ──
        if self.verbose:
            print(f"[FaceGate] ✓ PASSED — {faces_found} face(s) detected")

        return {
            "pass":       True,
            "reason":     "OK",
            "faces":      faces_found,
            "face_boxes": detection["face_boxes"],
            "method":     detection["method"],
            "skin_check": True,
        }

    def check_batch(self, image_paths: list) -> list:
        """Run gate on a list of images. Returns list of result dicts."""
        return [self.check(p) for p in image_paths]


# ─────────────────────────────────────────────
# HOW TO USE IN app.py (Streamlit)
# ─────────────────────────────────────────────

"""
Paste this into your Streamlit image tab (app.py):

─────────────────────────────────────────────────

from face_gate import FaceGate
gate = FaceGate()

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if uploaded:
    # Save temp file
    tmp_path = f"temp_{uploaded.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())

    st.image(tmp_path, caption="Uploaded Image", use_container_width=True)

    # ── RUN THE GATE FIRST ──
    gate_result = gate.check(tmp_path)

    if not gate_result["pass"]:
        st.error(f"⚠️ {gate_result['reason']}")
        st.info("This tool is designed for real human face photos only.")
        st.stop()   # ← stops here, never runs deepfake detector

    # ── Gate passed → run your detectors ──
    cnn_result = your_efficientnet.predict(tmp_path)
    fft_result = fft_analyzer.predict(tmp_path)
    final      = fuse_scores(cnn_result, fft_result["fake_prob"])

    # Show results...
    st.success(f"Verdict: {final['verdict']} ({final['confidence']:.1%})")

─────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python face_gate.py <image_path>")
        print("Example: python face_gate.py luffy.jpg")
    else:
        gate = FaceGate(verbose=True)
        result = gate.check(sys.argv[1])
        print("\n── Gate Result ──")
        print(f"  Pass      : {result['pass']}")
        print(f"  Reason    : {result['reason']}")
        print(f"  Faces     : {result['faces']}")
        print(f"  Method    : {result['method']}")
        print(f"  Skin check: {result['skin_check']}")
