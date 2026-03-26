"""
app.py — Deepfake Detection System
====================================
Streamlit web app with 3 tabs:
    Tab 1 — Image Detection  (EfficientNet + FFT + FaceGate)
    Tab 2 — Video Detection  (EfficientNet+LSTM + Audio + Fusion)
    Tab 3 — About            (how it works, datasets, architecture)
"""

import os
import sys
import tempfile
import time
import numpy as np
import torch
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Add project root to path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config (MUST be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark terminal aesthetic
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0a0f;
    color: #f0f0f5;
}

/* Hide default header */
header[data-testid="stHeader"] { display: none; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111118;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid rgba(255,255,255,0.07);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #9090a8;
    border-radius: 8px;
    padding: 8px 20px;
    letter-spacing: 0.05em;
}
.stTabs [aria-selected="true"] {
    background: #18181f !important;
    color: #00e5a0 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,229,160,0.3);
    border-radius: 12px;
    background: #111118;
    padding: 10px;
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00e5a0, #3b9eff);
    border-radius: 4px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #111118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.07); }

/* Spinner */
.stSpinner > div { border-top-color: #00e5a0 !important; }

/* Success/Error */
.stSuccess {
    background: rgba(0,229,160,0.1);
    border: 1px solid rgba(0,229,160,0.3);
    color: #00e5a0;
    border-radius: 8px;
}
.stError {
    background: rgba(255,91,91,0.1);
    border: 1px solid rgba(255,91,91,0.3);
    color: #ff5b5b;
    border-radius: 8px;
}
.stInfo {
    background: rgba(59,158,255,0.1);
    border: 1px solid rgba(59,158,255,0.25);
    color: #3b9eff;
    border-radius: 8px;
}
.stWarning {
    background: rgba(255,159,59,0.1);
    border: 1px solid rgba(255,159,59,0.3);
    color: #ff9f3b;
    border-radius: 8px;
}

/* Code blocks */
code {
    font-family: 'JetBrains Mono', monospace;
    background: #18181f;
    color: #00e5a0;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 20px;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                color:#00e5a0; letter-spacing:0.15em; margin-bottom:10px;">
        ── DEEPFAKE DETECTION SYSTEM ──
    </div>
    <h1 style="font-family:'Syne',sans-serif; font-size:36px; font-weight:800;
               color:#f0f0f5; letter-spacing:-0.02em; margin:0;">
        Is it <span style="color:#00e5a0;">Real</span>
        or <span style="color:#ff5b5b;">Fake</span>?
    </h1>
    <p style="color:#9090a8; font-size:13px; margin-top:10px;">
        Multimodal deepfake detection — images · videos · audio
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MODEL LOADERS (cached — only loads once per session)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_image_model():
    from src.models.image_detector import ImageDetector
    model = ImageDetector()
    weights = "models/image_model.pth"
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_fft_analyzer():
    from src.models.fft_analysis import FFTAnalyzer
    return FFTAnalyzer(model_path="models/fft_classifier.pth")

@st.cache_resource(show_spinner=False)
def load_video_model():
    from src.models.video_detector import VideoDetector
    model = VideoDetector()
    weights = "models/video_model.pth"
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_audio_model():
    from src.models.audio_detector import AudioDetector
    model = AudioDetector()
    weights = "models/audio_model.pth"
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_face_gate():
    from src.models.face_gate import FaceGate
    return FaceGate(verbose=False)


# ─────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────

def predict_image(model, fft_analyzer, img: Image.Image, tmp_path: str) -> dict:
    """Run EfficientNet + FFT on a PIL image. Returns result dict."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        cnn_prob = model(tensor).item()

    fft_result = fft_analyzer.predict(tmp_path)
    # Add this right after fft_result = fft_analyzer.predict(tmp_path)
    st.write(fft_result)  # ← shows all keys, find the right one
    fft_prob   = fft_result["fake_prob"]

    # Weighted fusion: 65% CNN + 35% FFT
    if fft_prob > 0.5:
        final_score = 0.45 * cnn_prob + 0.55 * fft_prob  # trust FFT more
    else:
        final_score = 0.65 * cnn_prob + 0.35 * fft_prob  # normal weights

    verdict    = "FAKE" if final_score > 0.5 else "REAL"
    confidence = final_score if verdict == "FAKE" else (1 - final_score)

    return {
        "cnn_prob":    cnn_prob,
        "fft_prob":    fft_prob,
        "final_score": final_score,
        "verdict":     verdict,
        "confidence":  round(confidence * 100, 1),
        "spectrum":    fft_result["spectrum"],
        "radial":      fft_result["radial"],
    }


def predict_video(video_path: str) -> dict:
    """Run video model + audio model on an MP4. Returns result dict."""
    import cv2

    # ── Visual score via video model ──
    try:
        video_model = load_video_model()
        from src.models.video_detector import predict_video_file
        visual_result = predict_video_file(video_model, video_path)
        visual_prob   = visual_result["fake_prob"]
        frame_scores  = visual_result.get("frame_scores", [])
    except Exception as e:
        st.warning(f"Video model error: {e} — using fallback")
        visual_prob  = 0.5
        frame_scores = []

    # ── Audio score ──
    try:
        audio_model = load_audio_model()
        from src.models.audio_detector import predict_audio_from_video
        audio_result = predict_audio_from_video(audio_model, video_path)
        audio_prob   = audio_result["fake_prob"]
        has_audio    = True
    except Exception as e:
        audio_prob = None
        has_audio  = False

    # ── Fusion ──
    if has_audio and audio_prob is not None:
        final_score = 0.6 * visual_prob + 0.4 * audio_prob
    else:
        final_score = visual_prob

    verdict    = "FAKE" if final_score > 0.5 else "REAL"
    confidence = final_score if verdict == "FAKE" else (1 - final_score)

    return {
        "visual_prob":  visual_prob,
        "audio_prob":   audio_prob,
        "has_audio":    has_audio,
        "final_score":  final_score,
        "verdict":      verdict,
        "confidence":   round(confidence * 100, 1),
        "frame_scores": frame_scores,
    }


# ─────────────────────────────────────────────────────────────────
# SHARED UI COMPONENTS
# ─────────────────────────────────────────────────────────────────

def show_verdict(verdict: str, confidence: float):
    """Big coloured verdict banner."""
    if verdict == "FAKE":
        st.markdown(f"""
        <div style="background:rgba(255,91,91,0.12); border:1px solid rgba(255,91,91,0.4);
                    border-radius:12px; padding:20px 24px; text-align:center; margin:16px 0;">
            <div style="font-size:32px; margin-bottom:4px;">⚠️</div>
            <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800;
                        color:#ff5b5b; letter-spacing:-0.01em;">DEEPFAKE DETECTED</div>
            <div style="color:#9090a8; font-size:13px; margin-top:6px;
                        font-family:'JetBrains Mono',monospace;">
                Confidence: {confidence}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(0,229,160,0.1); border:1px solid rgba(0,229,160,0.35);
                    border-radius:12px; padding:20px 24px; text-align:center; margin:16px 0;">
            <div style="font-size:32px; margin-bottom:4px;">✅</div>
            <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800;
                        color:#00e5a0; letter-spacing:-0.01em;">APPEARS REAL</div>
            <div style="color:#9090a8; font-size:13px; margin-top:6px;
                        font-family:'JetBrains Mono',monospace;">
                Confidence: {confidence}%
            </div>
        </div>
        """, unsafe_allow_html=True)


def show_scanning_animation():
    """
    Full-screen scanning animation shown while model is running.
    Returns the st.empty() placeholder so you can clear it when done.
    """
    placeholder = st.empty()
    placeholder.markdown("""
    <div id="scan-overlay" style="
        background: #0a0a0f;
        border: 1px solid rgba(0,229,160,0.2);
        border-radius: 14px;
        padding: 40px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin: 16px 0;
    ">
        <!-- Scan line sweep -->
        <div style="
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00e5a0, transparent);
            animation: scanline 1.6s ease-in-out infinite;
        "></div>
 
        <!-- Grid overlay -->
        <div style="
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(0,229,160,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,229,160,0.03) 1px, transparent 1px);
            background-size: 24px 24px;
            pointer-events: none;
        "></div>
 
        <!-- Icon -->
        <div style="
            width: 64px; height: 64px;
            border-radius: 50%;
            border: 2px solid rgba(0,229,160,0.3);
            border-top-color: #00e5a0;
            margin: 0 auto 20px;
            animation: spin 1s linear infinite;
            display: flex; align-items: center; justify-content: center;
        ">
            <div style="
                width: 40px; height: 40px;
                border-radius: 50%;
                border: 2px solid rgba(59,158,255,0.3);
                border-top-color: #3b9eff;
                animation: spin 0.7s linear infinite reverse;
            "></div>
        </div>
 
        <!-- Label -->
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: #00e5a0;
            letter-spacing: 0.1em;
            margin-bottom: 8px;
        ">ANALYSING IMAGE</div>
 
        <!-- Steps ticker -->
        <div id="step-text" style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #55556a;
            letter-spacing: 0.05em;
            height: 18px;
        ">
            <span id="step-ticker">Running EfficientNet...</span>
        </div>
 
        <!-- Progress bar -->
        <div style="
            background: #18181f;
            border-radius: 4px;
            height: 3px;
            margin: 16px auto 0;
            max-width: 240px;
            overflow: hidden;
        ">
            <div style="
                height: 100%;
                border-radius: 4px;
                background: linear-gradient(90deg, #00e5a0, #3b9eff);
                animation: progress 2.4s ease-in-out infinite;
            "></div>
        </div>
 
        <!-- Corner brackets -->
        <div style="position:absolute;top:10px;left:10px;width:14px;height:14px;
                    border-top:1px solid #00e5a0;border-left:1px solid #00e5a0;opacity:0.5;"></div>
        <div style="position:absolute;top:10px;right:10px;width:14px;height:14px;
                    border-top:1px solid #00e5a0;border-right:1px solid #00e5a0;opacity:0.5;"></div>
        <div style="position:absolute;bottom:10px;left:10px;width:14px;height:14px;
                    border-bottom:1px solid #00e5a0;border-left:1px solid #00e5a0;opacity:0.5;"></div>
        <div style="position:absolute;bottom:10px;right:10px;width:14px;height:14px;
                    border-bottom:1px solid #00e5a0;border-right:1px solid #00e5a0;opacity:0.5;"></div>
    </div>
 
    <style>
    @keyframes scanline {
        0%   { top: 0%;   opacity: 0; }
        10%  { opacity: 1; }
        90%  { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    @keyframes progress {
        0%   { width: 0%;   margin-left: 0; }
        50%  { width: 70%;  margin-left: 15%; }
        100% { width: 0%;   margin-left: 100%; }
    }
    </style>
 
    <script>
    const steps = [
        "Running face detection gate...",
        "Loading EfficientNet-B4...",
        "Extracting spatial features...",
        "Running FFT frequency analysis...",
        "Computing radial energy profile...",
        "Fusing CNN + FFT scores...",
        "Generating verdict..."
    ];
    let i = 0;
    const el = document.getElementById("step-ticker");
    if (el) {
        setInterval(() => {
            i = (i + 1) % steps.length;
            el.style.opacity = 0;
            setTimeout(() => {
                if (el) { el.textContent = steps[i]; el.style.opacity = 1; }
            }, 200);
        }, 900);
    }
    </script>
    """, unsafe_allow_html=True)
    return placeholder


def show_prob_bar(label: str, prob: float, color: str = "#3b9eff"):
    """Labelled probability bar."""
    pct = round(prob * 100, 1)
    st.markdown(f"""
    <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:11px;
                         color:#9090a8; letter-spacing:0.05em;">{label}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:11px;
                         color:{color}; font-weight:600;">{pct}%</span>
        </div>
        <div style="background:#18181f; border-radius:4px; height:6px; overflow:hidden;">
            <div style="width:{pct}%; height:100%;
                        background:{color}; border-radius:4px;
                        transition:width 0.4s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def plot_fft_spectrum(spectrum: np.ndarray, radial: np.ndarray):
    """Render FFT spectrum + radial profile chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("#0a0a0f")

    ax1.imshow(spectrum, cmap="magma")
    ax1.set_title("FFT Frequency Spectrum", color="white", fontsize=11, pad=8)
    ax1.axis("off")

    r = radial[:80]
    ax2.fill_between(range(len(r)), r, alpha=0.25, color="#3b9eff")
    ax2.plot(r, color="#3b9eff", linewidth=1.8)
    ax2.set_facecolor("#111118")
    ax2.set_title("Radial Energy Profile", color="white", fontsize=11, pad=8)
    ax2.tick_params(colors="#55556a", labelsize=8)
    ax2.set_xlabel("Frequency Ring", color="#55556a", fontsize=9)
    ax2.grid(alpha=0.15, color="#ffffff")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#18181f")

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_frame_timeline(frame_scores: list):
    """Per-frame fake probability timeline for video."""
    if not frame_scores:
        return

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#111118")

    x = list(range(len(frame_scores)))
    ax.fill_between(x, frame_scores, alpha=0.2, color="#ff9f3b")
    ax.plot(x, frame_scores, color="#ff9f3b", linewidth=1.8)
    ax.axhline(0.5, color="#ff5b5b", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_title("Per-Frame Fake Probability", color="white", fontsize=11, pad=8)
    ax.set_xlabel("Frame", color="#55556a", fontsize=9)
    ax.set_ylabel("Fake Probability", color="#55556a", fontsize=9)
    ax.tick_params(colors="#55556a", labelsize=8)
    ax.grid(alpha=0.15, color="#ffffff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#18181f")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🖼  IMAGE", "🎬  VIDEO", "ℹ  ABOUT"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE DETECTION
# ════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("""
    <p style="color:#9090a8; font-size:13px; margin-bottom:20px;">
    Upload a face photo — the system checks spatial features
    <span style="color:#00e5a0;">(EfficientNet)</span> and
    frequency artifacts <span style="color:#3b9eff;">(FFT)</span> to detect deepfakes.
    </p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # ── Face Gate ────────────────────────────────────────────
        tmp_path = os.path.join(tempfile.gettempdir(), f"df_{uploaded.name}")
        img.save(tmp_path)

        try:
            gate = load_face_gate()
            gate_result = gate.check(tmp_path)
            if not gate_result["pass"]:
                st.error(f"⚠️ {gate_result['reason']}")
                st.stop()
        except Exception as e:
            st.warning(f"Face gate unavailable ({e}) — proceeding without check.")

        # ── Detection ────────────────────────────────────────────
        with st.spinner("Analysing image..."):
            try:
                model        = load_image_model()
                fft_analyzer = load_fft_analyzer()
                result       = predict_image(model, fft_analyzer, img, tmp_path)

                st.divider()
                show_verdict(result["verdict"], result["confidence"])
                st.divider()

                # Score breakdown
                st.markdown("**Score Breakdown**")
                show_prob_bar("CNN  (EfficientNet spatial)", result["cnn_prob"],  "#00e5a0")
                show_prob_bar("FFT  (frequency artifacts)", result["fft_prob"],   "#3b9eff")
                show_prob_bar("FUSED  (65% CNN + 35% FFT)", result["final_score"],"#9b7fff")

                st.divider()

                # FFT visualisation
                with st.expander("🔬 View FFT Frequency Analysis", expanded=False):
                    plot_fft_spectrum(result["spectrum"], result["radial"])
                    st.caption(
                        "Real images have smooth frequency spectra. "
                        "Deepfakes often show grid-like spikes caused by GAN upsampling layers."
                    )

            except Exception as e:
                st.error(f"Detection failed: {e}")
                st.info("Make sure models are trained: `python src/train/train_image.py`")


# ════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO DETECTION
# ════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <p style="color:#9090a8; font-size:13px; margin-bottom:20px;">
    Upload an MP4 video — frames are analysed with
    <span style="color:#3b9eff;">EfficientNet+LSTM</span> for visual inconsistencies,
    and audio is checked with a
    <span style="color:#ff9f3b;">voice classifier</span>.
    Both scores are fused into a final verdict.
    </p>
    """, unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Choose a video...",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if uploaded_video:
        # Save to temp file
        tmp_video = os.path.join(tempfile.gettempdir(), f"df_{uploaded_video.name}")
        with open(tmp_video, "wb") as f:
            f.write(uploaded_video.read())

        st.video(tmp_video)

        with st.spinner("Extracting frames and analysing video..."):
            try:
                result = predict_video(tmp_video)

                st.divider()
                show_verdict(result["verdict"], result["confidence"])
                st.divider()

                # Score breakdown
                st.markdown("**Score Breakdown**")
                show_prob_bar("VISUAL  (EfficientNet+LSTM)", result["visual_prob"], "#3b9eff")

                if result["has_audio"] and result["audio_prob"] is not None:
                    show_prob_bar("AUDIO   (voice classifier)", result["audio_prob"],  "#ff9f3b")
                    show_prob_bar("FUSED   (60% visual + 40% audio)", result["final_score"], "#9b7fff")
                else:
                    st.caption("ℹ️ No audio track found — verdict based on visual only.")
                    show_prob_bar("VISUAL ONLY", result["final_score"], "#9b7fff")

                # Per-frame timeline
                if result["frame_scores"]:
                    st.divider()
                    with st.expander("📈 Per-Frame Fake Probability Timeline", expanded=True):
                        plot_frame_timeline(result["frame_scores"])
                        st.caption(
                            "Dashed red line = decision threshold (0.5). "
                            "Peaks above it indicate frames flagged as fake."
                        )

            except Exception as e:
                st.error(f"Detection failed: {e}")
                st.info("Make sure models are trained: `python src/train/train_video.py`")


# ════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                color:#55556a; letter-spacing:0.08em; margin-bottom:20px;">
    // SYSTEM ARCHITECTURE
    </div>
    """, unsafe_allow_html=True)

    st.code("""
VIDEO INPUT ──┬──▶ extract_frames()  ──▶ EfficientNet+LSTM ──▶ visual_score
              └──▶ extract_audio()   ──▶ CNN/MLP classifier ──▶ audio_score
                                                                      ▼
IMAGE INPUT ────────▶ EfficientNet ──────────────▶ score_fusion() ──▶ VERDICT
                         +FFT                    (weighted average)
    """, language="text")

    st.divider()

    st.markdown("**Datasets Used**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#111118; border:1px solid rgba(0,229,160,0.2);
                    border-radius:10px; padding:14px;">
            <div style="color:#00e5a0; font-size:11px; font-family:'JetBrains Mono',monospace;
                        margin-bottom:6px;">IMAGE + VIDEO</div>
            <div style="font-size:13px; font-weight:600;">FaceForensics++</div>
            <div style="color:#9090a8; font-size:11px; margin-top:4px;">Extracted frames dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#111118; border:1px solid rgba(59,158,255,0.2);
                    border-radius:10px; padding:14px;">
            <div style="color:#3b9eff; font-size:11px; font-family:'JetBrains Mono',monospace;
                        margin-bottom:6px;">IMAGE</div>
            <div style="font-size:13px; font-weight:600;">Deepfake & Real Images</div>
            <div style="color:#9090a8; font-size:11px; margin-top:4px;">Kaggle — manjilkarki</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#111118; border:1px solid rgba(255,159,59,0.2);
                    border-radius:10px; padding:14px;">
            <div style="color:#ff9f3b; font-size:11px; font-family:'JetBrains Mono',monospace;
                        margin-bottom:6px;">AUDIO</div>
            <div style="font-size:13px; font-weight:600;">DEEP-VOICE</div>
            <div style="color:#9090a8; font-size:11px; margin-top:4px;">DeepFake voice recognition</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**How to run**")
    st.code("streamlit run app.py", language="bash")
    st.markdown("**Train all models first**")
    st.code("""python src/train/train_image.py
python src/train/train_fft.py
python src/train/train_video.py
python src/train/train_audio.py""", language="bash")