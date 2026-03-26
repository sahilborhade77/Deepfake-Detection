import torch

def fuse_scores(visual_score, audio_score,
                visual_weight=0.6, audio_weight=0.4):
    """
    Combine visual and audio scores into final verdict.
    Both scores should be probabilities (0.0 to 1.0)
    where 1.0 = FAKE, 0.0 = REAL
    """
    final_score = (visual_weight * visual_score +
                   audio_weight  * audio_score)

    verdict = "FAKE" if final_score > 0.5 else "REAL"

    return {
        "visual_score" : round(visual_score, 4),
        "audio_score"  : round(audio_score,  4),
        "final_score"  : round(final_score,  4),
        "verdict"      : verdict,
        "confidence"   : round(abs(final_score - 0.5) * 200, 1)
    }


# ── Quick test ───────────────────────────────────
if __name__ == "__main__":
    print("Testing score fusion...\n")

    # Test 1 — both say FAKE
    result = fuse_scores(0.85, 0.90)
    print(f"Test 1 — Both FAKE   : {result}")

    # Test 2 — both say REAL
    result = fuse_scores(0.10, 0.15)
    print(f"Test 2 — Both REAL   : {result}")

    # Test 3 — visual says FAKE, audio says REAL
    result = fuse_scores(0.80, 0.20)
    print(f"Test 3 — Mixed       : {result}")

    print("\n✅ Fusion working!")