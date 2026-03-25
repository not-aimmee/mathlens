"""
test_model.py
═════════════════════════════════════════════════════════════
Test your trained model 3 ways:

  Mode 1 — Auto test   : runs 50 random images from MNIST +
                         generated symbols, prints results
  Mode 2 — Image file  : pass any image path as argument
  Mode 3 — Draw & test : opens a paint window, you draw,
                         model predicts in real-time

Usage:
    python test_model.py              ← auto test (Mode 1)
    python test_model.py myimage.png  ← test one image (Mode 2)
    python test_model.py --draw       ← draw yourself (Mode 3)
═════════════════════════════════════════════════════════════
"""

import sys
import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# ── Load model ───────────────────────────────────────────────
MODEL_PATH     = "models/digit_math_recognizer.keras"
LABEL_MAP_PATH = "models/label_map.json"

if not os.path.exists(MODEL_PATH):
    print("\n❌  Model not found at 'models/digit_math_recognizer.keras'")
    print("    Please run  train.py  first!\n")
    sys.exit(1)

print("\n⏳  Loading model …")
model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH) as f:
    LABEL_MAP = {int(k): v for k, v in json.load(f).items()}

CLASSES = [LABEL_MAP[i] for i in range(len(LABEL_MAP))]
print(f"✅  Model loaded  |  Classes: {CLASSES}\n")


# ─────────────────────────────────────────────────────────────
#  HELPER  — preprocess any image to (1, 28, 28, 1)
# ─────────────────────────────────────────────────────────────
def preprocess(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img.reshape(1, 28, 28, 1)


def predict(img: np.ndarray):
    x     = preprocess(img)
    probs = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return LABEL_MAP[idx], float(probs[idx]), probs


# ═════════════════════════════════════════════════════════════
#  MODE 1 — AUTO TEST  (50 random samples)
# ═════════════════════════════════════════════════════════════
def run_auto_test():
    print("═" * 55)
    print("  MODE 1 — Auto Test on 50 random samples")
    print("═" * 55)

    # load MNIST test set
    (_, _), (x_test, y_test) = mnist.load_data()

    rng  = np.random.default_rng(42)
    idxs = rng.choice(len(x_test), 50, replace=False)

    correct = 0
    results = []

    for i in idxs:
        img   = x_test[i]
        true  = str(y_test[i])
        pred, conf, probs = predict(img)
        ok    = pred == true
        correct += int(ok)
        results.append((img, true, pred, conf, ok))

    accuracy = correct / len(results) * 100
    print(f"\n  Accuracy on 50 samples: {correct}/50  ({accuracy:.0f}%)\n")

    # ── plot grid of 25 predictions ──────────────────────────
    fig, axes = plt.subplots(5, 10, figsize=(18, 10))
    fig.suptitle(
        f"Auto Test — 50 Random MNIST Samples   |   Accuracy: {accuracy:.0f}%",
        fontsize=14, fontweight="bold"
    )

    for ax, (img, true, pred, conf, ok) in zip(axes.flat, results):
        ax.imshow(img, cmap="gray")
        color = "#2ecc71" if ok else "#e74c3c"
        ax.set_title(
            f"T:{true}  P:{pred}\n{conf*100:.0f}%",
            fontsize=7.5, color=color, fontweight="bold"
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])

    correct_patch = mpatches.Patch(color="#2ecc71", label="Correct")
    wrong_patch   = mpatches.Patch(color="#e74c3c", label="Wrong")
    fig.legend(handles=[correct_patch, wrong_patch],
               loc="lower right", fontsize=11, frameon=True)

    plt.tight_layout()
    out = "outputs/test_results.png"
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  📊  Result grid saved → {out}")
    plt.show()

    # ── confidence distribution ──────────────────────────────
    confs   = [r[3] for r in results]
    correct_confs = [r[3] for r in results if r[4]]
    wrong_confs   = [r[3] for r in results if not r[4]]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(correct_confs, bins=20, alpha=0.7, color="#2ecc71",
            label=f"Correct ({len(correct_confs)})", edgecolor="black", lw=0.4)
    ax.hist(wrong_confs,   bins=20, alpha=0.7, color="#e74c3c",
            label=f"Wrong ({len(wrong_confs)})",   edgecolor="black", lw=0.4)
    ax.axvline(np.mean(confs), color="navy", linestyle="--",
               label=f"Mean conf: {np.mean(confs)*100:.1f}%")
    ax.set_title("Confidence Distribution — Correct vs Wrong Predictions",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(alpha=0.4)
    plt.tight_layout()
    out2 = "outputs/confidence_distribution.png"
    plt.savefig(out2, dpi=150)
    print(f"  📊  Confidence plot saved → {out2}")
    plt.show()


# ═════════════════════════════════════════════════════════════
#  MODE 2 — TEST A SINGLE IMAGE FILE
# ═════════════════════════════════════════════════════════════
def run_image_test(path: str):
    print("═" * 55)
    print(f"  MODE 2 — Testing image: {path}")
    print("═" * 55)

    if not os.path.exists(path):
        print(f"\n❌  File not found: {path}\n"); return

    img_bgr  = cv2.imread(path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    label, conf, probs = predict(img_gray)

    print(f"\n  Prediction  : '{label}'")
    print(f"  Confidence  : {conf*100:.1f}%\n")
    print("  Top 5 predictions:")
    top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
    for rank, (idx, p) in enumerate(top5, 1):
        bar = "█" * int(p * 40)
        print(f"    {rank}. '{LABEL_MAP[idx]}'  {bar:<40}  {p*100:5.1f}%")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title(f"Input image\nPrediction: '{label}'  ({conf*100:.1f}%)",
                      fontsize=13, fontweight="bold", color="#2c3e50")
    axes[0].axis("off")

    top_labels = [LABEL_MAP[i] for i, _ in top5]
    top_probs  = [p for _, p in top5]
    colors     = ["#2ecc71" if LABEL_MAP[i] == label else "#3498db"
                  for i, _ in top5]
    axes[1].barh(top_labels[::-1], top_probs[::-1], color=colors[::-1],
                 edgecolor="black", linewidth=0.5)
    axes[1].set_title("Top 5 Class Probabilities", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Confidence")
    axes[1].set_xlim(0, 1)
    axes[1].grid(axis="x", alpha=0.4)
    for i, (lbl, p) in enumerate(zip(top_labels[::-1], top_probs[::-1])):
        axes[1].text(p + 0.01, i, f"{p*100:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("outputs/single_image_result.png", dpi=150)
    print(f"\n  📊  Result saved → outputs/single_image_result.png")
    plt.show()


# ═════════════════════════════════════════════════════════════
#  MODE 3 — DRAW & PREDICT IN REAL-TIME
# ═════════════════════════════════════════════════════════════
def run_draw_test():
    print("═" * 55)
    print("  MODE 3 — Draw & Predict")
    print("  • Left-click and drag to draw")
    print("  • Press  R  to reset canvas")
    print("  • Press  Q  to quit")
    print("═" * 55)

    CANVAS_SIZE = 280   # 10× zoom of 28×28
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    drawing = [False]
    last_pos = [None]

    window_name = "Draw a digit or symbol  |  R=reset  Q=quit"
    cv2.namedWindow(window_name)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            last_pos[0] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            if last_pos[0]:
                cv2.line(canvas, last_pos[0], (x, y), 255, thickness=16,
                         lineType=cv2.LINE_AA)
            last_pos[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            last_pos[0] = None

    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n  Window opened — start drawing!\n")

    while True:
        display = canvas.copy()

        # predict on current canvas
        if canvas.max() > 0:
            small = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
            label, conf, probs = predict(small)

            # show prediction on canvas
            overlay = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(overlay, (0, 220), (280, 280), (0, 0, 0), -1)
            cv2.putText(overlay, f"'{label}'  {conf*100:.0f}%",
                        (10, 265), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 100), 2, cv2.LINE_AA)
            cv2.imshow(window_name, overlay)
        else:
            cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("r") or key == ord("R"):
            canvas[:] = 0
            print("  🔄  Canvas reset")
        elif key == ord("q") or key == ord("Q") or key == 27:
            break

    cv2.destroyAllWindows()
    print("\n  👋  Draw test closed.\n")


# ═════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_auto_test()
    elif sys.argv[1] == "--draw":
        run_draw_test()
    else:
        run_image_test(sys.argv[1])
