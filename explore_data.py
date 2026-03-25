"""
explore_data.py
───────────────
Run this to visualise samples from both MNIST digits
and the synthetic math symbols before training.
Saves a grid image to outputs/data_samples.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.datasets import mnist
import cv2, os

os.makedirs("outputs", exist_ok=True)

SEED = 42
rng  = np.random.default_rng(SEED)

LABEL_MAP = {
    **{i: str(i) for i in range(10)},
    10: "+", 11: "-", 12: "x", 13: "/", 14: "="
}

# ── load MNIST ───────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_mnist = np.concatenate([x_train, x_test])
y_mnist = np.concatenate([y_train, y_test])

# ── generate a few math symbols ──────────────────────────
SYMBOLS = ["+", "-", "x", "/", "="]
FONT    = cv2.FONT_HERSHEY_SIMPLEX
sym_images, sym_labels = [], []
for idx, sym in enumerate(SYMBOLS):
    for _ in range(5):
        img = np.zeros((28, 28), dtype=np.uint8)
        fs  = rng.uniform(0.7, 1.0)
        th  = int(rng.integers(1, 3))
        ts  = cv2.getTextSize(sym, FONT, fs, th)[0]
        xo  = max(1, int((28 - ts[0]) / 2) + rng.integers(-2, 3))
        yo  = max(1, int((28 + ts[1]) / 2) + rng.integers(-2, 3))
        cv2.putText(img, sym, (xo, yo), FONT, fs, 255, th, cv2.LINE_AA)
        angle = rng.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        img   = cv2.warpAffine(img, M, (28, 28))
        sym_images.append(img)
        sym_labels.append(10 + idx)

sym_images = np.array(sym_images)
sym_labels = np.array(sym_labels)

# ── plot grid ────────────────────────────────────────────
COLS = 10
rows_digit = 2      # 20 digit samples
rows_sym   = 1      #  5 symbol classes × 2 columns → 1 row of 10

fig, axes = plt.subplots(rows_digit + rows_sym + 1, COLS,
                          figsize=(18, (rows_digit + rows_sym + 1) * 2))
fig.suptitle("Data Samples  –  Digits (MNIST) & Math Symbols (Synthetic)",
             fontsize=16, fontweight="bold", y=1.01)

# digit rows
for row in range(rows_digit):
    for col in range(COLS):
        idx = rng.integers(len(x_mnist))
        axes[row, col].imshow(x_mnist[idx], cmap="gray")
        axes[row, col].set_title(str(y_mnist[idx]), fontsize=11)
        axes[row, col].axis("off")

# separator row (blank)
for col in range(COLS):
    axes[rows_digit, col].axis("off")

# symbol row
for col in range(COLS):
    if col < len(sym_images):
        axes[rows_digit + 1, col].imshow(sym_images[col], cmap="gray")
        axes[rows_digit + 1, col].set_title(
            LABEL_MAP[sym_labels[col]], fontsize=14)
        axes[rows_digit + 1, col].axis("off")

# legend patches
patches = [
    mpatches.Patch(color="royalblue",  label="MNIST digits (rows 1–2)"),
    mpatches.Patch(color="darkorange", label="Synthetic math symbols (row 4)"),
]
fig.legend(handles=patches, loc="lower center", ncol=2,
           fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
out = "outputs/data_samples.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"✅  Data sample grid saved → {out}")
plt.show()
