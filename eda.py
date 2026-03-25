"""
eda.py  —  Exploratory Data Analysis
Handwritten Digit & Math Expression Recognizer

  1. Class distribution
  2. Sample grid (all 15 classes)
  3. Pixel intensity statistics per class
  4. Mean & variance images per class
  5. Global pixel intensity histogram
  6. Per-class pixel intensity distribution (violin)
  7. Image sharpness / contrast analysis
  8. t-SNE 2-D embedding (cluster visualisation)
"""

import os, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist
import cv2

warnings.filterwarnings("ignore")
os.makedirs("outputs/eda", exist_ok=True)

SEED = 42
np.random.seed(SEED)
rng  = np.random.default_rng(SEED)

LABEL_MAP  = {
    **{i: str(i) for i in range(10)},
    10: "+", 11: "−", 12: "×", 13: "÷", 14: "="
}
CLASSES    = [LABEL_MAP[i] for i in range(15)]
NUM_CLASSES = 15
PALETTE    = sns.color_palette("tab20", NUM_CLASSES)

#  HELPERS

def save(name):
    path = f"outputs/eda/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅  saved → {path}")


def generate_symbols(n_per_class=4000):
    SYMBOLS = ["+", "-", "x", "/", "="]
    FONT    = cv2.FONT_HERSHEY_SIMPLEX
    imgs, labels = [], []
    for idx, sym in enumerate(SYMBOLS):
        for _ in range(n_per_class):
            img = np.zeros((28, 28), dtype=np.uint8)
            fs  = rng.uniform(0.6, 1.1)
            th  = int(rng.integers(1, 3))
            ts  = cv2.getTextSize(sym, FONT, fs, th)[0]
            xo  = max(1, int((28 - ts[0]) / 2) + rng.integers(-2, 3))
            yo  = max(1, int((28 + ts[1]) / 2) + rng.integers(-2, 3))
            cv2.putText(img, sym, (xo, yo), FONT, fs, 255, th, cv2.LINE_AA)
            angle = rng.uniform(-15, 15)
            M     = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            img   = cv2.warpAffine(img, M, (28, 28))
            noise = rng.normal(0, 8, img.shape).astype(np.int16)
            img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            imgs.append(img)
            labels.append(10 + idx)
    return np.array(imgs), np.array(labels)


# ─────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n📦  Loading data …")
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
x_digits  = np.concatenate([x_tr, x_te])          # (70000, 28, 28)
y_digits  = np.concatenate([y_tr, y_te])

print("    Generating math symbols …")
x_sym, y_sym = generate_symbols(n_per_class=4000)  # (20000, 28, 28)

X = np.concatenate([x_digits, x_sym])              # (90000, 28, 28)
Y = np.concatenate([y_digits, y_sym])

print(f"    Total: {len(X):,} images  |  {NUM_CLASSES} classes\n")

#  1. CLASS DISTRIBUTION
print("1️⃣   Class distribution …")

counts = [np.sum(Y == i) for i in range(NUM_CLASSES)]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("1 — Class Distribution", fontsize=15, fontweight="bold")

# bar chart
bars = axes[0].bar(CLASSES, counts, color=PALETTE, edgecolor="black", linewidth=0.5)
axes[0].set_title("Sample count per class")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
for bar, cnt in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{cnt:,}", ha="center", va="bottom", fontsize=8)
axes[0].tick_params(axis="x", labelsize=11)
axes[0].grid(axis="y", alpha=0.4)

# pie chart — digit vs symbol
digit_total  = np.sum(Y < 10)
symbol_total = np.sum(Y >= 10)
axes[1].pie(
    [digit_total, symbol_total],
    labels=[f"Digits\n{digit_total:,}", f"Math symbols\n{symbol_total:,}"],
    colors=["#4C9BE8", "#F28C38"],
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
axes[1].set_title("Digits vs Math Symbols")

plt.tight_layout()
save("01_class_distribution")

#  2. SAMPLE GRID  (5 samples per class, all 15 classes)
print("2️⃣   Sample grid …")

N_SAMPLES = 5
fig, axes = plt.subplots(NUM_CLASSES, N_SAMPLES, figsize=(N_SAMPLES * 1.8, NUM_CLASSES * 1.8))
fig.suptitle("2 — Random Samples per Class", fontsize=15, fontweight="bold", y=1.01)

for cls in range(NUM_CLASSES):
    idx = np.where(Y == cls)[0]
    chosen = rng.choice(idx, N_SAMPLES, replace=False)
    for col, img_idx in enumerate(chosen):
        ax = axes[cls, col]
        ax.imshow(X[img_idx], cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(LABEL_MAP[cls], fontsize=14, rotation=0,
                          labelpad=20, va="center", fontweight="bold",
                          color=PALETTE[cls])

plt.tight_layout()
save("02_sample_grid")

#  3. PIXEL INTENSITY STATISTICS PER CLASS
print("3️⃣   Pixel intensity statistics …")

stats = {"class": [], "mean": [], "std": [], "min": [], "max": [], "nonzero_%": []}
for cls in range(NUM_CLASSES):
    imgs = X[Y == cls].astype(float)
    stats["class"].append(LABEL_MAP[cls])
    stats["mean"].append(imgs.mean())
    stats["std"].append(imgs.std())
    stats["min"].append(imgs.min())
    stats["max"].append(imgs.max())
    stats["nonzero_%"].append((imgs > 0).mean() * 100)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("3 — Pixel Intensity Statistics per Class", fontsize=15, fontweight="bold")

x_pos = range(NUM_CLASSES)

# mean + std
axes[0].bar(x_pos, stats["mean"], color=PALETTE, edgecolor="black", linewidth=0.4)
axes[0].errorbar(x_pos, stats["mean"], yerr=stats["std"],
                 fmt="none", color="black", capsize=4, linewidth=1.2)
axes[0].set_xticks(x_pos); axes[0].set_xticklabels(CLASSES, fontsize=11)
axes[0].set_ylabel("Mean pixel value"); axes[0].set_title("Mean ± Std Dev")
axes[0].grid(axis="y", alpha=0.4)

# std only
axes[1].bar(x_pos, stats["std"], color=PALETTE, edgecolor="black", linewidth=0.4)
axes[1].set_xticks(x_pos); axes[1].set_xticklabels(CLASSES, fontsize=11)
axes[1].set_ylabel("Std deviation"); axes[1].set_title("Standard Deviation (higher = more varied strokes)")
axes[1].grid(axis="y", alpha=0.4)

# nonzero %
axes[2].bar(x_pos, stats["nonzero_%"], color=PALETTE, edgecolor="black", linewidth=0.4)
axes[2].set_xticks(x_pos); axes[2].set_xticklabels(CLASSES, fontsize=11)
axes[2].set_ylabel("% non-zero pixels"); axes[2].set_title("Ink Coverage (% pixels > 0)")
axes[2].grid(axis="y", alpha=0.4)

plt.tight_layout()
save("03_pixel_statistics")

#  4. MEAN & VARIANCE IMAGES PER CLASS
print("4️⃣   Mean & variance images …")

fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(NUM_CLASSES * 1.8, 4))
fig.suptitle("4 — Mean Image (top) & Variance Image (bottom) per Class",
             fontsize=13, fontweight="bold")

for cls in range(NUM_CLASSES):
    imgs = X[Y == cls].astype(float)
    mean_img = imgs.mean(axis=0)
    var_img  = imgs.var(axis=0)

    axes[0, cls].imshow(mean_img, cmap="hot", vmin=0, vmax=255)
    axes[0, cls].set_title(LABEL_MAP[cls], fontsize=13, fontweight="bold")
    axes[0, cls].axis("off")

    axes[1, cls].imshow(var_img, cmap="plasma")
    axes[1, cls].axis("off")

axes[0, 0].set_ylabel("Mean", fontsize=11, rotation=0, labelpad=30, va="center")
axes[1, 0].set_ylabel("Variance", fontsize=11, rotation=0, labelpad=30, va="center")

plt.tight_layout()
save("04_mean_variance_images")

#  5. GLOBAL PIXEL INTENSITY HISTOGRAM
print("5️⃣   Global pixel histogram …")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("5 — Global Pixel Intensity Distribution", fontsize=15, fontweight="bold")

# full range
flat = X.flatten()
axes[0].hist(flat, bins=64, color="#4C9BE8", edgecolor="navy", linewidth=0.3, alpha=0.85)
axes[0].set_title("All pixels (full range 0–255)")
axes[0].set_xlabel("Pixel value"); axes[0].set_ylabel("Frequency")
axes[0].grid(alpha=0.4)

# non-zero only
nonzero = flat[flat > 0]
axes[1].hist(nonzero, bins=64, color="#F28C38", edgecolor="saddlebrown",
             linewidth=0.3, alpha=0.85)
axes[1].set_title("Non-zero pixels only (ink pixels)")
axes[1].set_xlabel("Pixel value"); axes[1].set_ylabel("Frequency")
axes[1].grid(alpha=0.4)

plt.tight_layout()
save("05_global_histogram")

#  6. PER-CLASS PIXEL DISTRIBUTION  (violin plot)
print("6️⃣   Per-class violin plot …")

# sample 500 images per class to keep it fast
sample_data, sample_labels = [], []
for cls in range(NUM_CLASSES):
    idx  = np.where(Y == cls)[0]
    pick = rng.choice(idx, min(500, len(idx)), replace=False)
    means = X[pick].mean(axis=(1, 2))  # mean pixel per image
    sample_data.extend(means)
    sample_labels.extend([LABEL_MAP[cls]] * len(means))

fig, ax = plt.subplots(figsize=(16, 6))
sns.violinplot(x=sample_labels, y=sample_data, palette=PALETTE,
               order=CLASSES, inner="quartile", ax=ax)
ax.set_title("6 — Per-class Distribution of Mean Pixel Intensity",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Class", fontsize=12)
ax.set_ylabel("Mean pixel value per image", fontsize=12)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
save("06_violin_per_class")

#  7. SHARPNESS / CONTRAST ANALYSIS  (Laplacian variance)
print("7️⃣   Sharpness analysis …")

def laplacian_var(img):
    return cv2.Laplacian(img.astype(np.uint8), cv2.CV_64F).var()

sharp_by_class = []
for cls in range(NUM_CLASSES):
    idx   = np.where(Y == cls)[0]
    pick  = rng.choice(idx, min(300, len(idx)), replace=False)
    vals  = [laplacian_var(X[i]) for i in pick]
    sharp_by_class.append(vals)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("7 — Image Sharpness per Class (Laplacian Variance)",
             fontsize=14, fontweight="bold")

means_sharp = [np.mean(v) for v in sharp_by_class]
axes[0].bar(CLASSES, means_sharp, color=PALETTE, edgecolor="black", linewidth=0.4)
axes[0].set_title("Mean sharpness per class")
axes[0].set_xlabel("Class"); axes[0].set_ylabel("Laplacian variance")
axes[0].grid(axis="y", alpha=0.4)

sns.boxplot(data=sharp_by_class, palette=PALETTE, ax=axes[1])
axes[1].set_xticks(range(NUM_CLASSES))
axes[1].set_xticklabels(CLASSES, fontsize=11)
axes[1].set_title("Sharpness distribution (box plot)")
axes[1].set_xlabel("Class"); axes[1].set_ylabel("Laplacian variance")
axes[1].grid(axis="y", alpha=0.4)

plt.tight_layout()
save("07_sharpness_analysis")

#  8. t-SNE 2-D EMBEDDING
print("8️⃣   t-SNE embedding (this takes ~2 min) …")

# use 300 samples per class → 4 500 total for reasonable speed
tsne_x, tsne_y = [], []
for cls in range(NUM_CLASSES):
    idx  = np.where(Y == cls)[0]
    pick = rng.choice(idx, min(300, len(idx)), replace=False)
    tsne_x.append(X[pick].reshape(len(pick), -1))
    tsne_y.extend([cls] * len(pick))

tsne_x = np.vstack(tsne_x).astype(np.float32) / 255.0
tsne_y = np.array(tsne_y)

tsne   = TSNE(n_components=2, random_state=SEED, perplexity=40,
              max_iter=1000, verbose=0)
embed  = tsne.fit_transform(tsne_x)

fig, ax = plt.subplots(figsize=(13, 10))
for cls in range(NUM_CLASSES):
    mask = tsne_y == cls
    ax.scatter(embed[mask, 0], embed[mask, 1],
               color=PALETTE[cls], label=LABEL_MAP[cls],
               alpha=0.55, s=18, linewidths=0)

ax.legend(title="Class", bbox_to_anchor=(1.01, 1), loc="upper left",
          fontsize=11, title_fontsize=11)
ax.set_title("8 — t-SNE 2-D Embedding of All Classes",
             fontsize=14, fontweight="bold")
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
ax.grid(alpha=0.3)
plt.tight_layout()
save("08_tsne_embedding")

print("\n🎉  EDA complete!  All plots → outputs/eda/")
print("""
  Files generated:
    01_class_distribution.png
    02_sample_grid.png
    03_pixel_statistics.png
    04_mean_variance_images.png
    05_global_histogram.png
    06_violin_per_class.png
    07_sharpness_analysis.png
    08_tsne_embedding.png
""")
