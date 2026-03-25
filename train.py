"""
=============================================================
  Handwritten Digit & Math Expression Recognizer
  Pipeline: Data → Clean → Split → Train CNN → Evaluate → Save
=============================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Reproducibility ────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ──────────────────────────────────────────────────
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

#  STEP 1 – DATASET LOADING
#  Digits 0-9   : MNIST  (60 000 train / 10 000 test)
#  Math symbols : Synthetic images generated on-the-fly
#                 (+  −  ×  ÷  =)  → labels 10-14

LABEL_MAP = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "+", 11: "-", 12: "x", 13: "/", 14: "="
}
NUM_CLASSES = len(LABEL_MAP)   # 15


def load_mnist():
    """Load and basic-format MNIST digits."""
    print("[1/6] Loading MNIST digits …")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate([x_train, x_test], axis=0)   # 70 000 samples
    y = np.concatenate([y_train, y_test], axis=0)
    print(f"      MNIST loaded  → {x.shape[0]:,} images  (28×28 grayscale)")
    return x, y


def generate_math_symbols(samples_per_symbol: int = 4000):
    """
    Render each math symbol as a 28×28 white-on-black image using
    OpenCV + slight augmentation (rotation, scale, noise).
    Returns (images, labels) arrays.
    """
    import cv2

    SYMBOLS    = ["+", "-", "x", "/", "="]
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    images, labels = [], []

    print(f"[1/6] Generating synthetic math symbols  "
          f"({len(SYMBOLS)} symbols × {samples_per_symbol} samples) …")

    rng = np.random.default_rng(SEED)

    for sym_idx, sym in enumerate(SYMBOLS):
        label = 10 + sym_idx
        for _ in range(samples_per_symbol):
            img = np.zeros((28, 28), dtype=np.uint8)

            # randomise scale & position slightly
            font_scale = rng.uniform(0.6, 1.1)
            thickness  = rng.integers(1, 3)
            text_size  = cv2.getTextSize(sym, FONT, font_scale, thickness)[0]
            x_off = int((28 - text_size[0]) / 2) + rng.integers(-2, 3)
            y_off = int((28 + text_size[1]) / 2) + rng.integers(-2, 3)
            x_off = max(1, min(x_off, 27))
            y_off = max(1, min(y_off, 27))

            cv2.putText(img, sym, (x_off, y_off),
                        FONT, font_scale, 255, thickness, cv2.LINE_AA)

            # random rotation ±15°
            angle = rng.uniform(-15, 15)
            M     = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            img   = cv2.warpAffine(img, M, (28, 28))

            # gaussian noise
            noise = rng.normal(0, 8, img.shape).astype(np.int16)
            img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            images.append(img)
            labels.append(label)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    print(f"      Symbols generated → {images.shape[0]:,} images")
    return images, labels

#  STEP 2 – DATA CLEANING

def clean_data(x: np.ndarray, y: np.ndarray):
    """
    Cleaning steps:
      1. Remove blank / near-blank images  (mean pixel < 5)
      2. Remove corrupt images             (any NaN or Inf)
      3. Clip pixel range to [0, 255]
      4. Normalise to [0.0, 1.0]
      5. Reshape to (N, 28, 28, 1)
    """
    print("[2/6] Cleaning data …")
    before = len(x)

    # — remove near-blank
    mask_blank = x.mean(axis=(1, 2)) >= 5
    x, y = x[mask_blank], y[mask_blank]

    # — remove NaN / Inf (shouldn't happen with MNIST but good practice)
    mask_valid = ~np.any(np.isnan(x.astype(float)) |
                         np.isinf(x.astype(float)), axis=(1, 2))
    x, y = x[mask_valid], y[mask_valid]

    # — clip & normalise
    x = np.clip(x, 0, 255).astype(np.float32) / 255.0

    # — add channel dimension
    x = x.reshape(-1, 28, 28, 1)

    removed = before - len(x)
    print(f"      Removed {removed} blank/corrupt samples")
    print(f"      Clean dataset → {len(x):,} samples, shape {x.shape}")
    return x, y

#  STEP 3 – TRAIN / VALIDATION / TEST SPLIT

def split_data(x, y):
    """70 % train | 15 % validation | 15 % test  (stratified)"""
    print("[3/6] Splitting dataset (70 / 15 / 15 stratified) …")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=SEED)

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

    print(f"      Train : {len(x_train):>7,}")
    print(f"      Val   : {len(x_val):>7,}")
    print(f"      Test  : {len(x_test):>7,}")

    # one-hot encode
    y_train_oh = to_categorical(y_train, NUM_CLASSES)
    y_val_oh   = to_categorical(y_val,   NUM_CLASSES)
    y_test_oh  = to_categorical(y_test,  NUM_CLASSES)

    return (x_train, y_train_oh, y_train,
            x_val,   y_val_oh,   y_val,
            x_test,  y_test_oh,  y_test)

#  STEP 4 – CNN MODEL

def build_cnn(num_classes: int = NUM_CLASSES):
    """
    Architecture:
      Block 1 : Conv(32) → BN → Conv(32) → BN → MaxPool → Dropout(0.25)
      Block 2 : Conv(64) → BN → Conv(64) → BN → MaxPool → Dropout(0.25)
      Block 3 : Conv(128)→ BN → MaxPool  → Dropout(0.25)
      Head    : Flatten → Dense(256) → BN → Dropout(0.5) → Softmax(15)
    """
    inp = layers.Input(shape=(28, 28, 1), name="image_input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inp, out, name="digit_math_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

#  STEP 5 – TRAINING

def train_model(model, x_train, y_train, x_val, y_val):
    print("[4/6] Training CNN …")
    model.summary()

    # augmentation (only on training set)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10
    )

    cbs = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                monitor="val_accuracy", verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3,
                                    monitor="val_loss", verbose=1),
        callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "best_model.keras"),
            save_best_only=True, monitor="val_accuracy", verbose=1)
    ]

    BATCH = 128
    EPOCHS = 30

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH),
        steps_per_epoch=len(x_train) // BATCH,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=1
    )
    return history

#  STEP 6 – EVALUATION

def evaluate_and_save(model, history, x_test, y_test_oh, y_test_raw):
    print("[5/6] Evaluating on test set …")
    loss, acc = model.evaluate(x_test, y_test_oh, verbose=0)
    print(f"\n  ✅  Test Accuracy : {acc*100:.2f}%")
    print(f"      Test Loss     : {loss:.4f}\n")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    labels_str = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    print(classification_report(y_test_raw, y_pred, target_names=labels_str))

    # ── plots ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curve
    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True)

    # Loss curve
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "training_curves.png"), dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test_raw, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_str, yticklabels=labels_str, ax=ax)
    ax.set_title("Confusion Matrix – Test Set")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    print(f"      Plots saved → {OUTPUTS_DIR}/")

def save_artifacts(model):
    print("[6/6] Saving model & label map …")

    # Keras native format (.keras) — recommended
    model_path = os.path.join(MODELS_DIR, "digit_math_recognizer.keras")
    model.save(model_path)

    # Also save as SavedModel (TF serving / TFLite friendly)
    tf_path = os.path.join(MODELS_DIR, "saved_model")
    model.export(tf_path)

    # Label map JSON  ← your teammate needs this for the UI
    label_path = os.path.join(MODELS_DIR, "label_map.json")
    with open(label_path, "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"      ✅  Keras model  → {model_path}")
    print(f"      ✅  SavedModel   → {tf_path}/")
    print(f"      ✅  Label map    → {label_path}")

#  MAIN

def main():
    print("=" * 60)
    print("  Handwritten Digit & Math Expression Recognizer – CNN")
    print("=" * 60)

    # 1. Load
    x_digits, y_digits     = load_mnist()
    x_symbols, y_symbols   = generate_math_symbols(samples_per_symbol=4000)

    # Merge
    x_raw = np.concatenate([x_digits,  x_symbols], axis=0)
    y_raw = np.concatenate([y_digits,  y_symbols], axis=0)
    print(f"\n  Combined dataset → {len(x_raw):,} samples, {NUM_CLASSES} classes\n")

    # 2. Clean
    x_clean, y_clean = clean_data(x_raw, y_raw)

    # 3. Split
    (x_train, y_train_oh, _,
     x_val,   y_val_oh,   _,
     x_test,  y_test_oh,  y_test_raw) = split_data(x_clean, y_clean)

    # 4. Build
    model = build_cnn()

    # 5. Train
    history = train_model(model, x_train, y_train_oh, x_val, y_val_oh)

    # 6. Evaluate + save
    evaluate_and_save(model, history, x_test, y_test_oh, y_test_raw)
    save_artifacts(model)

    print("\n    Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
