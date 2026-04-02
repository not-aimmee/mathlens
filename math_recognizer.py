"""
Handwritten Math Expression Recognizer
=======================================
Usage:
  python math_recognizer.py --train
  python math_recognizer.py --test
  python math_recognizer.py --solve
  python math_recognizer.py --solve image.png
  python math_recognizer.py --eda
"""

import os, sys, json, re, warnings, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────
SEED        = 42
MODEL_PATH  = "models/math_recognizer.keras"
LABEL_PATH  = "models/label_map.json"
OPERATORS   = {"+", "-", "*", "/", "**"}

np.random.seed(SEED)
tf.random.set_seed(SEED)

LABEL_MAP = {
    0:"0", 1:"1", 2:"2", 3:"3", 4:"4",
    5:"5", 6:"6", 7:"7", 8:"8", 9:"9",
    10:"+", 11:"-", 12:"x", 13:"/",
    14:"^", 15:"(", 16:")", 17:"=",
}
NUM_CLASSES = len(LABEL_MAP)
OP_MAP      = {"x":"*", "^":"**"}


# ══════════════════════════════════════════════════════════════════
#  1. GEOMETRY-BASED SYMBOL + DIGIT DRAWING
#  Every symbol has multiple styles so the model sees every
#  possible way a human might write it.
# ══════════════════════════════════════════════════════════════════

def _rnd(rng, lo, hi):
    """Random integer in [lo, hi]."""
    return int(rng.integers(lo, hi+1))

def _rndf(rng, lo, hi):
    return float(rng.uniform(lo, hi))


# ── PLUS  (+) ────────────────────────────────────────────────────
# KEY FIX: arms are now LONG (40-48% of canvas) and ALWAYS symmetric
# so it can never be confused with a 7 or t
def draw_plus(size, rng):
    img = np.zeros((size,size), dtype=np.uint8)
    cx  = size//2 + _rnd(rng,-2,2)
    cy  = size//2 + _rnd(rng,-2,2)
    # long arms — at least 40% of canvas each side
    L   = _rnd(rng, int(size*0.38), int(size*0.46))
    th  = _rnd(rng, 2, 5)
    # horizontal arm — full width
    cv2.line(img, (cx-L, cy), (cx+L, cy), 255, th)
    # vertical arm — same length, centred
    cv2.line(img, (cx, cy-L), (cx, cy+L), 255, th)
    return img


# ── MINUS  (-) ───────────────────────────────────────────────────
def draw_minus(size, rng):
    img = np.zeros((size,size), dtype=np.uint8)
    cx  = size//2
    cy  = size//2 + _rnd(rng,-4,4)
    L   = _rnd(rng, int(size*0.35), int(size*0.46))
    th  = _rnd(rng, 2, 5)
    cv2.line(img, (cx-L, cy), (cx+L, cy), 255, th)
    return img


# ── MULTIPLY  (x) ────────────────────────────────────────────────
def draw_multiply(size, rng):
    img   = np.zeros((size,size), dtype=np.uint8)
    style = _rnd(rng, 0, 2)
    cx    = size//2 + _rnd(rng,-3,3)
    cy    = size//2 + _rnd(rng,-3,3)
    L     = _rnd(rng, int(size*0.30), int(size*0.42))
    th    = _rnd(rng, 2, 5)

    if style == 0:
        # classic × : two diagonal lines
        cv2.line(img, (cx-L,cy-L), (cx+L,cy+L), 255, th)
        cv2.line(img, (cx+L,cy-L), (cx-L,cy+L), 255, th)
    elif style == 1:
        # slightly rotated ×
        angle = _rndf(rng, 10, 30)
        for a in [45+angle, 45-angle]:
            p1 = (cx+int(L*np.cos(np.radians(a))), cy-int(L*np.sin(np.radians(a))))
            p2 = (cx-int(L*np.cos(np.radians(a))), cy+int(L*np.sin(np.radians(a))))
            cv2.line(img, p1, p2, 255, th)
    else:
        # cursive x — two curves
        pts1 = np.array([[cx-L,cy-L],[cx,cy+_rnd(rng,-4,4)],[cx+L,cy+L]])
        pts2 = np.array([[cx+L,cy-L],[cx,cy+_rnd(rng,-4,4)],[cx-L,cy+L]])
        cv2.polylines(img,[pts1],False,255,th)
        cv2.polylines(img,[pts2],False,255,th)
    return img


# ── DIVIDE  (/) ──────────────────────────────────────────────────
def draw_divide(size, rng):
    img    = np.zeros((size,size), dtype=np.uint8)
    style  = _rnd(rng, 0, 1)
    margin = _rnd(rng, int(size*0.10), int(size*0.18))
    th     = _rnd(rng, 2, 5)

    if style == 0:
        x1 = margin+_rnd(rng,-3,3); y1 = size-margin+_rnd(rng,-3,3)
        x2 = size-margin+_rnd(rng,-3,3); y2 = margin+_rnd(rng,-3,3)
        cv2.line(img,(x1,y1),(x2,y2),255,th)
    else:
        x1=margin; y1=size-margin; x2=size-margin; y2=margin
        cv2.line(img,(x1,y1),(x2,y2),255,th)
        dr=_rnd(rng,2,4)
        cv2.circle(img,(size//2+_rnd(rng,-2,2),_rnd(rng,5,9)),dr,255,-1)
        cv2.circle(img,(size//2+_rnd(rng,-2,2),size-_rnd(rng,5,9)),dr,255,-1)
    return img


# ── POWER  (^) ───────────────────────────────────────────────────
def draw_power(size, rng):
    img   = np.zeros((size,size), dtype=np.uint8)
    style = _rnd(rng, 0, 1)
    cx    = size//2 + _rnd(rng,-4,4)
    th    = _rnd(rng, 2, 5)
    if style == 0:
        top_y = _rnd(rng, int(size*0.18), int(size*0.30))
        bot_y = _rnd(rng, int(size*0.55), int(size*0.70))
        L     = _rnd(rng, int(size*0.22), int(size*0.34))
        cv2.line(img,(cx-L,bot_y),(cx,top_y),255,th)
        cv2.line(img,(cx,top_y),(cx+L,bot_y),255,th)
    else:
        top_y = _rnd(rng, int(size*0.25), int(size*0.38))
        bot_y = _rnd(rng, int(size*0.55), int(size*0.68))
        L     = _rnd(rng, int(size*0.28), int(size*0.40))
        cv2.line(img,(cx-L,bot_y+_rnd(rng,-3,3)),(cx,top_y),255,th)
        cv2.line(img,(cx,top_y),(cx+L,bot_y+_rnd(rng,-3,3)),255,th)
    return img


# ── LEFT PAREN  (() ──────────────────────────────────────────────
# The key visual feature of ( is: bulges LEFT, opens RIGHT
# We make the leftward bulge very pronounced and unmistakable
def draw_lparen(size, rng):
    img = np.zeros((size,size), dtype=np.uint8)
    th  = _rnd(rng, 2, 5)
    n   = 30  # many points = smooth curve

    # anchor points: top-right and bottom-right, bulge goes left
    top_x  = size//2 + _rnd(rng, 4, 10)
    top_y  = _rnd(rng, int(size*0.06), int(size*0.14))
    bot_x  = top_x + _rnd(rng, -4, 4)
    bot_y  = size - top_y + _rnd(rng, -4, 4)
    # leftmost point of the bulge
    mid_x  = _rnd(rng, int(size*0.08), int(size*0.22))
    mid_y  = size//2 + _rnd(rng, -4, 4)

    pts = []
    for i in range(n+1):
        t = i/n
        # quadratic bezier: top -> mid -> bot
        bx = int((1-t)**2*top_x + 2*(1-t)*t*mid_x + t**2*bot_x)
        by = int((1-t)**2*top_y + 2*(1-t)*t*mid_y + t**2*bot_y)
        pts.append([bx, by])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, 255, th)
    return img


# ── RIGHT PAREN  ()) ─────────────────────────────────────────────
# The key visual feature of ) is: bulges RIGHT, opens LEFT
def draw_rparen(size, rng):
    img = np.zeros((size,size), dtype=np.uint8)
    th  = _rnd(rng, 2, 5)
    n   = 30

    # anchor points: top-left and bottom-left, bulge goes right
    top_x  = size//2 - _rnd(rng, 4, 10)
    top_y  = _rnd(rng, int(size*0.06), int(size*0.14))
    bot_x  = top_x + _rnd(rng, -4, 4)
    bot_y  = size - top_y + _rnd(rng, -4, 4)
    # rightmost point of the bulge
    mid_x  = size - _rnd(rng, int(size*0.08), int(size*0.22))
    mid_y  = size//2 + _rnd(rng, -4, 4)

    pts = []
    for i in range(n+1):
        t = i/n
        bx = int((1-t)**2*top_x + 2*(1-t)*t*mid_x + t**2*bot_x)
        by = int((1-t)**2*top_y + 2*(1-t)*t*mid_y + t**2*bot_y)
        pts.append([bx, by])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, 255, th)
    return img


# ── EQUALS  (=) ──────────────────────────────────────────────────
def draw_equals(size, rng):
    img = np.zeros((size,size), dtype=np.uint8)
    cx  = size//2
    gap = _rnd(rng, int(size*0.07), int(size*0.13))
    L   = _rnd(rng, int(size*0.32), int(size*0.46))
    th  = _rnd(rng, 2, 4)
    o1  = _rnd(rng,-2,2); o2=_rnd(rng,-2,2)
    cv2.line(img,(cx-L+o1,cx-gap),(cx+L+o1,cx-gap),255,th)
    cv2.line(img,(cx-L+o2,cx+gap),(cx+L+o2,cx+gap),255,th)
    return img


# ── DIGIT DRAWING ────────────────────────────────────────────────
# KEY FIX: digit 7 uses GEOMETRY (two lines) so it can never be
# confused with + which also uses geometry.
# All other digits use font rendering (MNIST already covers them).

def draw_seven(size, rng):
    """7 drawn as two explicit strokes: top horizontal + diagonal down-left."""
    img = np.zeros((size,size), dtype=np.uint8)
    th  = _rnd(rng, 2, 5)
    top_y = _rnd(rng, int(size*0.18), int(size*0.28))
    x_l   = _rnd(rng, int(size*0.10), int(size*0.18))
    x_r   = _rnd(rng, int(size*0.78), int(size*0.88))
    # top horizontal bar
    cv2.line(img, (x_l, top_y), (x_r, top_y), 255, th)
    # diagonal stroke going down-left
    bot_x = _rnd(rng, int(size*0.18), int(size*0.35))
    bot_y = _rnd(rng, int(size*0.72), int(size*0.86))
    cv2.line(img, (x_r, top_y), (bot_x, bot_y), 255, th)
    # optional middle tick (some people add it)
    if _rnd(rng,0,1)==0:
        mid_x = (x_r+bot_x)//2 + _rnd(rng,-3,3)
        mid_y = (top_y+bot_y)//2 + _rnd(rng,-3,3)
        cv2.line(img,(mid_x-_rnd(rng,4,8),mid_y),(mid_x+_rnd(rng,4,8),mid_y),255,th)
    return img


def draw_six(size, rng):
    """6 = circle at bottom + vertical stroke coming down from top-right."""
    img = np.zeros((size,size), dtype=np.uint8)
    th  = _rnd(rng, 2, 4)
    # bottom circle
    cx  = size//2 + _rnd(rng,-3,3)
    cy  = size//2 + _rnd(rng,4,10)
    r   = _rnd(rng, int(size*0.22), int(size*0.30))
    cv2.circle(img,(cx,cy),r,255,th)
    # vertical stroke from top curving into circle
    top_x = cx + r - _rnd(rng,2,6)
    top_y = _rnd(rng, int(size*0.08), int(size*0.18))
    cv2.line(img,(top_x,top_y),(cx+r,cy),255,th)
    return img


def draw_nine(size, rng):
    """9 = circle at top + vertical stroke going down from bottom-left."""
    img = np.zeros((size,size), dtype=np.uint8)
    th  = _rnd(rng, 2, 4)
    # top circle
    cx  = size//2 + _rnd(rng,-3,3)
    cy  = size//2 - _rnd(rng,4,10)
    r   = _rnd(rng, int(size*0.22), int(size*0.30))
    cv2.circle(img,(cx,cy),r,255,th)
    # vertical stroke going down
    bot_x = cx + r - _rnd(rng,2,6)
    bot_y = size - _rnd(rng, int(size*0.08), int(size*0.18))
    cv2.line(img,(cx+r,cy),(bot_x,bot_y),255,th)
    return img


def draw_digit(digit, size, rng):
    """Render digit. Problem digits use geometry to avoid confusion."""
    if digit == 7:
        return draw_seven(size, rng)
    if digit == 6:
        return draw_six(size, rng)
    if digit == 9:
        return draw_nine(size, rng)
    img = np.zeros((size,size), dtype=np.uint8)
    F   = cv2.FONT_HERSHEY_SIMPLEX
    fs  = _rndf(rng, 0.9, 1.7)
    th  = _rnd(rng, 1, 4)
    s   = str(digit)
    ts  = cv2.getTextSize(s,F,fs,th)[0]
    xo  = max(1,min((size-ts[0])//2+_rnd(rng,-5,5),size-ts[0]-1))
    yo  = max(1,min((size+ts[1])//2+_rnd(rng,-5,5),size-1))
    cv2.putText(img,s,(xo,yo),F,fs,255,th,cv2.LINE_AA)
    return img


# ── AUGMENTATION ─────────────────────────────────────────────────
def augment(img, rng):
    """Rotation + mild perspective + noise. Keeps shapes recognisable."""
    h,w = img.shape

    # rotation
    angle = _rndf(rng, -18, 18)
    M     = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    img   = cv2.warpAffine(img, M, (w,h))

    # mild perspective warp
    mg  = int(w*0.08)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([
        [_rnd(rng,0,mg),   _rnd(rng,0,mg)],
        [w-_rnd(rng,0,mg), _rnd(rng,0,mg)],
        [w-_rnd(rng,0,mg), h-_rnd(rng,0,mg)],
        [_rnd(rng,0,mg),   h-_rnd(rng,0,mg)],
    ])
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src,dst), (w,h))

    # gaussian noise
    noise = rng.normal(0, _rndf(rng,3,10), img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16)+noise, 0, 255).astype(np.uint8)

    # random thickness variation
    op = _rnd(rng,0,2)
    ks = _rnd(rng,1,2)*2+1
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
    if   op==0: img = cv2.dilate(img,k)
    elif op==1: img = cv2.erode(img,k)
    return img


# ── DRAW DISPATCH ─────────────────────────────────────────────────
DRAW = {
    10: draw_plus,
    11: draw_minus,
    12: draw_multiply,
    13: draw_divide,
    14: draw_power,
    15: draw_lparen,
    16: draw_rparen,
    17: draw_equals,
}

def make_image(lbl, size, rng):
    """Make one training image for label lbl."""
    if lbl <= 9:
        img = draw_digit(lbl, size, rng)
    else:
        img = DRAW[lbl](size, rng)
    return img


# ══════════════════════════════════════════════════════════════════
#  2. DATASET
# ══════════════════════════════════════════════════════════════════

def load_mnist():
    print("[DATA] Loading MNIST ...")
    (a,b),(c,d) = mnist.load_data()
    x = np.concatenate([a,c]); y = np.concatenate([b,d])
    print(f"       {len(x):,} digit images")
    return x, y


def generate_all(n_digit=7000, n_symbol=15000):
    """
    Generate augmented training images for ALL 18 classes.
    Confused pairs get EXTRA samples:
      + and 7  -> 2x normal count  (most confused pair)
      - and =  -> 1.5x             (both horizontal lines)
      / and 1  -> 1.5x             (both diagonal/vertical)
    """
    SIZE = 56
    rng  = np.random.default_rng(SEED)
    imgs, labels = [], []

    # digits — how many extra samples per digit
    DIGIT_BOOST = {
        6: 2.0,   # 6 confused with 0
        7: 3.0,   # 7 confused with +
        9: 2.0,   # 9 confused with 0
    }
    print(f"[DATA] Generating digits ({n_digit:,}/class, 6/7/9 boosted) ...")
    for d in range(10):
        count = int(n_digit * DIGIT_BOOST.get(d, 1.0))
        for _ in range(count):
            raw   = make_image(d, SIZE, rng)
            aug   = augment(raw, rng)
            img28 = cv2.resize(aug,(28,28),interpolation=cv2.INTER_AREA)
            imgs.append(img28); labels.append(d)

    # symbols — boost the most confused ones
    SYMBOL_BOOST = {
        10: 3.0,   # +  gets 3x (most confused with 7)
        11: 1.5,   # -  gets 1.5x
        13: 1.5,   # /  gets 1.5x
        15: 3.0,   # (  gets 3x (was reading as 1)
        16: 2.0,   # )  gets 2x
        17: 1.5,   # =  gets 1.5x
    }
    print(f"[DATA] Generating symbols ({n_symbol:,}/class, + gets 3x) ...")
    for lbl in range(10, NUM_CLASSES):
        count = int(n_symbol * SYMBOL_BOOST.get(lbl, 1.0))
        for _ in range(count):
            raw   = make_image(lbl, SIZE, rng)
            aug   = augment(raw, rng)
            img28 = cv2.resize(aug,(28,28),interpolation=cv2.INTER_AREA)
            imgs.append(img28); labels.append(lbl)

    imgs   = np.array(imgs,   dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    print(f"       {len(imgs):,} total generated images")
    return imgs, labels


def clean_and_split(x, y):
    print("[DATA] Cleaning ...")
    mask = x.mean(axis=(1,2)) >= 3
    x,y  = x[mask], y[mask]
    x    = np.clip(x,0,255).astype(np.float32)/255.0
    x    = x.reshape(-1,28,28,1)
    print(f"       {len(x):,} samples kept")

    print("[DATA] Splitting 70/15/15 ...")
    x1,xt,y1,yt = train_test_split(x,y,test_size=0.30,stratify=y,random_state=SEED)
    xv,x2,yv,y2 = train_test_split(xt,yt,test_size=0.50,stratify=yt,random_state=SEED)
    print(f"       train={len(x1):,}  val={len(xv):,}  test={len(x2):,}")
    return (x1, to_categorical(y1,NUM_CLASSES), y1,
            xv, to_categorical(yv,NUM_CLASSES), yv,
            x2, to_categorical(y2,NUM_CLASSES), y2)


# ══════════════════════════════════════════════════════════════════
#  3. MODEL
# ══════════════════════════════════════════════════════════════════

def build_model():
    inp = layers.Input(shape=(28,28,1))
    x   = inp

    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters,3,padding="same",activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters,3,padding="same",activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)

    x   = layers.Flatten()(x)
    x   = layers.Dense(512,activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(256,activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(NUM_CLASSES,activation="softmax")(x)

    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def get_class_weights(y_raw):
    from collections import Counter
    counts  = Counter(y_raw.tolist())
    total   = len(y_raw)
    # per-class manual boost for the hardest pairs
    BOOST = {
        6:  2.0,   # digit 6  — confused with 0
        7:  3.0,   # digit 7  — confused with +
        9:  2.0,   # digit 9  — confused with 0
        10: 3.0,   # +        — confused with 7
        11: 2.0,   # -        — confused with =
        13: 2.0,   # /        — confused with 1
        15: 4.0,   # (        — was reading as 1
        16: 3.0,   # )        — needs boost
        17: 2.0,   # =        — confused with -
    }
    weights = {}
    for cls in range(NUM_CLASSES):
        cnt = counts.get(cls,1)
        w   = total / (NUM_CLASSES * cnt)
        if cls >= 10: w *= 2.0          # all symbols get base boost
        w *= BOOST.get(cls, 1.0)        # extra for confused pairs
        weights[cls] = w
    return weights


def train_model(model, x_tr, y_tr, y_tr_raw, xv, yv):
    print("[TRAIN] Starting ...")
    model.summary()
    cw  = get_class_weights(y_tr_raw)
    gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        shear_range=5,
    )
    cbs = [
        callbacks.EarlyStopping(patience=7, restore_best_weights=True,
                                monitor="val_accuracy", verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1,
                                    monitor="val_loss", min_lr=1e-6),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True,
                                  monitor="val_accuracy", verbose=1),
    ]
    return model.fit(
        gen.flow(x_tr, y_tr, batch_size=128),
        steps_per_epoch=len(x_tr)//128,
        validation_data=(xv,yv),
        epochs=40,
        callbacks=cbs,
        class_weight=cw,
        verbose=1
    )


def evaluate_model(model, history, x_te, y_te_oh, y_te_raw):
    loss,acc = model.evaluate(x_te,y_te_oh,verbose=0)
    print(f"\n  Test Accuracy: {acc*100:.2f}%  Loss: {loss:.4f}\n")
    y_pred = np.argmax(model.predict(x_te,verbose=0),axis=1)
    labels = [LABEL_MAP[i] for i in range(NUM_CLASSES)]
    print(classification_report(y_te_raw, y_pred, target_names=labels))

    fig,ax = plt.subplots(1,2,figsize=(14,5))
    ax[0].plot(history.history["accuracy"],    label="Train")
    ax[0].plot(history.history["val_accuracy"],label="Val")
    ax[0].set_title("Accuracy"); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(history.history["loss"],    label="Train")
    ax[1].plot(history.history["val_loss"],label="Val")
    ax[1].set_title("Loss"); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png",dpi=150); plt.close()

    cm = confusion_matrix(y_te_raw,y_pred)
    fig,ax = plt.subplots(figsize=(13,11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png",dpi=150); plt.close()
    print("  Plots saved -> outputs/")


# ══════════════════════════════════════════════════════════════════
#  4. PREPROCESSING & SEGMENTATION
# ══════════════════════════════════════════════════════════════════

def to_binary(src):
    """Load image -> binary (white strokes, black background)."""
    if isinstance(src, str):
        img = cv2.imread(src)
        if img is None: raise FileNotFoundError(src)
    else:
        img = src.copy()

    gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img.copy()
    display = gray.copy()

    h,w = gray.shape
    if h < 100:
        s    = 100/h
        gray = cv2.resize(gray,(int(w*s),100),interpolation=cv2.INTER_CUBIC)

    gray = cv2.GaussianBlur(gray,(3,3),0)

    # auto-detect: bright background (paper) needs inverting
    if gray.mean() > 127:
        binary = cv2.adaptiveThreshold(gray,255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,15,8)
    else:
        _, binary = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

    k      = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,k)
    return binary, display


def segment(binary):
    """Find individual symbol boxes. Only merge physically overlapping boxes."""
    contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []

    H,W   = binary.shape
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < H*W*0.0005: continue
        if w > W*0.90:       continue
        boxes.append([x,y,w,h])
    if not boxes: return []

    boxes  = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]

    for b in boxes[1:]:
        p   = merged[-1]
        gap = b[0] - (p[0]+p[2])

        # '=' detection: two short wide strokes stacked
        both_wide  = p[2]>W*0.04 and b[2]>W*0.04
        both_short = p[3]<H*0.45 and b[3]<H*0.45
        v_gap      = b[1]-(p[1]+p[3])
        eq_like    = (both_wide and both_short
                      and abs(gap)<p[2]*0.8 and 0<v_gap<H*0.30)

        if gap <= 2 or eq_like:
            nx  = min(p[0],b[0]); ny  = min(p[1],b[1])
            nx2 = max(p[0]+p[2],b[0]+b[2])
            ny2 = max(p[1]+p[3],b[1]+b[3])
            merged[-1] = [nx,ny,nx2-nx,ny2-ny]
        else:
            merged.append(list(b))

    return merged


# ══════════════════════════════════════════════════════════════════
#  5. CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

def classify(model, binary, boxes, pad=6):
    H,W   = binary.shape
    preds = []

    for x,y,w,h in boxes:
        x1 = max(0,x-pad); y1 = max(0,y-pad)
        x2 = min(W,x+w+pad); y2 = min(H,y+h+pad)
        crop = binary[y1:y2,x1:x2]
        if crop.size==0: continue

        ch,cw = crop.shape
        side  = max(ch,cw)
        sq    = np.zeros((side,side),dtype=np.uint8)
        sq[(side-ch)//2:(side-ch)//2+ch,
           (side-cw)//2:(side-cw)//2+cw] = crop
        img28 = cv2.resize(sq,(28,28),interpolation=cv2.INTER_AREA)
        probs = model.predict(img28.reshape(1,28,28,1).astype("float32")/255,
                              verbose=0)[0]

        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        label = LABEL_MAP[idx]

        # aspect-ratio hint for low-confidence predictions
        if conf < 0.65:
            aspect = w/max(h,1)
            top3   = np.argsort(probs)[::-1][:3]
            if aspect > 2.5:               # very wide -> - or =
                for i in top3:
                    if LABEL_MAP[i] in ("-","="):
                        idx,conf,label=i,float(probs[i]),LABEL_MAP[i]; break
            elif aspect < 0.55:            # tall & narrow -> ( or )
                for i in top3:
                    if LABEL_MAP[i] in ("(",")",'1'):
                        # pick ( or ) over 1 if they appear in top3
                        if LABEL_MAP[i] in ("(",")" ):
                            idx,conf,label=i,float(probs[i]),LABEL_MAP[i]; break
            elif 0.65 < aspect < 1.5:      # squarish -> maybe x
                for i in top3:
                    if LABEL_MAP[i]=="x" and probs[i]>0.20:
                        idx,conf,label=i,float(probs[i]),LABEL_MAP[i]; break

        preds.append((label,conf))
    return preds


# ══════════════════════════════════════════════════════════════════
#  6. EXPRESSION BUILDER & SOLVER
# ══════════════════════════════════════════════════════════════════

def build_expr(preds):
    tokens = [OP_MAP.get(lbl,lbl) for lbl,_ in preds]
    parts  = []; i=0
    while i < len(tokens):
        t = tokens[i]
        if t.isdigit():
            num=t
            while i+1<len(tokens) and tokens[i+1].isdigit():
                i+=1; num+=tokens[i]
            parts.append(num)
        elif t=="=":
            pass
        else:
            parts.append(t)
        i+=1

    while parts and parts[0]  in OPERATORS: parts.pop(0)
    while parts and parts[-1] in OPERATORS: parts.pop()
    return " ".join(parts)


def safe_eval(expr):
    e = expr.strip()
    if not e:
        return None,"Empty — no symbols recognised"
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.\*\*]+",e):
        return None,f"Unrecognised token in: '{e}'"
    try:
        result = eval(e)
        if isinstance(result,float) and result.is_integer():
            result=int(result)
        return result,None
    except ZeroDivisionError: return None,"Division by zero"
    except Exception as ex:   return None,str(ex)


# ══════════════════════════════════════════════════════════════════
#  7. SIMPLE EXPLANATION  (no API)
# ══════════════════════════════════════════════════════════════════

def explain(expression, result, error=None):
    if error:
        print(f"\n  Could not solve: {error}"); return
    print(f"\n  Result: {expression} = {result}")
    step=1
    if "(" in expression:
        m=re.search(r'\(([^)]+)\)',expression)
        if m:
            v=safe_eval(m.group(1))[0]
            print(f"  Step {step}: Brackets first -> {m.group(1)} = {v}"); step+=1
    if "**" in expression:
        print(f"  Step {step}: Evaluate powers"); step+=1
    if re.search(r'[\*/]',expression.replace("**","  ")):
        print(f"  Step {step}: Multiply/divide left to right"); step+=1
    if re.search(r'[\+\-]',expression):
        print(f"  Step {step}: Add/subtract left to right"); step+=1
    print(f"  Answer: {expression} = {result}")


# ══════════════════════════════════════════════════════════════════
#  8. VISUALISER
# ══════════════════════════════════════════════════════════════════

def show_result(display,binary,boxes,preds,expr,result,error):
    fig=plt.figure(figsize=(16,8),facecolor="#0d1117")
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.35)

    def dax(ax,title):
        ax.set_facecolor("#161b22")
        ax.set_title(title,color="#c9d1d9",fontsize=11,fontweight="bold",pad=8)
        ax.axis("off"); return ax

    dax(fig.add_subplot(gs[0,0]),"Input").imshow(display,cmap="gray")

    ax2=dax(fig.add_subplot(gs[0,1]),"Segmented")
    vis=cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
    COLS=[(0,255,128),(0,180,255),(255,180,0),(255,80,80),(180,80,255)]
    for i,(x,y,w,h) in enumerate(boxes):
        cv2.rectangle(vis,(x,y),(x+w,y+h),COLS[i%5],2)
    ax2.imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB))

    ax3=dax(fig.add_subplot(gs[0,2]),"Predictions")
    if preds:
        tbl=ax3.table(
            cellText=[[str(i+1),l,f"{c*100:.0f}%"] for i,(l,c) in enumerate(preds)],
            colLabels=["#","Symbol","Conf"],loc="center",cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1.2,1.8)
        for (r,c),cell in tbl.get_celld().items():
            cell.set_facecolor("#21262d" if r==0 else "#161b22")
            cell.set_text_props(color="#c9d1d9"); cell.set_edgecolor("#30363d")

    ax4=fig.add_subplot(gs[1,:]); ax4.set_facecolor("#161b22"); ax4.axis("off")
    txt=f"Error: {error}" if error else f"{expr}  =  {result}"
    col="#f85149" if error else "#3fb950"
    ax4.text(0.5,0.6,"RESULT",ha="center",va="center",fontsize=13,
             color="#8b949e",transform=ax4.transAxes)
    ax4.text(0.5,0.28,txt,ha="center",va="center",fontsize=38,
             color=col,fontweight="bold",transform=ax4.transAxes,
             bbox=dict(boxstyle="round,pad=0.5",facecolor="#0d1117",
                       edgecolor=col,linewidth=2))
    fig.suptitle("Handwritten Math Recognizer",color="#c9d1d9",
                 fontsize=14,fontweight="bold",y=0.98)
    plt.savefig("outputs/result.png",dpi=150,bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  Saved -> outputs/result.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════
#  9. PIPELINE
# ══════════════════════════════════════════════════════════════════

def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        print(f"\nNo model at '{MODEL_PATH}'. Run --train first.\n")
        sys.exit(1)
    print("Loading model ...")
    m=tf.keras.models.load_model(MODEL_PATH)
    print("Model ready")
    return m


def pipeline(model,src,show=True):
    print("\n"+"="*55)
    binary,display = to_binary(src)
    boxes          = segment(binary)
    print(f"  Segmented: {len(boxes)} symbol(s)")

    if not boxes:
        print("  No symbols found"); return None

    preds=classify(model,binary,boxes)
    for i,(lbl,conf) in enumerate(preds):
        flag=" [low conf]" if conf<0.65 else ""
        print(f"  [{i+1}] '{lbl}'  {conf*100:.0f}%{flag}")

    expr=build_expr(preds)
    result,error=safe_eval(expr)
    print(f"\n  Expression: '{expr}'")
    if error: print(f"  Error: {error}")
    else:     print(f"  Result: {expr} = {result}")

    explain(expr,result,error)
    if show: show_result(display,binary,boxes,preds,expr,result,error)
    return expr,result


# ══════════════════════════════════════════════════════════════════
#  10. DRAW MODE
# ══════════════════════════════════════════════════════════════════

def draw_mode(model):
    print("\n  DRAW MODE  |  SPACE=solve  R=reset  Q=quit\n")
    CW,CH=640,160
    canvas=np.ones((CH,CW),dtype=np.uint8)*255
    drawing=[False]; last=[None]; answer=["Draw expression, then press SPACE"]

    cv2.namedWindow("Math Recognizer",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Math Recognizer",CW*2,(CH+60)*2)

    def mouse(ev,x,y,*_):
        if   ev==cv2.EVENT_LBUTTONDOWN: drawing[0]=True;last[0]=(x,y)
        elif ev==cv2.EVENT_MOUSEMOVE and drawing[0]:
            if last[0]: cv2.line(canvas,last[0],(x,y),0,8,cv2.LINE_AA)
            last[0]=(x,y)
        elif ev==cv2.EVENT_LBUTTONUP: drawing[0]=False;last[0]=None

    cv2.setMouseCallback("Math Recognizer",mouse)
    while True:
        disp=cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR)
        bar=np.zeros((60,CW,3),dtype=np.uint8)
        cv2.putText(bar,answer[0],(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,255,128),2)
        cv2.imshow("Math Recognizer",np.vstack([disp,bar]))
        key=cv2.waitKey(20)&0xFF

        if key==ord(" "):
            inv=cv2.bitwise_not(canvas)
            binary,_=to_binary(inv)
            boxes=segment(binary)
            if boxes:
                preds=classify(model,binary,boxes)
                expr=build_expr(preds)
                res,err=safe_eval(expr)
                answer[0]=f"Error: {err}" if err else f"{expr} = {res}"
                if not err: explain(expr,res)
            else:
                answer[0]="No symbols detected"
        elif key in (ord("r"),ord("R")): canvas[:]=255; answer[0]="Draw expression, then press SPACE"
        elif key in (ord("q"),ord("Q"),27): break
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════
#  11. AUTO TEST
#  Uses the SAME geometry drawing as training so the test
#  actually measures what the model learned.
# ══════════════════════════════════════════════════════════════════

def _render_test_image(expr_str):
    """Render test expression using geometry draw functions — same as training."""
    CHAR_MAP={str(i):i for i in range(10)}
    CHAR_MAP.update({"+":10,"-":11,"x":12,"/":13,"^":14,"(":15,")":16,"=":17})
    rng  = np.random.default_rng()
    cell = 52; pad=10
    chars=[c for c in expr_str]
    cells=[]
    for ch in chars:
        if ch not in CHAR_MAP: continue
        lbl=CHAR_MAP[ch]
        img=make_image(lbl,cell,rng)   # use same draw fn as training
        cells.append(img)
    if not cells: return None

    W=len(cells)*(cell+pad)+pad
    canvas=np.ones((cell+pad*2,W),dtype=np.uint8)*255
    for i,c in enumerate(cells):
        x=pad+i*(cell+pad); y=pad
        canvas[y:y+cell,x:x+cell]=255-c   # invert to dark-on-white
    return canvas


def auto_test(model):
    tests=[
        ("8+2",   10),
        ("12-4",   8),
        ("6x3",   18),
        ("8/2",    4),
        ("2^3",    8),
        ("9-3",    6),
        ("7+5",   12),
        ("4x4",   16),
    ]
    print("\n"+"="*55+"\n  AUTO TEST\n"+"="*55)
    correct=0
    for expr_str,expected in tests:
        canvas=_render_test_image(expr_str)
        if canvas is None:
            print(f"  XX  '{expr_str}'  -- render failed"); continue
        res=pipeline(model,canvas,show=False)
        if res:
            got_expr,got_result=res
            ok=(got_result==expected)
            correct+=int(ok)
            tag="OK" if ok else "XX"
            print(f"  {tag}  '{expr_str}'  expected={expected}  "
                  f"got={got_result}  expr='{got_expr}'")
        else:
            print(f"  XX  '{expr_str}'  -- no output")
    print(f"\n  Score: {correct}/{len(tests)}\n")


# ══════════════════════════════════════════════════════════════════
#  12. EDA
# ══════════════════════════════════════════════════════════════════

def run_eda():
    os.makedirs("outputs/eda",exist_ok=True)
    rng=np.random.default_rng(SEED)
    PALETTE=sns.color_palette("tab20",NUM_CLASSES)
    CLASSES=[LABEL_MAP[i] for i in range(NUM_CLASSES)]

    def save(name):
        plt.savefig(f"outputs/eda/{name}.png",dpi=150,bbox_inches="tight")
        plt.close(); print(f"  saved -> outputs/eda/{name}.png")

    print("\n[EDA] Generating sample data ...")
    xd,yd=load_mnist()
    xg,yg=generate_all(n_digit=500,n_symbol=500)
    X=np.concatenate([xd,xg]); Y=np.concatenate([yd,yg])
    print(f"      {len(X):,} total samples\n")

    # class distribution
    counts=[int(np.sum(Y==i)) for i in range(NUM_CLASSES)]
    fig,ax=plt.subplots(1,2,figsize=(16,5))
    fig.suptitle("Class Distribution",fontsize=14,fontweight="bold")
    bars=ax[0].bar(CLASSES,counts,color=PALETTE,edgecolor="black",lw=0.5)
    for b,c in zip(bars,counts):
        ax[0].text(b.get_x()+b.get_width()/2,b.get_height()+50,
                   f"{c:,}",ha="center",va="bottom",fontsize=7)
    ax[0].grid(axis="y",alpha=0.4)
    ax[1].pie([int(np.sum(Y<10)),int(np.sum(Y>=10))],
              labels=[f"Digits\n{int(np.sum(Y<10)):,}",
                      f"Symbols\n{int(np.sum(Y>=10)):,}"],
              colors=["#4C9BE8","#F28C38"],autopct="%1.1f%%",
              wedgeprops={"edgecolor":"white","linewidth":2})
    plt.tight_layout(); save("01_class_distribution")

    # sample grid
    N=5; fig,axes=plt.subplots(NUM_CLASSES,N,figsize=(N*1.8,NUM_CLASSES*1.8))
    fig.suptitle("Samples per Class",fontsize=14,fontweight="bold",y=1.01)
    for cls in range(NUM_CLASSES):
        idx=np.where(Y==cls)[0]
        chosen=rng.choice(idx,min(N,len(idx)),replace=False)
        for col,ii in enumerate(chosen):
            a=axes[cls,col]; a.imshow(X[ii],cmap="gray"); a.axis("off")
            if col==0: a.set_ylabel(LABEL_MAP[cls],fontsize=12,rotation=0,
                                    labelpad=20,va="center",fontweight="bold",
                                    color=PALETTE[cls])
    plt.tight_layout(); save("02_sample_grid")
    print("\n  EDA done -> outputs/eda/\n")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--eda",  action="store_true")
    p.add_argument("--solve",nargs="?",const="__draw__",metavar="IMAGE")
    args=p.parse_args()

    if args.train:
        print("="*55+"\n  TRAINING\n"+"="*55)
        xd,yd = load_mnist()
        xg,yg = generate_all(n_digit=7000, n_symbol=15000)
        X = np.concatenate([xd,xg]); Y = np.concatenate([yd,yg])
        print(f"  Total: {len(X):,} samples | {NUM_CLASSES} classes\n")
        x_tr,y_tr,y_tr_r, xv,yv,_, x_te,y_te,y_te_r = clean_and_split(X,Y)
        m=build_model()
        h=train_model(m,x_tr,y_tr,y_tr_r,xv,yv)
        evaluate_model(m,h,x_te,y_te,y_te_r)
        with open(LABEL_PATH,"w") as f: json.dump(LABEL_MAP,f,indent=2)
        print(f"\n  Model saved -> {MODEL_PATH}")
        print("  Run: python math_recognizer.py --test\n")

    elif args.test:
        auto_test(load_model_safe())

    elif args.eda:
        run_eda()

    elif args.solve is not None:
        m=load_model_safe()
        if args.solve=="__draw__": draw_mode(m)
        else: pipeline(m,args.solve,show=True)

    else:
        p.print_help()
        print("\n  python math_recognizer.py --train")
        print("  python math_recognizer.py --test")
        print("  python math_recognizer.py --solve")
        print("  python math_recognizer.py --solve image.png\n")


if __name__=="__main__":
    main()
