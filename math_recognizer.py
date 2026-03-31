"""Handwritten Math Expression Recognizer
Usage:
  python math_recognizer.py --train
  python math_recognizer.py --test
  python math_recognizer.py --solve
  python math_recognizer.py --solve image.png
  python math_recognizer.py --eda"""

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
#  1. DATA
# ══════════════════════════════════════════════════════════════════

def load_mnist():
    print("[DATA] Loading MNIST …")
    (a,b),(c,d) = mnist.load_data()
    x = np.concatenate([a,c]); y = np.concatenate([b,d])
    print(f"       {len(x):,} digit images")
    return x, y


def make_symbol_image(sym, size=56):
    img = np.zeros((size,size), dtype=np.uint8)
    rng = np.random.default_rng()
    fs  = rng.uniform(0.9, 1.8)
    th  = int(rng.integers(1, 4))
    F   = cv2.FONT_HERSHEY_SIMPLEX
    ts  = cv2.getTextSize(sym, F, fs, th)[0]
    xo  = max(2, int((size-ts[0])/2) + rng.integers(-4,5))
    yo  = max(2, int((size+ts[1])/2) + rng.integers(-4,5))
    cv2.putText(img, sym, (xo,yo), F, fs, 255, th, cv2.LINE_AA)
    return img


def augment(img, rng):
    h,w = img.shape
    # rotation
    M   = cv2.getRotationMatrix2D((w/2,h/2), rng.uniform(-35,35), 1.0)
    img = cv2.warpAffine(img, M, (w,h))
    # perspective
    mg  = int(w*0.12)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([
        [rng.integers(0,mg),   rng.integers(0,mg)],
        [w-rng.integers(0,mg), rng.integers(0,mg)],
        [w-rng.integers(0,mg), h-rng.integers(0,mg)],
        [rng.integers(0,mg),   h-rng.integers(0,mg)],
    ])
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src,dst), (w,h))
    # elastic
    s  = rng.uniform(0,3)
    dx = cv2.GaussianBlur(rng.uniform(-1,1,(h,w)).astype(np.float32),(7,7),0)*s
    dy = cv2.GaussianBlur(rng.uniform(-1,1,(h,w)).astype(np.float32),(7,7),0)*s
    gx,gy = np.meshgrid(np.arange(w),np.arange(h))
    img = cv2.remap(img,
                    np.clip(gx+dx,0,w-1).astype(np.float32),
                    np.clip(gy+dy,0,h-1).astype(np.float32),
                    cv2.INTER_LINEAR)
    # noise
    noise = rng.normal(0, rng.uniform(5,18), img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16)+noise, 0, 255).astype(np.uint8)
    # morph
    ks = int(rng.integers(1,3))*2+1
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
    op = rng.integers(3)
    if   op==0: img = cv2.dilate(img,k)
    elif op==1: img = cv2.erode(img,k)
    return img


def generate_symbols(n=15000):
    # x gets 3 drawing styles so it is never confused with digits
    SYMS = [("+",10,[]), ("-",11,[]), ("x",12,["x","X"]),
            ("/",13,[]), ("^",14,[]), ("(",15,[]), (")",16,[]), ("=",17,[])]
    rng  = np.random.default_rng(SEED)
    imgs, labels = [], []
    print(f"[DATA] Generating symbols ({n:,} per class) …")
    for sym, lbl, extras in SYMS:
        styles    = [sym]+extras
        per_style = n // len(styles)
        for style in styles:
            count = per_style if style!=styles[-1] else n-per_style*(len(styles)-1)
            for _ in range(count):
                raw   = make_symbol_image(style)
                aug   = augment(raw, rng)
                img28 = cv2.resize(aug,(28,28),interpolation=cv2.INTER_AREA)
                imgs.append(img28); labels.append(lbl)
    imgs   = np.array(imgs,   dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    print(f"       {len(imgs):,} symbol images")
    return imgs, labels


def clean_and_split(x, y):
    print("[DATA] Cleaning …")
    mask = x.mean(axis=(1,2)) >= 4
    x,y  = x[mask], y[mask]
    x    = np.clip(x,0,255).astype(np.float32)/255.0
    x    = x.reshape(-1,28,28,1)
    print(f"       {len(x):,} samples after cleaning")

    print("[DATA] Splitting 70/15/15 …")
    x1,xt,y1,yt = train_test_split(x,y,test_size=0.30,stratify=y,random_state=SEED)
    xv,x2,yv,y2 = train_test_split(xt,yt,test_size=0.50,stratify=yt,random_state=SEED)
    print(f"       train={len(x1):,}  val={len(xv):,}  test={len(x2):,}")
    return (x1, to_categorical(y1,NUM_CLASSES), y1,
            xv, to_categorical(yv,NUM_CLASSES), yv,
            x2, to_categorical(y2,NUM_CLASSES), y2)


# ══════════════════════════════════════════════════════════════════
#  2. MODEL
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
    weights = {}
    for cls in range(NUM_CLASSES):
        cnt = counts.get(cls,1)
        w   = (total / (NUM_CLASSES * cnt))
        if cls >= 10: w *= 2.0   # extra weight for symbol classes
        weights[cls] = w
    return weights


def train_model(model, x_tr, y_tr, y_tr_raw, xv, yv):
    print("[TRAIN] Starting …")
    model.summary()
    cw  = get_class_weights(y_tr_raw)
    gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.12,
                             height_shift_range=0.12, zoom_range=0.12, shear_range=5)
    cbs = [
        callbacks.EarlyStopping(patience=6, restore_best_weights=True,
                                monitor="val_accuracy", verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1, min_lr=1e-6),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True,
                                  monitor="val_accuracy", verbose=1),
    ]
    return model.fit(gen.flow(x_tr,y_tr,batch_size=128),
                     steps_per_epoch=len(x_tr)//128,
                     validation_data=(xv,yv),
                     epochs=40, callbacks=cbs,
                     class_weight=cw, verbose=1)


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

    cm  = confusion_matrix(y_te_raw,y_pred)
    fig,ax = plt.subplots(figsize=(13,11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png",dpi=150); plt.close()
    print("  Plots → outputs/")


# ══════════════════════════════════════════════════════════════════
#  3. PREPROCESSING & SEGMENTATION
# ══════════════════════════════════════════════════════════════════

def to_binary(src):
    """
    Accept file path or numpy array.
    Always returns: binary with WHITE strokes on BLACK background.
    Auto-detects whether image needs inverting.
    """
    if isinstance(src, str):
        img = cv2.imread(src)
        if img is None: raise FileNotFoundError(src)
    else:
        img = src.copy()

    gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img.copy()
    display = gray.copy()

    # upscale tiny images
    h,w = gray.shape
    if h < 100:
        s    = 100/h
        gray = cv2.resize(gray,(int(w*s),100),interpolation=cv2.INTER_CUBIC)

    gray = cv2.GaussianBlur(gray,(3,3),0)

    # auto-invert: dark ink on bright paper → invert; already dark bg → keep
    if gray.mean() > 127:
        binary = cv2.adaptiveThreshold(gray,255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 8)
    else:
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    k      = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    return binary, display


def segment(binary):
    """
    Find individual symbol bounding boxes.

    KEY FIX: only merge boxes that physically OVERLAP (gap <= 2 px).
    The old code merged boxes within avg_width*0.35 which caused
    '3','7','2' to collapse into a single '372' box.

    Special case: '=' sign is two short horizontal strokes that sit
    close vertically — those ARE merged.
    """
    contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []

    H,W   = binary.shape
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < H*W*0.0005: continue   # noise
        if w > W*0.90:       continue   # full-width artifact
        boxes.append([x,y,w,h])
    if not boxes: return []

    boxes  = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]

    for b in boxes[1:]:
        p    = merged[-1]
        gap  = b[0] - (p[0]+p[2])           # >0 = space between, <0 = overlap

        # '=' detection: two wide short strokes stacked vertically
        both_wide  = p[2] > W*0.04 and b[2] > W*0.04
        both_short = p[3] < H*0.45 and b[3] < H*0.45
        v_gap      = b[1] - (p[1]+p[3])
        eq_like    = both_wide and both_short and abs(gap) < p[2]*0.8 and 0 < v_gap < H*0.3

        if gap <= 2 or eq_like:
            # merge
            nx  = min(p[0],b[0]); ny  = min(p[1],b[1])
            nx2 = max(p[0]+p[2],b[0]+b[2])
            ny2 = max(p[1]+p[3],b[1]+b[3])
            merged[-1] = [nx,ny,nx2-nx,ny2-ny]
        else:
            merged.append(list(b))

    return merged


# ══════════════════════════════════════════════════════════════════
#  4. CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

def classify(model, binary, boxes, pad=6):
    H,W   = binary.shape
    preds = []

    for x,y,w,h in boxes:
        x1 = max(0,x-pad); y1 = max(0,y-pad)
        x2 = min(W,x+w+pad); y2 = min(H,y+h+pad)
        crop = binary[y1:y2, x1:x2]
        if crop.size == 0: continue

        # make square
        ch,cw = crop.shape
        side  = max(ch,cw)
        sq    = np.zeros((side,side), dtype=np.uint8)
        sq[(side-ch)//2:(side-ch)//2+ch,
           (side-cw)//2:(side-cw)//2+cw] = crop

        img28 = cv2.resize(sq,(28,28),interpolation=cv2.INTER_AREA)
        probs = model.predict(img28.reshape(1,28,28,1).astype("float32")/255,
                              verbose=0)[0]

        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        label = LABEL_MAP[idx]

        # aspect-ratio hint when confidence is low
        if conf < 0.65:
            aspect = w / max(h,1)
            top3   = np.argsort(probs)[::-1][:3]
            if aspect > 2.5:                      # wide → - or =
                for i in top3:
                    if LABEL_MAP[i] in ("-","="):
                        idx,conf,label = i,float(probs[i]),LABEL_MAP[i]; break
            elif 0.65 < aspect < 1.5:             # squarish → maybe x
                for i in top3:
                    if LABEL_MAP[i]=="x" and probs[i]>0.20:
                        idx,conf,label = i,float(probs[i]),LABEL_MAP[i]; break

        preds.append((label,conf))
    return preds


# ══════════════════════════════════════════════════════════════════
#  5. EXPRESSION BUILDER & SOLVER
# ══════════════════════════════════════════════════════════════════

def build_expr(preds):
    """
    CNN labels → Python expression string.

    Steps:
      1. Map display symbols to Python operators  (x→*  ^→**)
      2. Merge consecutive digit tokens into numbers  (3,7,2 → 372)
      3. Drop '=' tokens
      4. Strip leading operators   (misread first symbol)
      5. Strip trailing operators  (was causing '372 /' EOF crash)
    """
    tokens = [OP_MAP.get(lbl, lbl) for lbl,_ in preds]

    parts = []; i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.isdigit():
            num = t
            while i+1 < len(tokens) and tokens[i+1].isdigit():
                i += 1; num += tokens[i]
            parts.append(num)
        elif t == "=":
            pass                   # strip equals sign
        else:
            parts.append(t)
        i += 1

    # strip leading operators
    while parts and parts[0] in OPERATORS:
        parts.pop(0)

    # strip trailing operators  ← THIS fixes the '372 /' EOF crash
    while parts and parts[-1] in OPERATORS:
        parts.pop()

    return " ".join(parts)


def safe_eval(expr):
    """Safely evaluate a whitelisted math expression."""
    e = expr.strip()
    if not e:
        return None, "Empty expression — no symbols were recognised"
    # whitelist: digits, spaces, operators, parens, dots
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.\*\*]+", e):
        return None, f"Unrecognised characters in: '{e}'"
    try:
        result = eval(e)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return result, None
    except ZeroDivisionError:
        return None, "Division by zero"
    except Exception as ex:
        return None, str(ex)


# ══════════════════════════════════════════════════════════════════
#  6. EXPLANATION  (local only — no API needed)
# ══════════════════════════════════════════════════════════════════

def explain(expression, result, error=None):
    if error:
        print(f"\n  ⚠️   Could not solve: {error}"); return
    lines = [f"\n  📖  {expression} = {result}", ""]
    step  = 1

    if "(" in expression:
        m = re.search(r'\(([^)]+)\)', expression)
        if m:
            v = safe_eval(m.group(1))[0]
            lines.append(f"  Step {step}: Solve brackets → {m.group(1)} = {v}"); step+=1

    if "**" in expression:
        lines.append(f"  Step {step}: Evaluate powers (^)"); step+=1

    muls = re.findall(r'\d+\s*[\*/]\s*\d+', expression.replace("**","  "))
    if muls:
        lines.append(f"  Step {step}: Multiply / divide left to right"); step+=1

    adds = re.findall(r'\d+\s*[\+\-]\s*\d+', expression)
    if adds:
        lines.append(f"  Step {step}: Add / subtract left to right"); step+=1

    lines.append(f"\n  ✅  Answer: {expression} = {result}")
    lines.append("  Well done!")
    print("\n".join(lines))


# ══════════════════════════════════════════════════════════════════
#  7. VISUALISER
# ══════════════════════════════════════════════════════════════════

def show_result(display, binary, boxes, preds, expr, result, error):
    fig = plt.figure(figsize=(16,8), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2,3, figure=fig, hspace=0.5, wspace=0.35)

    def dax(ax, title):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="#c9d1d9", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off"); return ax

    dax(fig.add_subplot(gs[0,0]), "① Input").imshow(display, cmap="gray")

    ax2  = dax(fig.add_subplot(gs[0,1]), "② Segmented")
    vis  = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    COLS = [(0,255,128),(0,180,255),(255,180,0),(255,80,80),(180,80,255)]
    for i,(x,y,w,h) in enumerate(boxes):
        cv2.rectangle(vis,(x,y),(x+w,y+h), COLS[i%5], 2)
    ax2.imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB))

    ax3  = dax(fig.add_subplot(gs[0,2]), "③ Predictions")
    if preds:
        tbl = ax3.table(
            cellText  = [[str(i+1),l,f"{c*100:.0f}%"] for i,(l,c) in enumerate(preds)],
            colLabels = ["#","Symbol","Conf"],
            loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1.2,1.8)
        for (r,c),cell in tbl.get_celld().items():
            cell.set_facecolor("#21262d" if r==0 else "#161b22")
            cell.set_text_props(color="#c9d1d9"); cell.set_edgecolor("#30363d")

    ax4  = fig.add_subplot(gs[1,:]); ax4.set_facecolor("#161b22"); ax4.axis("off")
    txt  = f"⚠  {error}" if error else f"{expr}  =  {result}"
    col  = "#f85149"      if error else "#3fb950"
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
    print("  📊  Saved → outputs/result.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════
#  8. PIPELINE
# ══════════════════════════════════════════════════════════════════

def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌  No model at '{MODEL_PATH}'\n   Run: python math_recognizer.py --train\n")
        sys.exit(1)
    print("⏳  Loading model …")
    m = tf.keras.models.load_model(MODEL_PATH)
    print("✅  Model ready")
    return m


def pipeline(model, src, show=True):
    print("\n" + "═"*55)
    binary, display = to_binary(src)
    boxes           = segment(binary)
    print(f"  Segmented: {len(boxes)} symbol(s)")

    if not boxes:
        print("  ❌  No symbols found — check image quality")
        return None

    preds = classify(model, binary, boxes)
    for i,(lbl,conf) in enumerate(preds):
        flag = "  ⚠ low conf" if conf < 0.65 else ""
        print(f"  [{i+1}] '{lbl}'  {conf*100:.0f}%{flag}")

    expr          = build_expr(preds)
    result, error = safe_eval(expr)

    print(f"\n  Expression : '{expr}'")
    if error: print(f"  ❌  {error}")
    else:     print(f"  ✅  {expr} = {result}")

    explain(expr, result, error)

    if show:
        show_result(display, binary, boxes, preds, expr, result, error)

    return expr, result


# ══════════════════════════════════════════════════════════════════
#  9. DRAW MODE
# ══════════════════════════════════════════════════════════════════

def draw_mode(model):
    print("\n  DRAW MODE  |  SPACE=solve  R=reset  Q=quit\n")
    CW,CH   = 640,160
    canvas  = np.ones((CH,CW),dtype=np.uint8)*255
    drawing = [False]; last=[None]; answer=["Draw, then press SPACE"]

    cv2.namedWindow("Math Recognizer",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Math Recognizer",CW*2,(CH+60)*2)

    def mouse(ev,x,y,*_):
        if   ev==cv2.EVENT_LBUTTONDOWN: drawing[0]=True;  last[0]=(x,y)
        elif ev==cv2.EVENT_MOUSEMOVE and drawing[0]:
            if last[0]: cv2.line(canvas,last[0],(x,y),0,8,cv2.LINE_AA)
            last[0]=(x,y)
        elif ev==cv2.EVENT_LBUTTONUP: drawing[0]=False; last[0]=None

    cv2.setMouseCallback("Math Recognizer",mouse)
    while True:
        disp = cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR)
        bar  = np.zeros((60,CW,3),dtype=np.uint8)
        cv2.putText(bar,answer[0],(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,255,128),2)
        cv2.imshow("Math Recognizer",np.vstack([disp,bar]))
        key = cv2.waitKey(20)&0xFF

        if key==ord(" "):
            # canvas = white bg + black strokes → invert → white strokes black bg
            inv    = cv2.bitwise_not(canvas)
            binary,_ = to_binary(inv)
            boxes  = segment(binary)
            if boxes:
                preds = classify(model,binary,boxes)
                expr  = build_expr(preds)
                res,err = safe_eval(expr)
                answer[0] = f"Error: {err}" if err else f"{expr} = {res}"
                if not err: explain(expr,res)
            else:
                answer[0] = "No symbols detected — try again"

        elif key in (ord("r"),ord("R")):
            canvas[:] = 255
            answer[0] = "Draw, then press SPACE"
        elif key in (ord("q"),ord("Q"),27):
            break

    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════
#  10. AUTO TEST
#  Tests use real OpenCV-rendered images (same font as training)
#  on white background — same as a real photo of handwriting.
# ══════════════════════════════════════════════════════════════════

def auto_test(model):
    F = cv2.FONT_HERSHEY_SIMPLEX
    tests = [
        ("8+2",    10),
        ("12-4",    8),
        ("6x3",    18),
        ("8/2",     4),
        ("2^3",     8),
        ("9-3",     6),
        ("7+5",    12),
        ("4x4",    16),
    ]
    print("\n"+"═"*55+"\n  AUTO TEST\n"+"═"*55)
    correct = 0
    for expr_str, expected in tests:
        canvas = np.ones((100,500),dtype=np.uint8)*255
        cv2.putText(canvas, expr_str, (20,75), F, 2.2, 0, 4, cv2.LINE_AA)
        res = pipeline(model, canvas, show=False)
        if res:
            got_expr, got_result = res
            ok = (got_result == expected)
            correct += int(ok)
            print(f"  {'✅' if ok else '❌'}  '{expr_str}'  "
                  f"expected={expected}  got={got_result}  (expr='{got_expr}')")
        else:
            print(f"  ❌  '{expr_str}'  — no output")
    print(f"\n  Score: {correct}/{len(tests)}\n")


# ══════════════════════════════════════════════════════════════════
#  11. EDA
# ══════════════════════════════════════════════════════════════════

def run_eda():
    from sklearn.manifold import TSNE
    os.makedirs("outputs/eda", exist_ok=True)
    rng     = np.random.default_rng(SEED)
    PALETTE = sns.color_palette("tab20", NUM_CLASSES)
    CLASSES = [LABEL_MAP[i] for i in range(NUM_CLASSES)]

    def save(name):
        plt.savefig(f"outputs/eda/{name}.png", dpi=150, bbox_inches="tight")
        plt.close(); print(f"  ✅  outputs/eda/{name}.png")

    print("\n[EDA] Loading data …")
    xd,yd = load_mnist()
    xs,ys = generate_symbols(n=2000)
    X = np.concatenate([xd,xs]); Y = np.concatenate([yd,ys])
    print(f"      {len(X):,} total\n")

    counts = [int(np.sum(Y==i)) for i in range(NUM_CLASSES)]
    fig,ax = plt.subplots(1,2,figsize=(16,5))
    fig.suptitle("Class Distribution",fontsize=14,fontweight="bold")
    bars = ax[0].bar(CLASSES,counts,color=PALETTE,edgecolor="black",lw=0.5)
    for b,c in zip(bars,counts):
        ax[0].text(b.get_x()+b.get_width()/2,b.get_height()+50,f"{c:,}",
                   ha="center",va="bottom",fontsize=7)
    ax[0].grid(axis="y",alpha=0.4)
    ax[1].pie([int(np.sum(Y<10)),int(np.sum(Y>=10))],
              labels=[f"Digits\n{int(np.sum(Y<10)):,}",
                      f"Symbols\n{int(np.sum(Y>=10)):,}"],
              colors=["#4C9BE8","#F28C38"],autopct="%1.1f%%",
              wedgeprops={"edgecolor":"white","linewidth":2})
    plt.tight_layout(); save("01_class_distribution")

    N=5; fig,axes=plt.subplots(NUM_CLASSES,N,figsize=(N*1.8,NUM_CLASSES*1.8))
    fig.suptitle("Sample Grid",fontsize=14,fontweight="bold",y=1.01)
    for cls in range(NUM_CLASSES):
        idx=np.where(Y==cls)[0]
        for col,ii in enumerate(rng.choice(idx,N,replace=False)):
            ax=axes[cls,col]; ax.imshow(X[ii],cmap="gray"); ax.axis("off")
            if col==0: ax.set_ylabel(LABEL_MAP[cls],fontsize=12,rotation=0,
                                     labelpad=20,va="center",fontweight="bold",
                                     color=PALETTE[cls])
    plt.tight_layout(); save("02_sample_grid")

    tx,ty=[],[]
    for cls in range(NUM_CLASSES):
        idx=np.where(Y==cls)[0]
        pick=rng.choice(idx,min(150,len(idx)),replace=False)
        tx.append(X[pick].reshape(len(pick),-1)); ty.extend([cls]*len(pick))
    tx=np.vstack(tx).astype(np.float32)/255; ty=np.array(ty)
    emb=TSNE(n_components=2,random_state=SEED,perplexity=25,verbose=0).fit_transform(tx)
    fig,ax=plt.subplots(figsize=(12,9))
    for cls in range(NUM_CLASSES):
        m=ty==cls
        ax.scatter(emb[m,0],emb[m,1],color=PALETTE[cls],
                   label=LABEL_MAP[cls],alpha=0.6,s=20,linewidths=0)
    ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",fontsize=10)
    ax.set_title("t-SNE Embedding",fontsize=13,fontweight="bold"); ax.grid(alpha=0.3)
    plt.tight_layout(); save("03_tsne")
    print("\n  EDA done → outputs/eda/\n")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--test",  action="store_true")
    p.add_argument("--eda",   action="store_true")
    p.add_argument("--solve", nargs="?", const="__draw__", metavar="IMAGE")
    args = p.parse_args()

    if args.train:
        print("="*55+"\n  TRAINING\n"+"="*55)
        xd,yd = load_mnist()
        xs,ys = generate_symbols(n=15000)
        X = np.concatenate([xd,xs]); Y = np.concatenate([yd,ys])
        print(f"  Total: {len(X):,} samples | {NUM_CLASSES} classes\n")
        x_tr,y_tr,y_tr_r, xv,yv,_, x_te,y_te,y_te_r = clean_and_split(X,Y)
        m = build_model()
        h = train_model(m, x_tr, y_tr, y_tr_r, xv, yv)
        evaluate_model(m, h, x_te, y_te, y_te_r)
        with open(LABEL_PATH,"w") as f: json.dump(LABEL_MAP,f,indent=2)
        print(f"\n  ✅  Saved → {MODEL_PATH}")
        print("  Now run: python math_recognizer.py --test\n")

    elif args.test:
        auto_test(load_model_safe())

    elif args.eda:
        run_eda()

    elif args.solve is not None:
        m = load_model_safe()
        if args.solve == "__draw__": draw_mode(m)
        else: pipeline(m, args.solve, show=True)

    else:
        p.print_help()
        print("\n  python math_recognizer.py --train")
        print("  python math_recognizer.py --test")
        print("  python math_recognizer.py --solve")
        print("  python math_recognizer.py --solve image.png\n")


if __name__ == "__main__":
    main()
