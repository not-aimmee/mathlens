"""
solve_expression.py
═══════════════════════════════════════════════════════════════
Takes a handwritten expression image → segments symbols →
classifies each with your CNN → evaluates the math → returns answer

Works on top of your existing model with ZERO changes.

Usage:
    python solve_expression.py                  ← test with synthetic image
    python solve_expression.py myexpr.png       ← your own image
    python solve_expression.py --draw           ← draw live, get answer
═══════════════════════════════════════════════════════════════
"""

import sys, os, json, re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.models import load_model

# ── Load model ───────────────────────────────────────────────
MODEL_PATH     = "models/digit_math_recognizer.keras"
LABEL_MAP_PATH = "models/label_map.json"

if not os.path.exists(MODEL_PATH):
    print("\n❌  Model not found. Run train.py first!\n")
    sys.exit(1)

model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH) as f:
    LABEL_MAP = {int(k): v for k, v in json.load(f).items()}

os.makedirs("outputs", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — PREPROCESS
#  Convert image to clean black-on-white binary
# ═══════════════════════════════════════════════════════════════
def preprocess_image(img_input):
    """
    Accepts:
      - file path (str)
      - numpy array (gray or BGR)
    Returns:
      - binary image (white strokes on black background, uint8)
      - original gray image for display
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {img_input}")
    else:
        img = img_input.copy()

    # to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    display_gray = gray.copy()

    # resize if too small (minimum 100px tall for good segmentation)
    h, w = gray.shape
    if h < 100:
        scale = 100 / h
        gray  = cv2.resize(gray, (int(w*scale), 100), interpolation=cv2.INTER_CUBIC)

    # denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # adaptive threshold → binary
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,   # white strokes on black
        blockSize=15, C=8
    )

    # morphological clean-up (remove tiny noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary, display_gray


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — SEGMENTATION
#  Find each symbol as a separate bounding box
# ═══════════════════════════════════════════════════════════════
def segment_symbols(binary):
    """
    Uses connected-component analysis to find individual symbols.
    Returns list of (x, y, w, h) bounding boxes sorted left→right.
    Merges boxes that are very close together (e.g. = sign two lines).
    """
    # find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    boxes = []
    img_h, img_w = binary.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # filter out tiny noise (less than 1% of image area)
        if w * h < (img_h * img_w * 0.001):
            continue
        # filter out full-image spanning boxes
        if w > img_w * 0.95:
            continue
        boxes.append([x, y, w, h])

    if not boxes:
        return []

    # sort left → right
    boxes = sorted(boxes, key=lambda b: b[0])

    # merge horizontally close boxes (handles = sign, ÷ dots, etc.)
    merged = [boxes[0]]
    for box in boxes[1:]:
        prev = merged[-1]
        gap  = box[0] - (prev[0] + prev[2])   # gap between boxes
        # merge if gap is small relative to average box width
        avg_w = np.mean([b[2] for b in boxes])
        if gap < avg_w * 0.4:
            # expand previous box to include this one
            new_x = min(prev[0], box[0])
            new_y = min(prev[1], box[1])
            new_x2 = max(prev[0]+prev[2], box[0]+box[2])
            new_y2 = max(prev[1]+prev[3], box[1]+box[3])
            merged[-1] = [new_x, new_y, new_x2-new_x, new_y2-new_y]
        else:
            merged.append(box)

    return merged


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — CLASSIFY EACH SYMBOL
#  Crop each bounding box → 28×28 → CNN prediction
# ═══════════════════════════════════════════════════════════════
def classify_symbols(binary, boxes, padding=6):
    """
    For each bounding box:
      1. Crop + pad
      2. Resize to 28×28
      3. Run through CNN
    Returns list of (label, confidence) per box.
    """
    results = []
    img_h, img_w = binary.shape

    for (x, y, w, h) in boxes:
        # add padding around the symbol
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        crop = binary[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # make square (pad shorter side)
        ch, cw = crop.shape
        side = max(ch, cw)
        square = np.zeros((side, side), dtype=np.uint8)
        offset_y = (side - ch) // 2
        offset_x = (side - cw) // 2
        square[offset_y:offset_y+ch, offset_x:offset_x+cw] = crop

        # resize to 28×28
        img28 = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        img28 = img28.astype(np.float32) / 255.0

        # predict
        x_in  = img28.reshape(1, 28, 28, 1)
        probs = model.predict(x_in, verbose=0)[0]
        idx   = int(np.argmax(probs))
        label = LABEL_MAP[idx]
        conf  = float(probs[idx])

        results.append((label, conf))

    return results


# ═══════════════════════════════════════════════════════════════
#  STEP 4 — EXPRESSION SOLVER
#  Turn symbol list into a math expression and evaluate
# ═══════════════════════════════════════════════════════════════

# Map CNN labels → Python math operators
SYMBOL_MAP = {
    "+": "+",
    "-": "-",
    "x": "*",
    "/": "/",
    "=": "=",
    **{str(i): str(i) for i in range(10)}
}

def build_expression(predictions):
    """
    Convert list of predicted labels into a math expression string.
    Handles:
      - Multi-digit numbers:  ['1','2','+','3'] → "12 + 3"
      - Equals sign:          ['5','+','3','='] → "5 + 3"  (strips =)
      - Implicit multiply:    ['2','x','3']     → "2 * 3"
    """
    tokens = []
    for label, conf in predictions:
        sym = SYMBOL_MAP.get(label, label)
        tokens.append(sym)

    # join digits into numbers
    expr_parts = []
    i = 0
    while i < len(tokens):
        if tokens[i].isdigit():
            num = tokens[i]
            while i+1 < len(tokens) and tokens[i+1].isdigit():
                i += 1
                num += tokens[i]
            expr_parts.append(num)
        elif tokens[i] == "=":
            pass   # strip equals sign (user wrote "3+5=")
        else:
            expr_parts.append(tokens[i])
        i += 1

    return " ".join(expr_parts)


def solve(expression_str):
    """
    Safely evaluate a math expression string.
    Returns (result, error_message).
    Supports: + - * /  and multi-digit numbers.
    """
    # whitelist: only digits, operators, spaces, parentheses, dots
    clean = expression_str.strip()
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", clean):
        return None, f"Unsafe expression: '{clean}'"

    if not clean:
        return None, "No expression found"

    try:
        result = eval(clean)   # safe — whitelisted above
        # return int if whole number
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return result, None
    except ZeroDivisionError:
        return None, "Division by zero!"
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════════
#  VISUALISE — show the full pipeline result
# ═══════════════════════════════════════════════════════════════
def visualise_result(display_gray, binary, boxes, predictions, expression, result, error):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#1a1a2e")

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── original image ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(display_gray, cmap="gray")
    ax1.set_title("Input Image", color="white", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # ── binary + bounding boxes ──────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    COLORS = [(0,255,128),(0,200,255),(255,180,0),(255,80,80),(180,80,255)]
    for i, (x, y, w, h) in enumerate(boxes):
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
    ax2.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax2.set_title("Segmented Symbols", color="white", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # ── per-symbol predictions ───────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#16213e")
    ax3.axis("off")
    ax3.set_title("Symbol Predictions", color="white", fontsize=12, fontweight="bold")
    if predictions:
        rows = [["#", "Symbol", "Confidence"]] + \
               [[str(i+1), lbl, f"{conf*100:.1f}%"] for i,(lbl,conf) in enumerate(predictions)]
        tbl = ax3.table(cellText=rows[1:], colLabels=rows[0],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(13)
        tbl.scale(1.2, 2.0)
        for (r,c), cell in tbl.get_celld().items():
            cell.set_facecolor("#0f3460" if r == 0 else "#16213e")
            cell.set_text_props(color="white")
            cell.set_edgecolor("#334466")

    # ── expression + result (big display) ────────
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor("#0f3460")
    ax4.axis("off")

    if error:
        ax4.text(0.5, 0.6, f"Expression:  {expression or '?'}",
                 ha="center", va="center", fontsize=18, color="#aaaaaa",
                 transform=ax4.transAxes)
        ax4.text(0.5, 0.25, f"⚠  {error}",
                 ha="center", va="center", fontsize=20, color="#e74c3c",
                 fontweight="bold", transform=ax4.transAxes)
    else:
        display_expr = expression + " = " + str(result)
        ax4.text(0.5, 0.55, "Result",
                 ha="center", va="center", fontsize=14, color="#aaaaaa",
                 transform=ax4.transAxes)
        ax4.text(0.5, 0.25, display_expr,
                 ha="center", va="center", fontsize=36, color="#2ecc71",
                 fontweight="bold", transform=ax4.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                           edgecolor="#2ecc71", linewidth=2))

    fig.suptitle("Handwritten Expression Solver", color="white",
                 fontsize=16, fontweight="bold", y=0.98)

    out = "outputs/expression_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  📊  Result saved → {out}")
    plt.show()


# ═══════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_pipeline(img_input, show=True):
    print("\n  🔍  Preprocessing …")
    binary, display_gray = preprocess_image(img_input)

    print("  ✂️   Segmenting symbols …")
    boxes = segment_symbols(binary)
    print(f"      Found {len(boxes)} symbol(s)")

    if not boxes:
        print("\n  ❌  No symbols found. Check image quality.\n")
        return

    print("  🧠  Classifying symbols …")
    predictions = classify_symbols(binary, boxes)
    for i, (lbl, conf) in enumerate(predictions):
        print(f"      [{i+1}] '{lbl}'  —  {conf*100:.1f}% confident")

    print("  🔢  Building expression …")
    expression = build_expression(predictions)
    print(f"      Expression: {expression}")

    print("  ✅  Solving …")
    result, error = solve(expression)

    if error:
        print(f"\n  ⚠️   Could not solve: {error}")
    else:
        print(f"\n  🎉  {expression} = {result}\n")

    if show:
        visualise_result(display_gray, binary, boxes,
                         predictions, expression, result, error)

    return expression, result, error


# ═══════════════════════════════════════════════════════════════
#  MODE — SYNTHETIC TEST IMAGE  (no real image needed)
# ═══════════════════════════════════════════════════════════════
def make_test_image(expression_str="3+5"):
    """Generate a synthetic handwritten-style expression image for testing."""
    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    canvas = np.zeros((80, 300), dtype=np.uint8)
    x_off  = 20
    for ch in expression_str:
        fs = 1.8
        th = 3
        ts = cv2.getTextSize(ch, FONT, fs, th)[0]
        cv2.putText(canvas, ch, (x_off, 60), FONT, fs, 255, th, cv2.LINE_AA)
        x_off += ts[0] + 10

    # invert so it looks like ink on white paper
    canvas = cv2.bitwise_not(canvas)
    path   = "outputs/test_expression.png"
    cv2.imwrite(path, canvas)
    print(f"  🖼️   Test image saved → {path}")
    return path


# ═══════════════════════════════════════════════════════════════
#  MODE — DRAW LIVE
# ═══════════════════════════════════════════════════════════════
def run_draw_solver():
    print("═" * 55)
    print("  DRAW MODE — Write an expression, press SPACE to solve")
    print("  • Draw with left-click")
    print("  • SPACE  → solve expression")
    print("  • R      → reset")
    print("  • Q      → quit")
    print("═" * 55)

    CANVAS_H, CANVAS_W = 120, 560
    canvas   = np.ones((CANVAS_H, CANVAS_W), dtype=np.uint8) * 255
    drawing  = [False]
    last_pos = [None]
    answer   = [""]

    win = "Write expression  |  SPACE=solve  R=reset  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, CANVAS_W * 2, CANVAS_H * 2 + 60)

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True; last_pos[0] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            if last_pos[0]:
                cv2.line(canvas, last_pos[0], (x, y), 0, 6, cv2.LINE_AA)
            last_pos[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False; last_pos[0] = None

    cv2.setMouseCallback(win, mouse)

    while True:
        display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        # answer bar at bottom
        bar = np.zeros((50, CANVAS_W, 3), dtype=np.uint8)
        txt = answer[0] if answer[0] else "Draw expression then press SPACE"
        cv2.putText(bar, txt, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2)
        combined = np.vstack([display, bar])
        cv2.imshow(win, combined)

        key = cv2.waitKey(20) & 0xFF
        if key == ord(" "):
            # solve current canvas
            print("\n" + "─"*40)
            inverted = cv2.bitwise_not(canvas)   # white strokes on black
            binary, _ = preprocess_image(inverted)
            boxes      = segment_symbols(binary)
            if boxes:
                preds      = classify_symbols(binary, boxes)
                expression = build_expression(preds)
                result, err = solve(expression)
                if err:
                    answer[0] = f"Error: {err}"
                    print(f"  ⚠️  {err}")
                else:
                    answer[0] = f"{expression} = {result}"
                    print(f"  🎉  {expression} = {result}")
            else:
                answer[0] = "No symbols found — try again"
                print("  ❌  No symbols found")

        elif key == ord("r") or key == ord("R"):
            canvas[:] = 255
            answer[0] = ""
        elif key == ord("q") or key == ord("Q") or key == 27:
            break

    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  Handwritten Expression Solver")
    print("═"*55)

    if len(sys.argv) == 1:
        # auto test with synthetic image
        print("\n  No image given — generating synthetic test image …")
        exprs = ["3+5", "12-4", "6x3", "8/2"]
        for expr in exprs:
            print(f"\n{'─'*40}")
            print(f"  Testing expression: '{expr}'")
            path = make_test_image(expr)
            run_pipeline(path, show=(expr == exprs[-1]))  # show last one

    elif sys.argv[1] == "--draw":
        run_draw_solver()

    else:
        run_pipeline(sys.argv[1], show=True)
