# Handwritten Digit & Math Expression Recognizer 🔢➕

A CNN-based classifier that recognises **handwritten digits (0–9)** and
**math symbols (+, −, ×, ÷, =)** from 28×28 grayscale images.

---

## Project Structure

```
digit_math_recognizer/
│
├── train.py            ← Full pipeline (run this first)
├── predict.py          ← Inference module for the UI team
├── explore_data.py     ← Visualise dataset samples
├── requirements.txt    ← Python dependencies
│
├── models/             ← Created after training
│   ├── digit_math_recognizer.keras   ← Main model file
│   ├── saved_model/                  ← TF SavedModel (TFLite ready)
│   └── label_map.json                ← {0:"0", …, 14:"="}
│
└── outputs/            ← Created after training
    ├── training_curves.png
    ├── confusion_matrix.png
    └── data_samples.png
```

---

## Quick Start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — (Optional) Explore the data
```bash
python explore_data.py
```

### 3 — Train the model
```bash
python train.py
```
Training takes **~10 min on GPU** / ~45 min on CPU.
After completion the `models/` folder contains everything your UI teammate needs.

---

## Dataset

| Source | Samples | Labels |
|--------|---------|--------|
| MNIST  | 70 000  | 0 – 9  |
| Synthetic (OpenCV rendered + augmented) | 20 000 | + − × ÷ = |
| **Total** | **~90 000** | **15 classes** |

---

## CNN Architecture

```
Input (28×28×1)
  │
  ├─ Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
  ├─ Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  ├─ Conv2D(128)→ BN → MaxPool → Dropout(0.25)
  │
  └─ Flatten → Dense(256) → BN → Dropout(0.5) → Softmax(15)
```

**Expected test accuracy** : ~98 % digits / ~95 % symbols

---

## For the UI Team — Using `predict.py`

```python
from predict import DigitMathPredictor

# Load once
predictor = DigitMathPredictor(
    model_path      = "models/digit_math_recognizer.keras",
    label_map_path  = "models/label_map.json"
)

# Predict from a file
label, confidence, all_probs = predictor.predict_image("my_drawing.png")
print(f"Predicted: {label}  ({confidence*100:.1f}% confident)")

# Predict from a NumPy array  (e.g. canvas pixel data from the UI)
import numpy as np
img_array = np.array(...)   # shape (28,28) or (28,28,1) or (H,W,3)
label, confidence, all_probs = predictor.predict_array(img_array)

# Batch prediction  (list of arrays)
results = predictor.predict_batch([img1, img2, img3])
```

### Input image requirements
| Property | Value |
|----------|-------|
| Size | Any — auto-resized to 28×28 |
| Colour | Any — auto-converted to grayscale |
| Format | PNG, JPG, BMP, or NumPy array |
| Background | Black (0) with white strokes preferred |

---

## Label Map Reference

```json
{
  "0":"0","1":"1","2":"2","3":"3","4":"4",
  "5":"5","6":"6","7":"7","8":"8","9":"9",
  "10":"+","11":"-","12":"x","13":"/","14":"="
}
```
