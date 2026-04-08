# Handwritten Math Solver

Solves handwritten mathematical expressions from images with near-perfect accuracy.
No model training required — uses a production-grade vision API for recognition and SymPy for exact math.

## How it works

```
Image → Preprocess (OpenCV) → Recognise (Gemini / Claude Vision) → Solve (SymPy) → Result
```

| Stage | Tool | What it does |
|---|---|---|
| Preprocessing | OpenCV | Upscale, denoise, deskew, sharpen, cap size |
| Recognition | Gemini 1.5 Flash **or** Claude Vision | Reads handwriting → clean expression string |
| Solving | SymPy | Arithmetic eval, algebraic simplification, equation solving |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Choosing a provider

| Provider | Cost | Limit | Key prefix |
|---|---|---|---|
| **Google Gemini** ⭐ recommended | **Free** | 1,500 req/day · 15 req/min | `AIza...` |
| Anthropic Claude | ~$0.004/image | Pay as you go | `sk-ant-...` |

### Gemini (free)

1. Get a free key at **aistudio.google.com/app/apikey**
2. Set the environment variable:

**Mac / Linux:**
```bash
export GEMINI_API_KEY="AIza..."
```
**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY = "AIza..."
```
**Permanent in VS Code** — add to User Settings JSON (`Ctrl+Shift+P` → Open User Settings JSON):
```json
"terminal.integrated.env.windows": {
    "GEMINI_API_KEY": "AIza..."
}
```

### Anthropic Claude (paid)

1. Get a key at **console.anthropic.com** → API Keys
2. Add credits under Plans & Billing ($5 covers hundreds of images)
3. Set the environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # Mac/Linux
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # Windows PowerShell
```

---

## Usage

### Single image — Python

```python
from solver import HandwrittenMathSolver

# Gemini (free)
solver = HandwrittenMathSolver(provider="gemini")
result = solver.solve("my_image.png")

# Claude (paid)
solver = HandwrittenMathSolver(provider="anthropic")
result = solver.solve("my_image.png")

print(result["result"])   # answer
print(result["steps"])    # step-by-step breakdown
print(result["latex"])    # LaTeX string
```

> Always use the **full image path** if you're not running from the same folder as the image:
> ```python
> result = solver.solve(r"C:\Users\you\Desktop\mathlens\image.png")
> ```

### Single image — command line

```bash
python example.py image.png
python example.py "C:\full\path\to\image.png"
```

### Batch processing (folder of images)

```bash
python batch_solver.py --images ./test_images
python batch_solver.py --images ./test_images --output results.json
python batch_solver.py --images ./test_images --verbose
```

---

## Result dictionary

```python
{
  "image":      "path/to/image.png",
  "provider":   "gemini",                       # or "anthropic"
  "recognised": "((3*x-4)*(3*x+4))**2",         # expression extracted from handwriting
  "type":       "simplify",                      # arithmetic | simplify | equation | error
  "result":     "81*x**4 - 288*x**2 + 256",     # simplified/solved form
  "expanded":   "81*x**4 - 288*x**2 + 256",     # expanded form (simplify only)
  "factored":   "(3*x - 4)**2*(3*x + 4)**2",    # factored form (simplify only)
  "numeric":    None,                             # float for arithmetic results, else None
  "steps":      [...],                            # list of step strings
  "latex":      "81 x^{4} - 288 x^{2} + 256"
}
```

---

## Supported expression types

| Type | Examples | Notes |
|---|---|---|
| Arithmetic | `2 + 3`, `(7 × 8) ÷ 4`, `√144` | Returns exact integer or fraction, never a rounding error |
| Algebraic | `3x² - 5x + 2`, `((3x-4)(3x+4))²` | Returns simplified, expanded, and factored forms |
| Equation | `2x + 5 = 13`, `x² - 9 = 0` | Returns all solutions per variable |
| Trailing `=` | `3 + 2 =` | Treated as arithmetic automatically |

---

## Supported image formats

| Format | Notes |
|---|---|
| `.jpg` / `.jpeg` | ✅ Recommended for phone photos |
| `.png` | ✅ Recommended for screenshots |
| `.bmp`, `.webp`, `.tiff` | ✅ Supported |
| `.heic` / `.heif` | ❌ iPhone default — convert to JPG first |
| `.pdf`, `.gif` | ❌ Not supported |

> **iPhone users:** Go to **Settings → Camera → Formats → Most Compatible** to shoot in JPG automatically.

---

## Common errors & fixes

| Error | Fix |
|---|---|
| `FileNotFoundError: Cannot read image` | Image is not in the current folder. Use the full path: `solver.solve(r"C:\full\path\image.png")` |
| `ValueError: Gemini API key required` | Set `GEMINI_API_KEY` in your terminal. Reopen terminal after setting it in system settings. |
| `credit balance is too low` | Anthropic only. Add credits at console.anthropic.com → Plans & Billing |
| `API error 400` | Image too large or wrong format. The solver auto-resizes but very unusual formats may fail. |
| Recognised expression is wrong | Try a clearer photo — better lighting, no shadows, pen contrast matters |

---

## Files

| File | Run it? | Purpose |
|---|---|---|
| `solver.py` | No — import it | Core engine: preprocessing + API client + SymPy solver |
| `example.py` | Yes | Quick demo — verifies your setup works end to end |
| `batch_solver.py` | Yes | CLI batch processor for a folder of images |
| `requirements.txt` | No — feed to pip | All Python dependencies |

---

## Why no training?

The vision model (Gemini / Claude) is already trained by Google / Anthropic on billions of images and already understands handwritten math, symbols, fractions, exponents, and brackets out of the box. Training a custom model from scratch would take months and still not match the accuracy. This pipeline uses those production models via an API call — you get world-class recognition without any training.
