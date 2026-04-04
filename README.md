# Handwritten Math Solver

Solves handwritten mathematical expressions from images with near-perfect accuracy.

## How it works

```
Image → Preprocess (OpenCV) → Recognise (Claude Vision) → Solve (SymPy) → Result
```

| Stage | Tool | What it does |
|---|---|---|
| Preprocessing | OpenCV | Denoise, deskew, sharpen, binarise |
| Recognition | Claude Vision API | Reads handwriting → clean expression string |
| Solving | SymPy | Arithmetic eval, algebraic simplification, equation solving |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Single image

```python
from solver import HandwrittenMathSolver

solver = HandwrittenMathSolver()                    # reads ANTHROPIC_API_KEY from env
result = solver.solve("my_image.png")

print(result["result"])   # answer
print(result["steps"])    # step-by-step breakdown
print(result["latex"])    # LaTeX string
```

Or with the API key inline:

```python
solver = HandwrittenMathSolver(api_key="sk-ant-...")
```

## Batch processing (folder of images)

```bash
python batch_solver.py --images ./test_images
python batch_solver.py --images ./test_images --output results.json
```

## Result dict

```python
{
  "image":      "path/to/image.png",
  "recognised": "((3*x-4)*(3*x+4))**2",   # extracted from handwriting
  "type":       "simplify",                 # arithmetic | simplify | equation | error
  "result":     "(9*x**2 - 16)**2",         # simplified/solved form
  "expanded":   "81*x**4 - 288*x**2 + 256",
  "factored":   "(3*x - 4)**2*(3*x + 4)**2",
  "numeric":    None,                        # float for arithmetic, else None
  "steps":      [...],                       # list of step strings
  "latex":      "\\left(9 x^{2} - 16\\right)^{2}"
}
```

## Supported expression types

| Type | Example |
|---|---|
| Arithmetic | `2 + 3`, `(7 × 8) ÷ 4`, `√144` |
| Algebraic expression | `3x² - 5x + 2`, `((3x-4)(3x+4))²` |
| Equation | `2x + 5 = 13`, `x² - 9 = 0` |

## Why this design?

Training a custom OCR model from scratch to perfect accuracy on handwritten math
is impractical — even academic state-of-the-art systems don't reach 100%.
This pipeline uses **Claude's Vision model** (the same model behind claude.ai)
for recognition, which already has world-class handwriting understanding,
combined with **SymPy** for exact symbolic mathematics.

The result is a production-grade solver that handles:
- Messy / cursive handwriting  
- Varied pen widths and orientations  
- Simple arithmetic through complex algebra  
- Equations with multiple variables
