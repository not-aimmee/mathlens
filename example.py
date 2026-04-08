"""
example.py  —  quick-start examples
=====================================
Shows how your teammate (or you) can integrate the solver.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python example.py path/to/image.png
"""

import sys
from solver import HandwrittenMathSolver

def demo(image_path: str, api_key: str | None = None):
    solver = HandwrittenMathSolver(api_key=api_key, verbose=True)
    result = solver.solve(image_path)
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example.py <image_path> [api_key]")
        sys.exit(1)

    path = sys.argv[1]
    key  = sys.argv[2] if len(sys.argv) > 2 else None
    demo(path, key)
