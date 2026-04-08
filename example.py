"""
example.py  —  quick-start examples
=====================================
Shows how your teammate (or you) can integrate the solver.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python example.py path/to/image.png
"""

import sys
from pathlib import Path
from solver import HandwrittenMathSolver

def resolve_image_path(image_path: str) -> str:
    path = Path(image_path)
    if path.exists():
        return str(path)

    if not path.suffix:
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            candidate = path.with_suffix(ext)
            if candidate.exists():
                return str(candidate)

    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        root = path.with_suffix("")
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = root.with_suffix(ext)
            if candidate.exists():
                return str(candidate)

    script_dir = Path(__file__).resolve().parent
    alt = script_dir / path
    if alt.exists():
        return str(alt)

    if not path.suffix:
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            candidate = script_dir / path.with_suffix(ext)
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(
        f"Cannot find image file: {image_path}.\n"
        "Check the file name, extension, and path."
    )

def demo(image_path: str, api_key: str | None = None):
    resolved_path = resolve_image_path(image_path)
    solver = HandwrittenMathSolver(api_key=api_key, verbose=True)
    result = solver.solve(resolved_path)

    print("\nSimple Result Summary:")
    print(f"  Recognised : {result.get('recognised', 'N/A')}")
    print(f"  Type       : {result.get('type', 'N/A')}")
    print(f"  Result     : {result.get('result', 'N/A')}")
    if result.get('expanded') is not None:
        print(f"  Expanded   : {result['expanded']}")
    if result.get('factored') is not None:
        print(f"  Factored   : {result['factored']}")
    if result.get('latex') is not None:
        print(f"  LaTeX      : {result['latex']}")

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example.py <image_path> [api_key]")
        sys.exit(1)

    path = sys.argv[1]
    key  = sys.argv[2] if len(sys.argv) > 2 else None
    demo(path, key)
