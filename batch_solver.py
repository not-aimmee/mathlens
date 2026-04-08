"""
batch_solver.py
===============
Process a folder of handwritten math images in one shot.

Usage
-----
    python batch_solver.py --images ./test_images --api-key sk-ant-...
    python batch_solver.py --images ./test_images          # uses ANTHROPIC_API_KEY env var
    python batch_solver.py --images ./test_images --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

from solver import HandwrittenMathSolver

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Batch handwritten math solver")
    parser.add_argument("--images",  required=True, help="Folder or single image file")
    parser.add_argument("--api-key", default=None,  help="Anthropic API key")
    parser.add_argument("--output",  default=None,  help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    solver = HandwrittenMathSolver(api_key=args.api_key, verbose=args.verbose)

    target = Path(args.images)
    if target.is_file():
        paths = [target]
    elif target.is_dir():
        paths = sorted(p for p in target.iterdir() if p.suffix.lower() in SUPPORTED)
    else:
        print(f"Error: {target} is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)

    if not paths:
        print("No supported image files found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for i, p in enumerate(paths, 1):
        print(f"\n[{i}/{len(paths)}] {p.name}")
        try:
            r = solver.solve(p)
            print(f"  → {r['type']:10s}  {r['result']}")
        except Exception as e:
            r = {"image": str(p), "type": "error", "result": str(e)}
            print(f"  ✗ ERROR: {e}", file=sys.stderr)
        results.append(r)

    print(f"\n{'─'*60}")
    print(f"Done. Processed {len(results)} image(s).")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"Results saved to {out}")
    else:
        print("\nJSON summary:")
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
