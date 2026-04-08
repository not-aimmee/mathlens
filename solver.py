"""
HandwrittenMathSolver
=====================
Solves handwritten mathematical expressions from images.

Pipeline:
  1. Preprocess image  (OpenCV)  — enhance, denoise, deskew
  2. Recognise expression        — Vision API (Anthropic Claude OR Google Gemini)
  3. Solve / simplify            — SymPy
  4. Return structured result

Usage
-----
    # --- Google Gemini (FREE — 1,500 requests/day) ---
    from solver import HandwrittenMathSolver
    solver = HandwrittenMathSolver(api_key="AIza...", provider="gemini")
    result = solver.solve("path/to/image.png")

    # --- Anthropic Claude (paid, ~$0.004/image) ---
    from solver import HandwrittenMathSolver
    solver = HandwrittenMathSolver(api_key="sk-ant-...")
    result = solver.solve("path/to/image.png")

API key env vars:
    Anthropic -> ANTHROPIC_API_KEY
    Gemini    -> GEMINI_API_KEY
"""

from __future__ import annotations

import base64
import json
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    import urllib.request as _urllib_request
    import urllib.error
    _HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


GEMINI_API_URL    = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"

MAX_TOKENS = 1024
RETRYABLE_STATUS = {429, 502, 503, 504}
MAX_HTTP_RETRIES = 4
BASE_RETRY_DELAY = 2.0

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

VISION_PROMPT = textwrap.dedent("""\
    You are a specialist OCR engine for handwritten mathematical expressions.
    Your ONLY job is to transcribe exactly what is written in the image into
    a clean, unambiguous mathematical expression that Python's SymPy library
    can parse.

    Rules:
    - Output ONLY the expression -- no explanation, no prose, no markdown.
    - Use ** for exponentiation  (e.g. x**2, not x^2 or x squared).
    - Use * for multiplication where it must be explicit.
    - Use / for division, sqrt() for square roots.
    - Preserve parentheses exactly as written.
    - For equations (containing =) output both sides separated by =.
    - If the image is unreadable output exactly: UNREADABLE
""")


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """Enhance a handwritten math image for best vision-model results."""

    @staticmethod
    def _find_alternative_path(path: Path) -> Path | None:
        if path.exists():
            return path

        root = path.with_suffix("")
        candidates = []
        if path.suffix:
            candidates.append(root)
        candidates.extend([root.with_suffix(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]])

        for candidate in candidates:
            if candidate.exists():
                return candidate

        cwd_candidate = Path.cwd() / path.name
        if cwd_candidate.exists():
            return cwd_candidate

        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            candidate = cwd_candidate.with_suffix(ext)
            if candidate.exists():
                return candidate

        return None

    @staticmethod
    def load(path):
        path = Path(path)
        fallback = ImagePreprocessor._find_alternative_path(path)
        if fallback is not None:
            path = fallback

        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img

    @staticmethod
    def to_grayscale(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def denoise(gray):
        return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    @staticmethod
    def deskew(gray):
        coords = np.column_stack(np.where(gray < 128))
        if coords.size == 0:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5:
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def upscale_if_small(gray, min_dim=512):
        h, w = gray.shape
        if min(h, w) < min_dim:
            scale = min_dim / min(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)
        return gray

    @staticmethod
    def cap_size(gray, max_dim=1568):
        h, w = gray.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        return gray

    @staticmethod
    def sharpen(gray):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(gray, -1, kernel)

    def process(self, path):
        img       = self.load(path)
        gray      = self.to_grayscale(img)
        gray      = self.upscale_if_small(gray)
        gray      = self.cap_size(gray)
        gray      = self.denoise(gray)
        gray      = self.deskew(gray)
        gray      = self.sharpen(gray)
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        _, buf    = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 92])
        b64       = base64.b64encode(buf.tobytes()).decode("utf-8")
        return processed, b64


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

def _post(url, headers, payload):
    body = json.dumps(payload).encode()
    attempt = 0

    while True:
        try:
            if _HAS_REQUESTS:
                resp = _requests.post(url, headers=headers, json=payload, timeout=60)
                status_code = resp.status_code
                text = resp.text
            else:
                req = _urllib_request.Request(url, data=body, headers=headers, method="POST")
                with _urllib_request.urlopen(req, timeout=60) as r:
                    status_code = r.getcode()
                    text = r.read().decode('utf-8')

            if 200 <= status_code < 300:
                return json.loads(text) if not _HAS_REQUESTS else resp.json()

            if status_code in RETRYABLE_STATUS and attempt < MAX_HTTP_RETRIES:
                attempt += 1
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                time.sleep(delay)
                continue

            raise RuntimeError(f"API error {status_code}: {text}")

        except Exception as exc:
            retriable_exc = False
            if _HAS_REQUESTS and isinstance(exc, _requests.exceptions.RequestException):
                retriable_exc = True
            elif not _HAS_REQUESTS and isinstance(exc, (_urllib_error.HTTPError, _urllib_error.URLError)):
                retriable_exc = True

            if retriable_exc and attempt < MAX_HTTP_RETRIES:
                attempt += 1
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                time.sleep(delay)
                continue
            raise


# ---------------------------------------------------------------------------
# Vision API clients
# ---------------------------------------------------------------------------


class GeminiVisionClient:
    """
    Wrapper around Google Gemini 1.5 Flash (FREE tier).
    Limits: 1,500 requests/day | 15 requests/minute
    Get your free key at: https://aistudio.google.com/app/apikey
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY.\n"
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )

    def recognise(self, b64_image):
        url     = f"{GEMINI_API_URL}?key={self.api_key}"
        headers = {"content-type": "application/json"}
        
        # Combine the system prompt and user prompt into one 'text' part
        combined_prompt = f"{VISION_PROMPT}\n\nTranscribe this handwritten mathematical expression exactly. Output only the expression -- nothing else."

        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}},
                    {"text": combined_prompt},
                ]
            }],
            "generationConfig": {
                "maxOutputTokens": MAX_TOKENS,
                "temperature": 0.0,
            },
        }
        data = _post(url, headers, payload)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Gemini response: {data}")


# ---------------------------------------------------------------------------
# Math solver / evaluator
# ---------------------------------------------------------------------------

class MathEngine:
    """Parse and solve/simplify expressions using SymPy."""

    REPLACEMENTS = [
        (r"\bx\^(\w+)", r"x**\1"),
        (r"\^",         r"**"),
        (r"÷",          r"/"),
        (r"×",          r"*"),
        (r"sqrt\(([^)]+)\)", r"sqrt(\1)"),
        (r"√\(([^)]+)\)", r"sqrt(\1)"),
        (r"√(\w+)",     r"sqrt(\1)"),
        (r"(\d)\s*\(",  r"\1*("),
        (r"\)\s*\(",    r")*("),
        (r"(\d)([a-zA-Z])", r"\1*\2"),
    ]

    def _clean(self, raw):
        expr = raw.strip()
        for pattern, repl in self.REPLACEMENTS:
            expr = re.sub(pattern, repl, expr)
        return expr

    def _parse(self, expr_str):
        return parse_expr(expr_str, transformations=TRANSFORMATIONS, evaluate=True)

    def solve(self, raw_expression):
        if raw_expression.upper() == "UNREADABLE":
            return {
                "recognised": raw_expression, "type": "error",
                "result": "Image was unreadable -- please try a clearer photo.",
                "numeric": None, "steps": [], "latex": "",
            }

        cleaned = self._clean(raw_expression)
        steps = [
            f"Recognised expression: {raw_expression}",
            f"Normalised for SymPy: {cleaned}",
        ]

        if "=" in cleaned:
            return self._solve_equation(cleaned, steps)

        try:
            expr = self._parse(cleaned)
        except Exception as e:
            return {
                "recognised": cleaned, "type": "error",
                "result": f"Could not parse expression: {e}",
                "numeric": None, "steps": steps, "latex": "",
            }

        free = expr.free_symbols
        steps.append(f"Free symbols detected: {', '.join(str(s) for s in free) or 'none'}")

        if not free:
            result  = sp.nsimplify(expr, rational=True)
            numeric = float(result.evalf())
            steps.append(f"Evaluated: {result}")
            return {
                "recognised": cleaned, "type": "arithmetic",
                "result": str(result), "numeric": numeric,
                "steps": steps, "latex": sp.latex(result),
            }
        else:
            expanded   = sp.expand(expr)
            simplified = sp.simplify(expr)
            factored   = sp.factor(expr)
            steps += [f"Expanded: {expanded}", f"Simplified: {simplified}", f"Factored: {factored}"]
            return {
                "recognised": cleaned, "type": "simplify",
                "result": str(simplified), "expanded": str(expanded),
                "factored": str(factored), "numeric": None,
                "steps": steps, "latex": sp.latex(simplified),
            }

    def _solve_equation(self, cleaned, steps):
        lhs_str, rhs_str = cleaned.split("=", 1)

        # Handle trailing = with nothing on the right (e.g. "3 + 2 =")
        # Vision models sometimes include the answer box -- treat as plain arithmetic
        if not rhs_str.strip():
            steps.append("Trailing '=' detected -- evaluating left-hand side as expression.")
            try:
                expr = self._parse(lhs_str.strip())
            except Exception as e:
                return {
                    "recognised": cleaned, "type": "error",
                    "result": f"Could not parse expression: {e}",
                    "numeric": None, "steps": steps, "latex": "",
                }
            free = expr.free_symbols
            if not free:
                result  = sp.nsimplify(expr, rational=True)
                numeric = float(result.evalf())
                steps.append(f"Evaluated: {result}")
                return {
                    "recognised": cleaned, "type": "arithmetic",
                    "result": str(result), "numeric": numeric,
                    "steps": steps, "latex": sp.latex(result),
                }
            else:
                simplified = sp.simplify(expr)
                steps.append(f"Simplified: {simplified}")
                return {
                    "recognised": cleaned, "type": "simplify",
                    "result": str(simplified), "expanded": str(sp.expand(expr)),
                    "factored": str(sp.factor(expr)), "numeric": None,
                    "steps": steps, "latex": sp.latex(simplified),
                }

        try:
            lhs = self._parse(lhs_str.strip())
            rhs = self._parse(rhs_str.strip())
        except Exception as e:
            return {
                "recognised": cleaned, "type": "error",
                "result": f"Could not parse equation sides: {e}",
                "numeric": None, "steps": steps, "latex": "",
            }

        equation  = sp.Eq(lhs, rhs)
        steps.append(f"Equation: {equation}")
        solutions = {}
        for sym in sorted(equation.free_symbols, key=str):
            sol = sp.solve(equation, sym)
            solutions[str(sym)] = [str(s) for s in sol]
            steps.append(f"Solving for {sym}: {sol}")

        return {
            "recognised": cleaned, "type": "equation",
            "result": solutions, "numeric": None,
            "steps": steps, "latex": sp.latex(equation),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class HandwrittenMathSolver:
    """
    End-to-end solver for handwritten math images.

    Parameters
    ----------
    api_key  : str, optional
        API key for the chosen provider.
        Anthropic -> falls back to ANTHROPIC_API_KEY env var.
        Gemini   -> falls back to GEMINI_API_KEY env var.
    provider : str
        'anthropic' (default, paid) or 'gemini' (free, 1500/day).
    verbose  : bool
        Print step-by-step info to stdout.
    """

    PROVIDERS = {
        "gemini":    GeminiVisionClient,
    }

    def __init__(self, api_key=None, provider="gemini", verbose=False):
        provider = provider.lower()
        if provider not in self.PROVIDERS:
            raise ValueError(f"provider must be one of {list(self.PROVIDERS)}. Got: '{provider}'")

        self.preprocessor = ImagePreprocessor()
        self.vision       = self.PROVIDERS[provider](api_key=api_key)
        self.engine       = MathEngine()
        self.verbose      = verbose
        self.provider     = provider

    def solve(self, image_path):
        path = Path(image_path)
        if self.verbose:
            print(f"[1/3] Preprocessing: {path.name}")

        _, b64 = self.preprocessor.process(path)

        label = "Gemini 2.0 (free)" if self.provider == "gemini" else "Claude Vision"
        if self.verbose:
            print(f"[2/3] Sending to {label} for expression recognition ...")

        raw_expr = self.vision.recognise(b64)

        if self.verbose:
            print(f"      Recognised: {raw_expr}")
            print("[3/3] Solving with SymPy ...")

        result             = self.engine.solve(raw_expr)
        result["image"]    = str(path)
        result["provider"] = self.provider

        if self.verbose:
            self._print_result(result)

        return result

    @staticmethod
    def _print_result(r):
        print("\n" + "=" * 60)
        print(f"  Image      : {r['image']}")
        print(f"  Provider   : {r.get('provider', 'unknown')}")
        print(f"  Recognised : {r['recognised']}")
        print(f"  Type       : {r['type']}")
        print(f"  Result     : {r['result']}")
        if r.get("numeric") is not None:
            print(f"  Numeric    : {r['numeric']}")
        if r.get("latex"):
            print(f"  LaTeX      : {r['latex']}")
        print("\n  Steps:")
        for i, s in enumerate(r.get("steps", []), 1):
            print(f"    {i}. {s}")
        print("=" * 60)
