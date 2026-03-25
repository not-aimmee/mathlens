"""
predict.py
──────────
Ready-to-use prediction module for your teammate's interface.

Usage:
    from predict import DigitMathPredictor

    predictor = DigitMathPredictor()
    label, confidence = predictor.predict_image("path/to/image.png")
    # or pass a numpy array directly
    label, confidence = predictor.predict_array(numpy_28x28_array)
"""

import json
import os
import numpy as np


class DigitMathPredictor:
    """
    Load the trained CNN and expose simple prediction methods.

    Parameters
    ----------
    model_path : str
        Path to the saved .keras model file.
    label_map_path : str
        Path to label_map.json.
    """

    def __init__(
        self,
        model_path: str = "models/digit_math_recognizer.keras",
        label_map_path: str = "models/label_map.json",
    ):
        import tensorflow as tf  # lazy import so file is importable without TF

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Run train.py first to generate it."
            )

        self.model = tf.keras.models.load_model(model_path)
        with open(label_map_path) as f:
            # JSON keys are strings; convert to int
            self.label_map = {int(k): v for k, v in json.load(f).items()}

        print(f"✅  Model loaded from '{model_path}'")
        print(f"    Classes : {list(self.label_map.values())}")

    # ── internal helper ──────────────────────────────────
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Accept any of these formats and normalise to (1, 28, 28, 1) float32:
          • (28, 28)       grayscale
          • (28, 28, 1)    grayscale with channel dim
          • (28, 28, 3)    RGB  → converted to grayscale
          • (H, W)         any size → resized to 28×28
        """
        import cv2  # opencv-python-headless

        # colour → gray
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]

        # resize if needed
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        # normalise
        img = img.astype(np.float32) / 255.0
        return img.reshape(1, 28, 28, 1)

    # ── public API ───────────────────────────────────────
    def predict_array(self, img: np.ndarray):
        """
        Predict from a NumPy array.

        Returns
        -------
        label      : str   e.g. '5' or '+'
        confidence : float 0.0 – 1.0
        all_probs  : dict  {label_str: probability}
        """
        x = self._preprocess(img)
        probs = self.model.predict(x, verbose=0)[0]
        idx   = int(np.argmax(probs))
        all_probs = {self.label_map[i]: float(probs[i])
                     for i in range(len(probs))}
        return self.label_map[idx], float(probs[idx]), all_probs

    def predict_image(self, path: str):
        """
        Predict from an image file path (.png / .jpg / .bmp …).

        Returns
        -------
        Same as predict_array.
        """
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at '{path}'")
        return self.predict_array(img)

    def predict_batch(self, images: list):
        """
        Predict a list of NumPy arrays at once (faster than looping).

        Returns
        -------
        List of (label, confidence) tuples.
        """
        batch = np.vstack([self._preprocess(img) for img in images])
        probs = self.model.predict(batch, verbose=0)
        results = []
        for p in probs:
            idx = int(np.argmax(p))
            results.append((self.label_map[idx], float(p[idx])))
        return results


# ── Quick demo ───────────────────────────────────────────
if __name__ == "__main__":
    import sys

    predictor = DigitMathPredictor()

    if len(sys.argv) > 1:
        # predict a file passed on the command line
        path = sys.argv[1]
        label, conf, _ = predictor.predict_image(path)
        print(f"\n  Image  : {path}")
        print(f"  Result : '{label}'  (confidence {conf*100:.1f}%)")
    else:
        # smoke-test with a random white-noise image
        dummy = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        label, conf, all_probs = predictor.predict_array(dummy)
        print(f"\n  Dummy image prediction : '{label}'  ({conf*100:.1f}%)")
        print("  All probabilities:")
        for lbl, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"    {lbl:>3}  {bar:<30}  {prob*100:5.1f}%")
