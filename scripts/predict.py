"""Standalone inference helper for the saved growth-phase classifier.

For the web team: this file only needs `joblib`, `numpy`, `opencv-python`,
and `scipy` (no PRISM synthesis dependencies) - safe to copy into a backend
service on its own, together with:
    output/classifier/models/best_growth_phase_classifier.joblib
    output/classifier/models/model_metadata.json

CLI usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/folder_of_images/

Programmatic usage:
    from predict import GrowthPhaseClassifier
    clf = GrowthPhaseClassifier()
    label, confidence, probs = clf.predict_path("sample.jpg")
"""
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _THIS_DIR.parent / "output" / "classifier" / "models"

CH_RANGES = [(0.0, 100.0), (-80.0, 80.0), (-80.0, 80.0)]


def _rgb_to_lab(rgb):
    def linearize(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r, g, b = linearize(rgb[..., 0]), linearize(rgb[..., 1]), linearize(rgb[..., 2])
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    xn, yn, zn = 0.95047, 1.00000, 1.08883

    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1).astype(np.float32)


def _extract_features(image_rgb_0_1):
    from scipy.stats import kurtosis, skew
    lab = _rgb_to_lab(image_rgb_0_1.astype(np.float32)).reshape(-1, 3)
    feat = []
    for ch, (lo, hi) in enumerate(CH_RANGES):
        hist, _ = np.histogram(lab[:, ch], bins=32, range=(lo, hi), density=True)
        feat.extend(hist)
    for ch in range(3):
        v = lab[:, ch]
        feat.extend([float(np.mean(v)), float(np.std(v)), float(skew(v)), float(kurtosis(v)),
                     float(np.percentile(v, 25)), float(np.percentile(v, 75))])
    feat.extend([float(np.corrcoef(lab[:, 0], lab[:, 1])[0, 1]),
                 float(np.corrcoef(lab[:, 1], lab[:, 2])[0, 1]),
                 float(np.corrcoef(lab[:, 0], lab[:, 2])[0, 1])])
    return np.nan_to_num(np.array(feat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class GrowthPhaseClassifier:
    def __init__(self, models_dir=_MODELS_DIR):
        models_dir = Path(models_dir)
        self.metadata = json.loads((models_dir / "model_metadata.json").read_text(encoding="utf8"))
        self.model = joblib.load(models_dir / "best_growth_phase_classifier.joblib")
        self.phases = self.metadata["phases_in_label_order"]
        self.image_size = tuple(self.metadata["preprocessing"]["resize"])

    def _load_and_preprocess(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img = cv2.cvtColor(cv2.resize(img, self.image_size), cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def predict_array(self, image_rgb_0_1):
        """image_rgb_0_1: HxWx3 RGB float array in [0, 1] (or uint8 0-255,
        auto-normalized)."""
        img = np.asarray(image_rgb_0_1, dtype=np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        if img.shape[:2] != self.image_size:
            img = cv2.resize(img, self.image_size)
        features = _extract_features(img).reshape(1, -1)
        pred_idx = int(self.model.predict(features)[0])
        probs = self.model.predict_proba(features)[0] if hasattr(self.model, "predict_proba") else None
        label = self.phases[pred_idx]
        confidence = float(probs[pred_idx]) if probs is not None else None
        prob_map = {self.phases[i]: float(p) for i, p in enumerate(probs)} if probs is not None else None
        return label, confidence, prob_map

    def predict_path(self, image_path):
        return self.predict_array(self._load_and_preprocess(image_path))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    clf = GrowthPhaseClassifier()
    print(f"Loaded model: {clf.metadata['best_model_name']}")

    paths = sorted(p for p in target.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
        if target.is_dir() else [target]

    for p in paths:
        label, confidence, probs = clf.predict_path(p)
        conf_str = f"{confidence:.3f}" if confidence is not None else "n/a"
        print(f"{p.name:<40} -> {label:<12} (confidence={conf_str})")
        if probs:
            print("   " + "  ".join(f"{k}={v:.3f}" for k, v in probs.items()))


if __name__ == "__main__":
    main()
