"""117-dimensional CIELAB feature extractor used for downstream growth-phase
classification (paper Section 3.4 / 4.2): per-channel 32-bin histograms
(96) + 6 summary statistics per channel (18) + 3 inter-channel correlations
(3) = 117.

This module has NO dependency on the rest of the PRISM package - it is safe
to import standalone from a web-service backend for inference.
"""
import numpy as np
from scipy.stats import kurtosis, skew

CH_RANGES = [(0.0, 100.0), (-80.0, 80.0), (-80.0, 80.0)]
FEATURE_DIM = 117


def rgb_to_lab(rgb):
    """Reference sRGB -> CIELAB conversion (D65), implemented without
    scikit-image so this module has zero heavy dependencies for deployment.
    """
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
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    return np.stack([L, a, b_], axis=-1).astype(np.float32)


def extract_cielab_features(image_rgb_0_1):
    """image_rgb_0_1: HxWx3 float array, RGB order, values in [0, 1]."""
    lab = rgb_to_lab(image_rgb_0_1.astype(np.float32)).reshape(-1, 3)
    feat = []
    for ch, (lo, hi) in enumerate(CH_RANGES):
        hist, _ = np.histogram(lab[:, ch], bins=32, range=(lo, hi), density=True)
        feat.extend(hist)
    for ch in range(3):
        v = lab[:, ch]
        feat.extend([
            float(np.mean(v)), float(np.std(v)), float(skew(v)), float(kurtosis(v)),
            float(np.percentile(v, 25)), float(np.percentile(v, 75)),
        ])
    feat.extend([
        float(np.corrcoef(lab[:, 0], lab[:, 1])[0, 1]),
        float(np.corrcoef(lab[:, 1], lab[:, 2])[0, 1]),
        float(np.corrcoef(lab[:, 0], lab[:, 2])[0, 1]),
    ])
    return np.nan_to_num(np.array(feat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_batch(images):
    return np.stack([extract_cielab_features(img) for img in images]).astype(np.float32)
